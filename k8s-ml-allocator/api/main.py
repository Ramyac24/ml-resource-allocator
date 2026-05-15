"""
api/main.py
FastAPI REST interface for the ML Resource Allocator.

Endpoints
---------
POST   /jobs              Submit a new ML job
GET    /jobs              List all jobs (optional ?status= filter)
GET    /jobs/{job_id}     Get single job details
DELETE /jobs/{job_id}     Cancel a queued job
POST   /jobs/{job_id}/scale  Manually override resource limits
GET    /cluster           Live cluster pod metrics snapshot
GET    /health            Health check
"""

import logging
import os
import sys

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

from allocator.controller import MLAllocatorController
from allocator.priority_queue import MLJob, Priority
from allocator.resource_monitor import ResourceMonitor

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

NAMESPACE = os.getenv("ML_NAMESPACE", "ml-workloads")

# Shared controller instance
controller: Optional[MLAllocatorController] = None
monitor:    Optional[ResourceMonitor]       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global controller, monitor
    controller = MLAllocatorController(namespace=NAMESPACE)
    monitor    = ResourceMonitor(namespace=NAMESPACE)
    logger.info(f"API ready — namespace: {NAMESPACE}")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="ML Resource Allocator API",
    description=(
        "REST API for submitting ML training/inference jobs to Kubernetes "
        "with dynamic resource allocation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────────

class JobSubmitRequest(BaseModel):
    name:           str  = Field(...,          example="iris-classifier")
    job_type:       str  = Field(...,          example="training",
                                               description="'training' or 'inference'")
    image:          str  = Field("ml-workload:latest", example="ml-workload:latest")
    priority:       str  = Field("MEDIUM",     example="HIGH",
                                               description="HIGH | MEDIUM | LOW")
    cpu_request_m:  int  = Field(200,          example=200,
                                               description="CPU request in millicores")
    mem_request_mi: int  = Field(256,          example=256,
                                               description="Memory request in MiB")
    cpu_limit_m:    int  = Field(500,          example=500)
    mem_limit_mi:   int  = Field(512,          example=512)
    env:            dict = Field(default_factory=dict,
                                               description="Environment variables")


class ScaleRequest(BaseModel):
    cpu_limit_m:  int = Field(..., example=800, description="New CPU limit (millicores)")
    mem_limit_mi: int = Field(..., example=768, description="New memory limit (MiB)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_priority(p: str) -> Priority:
    mapping = {"HIGH": Priority.HIGH, "MEDIUM": Priority.MEDIUM, "LOW": Priority.LOW}
    p_upper = p.upper()
    if p_upper not in mapping:
        raise HTTPException(status_code=400,
                            detail=f"priority must be one of {list(mapping.keys())}")
    return mapping[p_upper]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "namespace": NAMESPACE}


@app.post("/jobs", status_code=201, tags=["Jobs"])
def submit_job(req: JobSubmitRequest):
    """Submit a new ML job to the allocator queue."""
    if req.job_type not in ("training", "inference"):
        raise HTTPException(status_code=400,
                            detail="job_type must be 'training' or 'inference'")

    priority = _parse_priority(req.priority)
    job = MLJob.create(
        name=req.name,
        job_type=req.job_type,
        image=req.image,
        priority=priority,
        cpu_request_m=req.cpu_request_m,
        mem_request_mi=req.mem_request_mi,
        cpu_limit_m=req.cpu_limit_m,
        mem_limit_mi=req.mem_limit_mi,
        env=req.env,
    )
    job_id = controller.submit(job)

    # Immediately dispatch (single tick)
    controller._tick()

    return {"job_id": job_id, "status": "queued", "message": f"Job '{req.name}' submitted"}


@app.get("/jobs", tags=["Jobs"])
def list_jobs(status: Optional[str] = Query(None,
              description="Filter by status: queued | running | completed | failed")):
    """List all jobs, optionally filtered by status."""
    jobs = controller.queue.list_jobs(status=status)
    return {"jobs": [j.to_dict() for j in jobs], "total": len(jobs)}


@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: str):
    """Get details for a specific job."""
    job = controller.queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


@app.delete("/jobs/{job_id}", tags=["Jobs"])
def cancel_job(job_id: str):
    """Cancel a queued job (cannot cancel running jobs)."""
    cancelled = controller.queue.cancel(job_id)
    if not cancelled:
        job = controller.queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in status '{job.status}' — only 'queued' jobs can be cancelled"
        )
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/jobs/{job_id}/scale", tags=["Jobs"])
def scale_job(job_id: str, req: ScaleRequest):
    """Manually override resource limits for a running inference deployment."""
    job = controller.queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.job_type != "inference":
        raise HTTPException(
            status_code=400,
            detail="Only inference deployments support manual scaling"
        )

    deployment_name = f"mlinf-{job_id}"
    success = controller.scaler.force_scale(
        deployment_name=deployment_name,
        cpu_m=req.cpu_limit_m,
        mem_mi=req.mem_limit_mi,
    )
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scale deployment '{deployment_name}' — is it running?"
        )
    return {
        "job_id":       job_id,
        "deployment":   deployment_name,
        "new_cpu_m":    req.cpu_limit_m,
        "new_mem_mi":   req.mem_limit_mi,
        "status":       "scaled",
    }


@app.get("/cluster", tags=["Cluster"])
def cluster_snapshot():
    """Live snapshot of all pod resource usage in the namespace."""
    pods = monitor.snapshot()
    metrics_available = monitor.metrics_server_available()
    return {
        "namespace":        NAMESPACE,
        "metrics_server":   metrics_available,
        "pods": [
            {
                "name":           p.name,
                "phase":          p.phase,
                "cpu_usage_m":    round(p.cpu_usage_m, 1),
                "cpu_limit_m":    round(p.cpu_limit_m, 1),
                "cpu_util_pct":   round(p.cpu_utilization * 100, 1),
                "mem_usage_mi":   round(p.mem_usage_mi, 1),
                "mem_limit_mi":   round(p.mem_limit_mi, 1),
                "mem_util_pct":   round(p.mem_utilization * 100, 1),
                "labels":         p.labels,
            }
            for p in pods
        ],
        "total_pods": len(pods),
    }


@app.get("/queue", tags=["Queue"])
def queue_status():
    """Return current queue depth and job counts by status."""
    return controller.queue.queue_depth()
