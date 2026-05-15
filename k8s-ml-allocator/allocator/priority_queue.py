"""
priority_queue.py
In-memory priority queue for ML job submissions.
Jobs are ordered: HIGH > MEDIUM > LOW.
When the cluster is at capacity, low-priority jobs wait.
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("allocator.queue")


class Priority(IntEnum):
    HIGH   = 0    # lowest heap value = highest priority
    MEDIUM = 1
    LOW    = 2


JOB_TYPES = ("training", "inference")


@dataclass(order=True)
class MLJob:
    priority:   int             # Priority enum value (used for heap ordering)
    submitted:  float = field(compare=False)   # timestamp for FIFO within same priority
    job_id:     str   = field(compare=False)
    name:       str   = field(compare=False)
    job_type:   str   = field(compare=False)   # "training" | "inference"
    image:      str   = field(compare=False)
    cpu_request_m:   int  = field(compare=False, default=250)
    mem_request_mi:  int  = field(compare=False, default=256)
    cpu_limit_m:     int  = field(compare=False, default=500)
    mem_limit_mi:    int  = field(compare=False, default=512)
    env:        dict  = field(compare=False, default_factory=dict)
    status:     str   = field(compare=False, default="queued")
    # "queued" | "running" | "completed" | "failed" | "cancelled"

    @classmethod
    def create(cls, name: str, job_type: str, image: str,
               priority: Priority = Priority.MEDIUM,
               cpu_request_m: int = 250, mem_request_mi: int = 256,
               cpu_limit_m: int = 500, mem_limit_mi: int = 512,
               env: Optional[dict] = None) -> "MLJob":
        return cls(
            priority=int(priority),
            submitted=time.time(),
            job_id=str(uuid4())[:8],
            name=name,
            job_type=job_type,
            image=image,
            cpu_request_m=cpu_request_m,
            mem_request_mi=mem_request_mi,
            cpu_limit_m=cpu_limit_m,
            mem_limit_mi=mem_limit_mi,
            env=env or {},
        )

    @property
    def priority_name(self) -> str:
        return Priority(self.priority).name

    def to_dict(self) -> dict:
        return {
            "job_id":          self.job_id,
            "name":            self.name,
            "job_type":        self.job_type,
            "priority":        self.priority_name,
            "image":           self.image,
            "cpu_request_m":   self.cpu_request_m,
            "mem_request_mi":  self.mem_request_mi,
            "cpu_limit_m":     self.cpu_limit_m,
            "mem_limit_mi":    self.mem_limit_mi,
            "status":          self.status,
            "submitted_ts":    self.submitted,
        }


class JobQueue:
    """
    Thread-safe priority queue for ML jobs.
    Uses a min-heap: (priority, submitted_time, MLJob)
    """

    def __init__(self, max_concurrent: int = 3):
        self._heap:    list  = []
        self._all:     dict[str, MLJob] = {}
        self._lock     = threading.Lock()
        self.max_concurrent = max_concurrent

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def enqueue(self, job: MLJob) -> str:
        with self._lock:
            heapq.heappush(self._heap, (job.priority, job.submitted, job.job_id))
            self._all[job.job_id] = job
            logger.info(f"Queued [{job.priority_name}] {job.name} ({job.job_id})")
        return job.job_id

    # ── Dequeue ───────────────────────────────────────────────────────────────

    def dequeue(self) -> Optional[MLJob]:
        """Pop the highest-priority pending job."""
        with self._lock:
            running = sum(1 for j in self._all.values() if j.status == "running")
            if running >= self.max_concurrent:
                logger.debug(f"At capacity ({running}/{self.max_concurrent}) — queue holds")
                return None

            while self._heap:
                _, _, job_id = heapq.heappop(self._heap)
                job = self._all.get(job_id)
                if job and job.status == "queued":
                    job.status = "running"
                    logger.info(f"Dispatching [{job.priority_name}] {job.name} ({job.job_id})")
                    return job
        return None

    # ── Status updates ────────────────────────────────────────────────────────

    def mark_completed(self, job_id: str):
        with self._lock:
            if job_id in self._all:
                self._all[job_id].status = "completed"

    def mark_failed(self, job_id: str):
        with self._lock:
            if job_id in self._all:
                self._all[job_id].status = "failed"

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._all.get(job_id)
            if job and job.status == "queued":
                job.status = "cancelled"
                return True
        return False

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[MLJob]:
        return self._all.get(job_id)

    def list_jobs(self, status: Optional[str] = None) -> list[MLJob]:
        with self._lock:
            jobs = list(self._all.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: (j.priority, j.submitted))

    def queue_depth(self) -> dict:
        with self._lock:
            all_jobs = list(self._all.values())
        return {
            "queued":    sum(1 for j in all_jobs if j.status == "queued"),
            "running":   sum(1 for j in all_jobs if j.status == "running"),
            "completed": sum(1 for j in all_jobs if j.status == "completed"),
            "failed":    sum(1 for j in all_jobs if j.status == "failed"),
            "total":     len(all_jobs),
        }
