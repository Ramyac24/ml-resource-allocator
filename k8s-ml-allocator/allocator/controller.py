"""
controller.py
Main ML Resource Allocator control loop.

Every POLL_INTERVAL seconds:
  1. Dequeue the next pending MLJob and deploy it to K8s
  2. Snapshot current pod resource usage
  3. Run scaling decisions and patch deployments
  4. Print a live Rich table to the terminal
"""

import logging
import os
import sys
import time
import threading
from typing import Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box

from allocator.resource_monitor import ResourceMonitor, PodMetrics
from allocator.scaler import DynamicScaler, ScaleDecision
from allocator.priority_queue import JobQueue, MLJob, Priority

logger  = logging.getLogger("allocator.controller")
console = Console()

NAMESPACE     = os.getenv("ML_NAMESPACE",    "ml-workloads")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))    # seconds
IN_CLUSTER    = os.getenv("IN_CLUSTER", "false").lower() == "true"


class MLAllocatorController:
    """
    Orchestrates job dispatch + dynamic resource scaling for ML workloads.
    """

    def __init__(self, namespace: str = NAMESPACE):
        self.namespace = namespace
        self._load_k8s_config()
        self.batch_v1   = client.BatchV1Api()
        self.apps_v1    = client.AppsV1Api()
        self.core_v1    = client.CoreV1Api()
        self.monitor    = ResourceMonitor(namespace=namespace)
        self.scaler     = DynamicScaler(namespace=namespace)
        self.queue      = JobQueue(max_concurrent=3)
        self._running   = False
        self._iteration = 0

    def _load_k8s_config(self):
        try:
            if IN_CLUSTER:
                config.load_incluster_config()
                logger.info("Loaded in-cluster kubeconfig")
            else:
                config.load_kube_config()
                logger.info("Loaded local kubeconfig")
        except Exception as e:
            logger.error(f"Cannot load kubeconfig: {e}")
            sys.exit(1)

    # ── Job dispatch ──────────────────────────────────────────────────────────

    def _dispatch_job(self, job: MLJob):
        """Create a K8s Job or Deployment depending on job_type."""
        if job.job_type == "training":
            self._create_k8s_job(job)
        else:
            self._create_k8s_deployment(job)

    def _create_k8s_job(self, job: MLJob):
        """Submit a one-shot K8s Job for a training workload."""
        env_vars = [
            client.V1EnvVar(name=k, value=str(v))
            for k, v in job.env.items()
        ]
        body = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=f"mljob-{job.job_id}",
                namespace=self.namespace,
                labels={
                    "app":      f"mljob-{job.job_id}",
                    "ml-type":  "training",
                    "priority": job.priority_name.lower(),
                    "job-id":   job.job_id,
                },
            ),
            spec=client.V1JobSpec(
                ttl_seconds_after_finished=300,
                backoff_limit=2,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={
                        "app":     f"mljob-{job.job_id}",
                        "ml-type": "training",
                    }),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[client.V1Container(
                            name=f"mljob-{job.job_id}",
                            image=job.image,
                            image_pull_policy="IfNotPresent",
                            env=env_vars,
                            resources=client.V1ResourceRequirements(
                                requests={
                                    "cpu":    f"{job.cpu_request_m}m",
                                    "memory": f"{job.mem_request_mi}Mi",
                                },
                                limits={
                                    "cpu":    f"{job.cpu_limit_m}m",
                                    "memory": f"{job.mem_limit_mi}Mi",
                                },
                            ),
                        )],
                    ),
                ),
            ),
        )
        try:
            self.batch_v1.create_namespaced_job(namespace=self.namespace, body=body)
            logger.info(f"Created K8s Job: mljob-{job.job_id} ({job.name})")
        except ApiException as e:
            if e.status == 409:
                logger.debug(f"Job mljob-{job.job_id} already exists")
            else:
                logger.error(f"Failed to create job: {e}")
                self.queue.mark_failed(job.job_id)

    def _create_k8s_deployment(self, job: MLJob):
        """Create a K8s Deployment for an inference serving workload."""
        env_vars = [
            client.V1EnvVar(name=k, value=str(v))
            for k, v in job.env.items()
        ]
        body = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"mlinf-{job.job_id}",
                namespace=self.namespace,
                labels={
                    "app":     f"mlinf-{job.job_id}",
                    "ml-type": "inference",
                    "job-id":  job.job_id,
                },
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"mlinf-{job.job_id}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"mlinf-{job.job_id}", "ml-type": "inference"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[client.V1Container(
                            name=f"mlinf-{job.job_id}",
                            image=job.image,
                            image_pull_policy="IfNotPresent",
                            args=["ml_workloads/sample_inference.py"],
                            ports=[client.V1ContainerPort(container_port=5000)],
                            env=env_vars,
                            resources=client.V1ResourceRequirements(
                                requests={
                                    "cpu":    f"{job.cpu_request_m}m",
                                    "memory": f"{job.mem_request_mi}Mi",
                                },
                                limits={
                                    "cpu":    f"{job.cpu_limit_m}m",
                                    "memory": f"{job.mem_limit_mi}Mi",
                                },
                            ),
                        )],
                    ),
                ),
            ),
        )
        try:
            self.apps_v1.create_namespaced_deployment(namespace=self.namespace, body=body)
            logger.info(f"Created K8s Deployment: mlinf-{job.job_id} ({job.name})")
        except ApiException as e:
            if e.status == 409:
                logger.debug(f"Deployment mlinf-{job.job_id} already exists")
            else:
                logger.error(f"Failed to create deployment: {e}")
                self.queue.mark_failed(job.job_id)

    # ── Control loop ──────────────────────────────────────────────────────────

    def _tick(self):
        """One iteration of the control loop."""
        self._iteration += 1

        # 1. Dispatch queued jobs
        job = self.queue.dequeue()
        if job:
            self._dispatch_job(job)

        # 2. Snapshot metrics
        pod_metrics = self.monitor.snapshot()

        # 3. Scale decisions
        decisions = self.scaler.evaluate(pod_metrics)
        applied   = self.scaler.apply_all(decisions)
        if applied:
            logger.info(f"Applied {applied} scaling action(s)")

        # 4. Sync job statuses from K8s
        self._sync_job_statuses()

        return pod_metrics, decisions

    def _sync_job_statuses(self):
        """Update queue statuses based on actual K8s job/pod states."""
        try:
            jobs = self.batch_v1.list_namespaced_job(namespace=self.namespace)
            for kj in jobs.items:
                job_id = kj.metadata.labels.get("job-id") if kj.metadata.labels else None
                if not job_id:
                    continue
                q_job = self.queue.get_job(job_id)
                if not q_job:
                    continue
                if kj.status.succeeded and kj.status.succeeded > 0:
                    self.queue.mark_completed(job_id)
                elif kj.status.failed and kj.status.failed > 0:
                    self.queue.mark_failed(job_id)
        except ApiException:
            pass

    # ── Rich display ──────────────────────────────────────────────────────────

    def _build_table(self, pod_metrics: list[PodMetrics],
                     decisions: list[ScaleDecision]) -> Table:
        table = Table(
            title=f"[bold cyan]ML Allocator — iteration {self._iteration}[/bold cyan]",
            box=box.ROUNDED, show_header=True, header_style="bold magenta",
        )
        table.add_column("Pod",        style="cyan",  no_wrap=True)
        table.add_column("Phase",      style="green")
        table.add_column("CPU Use",    justify="right")
        table.add_column("CPU Lim",    justify="right")
        table.add_column("CPU Util",   justify="right")
        table.add_column("Mem Use",    justify="right")
        table.add_column("Mem Lim",    justify="right")
        table.add_column("Decision",   style="yellow")

        decision_map = {d.pod_name: d for d in decisions}

        for pm in pod_metrics:
            dec = decision_map.get(pm.name)
            action_str = ""
            if dec and dec.action != "none":
                action_str = (
                    f"[green]▲ {dec.action}[/green]"
                    if dec.action == "scale_up"
                    else f"[red]▼ {dec.action}[/red]"
                )

            cpu_util_str = f"{pm.cpu_utilization:.0%}"
            if pm.cpu_utilization > 0.75:
                cpu_util_str = f"[red]{cpu_util_str}[/red]"
            elif pm.cpu_utilization < 0.25:
                cpu_util_str = f"[dim]{cpu_util_str}[/dim]"

            table.add_row(
                pm.name[:40],
                pm.phase,
                f"{pm.cpu_usage_m:.0f}m",
                f"{pm.cpu_limit_m:.0f}m",
                cpu_util_str,
                f"{pm.mem_usage_mi:.0f}Mi",
                f"{pm.mem_limit_mi:.0f}Mi",
                action_str or "—",
            )

        # Queue summary footer
        q = self.queue.queue_depth()
        table.caption = (
            f"Queue: {q['queued']} queued | {q['running']} running | "
            f"{q['completed']} completed | {q['failed']} failed"
        )
        return table

    # ── Public interface ──────────────────────────────────────────────────────

    def submit(self, job: MLJob) -> str:
        """Add a job to the queue. Returns job_id."""
        return self.queue.enqueue(job)

    def start(self, once: bool = False):
        """
        Start the control loop.
        once=True runs a single tick (useful for testing).
        """
        self._running = True
        console.print(
            f"[bold green]ML Allocator started[/bold green] — "
            f"namespace=[cyan]{self.namespace}[/cyan] "
            f"poll=[cyan]{POLL_INTERVAL}s[/cyan]"
        )

        metrics_ok = self.monitor.metrics_server_available()
        if not metrics_ok:
            console.print(
                "[yellow]⚠  metrics-server not available — "
                "run: minikube addons enable metrics-server[/yellow]"
            )

        if once:
            pod_metrics, decisions = self._tick()
            console.print(self._build_table(pod_metrics, decisions))
            return

        with Live(console=console, refresh_per_second=0.5) as live:
            while self._running:
                try:
                    pod_metrics, decisions = self._tick()
                    live.update(self._build_table(pod_metrics, decisions))
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Controller error: {e}", exc_info=True)
                time.sleep(POLL_INTERVAL)

        console.print("[bold red]Allocator stopped.[/bold red]")

    def stop(self):
        self._running = False


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="ML Resource Allocator Controller")
    parser.add_argument("--namespace", default=NAMESPACE,
                        help=f"K8s namespace to watch (default: {NAMESPACE})")
    parser.add_argument("--once", action="store_true",
                        help="Run a single tick and exit")
    args = parser.parse_args()

    ctrl = MLAllocatorController(namespace=args.namespace)

    ctrl.submit(MLJob.create(
        name="iris-training", job_type="training", image="ml-workload:latest",
        priority=Priority.HIGH, cpu_request_m=200, mem_request_mi=256,
        cpu_limit_m=500, mem_limit_mi=512,
    ))
    ctrl.submit(MLJob.create(
        name="iris-inference", job_type="inference", image="ml-workload:latest",
        priority=Priority.MEDIUM, cpu_request_m=100, mem_request_mi=128,
        cpu_limit_m=250, mem_limit_mi=256,
    ))

    ctrl.start(once=args.once)

