"""
resource_monitor.py
Watches Kubernetes pod resource usage via the metrics-server API.
Falls back to pod spec requests when metrics-server is unavailable.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger("allocator.monitor")


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class PodMetrics:
    name:          str
    namespace:     str
    cpu_usage_m:   float   # millicores
    mem_usage_mi:  float   # MiB
    cpu_limit_m:   float   # millicores (from spec)
    mem_limit_mi:  float   # MiB (from spec)
    cpu_request_m: float
    mem_request_mi: float
    phase:         str = "Unknown"
    labels:        dict = field(default_factory=dict)

    @property
    def cpu_utilization(self) -> float:
        """CPU usage / limit (0-1). Returns 0 if limit is 0."""
        return self.cpu_usage_m / self.cpu_limit_m if self.cpu_limit_m > 0 else 0.0

    @property
    def mem_utilization(self) -> float:
        return self.mem_usage_mi / self.mem_limit_mi if self.mem_limit_mi > 0 else 0.0


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_cpu(cpu_str: str) -> float:
    """Convert K8s CPU string to millicores. e.g. '250m' → 250, '1' → 1000."""
    if cpu_str is None:
        return 0.0
    cpu_str = str(cpu_str).strip()
    if cpu_str.endswith("m"):
        return float(cpu_str[:-1])
    if cpu_str.endswith("n"):          # nanocores (from metrics-server)
        return float(cpu_str[:-1]) / 1_000_000
    try:
        return float(cpu_str) * 1000   # cores → millicores
    except ValueError:
        return 0.0


def _parse_mem(mem_str: str) -> float:
    """Convert K8s memory string to MiB. e.g. '256Mi' → 256, '1Gi' → 1024."""
    if mem_str is None:
        return 0.0
    mem_str = str(mem_str).strip()
    units = {
        "Ki": 1 / 1024,
        "Mi": 1.0,
        "Gi": 1024.0,
        "Ti": 1024.0 ** 2,
        "k":  1 / 1024,
        "M":  1.0,
        "G":  1024.0,
    }
    for suffix, factor in units.items():
        if mem_str.endswith(suffix):
            try:
                return float(mem_str[:-len(suffix)]) * factor
            except ValueError:
                return 0.0
    try:
        return float(mem_str) / (1024 * 1024)   # bytes → MiB
    except ValueError:
        return 0.0


# ── ResourceMonitor ────────────────────────────────────────────────────────────

class ResourceMonitor:
    """
    Queries the Kubernetes metrics-server for live pod CPU/memory usage.
    Gracefully degrades to returning zeros when metrics-server is unavailable.
    """

    METRICS_API   = "metrics.k8s.io"
    METRICS_VER   = "v1beta1"
    METRICS_PATH  = "/apis/metrics.k8s.io/v1beta1/namespaces/{ns}/pods"

    def __init__(self, namespace: str = "ml-workloads", in_cluster: bool = False):
        self.namespace   = namespace
        self.in_cluster  = in_cluster
        self._load_config()
        self.core_v1     = client.CoreV1Api()
        self.custom_api  = client.CustomObjectsApi()

    def _load_config(self):
        try:
            if self.in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config()
        except Exception as e:
            logger.warning(f"Could not load kubeconfig: {e}")

    # ── Pod spec resources ────────────────────────────────────────────────────

    def get_pod_specs(self) -> dict[str, dict]:
        """Return {pod_name: {cpu_request_m, mem_request_mi, cpu_limit_m, mem_limit_mi, phase, labels}}"""
        specs = {}
        try:
            pods = self.core_v1.list_namespaced_pod(namespace=self.namespace)
            for pod in pods.items:
                name = pod.metadata.name
                phase = pod.status.phase or "Unknown"
                labels = pod.metadata.labels or {}
                cpu_req = mem_req = cpu_lim = mem_lim = 0.0
                for container in (pod.spec.containers or []):
                    res = container.resources
                    if res:
                        if res.requests:
                            cpu_req += _parse_cpu(res.requests.get("cpu"))
                            mem_req += _parse_mem(res.requests.get("memory"))
                        if res.limits:
                            cpu_lim += _parse_cpu(res.limits.get("cpu"))
                            mem_lim += _parse_mem(res.limits.get("memory"))
                specs[name] = {
                    "cpu_request_m":  cpu_req,
                    "mem_request_mi": mem_req,
                    "cpu_limit_m":    cpu_lim,
                    "mem_limit_mi":   mem_lim,
                    "phase":          phase,
                    "labels":         labels,
                }
        except ApiException as e:
            logger.error(f"Failed to list pods: {e}")
        return specs

    # ── Metrics-server usage ──────────────────────────────────────────────────

    def get_pod_usage(self) -> dict[str, tuple[float, float]]:
        """
        Return {pod_name: (cpu_millicores, mem_mib)} from metrics-server.
        Returns empty dict if metrics-server is not available.
        """
        usage = {}
        try:
            result = self.custom_api.list_namespaced_custom_object(
                group=self.METRICS_API,
                version=self.METRICS_VER,
                namespace=self.namespace,
                plural="pods",
            )
            for item in result.get("items", []):
                name = item["metadata"]["name"]
                cpu_total = mem_total = 0.0
                for container in item.get("containers", []):
                    cpu_total += _parse_cpu(container["usage"].get("cpu", "0"))
                    mem_total += _parse_mem(container["usage"].get("memory", "0"))
                usage[name] = (cpu_total, mem_total)
        except ApiException as e:
            if e.status == 404:
                logger.debug("metrics-server not available — usage will be zero")
            else:
                logger.warning(f"Metrics API error: {e}")
        except Exception as e:
            logger.debug(f"metrics-server unavailable: {e}")
        return usage

    # ── Combined snapshot ────────────────────────────────────────────────────

    def snapshot(self) -> list[PodMetrics]:
        """Return a full snapshot of all pods with usage + spec resources."""
        specs = self.get_pod_specs()
        usage = self.get_pod_usage()

        metrics = []
        for pod_name, spec in specs.items():
            cpu_use, mem_use = usage.get(pod_name, (0.0, 0.0))
            metrics.append(PodMetrics(
                name=pod_name,
                namespace=self.namespace,
                cpu_usage_m=cpu_use,
                mem_usage_mi=mem_use,
                cpu_limit_m=spec["cpu_limit_m"],
                mem_limit_mi=spec["mem_limit_mi"],
                cpu_request_m=spec["cpu_request_m"],
                mem_request_mi=spec["mem_request_mi"],
                phase=spec["phase"],
                labels=spec["labels"],
            ))
        return metrics

    def metrics_server_available(self) -> bool:
        """Quick check — returns True if metrics-server responds."""
        try:
            self.custom_api.list_namespaced_custom_object(
                group=self.METRICS_API, version=self.METRICS_VER,
                namespace=self.namespace, plural="pods",
            )
            return True
        except Exception:
            return False
