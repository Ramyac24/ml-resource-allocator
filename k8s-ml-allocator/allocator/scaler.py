"""
scaler.py
Dynamic resource scaling logic for ML workloads.

Scale-Up  : CPU util > HIGH_THRESHOLD  → increase limits by SCALE_UP_FACTOR
Scale-Down: CPU util < LOW_THRESHOLD   → decrease limits by SCALE_DOWN_FACTOR
Caps      : per-pod min/max limits enforced
"""

import logging
from dataclasses import dataclass
from typing import Optional

from kubernetes import client
from kubernetes.client.rest import ApiException

from allocator.resource_monitor import PodMetrics

logger = logging.getLogger("allocator.scaler")


# ── Thresholds & factors ───────────────────────────────────────────────────────
CPU_HIGH_THRESHOLD  = 0.75   # scale up when CPU util > 75%
CPU_LOW_THRESHOLD   = 0.25   # scale down when CPU util < 25%
MEM_HIGH_THRESHOLD  = 0.80
MEM_LOW_THRESHOLD   = 0.30

SCALE_UP_FACTOR     = 1.30   # +30%
SCALE_DOWN_FACTOR   = 0.80   # -20%

# Per-pod hard limits (millicores / MiB)
CPU_MIN_M   = 100
CPU_MAX_M   = 2000
MEM_MIN_MI  = 128
MEM_MAX_MI  = 2048

# How many consecutive high/low readings before acting
COOLDOWN_CYCLES = 2


@dataclass
class ScaleDecision:
    pod_name:       str
    namespace:      str
    action:         str          # "scale_up" | "scale_down" | "none"
    reason:         str
    new_cpu_limit:  Optional[float] = None   # millicores
    new_mem_limit:  Optional[float] = None   # MiB
    applied:        bool = False


class DynamicScaler:
    """
    Evaluates PodMetrics snapshots and patches K8s resource limits
    for pods that are over- or under-utilised.
    """

    def __init__(self, namespace: str = "ml-workloads"):
        self.namespace = namespace
        self.apps_v1   = client.AppsV1Api()
        self.core_v1   = client.CoreV1Api()
        # cooldown counters: pod_name → consecutive high/low count
        self._high_counts: dict[str, int] = {}
        self._low_counts:  dict[str, int] = {}

    # ── Decision logic ─────────────────────────────────────────────────────────

    def evaluate(self, metrics: list[PodMetrics]) -> list[ScaleDecision]:
        """Return a ScaleDecision for every running ML pod."""
        decisions = []
        for pm in metrics:
            if pm.phase not in ("Running", "Pending"):
                continue
            if not pm.labels.get("app"):
                continue   # skip non-app pods

            decisions.append(self._decide(pm))
        return decisions

    def _decide(self, pm: PodMetrics) -> ScaleDecision:
        name = pm.name

        # Update cooldown counters
        if pm.cpu_utilization > CPU_HIGH_THRESHOLD or pm.mem_utilization > MEM_HIGH_THRESHOLD:
            self._high_counts[name] = self._high_counts.get(name, 0) + 1
            self._low_counts[name]  = 0
        elif pm.cpu_utilization < CPU_LOW_THRESHOLD and pm.mem_utilization < MEM_LOW_THRESHOLD:
            self._low_counts[name]  = self._low_counts.get(name, 0) + 1
            self._high_counts[name] = 0
        else:
            self._high_counts[name] = 0
            self._low_counts[name]  = 0

        # Scale up?
        if self._high_counts.get(name, 0) >= COOLDOWN_CYCLES:
            new_cpu = min(pm.cpu_limit_m   * SCALE_UP_FACTOR, CPU_MAX_M)
            new_mem = min(pm.mem_limit_mi  * SCALE_UP_FACTOR, MEM_MAX_MI)
            self._high_counts[name] = 0
            return ScaleDecision(
                pod_name=name, namespace=self.namespace,
                action="scale_up",
                reason=f"CPU util {pm.cpu_utilization:.0%} / Mem util {pm.mem_utilization:.0%} exceeded thresholds",
                new_cpu_limit=round(new_cpu), new_mem_limit=round(new_mem),
            )

        # Scale down?
        if self._low_counts.get(name, 0) >= COOLDOWN_CYCLES:
            new_cpu = max(pm.cpu_limit_m   * SCALE_DOWN_FACTOR, CPU_MIN_M)
            new_mem = max(pm.mem_limit_mi  * SCALE_DOWN_FACTOR, MEM_MIN_MI)
            self._low_counts[name] = 0
            return ScaleDecision(
                pod_name=name, namespace=self.namespace,
                action="scale_down",
                reason=f"CPU util {pm.cpu_utilization:.0%} / Mem util {pm.mem_utilization:.0%} below thresholds",
                new_cpu_limit=round(new_cpu), new_mem_limit=round(new_mem),
            )

        return ScaleDecision(
            pod_name=name, namespace=self.namespace,
            action="none",
            reason=f"CPU {pm.cpu_utilization:.0%} / Mem {pm.mem_utilization:.0%} — within bounds",
        )

    # ── Apply to Deployment ───────────────────────────────────────────────────

    def apply(self, decision: ScaleDecision) -> bool:
        """
        Patch the parent Deployment's container resource limits.
        Returns True if patched successfully.
        """
        if decision.action == "none":
            return False

        # Find which deployment owns this pod via label selector
        deployment_name = decision.pod_name.rsplit("-", 2)[0]  # strip pod hash suffix

        patch_body = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": deployment_name,
                            "resources": {
                                "limits": {
                                    "cpu":    f"{int(decision.new_cpu_limit)}m",
                                    "memory": f"{int(decision.new_mem_limit)}Mi",
                                },
                                "requests": {
                                    "cpu":    f"{int(decision.new_cpu_limit // 2)}m",
                                    "memory": f"{int(decision.new_mem_limit // 2)}Mi",
                                },
                            }
                        }]
                    }
                }
            }
        }

        try:
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=decision.namespace,
                body=patch_body,
            )
            decision.applied = True
            logger.info(
                f"[{decision.action.upper()}] {deployment_name} → "
                f"CPU {decision.new_cpu_limit}m / Mem {decision.new_mem_limit}Mi"
            )
            return True
        except ApiException as e:
            if e.status == 404:
                logger.debug(f"Deployment {deployment_name} not found (may be a Job pod)")
            else:
                logger.warning(f"Patch failed for {deployment_name}: {e}")
            return False

    def apply_all(self, decisions: list[ScaleDecision]) -> int:
        """Apply all non-'none' decisions. Returns count of applied patches."""
        return sum(1 for d in decisions if self.apply(d))

    # ── Manual override ───────────────────────────────────────────────────────

    def force_scale(self, deployment_name: str,
                    cpu_m: int, mem_mi: int) -> bool:
        """Manually set resource limits on a deployment."""
        d = ScaleDecision(
            pod_name=deployment_name,
            namespace=self.namespace,
            action="scale_up",
            reason="manual override",
            new_cpu_limit=cpu_m,
            new_mem_limit=mem_mi,
        )
        return self.apply(d)
