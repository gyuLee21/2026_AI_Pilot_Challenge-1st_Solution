"""Observe-only PPO/data-health diagnostics for long-horizon training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class HealthReport:
    iteration: int
    healthy: bool
    complete: bool
    alerts: tuple[str, ...]
    missing: tuple[str, ...]
    metrics: dict
    automatic_action: None = None

    def to_dict(self) -> dict:
        return asdict(self)


class PPOHealthMonitor:
    REQUIRED = (
        "approx_kl", "clipfrac", "actor_grad_norm", "critic_grad_norm",
        "entropy", "explained_variance", "fresh_fraction", "policy_lag",
    )

    def __init__(self, config):
        self.config = config
        self.config.validate()

    def evaluate(self, iteration: int, metrics: dict) -> HealthReport:
        clean = {}
        missing = []
        for name in self.REQUIRED:
            if name not in metrics or metrics[name] is None:
                missing.append(name)
                continue
            value = float(metrics[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite PPO health metric: {name}")
            clean[name] = value
        # Optional per-action-head diagnostics and ratio quantiles remain in the
        # receipt when supplied by the full vNext trainer probe.
        for name, value in metrics.items():
            if name in clean or value is None or isinstance(value, (str, bool, dict, list, tuple)):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                raise FloatingPointError(f"non-finite PPO health metric: {name}")
            clean[name] = numeric

        alerts = []
        if missing:
            alerts.append("required_health_metrics_missing")
        if clean.get("approx_kl", 0.0) > self.config.maximum_policy_kl:
            alerts.append("policy_kl_high")
        if clean.get("clipfrac", 0.0) > self.config.maximum_clip_fraction:
            alerts.append("clip_fraction_high")
        if "support_retention" in clean and clean["support_retention"] < self.config.minimum_support_retention:
            alerts.append("action_support_contraction")
        if clean.get("policy_lag", 0.0) > self.config.maximum_policy_lag:
            alerts.append("policy_lag_high")
        if clean.get("fresh_fraction", 1.0) < self.config.minimum_fresh_fraction:
            alerts.append("fresh_data_fraction_low")
        if clean.get("actor_grad_norm", 1.0) <= 1e-10:
            alerts.append("actor_gradient_vanished")
        if clean.get("critic_grad_norm", 1.0) <= 1e-10:
            alerts.append("critic_gradient_vanished")
        return HealthReport(
            int(iteration), not alerts and not missing, not missing,
            tuple(alerts), tuple(missing), clean, None)


__all__ = ["HealthReport", "PPOHealthMonitor"]
