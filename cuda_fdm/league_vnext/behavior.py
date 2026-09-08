"""Log-only tactical behaviour descriptors; never a reward term."""
from __future__ import annotations

import math

import numpy as np


FEATURES = (
    "altitude_mean_m", "altitude_std_m", "altitude_min_m", "low_altitude_fraction",
    "speed_mean_mps", "speed_std_mps", "specific_energy_mean", "energy_loss_rate",
    "load_factor_mean", "load_factor_p95", "turn_rate_mean", "reversal_rate",
    "vertical_speed_mean", "vertical_speed_std", "throttle_mean", "throttle_std",
    "range_mean_m", "range_min_m", "closure_rate_mean", "ata_mean_rad",
    "aa_mean_rad", "offensive_wez_fraction", "defensive_wez_fraction",
    "time_to_advantage_s", "overshoot_rate", "minimum_separation_m",
    "damage_dealt", "damage_received", "attack_events", "terminal_score",
    "altitude_loss", "timeout_fraction",
)


class BehaviorDescriptorAccumulator:
    """Weighted feature means plus an observation mask.

    Missing fields remain explicitly masked rather than being imputed as a
    tactical zero. The descriptor is intended for indexing and audit only.
    """

    def __init__(self, state: dict | None = None):
        size = len(FEATURES)
        self.weight = np.zeros(size, dtype=np.float64)
        self.total = np.zeros(size, dtype=np.float64)
        self.total_sq = np.zeros(size, dtype=np.float64)
        self.episodes = 0
        if state is not None:
            self.load_state_dict(state)

    def update(self, values: dict, *, weight: float = 1.0) -> None:
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("behaviour descriptor weight must be finite and positive")
        for index, name in enumerate(FEATURES):
            if name not in values or values[name] is None:
                continue
            value = float(values[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite behaviour feature: {name}")
            self.weight[index] += weight
            self.total[index] += weight * value
            self.total_sq[index] += weight * value * value
        self.episodes += 1

    def descriptor(self) -> list[float]:
        observed = self.weight > 0.0
        mean = np.divide(self.total, self.weight, out=np.zeros_like(self.total), where=observed)
        variance = np.divide(self.total_sq, self.weight, out=np.zeros_like(self.total),
                             where=observed) - mean * mean
        std = np.sqrt(np.maximum(variance, 0.0))
        mask = observed.astype(np.float64)
        value = np.concatenate([mean, std, mask])
        if np.any(~np.isfinite(value)):
            raise FloatingPointError("non-finite behaviour descriptor")
        return value.tolist()

    def state_dict(self) -> dict:
        return {
            "features": list(FEATURES), "episodes": self.episodes,
            "weight": self.weight.tolist(), "total": self.total.tolist(),
            "total_sq": self.total_sq.tolist(), "log_only": True,
        }

    def load_state_dict(self, state: dict) -> None:
        if tuple(state.get("features", ())) != FEATURES or state.get("log_only") is not True:
            raise ValueError("behaviour descriptor contract mismatch")
        self.episodes = int(state.get("episodes", 0))
        for name in ("weight", "total", "total_sq"):
            array = np.asarray(state[name], dtype=np.float64)
            if array.shape != (len(FEATURES),) or np.any(~np.isfinite(array)):
                raise ValueError("invalid saved behaviour descriptor")
            setattr(self, name, array.copy())


__all__ = ["FEATURES", "BehaviorDescriptorAccumulator"]
