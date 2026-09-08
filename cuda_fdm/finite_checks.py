"""Fail-closed numerical checks shared by regression and experiment gates.

Never use max(previous_error, nan): Python can keep previous_error and report PASS.
These checks intentionally inspect inputs before observation sanitization/comparison.
"""
from __future__ import annotations

import math
import json
import os
import time
import numbers
from pathlib import Path
import numpy as np
import torch


def require_finite(value, label="value"):
    """Raise on any NaN/Inf, including matching NaNs and matching infinities."""
    # Checkpoints contain optional states, labels and 128-bit RNG integers as
    # well as numeric tensors. Integer RNG state is finite without float casts.
    if value is None or isinstance(value, (str, bytes, numbers.Integral)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
        return
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite {label}")
    else:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise FloatingPointError(f"non-finite {label}")


def finite_max_abs_error(actual, expected, label="comparison"):
    """A finite, shape-exact maximum error, never an ignored NaN reduction."""
    require_finite(actual, f"{label}.actual")
    require_finite(expected, f"{label}.expected")
    if torch.is_tensor(actual):
        actual = actual.detach().cpu().numpy()
    if torch.is_tensor(expected):
        expected = expected.detach().cpu().numpy()
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch {label}: {actual.shape} != {expected.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(actual - expected)
    require_finite(difference, f"{label}.difference")
    return float(difference.max()) if difference.size else 0.0


def require_finite_training_stats(stats):
    """Missing episode aggregates may be NaN only when no episode completed."""
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
             "explained_variance", "steps_per_sec", "elapsed_sec", "completed_episodes")
    for name in names:
        if not math.isfinite(float(getattr(stats, name))):
            raise FloatingPointError(f"non-finite PPO metric {name} at iteration {stats.iteration}")
    if stats.completed_episodes > 0:
        for name in ("mean_return", "mean_length", "win_rate"):
            if not math.isfinite(float(getattr(stats, name))):
                raise FloatingPointError(f"non-finite episode metric {name} at iteration {stats.iteration}")


def record_integrity_failure(folder, error, phase):
    """Durable failure marker consumed by the experiment manager; no auto retry."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "INTEGRITY_FAILURE.json"
    value = dict(phase=str(phase), error_type=type(error).__name__, error=str(error),
                 pid=os.getpid(), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                 requires_review=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path
