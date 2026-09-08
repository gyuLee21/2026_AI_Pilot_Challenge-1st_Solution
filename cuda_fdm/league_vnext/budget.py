"""Bounded wall-clock accounting and rolling 100K ETA receipts."""
from __future__ import annotations

from collections import deque
import math

import numpy as np


BUDGET_PROTOCOL = "active_league_wallclock_budget_v1"
QUANTILES = (0.50, 0.90, 0.95, 0.99)


class WallclockBudgetTracker:
    def __init__(self, *, target_iteration: int = 100000, window: int = 512,
                 state: dict | None = None):
        self.target_iteration = int(target_iteration)
        self.window = int(window)
        if self.target_iteration < 1 or self.window < 16:
            raise ValueError("invalid wall-clock budget tracker bounds")
        self.samples = {name: deque(maxlen=self.window)
                        for name in ("main", "evaluation", "side")}
        self.main_transitions = 0
        self.completed_games = 0
        if state is not None:
            self.load_state_dict(state)

    def add(self, category: str, seconds: float) -> None:
        if category not in self.samples:
            raise ValueError("unknown wall-clock category")
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("wall-clock sample must be finite and non-negative")
        self.samples[category].append(seconds)

    def observe_main(self, *, seconds: float, transitions: int,
                     completed_games: int) -> None:
        self.add("main", seconds)
        if int(transitions) < 0 or int(completed_games) < 0:
            raise ValueError("experience counters cannot be negative")
        self.main_transitions += int(transitions)
        self.completed_games += int(completed_games)

    @staticmethod
    def _summary(values) -> dict:
        if not values:
            return {"count": 0, **{f"p{int(q * 100):02d}": None for q in QUANTILES}}
        array = np.asarray(values, dtype=np.float64)
        result = {"count": int(array.size)}
        for quantile, value in zip(QUANTILES, np.quantile(array, QUANTILES)):
            result[f"p{int(quantile * 100):02d}"] = float(value)
        return result

    def report(self, *, iteration: int, milestone_period: int = 500) -> dict:
        summaries = {name: self._summary(values)
                     for name, values in self.samples.items()}
        remaining = max(0, self.target_iteration - int(iteration))
        main_p50 = summaries["main"]["p50"]
        eval_p95 = summaries["evaluation"]["p95"] or 0.0
        side_p95 = summaries["side"]["p95"] or 0.0
        period = int(milestone_period)
        # Count actual future boundaries, including the final target if it is
        # a milestone. ceil(remaining/period) overcounts non-multiple targets.
        milestones = (max(0, self.target_iteration // period - int(iteration) // period)
                      if period > 0 else 0)
        eta = None if main_p50 is None else (
            remaining * main_p50 + milestones * (eval_p95 + side_p95))
        return {
            "protocol": BUDGET_PROTOCOL, "iteration": int(iteration),
            "target_iteration": self.target_iteration,
            "main_transitions": self.main_transitions,
            "completed_games": self.completed_games,
            "seconds": summaries, "rolling_eta_seconds": eta,
            "remaining_main_iterations": remaining,
            "remaining_milestones": milestones,
        }

    def state_dict(self) -> dict:
        return {
            "protocol": BUDGET_PROTOCOL, "target_iteration": self.target_iteration,
            "window": self.window, "samples": {name: list(values)
                                                for name, values in self.samples.items()},
            "main_transitions": self.main_transitions,
            "completed_games": self.completed_games,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("protocol") != BUDGET_PROTOCOL:
            raise ValueError("wall-clock budget protocol mismatch")
        if (int(state.get("target_iteration", -1)) != self.target_iteration
                or int(state.get("window", -1)) != self.window):
            raise ValueError("wall-clock budget configuration changed across resume")
        for name in self.samples:
            values = [float(value) for value in state.get("samples", {}).get(name, [])]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("invalid saved wall-clock sample")
            self.samples[name].extend(values)
        self.main_transitions = int(state.get("main_transitions", 0))
        self.completed_games = int(state.get("completed_games", 0))


__all__ = ["BUDGET_PROTOCOL", "QUANTILES", "WallclockBudgetTracker"]
