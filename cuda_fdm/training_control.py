"""Operational horizon/deadline controls; never change PPO or league rules."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time


def retarget_resume_state(state, *, target, completed, saved_main_target=None,
                          allow_change=False):
    """Explicitly retarget only control metadata; keep all evidence and samples."""
    target, completed = int(target), int(completed)
    if target < 1 or target < completed:
        raise ValueError("target iterations must be positive and not precede the checkpoint")
    old_targets = {}
    if saved_main_target is not None:
        old_targets['main'] = int(saved_main_target)
    if state is not None:
        old_targets['vnext'] = int(state['config']['target_iteration'])
        if state.get('wallclock_budget'):
            old_targets['budget'] = int(state['wallclock_budget']['target_iteration'])
    changed = {k:v for k,v in old_targets.items() if v != target}
    if changed and not allow_change:
        raise ValueError("target changed across resume; pass --accept-target-change explicitly")
    if not changed:
        return state, None
    updated = copy.deepcopy(state)
    if updated is not None:
        updated['config']['target_iteration'] = target
        if updated.get('wallclock_budget'):
            updated['wallclock_budget']['target_iteration'] = target
    return updated, dict(previous_targets=old_targets, target_iteration=target,
                         completed_iteration=completed)


@dataclass(frozen=True)
class DeadlineGuard:
    deadline: str
    reserve_seconds: float = 1800.
    milestone_floor_seconds: float = 3600.
    safety_factor: float = 1.5

    def __post_init__(self):
        stamp = datetime.fromisoformat(self.deadline.replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("deadline must include a timezone, e.g. 2026-09-13T16:00:00+09:00")
        if (not all(math.isfinite(v) for v in (self.reserve_seconds,
                     self.milestone_floor_seconds, self.safety_factor))
                or self.reserve_seconds < 0 or self.milestone_floor_seconds <= 0
                or self.safety_factor < 1):
            raise ValueError("invalid deadline safety settings")

    @property
    def epoch(self):
        return datetime.fromisoformat(self.deadline.replace('Z', '+00:00')).timestamp()

    def state_dict(self):
        return asdict(self)

    def check(self, *, next_iteration, target_iteration, milestone_period, budget, now=None):
        """Refuse a whole iteration before mutation, especially its milestone.

        Estimated durations are not a hard upper bound. We never interrupt or
        skip a partially started milestone to pretend a deadline was met.
        """
        now = time.time() if now is None else float(now)
        summary = budget.report(iteration=next_iteration-1,
                                milestone_period=milestone_period)['seconds']
        main = summary['main']['p95']
        main = 10. if main is None else max(1., main * self.safety_factor)
        extra = max(self.milestone_floor_seconds, self.safety_factor * (
            (summary['evaluation']['p95'] or 0.) + (summary['side']['p95'] or 0.)))
        period = int(milestone_period)
        milestone = period > 0 and next_iteration % period == 0
        available = self.epoch - now - self.reserve_seconds
        required = main + (extra if milestone else 0.)
        # Read-only recommendation; never silently alter the configured target.
        completed = next_iteration-1
        lo, hi = completed, max(completed, int(target_iteration))
        while lo < hi:
            mid = (lo + hi + 1)//2
            count = max(0,mid//period-completed//period) if period > 0 else 0
            cost = (mid-completed)*main + count*extra
            if cost <= available: lo = mid
            else: hi = mid-1
        return dict(allow=available >= required, next_iteration=int(next_iteration),
                    is_milestone=milestone, deadline_utc=datetime.fromtimestamp(
                        self.epoch,timezone.utc).isoformat(),
                    remaining_seconds=self.epoch-now, reserve_seconds=self.reserve_seconds,
                    estimated_next_seconds=required, estimated_safe_target_iteration=lo,
                    estimate_is_hard_bound=False)


def resolve_deadline(*, saved=None, deadline=None, clear=False, reserve=None,
                     milestone_floor=None, safety_factor=None):
    if clear:
        if deadline is not None:
            raise ValueError("--deadline and --clear-deadline are mutually exclusive")
        return None
    values = dict(saved or {})
    if deadline is not None: values['deadline'] = deadline
    if 'deadline' not in values:
        if any(v is not None for v in (reserve,milestone_floor,safety_factor)):
            raise ValueError("deadline safety settings require --deadline or a saved deadline")
        return None
    for name,value in (('reserve_seconds',reserve),('milestone_floor_seconds',milestone_floor),
                       ('safety_factor',safety_factor)):
        if value is not None: values[name] = float(value)
    return DeadlineGuard(**values)


class RunLease:
    """OS-owned per-run lock, released on every exit including process death."""
    def __init__(self, root, *, resume):
        self.root, self.resume, self.stream = Path(root).resolve(), bool(resume), None

    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=True)
        lock = self.root/'.training.lock'
        stream = lock.open('a+b')
        try:
            stream.seek(0,2)
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError(f"another training process owns this run: {self.root}") from exc
        self.stream = stream
        try:
            if not self.resume and any(p.is_file() and p != lock for p in self.root.rglob('*')):
                raise ValueError("fresh run directory contains files; use resume or a new scenario/run root")
            if (self.root/'STOP').exists():
                raise ValueError("STOP exists; inspect and remove it explicitly before launch")
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*args):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        # Keep the lock inode/path: unlinking permits a racing second owner.
