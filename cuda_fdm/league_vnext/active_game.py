"""Bounded solver membership and deterministic historical counter audits.

This module deliberately has no policy-loading or evaluator code. Selection is
cheap and deterministic. Both migration-era records and retired archive-only
policies are auditable. Fresh candidates use the admission/successor path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


HISTORICAL_AUDIT_PROTOCOL = "pre20k_structured_counter_audit_v1"


@dataclass(frozen=True)
class HistoricalAuditTarget:
    archive_id: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def strategic_active_ids(entries, *, current_id: int, cap: int = 20) -> list[int]:
    """Return exactly the resident policies eligible for the empirical game.

    Current Main is included. The four recent snapshots and the latest clone
    are rollout curriculum only; including them would spend evaluator time on
    transient copies and can duplicate current Main.
    """
    result = [int(current_id)]
    for entry in entries:
        if entry.get("role") not in {"core", "challenger"}:
            continue
        identity = entry.get("archive_id")
        if identity is not None:
            result.append(int(identity))
    result = list(dict.fromkeys(result))
    if len(result) > int(cap):
        raise RuntimeError(
            f"strategic active game has {len(result)} policies; cap is {int(cap)}")
    return result


def _audit_age(record: dict) -> tuple[int, int]:
    metrics = record.get("metrics") or {}
    last = metrics.get("last_historical_audit_iteration")
    # Never-audited policies sort before every measured policy.
    return (-1 if last is None else int(last), int(record.get("iteration", -1)))


def select_historical_audits(records: dict[int, dict], *, source_iteration: int,
                             active_ids=(), cycle_ids=(), cursor: int,
                             config, include_all=False) -> tuple[list[HistoricalAuditTarget], int]:
    """Select a bounded category-reserved set including retired policies.

    Empty categories are backfilled from the deterministic stale order. This
    avoids both global-top-K starvation and a growing collection of special
    cases. Never-audited/oldest records have six reserved seats by default,
    after up to two risk sentinels; active policies remain excluded.
    """
    excluded = set(map(int, active_ids))

    def _auditable(identity: int, record: dict) -> bool:
        if int(identity) in excluded:
            return False
        if include_all:
            return (not record.get("invalid", False)
                    and record.get("safety_status", "valid") == "valid"
                    and record.get("duplicate_of_archive_id") is None
                    and bool(record.get("file")))
        if not record.get("payoff_eligible", True) or record.get("invalid", False):
            return False
        if int(record.get("iteration", source_iteration)) <= int(source_iteration):
            return True
        # 2026-09-04: the iteration gate above is a migration-era artifact --
        # it selects the pre-20K archive, which for a from-scratch run
        # (source_iteration=0) is empty, leaving this whole audit/reactivation
        # path permanently inert. A policy that actually held a seat and then
        # lost it is exactly what this mechanism is for: it asks whether Main
        # has regressed against something it used to beat. Without this, a
        # retired member can never come back, and eviction becomes permanent
        # deletion.
        return str(record.get("admission_status", "")) == "archive_only"

    eligible = {
        int(identity): record for identity, record in records.items()
        if _auditable(int(identity), record)
    }
    if not eligible:
        return [], int(cursor)

    selected: list[HistoricalAuditTarget] = []
    used: set[int] = set()

    def take(order, count: int, reason: str) -> None:
        if int(count) <= 0:
            return
        for identity in order:
            identity = int(identity)
            if identity not in eligible or identity in used:
                continue
            selected.append(HistoricalAuditTarget(identity, reason))
            used.add(identity)
            if sum(item.reason == reason for item in selected) >= int(count):
                break

    cycles = sorted(
        (int(identity) for identity in cycle_ids if int(identity) in eligible),
        key=lambda identity: (_audit_age(eligible[identity]), identity))
    take(cycles, config.cold_cycle_quota, "cycle_sentinel")

    regression = sorted(eligible, key=lambda identity: (
        -float(eligible[identity].get("regression", 0.0)),
        float(eligible[identity].get("current_score", 0.5)),
        _audit_age(eligible[identity]), identity))
    take(regression, config.cold_regression_quota, "regression_sentinel")

    stale = sorted(eligible, key=lambda identity: (
        _audit_age(eligible[identity]), identity))
    take(stale, config.cold_stale_quota, "least_recently_audited")

    rotation = sorted(eligible)
    start = int(cursor) % len(rotation)
    scanned = 0
    rotation_added = 0
    for offset in range(len(rotation)):
        if rotation_added >= config.cold_rotation_quota:
            break
        scanned = offset + 1
        identity = rotation[(start + offset) % len(rotation)]
        if identity in used:
            continue
        selected.append(HistoricalAuditTarget(identity, "round_robin"))
        used.add(identity)
        rotation_added += 1
    next_cursor = ((start + max(1, scanned)) % len(rotation)
                   if config.cold_rotation_quota else int(cursor))

    # Category overlap is common (an old hard policy may also be stale). Fill
    # the unused reservation without raising the configured policy budget.
    remaining = config.cold_audit_total - len(selected)
    if remaining > 0:
        take(stale, remaining, "quota_backfill")
    if len(selected) > config.cold_audit_total:
        raise RuntimeError("historical audit selector exceeded its fixed budget")
    return selected, next_cursor


__all__ = [
    "HISTORICAL_AUDIT_PROTOCOL", "HistoricalAuditTarget",
    "select_historical_audits", "strategic_active_ids",
]
