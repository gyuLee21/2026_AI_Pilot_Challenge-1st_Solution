"""Role-scoped adaptive reward-profile statistics for side learners."""
from __future__ import annotations

import math


PROFILE_STATS_PROTOCOL = "role_scoped_exploiter_profile_stats_v1"
# 2026-09-03 (user decision): this isolated 100K package trains from scratch
# and never schedules LE (across all LE sessions run on the 20K lineage,
# every one crossed its training win-rate target yet zero were ever
# admitted -- LE trains against an easier league mixture than it is
# evaluated against). Unlike 20K/v17, which kept LE's machinery in place
# behind a scheduling exclusion for checkpoint-resume compatibility, this
# package has no existing checkpoint to preserve, so LE is dropped here
# outright rather than soft-disabled.
EXPLOITER_ROLES = ("ME-EIE", "ME-ERE")
EXPLOITER_PROFILES = ("standard", "altitude_hunt", "attack", "defense")


def empty_profile_stats() -> dict[str, dict[str, dict[str, float | int]]]:
    # P2 fix (audit 2026-09-02, B6): "utility" is a clean paired-evaluation
    # signal (target_eval["score"] + 0.25*novelty), only ever produced by the
    # non-staged admission path. The staged (P1 live-adapter) path has no
    # fresh evaluation at exploiter-training time and instead reports the
    # training-time win-rate EMA against a different opponent mixture per
    # role -- a different scale and, for LE, a different meaning entirely.
    # Before this fix both were folded into the same running "utility"
    # average. Keep the two evidence streams in separate fields.
    # 2026-09-04: _select_exploiter_profile() reads "staged_count"/
    # "staged_utility" whenever the staged live adapter is installed (always
    # in this package); "count"/"utility" are then only the fallback for the
    # legacy archive-None path. The pre-2026-09-04 sentence below claiming
    # selection continues to use only "utility" is superseded.
    return {
        role: {profile: {"count": 0, "utility": 0.0,
                         "staged_count": 0, "staged_utility": 0.0}
               for profile in EXPLOITER_PROFILES}
        for role in EXPLOITER_ROLES
    }


def target_scope(role: str) -> str:
    if role not in EXPLOITER_ROLES:
        raise ValueError("unknown exploiter role")
    return "frozen_current_main"


def normalise_profile_stats(value) -> dict[str, dict[str, dict[str, float | int]]]:
    """Load v1 role-scoped state; discard the inseparable legacy flat state.

    The legacy map combined LE mixture trials with current-main trials. There is
    no sound way to split those aggregate counts after the fact, so migration
    starts the scheduler feedback at zero while preserving policy/archive state.
    """
    if not isinstance(value, dict) or not all(role in value for role in EXPLOITER_ROLES):
        return empty_profile_stats()
    result = empty_profile_stats()
    for role in EXPLOITER_ROLES:
        role_state = value.get(role)
        if not isinstance(role_state, dict):
            raise ValueError("role-scoped profile statistics are malformed")
        for profile in EXPLOITER_PROFILES:
            item = role_state.get(profile)
            if not isinstance(item, dict):
                raise ValueError("role-scoped profile entry is missing")
            count = int(item.get("count", -1))
            utility = float(item.get("utility", float("nan")))
            if count < 0 or not math.isfinite(utility):
                raise ValueError("role-scoped profile statistics are invalid")
            # staged_count/staged_utility postdate this fix; a checkpoint
            # written before it simply has no staged evidence yet.
            staged_count = int(item.get("staged_count", 0))
            staged_utility = float(item.get("staged_utility", 0.0))
            if staged_count < 0 or not math.isfinite(staged_utility):
                raise ValueError("role-scoped staged profile statistics are invalid")
            result[role][profile] = {
                "count": count, "utility": utility,
                "staged_count": staged_count, "staged_utility": staged_utility}
    return result


__all__ = [
    "PROFILE_STATS_PROTOCOL", "EXPLOITER_ROLES", "EXPLOITER_PROFILES",
    "empty_profile_stats", "normalise_profile_stats", "target_scope",
]
