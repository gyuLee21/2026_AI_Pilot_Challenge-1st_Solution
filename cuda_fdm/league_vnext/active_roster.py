"""Bounded active-roster proposals and realized-exposure accounting."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from cuda_fdm.league import role_mixture


LAYOUT = {"latest": 1, "recent": 4, "core": 16, "challenger": 3}


@dataclass(frozen=True)
class RosterProposal:
    latest: tuple[int, ...]
    recent: tuple[int, ...]
    core: tuple[int, ...]
    challenger: tuple[int, ...]

    @property
    def all_ids(self) -> tuple[int, ...]:
        return self.latest + self.recent + self.core + self.challenger

    def to_dict(self) -> dict:
        return {role: list(getattr(self, role)) for role in LAYOUT}


def ordered_probationary_ids(records: dict[int, dict]) -> list[int]:
    """Probationary challengers in fair-queue order.

    Shared by the shadow proposal and the staged mutation so both see the same
    queue. No candidate type may starve once it has waited: a policy already
    waiting for solver completion goes first, then a reactivated historical
    counter, then brand-new successors newest-first.
    """
    def order_key(identity: int):
        record = records[identity]
        metrics = record.get("metrics") or {}
        historical = metrics.get("historical_counter_reactivated_at_iteration")
        pending = metrics.get("solver_pending_since_iteration")
        created = int(record.get("iteration", -1))
        activations = int(metrics.get("challenger_activation_count", 0))
        last_active = int(metrics.get("last_challenger_activation_iteration", -1))
        if pending is not None:
            return (0, activations, int(pending), last_active, identity)
        if historical is not None:
            return (1, activations, int(historical), last_active, identity)
        # Once every candidate has entered at least once, the least-exposed
        # and longest-idle candidate goes first.  Newest-first remains only a
        # final tie-break, so a stream of fresh admissions cannot starve an
        # older probationary policy forever.
        return (2, activations, last_active, -created, -identity)

    return sorted(
        (int(identity) for identity, record in records.items()
         if record.get("admitted") is True
         and str(record.get("admission_status", "")) == "probationary"),
        key=order_key)


def rank_roster_candidates(records: dict[int, dict], *, solver_ids,
                           recent_ids=(), current_id: int | None = None,
                           nash_override: dict[int, float] | None = None):
    """Ordered (strategic, challenger) candidate lists for ``propose()``.

    Ranking only -- ``ActiveRosterSelector.propose()`` performs the bounded
    allocation (role dedup, cap enforcement, challenger-before-core). Audit
    2026-09-02: the shadow proposal and the staged ``refresh_roster()`` used to
    rank and allocate independently, so what P0 observed in
    ``active_roster`` did not predict what P1 would actually apply. Both now
    rank here and allocate there.

    A policy appearing in both lists is resolved by ``propose()`` in favour of
    the challenger seat. That precedence is meant for *unproven* policies --
    ``propose()``'s own note is about a newly admitted challenger -- so the
    challenger list leads with the probationary queue, and the seats it cannot
    fill from that queue are filled from the *bottom* of the strategic ranking
    (the policies core wants least). Ranking the fill by recency instead would
    hand a challenger seat to a policy that just completed probation on the
    strength of a high Nash mass, which is precisely the policy maturation
    exists to promote into core.
    """
    excluded = {int(value) for value in recent_ids}
    if current_id is not None:
        excluded.add(int(current_id))
    eligible = [identity for identity in dict.fromkeys(int(v) for v in solver_ids)
                if identity not in excluded
                and records.get(identity, {}).get("admitted", True)]
    eligible_set = set(eligible)
    probationary = [identity for identity in ordered_probationary_ids(records)
                    if identity in eligible_set]

    def nash(identity: int) -> float:
        if nash_override is not None:
            return float(nash_override.get(identity, 0.0))
        return float(records.get(identity, {}).get("nash_mass", 0.0))

    def created(identity: int) -> int:
        return int(records.get(identity, {}).get("iteration", -1))

    strategic = sorted(eligible, key=lambda identity: (nash(identity), identity),
                       reverse=True)
    remaining = [identity for identity in eligible if identity not in set(probationary)]
    challenger = [*probationary,
                  *sorted(remaining,
                          key=lambda identity: (nash(identity), created(identity), identity))]
    return strategic, challenger


class ActiveRosterSelector:
    def __init__(self, *, cap: int = 24):
        if int(cap) != sum(LAYOUT.values()):
            raise ValueError("vNext active roster must preserve the validated 24-policy layout")
        self.cap = int(cap)

    def propose(self, *, latest_id: int, recent_ids=(), strategic_ids=(),
                challenger_ids=()) -> RosterProposal:
        used = set()

        def take(values, count):
            result = []
            for value in values:
                identity = int(value)
                if identity in used:
                    continue
                used.add(identity)
                result.append(identity)
                if len(result) >= count:
                    break
            return tuple(result)

        latest = take([latest_id], LAYOUT["latest"])
        if len(latest) != 1:
            raise ValueError("active roster requires exactly one latest clone")
        recent = take(recent_ids, LAYOUT["recent"])
        # Reserve newly admitted challengers before filling the strategic core;
        # otherwise a high-Nash challenger can consume both roles.
        challenger = take(challenger_ids, LAYOUT["challenger"])
        core = take(strategic_ids, LAYOUT["core"])
        proposal = RosterProposal(latest, recent, core, challenger)
        if len(proposal.all_ids) != len(set(proposal.all_ids)):
            raise RuntimeError("active roster contains duplicate policy identities")
        if len(proposal.all_ids) > self.cap:
            raise RuntimeError("active roster exceeds GPU policy cap")
        return proposal

    @staticmethod
    def fallback_distribution(proposal: RosterProposal, records: dict[int, dict]) -> dict[int, float]:
        entries = []
        for role in LAYOUT:
            for identity in getattr(proposal, role):
                record = records.get(identity, {})
                entries.append({
                    "id": identity, "role": role, "retired": False,
                    "ema": float(record.get("current_score", 0.5)),
                    "past_best": float(record.get("past_best", 0.5)),
                    "coverage": bool(record.get("coverage", False)),
                })
        weights = role_mixture(entries)
        result = {int(entry["id"]): float(weight)
                  for entry, weight in zip(entries, weights)}
        if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
            raise FloatingPointError("active roster probabilities do not sum to one")
        return result


class ExposureTracker:
    """Compares planned probability with completed-game exposure."""

    def __init__(self):
        self.completed_games: dict[int, int] = {}
        self.window_completed_games: dict[int, int] = {}
        self.completed_transitions: dict[int, int] = {}
        self.window_completed_transitions: dict[int, int] = {}
        self.last_exposure_iteration: dict[int, int] = {}

    def update(self, completed_by_policy: dict[int, int], *, iteration: int,
               transitions_by_policy: dict[int, int] | None = None) -> None:
        for identity, games in completed_by_policy.items():
            identity, games = int(identity), int(games)
            if games < 0:
                raise ValueError("completed games cannot be negative")
            self.completed_games[identity] = self.completed_games.get(identity, 0) + games
            self.window_completed_games[identity] = (
                self.window_completed_games.get(identity, 0) + games)
            if games:
                self.last_exposure_iteration[identity] = int(iteration)
        for identity, transitions in (transitions_by_policy or {}).items():
            identity, transitions = int(identity), int(transitions)
            if transitions < 0:
                raise ValueError("completed transitions cannot be negative")
            self.completed_transitions[identity] = (
                self.completed_transitions.get(identity, 0) + transitions)
            self.window_completed_transitions[identity] = (
                self.window_completed_transitions.get(identity, 0) + transitions)

    def report(self, planned: dict[int, float], *, iteration: int,
               consume_window: bool = False) -> dict:
        ids = sorted(map(int, planned))
        probabilities = np.asarray([float(planned[x]) for x in ids], dtype=np.float64)
        if (np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0)
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("invalid planned matchmaking distribution")
        games = np.asarray([self.window_completed_games.get(x, 0) for x in ids], dtype=np.float64)
        realized = games / games.sum() if games.sum() else np.zeros_like(games)
        ess = 1.0 / float(np.square(realized).sum()) if np.any(realized) else 0.0
        report = {
            "planned": {str(x): float(p) for x, p in zip(ids, probabilities)},
            "realized": {str(x): float(p) for x, p in zip(ids, realized)},
            "opponent_ess": ess,
            "maximum_absolute_gap": float(np.max(np.abs(realized - probabilities))),
            "last_exposure_age": {
                str(x): (None if x not in self.last_exposure_iteration
                         else int(iteration) - self.last_exposure_iteration[x])
                for x in ids},
            "window_completed_games": {str(x): int(self.window_completed_games.get(x, 0))
                                       for x in ids},
            "lifetime_completed_games": {str(x): int(self.completed_games.get(x, 0))
                                         for x in ids},
            "window_completed_transitions": {
                str(x): int(self.window_completed_transitions.get(x, 0)) for x in ids},
        }
        if consume_window:
            self.window_completed_games.clear()
            self.window_completed_transitions.clear()
        return report

    def state_dict(self) -> dict:
        return {
            "completed_games": self.completed_games,
            "window_completed_games": self.window_completed_games,
            "completed_transitions": self.completed_transitions,
            "window_completed_transitions": self.window_completed_transitions,
            "last_exposure_iteration": self.last_exposure_iteration,
        }


__all__ = ["LAYOUT", "RosterProposal", "ActiveRosterSelector", "ExposureTracker",
           "ordered_probationary_ids", "rank_roster_candidates"]
