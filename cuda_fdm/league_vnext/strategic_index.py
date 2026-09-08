"""Bounded strategic representatives over an immutable cold archive."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


def _vector(record: dict, name: str) -> np.ndarray | None:
    value = record.get(name)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or np.any(~np.isfinite(array)):
        return None
    return array


def _distance(left: dict, right: dict, name: str) -> float:
    a, b = _vector(left, name), _vector(right, name)
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / math.sqrt(a.size))


@dataclass(frozen=True)
class StrategicIndexResult:
    selected_ids: tuple[int, ...]
    reasons: dict[int, tuple[str, ...]]
    redundant_with: dict[int, int]
    payoff_neighbors: dict[int, tuple[int, ...]]
    behavior_neighbors: dict[int, tuple[int, ...]]
    quota_counts: dict[str, int]
    solver_epoch: int
    memberships: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "selected_ids": list(self.selected_ids),
            "reasons": {str(k): list(v) for k, v in self.reasons.items()},
            "redundant_with": {str(k): v for k, v in self.redundant_with.items()},
            "payoff_neighbors": {str(k): list(v) for k, v in self.payoff_neighbors.items()},
            "behavior_neighbors": {str(k): list(v) for k, v in self.behavior_neighbors.items()},
            "quota_counts": self.quota_counts,
            "solver_epoch": self.solver_epoch,
            "memberships": list(self.memberships),
        }


@dataclass(frozen=True)
class SolverMembershipRecord:
    archive_id: int
    solver_epoch: int
    membership_reason: tuple[str, ...]
    priority_components: dict
    entered_at_main_iter: int
    last_used_at_main_iter: int
    protected_until_main_iter: int | None


class StrategicIndexSelector:
    """Selects representatives by protected roles and measured contribution.

    Redundancy removal is deliberately conservative: policy embedding, payoff
    response, and behaviour descriptor must all be present and close. Missing
    evidence therefore cannot silently evict a potentially novel strategy.
    """

    def __init__(self, config, *, state: dict | None = None):
        self.config = config
        self.config.validate()
        self.epoch = 0
        self.memberships: dict[int, SolverMembershipRecord] = {}
        self.audit_cursor = 0
        if state is not None:
            self.load_state_dict(state)

    def _redundant(self, candidate: dict, representative: dict) -> bool:
        return (
            _distance(candidate, representative, "_normalised_payoff")
            <= self.config.payoff_distance_epsilon
            and _distance(candidate, representative, "_normalised_behavior")
            <= self.config.behavior_distance_epsilon
        )

    @staticmethod
    def _normalise_view(records: dict[int, dict], source: str, target: str) -> None:
        groups: dict[int, list[tuple[int, np.ndarray]]] = {}
        for identity, record in records.items():
            vector = _vector(record, source)
            if vector is not None:
                groups.setdefault(int(vector.size), []).append((identity, vector))
        for values in groups.values():
            matrix = np.stack([vector for _, vector in values])
            centre = np.median(matrix, axis=0)
            scale = np.median(np.abs(matrix - centre), axis=0) * 1.4826
            fallback = np.std(matrix, axis=0)
            scale = np.where(scale > 1e-9, scale, np.where(fallback > 1e-9, fallback, 1.0))
            for (identity, _), row in zip(values, (matrix - centre) / scale):
                records[identity][target] = row.tolist()

    @staticmethod
    def _eligible(record: dict) -> bool:
        return bool(record.get("payoff_eligible", True)
                    and record.get("admitted", True)
                    and not record.get("invalid", False))

    def select(self, records: dict[int, dict], *, champion_id: int | None,
               sentinel_ids=(), baseline_ids=(), cycle_ids=(),
               iteration: int = 0) -> StrategicIndexResult:
        normalized = {int(identity): dict(record) for identity, record in records.items()
                      if self._eligible(record)}
        self._normalise_view(normalized, "payoff_fingerprint", "_normalised_payoff")
        self._normalise_view(normalized, "behavior_descriptor", "_normalised_behavior")
        preserve = []
        # Champion is an evaluation anchor. It consumes a solver slot only when
        # it has an independent strategic reason (Nash/regression/unique probe).
        if champion_id is not None and int(champion_id) in normalized:
            champion = normalized[int(champion_id)]
            if (float(champion.get("nash_mass", 0.0)) > 0.0
                    or float(champion.get("regression", 0.0)) > 0.0
                    or champion.get("champion_probe_required", False)):
                preserve.append((int(champion_id), "champion_probe"))
        for role, values in (("sentinel", sentinel_ids), ("baseline", baseline_ids),
                             ("nontransitive_cycle", cycle_ids)):
            for identity in values:
                if int(identity) in normalized:
                    preserve.append((int(identity), role))
        for identity, membership in self.memberships.items():
            if (identity in normalized and membership.protected_until_main_iter is not None
                    and int(iteration) <= membership.protected_until_main_iter):
                preserve.append((identity, "protected_recent_member"))
        preserve = list(dict.fromkeys(preserve))
        if len({identity for identity, _ in preserve}) > self.config.maximum_budget:
            raise RuntimeError("protected strategic policies exceed the hard budget")

        selected: list[int] = []
        reasons: dict[int, list[str]] = {}
        redundant: dict[int, int] = {}
        quota_counts: dict[str, int] = {}

        def admit(identity: int, reason: str, *, protected=False) -> bool:
            identity = int(identity)
            if identity not in normalized:
                return False
            if identity in selected:
                if reason not in reasons[identity]:
                    reasons[identity].append(reason)
                return False
            if not protected:
                for representative in selected:
                    if self._redundant(normalized[identity], normalized[representative]):
                        redundant[identity] = representative
                        return False
            selected.append(identity)
            reasons[identity] = [reason]
            quota_counts[reason] = quota_counts.get(reason, 0) + 1
            return True

        for identity, reason in preserve:
            admit(identity, reason, protected=True)

        budget = max(
            self.config.soft_budget,
            len(selected) + self.config.cold_audit_per_epoch)
        budget = min(budget, self.config.maximum_budget)
        cold_reserve = min(
            self.config.cold_audit_per_epoch, max(0, budget - len(selected)))
        selection_limit = max(len(selected), budget - cold_reserve)

        def ranked(reason: str, order, quota: int):
            added = 0
            for identity in order:
                if len(selected) >= selection_limit or added >= int(quota):
                    break
                if admit(identity, reason):
                    added += 1

        candidates = list(normalized)
        recent = sorted(candidates, key=lambda x: (
            bool(normalized[x].get("champion_lineage", False)),
            int(normalized[x].get("iteration", -1))), reverse=True)
        ranked("recent_champion", recent, self.config.recent_champions)
        ranked("nash_support", sorted(candidates, key=lambda x: (
            float(normalized[x].get("nash_mass", 0.0)),
            int(normalized[x].get("iteration", -1))), reverse=True),
               self.config.nash_support)
        ranked("hard", sorted(candidates, key=lambda x: (
            float(normalized[x].get("current_score", 0.5)),
            -int(normalized[x].get("iteration", -1)))), self.config.hard_opponents)
        ranked("regression", sorted(candidates, key=lambda x: (
            float(normalized[x].get("regression", 0.0)),
            -float(normalized[x].get("current_score", 0.5))), reverse=True),
               self.config.regression_sentinels)
        ranked("redteam", sorted((x for x in candidates
                                    if normalized[x].get("redteam", False)),
                                   key=lambda x: normalized[x].get("iteration", -1), reverse=True),
               self.config.redteam_representatives)
        ranked("scenario", sorted((x for x in candidates
                                     if normalized[x].get("scenario_specialist", False)),
                                    key=lambda x: normalized[x].get("scenario_utility", 0.0),
                                    reverse=True), self.config.scenario_representatives)

        def farthest(name: str, quota: int, reason: str):
            added = 0
            remaining = [x for x in candidates if x not in selected]
            while remaining and len(selected) < selection_limit and added < quota:
                with_vector = [x for x in remaining if _vector(normalized[x], name) is not None]
                if not with_vector:
                    break
                if not selected:
                    pick = max(with_vector, key=lambda x: int(normalized[x].get("iteration", -1)))
                else:
                    pick = max(with_vector, key=lambda x: min(
                        _distance(normalized[x], normalized[y], name) for y in selected))
                remaining.remove(pick)
                if admit(pick, reason):
                    added += 1

        farthest("_normalised_behavior", self.config.behavior_representatives,
                 "behavior_novelty")
        farthest("_normalised_payoff", self.config.payoff_representatives,
                 "payoff_novelty")

        cold_candidates = sorted(identity for identity in candidates if identity not in selected)
        selection_limit = budget
        if cold_candidates:
            count = min(cold_reserve, len(cold_candidates))
            start = self.audit_cursor % len(cold_candidates)
            scanned = 0
            added = 0
            for offset in range(len(cold_candidates)):
                if added >= count:
                    break
                scanned = offset + 1
                identity = cold_candidates[(start + offset) % len(cold_candidates)]
                # Audit sampling is evidence collection, not a strategic
                # admission. Redundancy must not erase its reserved slot.
                if admit(identity, "cold_archive_audit", protected=True):
                    added += 1
            self.audit_cursor = (start + max(1, scanned)) % len(cold_candidates)
            if added != count:
                raise RuntimeError("cold archive reservation was not filled")

        # Contribution-weighted fill keeps the index full without turning age
        # alone into a selection signal.
        fill = sorted(candidates, key=lambda x: (
            float(normalized[x].get("nash_mass", 0.0))
            + float(normalized[x].get("regression", 0.0))
            + float(normalized[x].get("scenario_utility", 0.0))
            + (1.0 - float(normalized[x].get("current_score", 0.5))),
            int(normalized[x].get("iteration", -1))), reverse=True)
        ranked("contribution_fill", fill, budget)

        def nearest(name: str, identity: int, count: int) -> tuple[int, ...]:
            choices = []
            for other in selected:
                if other == identity:
                    continue
                distance = _distance(normalized[identity], normalized[other], name)
                if math.isfinite(distance):
                    choices.append((distance, other))
            choices.sort()
            return tuple(other for _, other in choices[:count])

        payoff_neighbors = {identity: nearest("_normalised_payoff", identity, 4)
                            for identity in selected}
        behavior_neighbors = {identity: nearest("_normalised_behavior", identity, 4)
                              for identity in selected}
        self.epoch += 1
        next_memberships = {}
        for identity in selected:
            previous = self.memberships.get(identity)
            record = normalized[identity]
            kind = str(record.get("kind", ""))
            admission_status = str(record.get("admission_status", ""))
            recent = (kind.startswith("exploiter")
                      or admission_status in {"probationary", "recent_challenger"})
            protected_until = (int(iteration) + self.config.recent_protection_main_iters
                               if recent and previous is None
                               else (None if previous is None
                                     else previous.protected_until_main_iter))
            next_memberships[identity] = SolverMembershipRecord(
                archive_id=identity, solver_epoch=self.epoch,
                membership_reason=tuple(reasons[identity]),
                priority_components={
                    "nash_mass": float(record.get("nash_mass", 0.0)),
                    "regression": float(record.get("regression", 0.0)),
                    "current_score": float(record.get("current_score", 0.5)),
                },
                entered_at_main_iter=(int(iteration) if previous is None
                                      else previous.entered_at_main_iter),
                last_used_at_main_iter=int(iteration),
                protected_until_main_iter=protected_until,
            )
        self.memberships = next_memberships
        return StrategicIndexResult(
            tuple(selected), {k: tuple(v) for k, v in reasons.items()}, redundant,
            payoff_neighbors, behavior_neighbors, quota_counts, self.epoch,
            tuple(asdict(self.memberships[key]) for key in sorted(self.memberships)),
        )

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "audit_cursor": self.audit_cursor,
                "memberships": [asdict(self.memberships[key])
                                for key in sorted(self.memberships)]}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.audit_cursor = int(state.get("audit_cursor", 0))
        self.memberships = {}
        for value in state.get("memberships", []):
            value = dict(value)
            value["membership_reason"] = tuple(value.get("membership_reason", ()))
            membership = SolverMembershipRecord(**value)
            if membership.archive_id in self.memberships:
                raise ValueError("duplicate solver membership")
            self.memberships[membership.archive_id] = membership


__all__ = ["SolverMembershipRecord", "StrategicIndexResult", "StrategicIndexSelector"]
