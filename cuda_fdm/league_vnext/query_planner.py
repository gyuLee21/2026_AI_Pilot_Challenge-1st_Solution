"""Value-of-information query planner for a bounded sparse payoff graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math

from .payoff_graph import SparsePayoffGraph


@dataclass(frozen=True)
class PayoffQuery:
    left: int
    right: int
    priority: float
    reason: str
    minimum_blocks: int
    maximum_blocks: int
    phase: str
    scenario_bank_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class PayoffQueryPlanner:
    def __init__(self, config):
        self.config = config
        self.config.validate()

    @staticmethod
    def _canonical(left: int, right: int) -> tuple[int, int]:
        left, right = int(left), int(right)
        if left == right:
            raise ValueError("self payoff query is invalid")
        return (left, right) if left < right else (right, left)

    def plan(self, graph: SparsePayoffGraph, records: dict[int, dict], *,
             iteration: int, active_ids=(), strategic_ids=(), champion_id=None,
             sentinel_ids=(), candidate_ids=(), champion_candidate_id=None,
             admission_target_id=None) -> list[PayoffQuery]:
        active = sorted(set(map(int, active_ids)))
        strategic = sorted(set(map(int, strategic_ids)))
        sentinels = sorted(set(map(int, sentinel_ids)))
        candidates = sorted(set(map(int, candidate_ids)))
        incumbent_by_candidate = {
            candidate: int(records[candidate]["successor_incumbent_id"])
            for candidate in candidates
            if candidate in records
            and records[candidate].get("successor_incumbent_id") is not None
        }
        pool: dict[tuple[int, int], set[str]] = {}

        def add(left, right, reason):
            if int(left) == int(right):
                return
            pair = self._canonical(left, right)
            pool.setdefault(pair, set()).add(str(reason))

        # Active edges are operationally important, but the query budget still
        # prevents a full all-pairs burst from blocking the learner.
        for left, right in itertools.combinations(active, 2):
            add(left, right, "active")
        if champion_id is not None:
            for identity in set(active + strategic + sentinels + candidates):
                add(champion_id, identity, "champion_anchor")
        if champion_candidate_id is not None and champion_id is not None:
            add(champion_candidate_id, champion_id, "champion_promotion")
            for identity in sentinels:
                add(champion_candidate_id, identity, "champion_promotion")
        for candidate in candidates:
            for identity in set(active + sentinels):
                # Only the frozen current-Main target decides admission. Other
                # anchors are bounded solver probes and must not accidentally
                # consume the 128-block confirmatory budget.
                add(candidate, identity, "admission" if admission_target_id is None
                    else "candidate_probe")
            if admission_target_id is not None:
                # The current learner is the admission and altitude-red-team
                # target. Give this edge an explicit reason so a bounded query
                # budget cannot spend every slot on incidental archive anchors.
                add(candidate, admission_target_id, "admission_target")
                incumbent = incumbent_by_candidate.get(candidate)
                if incumbent is not None:
                    add(incumbent, admission_target_id, "successor_incumbent_baseline")
        for sentinel in sentinels:
            for identity in active:
                add(sentinel, identity, "regression")
        # Neighbour hints are computed by the strategic index from payoff and
        # behaviour descriptors.  They bound graph growth without trusting a
        # learned payoff predictor as evidence.
        for identity in strategic:
            record = records.get(identity, {})
            for neighbor in record.get("payoff_neighbors", [])[:self.config.fingerprint_neighbors]:
                add(identity, neighbor, "payoff_neighbor")
            for neighbor in record.get("behavior_neighbors", [])[:self.config.behavior_neighbors]:
                add(identity, neighbor, "behavior_neighbor")

        ordered_candidates = sorted(
            candidates,
            key=lambda identity: (
                int(records.get(identity, {}).get("iteration", -1)), identity),
            reverse=True)
        candidate_recency_by_id = {
            identity: max(
                0.0, 4.0 * (len(ordered_candidates) - rank)
                / max(1, len(ordered_candidates)))
            for rank, identity in enumerate(ordered_candidates)}
        ranked = []
        for (left, right), reasons in pool.items():
            admission_candidate = next((identity for identity in (left, right)
                                        if identity in candidates), None)
            incumbent_candidates = [
                candidate for candidate, incumbent in incumbent_by_candidate.items()
                if admission_target_id is not None
                and self._canonical(incumbent, admission_target_id) == (left, right)
            ]
            incumbent_candidate = max(
                incumbent_candidates,
                key=lambda identity: (
                    int(records.get(identity, {}).get("iteration", -1)), identity),
                default=None)
            decision_candidate = (admission_candidate if admission_candidate is not None
                                  else incumbent_candidate)
            screening_passed = False
            if decision_candidate is not None:
                record = records.get(decision_candidate, {})
                screening_passed = str(record.get(
                    "screening_status",
                    record.get("metrics", {}).get("screening_status", "pending"))) == "passed"
            is_admission = bool(reasons.intersection(
                {"admission", "admission_target", "successor_incumbent_baseline"}))
            if is_admission and not screening_passed:
                phase = "screening"
                minimum = self.config.screening_paired_blocks
                scenario_bank = self.config.screening_scenario_bank
            elif is_admission or "champion_promotion" in reasons:
                phase = "confirmatory"
                minimum = self.config.confirmatory_paired_blocks
                scenario_bank = self.config.confirmatory_scenario_bank
            else:
                phase = "solver"
                minimum = self.config.solver_paired_blocks
                scenario_bank = self.config.solver_scenario_bank
            if not graph.query_eligible(
                    left, right, phase=phase, required_blocks=minimum,
                    maximum_blocks=self.config.maximum_paired_blocks,
                    iteration=iteration,
                    stale_after_iterations=self.config.stale_after_iterations):
                continue
            estimate = graph.estimate(left, right)
            record_l = records.get(left, {})
            record_r = records.get(right, {})
            uncertainty = 1.0 if not estimate.known else estimate.width
            age = (self.config.stale_after_iterations if estimate.last_iteration is None
                   else max(0, int(iteration) - estimate.last_iteration))
            staleness = min(1.0, age / self.config.stale_after_iterations)
            meta = min(1.0, float(record_l.get("nash_mass", 0.0))
                       + float(record_r.get("nash_mass", 0.0)))
            decision = 1.0 if reasons.intersection(
                {"champion_promotion", "admission", "admission_target",
                 "successor_incumbent_baseline", "regression"}) else 0.25
            mandatory = 1.0 if reasons.intersection(
                {"champion_promotion", "admission", "admission_target",
                 "successor_incumbent_baseline"}) else 0.0
            target_bonus = (25.0 if "admission_target" in reasons else
                            24.75 if "successor_incumbent_baseline" in reasons else 0.0)
            candidate_recency = candidate_recency_by_id.get(decision_candidate, 0.0)
            priority = (100.0 * mandatory + 8.0 * decision + 5.0 * uncertainty
                        + 3.0 * meta + 2.0 * staleness + target_bonus
                        + candidate_recency)
            if not math.isfinite(priority):
                raise FloatingPointError("non-finite payoff query priority")
            ranked.append(PayoffQuery(
                left, right, priority, "+".join(sorted(reasons)),
                minimum, self.config.maximum_paired_blocks,
                phase, scenario_bank,
            ))
        ranked.sort(key=lambda item: (-item.priority, item.left, item.right))
        return ranked[:self.config.query_budget_per_milestone]


__all__ = ["PayoffQuery", "PayoffQueryPlanner"]
