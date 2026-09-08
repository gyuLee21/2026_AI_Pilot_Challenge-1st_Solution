"""Frozen Champion pointer, paired promotion gate and bounded Pareto frontier."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class PromotionEvidence:
    candidate_id: int
    evaluation_suite_version: str
    protocol_version: str
    primary_difference_lcb: float
    safety_difference_lcb: float
    worst_cluster_difference_lcb: float
    heldout_difference_lcb: float
    redteam_difference_lcb: float
    paired_blocks: int
    primary_metric: float
    worst_cluster_metric: float
    heldout_metric: float
    safety_metric: float
    fresh_confirmatory: bool = True
    critical_regressions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    frontier_only: bool
    candidate_id: int
    previous_champion_id: int | None
    reasons: tuple[str, ...]
    evidence: PromotionEvidence

    def to_dict(self) -> dict:
        value = asdict(self)
        value["evidence"] = asdict(self.evidence)
        return value


@dataclass(frozen=True)
class ChampionFrontierRecord:
    archive_id: int
    champion_epoch: int
    evaluation_suite_version: str
    protocol_version: str
    metrics: dict = field(default_factory=dict)


class ChampionManager:
    def __init__(self, config, *, champion_id: int | None = None,
                 history: list[dict] | None = None,
                 frontier: list[dict] | None = None,
                 expected_suite_version: str = "champion_suite_v1",
                 expected_protocol_version: str = "active_league_vnext_100k_v2"):
        self.config = config
        self.config.validate()
        self.champion_id = None if champion_id is None else int(champion_id)
        self.history = list(history or [])
        self.frontier = [ChampionFrontierRecord(**value) for value in (frontier or [])]
        self.expected_suite_version = str(expected_suite_version)
        self.expected_protocol_version = str(expected_protocol_version)
        self.epoch = max((value.champion_epoch for value in self.frontier), default=0)

    def evaluate(self, evidence: PromotionEvidence) -> PromotionDecision:
        numeric = (
            evidence.primary_difference_lcb, evidence.safety_difference_lcb,
            evidence.worst_cluster_difference_lcb, evidence.heldout_difference_lcb,
            evidence.redteam_difference_lcb, evidence.primary_metric,
            evidence.worst_cluster_metric, evidence.heldout_metric,
            evidence.safety_metric,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise FloatingPointError("non-finite champion promotion evidence")
        failures = []
        if evidence.evaluation_suite_version != self.expected_suite_version:
            failures.append("evaluation_suite_version_mismatch")
        if evidence.protocol_version != self.expected_protocol_version:
            failures.append("champion_protocol_version_mismatch")
        if not evidence.fresh_confirmatory:
            failures.append("confirmatory_seeds_are_not_fresh")
        if evidence.primary_difference_lcb < -self.config.primary_noninferiority_margin:
            failures.append("primary_metric_inferior")
        if evidence.safety_difference_lcb < -self.config.safety_noninferiority_margin:
            failures.append("safety_metric_inferior")
        if evidence.worst_cluster_difference_lcb < -self.config.worst_cluster_noninferiority_margin:
            failures.append("worst_cluster_inferior")
        robustness = max(evidence.worst_cluster_difference_lcb,
                         evidence.heldout_difference_lcb,
                         evidence.redteam_difference_lcb)
        if robustness <= self.config.minimum_robustness_improvement:
            failures.append("no_confirmed_robustness_improvement")
        if evidence.paired_blocks < self.config.minimum_paired_blocks:
            failures.append("insufficient_paired_blocks")
        if evidence.critical_regressions:
            failures.append("critical_scenario_regression")
        frontier_only = bool(failures and set(failures).issubset({
            "primary_metric_inferior", "no_confirmed_robustness_improvement"}))
        return PromotionDecision(
            not failures, frontier_only, int(evidence.candidate_id), self.champion_id,
            tuple(failures or ["paired_noninferiority_and_robustness_gate_passed"]), evidence)

    @staticmethod
    def _dominates(left: ChampionFrontierRecord,
                   right: ChampionFrontierRecord) -> bool:
        names = ("primary", "worst_cluster", "heldout", "safety")
        left_values = [float(left.metrics[name]) for name in names]
        right_values = [float(right.metrics[name]) for name in names]
        return (all(a >= b for a, b in zip(left_values, right_values))
                and any(a > b for a, b in zip(left_values, right_values)))

    def _insert_frontier(self, evidence: PromotionEvidence) -> None:
        self.epoch += 1
        candidate = ChampionFrontierRecord(
            archive_id=int(evidence.candidate_id), champion_epoch=self.epoch,
            evaluation_suite_version=evidence.evaluation_suite_version,
            protocol_version=evidence.protocol_version,
            metrics={"primary": float(evidence.primary_metric),
                     "worst_cluster": float(evidence.worst_cluster_metric),
                     "heldout": float(evidence.heldout_metric),
                     "safety": float(evidence.safety_metric)},
        )
        if any(self._dominates(value, candidate) for value in self.frontier):
            return
        self.frontier = [value for value in self.frontier
                         if not self._dominates(candidate, value)
                         and value.archive_id != candidate.archive_id]
        self.frontier.append(candidate)
        self.frontier.sort(key=lambda value: (
            min(value.metrics.values()), value.metrics["primary"],
            -value.archive_id), reverse=True)
        self.frontier = self.frontier[:self.config.frontier_cap]

    def commit(self, decision: PromotionDecision, *, iteration: int) -> None:
        if not decision.accepted:
            raise RuntimeError("rejected champion candidate cannot be committed")
        if decision.previous_champion_id != self.champion_id:
            raise RuntimeError("champion changed after promotion evaluation")
        if self.champion_id == decision.candidate_id:
            raise RuntimeError("candidate is already champion")
        self._insert_frontier(decision.evidence)
        self.history.append({"iteration": int(iteration), **decision.to_dict()})
        self.champion_id = int(decision.candidate_id)

    def consider_for_frontier(self, evidence: PromotionEvidence) -> None:
        self._insert_frontier(evidence)

    def state_dict(self) -> dict:
        return {
            "champion_id": self.champion_id, "history": self.history,
            "frontier": [asdict(value) for value in self.frontier],
            "expected_suite_version": self.expected_suite_version,
            "expected_protocol_version": self.expected_protocol_version,
        }


__all__ = ["PromotionEvidence", "PromotionDecision", "ChampionFrontierRecord",
           "ChampionManager"]
