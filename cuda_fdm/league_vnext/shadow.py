"""Read-only orchestration of vNext decisions before staged activation."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .active_roster import (
    LAYOUT, ActiveRosterSelector, ExposureTracker, rank_roster_candidates)
from .archive_index import ArchiveMaterializedIndex
from .budget import WallclockBudgetTracker
from .champion import ChampionManager
from .contracts import VNextConfig, VNextMode
from .decision_log import DecisionLog
from .health import PPOHealthMonitor
from .lineages import LineageManager
from .payoff_graph import SparsePayoffGraph
from .query_planner import PayoffQueryPlanner
from .strategic_index import StrategicIndexSelector


SHADOW_STATE_PROTOCOL = "active_league_100k_shadow_state_v2"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class VNextShadowController:
    """Produces auditable proposals and zero training mutations in shadow mode."""

    def __init__(self, root, config: VNextConfig | None = None, *, state: dict | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or VNextConfig()
        self.config.validate()
        self.log = DecisionLog(self.root / "decisions.jsonl")
        self.graph = SparsePayoffGraph(confidence=self.config.payoff_graph.confidence)
        self.champion = ChampionManager(self.config.champion)
        self.lineages = LineageManager()
        self.index = StrategicIndexSelector(self.config.strategic_index)
        self.archive_index = ArchiveMaterializedIndex(self.root / "archive_index")
        self.budget = WallclockBudgetTracker(target_iteration=self.config.target_iteration)
        self.planner = PayoffQueryPlanner(self.config.payoff_graph)
        self.health = PPOHealthMonitor(self.config.health)
        self.roster = ActiveRosterSelector(cap=self.config.active_cap)
        self.exposure = ExposureTracker()
        self.last_exposure_report = None
        self.last_iteration = self.config.source_iteration
        self.last_index = None
        self.last_solver_ids: list[int] = []
        self.historical_audit_cursor = 0
        self.freeze_reasons: list[str] = []
        self.kl_healthy_streak = 0
        self.consecutive_milestones_without_admission = 0
        if state is not None:
            self.load_state_dict(state)

    def observe_admission_cadence(self, iteration: int, *, admitted_count: int) -> bool:
        """Track milestones with zero new admissions; log an alert past the
        configured streak. Returns True exactly when this call raised one."""
        if int(admitted_count) < 0:
            raise ValueError("admitted_count cannot be negative")
        if admitted_count > 0:
            self.consecutive_milestones_without_admission = 0
            return False
        self.consecutive_milestones_without_admission += 1
        threshold = self.config.health.maximum_milestones_without_admission
        if self.consecutive_milestones_without_admission >= threshold:
            self.log.append(
                "admission_cadence_alert",
                {"consecutive_milestones_without_admission":
                     self.consecutive_milestones_without_admission,
                 "threshold": threshold},
                iteration=int(iteration))
            return True
        return False

    @property
    def may_mutate_training(self) -> bool:
        return self.config.mode == VNextMode.STAGED and not self.freeze_reasons

    def observe_iteration(self, iteration: int, metrics: dict) -> dict:
        iteration = int(iteration)
        if iteration < self.config.source_iteration:
            raise ValueError("vNext observations cannot precede the final-20000 migration")
        if iteration < self.last_iteration:
            raise ValueError("vNext iteration moved backwards")
        report = self.health.evaluate(iteration, metrics)
        if self.config.decision_freeze_on_anomaly:
            if report.alerts == ("policy_kl_high",):
                if "transient_policy_kl_high" not in self.freeze_reasons:
                    self.freeze_reasons.append("transient_policy_kl_high")
                self.kl_healthy_streak = 0
            elif not report.healthy:
                if "health_anomaly" not in self.freeze_reasons:
                    self.freeze_reasons.append("health_anomaly")
                self.kl_healthy_streak = 0
            elif "transient_policy_kl_high" in self.freeze_reasons:
                self.kl_healthy_streak = (self.kl_healthy_streak + 1
                    if iteration == self.last_iteration + 1 else 1)
                if self.kl_healthy_streak >= 10:
                    self.freeze_reasons.remove("transient_policy_kl_high")
                    self.kl_healthy_streak = 0
                    self.log.append("policy_kl_freeze_recovered", {
                        "consecutive_healthy_iterations": 10,
                        "remaining_freeze_reasons": list(self.freeze_reasons)}, iteration=iteration)
        if iteration % self.config.health.log_period == 0 or not report.healthy:
            self.log.append("ppo_health", report.to_dict(), iteration=iteration)
        self.last_iteration = iteration
        if "elapsed_sec" in metrics and "environment_transitions" in metrics:
            self.budget.observe_main(
                seconds=float(metrics["elapsed_sec"]),
                transitions=int(metrics["environment_transitions"]),
                completed_games=int(metrics.get("completed_games", 0)))
        if float(metrics.get("evaluation_elapsed_sec", 0.0)) > 0.0:
            self.budget.add("evaluation", float(metrics["evaluation_elapsed_sec"]))
        if float(metrics.get("side_elapsed_sec", 0.0)) > 0.0:
            self.budget.add("side", float(metrics["side_elapsed_sec"]))
        exposure = metrics.get("opponent_exposure") or []
        if exposure:
            completed = {int(item["resident_id"]): int(item["completed_games"])
                         for item in exposure}
            transitions = {int(item["resident_id"]): int(item["environment_transitions"])
                           for item in exposure}
            planned = {int(item["resident_id"]): float(item["planned_probability"])
                       for item in exposure if float(item["planned_probability"]) > 0.0}
            self.exposure.update(completed, iteration=iteration,
                                 transitions_by_policy=transitions)
            if planned:
                total = sum(planned.values())
                planned = {identity: value / total for identity, value in planned.items()}
                self.last_exposure_report = self.exposure.report(
                    planned, iteration=iteration,
                    consume_window=(iteration % 500 == 0))
        # The model checkpoint receives this in-memory state every recovery
        # boundary.  Do not advance the external bootstrap state every
        # iteration: after a crash it could otherwise be newer than the model
        # checkpoint. Milestones persist an external snapshot because they are
        # complete transactional league boundaries.
        return report.to_dict()

    def observe_milestone(self, iteration: int, archive_state: dict, *,
                          current_id: int | None, active_ids=(), sentinel_ids=(),
                          baseline_ids=(), cycle_ids=(), candidate_ids=(),
                          commit: bool = True) -> dict:
        iteration = int(iteration)
        records = self.archive_index.rebuild(
            archive_state, self.graph, iteration=iteration, persist=commit)
        champion_id = self.champion.champion_id
        if champion_id is None:
            champion_id = current_id
        result = self.index.select(
            records, champion_id=champion_id, sentinel_ids=sentinel_ids,
            baseline_ids=baseline_ids, cycle_ids=cycle_ids,
            iteration=iteration)
        planner_ids = (set(result.selected_ids) | set(map(int, candidate_ids))
                       | set(map(int, sentinel_ids)))
        if champion_id is not None:
            planner_ids.add(int(champion_id))
        enriched = {identity: dict(records[identity]) for identity in planner_ids
                    if identity in records}
        for identity in result.selected_ids:
            enriched[identity]["payoff_neighbors"] = list(result.payoff_neighbors[identity])
            enriched[identity]["behavior_neighbors"] = list(result.behavior_neighbors[identity])
        queries = self.planner.plan(
            self.graph, enriched, iteration=iteration, active_ids=active_ids,
            strategic_ids=result.selected_ids, champion_id=champion_id,
            sentinel_ids=sentinel_ids, candidate_ids=candidate_ids,
            admission_target_id=current_id)
        recent_ids = sorted((identity for identity, record in records.items()
                             if str(record.get("kind", "")).startswith("recent")),
                            key=lambda identity: int(records[identity].get("iteration", -1)),
                            reverse=True)
        if current_id is None:
            raise RuntimeError("vNext active roster requires the current learner identity")
        # Audit 2026-09-02: this shadow proposal and the staged
        # VNextMilestoneAdapter.refresh_roster() used to rank candidates by two
        # independently written rules, so observing "active_roster" during P0
        # did not predict what P1 would actually apply. Both now rank through
        # rank_roster_candidates() and allocate through propose().
        strategic_order, challenger_order = rank_roster_candidates(
            records, solver_ids=result.selected_ids,
            recent_ids=recent_ids[:LAYOUT["recent"]], current_id=current_id)
        roster = self.roster.propose(
            latest_id=current_id, recent_ids=recent_ids,
            strategic_ids=strategic_order, challenger_ids=challenger_order)
        proposal = {
            "mode": self.config.mode.value,
            "training_mutation_applied": False,
            "strategic_index": result.to_dict(),
            "payoff_queries": [query.to_dict() for query in queries],
            "active_roster": roster.to_dict(),
            "champion_id": champion_id,
            "freeze_reasons": list(self.freeze_reasons),
            "budget": self.budget.report(iteration=iteration),
        }
        # Even staged mode only emits proposals here. The future live adapter
        # must commit each mutation transactionally after its own release gate.
        self.last_index = proposal["strategic_index"]
        self.last_iteration = max(self.last_iteration, iteration)
        if commit:
            self.log.append("milestone_shadow_proposal", proposal, iteration=iteration)
            self.persist()
        return proposal

    def state_dict(self) -> dict:
        return {
            "protocol": SHADOW_STATE_PROTOCOL,
            "config": self.config.to_dict(),
            "last_iteration": self.last_iteration,
            "last_index": self.last_index,
            "last_solver_ids": list(self.last_solver_ids),
            "historical_audit_cursor": int(self.historical_audit_cursor),
            "freeze_reasons": list(self.freeze_reasons),
            "kl_healthy_streak": int(self.kl_healthy_streak),
            "consecutive_milestones_without_admission": int(
                self.consecutive_milestones_without_admission),
            "payoff_graph": self.graph.state_dict(),
            "champion": self.champion.state_dict(),
            "lineages": self.lineages.state_dict(),
            "strategic_index_state": self.index.state_dict(),
            "archive_index": self.archive_index.state_dict(),
            "wallclock_budget": self.budget.state_dict(),
            "exposure_tracker": self.exposure.state_dict(),
            "last_exposure_report": self.last_exposure_report,
            "decision_log_sha256": self.log.last_hash,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("protocol") != SHADOW_STATE_PROTOCOL:
            raise ValueError("vNext shadow state protocol mismatch")
        saved_config = VNextConfig.from_dict(state["config"])
        if saved_config.to_dict() != self.config.to_dict():
            raise ValueError("vNext configuration changed across resume")
        checkpoint_log_hash = state.get("decision_log_sha256")
        if checkpoint_log_hash != self.log.last_hash:
            if not self.log.contains_hash(checkpoint_log_hash):
                raise RuntimeError("vNext decision log does not match checkpoint state")
            # 2026-09-03 (found via a real resume): --save-every writes the
            # model checkpoint on its own 100-iteration cadence, independent
            # of on_milestone()'s persist() -- a milestone's decisions can
            # commit to decisions.jsonl and archive_manifest.json before the
            # next model checkpoint captures that iteration. Killing the
            # process in that window (exactly what happened here: the log's
            # last entry was milestone 4000's commit, but checkpoint.pt was
            # still at iteration 3900) left the on-disk log ahead of the
            # checkpoint. This branch already restores every field below from
            # the checkpoint's own embedded snapshot regardless of mode
            # (archive.load_state_dict() does the same from checkpoint.pt's
            # embedded league_archive, ignoring archive_manifest.json on
            # disk) -- the mode restriction here was narrower than the
            # comment already claimed ("a staged/live mutation tail is never
            # auto-adopted"), which was already the intended behavior for
            # staged, just not implemented for it. Keep the orphaned tail as
            # evidence, mark the replay, and resume the checkpoint-bound
            # control state -- never adopt or rewind it, in any mode.
            self.log.append(
                "vnext_resume_from_checkpoint",
                {"checkpoint_log_sha256": checkpoint_log_hash,
                 "discarded_tail_sha256": self.log.last_hash,
                 "mode": self.config.mode.value},
                iteration=int(state["last_iteration"]))
        self.last_iteration = int(state["last_iteration"])
        self.last_index = state.get("last_index")
        self.last_solver_ids = [int(value) for value in state.get("last_solver_ids", [])]
        self.historical_audit_cursor = int(state.get("historical_audit_cursor", 0))
        self.freeze_reasons = list(state.get("freeze_reasons", []))
        self.kl_healthy_streak = int(state.get("kl_healthy_streak", 0))
        self.consecutive_milestones_without_admission = int(
            state.get("consecutive_milestones_without_admission", 0))
        self.graph = SparsePayoffGraph(
            confidence=self.config.payoff_graph.confidence,
            state=state["payoff_graph"],
        )
        champion_state = state.get("champion", {})
        self.champion = ChampionManager(
            self.config.champion, champion_id=champion_state.get("champion_id"),
            history=champion_state.get("history", []),
            frontier=champion_state.get("frontier", []),
            expected_suite_version=champion_state.get(
                "expected_suite_version", "champion_suite_v1"),
            expected_protocol_version=champion_state.get(
                "expected_protocol_version", "active_league_vnext_100k_v2"))
        self.lineages = LineageManager(state.get("lineages", {}))
        self.index = StrategicIndexSelector(
            self.config.strategic_index, state=state.get("strategic_index_state", {}))
        self.archive_index = ArchiveMaterializedIndex(
            self.root / "archive_index", state=state.get("archive_index", {
                "protocol": "active_league_archive_materialized_index_v1",
                "epoch": 0, "records": [], "anchor_ids": []}))
        self.budget = WallclockBudgetTracker(
            target_iteration=self.config.target_iteration,
            state=state.get("wallclock_budget", {
                "protocol": "active_league_wallclock_budget_v1",
                "target_iteration": self.config.target_iteration, "window": 512,
                "samples": {}, "main_transitions": 0, "completed_games": 0}))
        self.exposure = ExposureTracker()
        exposure_state = state.get("exposure_tracker", {})
        for name in ("completed_games", "window_completed_games",
                     "completed_transitions", "window_completed_transitions",
                     "last_exposure_iteration"):
            setattr(self.exposure, name, {
                int(key): int(value) for key, value in exposure_state.get(name, {}).items()})
        self.last_exposure_report = state.get("last_exposure_report")

    def persist(self) -> None:
        _atomic_json(self.root / "shadow_state.json", self.state_dict())


__all__ = ["SHADOW_STATE_PROTOCOL", "VNextShadowController"]
