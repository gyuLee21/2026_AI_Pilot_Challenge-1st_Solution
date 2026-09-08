from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from cuda_fdm.league_vnext.behavior import BehaviorDescriptorAccumulator, FEATURES
from cuda_fdm.league_vnext.active_roster import ActiveRosterSelector, ExposureTracker
from cuda_fdm.league_vnext.champion import ChampionManager, PromotionEvidence
from cuda_fdm.league_vnext.contracts import (
    ChampionConfig, StrategicIndexConfig, VNextConfig, VNextMode, VNextStage)
from cuda_fdm.league_vnext.decision_log import DecisionLog
from cuda_fdm.league_vnext.health import PPOHealthMonitor
from cuda_fdm.league_vnext.lineages import LineageManager, LineageRecord
from cuda_fdm.league_vnext.payoff_graph import SparsePayoffGraph
from cuda_fdm.league_vnext.query_planner import PayoffQueryPlanner
from cuda_fdm.league_vnext.shadow import VNextShadowController
from cuda_fdm.league_vnext.strategic_index import StrategicIndexSelector


def record(identity, **extra):
    base = {
        "id": identity, "admitted": True, "payoff_eligible": True,
        "iteration": identity * 100, "nash_mass": 0.01 * identity,
        "current_score": 0.5, "regression": 0.0,
        "policy_embedding": [identity * 0.1, 0.0],
        "payoff_fingerprint": [identity * 0.1, 0.2],
        "behavior_descriptor": [identity * 0.1, -0.1],
    }
    base.update(extra)
    return base


class ContractsTest(unittest.TestCase):
    def test_default_is_shadow_locked_and_single_gpu(self):
        config = VNextConfig()
        config.validate()
        self.assertEqual(config.mode, VNextMode.SHADOW)
        self.assertEqual(config.stage, VNextStage.PREPARED)
        self.assertEqual(config.active_cap, 24)
        self.assertEqual(config.maximum_gpu_learners, 1)

    def test_live_prepared_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            VNextConfig(mode=VNextMode.STAGED).validate()


class SparsePayoffGraphTest(unittest.TestCase):
    def test_unknown_is_not_an_exact_draw(self):
        estimate = SparsePayoffGraph().estimate(1, 2)
        self.assertFalse(estimate.known)
        self.assertIsNone(estimate.posterior_mean)
        self.assertEqual(estimate.width, 1.0)

    def test_transaction_orientation_and_duplicate_block(self):
        graph = SparsePayoffGraph()
        tx = graph.plan(2, 1, "seed-a", reason="test", iteration=20000)
        graph.start(tx)
        forward = graph.commit(tx, [0.75, 1.0, 0.5], iteration=20000)
        reverse = graph.estimate(2, 1)
        canonical = graph.estimate(1, 2)
        self.assertAlmostEqual(forward.posterior_mean, canonical.posterior_mean)
        self.assertAlmostEqual(reverse.posterior_mean, 1.0 - canonical.posterior_mean)
        with self.assertRaises(RuntimeError):
            graph.plan(1, 2, "seed-a", reason="duplicate", iteration=20001)

    def test_state_roundtrip_and_finite_extreme_interval(self):
        graph = SparsePayoffGraph()
        tx = graph.plan(1, 3, 77, reason="all-win", iteration=20000)
        estimate = graph.commit(tx, [1.0] * 16, iteration=20000)
        self.assertLess(estimate.lcb, 1.0)
        self.assertEqual(estimate.ucb, 1.0)
        restored = SparsePayoffGraph(state=graph.state_dict())
        self.assertEqual(restored.estimate(1, 3).paired_blocks, 16)
        self.assertFalse(restored.needs_more(1, 3, minimum_blocks=16,
                                             maximum_blocks=128))


class StrategicIndexTest(unittest.TestCase):
    def test_protected_and_multiview_selection(self):
        config = StrategicIndexConfig(
            minimum_budget=4, soft_budget=8, maximum_budget=8,
            recent_champions=1, nash_support=2, hard_opponents=2,
            regression_sentinels=1, behavior_representatives=1,
            payoff_representatives=1, redteam_representatives=1,
            scenario_representatives=1)
        records = {i: record(i) for i in range(1, 13)}
        records[2]["current_score"] = 0.1
        records[3]["regression"] = 0.4
        records[4]["redteam"] = True
        records[5]["scenario_specialist"] = True
        records[5]["scenario_utility"] = 0.8
        result = StrategicIndexSelector(config).select(
            records, champion_id=1, sentinel_ids=[3], baseline_ids=[6], cycle_ids=[7])
        self.assertLessEqual(len(result.selected_ids), 8)
        self.assertTrue({1, 3, 6, 7}.issubset(result.selected_ids))
        self.assertIn("champion_probe", result.reasons[1])

    def test_redundancy_requires_payoff_and_behavior_evidence(self):
        config = StrategicIndexConfig(minimum_budget=1, soft_budget=2, maximum_budget=2,
                                      recent_champions=0, nash_support=2,
                                      hard_opponents=0, regression_sentinels=0,
                                      behavior_representatives=0,
                                      payoff_representatives=0,
                                      redteam_representatives=0,
                                      scenario_representatives=0,
                                      cold_audit_per_epoch=0)
        records = {1: record(1), 2: record(2)}
        for field in ("policy_embedding", "payoff_fingerprint", "behavior_descriptor"):
            records[2][field] = list(records[1][field])
        result = StrategicIndexSelector(config).select(records, champion_id=1)
        self.assertEqual(result.redundant_with[2], 1)
        del records[2]["behavior_descriptor"]
        result = StrategicIndexSelector(config).select(records, champion_id=1)
        self.assertIn(2, result.selected_ids)

    def test_cold_audit_reservation_cannot_be_starved_by_hot_quotas(self):
        config = StrategicIndexConfig(
            minimum_budget=4, soft_budget=4, maximum_budget=4,
            recent_champions=4, nash_support=4, hard_opponents=4,
            regression_sentinels=4, behavior_representatives=0,
            payoff_representatives=0, redteam_representatives=0,
            scenario_representatives=0, cold_audit_per_epoch=1)
        records = {identity: record(identity) for identity in range(1, 9)}
        result = StrategicIndexSelector(config).select(records, champion_id=None)
        self.assertEqual(len(result.selected_ids), 4)
        self.assertEqual(result.quota_counts.get("cold_archive_audit"), 1)


class QueryPlannerTest(unittest.TestCase):
    def test_current_main_admission_edge_survives_a_one_query_budget(self):
        config = replace(VNextConfig().payoff_graph, query_budget_per_milestone=1)
        planner = PayoffQueryPlanner(config)
        records = {i: record(i) for i in range(1, 9)}
        records[8].update(admitted=False, iteration=20500)
        result = planner.plan(
            SparsePayoffGraph(), records, iteration=20500,
            active_ids=[1, 2, 3, 4, 5], strategic_ids=[1, 2, 3, 4, 5],
            champion_id=1, candidate_ids=[8], admission_target_id=5)
        self.assertEqual(len(result), 1)
        self.assertEqual({result[0].left, result[0].right}, {5, 8})
        self.assertIn("admission_target", result[0].reason)

    def test_admission_and_champion_edges_outrank_coverage(self):
        config = VNextConfig().payoff_graph
        planner = PayoffQueryPlanner(replace(config, query_budget_per_milestone=3))
        records = {i: record(i) for i in range(1, 6)}
        records[2]["payoff_neighbors"] = [3]
        result = planner.plan(SparsePayoffGraph(), records, iteration=20500,
                              active_ids=[1, 2, 3], strategic_ids=[1, 2, 3],
                              champion_id=1, candidate_ids=[5])
        self.assertEqual(len(result), 3)
        self.assertTrue(any("admission" in item.reason for item in result))
        self.assertGreaterEqual(result[0].priority, result[-1].priority)


class ActiveRosterTest(unittest.TestCase):
    def test_roster_is_unique_bounded_and_exposure_is_audited(self):
        selector = ActiveRosterSelector()
        proposal = selector.propose(
            latest_id=1, recent_ids=[2, 3, 4, 5, 6],
            strategic_ids=range(4, 30), challenger_ids=[30, 31, 32, 33])
        self.assertEqual(len(proposal.all_ids), 24)
        self.assertEqual(len(set(proposal.all_ids)), 24)
        records = {identity: record(identity) for identity in proposal.all_ids}
        planned = selector.fallback_distribution(proposal, records)
        tracker = ExposureTracker()
        tracker.update({identity: 10 for identity in proposal.all_ids}, iteration=21000)
        report = tracker.report(planned, iteration=21010)
        self.assertGreater(report["opponent_ess"], 20.0)
        self.assertLessEqual(max(planned.values()), 0.20)


class ChampionAndLineageTest(unittest.TestCase):
    def test_champion_requires_holdout_and_confidence(self):
        manager = ChampionManager(ChampionConfig(), champion_id=10)
        evidence = PromotionEvidence(
            candidate_id=11, evaluation_suite_version="champion_suite_v1",
            protocol_version="active_league_vnext_100k_v2",
            primary_difference_lcb=0.01, safety_difference_lcb=0.0,
            worst_cluster_difference_lcb=0.02, heldout_difference_lcb=0.01,
            redteam_difference_lcb=0.0, paired_blocks=128,
            primary_metric=0.55, worst_cluster_metric=0.52,
            heldout_metric=0.53, safety_metric=0.99)
        decision = manager.evaluate(evidence)
        self.assertTrue(decision.accepted)
        manager.commit(decision, iteration=25000)
        self.assertEqual(manager.champion_id, 11)
        rejected = manager.evaluate(PromotionEvidence(
            candidate_id=12, evaluation_suite_version="champion_suite_v1",
            protocol_version="active_league_vnext_100k_v2",
            primary_difference_lcb=0.10, safety_difference_lcb=-0.10,
            worst_cluster_difference_lcb=0.02, heldout_difference_lcb=0.01,
            redteam_difference_lcb=0.0, paired_blocks=128,
            primary_metric=0.70, worst_cluster_metric=0.65,
            heldout_metric=0.62, safety_metric=0.80))
        self.assertFalse(rejected.accepted)
        self.assertIn("safety_metric_inferior", rejected.reasons)

    def test_direct_head_to_head_without_robustness_cannot_promote(self):
        manager = ChampionManager(ChampionConfig(), champion_id=10)
        evidence = PromotionEvidence(
            candidate_id=11, evaluation_suite_version="champion_suite_v1",
            protocol_version="active_league_vnext_100k_v2",
            primary_difference_lcb=0.20, safety_difference_lcb=0.0,
            worst_cluster_difference_lcb=0.0, heldout_difference_lcb=0.0,
            redteam_difference_lcb=0.0, paired_blocks=128,
            primary_metric=0.70, worst_cluster_metric=0.50,
            heldout_metric=0.50, safety_metric=1.0)
        decision = manager.evaluate(evidence)
        self.assertFalse(decision.accepted)
        self.assertIn("no_confirmed_robustness_improvement", decision.reasons)

    def test_only_one_persistent_lineage_owns_gpu(self):
        manager = LineageManager()
        for suffix, role in (("a", "ME-EIE"), ("b", "LE"), ("c", "ME-ERE")):
            manager.register(LineageRecord(suffix, role, [1], "m", "o", "n", "r"))
        chosen = manager.choose_next(iteration=21000)
        with self.assertRaises(RuntimeError):
            manager.choose_next(iteration=21000)
        manager.finish_slice(chosen, iteration=21010, useful=True,
                             plasticity={"kl": 0.01})
        restored = LineageManager(manager.state_dict())
        self.assertIsNone(restored.active_lineage_id)
        with self.assertRaises(ValueError):
            restored.reset(chosen, reason="because", iteration=21020, evidence={"x": 1})


class BehaviorAndHealthTest(unittest.TestCase):
    def test_descriptor_is_log_only_and_masks_missing_fields(self):
        acc = BehaviorDescriptorAccumulator()
        acc.update({"altitude_mean_m": 1200.0, "damage_dealt": 35.0})
        descriptor = acc.descriptor()
        self.assertEqual(len(descriptor), len(FEATURES) * 3)
        state = acc.state_dict()
        self.assertTrue(state["log_only"])
        self.assertEqual(BehaviorDescriptorAccumulator(state).descriptor(), descriptor)

    def test_health_monitor_never_returns_automatic_action(self):
        monitor = PPOHealthMonitor(VNextConfig().health)
        report = monitor.evaluate(20020, {
            "approx_kl": 0.1, "clipfrac": 0.1, "actor_grad_norm": 1.0,
            "critic_grad_norm": 1.0, "entropy": 1.2,
            "explained_variance": 0.5, "fresh_fraction": 1.0, "policy_lag": 0})
        self.assertFalse(report.healthy)
        self.assertIsNone(report.automatic_action)
        self.assertIn("policy_kl_high", report.alerts)

    def test_health_monitor_keeps_head_probe_and_flags_support_collapse(self):
        monitor = PPOHealthMonitor(VNextConfig().health)
        report = monitor.evaluate(20021, {
            "approx_kl": 0.01, "clipfrac": 0.1, "actor_grad_norm": 1.0,
            "critic_grad_norm": 1.0, "entropy": 1.2,
            "explained_variance": 0.5, "fresh_fraction": 1.0, "policy_lag": 0,
            "support_retention": 0.10, "support_retention_head0": 0.08,
            "normalized_entropy_head0": 0.41, "ratio_q99": 1.17,
            "clipfrac_low": 0.03, "clipfrac_high": 0.07})
        self.assertFalse(report.healthy)
        self.assertIsNone(report.automatic_action)
        self.assertIn("action_support_contraction", report.alerts)
        self.assertEqual(report.metrics["ratio_q99"], 1.17)

    def test_missing_health_metrics_fail_closed(self):
        report = PPOHealthMonitor(VNextConfig().health).evaluate(20022, {
            "approx_kl": 0.01, "clipfrac": 0.1})
        self.assertFalse(report.healthy)
        self.assertFalse(report.complete)
        self.assertIn("required_health_metrics_missing", report.alerts)


class ShadowControllerTest(unittest.TestCase):
    @staticmethod
    def healthy_metrics():
        return {
            "approx_kl": 0.01, "clipfrac": 0.1,
            "actor_grad_norm": 1.0, "critic_grad_norm": 1.0,
            "entropy": 1.0, "explained_variance": 0.3,
            "fresh_fraction": 1.0, "policy_lag": 0}

    def test_shadow_proposes_without_training_mutation_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = VNextShadowController(folder)
            controller.observe_iteration(20020, self.healthy_metrics())
            archive = {"records": [record(i) for i in range(1, 6)]}
            proposal = controller.observe_milestone(
                20500, archive, current_id=5, active_ids=[1, 2, 3, 4])
            self.assertFalse(proposal["training_mutation_applied"])
            self.assertFalse(controller.may_mutate_training)
            state = controller.state_dict()
            restored = VNextShadowController(folder, state=state)
            self.assertEqual(restored.last_iteration, 20500)
            self.assertEqual(DecisionLog(Path(folder) / "decisions.jsonl").last_hash,
                             state["decision_log_sha256"])

    def test_shadow_log_tail_can_replay_from_older_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = VNextShadowController(folder)
            controller.persist()
            checkpoint_state = controller.state_dict()
            controller.observe_iteration(20020, self.healthy_metrics())
            controller.observe_iteration(20040, self.healthy_metrics())
            self.assertNotEqual(checkpoint_state["decision_log_sha256"],
                                controller.log.last_hash)
            replayed = VNextShadowController(folder, state=checkpoint_state)
            self.assertEqual(replayed.last_iteration, 20000)
            self.assertTrue(replayed.log.contains_hash(
                checkpoint_state["decision_log_sha256"]))

    def test_staged_log_tail_can_replay_from_older_model_checkpoint(self):
        """2026-09-03: found via a real resume of the 100K run.

        --save-every writes the model checkpoint on its own cadence,
        independent of on_milestone()'s persist(); a milestone's decisions
        can commit to decisions.jsonl before the next model checkpoint
        captures that iteration. Killing the process in that window (exactly
        what happened: the on-disk log's last entry was milestone 4000's
        commit, but checkpoint.pt was still at iteration 3900) used to raise
        unconditionally in staged mode -- the crash-tolerant replay this
        mirrors (test_shadow_log_tail_can_replay_from_older_model_checkpoint,
        above) was implemented for shadow mode only, even though every field
        restored afterward comes from the checkpoint's own embedded snapshot
        in either mode.
        """
        config = VNextConfig(mode=VNextMode.STAGED, stage=VNextStage.SPARSE_LEAGUE,
                             source_iteration=0)
        with tempfile.TemporaryDirectory() as folder:
            controller = VNextShadowController(folder, config=config)
            controller.persist()
            checkpoint_state = controller.state_dict()
            archive = {"records": [record(i) for i in range(1, 6)]}
            controller.observe_milestone(
                4000, archive, current_id=5, active_ids=[1, 2, 3, 4], commit=True)
            self.assertNotEqual(checkpoint_state["decision_log_sha256"],
                                controller.log.last_hash)
            replayed = VNextShadowController(folder, config=config, state=checkpoint_state)
            self.assertEqual(replayed.last_iteration, 0)
            self.assertTrue(replayed.log.contains_hash(
                checkpoint_state["decision_log_sha256"]))
            last_line = replayed.log.path.read_text(encoding="utf-8").splitlines()[-1]
            self.assertEqual(json.loads(last_line)["kind"],
                             "vnext_resume_from_checkpoint")

    def test_live_proposal_mode_does_not_persist_before_evaluation_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = VNextShadowController(folder)
            archive = {"records": [record(i) for i in range(1, 6)]}
            controller.observe_milestone(
                20500, archive, current_id=5, active_ids=[1, 2, 3, 4],
                commit=False)
            root = Path(folder)
            self.assertFalse((root / "shadow_state.json").exists())
            self.assertFalse((root / "decisions.jsonl").exists())
            self.assertFalse((root / "archive_index" / "archive_index.json").exists())


if __name__ == "__main__":
    unittest.main()
