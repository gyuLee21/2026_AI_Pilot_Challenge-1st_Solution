from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from cuda_fdm.league import LeagueArchive
from cuda_fdm.league_vnext.active_game import (
    select_historical_audits, strategic_active_ids)
from cuda_fdm.league_vnext.active_roster import LAYOUT
from cuda_fdm.league_vnext.archive_index import ArchiveMaterializedIndex
from cuda_fdm.league_vnext.budget import WallclockBudgetTracker
from cuda_fdm.league_vnext.contracts import PayoffGraphConfig, VNextConfig
from cuda_fdm.league_vnext.lineages import LineageManager, LineageRecord
from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
from cuda_fdm.league_vnext.payoff_graph import SparsePayoffGraph
from cuda_fdm.league_vnext.profile_stats import (
    EXPLOITER_PROFILES, EXPLOITER_ROLES,
    empty_profile_stats, normalise_profile_stats)
from cuda_fdm.league_vnext.query_planner import PayoffQueryPlanner
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.tests.pool_identity_audit import ToyEnv


def policy_record(identity: int, *, admitted=True):
    return {
        "id": identity, "kind": "milestone_main", "iteration": identity * 500,
        "admitted": admitted, "payoff_eligible": True,
        "current_score": 0.5, "past_best": 0.5, "regression": 0.0,
        "nash_mass": 0.0, "sha256": f"{identity:064x}", "metrics": {},
    }


class SparseTransactionRecoveryTest(unittest.TestCase):
    def test_partial_resume_is_idempotent_and_wdl_is_not_doubled(self):
        graph = SparsePayoffGraph()
        txid = graph.plan(
            1, 2, "block", reason="resume", iteration=20000,
            phase="solver", evaluator_protocol="eval-v1",
            scenario_bank_version="solver-v1")
        graph.commit(
            txid, [1.0, 0.0], iteration=20000, complete=False,
            paired_game_ids=["g0", "g1"], wins=1, draws=0, losses=1)
        restored = SparsePayoffGraph(state=graph.state_dict())
        restored.commit(
            txid, [1.0, 0.0], iteration=20001, complete=False,
            paired_game_ids=["g0", "g1"], wins=1, draws=0, losses=1)
        self.assertEqual(restored.transactions[txid]["pending_wdl"]["wins"], 1.0)
        estimate = restored.commit(
            txid, [1.0, 0.0], iteration=20001, complete=True,
            paired_game_ids=["g2", "g3"], wins=1, draws=0, losses=1)
        self.assertEqual(estimate.wins, 2.0)
        self.assertEqual(estimate.losses, 2.0)
        self.assertEqual(estimate.paired_blocks, 4)

    def test_partial_overlap_is_rejected(self):
        graph = SparsePayoffGraph()
        txid = graph.plan(1, 2, "block", reason="overlap", iteration=1)
        graph.commit(txid, [0.5, 0.5], iteration=1, complete=False,
                     paired_game_ids=["g0", "g1"])
        with self.assertRaises(RuntimeError):
            graph.commit(txid, [0.5, 0.5], iteration=2, complete=False,
                         paired_game_ids=["g1", "g2"])

    def test_screening_is_not_solver_evidence_and_uncertainty_is_neutral(self):
        graph = SparsePayoffGraph()
        screen = graph.plan(
            1, 2, "screen", reason="screen", iteration=1,
            phase="screening", evaluator_protocol="eval",
            scenario_bank_version="screen-bank")
        graph.commit(screen, [1.0] * 16, iteration=1)
        self.assertFalse(graph.estimate(1, 2).known)
        self.assertTrue(graph.estimate(1, 2, include_screening=True).known)
        solver = graph.plan(
            1, 2, "solver", reason="solver", iteration=2,
            phase="solver", evaluator_protocol="eval",
            scenario_bank_version="solver-bank")
        graph.commit(solver, [0.0, 1.0] * 16, iteration=2)
        self.assertEqual(graph.conservative_value(1, 2), 0.5)
        nash = graph.conservative_nash([1, 2])
        self.assertAlmostEqual(nash[1], 0.5, places=6)
        self.assertAlmostEqual(nash[2], 0.5, places=6)

    def test_missing_edge_is_rejected_by_nash_instead_of_becoming_a_draw(self):
        graph = SparsePayoffGraph()
        with self.assertRaisesRegex(RuntimeError, "missing payoff edges"):
            graph.conservative_nash([1, 2])
        screen = graph.plan(
            1, 2, "screen-only", reason="screen", iteration=1,
            phase="screening", evaluator_protocol="eval",
            scenario_bank_version="screen-bank")
        graph.commit(screen, [0.5] * 64, iteration=1)
        with self.assertRaises(RuntimeError):
            graph.conservative_nash([1, 2])

    def test_solver_completion_is_scoped_to_compatible_protocol_and_bank(self):
        config = VNextConfig().payoff_graph
        graph = SparsePayoffGraph()
        wrong = graph.plan(
            1, 2, "wrong", reason="wrong", iteration=1,
            phase="solver", evaluator_protocol="obsolete-evaluator",
            scenario_bank_version=config.solver_scenario_bank)
        graph.commit(wrong, [1.0] * config.solver_paired_blocks, iteration=1)
        self.assertFalse(graph.solver_edge_eligible(
            1, 2, minimum_blocks=config.solver_paired_blocks,
            evaluator_protocol=config.evaluator_protocol,
            scenario_bank_versions=config.solver_compatible_scenario_banks))
        migrated = graph.plan(
            1, 2, "legacy-compatible", reason="migrated", iteration=2,
            phase="migrated", evaluator_protocol=config.evaluator_protocol,
            scenario_bank_version="legacy_clean_bank_v1")
        graph.commit(migrated, [0.5] * config.solver_paired_blocks, iteration=2)
        self.assertTrue(graph.solver_edge_eligible(
            1, 2, minimum_blocks=config.solver_paired_blocks,
            evaluator_protocol=config.evaluator_protocol,
            scenario_bank_versions=config.solver_compatible_scenario_banks))

    def test_confident_cycle_requires_decisive_solver_edges(self):
        graph = SparsePayoffGraph()
        for left, right, score in ((1, 2, 1.0), (2, 3, 1.0), (1, 3, 0.0)):
            txid = graph.plan(
                left, right, f"{left}:{right}", reason="cycle", iteration=1,
                phase="solver", evaluator_protocol="eval",
                scenario_bank_version="solver-bank")
            graph.commit(txid, [score] * 64, iteration=1)
        self.assertEqual(graph.confident_cycle_members([1, 2, 3]), (1, 2, 3))


class ActiveGameAndHistoricalAuditTest(unittest.TestCase):
    def test_solver_membership_excludes_latest_and_recent_curriculum(self):
        entries = [
            {"role": "latest", "archive_id": 99},
            {"role": "recent", "archive_id": 98},
            {"role": "core", "archive_id": 1},
            {"role": "challenger", "archive_id": 2},
            {"role": "core", "archive_id": 7},
        ]
        self.assertEqual(
            strategic_active_ids(entries, current_id=7), [7, 1, 2])

    def test_structured_audit_is_bounded_and_only_uses_pre20k_archive(self):
        records = {identity: policy_record(identity) for identity in range(1, 9)}
        records[1]["regression"] = 0.4
        records[2]["current_score"] = 0.1
        records[8]["iteration"] = 20500
        targets, cursor = select_historical_audits(
            records, source_iteration=20000, active_ids=[7], cycle_ids=[3],
            # This fixture tests a four-slot round-robin allocation, not the
            # live sixteen-slot (2/2/12/0) default. Keep every quota explicit
            # so changing production coverage cannot silently resize it.
            cursor=0, config=replace(VNextConfig().active_game,
                                     cold_cycle_quota=1,
                                     cold_regression_quota=1,
                                     cold_stale_quota=1,
                                     cold_rotation_quota=1))
        ids = [target.archive_id for target in targets]
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), 4)
        self.assertIn(3, ids)
        self.assertNotIn(7, ids)
        self.assertNotIn(8, ids)
        self.assertGreaterEqual(cursor, 0)
        second, _ = select_historical_audits(
            records, source_iteration=20000, active_ids=[7], cycle_ids=[3],
            cursor=cursor, config=replace(VNextConfig().active_game,
                                          cold_cycle_quota=1,
                                          cold_regression_quota=1,
                                          cold_stale_quota=1,
                                          cold_rotation_quota=1))
        first_rotation = next(target.archive_id for target in targets
                              if target.reason == "round_robin")
        second_rotation = next(target.archive_id for target in second
                               if target.reason == "round_robin")
        self.assertNotEqual(first_rotation, second_rotation)

    def test_active_game_completion_fills_only_missing_edges_then_nash_is_strict(self):
        config = VNextConfig()
        graph = SparsePayoffGraph()

        def commit(left, right, name, blocks):
            txid = graph.plan(
                left, right, name, reason=name, iteration=21000,
                phase="solver", evaluator_protocol=config.payoff_graph.evaluator_protocol,
                scenario_bank_version=config.payoff_graph.solver_scenario_bank)
            graph.commit(txid, [0.5] * blocks, iteration=21000)

        commit(1, 2, "cached", config.payoff_graph.solver_paired_blocks)
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=config, graph=graph)

        def execute(_trainer, query, **_kwargs):
            commit(query["left"], query["right"], query["reason"],
                   query["minimum_blocks"])
            return {}

        adapter._execute_query = execute
        queries = adapter._complete_active_solver_game(
            SimpleNamespace(), policy_ids=[1, 2, 3], iteration=21000,
            current_id=3, candidate_ids=set())
        self.assertEqual(len(queries), 2)
        self.assertEqual(
            set(graph.missing_solver_pairs(
                [1, 2, 3], minimum_blocks=config.payoff_graph.solver_paired_blocks)),
            set())
        self.assertEqual(set(graph.conservative_nash(
            [1, 2, 3], minimum_blocks=config.payoff_graph.solver_paired_blocks)),
            {1, 2, 3})

    def test_challenger_rows_over_pause_cap_remain_pending(self):
        base = VNextConfig()
        config = replace(
            base, active_game=replace(base.active_game, completion_edge_cap=6))
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=config, graph=SparsePayoffGraph())
        records = {
            identity: {
                "id": identity, "iteration": identity,
                "admitted": True, "admission_status": "probationary",
                "metrics": {}}
            for identity in (20, 21, 22)}
        records[20]["metrics"]["historical_counter_reactivated_at_iteration"] = 21900
        records[21]["metrics"]["solver_pending_since_iteration"] = 20500
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        accepted, pending = adapter._select_solver_challengers(
            trainer, current_id=30,
            incumbent_entries=[{"role": "core", "archive_id": 1},
                               {"role": "core", "archive_id": 2}],
            iteration=22000)
        self.assertEqual(accepted, [21])
        self.assertEqual(pending, [20, 22])
        self.assertNotIn("solver_pending_since_iteration", records[21]["metrics"])
        self.assertEqual(records[20]["metrics"]["solver_pending_since_iteration"], 22000)
        self.assertEqual(records[22]["metrics"]["solver_pending_since_iteration"], 22000)


class PlannerAndIndexTest(unittest.TestCase):
    def test_screening_then_fresh_confirmatory_and_resolved_edge_skip(self):
        config = PayoffGraphConfig(query_budget_per_milestone=4)
        planner = PayoffQueryPlanner(config)
        graph = SparsePayoffGraph()
        records = {1: policy_record(1), 2: policy_record(2, admitted=False)}
        first = planner.plan(graph, records, iteration=20000, active_ids=[1],
                             strategic_ids=[1], candidate_ids=[2])
        self.assertEqual(first[0].phase, "screening")
        txid = graph.plan(
            1, 2, "screen", reason="admission", iteration=20000,
            phase="screening", evaluator_protocol=config.evaluator_protocol,
            scenario_bank_version=config.screening_scenario_bank)
        graph.commit(txid, [0.75] * config.screening_paired_blocks, iteration=20000)
        self.assertEqual(planner.plan(
            graph, records, iteration=20500, active_ids=[1],
            strategic_ids=[1], candidate_ids=[2]), [])
        records[2]["screening_status"] = "passed"
        confirm = planner.plan(
            graph, records, iteration=20500, active_ids=[1],
            strategic_ids=[1], candidate_ids=[2])
        self.assertEqual(confirm[0].phase, "confirmatory")
        self.assertNotEqual(confirm[0].scenario_bank_version,
                            config.screening_scenario_bank)

    def test_materialized_index_rebuild_is_deterministic_and_unsynthesized(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = {"protocol": "p", "records": [policy_record(1), policy_record(2)]}
            graph = SparsePayoffGraph()
            index = ArchiveMaterializedIndex(folder)
            first = index.rebuild(archive, graph, iteration=20000)
            second = index.rebuild(archive, graph, iteration=20000)
            self.assertEqual(first, second)
            self.assertIsNone(first[1]["behavior_descriptor"])
            self.assertEqual(first[1]["payoff_fingerprint"], [0.5, 0.5])
            self.assertEqual(index.lookup(2)["archive_id"], 2)

    def test_altitude_sentinel_role_is_exact_and_old_state_is_repaired_on_load(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = policy_record(7)
            sentinel["kind"] = "altitude_sentinel"
            archive = {"protocol": "p", "records": [sentinel]}
            index = ArchiveMaterializedIndex(folder)
            index.rebuild(archive, SparsePayoffGraph(), iteration=13500,
                          persist=False)
            self.assertEqual(index.lookup(7)["policy_role"], "altitude_sentinel")

            stale = index.state_dict()
            stale["records"][0]["policy_role"] = "main_milestone"
            restored = ArchiveMaterializedIndex(folder, state=stale)
            self.assertEqual(restored.lookup(7)["policy_role"],
                             "altitude_sentinel")

    def test_non_target_candidate_anchor_is_a_solver_probe_not_confirmatory(self):
        config = PayoffGraphConfig(query_budget_per_milestone=4)
        records = {1: policy_record(1), 2: policy_record(2),
                   9: policy_record(9, admitted=False)}
        records[9]["screening_status"] = "passed"
        queries = PayoffQueryPlanner(config).plan(
            SparsePayoffGraph(), records, iteration=20500,
            active_ids=[1, 2], strategic_ids=[1, 2],
            candidate_ids=[9], admission_target_id=1)
        target = next(query for query in queries
                      if {query.left, query.right} == {1, 9})
        probe = next(query for query in queries
                     if {query.left, query.right} == {2, 9})
        self.assertEqual(target.phase, "confirmatory")
        self.assertEqual(probe.phase, "solver")
        self.assertIn("candidate_probe", probe.reason)


class ArchiveAndLineageTest(unittest.TestCase):
    def test_content_addressed_archive_deduplicates_without_identity_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = LeagueArchive(folder)
            model = {"weight": torch.tensor([1.0, 2.0])}
            norm = {"mean": torch.tensor([0.0])}
            first = archive.add(model, norm, kind="milestone_main", iteration=1)
            second = archive.add(model, norm, kind="milestone_main", iteration=2)
            self.assertNotEqual(first, second)
            self.assertEqual(archive.records[first]["file"], archive.records[second]["file"])
            self.assertEqual(len(list((Path(folder) / "policies").glob("*.pt"))), 1)

    def test_new_vnext_identity_never_reuses_identity_bound_legacy_payload(self):
        from cuda_fdm.league import (
            LEGACY_LEAGUE_PROTOCOL, LEAGUE_PROTOCOL, policy_content_sha256)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "policies").mkdir()
            model = {"weight": torch.tensor([1.0])}
            norm = {"mean": torch.tensor([0.0])}
            legacy_path = root / "policies" / "policy_00000.pt"
            torch.save({"protocol": LEGACY_LEAGUE_PROTOCOL, "archive_id": 0,
                        "model": model, "norm": norm}, legacy_path)
            content = policy_content_sha256(model, norm)
            import hashlib
            file_hash = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
            archive = LeagueArchive(folder, state={
                "protocol": LEAGUE_PROTOCOL, "root": str(root), "next_id": 1,
                "records": [{
                    "id": 0, "file": legacy_path.name, "sha256": file_hash,
                    "content_sha256": content,
                    "policy_protocol": LEGACY_LEAGUE_PROTOCOL,
                    "kind": "milestone_main", "iteration": 0,
                    "admitted": True, "payoff_eligible": True,
                }], "payoff": {}})
            identity = archive.add(
                model, norm, kind="milestone_main", iteration=1)
            self.assertEqual(identity, 1)
            self.assertNotEqual(archive.records[1]["file"], legacy_path.name)
            self.assertEqual(
                archive.load_policy(1)["protocol"], LEAGUE_PROTOCOL)

    def test_lineage_ghost_owner_is_rejected_and_redteam_queue_survives(self):
        record = LineageRecord("le", "LE", [1], "m", "o", "n", "r")
        manager = LineageManager()
        manager.register(record)
        manager.queue_candidate(
            7, role="LE", reason="heldout_altitude_redteam", iteration=21000,
            evidence={"target_main_alt_loss_rate": 0.25})
        restored = LineageManager(manager.state_dict())
        self.assertEqual(restored.candidate_queue[0]["archive_id"], 7)
        bad = manager.state_dict()
        bad["active_lineage_id"] = "le"
        with self.assertRaises(ValueError):
            LineageManager(bad)


class BudgetBoundaryAndLaunchTest(unittest.TestCase):
    def test_pool_rollout_scatters_both_altitude_directions(self):
        class AltitudeToy(ToyEnv):
            def __init__(self):
                super().__init__(nenv=4, episode_steps=1)

            def step(self, controls):
                obs, reward, done, info = super().step(controls)
                altitude = torch.full((4, 2), 1000.0)
                altitude[0:2, 0] = 100.0
                altitude[2, 1] = 100.0
                info["terminal_alt_m"] = altitude
                return obs, reward, done, info

        config = PPOGPUConfig(
            device="cpu", architecture="mlp", hidden=(8, 8), num_bins=3,
            normalize_obs=False, opp_sample=False, rollout_steps=1,
            sched_period=0, milestone_period=0, exploiter_iters=0)
        trainer = PPOGPUTrainer(AltitudeToy(), config)
        _, _, stats = trainer.collect_rollout(opp_kind="pool")
        self.assertTrue(stats["altitude_loss_measured"])
        self.assertEqual(float(stats["alt_loss_sum"]), 2.0)
        self.assertEqual(float(stats["opponent_alt_loss_sum"]), 1.0)
        self.assertEqual(float(stats["main_alt_loss_by_opp"].sum()), 2.0)
        self.assertEqual(float(stats["opponent_alt_loss_by_opp"].sum()), 1.0)
        self.assertEqual(float(stats["ep_by_opp"].sum()), 4.0)

    def test_staged_side_candidate_defers_legacy_dense_evaluation(self):
        trainer = object.__new__(PPOGPUTrainer)
        trainer.iteration = 20500
        trainer.vnext_milestone_adapter = object()
        trainer._profile_bandit = empty_profile_stats()
        trainer._archive_bundle = lambda candidate, kind, **kwargs: 77
        trainer._evaluate_pair = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy evaluator must not run for a staged candidate"))
        accepted, archive_id, metrics = trainer._evaluate_and_admit_exploiter(
            {"model": {}}, {"model": {}}, "ME-EIE", "altitude_hunt", 0.78)
        self.assertFalse(accepted)
        self.assertEqual(archive_id, 77)
        self.assertFalse(metrics["altitude_loss_measured"])
        self.assertIsNone(metrics["target_main_alt_loss_rate"])
        self.assertTrue(metrics["vnext_fresh_evaluation_pending"])

    def test_adaptive_profile_statistics_are_role_scoped(self):
        trainer = object.__new__(PPOGPUTrainer)
        trainer.archive = SimpleNamespace(records={})
        trainer._profile_bandit = empty_profile_stats()
        trainer._profile_bandit["ME-ERE"]["altitude_hunt"] = {
            "count": 50, "utility": 0.95}
        profile, _ = trainer._select_exploiter_profile("ME-EIE")
        self.assertEqual(profile, "standard")
        trainer._update_profile_stat("ME-EIE", "attack", 0.8)
        self.assertEqual(trainer._profile_bandit["ME-EIE"]["attack"]["count"], 1)
        self.assertEqual(trainer._profile_bandit["ME-ERE"]["altitude_hunt"]["count"], 50)

    def test_legacy_flat_profile_statistics_are_not_reused(self):
        legacy = {profile: {"count": 10, "utility": 0.9}
                  for profile in EXPLOITER_PROFILES}
        migrated = normalise_profile_stats(legacy)
        self.assertEqual(set(migrated), set(EXPLOITER_ROLES))
        self.assertTrue(all(
            item["count"] == 0
            for role in migrated.values() for item in role.values()))

    def test_successor_admission_is_role_profile_scoped_and_keeps_incumbent(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        records = {
            9: {"id": 9, "profile": "altitude_hunt", "admitted": True,
                "current_score": 0.20, "iteration": 9000,
                "metrics": {"role": "ME-EIE"}},
            10: {"id": 10, "profile": "altitude_hunt", "admitted": True,
                 "current_score": 0.91, "iteration": 10000,
                 "metrics": {"role": "ME-ERE"}},
            11: {"id": 11, "profile": "standard", "admitted": True,
                 "current_score": 0.80, "metrics": {"role": "ME-ERE"}},
            12: {"id": 12, "profile": "altitude_hunt", "admitted": True,
                 "current_score": 0.05, "metrics": {"role": "ME-ERE"}},
            20: {"id": 20, "profile": "altitude_hunt", "admitted": False,
                 "kind": "heldout_vnext", "metrics": {"role": "ME-EIE"}},
        }
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            pool=SimpleNamespace(active_entries=lambda: [
                {"role": "core", "archive_id": 9},
                {"role": "core", "archive_id": 10},
                {"role": "core", "archive_id": 11},
                {"role": "core", "archive_id": 12},
            ]))
        adapter._assign_successor_incumbents(trainer, [20])
        self.assertEqual(records[20]["successor_incumbent_id"], 9)
        evidence = adapter._successor_evidence(records[20]["metrics"], "confirmatory")
        evidence["candidate"] = {
            "score": 0.463, "lcb95": 0.401, "ucb95": 0.525,
            "current_id": 30, "iteration": 21000,
            "scenario_bank_version": "confirm-bank", "evaluation_seed_block": 77}
        evidence["incumbent"] = {
            "archive_id": 9, "score": 0.09, "lcb95": 0.04, "ucb95": 0.18,
            "current_id": 30, "iteration": 21000,
            "scenario_bank_version": "confirm-bank", "evaluation_seed_block": 77}
        records[20]["metrics"]["direct_admission_passed"] = False
        incumbent_before = deepcopy(records[9])
        adapter._resolve_successor_admission(trainer, 20, "confirmatory")
        self.assertTrue(records[20]["admitted"])
        self.assertEqual(records[20]["admission_status"], "probationary")
        self.assertTrue(records[20]["metrics"]["successor_admission_passed"])
        self.assertEqual(records[20]["metrics"]["successor_of_archive_id"], 9)
        self.assertEqual(records[9], incumbent_before)
        self.assertNotIn("successor_incumbent_id", records[12])

    def test_successor_candidate_and_incumbent_share_exact_scenario_seed(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        records = {
            1: {"id": 1, "admitted": True},
            5: {"id": 5, "profile": "altitude_hunt", "admitted": True,
                "metrics": {"role": "ME-EIE"}},
            8: {"id": 8, "profile": "altitude_hunt", "admitted": False,
                "iteration": 20500, "successor_incumbent_id": 5,
                "metrics": {"role": "ME-EIE"}},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        common = {
            "phase": "confirmatory", "scenario_bank_version": "confirm-bank",
        }
        candidate = adapter._successor_evaluation_context(
            trainer, {**common, "left": 1, "right": 8,
                      "reason": "admission_target"},
            candidate_ids={8}, current_id=1, iteration=20500)
        incumbent = adapter._successor_evaluation_context(
            trainer, {**common, "left": 1, "right": 5,
                      "reason": "successor_incumbent_baseline"},
            candidate_ids={8}, current_id=1, iteration=20500)
        self.assertIsNotNone(candidate)
        self.assertIsNotNone(incumbent)
        self.assertEqual(candidate["seed_block"], incumbent["seed_block"])
        self.assertEqual(candidate["seed_text"], incumbent["seed_text"])

    def test_successor_rejects_mismatched_scenario_seed_evidence(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        records = {
            9: {"id": 9, "admitted": True},
            20: {"id": 20, "admitted": False,
                 "metrics": {"successor_incumbent_id": 9,
                             "direct_admission_passed": False}},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        evidence = adapter._successor_evidence(
            records[20]["metrics"], "confirmatory")
        evidence["candidate"] = {
            "score": .8, "lcb95": .7, "ucb95": .9,
            "current_id": 30, "iteration": 21000,
            "scenario_bank_version": "confirm-bank", "evaluation_seed_block": 1}
        evidence["incumbent"] = {
            "archive_id": 9, "score": .2, "lcb95": .1, "ucb95": .3,
            "current_id": 30, "iteration": 21000,
            "scenario_bank_version": "confirm-bank", "evaluation_seed_block": 2}
        adapter._resolve_successor_admission(trainer, 20, "confirmatory")
        self.assertFalse(records[20]["admitted"])
        self.assertEqual(
            evidence["status"], "awaiting_same_scenario_seed_results")

    def test_successor_baseline_shares_candidate_phase_and_budget(self):
        config = PayoffGraphConfig(query_budget_per_milestone=2)
        records = {
            1: policy_record(1),
            5: policy_record(5),
            8: policy_record(8, admitted=False),
        }
        records[8]["successor_incumbent_id"] = 5
        queries = PayoffQueryPlanner(config).plan(
            SparsePayoffGraph(), records, iteration=20500,
            active_ids=[1, 5], strategic_ids=[1, 5],
            candidate_ids=[8], admission_target_id=1)
        self.assertEqual(len(queries), 2)
        self.assertEqual(
            [{query.left, query.right} for query in queries], [{1, 8}, {1, 5}])
        self.assertTrue(all(query.phase == "screening" for query in queries))
        self.assertIn("successor_incumbent_baseline", queries[1].reason)

    def test_newest_probationary_successor_gets_a_challenger_slot(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            last_solver_ids=[1, 21, 22, 23], last_index={
                "selected_ids": [1], "solver_eligible_ids": [1, 21, 22, 23],
                "reasons": {"1": ["nash_support"]}},
            log=SimpleNamespace(append=lambda *a, **k: None))
        records = {
            1: {"id": 1, "admitted": True, "nash_mass": 1.0},
            **{
                identity: {
                    "id": identity, "admitted": True,
                    "admission_status": "probationary", "nash_mass": 0.0}
                for identity in (20, 21, 22, 23)
            },
        }
        captured = {}
        pool = SimpleNamespace(
            active_entries=lambda: [],
            sync_archive_roles=lambda archive, core, challengers, coverage_ids=():
                captured.update(core=list(core), challengers=list(challengers)))
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records), pool=pool,
            _refresh_weights=lambda: None)
        adapter.refresh_roster(trainer, iteration=21000, current_id=1)
        self.assertEqual(captured["challengers"], [23, 22, 21])
        self.assertEqual(captured["core"], [])

    def test_post_side_candidate_can_be_forced_ahead_of_a_full_queue(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            last_solver_ids=[1, 20, 21, 22, 23], last_index={
                "selected_ids": [],
                "solver_eligible_ids": [1, 20, 21, 22, 23],
                "reasons": {}},
            log=SimpleNamespace(append=lambda *a, **k: None))
        records = {
            1: {"id": 1, "admitted": True, "nash_mass": 1.0},
            **{
                identity: {
                    "id": identity, "iteration": identity,
                    "admitted": True, "admission_status": "probationary",
                    "nash_mass": 0.0, "metrics": {}}
                for identity in (20, 21, 22, 23)
            },
        }
        captured = {}
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            pool=SimpleNamespace(
                active_entries=lambda: [],
                sync_archive_roles=lambda archive, core, challengers,
                coverage_ids=(): captured.update(
                    core=list(core), challengers=list(challengers))),
            _refresh_weights=lambda: None)
        adapter.refresh_roster(
            trainer, iteration=21000, current_id=1,
            forced_challenger_ids=(20,))
        self.assertEqual(captured["challengers"], [20, 23, 22])
        self.assertEqual(len(captured["challengers"]), 3)

    def test_full_resident_pool_accepts_post_side_swap_when_current_is_recent(self):
        """The live latest has no archive id, while its frozen snapshot is recent.

        At a milestone ``current_id`` therefore legitimately appears among
        the four resident recent ids.  The abstract selector deduplicates it
        against latest, but strategic refresh must validate the realized
        resident 1/4/16/3 layout instead of rejecting a safe challenger swap.
        """
        adapter = object.__new__(VNextMilestoneAdapter)
        selected = [136, 137, *range(3, 21)]
        adapter.controller = SimpleNamespace(
            last_solver_ids=[136, *range(2, 21)],
            last_index={"solver_eligible_ids": [136, *range(2, 21)],
                        "reasons": {}},
            log=SimpleNamespace(append=lambda *a, **k: None))
        records = {
            identity: {"id": identity, "iteration": identity,
                       "admitted": True, "admission_status": "core",
                       "nash_mass": float(identity), "metrics": {}}
            for identity in range(2, 21)}
        records.update({
            identity: {"id": identity, "iteration": identity,
                       "admitted": True, "nash_mass": 0.0, "metrics": {}}
            for identity in (133, 134, 135, 136)})
        records[137] = {
            "id": 137, "iteration": 11500, "admitted": True,
            "admission_status": "probationary", "nash_mass": 0.0,
            "metrics": {}}
        fixed = [
            {"role": "latest", "archive_id": None, "retired": False,
             "coverage": False},
            *({"role": "recent", "archive_id": identity, "retired": False,
               "coverage": False} for identity in (133, 134, 135, 136)),
        ]
        strategic = [
            *({"role": "core", "archive_id": identity, "retired": False,
               "coverage": False} for identity in range(2, 18)),
            *({"role": "challenger", "archive_id": identity,
               "retired": False, "coverage": False}
              for identity in range(18, 21)),
        ]

        class FullPool:
            def __init__(self):
                self.entries = [*fixed, *strategic]
                self.next_id = 24

            def active_entries(self):
                return [entry for entry in self.entries
                        if not entry.get("retired", False)]

            def sync_archive_roles(self, _archive, core, challengers,
                                   coverage_ids=()):
                coverage = set(coverage_ids)
                self.entries = [*fixed, *(
                    {"role": role, "archive_id": identity, "retired": False,
                     "coverage": identity in coverage}
                    for role, values in (("core", core),
                                         ("challenger", challengers))
                    for identity in values)]

            def _reindex(self):
                return None

            def _validate_layout(self):
                return None

        pool = FullPool()
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records), pool=pool,
            _refresh_weights=lambda: None,
            cfg=SimpleNamespace(milestone_period=500))
        adapter.refresh_roster(
            trainer, iteration=11500, current_id=136,
            forced_challenger_ids=(137,), selected_ids=selected,
            nash_override={identity: records[identity].get("nash_mass", 0.0)
                           for identity in selected})
        active = pool.active_entries()
        self.assertEqual(
            {role: sum(entry["role"] == role for entry in active)
             for role in LAYOUT}, LAYOUT)
        self.assertTrue(any(entry["role"] == "challenger"
                            and entry["archive_id"] == 137 for entry in active))
        archive_ids = [entry["archive_id"] for entry in active
                       if entry["archive_id"] is not None]
        self.assertEqual(len(archive_ids), len(set(archive_ids)))

    def test_full_resident_pool_allows_intentional_stale_retirement_underfill(self):
        """A full pool is a cap, not a promise that solved seats never empty.

        Stale retirement can remove up to two strategic members at a milestone.
        If no qualified replacement exists, the target roster must shrink by
        exactly that amount instead of treating the intentional vacancy as the
        full-pool aliasing bug guarded by the post-side swap test above.
        """
        for retired_count in (1, 2):
            with self.subTest(retired_count=retired_count):
                adapter = object.__new__(VNextMilestoneAdapter)
                previous_solver = [136, *range(2, 21)]
                selected = [136, *range(2, 21 - retired_count)]
                adapter.controller = SimpleNamespace(
                    last_solver_ids=previous_solver,
                    last_index={"solver_eligible_ids": previous_solver,
                                "reasons": {}},
                    log=SimpleNamespace(append=lambda *a, **k: None))
                records = {
                    identity: {
                        "id": identity, "iteration": identity,
                        "admitted": True, "admission_status": "core",
                        "nash_mass": float(identity), "metrics": {}}
                    for identity in range(2, 21)
                }
                records.update({
                    identity: {
                        "id": identity, "iteration": identity,
                        "admitted": True, "nash_mass": 0.0, "metrics": {}}
                    for identity in (133, 134, 135, 136)
                })
                fixed = [
                    {"role": "latest", "archive_id": None,
                     "retired": False, "coverage": False},
                    *({"role": "recent", "archive_id": identity,
                       "retired": False, "coverage": False}
                      for identity in (133, 134, 135, 136)),
                ]
                strategic = [
                    *({"role": "core", "archive_id": identity,
                       "retired": False, "coverage": False}
                      for identity in range(2, 18)),
                    *({"role": "challenger", "archive_id": identity,
                       "retired": False, "coverage": False}
                      for identity in range(18, 21)),
                ]

                class FullPool:
                    def __init__(self):
                        self.entries = [*fixed, *strategic]
                        self.next_id = 24

                    def active_entries(self):
                        return [entry for entry in self.entries
                                if not entry.get("retired", False)]

                    def sync_archive_roles(self, _archive, core, challengers,
                                           coverage_ids=()):
                        coverage = set(coverage_ids)
                        self.entries = [*fixed, *(
                            {"role": role, "archive_id": identity,
                             "retired": False,
                             "coverage": identity in coverage}
                            for role, values in (("core", core),
                                                 ("challenger", challengers))
                            for identity in values)]

                    def _reindex(self):
                        return None

                    def _validate_layout(self):
                        return None

                pool = FullPool()
                trainer = SimpleNamespace(
                    archive=SimpleNamespace(records=records), pool=pool,
                    _refresh_weights=lambda: None,
                    cfg=SimpleNamespace(milestone_period=500))
                adapter.refresh_roster(
                    trainer, iteration=13500, current_id=136,
                    selected_ids=selected,
                    nash_override={identity: records[identity].get(
                        "nash_mass", 0.0) for identity in selected})
                active = pool.active_entries()
                counts = {
                    role: sum(entry["role"] == role for entry in active)
                    for role in LAYOUT}
                self.assertEqual(counts["latest"], 1)
                self.assertEqual(counts["recent"], 4)
                self.assertEqual(counts["challenger"], 3)
                self.assertEqual(counts["core"], 16 - retired_count)
                self.assertEqual(len(active), 24 - retired_count)
                archive_ids = [entry["archive_id"] for entry in active
                               if entry["archive_id"] is not None]
                self.assertEqual(len(archive_ids), len(set(archive_ids)))

    def test_roster_sync_failure_rolls_back_resident_and_lifecycle_state(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            last_solver_ids=[1, 2],
            last_index={"solver_eligible_ids": [1, 2], "reasons": {}},
            log=SimpleNamespace(append=lambda *a, **k: None))
        records = {
            1: {"id": 1, "admitted": True, "admission_status": "admitted",
                "nash_mass": 1.0, "metrics": {}},
            2: {"id": 2, "admitted": True, "admission_status": "probationary",
                "nash_mass": 0.0, "metrics": {}},
        }
        entries = [
            {"id": 0, "role": "latest", "archive_id": None,
             "retired": False, "coverage": False},
            {"id": 1, "role": "core", "archive_id": 2,
             "retired": False, "coverage": False},
        ]

        class FailingPool:
            def __init__(self):
                self.entries = entries
                self.next_id = 2

            def active_entries(self):
                return [entry for entry in self.entries if not entry["retired"]]

            def sync_archive_roles(self, *_args, **_kwargs):
                self.entries[1]["retired"] = True
                self.entries.append({"id": 2, "role": "challenger",
                                     "archive_id": 2, "retired": False,
                                     "coverage": False})
                self.next_id = 3
                raise RuntimeError("synthetic load failure")

            def _reindex(self):
                return None

            def _validate_layout(self):
                return None

        pool = FailingPool()
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records), pool=pool,
            _refresh_weights=lambda: None,
            cfg=SimpleNamespace(milestone_period=500))
        with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
            adapter.refresh_roster(
                trainer, iteration=21000, current_id=1,
                forced_challenger_ids=(2,))
        self.assertEqual(pool.next_id, 2)
        self.assertEqual(len(pool.entries), 2)
        self.assertFalse(pool.entries[1]["retired"])
        self.assertEqual(records[2]["admission_status"], "probationary")

    def test_completed_probationary_challenger_can_compete_into_core(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            last_solver_ids=list(range(1, 21)),
            last_index={"solver_eligible_ids": list(range(1, 21)), "reasons": {}},
            log=SimpleNamespace(append=lambda *a, **k: None))
        records = {
            1: {"id": 1, "iteration": 21000, "admitted": True,
                "admission_status": "admitted", "nash_mass": 0.1, "metrics": {}},
            **{
                identity: {"id": identity, "iteration": 20000,
                           "admitted": True, "admission_status": "core",
                           "nash_mass": 0.5 - identity / 100.0, "metrics": {}}
                for identity in range(2, 20)
            },
            20: {"id": 20, "iteration": 20500, "admitted": True,
                 "admission_status": "probationary", "nash_mass": 0.9,
                 "metrics": {}},
        }
        records[21] = {"id": 21, "iteration": 19000, "admitted": True,
                       "admission_status": "core", "nash_mass": 0.0,
                       "metrics": {}}
        records[22] = {"id": 22, "iteration": 21000, "admitted": True,
                       "admission_status": "probationary", "nash_mass": 0.0,
                       "metrics": {}}
        entries = [{"role": "challenger", "archive_id": 20,
                    "coverage": False, "games": 0.0},
                   {"role": "core", "archive_id": 21,
                    "coverage": False, "games": 100.0},
                   {"role": "challenger", "archive_id": 22,
                    "coverage": False, "games": 0.0}]
        captured = {}
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            pool=SimpleNamespace(
                active_entries=lambda: entries,
                sync_archive_roles=lambda archive, core, challengers, coverage_ids=():
                    captured.update(core=list(core), challengers=list(challengers))),
            _refresh_weights=lambda: None)
        adapter.controller.config = VNextConfig()
        adapter._mature_completed_challengers(trainer, iteration=21500)
        self.assertEqual(records[20]["admission_status"], "probationary")
        entries[0]["games"] = 32.0
        adapter._mature_completed_challengers(trainer, iteration=21500)
        self.assertEqual(records[20]["admission_status"], "solver_eligible")
        adapter.refresh_roster(trainer, iteration=21500, current_id=1)
        self.assertIn(20, captured["core"])
        self.assertEqual(len(captured["core"]), 16)
        self.assertEqual(len(captured["challengers"]), 3)
        self.assertEqual(set(captured["core"] + captured["challengers"]), set(range(2, 21)))
        self.assertEqual(records[20]["admission_status"], "core")
        self.assertEqual(records[21]["admission_status"], "archive_only")
        # 21 held a core seat and lost it in this milestone's re-rank -- more
        # specific than the old blanket "bounded_solver_refresh", and exactly
        # what the core_roster_diff churn log now needs to distinguish a rank
        # eviction from any other reason a policy can leave the active set.
        self.assertEqual(
            records[21]["metrics"]["active_eviction_reason"],
            "strategic_capacity_displacement")
        self.assertEqual(records[22]["admission_status"], "probationary")
        self.assertEqual(records[22]["metrics"]["active_eviction_reason"],
                         "challenger_capacity_displacement")
        self.assertEqual(records[22]["metrics"]["solver_pending_since_iteration"],
                         21500)

    def test_pre20k_counter_reactivation_preserves_identity_and_gets_challenger_priority(self):
        adapter = object.__new__(VNextMilestoneAdapter)
        config = VNextConfig()
        graph = SparsePayoffGraph()
        txid = graph.plan(
            2, 30, "old-beats-current", reason="historical", iteration=21000,
            phase="solver", evaluator_protocol=config.payoff_graph.evaluator_protocol,
            scenario_bank_version=config.payoff_graph.solver_scenario_bank)
        graph.commit(txid, [1.0] * config.payoff_graph.solver_paired_blocks,
                     iteration=21000)
        adapter.controller = SimpleNamespace(
            config=config, graph=graph, last_solver_ids=[30, 2, 40],
            last_index={"solver_eligible_ids": [30, 2, 40], "reasons": {}},
            log=SimpleNamespace(append=lambda *a, **k: None))
        records = {
            2: {"id": 2, "kind": "heldout_old", "iteration": 9000,
                "admitted": False, "payoff_eligible": True,
                "current_score": 0.5, "past_best": 0.8, "metrics": {}},
            30: {"id": 30, "kind": "milestone_main", "iteration": 21000,
                 "admitted": True, "nash_mass": 1.0, "metrics": {}},
            40: {"id": 40, "kind": "exploiter", "iteration": 20500,
                 "admitted": True, "admission_status": "probationary",
                 "nash_mass": 0.0, "metrics": {}},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        self.assertTrue(adapter._record_historical_audit(
            trainer, archive_id=2, current_id=30, iteration=21000,
            reason="regression_sentinel"))
        self.assertEqual(records[2]["kind"], "heldout_old")
        self.assertEqual(records[2]["admission_status"], "probationary")
        self.assertEqual(
            records[2]["metrics"]["historical_counter_reactivated_at_iteration"],
            21000)

        captured = {}
        trainer.pool = SimpleNamespace(
            active_entries=lambda: [],
            sync_archive_roles=lambda archive, core, challengers, coverage_ids=():
                captured.update(core=list(core), challengers=list(challengers)))
        trainer._refresh_weights = lambda: None
        adapter.refresh_roster(trainer, iteration=21000, current_id=30)
        self.assertEqual(captured["challengers"][:2], [2, 40])

    def test_budget_reports_all_quantiles_and_eta(self):
        tracker = WallclockBudgetTracker(target_iteration=100000, window=16)
        for index in range(16):
            tracker.observe_main(seconds=index + 1, transitions=100,
                                 completed_games=index)
        tracker.add("evaluation", 5.0)
        tracker.add("side", 7.0)
        report = tracker.report(iteration=20000, milestone_period=500)
        self.assertEqual(set(report["seconds"]["main"]),
                         {"count", "p50", "p90", "p95", "p99"})
        self.assertGreater(report["rolling_eta_seconds"], 0.0)
        self.assertEqual(report["main_transitions"], 1600)

    def test_inflight_checkpoint_is_refused_before_state_serialization(self):
        trainer = object.__new__(PPOGPUTrainer)
        trainer._iteration_inflight = True
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                trainer.save(Path(folder) / "checkpoint.pt")

    def test_altitude_candidate_perspective_swaps_both_policy_and_crash(self):
        result = {
            "score": 0.3, "lcb95": 0.2, "ucb95": 0.4,
            "left_alt_loss_rate": 0.05, "right_alt_loss_rate": 0.25}
        score, lcb, target_alt = VNextMilestoneAdapter._candidate_perspective(
            result, candidate_id=1, left_id=1)
        self.assertEqual((score, lcb, target_alt), (0.3, 0.2, 0.25))
        score, lcb, target_alt = VNextMilestoneAdapter._candidate_perspective(
            result, candidate_id=2, left_id=1)
        self.assertAlmostEqual(score, 0.7)
        self.assertAlmostEqual(lcb, 0.6)
        self.assertEqual(target_alt, 0.05)


if __name__ == "__main__":
    unittest.main()
