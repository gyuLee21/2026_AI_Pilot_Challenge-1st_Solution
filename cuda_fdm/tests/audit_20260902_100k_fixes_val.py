"""CPU regressions for the 2026-09-02 audit fixes applied to the 100K package
(B1 migration contract derivation, B4 admission-cadence alarm, A4 bounded
payoff-row sweep, A7 score-provenance split).

No CUDA environment; no production control directory is touched.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import tempfile
import unittest

import torch

from cuda_fdm.league_vnext.active_roster import (
    LAYOUT, ActiveRosterSelector, rank_roster_candidates)
from cuda_fdm.league_vnext.contracts import HealthConfig, VNextConfig
from cuda_fdm.league_vnext.shadow import VNextShadowController
from cuda_fdm.ppo_gpu import EXPLOITER_CURRICULUM_BLOCK_ITERS, PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.tests.pool_episode_val import StatefulToy


_STUB_EVAL_RESULT = {
    "score": .5, "games": 8, "wins": 0, "draws": 8, "losses": 0,
    "lcb95": .5, "ucb95": .5, "paired_blocks": 4,
    "alt_loss_rate": .0, "opponent_alt_loss_rate": .0,
    "wins_by_opponent_altitude": 0, "wins_by_opponent_altitude_rate": 0.,
    "wins_by_opponent_altitude_share": 0., "losses_by_own_altitude": 0,
    "losses_by_own_altitude_rate": 0.,
    "altitude_diagnostic_protocol": "mirrored_terminal_cause_observability_v2",
}


def _league_trainer(temp, **overrides):
    cfg_kwargs = dict(
        device="cpu", architecture="mlp", hidden=(8, 8), num_bins=3, gru_size=0,
        normalize_obs=False, opp_sample=False, rollout_steps=1,
        sched_period=0, milestone_period=0, exploiter_iters=0,
        league_enabled=True, league_dir=str(Path(temp) / "league"),
        league_active_cap=24, aux_pred=False, seed=3)
    cfg_kwargs.update(overrides)
    return PPOGPUTrainer(StatefulToy(8), PPOGPUConfig(**cfg_kwargs))


class StagedExploiterAdmissionPathTest(unittest.TestCase):
    """2026-09-03: a from-scratch staged run archived two exploiters that both
    crossed their win target, and neither ever entered the pool. The cause was
    not a broken admission rule -- in staged mode
    `_evaluate_and_admit_exploiter` deliberately performs no inline evaluation
    (that cost up to 1,024 games per candidate) and instead defers to the
    payoff-graph budget: screening at the next milestone, confirmatory at the
    one after, so admission lands two milestones (1,000 main iterations) after
    the exploiter is trained. The run was stopped at iteration 1,100, before
    the first admission was even due. These tests pin that state machine down
    so "the exploiter never got admitted" is always distinguishable from
    "the exploiter has not reached its confirmatory milestone yet".
    """

    @staticmethod
    def _adapter_and_trainer(candidate_id=5, current_id=10):
        from types import SimpleNamespace
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=VNextConfig())
        records = {
            current_id: {"id": current_id, "admitted": True, "kind": "milestone_main"},
            candidate_id: {"id": candidate_id, "admitted": False,
                            "kind": "heldout_vnext_me-eie",
                            "admission_status": "heldout",
                            "profile": "standard", "metrics": {"role": "ME-EIE"}},
        }
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            cfg=SimpleNamespace(league_admission_score=0.65,
                                league_admission_lcb=0.60,
                                league_altitude_redteam_threshold=0.1))
        return adapter, trainer, records

    @staticmethod
    def _result(score, lcb, ucb):
        return {"score": score, "lcb95": lcb, "ucb95": ucb,
                "altitude_loss_measured": False,
                "scenario_bank_version": "bank", "evaluation_seed_block": 1}

    def test_screening_then_confirmatory_admits_a_strong_exploiter(self):
        adapter, trainer, records = self._adapter_and_trainer()
        # Milestone N+1: screening. The candidate is `left`, current is `right`.
        adapter._update_candidate_status(
            trainer, {"left": 5, "right": 10, "phase": "screening",
                      "reason": "admission_target"},
            self._result(0.72, 0.60, 0.84),
            candidate_ids={5}, current_id=10, iteration=1000)
        self.assertEqual(records[5]["screening_status"], "passed")
        # Screening alone must never admit -- that is the whole point of the
        # two-stage design.
        self.assertFalse(records[5]["admitted"])

        # Milestone N+2: confirmatory, comfortably over both thresholds.
        adapter._update_candidate_status(
            trainer, {"left": 5, "right": 10, "phase": "confirmatory",
                      "reason": "admission_target"},
            self._result(0.74, 0.66, 0.82),
            candidate_ids={5}, current_id=10, iteration=1500)
        self.assertTrue(records[5]["admitted"])
        self.assertEqual(records[5]["admission_status"], "probationary")
        self.assertEqual(records[5]["kind"], "exploiter_probationary")

    def test_weak_screening_is_marked_failed_and_stays_heldout(self):
        adapter, trainer, records = self._adapter_and_trainer()
        adapter._update_candidate_status(
            trainer, {"left": 5, "right": 10, "phase": "screening",
                      "reason": "admission_target"},
            self._result(0.40, 0.30, 0.50),
            candidate_ids={5}, current_id=10, iteration=1000)
        self.assertEqual(records[5]["screening_status"], "failed")
        self.assertFalse(records[5]["admitted"])

    def test_screening_pass_triggers_confirmatory_in_the_same_milestone(self):
        # 2026-09-03 (user decision): confirmatory must fire the same cycle
        # screening clears, not wait for the planner's next milestone. This
        # stubs _execute_query itself (already covered by
        # test_screening_then_confirmatory_admits_a_strong_exploiter above)
        # so this test is purely about whether the follow-up call happens
        # and is wired with the right phase/blocks/bank, not about what
        # _update_candidate_status does with the result.
        adapter, trainer, records = self._adapter_and_trainer()
        calls = []

        def fake_execute_query(trainer_, query, *, iteration, candidate_ids, current_id):
            calls.append(dict(query))
            if query["phase"] == "screening":
                records[5]["screening_status"] = "passed"
            return {"score": 0.9}

        adapter._execute_query = fake_execute_query
        screening_query = {"left": 5, "right": 10, "phase": "screening",
                           "reason": "admission_target", "minimum_blocks": 16,
                           "maximum_blocks": 128, "scenario_bank_version": "screening_v1",
                           "priority": 108.0}
        adapter._execute_query(trainer, screening_query, iteration=1000,
                               candidate_ids={5}, current_id=10)
        adapter._maybe_run_immediate_confirmatory(
            trainer, screening_query, iteration=1000, candidate_ids={5}, current_id=10)

        self.assertEqual(len(calls), 2)
        confirmatory_call = calls[1]
        self.assertEqual(confirmatory_call["phase"], "confirmatory")
        self.assertEqual(confirmatory_call["left"], 5)
        self.assertEqual(confirmatory_call["right"], 10)
        self.assertEqual(confirmatory_call["minimum_blocks"],
                         adapter.controller.config.payoff_graph.confirmatory_paired_blocks)
        self.assertEqual(confirmatory_call["scenario_bank_version"],
                         adapter.controller.config.payoff_graph.confirmatory_scenario_bank)

    def test_failed_screening_does_not_trigger_confirmatory(self):
        adapter, trainer, records = self._adapter_and_trainer()
        calls = []

        def fake_execute_query(trainer_, query, *, iteration, candidate_ids, current_id):
            calls.append(dict(query))
            records[5]["screening_status"] = "failed"
            return {"score": 0.3}

        adapter._execute_query = fake_execute_query
        screening_query = {"left": 5, "right": 10, "phase": "screening",
                           "reason": "admission_target", "minimum_blocks": 16,
                           "maximum_blocks": 128, "scenario_bank_version": "screening_v1",
                           "priority": 108.0}
        adapter._execute_query(trainer, screening_query, iteration=1000,
                               candidate_ids={5}, current_id=10)
        result = adapter._maybe_run_immediate_confirmatory(
            trainer, screening_query, iteration=1000, candidate_ids={5}, current_id=10)

        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)  # no second (confirmatory) call

    def test_confirmatory_phase_query_is_never_re_chained(self):
        # A query that is *already* confirmatory (or solver) must not trigger
        # another confirmatory call, regardless of any stale screening_status
        # left over on the record.
        adapter, trainer, records = self._adapter_and_trainer()
        records[5]["screening_status"] = "passed"
        adapter._execute_query = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not be called"))
        confirmatory_query = {"left": 5, "right": 10, "phase": "confirmatory",
                              "reason": "admission_target", "minimum_blocks": 128,
                              "maximum_blocks": 128, "scenario_bank_version": "confirmatory_v1",
                              "priority": 108.0}
        result = adapter._maybe_run_immediate_confirmatory(
            trainer, confirmatory_query, iteration=1500, candidate_ids={5}, current_id=10)
        self.assertIsNone(result)

    def test_confirmatory_below_lcb_threshold_is_refused(self):
        # score clears 0.65 but the lower confidence bound does not clear 0.60:
        # the admission rule requires both, exactly as the legacy path did.
        adapter, trainer, records = self._adapter_and_trainer()
        adapter._update_candidate_status(
            trainer, {"left": 5, "right": 10, "phase": "confirmatory",
                      "reason": "admission_target"},
            self._result(0.68, 0.52, 0.84),
            candidate_ids={5}, current_id=10, iteration=1500)
        self.assertFalse(records[5]["admitted"])
        self.assertEqual(records[5]["admission_status"], "heldout")
        self.assertEqual(records[5]["metrics"]["confirmatory_status"], "failed")


class AddArchivePolicyRoleScopedGuardTest(unittest.TestCase):
    def test_add_archive_policy_allows_same_id_under_two_roles(self):
        # 2026-09-03 (ported from 20K/v17 -- missing here was a real bug):
        # once a milestone can reuse a still-resident "recent" archive_id,
        # select_core() can later legitimately want the exact same
        # archive_id as "core" while the "recent" copy is still active. The
        # id-only guard this file had skipped an add whenever *any* role
        # already held that archive_id; it must be scoped to (archive_id,
        # role) or sync_archive_roles() silently loses a pool member
        # select_core() actually chose.
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp)
            bundle = trainer._policy_bundle()
            archive_id = trainer.archive.add(bundle["model"], bundle["norm"],
                                             kind="recent_main", iteration=1,
                                             admitted=True, payoff_eligible=True)
            first = trainer.pool.add_archive_policy(trainer.archive, archive_id, "recent")
            self.assertIsNotNone(first)
            # Same (archive_id, role) again must still be skipped (original guard).
            self.assertIsNone(
                trainer.pool.add_archive_policy(trainer.archive, archive_id, "recent"))
            # A different role for the same archive_id must now succeed.
            second = trainer.pool.add_archive_policy(trainer.archive, archive_id, "core")
            self.assertIsNotNone(second)
            roles = sorted(e["role"] for e in trainer.pool.entries
                           if e.get("archive_id") == archive_id)
            self.assertEqual(roles, ["core", "recent"])


class ExploiterEarlyStopTest(unittest.TestCase):
    def test_me_eie_still_stops_promptly_when_confidently_dominant(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(
                temp, exploiter_iters=200, exploiter_win_target=0.55,
                league_admission_games=400, rollout_steps=1,
                selfplay_ema_alpha=0.5, league_ema_half_life_games=4.0)
            trainer._exploiter_role = lambda: "ME-EIE"
            trainer._evaluate_pair = lambda *a, **k: dict(_STUB_EVAL_RESULT)
            trainer.train_exploiter()
            record = trainer.exploiter_history[-1]
            self.assertEqual(record["role"], "ME-EIE")
            # 2026-09-04: the bound moved 5 -> 7 deliberately, one iteration
            # for each of two fixes, both of which trade a single iteration
            # for a measurement that cannot be produced by one sample:
            #   +1  the session's opening _reset_env_state() synchronizes
            #       every lane, so that first iteration only completes the
            #       fastest-resolving episodes -- a censored, win-biased
            #       sample now excluded from wr_ema
            #       (ExploiterResetCensoringBiasTest).
            #   +1  early stop now needs the target to hold on consecutive
            #       credited iterations, since one game-heavy iteration can
            #       move the EMA most of the way alone
            #       (ExploiterEarlyStopConfirmationTest).
            # What this test guards -- a confidently dominant ME-EIE stops
            # promptly instead of burning all 200 -- is unchanged.
            # This toy finishes episodes every THREE rollout ticks. The first
            # batch is censored; two fresh confirming batches arrive at 6/9.
            self.assertEqual(record["iterations"], 9)


class MilestoneRecentDedupTest(unittest.TestCase):
    def test_milestone_reuses_coincident_recent_snapshot(self):
        # 2026-09-03 (ported from 20K/v17 -- missing here): when
        # recent_period divides milestone_period, without this fix the
        # milestone write and the recent-ring write each archive their own
        # byte-identical copy of the same snapshot at the same iteration.
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(
                temp, total_iterations=2, rollout_steps=1,
                league_recent_period=2, league_latest_period=1000,
                milestone_period=2, exploiter_iters=0,
                league_payoff_refresh_period=1000, league_redteam_period=1000)
            trainer.train()
            records_at_it2 = [r for r in trainer.archive.records.values()
                              if int(r["iteration"]) == 2]
            self.assertEqual(len(records_at_it2), 1,
                             "milestone and recent snapshot at the same "
                             "iteration must share one archive record")
            record = records_at_it2[0]
            self.assertEqual(record["kind"], "milestone_main")
            self.assertTrue(record["payoff_eligible"])
            self.assertEqual(trainer._last_milestone_archive_id, record["id"])


class AdmissionCadenceAlarmTest(unittest.TestCase):
    def _controller(self, root, threshold=3):
        cfg = VNextConfig(health=dataclasses.replace(
            HealthConfig(), maximum_milestones_without_admission=threshold))
        return VNextShadowController(root, config=cfg)

    def test_alert_fires_only_once_streak_reaches_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(temp, threshold=3)
            fired = [controller.observe_admission_cadence(it * 500, admitted_count=0)
                     for it in range(1, 6)]
            self.assertEqual(fired, [False, False, True, True, True])
            self.assertEqual(controller.consecutive_milestones_without_admission, 5)

    def test_any_admission_resets_the_streak(self):
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(temp, threshold=3)
            controller.observe_admission_cadence(500, admitted_count=0)
            controller.observe_admission_cadence(1000, admitted_count=0)
            fired = controller.observe_admission_cadence(1500, admitted_count=1)
            self.assertFalse(fired)
            self.assertEqual(controller.consecutive_milestones_without_admission, 0)

    def test_streak_survives_a_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(temp, threshold=3)
            controller.observe_admission_cadence(500, admitted_count=0)
            controller.observe_admission_cadence(1000, admitted_count=0)
            state = controller.state_dict()
            self.assertEqual(
                state["consecutive_milestones_without_admission"], 2)
            reloaded = self._controller(temp, threshold=3)
            reloaded.log = controller.log
            reloaded.load_state_dict(state)
            self.assertEqual(reloaded.consecutive_milestones_without_admission, 2)
            fired = reloaded.observe_admission_cadence(1500, admitted_count=0)
            self.assertTrue(fired)

    def test_negative_admitted_count_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            controller = self._controller(temp)
            with self.assertRaises(ValueError):
                controller.observe_admission_cadence(500, admitted_count=-1)


class BoundedPayoffRowTest(unittest.TestCase):
    def test_candidates_are_capped_to_resident_core_challenger_roster(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp, league_payoff_row_cap=3)
            bundle = trainer._policy_bundle()
            core_ids = []
            for _ in range(6):
                archive_id = trainer.archive.add(
                    bundle["model"], bundle["norm"], kind="milestone_main",
                    iteration=1, admitted=True, payoff_eligible=True)
                trainer.pool.add_archive_policy(trainer.archive, archive_id, "core")
                core_ids.append(archive_id)
            target_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=2, admitted=True, payoff_eligible=True)
            candidates = trainer._bounded_payoff_row_candidates(target_id, force_ids=())
            self.assertLessEqual(len(candidates), 3)
            self.assertTrue(set(candidates).issubset(set(core_ids)))

    def test_forced_ids_are_included_even_beyond_the_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp, league_payoff_row_cap=2)
            bundle = trainer._policy_bundle()
            for _ in range(4):
                archive_id = trainer.archive.add(
                    bundle["model"], bundle["norm"], kind="milestone_main",
                    iteration=1, admitted=True, payoff_eligible=True)
                trainer.pool.add_archive_policy(trainer.archive, archive_id, "core")
            forced_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=3, admitted=False, payoff_eligible=True)
            target_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=4, admitted=True, payoff_eligible=True)
            candidates = trainer._bounded_payoff_row_candidates(
                target_id, force_ids=(forced_id,))
            self.assertIn(forced_id, candidates)

    def test_target_id_never_appears_in_its_own_candidate_list(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp, league_payoff_row_cap=24)
            bundle = trainer._policy_bundle()
            target_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=1, admitted=True, payoff_eligible=True)
            trainer.pool.add_archive_policy(trainer.archive, target_id, "core")
            candidates = trainer._bounded_payoff_row_candidates(
                target_id, force_ids=(target_id,))
            self.assertNotIn(target_id, candidates)


class ScoreProvenanceTest(unittest.TestCase):
    def test_new_record_starts_with_no_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp)
            bundle = trainer._policy_bundle()
            archive_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=1, admitted=True, payoff_eligible=True)
            record = trainer.archive.records[archive_id]
            self.assertEqual(record["current_score_source"], "default")
            self.assertIsNone(record["online_ema_score"])
            self.assertIsNone(record["clean_paired_score"])

    def test_sync_writes_online_ema_without_touching_clean_score(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp)
            bundle = trainer._policy_bundle()
            archive_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=1, admitted=True, payoff_eligible=True)
            trainer.pool.add_archive_policy(trainer.archive, archive_id, "core")
            entry = next(e for e in trainer.pool.entries if e["archive_id"] == archive_id)
            entry["ema"] = 0.73
            trainer._sync_active_scores_to_archive()
            record = trainer.archive.records[archive_id]
            self.assertEqual(record["current_score"], 0.73)
            self.assertEqual(record["online_ema_score"], 0.73)
            self.assertIsNone(record["clean_paired_score"])
            self.assertEqual(record["current_score_source"], "online_ema")

    def test_refresh_meta_overwrites_with_clean_paired_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp)
            bundle = trainer._policy_bundle()
            current_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=1, admitted=True, payoff_eligible=True)
            other_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=2, admitted=True, payoff_eligible=True)
            trainer.pool.add_archive_policy(trainer.archive, other_id, "core")
            entry = next(e for e in trainer.pool.entries if e["archive_id"] == other_id)
            entry["ema"] = 0.20  # stale online EMA; clean edge below must win
            trainer._sync_active_scores_to_archive()
            trainer.archive.set_payoff(current_id, other_id, 0.85, 64)
            trainer.archive.refresh_meta(current_id)
            record = trainer.archive.records[other_id]
            self.assertAlmostEqual(record["clean_paired_score"], 0.85)
            self.assertAlmostEqual(record["current_score"], 0.85)
            self.assertEqual(record["current_score_source"], "clean_paired")
            # The stale online measurement stays visible, just no longer
            # the one "current_score" reflects.
            self.assertAlmostEqual(record["online_ema_score"], 0.20)

    def test_recent_only_record_keeps_online_ema_as_current(self):
        # A payoff_eligible=False "recent" snapshot never gets a payoff edge
        # against the current milestone, so refresh_meta() never touches it;
        # it should keep reporting the online EMA as its provenance.
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(temp)
            bundle = trainer._policy_bundle()
            current_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="milestone_main",
                iteration=1, admitted=True, payoff_eligible=True)
            recent_id = trainer.archive.add(
                bundle["model"], bundle["norm"], kind="recent_main",
                iteration=2, admitted=True, payoff_eligible=False)
            trainer.pool.add_archive_policy(trainer.archive, recent_id, "recent")
            entry = next(e for e in trainer.pool.entries if e["archive_id"] == recent_id)
            entry["ema"] = 0.61
            trainer._sync_active_scores_to_archive()
            trainer.archive.refresh_meta(current_id)
            record = trainer.archive.records[recent_id]
            self.assertEqual(record["current_score_source"], "online_ema")
            self.assertAlmostEqual(record["current_score"], 0.61)


class RosterRankingTest(unittest.TestCase):
    """The shadow proposal and the staged mutation now share this ranking."""

    @staticmethod
    def _records(overrides=None):
        records = {
            identity: {"id": identity, "iteration": 20000, "admitted": True,
                       "admission_status": "core", "nash_mass": identity / 100.0,
                       "metrics": {}}
            for identity in range(1, 8)
        }
        for identity, patch in (overrides or {}).items():
            records[identity].update(patch)
        return records

    def _allocate(self, records, solver_ids, *, recent_ids=(), current_id=None):
        strategic, challenger = rank_roster_candidates(
            records, solver_ids=solver_ids, recent_ids=recent_ids,
            current_id=current_id)
        proposal = ActiveRosterSelector(cap=sum(LAYOUT.values())).propose(
            latest_id=current_id, recent_ids=recent_ids,
            strategic_ids=strategic, challenger_ids=challenger)
        return proposal

    def test_probationary_takes_a_challenger_seat_before_core(self):
        # id 7 has the highest Nash mass but has not finished probation, so
        # it must serve as a challenger rather than occupy a core seat.
        records = self._records({7: {"admission_status": "probationary"}})
        proposal = self._allocate(records, range(1, 8), current_id=1)
        self.assertIn(7, proposal.challenger)
        self.assertNotIn(7, proposal.core)

    def test_matured_challenger_still_competes_into_core(self):
        # Same policy after maturation: probation is over, so its Nash mass
        # should win it a core seat instead of holding a challenger seat.
        records = self._records({7: {"admission_status": "solver_eligible"}})
        proposal = self._allocate(records, range(1, 8), current_id=1)
        self.assertIn(7, proposal.core)
        self.assertNotIn(7, proposal.challenger)

    def test_challenger_fill_takes_the_weakest_not_the_newest(self):
        # With no probationary queue the three challenger seats go to the
        # policies core wants least, so a newly created high-Nash policy is
        # not consumed by a challenger seat.
        records = self._records({7: {"iteration": 99999}})
        strategic, challenger = rank_roster_candidates(
            records, solver_ids=range(1, 8), current_id=1)
        self.assertEqual(strategic[0], 7)
        self.assertEqual(challenger[:3], [2, 3, 4])

    def test_recent_current_and_unadmitted_are_never_ranked(self):
        records = self._records({5: {"admitted": False}})
        strategic, challenger = rank_roster_candidates(
            records, solver_ids=range(1, 8), recent_ids=(6,), current_id=1)
        for excluded in (1, 5, 6):
            self.assertNotIn(excluded, strategic)
            self.assertNotIn(excluded, challenger)


class OpponentCategorySwitchResetTest(unittest.TestCase):
    def test_environment_resets_exactly_on_category_changes(self):
        # 2026-09-03 (ported from 20K/v17): the exploiter's opponent category
        # (frozen-main vs an easier variance-curriculum mixture) is decided
        # once per EXPLOITER_CURRICULUM_BLOCK_ITERS block via a smooth
        # probability band (_curriculum_probability), not a hard wr_ema<0.20
        # on/off gate, and consecutive curriculum blocks are forbidden so
        # wr_ema always gets refreshed against the real target at least every
        # other block. wr_ema starts at 0.5, above BAND_HIGH (0.30), so block
        # 1 is decided with p_curriculum=0 -> deterministically frozen-main.
        # The all-loss stub then drives wr_ema to 0 by block 1's end, below
        # BAND_LOW (0.10), so block 2 is decided with p_curriculum=1 ->
        # deterministically curriculum. Block 3 would see the same wr_ema=0
        # (curriculum never updates it) but the no-consecutive guard forces
        # it back to frozen-main regardless of probability. Sequence:
        # F(x block), C(x block), F(x1) -- deterministic at both probability
        # extremes, no seed sensitivity.
        block = EXPLOITER_CURRICULUM_BLOCK_ITERS
        with tempfile.TemporaryDirectory() as temp:
            trainer = _league_trainer(
                temp, exploiter_iters=2 * block + 1, exploiter_win_target=0.999,
                rollout_steps=1, selfplay_ema_alpha=0.5,
                league_ema_half_life_games=4.0)
            trainer._exploiter_role = lambda: "ME-EIE"
            trainer._evaluate_pair = lambda *a, **k: dict(_STUB_EVAL_RESULT)

            nenv = trainer.nenv
            real_collect = trainer.collect_rollout

            def fake_collect_rollout(opp_kind="pool", frozen_opp=None, update_norm=True):
                adv = torch.zeros(1, nenv)
                ret = torch.zeros(1, nenv)
                rs = {
                    "ret_sum": torch.tensor(0.0), "len_sum": torch.tensor(0.0),
                    "ep_count": torch.tensor(float(nenv)),
                    "win_sum": torch.tensor(0.0), "loss_sum": torch.tensor(float(nenv)),
                    "alt_loss_sum": torch.tensor(0.0),
                    "opponent_alt_loss_sum": torch.tensor(0.0),
                    "opponent_ids": trainer.pool.ids.clone(),
                    "win_by_opp": torch.zeros(trainer.pool.resident_size()),
                    "loss_by_opp": torch.zeros(trainer.pool.resident_size()),
                    "ep_by_opp": torch.zeros(trainer.pool.resident_size()),
                }
                return adv, ret, rs

            trainer.collect_rollout = fake_collect_rollout
            reset_calls = {"n": 0}
            real_reset = trainer._reset_env_state

            def counting_reset(*args, **kwargs):
                reset_calls["n"] += 1
                return real_reset(*args, **kwargs)

            trainer._reset_env_state = counting_reset
            try:
                trainer.train_exploiter()
            finally:
                trainer.collect_rollout = real_collect
            # _train_exploiter_inner resets once unconditionally before the
            # loop, then once more at each category change relative to the
            # previous iteration (iteration 1 counts as a change from the
            # initial "no category yet" sentinel): start->F, F->C, C->F is
            # three changes plus the unconditional reset: 1 + 3 = 4.
            self.assertEqual(reset_calls["n"], 4)


class MilestoneMainCorePromotionTest(unittest.TestCase):
    """2026-09-03: at iteration 3,900 of the from-scratch run all 16 core seats
    were still empty while seven `milestone_main` snapshots sat unused in the
    archive. Only exploiter admission (or a historical reactivation) ever set
    `admission_status`, and _select_solver_challengers() reads exactly that
    field, so Main's own past selves could not reach solver_ids and were never
    rankable for a core seat. An empty core also keeps `role_mixture` on its
    warm-up split, so the hard/variance/forgotten/coverage channels stay dark.
    """

    @staticmethod
    def _adapter_and_trainer(*, missing_pairs=0):
        from types import SimpleNamespace
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            graph=SimpleNamespace(
                missing_solver_pairs=lambda *a, **k: list(range(missing_pairs))))
        records = {
            4: {"id": 4, "kind": "milestone_main", "iteration": 500, "admitted": True},
            10: {"id": 10, "kind": "milestone_main", "iteration": 1000, "admitted": True},
            16: {"id": 16, "kind": "milestone_main", "iteration": 1500, "admitted": True},
            11: {"id": 11, "kind": "exploiter_probationary", "iteration": 1000,
                 "admitted": True, "admission_status": "probationary"},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        return adapter, trainer, records

    def test_past_main_snapshot_becomes_core_eligible(self):
        adapter, trainer, records = self._adapter_and_trainer()
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        # Cold start seeds core with the newest past self, not a near-random
        # snapshot from the first few hundred iterations.
        self.assertEqual(promoted, [16])
        self.assertEqual(records[16]["admission_status"], "solver_eligible")
        self.assertEqual(records[16]["metrics"]["core_promoted_at_iteration"], 2000)
        # Untouched snapshots stay candidates for a later milestone.
        self.assertIsNone(records[4].get("admission_status"))
        self.assertIsNone(records[10].get("admission_status"))

    def test_second_promotion_prefers_the_widest_history_gap(self):
        adapter, trainer, records = self._adapter_and_trainer()
        records[16]["admission_status"] = "solver_eligible"
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        # 1500 is held, so 500 (gap 1000) beats 1000 (gap 500).
        self.assertEqual(promoted, [4])

    def test_promotion_respects_the_completion_edge_cap(self):
        adapter, trainer, records = self._adapter_and_trainer(missing_pairs=999)
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [])
        self.assertIsNone(records[16].get("admission_status"))

    def test_promotion_never_consumes_a_challenger_seat(self):
        adapter, trainer, records = self._adapter_and_trainer()
        adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        # "probationary" is the challenger lane and belongs to unproven
        # policies; a past self skips it and goes straight to core candidacy.
        self.assertNotEqual(records[16]["admission_status"], "probationary")
        from cuda_fdm.league_vnext.active_roster import ordered_probationary_ids
        self.assertEqual(ordered_probationary_ids(records), [11])

    def test_promotion_stops_at_the_solver_policy_cap(self):
        """_select_solver_challengers() fills challenger seats up to
        19 - len(incumbents), so current Main + incumbents + challengers can
        already sit exactly on solver_policy_cap (20). Stacking a promotion on
        top of a full population made strategic_active_ids() raise.
        """
        adapter, trainer, records = self._adapter_and_trainer()
        cap = adapter.controller.config.active_game.solver_policy_cap
        incumbents = [{"role": "core", "archive_id": 100 + i}
                      for i in range(cap - 4)]
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22,
            incumbent_entries=incumbents,
            reserved_challenger_ids=[201, 202, 203])
        self.assertEqual(promoted, [])
        self.assertIsNone(records[16].get("admission_status"))

    def test_the_current_milestone_main_is_never_promoted(self):
        adapter, trainer, records = self._adapter_and_trainer()
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=1500, current_id=16, incumbent_entries=[])
        self.assertNotIn(16, promoted)


class AltitudeSentinelDirectEntryTest(unittest.TestCase):
    """2026-09-03 (user decision): every ALTITUDE_SENTINEL_PERIOD iterations the
    scheduled exploiter is replaced by a forced altitude hunter that enters the
    roster without screening/confirmatory. Losing altitude is an instant loss,
    so a specialist that forces it is worth practising against even at a win
    rate the admission bar rejects. Entry is all that is granted: the sentinel
    is then ranked and evicted by the same Nash machinery as any challenger.
    """

    def test_schedule_preserves_history_then_aligns_after_36000(self):
        from cuda_fdm.ppo_gpu import (ALTITUDE_SENTINEL_LEGACY_THROUGH,
                                      ALTITUDE_SENTINEL_OFFSET,
                                      ALTITUDE_SENTINEL_PERIOD,
                                      PPOGPUTrainer)
        fires = PPOGPUTrainer._is_altitude_sentinel_milestone
        self.assertTrue(fires(ALTITUDE_SENTINEL_OFFSET))
        self.assertTrue(fires(ALTITUDE_SENTINEL_OFFSET + ALTITUDE_SENTINEL_PERIOD))
        self.assertTrue(fires(ALTITUDE_SENTINEL_OFFSET + 2 * ALTITUDE_SENTINEL_PERIOD))
        self.assertTrue(fires(ALTITUDE_SENTINEL_LEGACY_THROUGH))
        self.assertTrue(fires(40000))
        self.assertTrue(fires(45000))
        self.assertTrue(fires(50000))
        # Ordinary milestones keep the scheduled role/profile bandit.
        self.assertFalse(fires(500))
        self.assertFalse(fires(1500))
        self.assertFalse(fires(ALTITUDE_SENTINEL_OFFSET + 500))
        self.assertFalse(fires(36500))
        self.assertFalse(fires(39500))
        self.assertFalse(fires(41000))
        self.assertFalse(fires(46000))
        self.assertFalse(fires(0))

    def test_sentinel_enters_as_probationary_without_evaluation(self):
        from types import SimpleNamespace
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        trainer = object.__new__(PPOGPUTrainer)
        records = {}
        archived = {}

        def fake_add(model, norm, *, kind, iteration, profile, admitted,
                     payoff_eligible, metrics):
            archived.update(kind=kind, admitted=admitted,
                            payoff_eligible=payoff_eligible)
            records[7] = {"id": 7, "kind": kind, "admitted": admitted,
                          "metrics": metrics}
            return 7

        trainer.archive = SimpleNamespace(records=records, add=fake_add)
        trainer.vnext_milestone_adapter = object()
        trainer.iteration = 6000
        trainer._altitude_sentinel_pending = True
        trainer._profile_bandit = {
            "ME-ERE": {"altitude_hunt": {"count": 0, "utility": 0.0,
                                         "staged_count": 0, "staged_utility": 0.0}}}

        accepted, archive_id, metrics = trainer._evaluate_and_admit_exploiter(
            {"model": None, "norm": None}, None, "ME-ERE", "altitude_hunt", 0.62)

        self.assertTrue(accepted)
        self.assertEqual(archive_id, 7)
        self.assertEqual(archived["kind"], "altitude_sentinel")
        self.assertTrue(archived["admitted"])
        # payoff_eligible keeps it inside the solver game -- that is what later
        # measures it and lets Nash mass drop it once Main outgrows it.
        self.assertTrue(archived["payoff_eligible"])
        self.assertEqual(records[7]["admission_status"], "probationary")
        self.assertTrue(metrics["altitude_sentinel"])
        self.assertFalse(metrics["vnext_fresh_evaluation_pending"])

    def test_ordinary_exploiter_still_defers_to_the_payoff_graph(self):
        from types import SimpleNamespace
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        trainer = object.__new__(PPOGPUTrainer)
        records = {}
        archived = {}

        def fake_add(model, norm, *, kind, iteration, profile, admitted,
                     payoff_eligible, metrics):
            archived.update(kind=kind, admitted=admitted)
            records[9] = {"id": 9, "kind": kind, "admitted": admitted,
                          "metrics": metrics}
            return 9

        trainer.archive = SimpleNamespace(records=records, add=fake_add)
        trainer.vnext_milestone_adapter = object()
        trainer.iteration = 2000
        trainer._altitude_sentinel_pending = False
        trainer._profile_bandit = {
            "ME-EIE": {"standard": {"count": 0, "utility": 0.0,
                                    "staged_count": 0, "staged_utility": 0.0}}}

        accepted, archive_id, metrics = trainer._evaluate_and_admit_exploiter(
            {"model": None, "norm": None}, None, "ME-EIE", "standard", 0.81)

        self.assertFalse(accepted)
        self.assertEqual(archived["kind"], "heldout_vnext_me-eie")
        self.assertFalse(archived["admitted"])
        self.assertFalse(metrics["altitude_sentinel"])
        self.assertTrue(metrics["vnext_fresh_evaluation_pending"])
        self.assertNotIn("admission_status", records[9])


class AdmissionEvidenceIntegrityTest(unittest.TestCase):
    """2026-09-03: id=11 displayed fresh_candidate_lcb95=0.490 -- below the
    0.50 bar -- while showing confirmatory_status="passed", which reads as an
    admission bug. It was not: the confirmatory that admitted it measured
    lcb=0.532, and a *solver* query later in the same milestone overwrote the
    shared fresh_candidate_* fields. on_milestone() freezes candidate_id_set
    before its query loop, so a policy admitted mid-loop stays in that set for
    every later query. Admission evidence is now written once and kept.
    """

    def _adapter_and_trainer(self):
        from types import SimpleNamespace
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=VNextConfig())
        records = {
            10: {"id": 10, "admitted": True, "kind": "milestone_main"},
            5: {"id": 5, "admitted": False, "kind": "heldout_vnext_me-ere",
                "admission_status": "heldout", "profile": "standard",
                "metrics": {"role": "ME-ERE"}},
        }
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            cfg=SimpleNamespace(league_admission_score=0.55,
                                league_admission_lcb=0.50,
                                league_altitude_redteam_threshold=0.1))
        return adapter, trainer, records

    @staticmethod
    def _result(score, lcb, ucb):
        return {"score": score, "lcb95": lcb, "ucb95": ucb,
                "altitude_loss_measured": False,
                "scenario_bank_version": "bank", "evaluation_seed_block": 1}

    def _run(self, adapter, trainer, phase, result, iteration=1500):
        adapter._update_candidate_status(
            trainer, {"left": 5, "right": 10, "phase": phase,
                      "reason": "admission_target"},
            result, candidate_ids={5}, current_id=10, iteration=iteration)

    def test_solver_traffic_cannot_rewrite_admission_evidence(self):
        adapter, trainer, records = self._adapter_and_trainer()
        self._run(adapter, trainer, "screening", self._result(0.60, 0.40, 0.80))
        self._run(adapter, trainer, "confirmatory", self._result(0.619, 0.532, 0.70))
        metrics = records[5]["metrics"]
        self.assertEqual(metrics["confirmatory_status"], "passed")
        self.assertTrue(records[5]["admitted"])
        decisive = metrics["admission_decision_eval"]
        # A solver query arriving later in the same milestone, with the stale
        # candidate set, must not touch the record that justified admission.
        self._run(adapter, trainer, "solver", self._result(0.613, 0.4896, 0.728))
        self.assertEqual(metrics["admission_decision_eval"], decisive)
        self.assertAlmostEqual(metrics["fresh_candidate_lcb95"], 0.532)
        self.assertGreaterEqual(metrics["fresh_candidate_lcb95"],
                                trainer.cfg.league_admission_lcb)

    def test_every_evaluation_is_appended_not_overwritten(self):
        adapter, trainer, records = self._adapter_and_trainer()
        self._run(adapter, trainer, "screening", self._result(0.60, 0.40, 0.80))
        self._run(adapter, trainer, "confirmatory", self._result(0.50, 0.30, 0.70))
        history = records[5]["metrics"]["eval_history"]
        self.assertEqual([event["phase"] for event in history],
                         ["screening", "confirmatory"])
        self.assertAlmostEqual(history[0]["score"], 0.60)
        self.assertAlmostEqual(history[1]["score"], 0.50)
        self.assertEqual(history[1]["opponent_id"], 10)

    def test_history_is_bounded(self):
        from cuda_fdm.league_vnext.live_adapter import EVAL_HISTORY_LIMIT
        adapter, trainer, records = self._adapter_and_trainer()
        for index in range(EVAL_HISTORY_LIMIT + 8):
            self._run(adapter, trainer, "screening",
                      self._result(0.60, 0.40, 0.80), iteration=1000 + index)
        history = records[5]["metrics"]["eval_history"]
        self.assertEqual(len(history), EVAL_HISTORY_LIMIT)
        # The oldest events are the ones dropped.
        self.assertEqual(history[-1]["iteration"], 1000 + EVAL_HISTORY_LIMIT + 7)


class DormantRejectionTest(unittest.TestCase):
    """2026-09-03: id=5 (screening ucb95 0.205) and id=17 (0.344) were re-tested
    at every milestone even though the optimistic end of their interval could
    never reach the 0.55 point threshold. Their admission_target edge carries
    +100 mandatory priority, so each one permanently reserved query budget.
    """

    def _adapter_and_trainer(self):
        return AdmissionEvidenceIntegrityTest._adapter_and_trainer(self)

    def _run(self, adapter, trainer, result):
        adapter._update_candidate_status(
            trainer, {"left": 5, "right": 10, "phase": "screening",
                      "reason": "admission_target"},
            result, candidate_ids={5}, current_id=10, iteration=2500)

    def test_hopeless_screening_retires_the_candidate(self):
        adapter, trainer, records = self._adapter_and_trainer()
        self._run(adapter, trainer,
                  AdmissionEvidenceIntegrityTest._result(0.09, 0.0, 0.205))
        self.assertEqual(records[5]["metrics"]["screening_status"], "failed")
        self.assertEqual(records[5]["admission_status"], "dormant_rejected")
        self.assertEqual(records[5]["metrics"]["dormant_rejected_at_iteration"], 2500)
        # Retirement is not deletion: the record stays available to the
        # historical audit, which can re-enrol it if Main later regresses.
        self.assertFalse(records[5]["admitted"])

    def test_borderline_failure_keeps_retrying(self):
        adapter, trainer, records = self._adapter_and_trainer()
        # Point estimate fails screening, but the interval still reaches past
        # the admission threshold -- a later Main could plausibly lose to it.
        self._run(adapter, trainer,
                  AdmissionEvidenceIntegrityTest._result(0.48, 0.30, 0.66))
        self.assertEqual(records[5]["metrics"]["screening_status"], "failed")
        self.assertNotEqual(records[5].get("admission_status"), "dormant_rejected")


class ScheduledProfileDrawTest(unittest.TestCase):
    """2026-09-03 (user decision): altitude_hunt has its own guaranteed
    schedule, so the ordinary milestone bandit must draw from the remaining
    profiles instead of duplicating it.
    """

    def test_altitude_hunt_is_never_drawn_on_an_ordinary_milestone(self):
        from cuda_fdm.ppo_gpu import (EXPLOITER_PROFILES,
                                      SCHEDULED_EXPLOITER_PROFILES)
        self.assertNotIn("altitude_hunt", SCHEDULED_EXPLOITER_PROFILES)
        self.assertEqual(set(SCHEDULED_EXPLOITER_PROFILES),
                         {"standard", "attack"})
        # The stored statistics still span every profile, so sentinel results
        # keep accumulating and the checkpoint schema is unchanged.
        self.assertIn("altitude_hunt", EXPLOITER_PROFILES)

    def test_both_roles_cycle_the_enabled_scheduled_profiles(self):
        from types import SimpleNamespace
        from cuda_fdm.ppo_gpu import PPOGPUTrainer, SCHEDULED_EXPLOITER_PROFILES
        from cuda_fdm.league_vnext.profile_stats import (EXPLOITER_ROLES,
                                                         empty_profile_stats)
        trainer = object.__new__(PPOGPUTrainer)
        trainer.archive = SimpleNamespace(records={})
        trainer.vnext_milestone_adapter = object()
        trainer._profile_bandit = empty_profile_stats()
        for role in EXPLOITER_ROLES:
            drawn = []
            for _ in SCHEDULED_EXPLOITER_PROFILES:
                profile, _mode = trainer._select_exploiter_profile(role)
                drawn.append(profile)
                trainer._update_profile_stat(role, profile, 0.8,
                                             source="staged_training_ema")
            self.assertEqual(sorted(drawn), sorted(SCHEDULED_EXPLOITER_PROFILES))


class FreshCandidateScreeningTest(unittest.TestCase):
    """2026-09-03: the side learner runs after on_milestone(), so its candidate
    could not be planned until the next milestone -- 500 iterations later,
    against a Main that had kept training. That single delayed number conflated
    "did it find a weakness" with "did the weakness survive 500 iterations".
    """

    @staticmethod
    def _adapter(records):
        from types import SimpleNamespace
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=VNextConfig())
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        return adapter, trainer

    def test_a_new_candidate_is_screened_in_its_own_milestone(self):
        records = {7: {"id": 7, "admitted": False, "kind": "heldout_vnext_me-eie"},
                   6: {"id": 6, "admitted": True, "kind": "milestone_main"}}
        adapter, trainer = self._adapter(records)
        seen = []
        adapter._execute_query = lambda trainer_, query, **kw: seen.append(query) or {}
        adapter._maybe_run_immediate_confirmatory = lambda *a, **k: None
        adapter.screen_fresh_candidate(trainer, archive_id=7, iteration=2000,
                                       current_id=6)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["phase"], "screening")
        self.assertEqual({seen[0]["left"], seen[0]["right"]}, {6, 7})
        self.assertEqual(
            seen[0]["minimum_blocks"],
            adapter.controller.config.payoff_graph.screening_paired_blocks)

    def test_an_altitude_sentinel_skips_screening(self):
        # Sentinels enter without evaluation by design; screening them would
        # re-impose the gate the schedule exists to bypass.
        records = {7: {"id": 7, "admitted": True, "kind": "altitude_sentinel"},
                   6: {"id": 6, "admitted": True, "kind": "milestone_main"}}
        adapter, trainer = self._adapter(records)
        seen = []
        adapter._execute_query = lambda trainer_, query, **kw: seen.append(query) or {}
        self.assertFalse(adapter.screen_fresh_candidate(
            trainer, archive_id=7, iteration=2000, current_id=6))
        self.assertEqual(seen, [])


class CoreRosterChurnLoggingTest(unittest.TestCase):
    """2026-09-03 (user request): propose() recomputes core from a full
    re-rank every milestone with no incumbent protection (ActiveRosterSelector
    has no tenure/hysteresis). take() only truncates once solver_eligible core
    candidates exceed 16 seats, so churn cannot occur while core is underfull
    -- but the enter/exit/tenure/occupancy trail needs to exist from the
    start so the first real 17-candidate contention is visible in the record.
    """

    @staticmethod
    def _adapter_and_trainer(*, core_role_ids=(), solver_eligible_ids=()):
        from types import SimpleNamespace
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        log = []
        adapter.controller = SimpleNamespace(
            config=VNextConfig(), last_solver_ids=list(solver_eligible_ids),
            last_index={"solver_eligible_ids": list(solver_eligible_ids), "reasons": {}},
            log=SimpleNamespace(append=lambda kind, payload, iteration:
                                log.append((kind, payload, iteration))))
        records = {identity: {"id": identity, "kind": "milestone_main",
                              "admitted": True, "admission_status": "solver_eligible",
                              "nash_mass": 0.1 * identity, "metrics": {}}
                  for identity in solver_eligible_ids}
        records[99] = {"id": 99, "kind": "milestone_main", "admitted": True}
        pool_entries = [{"role": "core", "archive_id": identity, "retired": False}
                        for identity in core_role_ids]
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records,
                                    persist=lambda: None,
                                    state_dict=lambda: {}),
            pool=SimpleNamespace(active_entries=lambda: pool_entries,
                                 sync_archive_roles=lambda *a, **k: None),
            cfg=SimpleNamespace(milestone_period=500),
            _refresh_weights=lambda: None)
        return adapter, trainer, records, log

    # rank_roster_candidates()'s challenger list is filled from the *bottom*
    # of the strategic ranking (propose()'s own note: "the policies core
    # wants least"), and any non-probationary eligible policy can appear on
    # it -- not just probationary ones. take() resolves an overlap in favour
    # of the challenger seat. With more eligible candidates than the 3
    # challenger seats, the lowest-nash_mass ones absorb that fill and the
    # top-ranked target is never at risk of being pulled into it by accident.
    _DECOYS = (1, 2, 3, 4)   # nash_mass 0.1..0.4, always ranked below target=9

    def test_new_core_entrant_is_stamped_and_logged(self):
        adapter, trainer, records, log = self._adapter_and_trainer(
            core_role_ids=(), solver_eligible_ids=(*self._DECOYS, 9))
        adapter.refresh_roster(trainer, iteration=2000, current_id=99)
        self.assertEqual(records[9]["metrics"]["core_entered_at_iteration"], 2000)
        diffs = [payload for kind, payload, it in log if kind == "core_roster_diff"]
        self.assertEqual(len(diffs), 1)
        entered_ids = {e["archive_id"] for e in diffs[0]["core_entered"]}
        self.assertIn(9, entered_ids)
        self.assertEqual(diffs[0]["core_exited"], [])

    def test_no_churn_while_core_has_open_seats(self):
        # Two milestones with the same underfull population: nothing enters
        # or exits a second time.
        adapter, trainer, records, log = self._adapter_and_trainer(
            core_role_ids=(), solver_eligible_ids=(*self._DECOYS, 9))
        adapter.refresh_roster(trainer, iteration=2000, current_id=99)
        first_core = [e["archive_id"] for e in
                     [payload for kind, payload, it in log
                      if kind == "core_roster_diff"][0]["core_entered"]]
        trainer.pool.active_entries = lambda: [
            {"role": "core", "archive_id": identity, "retired": False}
            for identity in first_core]
        adapter.refresh_roster(trainer, iteration=2500, current_id=99)
        second_diff = [payload for kind, payload, it in log
                      if kind == "core_roster_diff"][1]
        self.assertEqual(second_diff["core_entered"], [])
        self.assertEqual(second_diff["core_exited"], [])

    def test_exit_records_tenure_and_reason(self):
        # 17 eligible candidates for 16 core seats: exactly one, the lowest
        # nash_mass, must lose its seat this milestone -- the only situation
        # where take()'s full re-rank can evict an incumbent. Three
        # probationary decoys fill all 3 challenger seats first, so the
        # challenger fill (which draws from the *bottom* of this same
        # ranking) never reaches into the 17 core candidates and this stays a
        # clean core-only eviction.
        solver_eligible_ids = tuple(range(1, 18))
        adapter, trainer, records, log = self._adapter_and_trainer(
            core_role_ids=(1,), solver_eligible_ids=solver_eligible_ids)
        for identity in solver_eligible_ids:
            records[identity]["admission_status"] = "solver_eligible"
        for identity in (901, 902, 903):
            records[identity] = {"id": identity, "kind": "exploiter_probationary",
                                 "admitted": True, "admission_status": "probationary",
                                 "nash_mass": 0.0, "metrics": {}}
        adapter.controller.last_solver_ids = list(solver_eligible_ids) + [901, 902, 903]
        adapter.controller.last_index = {
            "solver_eligible_ids": adapter.controller.last_solver_ids, "reasons": {}}
        records[1]["metrics"]["core_entered_at_iteration"] = 1000
        adapter.refresh_roster(trainer, iteration=2000, current_id=99)
        diffs = [payload for kind, payload, it in log if kind == "core_roster_diff"]
        exited = diffs[0]["core_exited"]
        self.assertEqual([e["archive_id"] for e in exited], [1])
        self.assertEqual(exited[0]["core_tenure_milestones"], 2)  # (2000-1000)/500
        self.assertEqual(exited[0]["exit_reason"], "strategic_capacity_displacement")


if __name__ == "__main__":
    unittest.main()
