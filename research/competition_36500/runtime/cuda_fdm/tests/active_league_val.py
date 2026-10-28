from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cuda_fdm.league import (ACTIVE_LAYOUT, LEAGUE_PROTOCOL, LeagueArchive,
                             count_aware_alpha, empirical_nash,
                             payoff_posterior_mean, role_mixture,
                             wilson_lower, paired_score_interval,
                             score_block_interval)
from cuda_fdm.ppo_gpu import (TRAINING_PROTOCOL, ActiveLeaguePool, PPOGPUConfig,
                              PPOGPUTrainer, RunningNorm, build_actor_critic,
                              role_profile_bandit_from_history)
from cuda_fdm.tests.pool_identity_audit import ToyEnv
from cuda_fdm.tests.pool_episode_val import StatefulToy


MODEL_KW = dict(obs_dim=12, act_dim=4, num_bins=5, hidden=(16, 16),
                activation="tanh", gru_size=0, architecture="mlp", encoder_depth=2)


class ActiveLeagueContractTest(unittest.TestCase):
    def test_role_profile_scheduler_does_not_mix_le_altitude_evidence_into_me(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, aux_pred=False, seed=5)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            for profile in ("standard", "attack", "defense"):
                trainer._role_profile_bandit["ME-EIE"][profile] = {
                    "count": 1, "utility": .7}
            trainer._role_profile_bandit["LE"]["altitude_hunt"] = {
                "count": 100, "utility": .95}
            profile, _ = trainer._select_exploiter_profile("ME-EIE")
            self.assertEqual(profile, "altitude_hunt")
            self.assertEqual(trainer._role_profile_bandit["LE"]["altitude_hunt"]["count"], 100)

    def test_le_profile_choice_ignores_other_role_admission_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, aux_pred=False, seed=5)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            trainer._role_profile_bandit["LE"] = {
                "standard": {"count": 2, "utility": .5},
                "altitude_hunt": {"count": 3, "utility": .8},
                "attack": {"count": 1, "utility": .9},
                "defense": {"count": 1, "utility": .1},
            }
            trainer.archive.records = {
                identity: {
                    "id": identity, "admitted": True,
                    "kind": "exploiter_me-eie", "profile": "attack",
                    "metrics": {"role": "ME-EIE"}}
                for identity in range(20)
            }
            profile, _ = trainer._select_exploiter_profile("LE")
            self.assertEqual(profile, "attack")

    def test_role_profile_scheduler_reconstructs_exact_role_evidence(self):
        history = [
            {"role": "LE", "reward_mode": "altitude_hunt",
             "admission": {"target_score": .4, "payoff_novelty": .2}},
            {"role": "LE", "reward_mode": "altitude_hunt",
             "admission": {"target_score": .6, "payoff_novelty": .0}},
            {"role": "ME-EIE", "reward_mode": "altitude_hunt",
             "admission": {"target_score": .8, "payoff_novelty": .4}},
        ]
        state = role_profile_bandit_from_history(history)
        self.assertEqual(state["LE"]["altitude_hunt"]["count"], 2)
        self.assertAlmostEqual(state["LE"]["altitude_hunt"]["utility"], .525)
        self.assertEqual(state["ME-EIE"]["altitude_hunt"]["count"], 1)
        self.assertAlmostEqual(state["ME-EIE"]["altitude_hunt"]["utility"], .9)
        self.assertEqual(state["ME-ERE"]["altitude_hunt"]["count"], 0)

    def test_pool_opponent_altitude_diagnostic_is_measured_not_zero_filled(self):
        rollout_source = inspect.getsource(PPOGPUTrainer.collect_rollout)
        exploiter_source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        self.assertNotIn('if opp_kind != "pool"', rollout_source)
        self.assertIn('win & (ta[:, 1] < self.min_alt) & done_b', rollout_source)
        self.assertIn('"observed_opponent_alt_loss_rate_v2"', exploiter_source)
        self.assertNotIn('"opponent_alt_loss_rate": target_alt_lr', exploiter_source)

    def test_role_mixture_and_cap(self):
        entries = [{"role": "latest", "ema": .5}]
        entries += [{"role": "recent", "ema": .6} for _ in range(4)]
        entries += [{"role": "core", "ema": .3, "coverage": i < 2} for i in range(16)]
        entries += [{"role": "challenger", "ema": .5} for _ in range(3)]
        probability = role_mixture(entries)
        self.assertEqual(len(probability), 24)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(probability[0]), .20, places=12)
        self.assertLessEqual(float(probability[1:].max()), .12 + 1e-12)
        self.assertTrue(np.all(probability > 0))

    def test_explicit_early_league_schedule(self):
        one_recent = role_mixture([
            {"role": "latest", "ema": .5}, {"role": "recent", "ema": .5}])
        np.testing.assert_allclose(one_recent, [.5, .5], rtol=0, atol=1e-12)
        four_recent = role_mixture(
            [{"role": "latest", "ema": .5}]
            + [{"role": "recent", "ema": .5} for _ in range(4)])
        np.testing.assert_allclose(four_recent, [.5, .125, .125, .125, .125],
                                   rtol=0, atol=1e-12)

    def test_count_ema_and_wilson(self):
        self.assertAlmostEqual(count_aware_alpha(512, 512), .5, places=12)
        self.assertAlmostEqual(count_aware_alpha(1024, 512), .75, places=12)
        self.assertGreater(wilson_lower(.7, 1024), .65)
        self.assertLess(wilson_lower(.7, 16), .6)

    def test_meta_nash_rock_paper_scissors(self):
        payoff = {}
        for left, right, score in ((0, 1, 1.), (1, 2, 1.), (2, 0, 1.)):
            payoff.setdefault(str(left), {})[str(right)] = {"score": score, "games": 64}
            payoff.setdefault(str(right), {})[str(left)] = {"score": 1-score, "games": 64}
        nash = empirical_nash([0, 1, 2], payoff, iterations=12000)
        self.assertAlmostEqual(sum(nash.values()), 1., places=9)
        for value in nash.values():
            self.assertLess(abs(value - 1/3), .03)

    def test_paired_bootstrap_uses_mirrored_blocks(self):
        first = np.asarray([1., 1., 0., 0.])
        swapped = np.asarray([0., 0., 1., 1.])
        low, high = paired_score_interval(first, swapped, seed=17)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_paired_interval_never_claims_zero_boundary_uncertainty(self):
        for count in (64, 128):
            all_win = np.ones(count)
            all_loss = np.zeros(count)
            win_low, win_high = score_block_interval(all_win, seed=count)
            loss_low, loss_high = score_block_interval(all_loss, seed=count)
            self.assertLess(win_low, 1.0)
            self.assertEqual(win_high, 1.0)
            self.assertEqual(loss_low, 0.0)
            self.assertGreater(loss_high, 0.0)
            self.assertAlmostEqual(win_low, 1.0 - loss_high, delta=.015)

    def test_payoff_accumulates_mirrored_wdl_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = LeagueArchive(Path(temp) / "league")
            model = build_actor_critic(**MODEL_KW)
            norm = RunningNorm(12, "cpu")
            left = archive.add(model.state_dict(), norm.state_dict(), kind="milestone_main",
                               iteration=1, admitted=True, payoff_eligible=True)
            right = archive.add(model.state_dict(), norm.state_dict(), kind="milestone_main",
                                iteration=2, admitted=True, payoff_eligible=True)
            archive.accumulate_payoff(left, right, wins=30, draws=4, losses=30,
                                      seed_block=1, lcb95=.4, ucb95=.6, paired_blocks=32,
                                      paired_scores=[0., 1.] * 16)
            width1 = archive.payoff_uncertainty(left, right)
            archive.accumulate_payoff(left, right, wins=60, draws=8, losses=60,
                                      seed_block=2, lcb95=.44, ucb95=.56, paired_blocks=64,
                                      paired_scores=[0., 1.] * 32)
            entry = archive.payoff[str(left)][str(right)]
            reverse = archive.payoff[str(right)][str(left)]
            self.assertEqual((entry["wins"], entry["draws"], entry["losses"]),
                             (90., 12., 90.))
            self.assertEqual((reverse["wins"], reverse["draws"], reverse["losses"]),
                             (90., 12., 90.))
            self.assertEqual([block["seed_block"] for block in entry["blocks"]], [1, 2])
            self.assertEqual(entry["blocks"][1]["paired_blocks"], 64)
            self.assertEqual(entry["paired_blocks"], 96)
            self.assertEqual(reverse["paired_blocks"], 96)
            self.assertLess(archive.payoff_uncertainty(left, right), width1)

    def test_meta_nash_keeps_uncertain_measured_edge(self):
        payoff = {
            "0": {"1": {"score": .54, "games": 607, "lcb95": .49, "ucb95": .59}},
            "1": {"0": {"score": .46, "games": 607, "lcb95": .41, "ucb95": .51}},
        }
        nash = empirical_nash([0, 1], payoff, iterations=2000)
        self.assertGreater(nash[0], .5)
        self.assertLess(nash[1], .5)

    def test_meta_nash_is_invariant_to_confidence_interval_threshold(self):
        base = {
            "0": {"1": {"score": .54, "games": 607, "lcb95": .49, "ucb95": .59}},
            "1": {"0": {"score": .46, "games": 607, "lcb95": .41, "ucb95": .51}},
        }
        narrow = copy.deepcopy(base)
        narrow["0"]["1"].update(lcb95=.53, ucb95=.55)
        narrow["1"]["0"].update(lcb95=.45, ucb95=.47)
        uncertain = empirical_nash([0, 1], base, iterations=2000)
        confident = empirical_nash([0, 1], narrow, iterations=2000)
        self.assertAlmostEqual(uncertain[0], confident[0], places=12)
        self.assertAlmostEqual(uncertain[1], confident[1], places=12)

    def test_paired_jeffreys_posterior_is_finite_and_complement_symmetric(self):
        forward = {
            "score": 1.0, "games": 4, "lcb95": .4, "ucb95": 1.0,
            "paired_blocks": 4,
            "blocks": [{"paired_scores": [1.0, 1.0, 1.0, 1.0]}],
        }
        reverse = {
            "score": 0.0, "games": 4, "lcb95": 0.0, "ucb95": .6,
            "paired_blocks": 4,
            "blocks": [{"paired_scores": [0.0, 0.0, 0.0, 0.0]}],
        }
        left = payoff_posterior_mean(forward)
        right = payoff_posterior_mean(reverse)
        self.assertLess(left, 1.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(left + right, 1.0, places=12)

    def test_archive_is_append_only_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = LeagueArchive(Path(temp) / "league")
            model = build_actor_critic(**MODEL_KW)
            norm = RunningNorm(12, "cpu")
            identity = archive.add(model.state_dict(), norm.state_dict(), kind="milestone_main",
                                   iteration=500, admitted=True, payoff_eligible=True)
            loaded = archive.load_policy(identity)
            self.assertEqual(loaded["protocol"], LEAGUE_PROTOCOL)
            restored = LeagueArchive(Path(temp) / "league", archive.state_dict())
            self.assertEqual(restored.records[identity]["iteration"], 500)

    def test_orphan_policy_after_failure_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "league"
            archive = LeagueArchive(root)
            model = build_actor_critic(**MODEL_KW)
            norm = RunningNorm(12, "cpu")
            first = archive.add(model.state_dict(), norm.state_dict(), kind="milestone_main",
                                iteration=500, admitted=True, payoff_eligible=True)
            recovery_state = copy.deepcopy(archive.state_dict())
            orphan = archive.add(model.state_dict(), norm.state_dict(), kind="milestone_main",
                                 iteration=1000, admitted=True, payoff_eligible=True)

            restored = LeagueArchive(root, recovery_state)
            self.assertEqual(set(restored.records), {first})
            self.assertEqual(restored.next_id, orphan + 1)
            replacement = restored.add(model.state_dict(), norm.state_dict(), kind="milestone_main",
                                       iteration=1000, admitted=True, payoff_eligible=True)
            self.assertEqual(replacement, orphan + 1)
            self.assertTrue((root / f"policies/policy_{orphan:05d}.pt").is_file())

    def test_active_layout_and_deferred_latest_retirement(self):
        model = build_actor_critic(**MODEL_KW)
        norm = RunningNorm(12, "cpu")
        pool = ActiveLeaguePool(MODEL_KW, "cpu", active_cap=24, sample=False, seed=7)
        first = pool.add(model, norm, role="latest", created_iteration=0)
        assignment = torch.full((8,), first, dtype=torch.long)
        done = torch.zeros(8, dtype=torch.bool)
        second = pool.replace_latest(model, norm, 20)
        self.assertNotEqual(first, second)
        self.assertEqual(pool.size(), 1)
        self.assertEqual(pool.resident_size(), 2)
        actions = pool.act(torch.zeros(8, 12), assignment, done.float())
        self.assertEqual(tuple(actions.shape), (8, 4))
        pool.refresh_residents(assignment, done)
        self.assertEqual(pool.resident_size(), 2)
        pool.refresh_residents(torch.full((8,), second), torch.ones(8, dtype=torch.bool))
        self.assertEqual(pool.resident_size(), 1)

    def test_full_quota_never_exceeds_24(self):
        model = build_actor_critic(**MODEL_KW)
        norm = RunningNorm(12, "cpu")
        pool = ActiveLeaguePool(MODEL_KW, "cpu", active_cap=24, sample=False, seed=8)
        pool.add(model, norm, role="latest")
        for i in range(ACTIVE_LAYOUT["recent"]):
            pool.add(model, norm, role="recent", archive_id=100+i, created_iteration=i)
        for i in range(ACTIVE_LAYOUT["core"]):
            pool.add(model, norm, role="core", archive_id=200+i)
        for i in range(ACTIVE_LAYOUT["challenger"]):
            pool.add(model, norm, role="challenger", archive_id=300+i)
        self.assertEqual(pool.size(), 24)
        self.assertEqual(pool.role_counts(), ACTIVE_LAYOUT)
        self.assertAlmostEqual(float(pool.weights().sum()), 1., places=6)

    def test_recent_ring_evicts_strictly_oldest(self):
        model = build_actor_critic(**MODEL_KW)
        norm = RunningNorm(12, "cpu")
        pool = ActiveLeaguePool(MODEL_KW, "cpu", active_cap=24, sample=False, seed=9)
        pool.add(model, norm, role="latest")
        for iteration in (100, 200, 300, 400):
            pool.add_recent_bundle(model, norm, iteration, iteration)
        oldest = min((e for e in pool.active_entries() if e["role"] == "recent"),
                     key=lambda e: e["created_iteration"])
        oldest["ema"] = 0.0  # Difficulty must not protect a temporal snapshot.
        pool.add_recent_bundle(model, norm, 500, 500)
        active_iterations = sorted(e["created_iteration"] for e in pool.active_entries()
                                   if e["role"] == "recent")
        self.assertEqual(active_iterations, [200, 300, 400, 500])

    def test_league_exploiter_warmup_without_core_is_finite(self):
        """The first LE can run before any core/challenger has been selected."""
        model = build_actor_critic(**MODEL_KW)
        norm = RunningNorm(12, "cpu")
        pool = ActiveLeaguePool(MODEL_KW, "cpu", active_cap=24, sample=True, seed=11)
        pool.add(model, norm, role="latest", created_iteration=20)
        pool.add(model, norm, role="recent", archive_id=100, created_iteration=0)

        weights = pool.league_exploiter_weights()
        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertTrue(bool((weights > 0).all()))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        sampled = torch.multinomial(weights, 4096, replacement=True)
        self.assertEqual(tuple(sampled.shape), (4096,))

    def test_league_exploiter_rejects_nonfinite_metadata_before_sampling(self):
        model = build_actor_critic(**MODEL_KW)
        norm = RunningNorm(12, "cpu")
        pool = ActiveLeaguePool(MODEL_KW, "cpu", active_cap=24, sample=False, seed=12)
        pool.add(model, norm, role="latest")
        pool.entries[0]["nash_mass"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pool.league_exploiter_weights()

    def test_trainer_checkpoint_restores_archive_and_active_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp", hidden=(8, 8),
                               num_bins=3, gru_size=0, normalize_obs=False,
                               rollout_steps=2, sched_period=0, milestone_period=0,
                               exploiter_iters=0, league_enabled=True,
                               league_dir=str(Path(temp) / "league"), league_active_cap=24,
                               aux_pred=False, seed=19)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            trainer._evaluate_pair = lambda *args, **kwargs: {
                "score": .5, "games": 8, "wins": 0, "draws": 8, "losses": 0,
                "lcb95": .5, "ucb95": .5, "paired_blocks": 4,
                "alt_loss_rate": .125, "opponent_alt_loss_rate": .25,
                "wins_by_opponent_altitude": 0,
                "wins_by_opponent_altitude_rate": 0.,
                "wins_by_opponent_altitude_share": 0.,
                "losses_by_own_altitude": 0,
                "losses_by_own_altitude_rate": 0.,
                "altitude_diagnostic_protocol":
                    "mirrored_terminal_cause_observability_v2"}
            trainer.train(start_iteration=1)
            trainer._heldout_audit_cursor = 7
            checkpoint = Path(temp) / "checkpoint.pt"
            trainer.save(checkpoint)
            saved_rng = trainer.pool.entries[0]["rng"].get_state().clone()
            restored = PPOGPUTrainer(ToyEnv(8, episode_steps=2), cfg)
            content = restored.load(checkpoint)
            self.assertEqual(content["training_protocol"], TRAINING_PROTOCOL)
            self.assertEqual(restored.pool.role_counts()["latest"], 1)
            self.assertEqual(restored._heldout_audit_cursor, 7)
            self.assertEqual(len(restored.archive), 0)
            torch.testing.assert_close(restored.pool.entries[0]["rng"].get_state(), saved_rng,
                                       rtol=0, atol=0)

    def test_evaluator_isolated_runner_restores_all_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, aux_pred=False, seed=37)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            samples = []
            before = trainer._runtime_state()

            def fake_runner(_left, _right, _minimum_games, _seed_block):
                samples.append((float(trainer.env.rng.random()),
                                 float(np.random.random()), float(torch.rand(()))))
                return {"score": 0.5, "games": 8, "wins": 0, "losses": 0,
                        "draws": 8, "lcb95": .4, "ucb95": .6,
                        "paired_blocks": 4}

            trainer._run_clean_paired_evaluation = fake_runner
            bundle = trainer._policy_bundle()
            trainer._evaluate_pair(bundle, bundle, 1, paired=True, seed_block=4)
            self.assertEqual(len(samples), 1)
            after = trainer._runtime_state()
            torch.testing.assert_close(after["torch_rng"], before["torch_rng"], rtol=0, atol=0)
            self.assertEqual(after["env_rng"], before["env_rng"])
            for name in before["trainer"]:
                torch.testing.assert_close(after["trainer"][name], before["trainer"][name],
                                           rtol=0, atol=0)
            source = inspect.getsource(PPOGPUTrainer._run_clean_paired_evaluation)
            self.assertNotIn("collect_rollout", source)
            self.assertNotIn("self.update", source)

    def test_chronological_recent_does_not_call_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, league_latest_period=1000,
                               league_recent_period=100, aux_pred=False, seed=47)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            trainer._action_kl_to_entry = lambda _entry: 0.0
            trainer._evaluate_pair = lambda *args, **kwargs: self.fail(
                "chronological recent must not invoke policy evaluation")
            self.assertEqual(trainer._maybe_update_latest_and_recent(100), "recent")
            recent = [entry for entry in trainer.pool.active_entries()
                      if entry["role"] == "recent"]
            self.assertEqual(len(recent), 1)
            record = trainer.archive.records[recent[0]["archive_id"]]
            self.assertEqual(record["metrics"]["selection"],
                             "chronological_periodic_fifo_v1")

    def test_runtime_restore_rewinds_each_opponent_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, aux_pred=False, seed=29)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            runtime = trainer._runtime_state()
            before = trainer.pool.entries[0]["rng"].get_state().clone()
            torch.rand(37, generator=trainer.pool.entries[0]["rng"])
            self.assertFalse(torch.equal(trainer.pool.entries[0]["rng"].get_state(), before))
            trainer._restore_runtime(runtime)
            torch.testing.assert_close(trainer.pool.entries[0]["rng"].get_state(), before,
                                       rtol=0, atol=0)

    def test_milestone_exploiter_uses_independent_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0, normalize_obs=False,
                               rollout_steps=1, update_epochs=1, num_minibatches=1,
                               sched_period=0, milestone_period=1, exploiter_iters=1,
                               exploiter_win_target=.75, league_enabled=True,
                               league_dir=str(Path(temp) / "league"), league_active_cap=24,
                               league_payoff_games=1, league_admission_games=1,
                               aux_pred=False, seed=23,
                               opp_sample=False)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            trainer._evaluate_pair = lambda *args, **kwargs: {
                "score": .5, "games": 8, "wins": 0, "draws": 8, "losses": 0,
                "lcb95": .5, "ucb95": .5, "paired_blocks": 4,
                "alt_loss_rate": .125, "opponent_alt_loss_rate": .25,
                "wins_by_opponent_altitude": 0,
                "wins_by_opponent_altitude_rate": 0.,
                "wins_by_opponent_altitude_share": 0.,
                "losses_by_own_altitude": 0,
                "losses_by_own_altitude_rate": 0.,
                "altitude_diagnostic_protocol":
                    "mirrored_terminal_cause_observability_v2"}
            trainer.train(start_iteration=1)
            self.assertEqual(len(trainer.exploiter_history), 1)
            result = trainer.exploiter_history[0]
            self.assertIn("accepted", result)
            self.assertIn("target_lcb95", result["admission"])
            self.assertEqual(result["diagnostic_protocol"],
                             "opponent_altitude_outcome_observability_v2")
            self.assertTrue(any(r["kind"] == "milestone_main"
                                for r in trainer.archive.records.values()))
            self.assertLessEqual(trainer.pool.size(), 24)

    def test_exploiter_admission_requires_point_and_confidence(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1,
                               update_epochs=1, num_minibatches=1,
                               sched_period=0, milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, league_payoff_games=4,
                               league_admission_games=4, aux_pred=False, seed=29)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            candidate = trainer._policy_bundle()
            weak_confidence = {"score": .70, "games": 512, "wins": 358,
                               "draws": 0, "losses": 154, "lcb95": .59,
                               "ucb95": .74, "paired_blocks": 256,
                               "alt_loss_rate": .10,
                               "opponent_alt_loss_rate": .20,
                               "wins_by_opponent_altitude": 80,
                               "wins_by_opponent_altitude_rate": 80 / 512,
                               "wins_by_opponent_altitude_share": 80 / 358,
                               "losses_by_own_altitude": 40,
                               "losses_by_own_altitude_rate": 40 / 512,
                               "altitude_diagnostic_protocol":
                                   "mirrored_terminal_cause_observability_v2"}
            trainer._evaluate_pair = lambda *args, **kwargs: dict(weak_confidence)
            accepted, _, metrics = trainer._evaluate_and_admit_exploiter(
                candidate, candidate, "ME-EIE", "standard", .75)
            self.assertFalse(accepted)
            self.assertFalse(metrics["target_confident"])
            strong_confidence = dict(weak_confidence, lcb95=.61)
            trainer._evaluate_pair = lambda *args, **kwargs: dict(strong_confidence)
            accepted, _, metrics = trainer._evaluate_and_admit_exploiter(
                candidate, candidate, "ME-EIE", "standard", .75)
            self.assertTrue(accepted)
            self.assertTrue(metrics["target_confident"])
            self.assertEqual(metrics["admission_rule"],
                             "point_lcb_or_similar_successor_v3")
            self.assertEqual(metrics["wins_by_target_main_altitude"], 80)
            self.assertAlmostEqual(metrics["target_main_alt_loss_rate"], .20)

    def test_failed_normal_admission_can_become_similar_successor(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, league_payoff_games=4,
                               league_admission_games=4, aux_pred=False, seed=31)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            incumbent = trainer._archive_current(
                "exploiter_me-ere", profile="altitude_hunt",
                admitted=True, payoff_eligible=True, metrics={"role": "LE"})
            trainer.pool.add_archive_policy(
                trainer.archive, incumbent, "challenger")
            trainer._candidate_novelty = lambda _scores: .03
            candidate_result = {
                "score": .46, "games": 512, "wins": 230, "draws": 11, "losses": 271,
                "lcb95": .40, "ucb95": .52, "paired_blocks": 256,
                "alt_loss_rate": .08, "opponent_alt_loss_rate": .17,
                "wins_by_opponent_altitude": 40,
                "wins_by_opponent_altitude_rate": 40 / 512,
                "wins_by_opponent_altitude_share": 40 / 230,
                "losses_by_own_altitude": 32,
                "losses_by_own_altitude_rate": 32 / 512,
                "altitude_diagnostic_protocol": "mirrored_terminal_cause_observability_v2"}
            incumbent_result = dict(
                candidate_result, score=.08, wins=35, draws=12, losses=465,
                lcb95=.04, ucb95=.14,
                wins_by_opponent_altitude=3,
                wins_by_opponent_altitude_rate=3 / 512,
                wins_by_opponent_altitude_share=3 / 35)
            anchor_result = dict(candidate_result, score=.50, wins=245, draws=22,
                                 losses=245, lcb95=.43, ucb95=.57)
            results = iter((candidate_result, anchor_result, incumbent_result))
            trainer._evaluate_pair = lambda *args, **kwargs: dict(next(results))
            trainer._refresh_payoff_row = lambda *args, **kwargs: None
            bundle = trainer._policy_bundle()
            accepted, identity, metrics = trainer._evaluate_and_admit_exploiter(
                bundle, bundle, "LE", "altitude_hunt", .75)
            self.assertTrue(accepted)
            self.assertTrue(metrics["similar_successor"])
            self.assertEqual(metrics["successor_of"], incumbent)
            self.assertEqual(metrics["successor_family"],
                             {"role": "LE", "profile": "altitude_hunt"})
            self.assertEqual(metrics["admission_route"], "similar_successor")
            self.assertTrue(trainer.archive.records[identity]["kind"].startswith(
                "exploiter_successor_"))
            self.assertEqual(trainer.archive.records[identity]["admission_status"],
                             "probationary")
            self.assertTrue(trainer.archive.records[incumbent]["admitted"])
            trainer._last_milestone_archive_id = None
            trainer._refresh_active_core()
            self.assertTrue(any(entry.get("archive_id") == identity
                                and entry.get("role") == "challenger"
                                for entry in trainer.pool.active_entries()))

    def test_similar_successor_requires_nonoverlapping_confidence(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, league_payoff_games=4,
                               league_admission_games=4, aux_pred=False, seed=33)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            incumbent = trainer._archive_current(
                "exploiter_me-ere", profile="altitude_hunt",
                admitted=True, payoff_eligible=True, metrics={"role": "LE"})
            trainer.pool.add_archive_policy(
                trainer.archive, incumbent, "challenger")
            trainer._candidate_novelty = lambda _scores: .03
            base = {
                "games": 512, "draws": 0, "paired_blocks": 256,
                "alt_loss_rate": .0, "opponent_alt_loss_rate": .0,
                "wins_by_opponent_altitude": 0,
                "wins_by_opponent_altitude_rate": 0.,
                "wins_by_opponent_altitude_share": 0.,
                "losses_by_own_altitude": 0, "losses_by_own_altitude_rate": 0.,
                "altitude_diagnostic_protocol": "mirrored_terminal_cause_observability_v2"}
            candidate = dict(base, score=.46, wins=236, losses=276, lcb95=.40, ucb95=.52)
            incumbent_eval = dict(base, score=.30, wins=154, losses=358,
                                  lcb95=.24, ucb95=.45)
            anchor_result = dict(candidate, score=.50, wins=256, losses=256,
                                 lcb95=.43, ucb95=.57)
            results = iter((candidate, anchor_result, incumbent_eval))
            trainer._evaluate_pair = lambda *args, **kwargs: dict(next(results))
            bundle = trainer._policy_bundle()
            accepted, identity, metrics = trainer._evaluate_and_admit_exploiter(
                bundle, bundle, "LE", "altitude_hunt", .75)
            self.assertFalse(accepted)
            self.assertIsNone(identity)
            self.assertFalse(metrics["similar_successor"])

    def test_successor_family_is_exact_role_x_profile_and_incumbent_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, aux_pred=False, seed=35)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            identity = trainer._archive_current(
                "exploiter_me-ere", profile="altitude_hunt",
                admitted=True, payoff_eligible=True, metrics={"role": "LE"})
            original = copy.deepcopy(trainer.archive.records[identity])
            trainer.pool.add_archive_policy(trainer.archive, identity, "core")
            self.assertEqual(
                trainer._active_family_incumbent("LE", "altitude_hunt"), identity)
            self.assertIsNone(
                trainer._active_family_incumbent("ME-EIE", "altitude_hunt"))
            self.assertIsNone(trainer._active_family_incumbent("LE", "attack"))
            self.assertEqual(trainer.archive.records[identity], original)

    def test_first_le_sees_previous_milestone_core_before_training(self):
        """Milestone payoff/core refresh must precede the LE side learner."""
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=2, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1,
                               update_epochs=1, num_minibatches=1,
                               sched_period=0, milestone_period=1, exploiter_iters=1,
                               league_enabled=True,
                               league_dir=str(Path(temp) / "league"), league_active_cap=24,
                               league_payoff_games=1, league_admission_games=1,
                               aux_pred=False, seed=31,
                               opp_sample=False)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            sightings = []
            trainer._evaluate_pair = lambda *args, **kwargs: {
                "score": .5, "games": 8, "wins": 0, "draws": 8, "losses": 0,
                "lcb95": .5, "ucb95": .5, "paired_blocks": 4}

            def observe_side_learner(log=None, metric_cb=None):
                core = [entry.get("archive_id") for entry in trainer.pool.active_entries()
                        if entry.get("role") == "core"]
                sightings.append({"iteration": trainer.iteration,
                                  "role": trainer._exploiter_role(),
                                  "core": core,
                                  "current": trainer._last_milestone_archive_id})
                return 0.5

            trainer.train_exploiter = observe_side_learner
            trainer.train(start_iteration=1)

            # 2026-09-02: LE was dropped from the scheduled rotation (see
            # EXPLOITER_ACTIVE_ROLES in ppo_gpu.py) -- the second milestone
            # now schedules ME-ERE instead of LE.
            self.assertEqual([item["role"] for item in sightings], ["ME-EIE", "ME-ERE"])
            self.assertEqual(sightings[0]["core"], [])
            self.assertEqual(len(sightings[1]["core"]), 1)
            prior = sightings[1]["core"][0]
            self.assertNotEqual(prior, sightings[1]["current"])
            self.assertEqual(trainer.archive.records[prior]["kind"], "milestone_main")
            self.assertEqual(trainer.archive.records[prior]["iteration"], 1)

    def test_payoff_refresh_period_is_independent_of_milestone(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=2, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1,
                               update_epochs=1, num_minibatches=1,
                               sched_period=0, milestone_period=2, exploiter_iters=0,
                               league_enabled=True,
                               league_dir=str(Path(temp) / "league"), league_active_cap=24,
                               league_payoff_games=1, league_admission_games=1,
                               league_payoff_refresh_period=1, league_redteam_period=99,
                               aux_pred=False, seed=41, opp_sample=False)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            sightings = []
            trainer._refresh_uncertain_payoff_edge = (
                lambda *, seed_block: sightings.append((trainer.iteration, seed_block)))
            trainer.train(start_iteration=1)
            self.assertEqual([iteration for iteration, _ in sightings], [1, 2])

    def test_heldout_audit_rotates_and_requires_confident_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True,
                               league_dir=str(Path(temp) / "league"), league_active_cap=24,
                               league_payoff_games=1, league_admission_games=1,
                               league_redteam_period=1, aux_pred=False, seed=43)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            identities = [trainer._archive_current(
                "heldout_me-eie", admitted=False, payoff_eligible=True)
                for _ in range(10)]
            result = {"score": .35, "games": 16, "wins": 5, "draws": 1,
                      "losses": 10, "lcb95": .29, "ucb95": .71,
                      "paired_blocks": 8}
            trainer._evaluate_pair = lambda *args, **kwargs: dict(result)
            self.assertEqual(trainer._audit_heldout_redteam(1), [])
            trainer._audit_heldout_redteam(2)
            self.assertIn("last_audit_iteration",
                          trainer.archive.records[identities[-1]]["metrics"])
            # Point score alone is insufficient while the current-main UCB
            # still crosses .70 (approved common recovery gate).
            self.assertFalse(trainer.archive.records[identities[0]]["admitted"])
            result["ucb95"] = .70
            trainer._heldout_audit_cursor = 0
            promoted = trainer._audit_heldout_redteam(3)
            self.assertIn(identities[0], promoted)

    def test_failed_evaluation_commits_no_candidate_or_payoff(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, league_payoff_games=1,
                               league_admission_games=1, aux_pred=False, seed=53)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            before = copy.deepcopy(trainer.archive.state_dict())
            bandit = copy.deepcopy(trainer._role_profile_bandit)
            trainer._evaluate_pair = lambda *args, **kwargs: (_ for _ in ()).throw(
                FloatingPointError("synthetic evaluator NaN"))
            bundle = trainer._policy_bundle()
            with self.assertRaisesRegex(FloatingPointError, "synthetic evaluator NaN"):
                trainer._evaluate_and_admit_exploiter(
                    bundle, bundle, "ME-EIE", "standard", .5)
            self.assertEqual(trainer.archive.state_dict(), before)
            self.assertEqual(trainer._role_profile_bandit, bandit)

    def test_old_evaluator_contract_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = PPOGPUConfig(total_iterations=1, device="cpu", architecture="mlp",
                               hidden=(8, 8), num_bins=3, gru_size=0,
                               normalize_obs=False, rollout_steps=1, sched_period=0,
                               milestone_period=0, exploiter_iters=0,
                               league_enabled=True, league_dir=str(Path(temp) / "league"),
                               league_active_cap=24, aux_pred=False, seed=59)
            trainer = PPOGPUTrainer(StatefulToy(8), cfg)
            checkpoint = Path(temp) / "checkpoint.pt"
            trainer.save(checkpoint)
            content = torch.load(checkpoint, map_location="cpu", weights_only=False)
            content["league_contract"]["evaluator"] = "legacy_random_stagger"
            torch.save(content, checkpoint)
            restored = PPOGPUTrainer(StatefulToy(8), cfg)
            with self.assertRaisesRegex(ValueError, "active-league contract differs"):
                restored.load(checkpoint)



if __name__ == "__main__":
    unittest.main()
