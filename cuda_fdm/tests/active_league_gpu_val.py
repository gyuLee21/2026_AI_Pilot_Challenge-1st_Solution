from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv


def assert_nested_equal(test, left, right, path="root"):
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            detail = f"{path}: shape {tuple(left.shape)} -> {tuple(right.shape)}"
            if left.shape == right.shape and left.numel() and left.dtype != torch.bool:
                detail += f", max_abs={float((left-right).abs().max())}, left={left}, right={right}"
            test.fail(detail)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_equal(test, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested_equal(test, a, b, f"{path}[{index}]")
    else:
        test.assertEqual(left, right, path)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ActiveLeagueGPUContractTest(unittest.TestCase):
    def make_trainer(self, root):
        env = GpuDogfightVecEnv(32, seed=71)
        cfg = PPOGPUConfig(
            total_iterations=1, device="cuda", architecture="mlp",
            hidden=(384, 384, 384), num_bins=21, normalize_obs=True,
            rollout_steps=2, sched_period=0, milestone_period=0,
            exploiter_iters=0, league_enabled=True,
            league_dir=str(Path(root) / "league"), league_active_cap=24,
            league_payoff_games=64, league_admission_games=256,
            aux_pred=True, aux_coef=.1, seed=71, opp_sample=True)
        return PPOGPUTrainer(env, cfg)

    def test_paired_bank_and_step_only_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = self.make_trainer(temp)
            evaluator, per_order = trainer._league_eval_env(64)
            self.assertEqual((evaluator.nenv, per_order), (128, 64))
            evaluator.reset_evaluation(paired=True, seed_block=9)
            states = evaluator.sim.states.view(evaluator.nenv, 2, -1)
            torch.testing.assert_close(states[:per_order], states[per_order:], rtol=0, atol=0)
            meta = evaluator._evaluation_seed_metadata
            self.assertEqual((meta["three_nine"], meta["headon"]), (48, 16))
            self.assertEqual(meta["three_nine_distance_counts"],
                             {"2000": 16, "2500": 16, "3000": 16})
            reconstruction = evaluator.obr.clone_state()
            self.assertTrue(bool((reconstruction["t_sec"] == 0).all()))
            self.assertTrue(bool((reconstruction["act_hist"] == 0).all()))
            for name, value in reconstruction.items():
                if value.shape[0] == evaluator.nenv:
                    torch.testing.assert_close(value[:per_order], value[per_order:], rtol=0, atol=0)
                elif value.shape[0] == evaluator.nac:
                    paired = value.view(evaluator.nenv, 2, *value.shape[1:])
                    torch.testing.assert_close(paired[:per_order], paired[per_order:], rtol=0, atol=0)

            # Admission contract is 256 complete games in each ordering.
            admission_env, admission_per_order = trainer._league_eval_env(256)
            self.assertEqual((admission_env.nenv, admission_per_order), (512, 256))

            before_runtime = trainer._runtime_state()
            before_learner = trainer._snapshot_learner()
            before_archive = copy.deepcopy(trainer.archive.state_dict())
            bundle = trainer._policy_bundle()
            first = trainer._evaluate_pair(bundle, bundle, 4, paired=True, seed_block=11)
            second = trainer._evaluate_pair(bundle, bundle, 4, paired=True, seed_block=11)
            self.assertEqual(first, second)
            self.assertEqual((first["games"], first["paired_blocks"]), (8, 4))
            self.assertEqual((first["forward"]["games"], first["reverse"]["games"]), (4, 4))
            self.assertLessEqual(first["lcb95"], first["score"])
            self.assertGreaterEqual(first["ucb95"], first["score"])
            self.assertFalse(first["deterministic"])
            self.assertTrue(first["stochastic"])
            self.assertIsInstance(first["policy_rng_seed"], int)
            assert_nested_equal(self, before_runtime, trainer._runtime_state(), "runtime")
            assert_nested_equal(self, before_learner, trainer._snapshot_learner(), "learner")
            self.assertEqual(before_archive, trainer.archive.state_dict())

    def test_real_side_learner_restores_main_and_first_le_has_core(self):
        with tempfile.TemporaryDirectory() as temp:
            env = GpuDogfightVecEnv(32, seed=83)
            cfg = PPOGPUConfig(
                total_iterations=2, device="cuda", architecture="mlp",
                hidden=(32, 32, 32), num_bins=5, normalize_obs=True,
                rollout_steps=1, update_epochs=1, num_minibatches=1,
                sched_period=0, milestone_period=1, exploiter_iters=1,
                exploiter_win_target=.75, league_enabled=True,
                league_dir=str(Path(temp) / "league"), league_active_cap=24,
                league_payoff_games=4, league_admission_games=4,
                league_payoff_refresh_period=1, league_redteam_period=99,
                aux_pred=False, seed=83, opp_sample=False)
            trainer = PPOGPUTrainer(env, cfg)
            original = trainer.train_exploiter
            sightings = []

            def checked_side_learner(*args, **kwargs):
                role = trainer._exploiter_role()
                core = [entry.get("archive_id") for entry in trainer.pool.active_entries()
                        if entry.get("role") == "core"]
                # Mirror train_exploiter's pre-snapshot resident refresh.  A
                # newly replaced latest can legitimately add a retired resident
                # row while its old episodes finish; that is a league event, not
                # side-learner mutation.
                trainer._refresh_weights()
                before_runtime = trainer._runtime_state()
                before_learner = trainer._snapshot_learner()
                before_pool = copy.deepcopy(trainer.pool.state_dicts())
                result = original(*args, **kwargs)
                assert_nested_equal(self, before_runtime, trainer._runtime_state(),
                                    f"{role}.runtime")
                assert_nested_equal(self, before_learner, trainer._snapshot_learner(),
                                    f"{role}.learner")
                assert_nested_equal(self, before_pool, trainer.pool.state_dicts(),
                                    f"{role}.pool")
                sightings.append((role, core))
                return result

            trainer.train_exploiter = checked_side_learner
            trainer.train(start_iteration=1)
            # 2026-09-03: LE was dropped entirely from this package (see
            # EXPLOITER_ROLES in ppo_gpu.py) -- the second milestone
            # schedules ME-ERE.
            self.assertEqual([role for role, _ in sightings], ["ME-EIE", "ME-ERE"])
            self.assertEqual(sightings[0][1], [])
            self.assertEqual(len(sightings[1][1]), 1)
            self.assertEqual(trainer.iteration, 2)
            self.assertTrue(all(torch.isfinite(parameter).all()
                                for parameter in trainer.model.parameters()))
            checkpoint = Path(temp) / "fresh_milestone_smoke.pt"
            trainer.save(checkpoint)
            archive_before = copy.deepcopy(trainer.archive.state_dict())
            learner_before = trainer._snapshot_learner()
            restored_env = GpuDogfightVecEnv(32, seed=83)
            restored = PPOGPUTrainer(restored_env, cfg)
            loaded = restored.load(checkpoint)
            self.assertEqual(loaded["iteration"], 2)
            self.assertEqual(restored.archive.state_dict(), archive_before)
            assert_nested_equal(self, learner_before, restored._snapshot_learner(),
                                "fresh_milestone_save_reload")
            self.assertLessEqual(restored.pool.size(), 24)

    def test_policy_order_complement(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = self.make_trainer(temp)
            left = trainer._policy_bundle()
            right = copy.deepcopy(left)
            left_bias_name = [name for name in left["model"]
                              if name.startswith("actor_logits") and name.endswith(".bias")][-1]
            output_bias = [name for name in right["model"]
                           if name.startswith("actor_logits") and name.endswith(".bias")][-1]
            left_forced = left["model"][left_bias_name].view(4, trainer.cfg.num_bins)
            forced = right["model"][output_bias].view(4, trainer.cfg.num_bins)
            left_forced.fill_(-50.0)
            left_forced[:, -1] = 50.0
            forced.fill_(-50.0)
            forced[:, 0] = 50.0
            ab = trainer._evaluate_pair(left, right, 4, paired=True, seed_block=19)
            ba = trainer._evaluate_pair(right, left, 4, paired=True, seed_block=19)
            # This deliberately asymmetric policy pair prevents a vacuous
            # complement test where both incorrectly aggregated scores are 0.5.
            self.assertNotAlmostEqual(ab["score"], 0.5, places=4)
            self.assertAlmostEqual(ab["score"] + ba["score"], 1.0, places=12,
                                   msg=f"ab={ab} ba={ba}")
            self.assertEqual((ab["wins"], ab["draws"], ab["losses"]),
                             (ba["losses"], ba["draws"], ba["wins"]))
            self.assertAlmostEqual(ab["lcb95"], 1.0 - ba["ucb95"], places=12)
            self.assertAlmostEqual(ab["ucb95"], 1.0 - ba["lcb95"], places=12)

    def test_cuda_termination_uses_direct_msl_boundary(self):
        """An aircraft above 1000ft MSL stays alive despite tangent-D curvature."""
        from cuda_fdm.ic import build_seed_vector
        from cuda_fdm.rl_env import STATE_N

        env = GpuDogfightVecEnv(4, seed=97)
        env.reset(stagger=False)
        altitude_m = 305.5
        rows = np.zeros((env.nac, STATE_N), dtype=np.float64)
        for index in range(env.nac):
            north = 3957.2 + 1000.0 * (index % 2)
            east = 1000.0 * (index // 2)
            heading = 90.0 if index % 2 == 0 else 270.0
            spec = env._ic_dict(north, east, -altitude_m, heading, 250.0)
            rows[index] = build_seed_vector(**spec)
        env.sim.load_seed(rows)
        env.obr.reset_all()
        env.obr.kernel_init_reward_state(env.sim.states)
        state = env.state9_flat()
        torch.testing.assert_close(-state[:, 2],
                                   torch.full_like(state[:, 2], altitude_m),
                                   rtol=0, atol=2e-6)
        actions = torch.zeros(env.nac, 4, dtype=torch.float64, device="cuda")
        actions[:, 3] = .8
        _, terminated, truncated = env.obr.kernel_advance(
            env.sim.states, actions, min_alt=304.8, max_time=200.0)
        torch.cuda.synchronize()
        state_after = env.state9_flat()
        self.assertGreater(float((-state_after[:, 2]).min()), 304.8)
        self.assertFalse(bool(terminated.any()))
        self.assertFalse(bool(truncated.any()))


if __name__ == "__main__":
    unittest.main()
