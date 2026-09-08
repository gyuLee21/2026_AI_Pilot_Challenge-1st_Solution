"""Finite-horizon reward/GAE/clock regressions on CPU; no optimization or CUDA."""
import re
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from cuda_fdm import obs_reward as OR
from cuda_fdm.ppo_gpu import (LEGACY_TRAINING_PROTOCOL, PPOGPUConfig, PPOGPUTrainer,
                              TRAINING_PROTOCOL)
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.tests.pool_identity_audit import toy_trainer
from cuda_fdm.tests.time_limit_value_audit import BoundaryEnv, collect_case, reward_and_clock
from dogfight.envs.termination import (
    TIME_LIMIT_TOLERANCE_SEC, evaluate_termination, time_limit_reached,
)
from dogfight.sim.state_schema import StateIndex


ROOT = Path(__file__).resolve().parents[2]


def termination_fixture():
    env = GpuDogfightVecEnv.__new__(GpuDogfightVecEnv)
    env.nenv = 1
    env.min_altitude_m = 304.8
    env.max_engage_time_s = 200.0
    env.obr = SimpleNamespace(hp=torch.ones(2), t_sec=torch.zeros(1, dtype=torch.float64))
    s9 = torch.zeros(2, 9, dtype=torch.float64)
    s9[:, 2] = -3000.
    return env, s9


class TimeoutTerminalTests(unittest.TestCase):
    def test_mlp_and_gru_timeout_targets_are_raw_and_live_cutoff_still_bootstraps(self):
        for architecture in ("mlp", "gru"):
            with self.subTest(architecture=architecture):
                case = collect_case(architecture)
                np.testing.assert_allclose(case["first_step_gae_targets"],
                                           [5., -5., -4., 5., 10.97], atol=2e-6)
                np.testing.assert_allclose(case["timeout_target_error"], 0., atol=2e-6)

    def test_later_episode_rewards_and_values_cannot_enter_terminal_target(self):
        for architecture in ("mlp", "gru"):
            baseline = collect_case(architecture, rollout_steps=2, next_reward=0., fresh_value=100.)
            perturbed = collect_case(architecture, rollout_steps=2, next_reward=500., fresh_value=999.)
            np.testing.assert_allclose(baseline["first_step_gae_targets"][:4],
                                       [5., -5., -4., 5.], atol=2e-6)
            np.testing.assert_allclose(baseline["first_step_gae_targets"][:4],
                                       perturbed["first_step_gae_targets"][:4], atol=2e-6)
            self.assertNotEqual(baseline["first_step_gae_targets"][4],
                                perturbed["first_step_gae_targets"][4])

    def test_real_normalizer_and_network_keep_terminal_targets_and_no_extra_value_call(self):
        for architecture in ("mlp", "gru"):
            cfg = PPOGPUConfig(device="cpu", architecture=architecture, hidden=(8, 8, 8),
                               gru_size=4 if architecture == "gru" else 0,
                               num_bins=3, normalize_obs=True, rollout_steps=3,
                               sched_period=0, milestone_period=0, exploiter_iters=0)
            trainer = PPOGPUTrainer(BoundaryEnv(), cfg)
            calls = []
            original_value = trainer.model.value_step
            def count_value(self, obs, state=None, episode_start=None):
                calls.append(obs.shape)
                return original_value(obs, state=state, episode_start=episode_start)
            trainer.model.value_step = MethodType(count_value, trainer.model)
            _, ret, _ = trainer.collect_rollout()
            np.testing.assert_allclose(ret[0, :4].tolist(), [5., -5., -4., 5.], atol=2e-6)
            self.assertEqual(len(calls), cfg.rollout_steps + 1)

    def test_cpu_and_cuda_reference_inclusive_clock_and_tolerance_match(self):
        report = reward_and_clock()
        self.assertEqual(report["current_inclusive_horizon"]["steps"], 2000)
        self.assertAlmostEqual(report["current_inclusive_horizon"]["time"], 200.0, places=8)
        self.assertLessEqual(report["geometry_theoretical_abs_bound_at_current_duration"], 5. + 1e-10)
        env, s9 = termination_fixture()
        for timestamp, expected in ((199.9, False), (200. - 2e-8, False),
                                    (200. - 5e-9, True), (200., True), (200.1, True)):
            env.obr.t_sec.fill_(timestamp)
            term, trunc = env._termination(s9)
            self.assertFalse(bool(term[0]))
            self.assertEqual(bool(trunc[0]), expected)
            self.assertEqual(bool(time_limit_reached(timestamp, 200.)), expected)
        kernel = (ROOT / "cuda_fdm/gen/obs_kernel.cu").read_text(encoding="utf-8")
        match = re.search(r"#define TIME_LIMIT_TOLERANCE_SEC\s+([0-9.eE+-]+)", kernel)
        self.assertIsNotNone(match)
        self.assertEqual(float(match.group(1)), TIME_LIMIT_TOLERANCE_SEC)
        self.assertIn("(t_new >= max_time - TIME_LIMIT_TOLERANCE_SEC)", kernel)

    def test_destruction_and_low_altitude_still_precede_timeout(self):
        env, s9 = termination_fixture()
        own = np.zeros(46)
        enemy = own.copy()
        for state in (own, enemy):
            state[StateIndex.HEALTH] = state[StateIndex.FUEL] = 1.
            state[StateIndex.ALT] = 3000.
            state[StateIndex.SIM_TIME] = 200.
        sim = SimpleNamespace(fdm_update_success=True)
        for reason in ("altitude", "health"):
            env.obr.t_sec.fill_(200.)
            env.obr.hp.fill_(1.)
            s9[:, 2] = -3000.
            own[StateIndex.ALT] = 3000.
            own[StateIndex.HEALTH] = 1.
            if reason == "altitude":
                s9[0, 2] = -100.
                own[StateIndex.ALT] = 100.
            else:
                env.obr.hp[0] = 0.
                own[StateIndex.HEALTH] = 0.
            term, trunc = env._termination(s9)
            self.assertTrue(bool(term[0]))
            self.assertFalse(bool(trunc[0]))
            cpu = evaluate_termination(own, enemy, sim, sim, 200., 300., 2000, None)
            self.assertTrue(cpu[0])
            self.assertFalse(cpu[1])
            self.assertNotEqual(cpu[2], "max time out")

    def test_timeout_reward_damage_and_potential_reset_are_preserved(self):
        report = reward_and_clock()
        np.testing.assert_allclose(report["torch_reference_rewards_both_sides"],
                                   [[5., -5.], [-5., 5.], [-4., -4.]])
        np.testing.assert_allclose(report["last_step_damage_plus_timeout_reward"], [5.2, -5.2])
        s9 = torch.zeros(2, 9, dtype=torch.float64)
        s9[:, 2], s9[:, 6] = -3000., 250.
        s9[1, 0] = 1000.
        used = OR.BatchObsReward(1, device="cpu", enable_kernel=False)
        fresh = OR.BatchObsReward(1, device="cpu", enable_kernel=False)
        used.prev_x.copy_(torch.tensor([.9, -.9]))
        used.prev_x_valid.fill_(True)
        used.prev_alt_log.fill_(-7.)
        used.t_sec.fill_(200.)
        used.reset_envs(torch.ones(1, dtype=torch.bool))
        used.initialize_reward_state(s9)
        fresh.initialize_reward_state(s9)
        mask = torch.zeros(1, dtype=torch.bool)
        torch.testing.assert_close(used.compute_reward(s9, mask), fresh.compute_reward(s9, mask),
                                   rtol=0, atol=0)

    def test_previous_fixed_pool_protocol_cannot_resume_as_corrected_targets(self):
        trainer = toy_trainer(cap=4)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "current.pt"
            trainer.save(checkpoint)
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(data["training_protocol"], LEGACY_TRAINING_PROTOCOL)
            self.assertEqual(LEGACY_TRAINING_PROTOCOL,
                             "cuda_mlp_flat_finite_horizon_diverse_h3_reset_v7")
            self.assertNotEqual(TRAINING_PROTOCOL, LEGACY_TRAINING_PROTOCOL)
            data["training_protocol"] = "cuda_episode_opponent_v2"
            old = Path(tmp) / "old_target_rules.pt"
            torch.save(data, old)
            before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "finite-horizon"):
                trainer.load(old)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            trainer.load(checkpoint)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
