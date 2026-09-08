"""Tiny CPU-only regression checks for the pre-update trajectory integrity gate.

Run: python -B -m cuda_fdm.tests.rollout_finite_val
No CUDA environment, checkpoints, or experiment jobs are used.
"""
import math
import unittest
from unittest import mock

import torch

from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.tests.pool_identity_audit import ToyEnv, reproduce_nonfinite_scoring


class FiniteFlagToyEnv(ToyEnv):
    """Every step autoresets; invalid raw physics is represented only by its flag."""

    def __init__(self, invalid_step=None):
        super().__init__(nenv=4, episode_steps=1)
        self.invalid_step = invalid_step
        self.calls = 0
        self.returned_numeric_finite = []
        self.raw_flags = []

    def step(self, controls):
        obs, reward, done, info = super().step(controls)
        self.calls += 1
        if self.calls == self.invalid_step:
            # A single invalid lane must reject the complete PPO rollout.
            info["terminal_state_finite"][0] = False
        self.returned_numeric_finite.append(all(bool(torch.isfinite(x).all()) for x in
            (obs, reward, info["terminal_obs"], info["terminal_hp"], info["terminal_alt_m"])))
        self.raw_flags.append(bool(info["terminal_state_finite"].all()))
        assert bool(done.all()) and bool((obs == 0).all()), "fixture must emulate autoreset"
        return obs, reward, done, info


def make_trainer(architecture, invalid_step=None):
    env = FiniteFlagToyEnv(invalid_step)
    cfg = PPOGPUConfig(device="cpu", architecture=architecture,
                       hidden=(8, 8, 8), gru_size=4 if architecture == "gru" else 0,
                       num_bins=3, normalize_obs=False, opp_sample=False,
                       total_iterations=1, rollout_steps=3, update_epochs=1,
                       num_minibatches=1, recurrent_seq_len=3,
                       selfplay_gate_threshold=2., sched_period=0,
                       milestone_period=0, exploiter_iters=0)
    return PPOGPUTrainer(env, cfg)


class RolloutFiniteTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_raw_flag_rejects_before_update_despite_finite_autoreset(self):
        for architecture in ("mlp", "gru"):
            for invalid_step in (1, 3):
                with self.subTest(architecture=architecture, invalid_step=invalid_step):
                    tr = make_trainer(architecture, invalid_step)
                    before = {k: v.clone() for k, v in tr.model.state_dict().items()}
                    with mock.patch.object(tr, "update", wraps=tr.update) as update:
                        with self.assertRaisesRegex(FloatingPointError, "update refused"):
                            tr.train()
                    update.assert_not_called()
                    self.assertEqual(len(tr.actor_opt.state), 0)
                    self.assertEqual(len(tr.critic_opt.state), 0)
                    self.assertTrue(all(torch.equal(v, tr.model.state_dict()[k]) for k, v in before.items()))
                    self.assertEqual(tr.env.calls, 3)
                    self.assertTrue(all(tr.env.returned_numeric_finite))
                    self.assertFalse(tr.env.raw_flags[invalid_step - 1])
                    if invalid_step == 1:
                        self.assertTrue(tr.env.raw_flags[-1], "early failure must survive later finite steps")
                    for buffer in (tr.b_obs, tr.b_rew, tr.b_val, tr.b_logp):
                        self.assertTrue(bool(torch.isfinite(buffer).all()))

    def test_finite_toy_allows_normal_rollout_and_one_cpu_update(self):
        for architecture in ("mlp",):
            with self.subTest(architecture=architecture):
                tr = make_trainer(architecture)
                with mock.patch.object(tr, "update", wraps=tr.update) as update:
                    history = tr.train()
                update.assert_called_once()
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0].completed_episodes, 12.)
                self.assertTrue(math.isfinite(history[0].policy_loss))
                self.assertTrue(math.isfinite(history[0].value_loss))
                self.assertTrue(all(tr.env.raw_flags))
                self.assertTrue(all(tr.env.returned_numeric_finite))
                self.assertGreater(len(tr.actor_opt.state), 0)
                self.assertGreater(len(tr.critic_opt.state), 0)

    def test_nonfinite_draw_fixture_is_explicitly_legacy_scoring_only(self):
        cases = reproduce_nonfinite_scoring()
        for case in cases.values():
            self.assertEqual(case["semantics"], "legacy scoring arithmetic without the new raw-state gate")
            self.assertFalse(case["raw_terminal_state_finite"])
            self.assertEqual(case["score"], .5)
            self.assertTrue(case["legacy_transformed_data_check_accepts"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
