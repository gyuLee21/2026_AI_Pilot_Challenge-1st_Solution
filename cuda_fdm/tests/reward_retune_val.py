"""Approved reward retune: independent altitude scale, parity and safe resume."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from cuda_fdm import obs_reward as OR
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.tests.main_schedule_val import fixture
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal
from cuda_fdm.tests.time_limit_value_audit import BoundaryEnv


def reward_cfg():
    return dict(OR.RW.MY_REWARD_CONFIG, damage_scale=2.,
                altitude_settlement_scale=2., timeout_draw_reward=0.,
                altitude_terminal_mode="result_remaining_hp", altitude_win_reward=5., altitude_loss_reward=-5.,
                shaping_reward_scale=0.)


class RewardRetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_damage_terminal_and_unchanged_altitude(self):
        b = OR.BatchObsReward(6, device="cpu", enable_kernel=False)
        s = torch.zeros(12, 9, dtype=torch.float64)
        s[:, 2] = -3000.; s[:, 6] = 200.; s[1::2, 0] = 700.
        s[0, 2] = -299.
        b.hp.copy_(torch.tensor([.3, .8, .6, 0., .8, .6, .6, .8, 1., 1., .8, .7]))
        b.hp_loss.zero_(); b.hp_loss[10:] = torch.tensor([.1, .2])
        term = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
        trunc = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.bool)
        got = b.compute_reward(s, term, reward_cfg(), trunc).reshape(6, 2)
        expected = torch.tensor([[-5.6, 5.6], [5., -5.], [5., -5.],
                                 [-5., 5.], [0., 0.], [.2, -.2]], dtype=torch.float64)
        torch.testing.assert_close(got, expected, atol=2e-7, rtol=0)
        independent = b.compute_reward(s, term, dict(reward_cfg(), altitude_settlement_scale=10.), trunc)
        self.assertAlmostEqual(float(independent[0]), -8., places=6)
        # The independent CPU reference must preserve the same crash settlement.
        from cuda_fdm.tests.altitude_reward_val import pair
        OR.RW.reset_distance_tracker()
        _, parts = OR.RW.compute_reward(*pair(299., own_hp=.3), 0., 0., None,
                                       {}, reward_cfg(), True, False, OR.RW._OWNSHIP_ALT_END)
        self.assertAlmostEqual(parts["terminal"], -5.6)

    def test_terminal_step_damage_not_double_counted_and_kill_crash_equal(self):
        b = OR.BatchObsReward(1, device="cpu", enable_kernel=False)
        s = torch.zeros(2, 9, dtype=torch.float64)
        s[:, 2] = -3000.; s[:, 6] = 200.; s[1, 0] = 700.
        term = torch.tensor([True]); trunc = torch.tensor([False])
        for remaining_before in (0.1, 0.3, 0.7, 1.0):
            damage = remaining_before * .4
            b.hp[:] = torch.tensor([remaining_before-damage, 1.], dtype=torch.float64)
            b.hp_loss[:] = torch.tensor([damage, 0.], dtype=torch.float64)
            s[0, 2] = -299.
            crash = b.compute_reward(s, term, reward_cfg(), trunc).clone()
            b.hp[0] = 0.; b.hp_loss[0] = remaining_before; s[0, 2] = -3000.
            kill = b.compute_reward(s, term, reward_cfg(), trunc)
            torch.testing.assert_close(crash, kill, atol=1e-12, rtol=0)
            self.assertAlmostEqual(float(crash[0]), -5. - 2.*remaining_before)
        # Simultaneous fatalities follow match scoring: draw terminal, not two wins.
        s[:, 2] = -299.; b.hp.fill_(.3); b.hp_loss.zero_()
        torch.testing.assert_close(b.compute_reward(s, term, reward_cfg(), trunc),
                                   torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)

    def test_gamma_0999_does_not_bootstrap_task_terminal(self):
        cfg = PPOGPUConfig(device="cpu", hidden=(8, 8), num_bins=3,
                           normalize_obs=False, rollout_steps=1, gamma=.999,
                           milestone_period=0, exploiter_iters=0, sched_period=0)
        tr = PPOGPUTrainer(BoundaryEnv(fresh_value=1000000.), cfg)
        def value(self, obs, state=None, episode_start=None):
            return obs[:, 0].clone(), state
        tr.model.value_step = MethodType(value, tr.model)
        _, returns, _ = tr.collect_rollout()
        # Synthetic terminal rewards stay unmodified; only the live lane bootstraps.
        torch.testing.assert_close(returns[0], torch.tensor([5., -5., -4., 5., 10.99]), atol=2e-6, rtol=0)

    def test_resume_guard_migration_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = fixture()
            for opt, params in ((old.actor_opt, old.model.actor_parameters()),
                                (old.critic_opt, old.model.critic_parameters())):
                opt.zero_grad(); sum(p.square().sum() for p in params).backward(); opt.step()
            old.iteration = old._committed_iteration = 31956
            path = Path(tmp) / "legacy.pt"
            old.save(path)
            ckpt = torch.load(path, weights_only=False)
            ckpt.pop("training_objective"); ckpt.pop("training_objective_history")
            torch.save(ckpt, path)
            new = fixture(gamma=.999)
            new.env.reward_cfg.update(reward_cfg())
            # Keep the immutable shaping base identical to the original run.
            new.env.reward_cfg["shaping_reward_scale"] = 1e-4
            with self.assertRaisesRegex(ValueError, "objective-change"):
                new.load(path)
            new.load(path, allow_objective_change=True)
            for key, actual in (("model", new.model.state_dict()),
                                ("actor_opt", new.actor_opt.state_dict()),
                                ("critic_opt", new.critic_opt.state_dict())):
                assert_tree_equal(ckpt[key], actual)
            self.assertEqual(new.iteration, 31956)
            self.assertEqual(len(new.training_objective_history), 1)
            new.save(path)
            again = fixture(gamma=.999); again.env.reward_cfg.update(new.env.reward_cfg)
            again.load(path)
            self.assertEqual(again._training_objective(), new._training_objective())
            self.assertEqual(again.training_objective_history, new.training_objective_history)
            with self.assertRaisesRegex(ValueError, "objective-change"):
                fixture().load(path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_parity_all_modes_with_old_and_new_coefficients(self):
        from cuda_fdm.tests.altitude_reward_gpu_val import seeds_for
        env = GpuDogfightVecEnv(8, seed=541)
        heights = np.array([[299., 1000.], [1000., 299.], [299., 299.],
                            [1000., 1000.], [1000., 500.], [1000., 2000.],
                            [305., 305.], [1000., 1000.]]).flatten()
        initial = seeds_for(env, np.full(16, 1000.))
        final = seeds_for(env, heights)
        from cuda_fdm.ic import build_seed_vector
        # No final-interval gun damage in the timeout-draw fixture.
        final[-1] = build_seed_vector(**env._ic_dict(10000., 0., -1000., 0., 200.))
        hp = torch.tensor([.3,.8,.3,.8,.4,.7,0.,.8,.6,.7,.7,.4,.3,.2,1.,1.],
                          dtype=torch.float64, device="cuda")
        controls = torch.zeros(16, 4, dtype=torch.float64, device="cuda")
        for cfg in (dict(OR.RW.MY_REWARD_CONFIG, shaping_reward_scale=0.), reward_cfg()):
            for mode in (0, 1, 2, 3):
                env.sim.load_seed(initial); env.obr.reset_all(); env.obr.hp.copy_(hp)
                env.obr.kernel_init_reward_state(env.sim.states)
                ref = OR.BatchObsReward(8, enable_kernel=False)
                ref.hp.copy_(hp); ref.initialize_reward_state(env.state9_flat().clone(), cfg=cfg)
                env.obr.t_sec[-1] = ref.t_sec[-1] = 199.9
                env.sim.load_seed(final); s9 = env.state9_flat().clone()
                ref.push_actions(controls); ref.advance(s9)
                got, term, trunc = env.obr.kernel_advance(env.sim.states, controls, cfg=cfg, reward_mode=mode)
                expected = ref.compute_reward(s9, term.bool(), cfg=cfg,
                                              truncated_env=trunc.bool(), reward_mode=mode)
                torch.testing.assert_close(got, expected, atol=2e-7, rtol=2e-7)
                if mode == 0:
                    base = -5. if cfg.get("altitude_terminal_mode") == "result_remaining_hp" else 0.
                    self.assertAlmostEqual(float(got[0]), base - .3 * cfg.get("altitude_settlement_scale", cfg["damage_scale"]), places=6)
                    self.assertAlmostEqual(float(got[-1]), cfg["timeout_draw_reward"], places=6)


if __name__ == "__main__":
    unittest.main()
