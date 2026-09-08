"""CPU boundary regression using real reset/capture/restore/reward code."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from claude_code.my_reward import MY_REWARD_CONFIG
from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.rl_env import GpuDogfightVecEnv


def warmup_fixture(*, delta=100., max_steps=4, phases=None, altitude=1000.):
    phases = np.arange(4) if phases is None else np.asarray(phases)
    class FixedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, max_steps, 4)
            return phases.copy()
    class Physics:
        def load_seed(self, seeds):
            self.states = torch.tensor(seeds, dtype=torch.float64)
        def step(self, actions, substeps):
            self.states[1::2, 2] += delta
            self.states[:, 0] += torch.arange(8, dtype=torch.float64)
            self.states[:, 3] += .001
    env = GpuDogfightVecEnv.__new__(GpuDogfightVecEnv)
    env.nenv, env.nac, env.device, env.substeps = 4, 8, "cpu", 6
    env.sim, env.rng = Physics(), FixedRng()
    env.obr = BatchObsReward(4, device="cpu", enable_kernel=False)
    seeds = np.zeros((8, 101)); seeds[:, 2] = -altitude; seeds[:, 6] = 200.
    seeds[1::2, 0] = 8000.
    env._build_all_seeds = lambda: seeds.copy()
    env.state9_flat = lambda: env.sim.states[:, :9]
    env._random_actions = lambda: torch.tensor([[.1, -.1, .05, .8]]*8, dtype=torch.float64)
    env._seed_envs = lambda indices: [env.sim.states.__setitem__(slice(2*i, 2*i+2),
                                    torch.tensor(seeds[2*i:2*i+2])) for i in indices]
    calls = []
    def initialize(states, mask=None):
        before = env.obr.clone_state()
        env.obr.initialize_reward_state(states[:, :9], mask)
        after = env.obr.clone_state()
        for key in before:
            if key not in ("prev_x", "prev_x_valid", "prev_alt_log"):
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        calls.append(mask)
    env.obr.kernel_init_reward_state = initialize
    return env, calls


class WarmupRewardTests(unittest.TestCase):
    def test_no_imaginary_hunt_reward_after_different_warmup_phases(self):
        for delta in (100., -100., 0.):
            env, calls = warmup_fixture(delta=delta)
            env.reset(stagger=True, max_stagger_steps=4)
            state = env.state9_flat().clone()
            torch.testing.assert_close(env.obr.prev_alt_log, env.obr._altitude_log(state), atol=0, rtol=0)
            env.obr.advance(state)
            reward = env.obr.compute_reward(state, torch.zeros(4, dtype=torch.bool), reward_mode=1)
            torch.testing.assert_close(reward, torch.zeros_like(reward), atol=1e-12, rtol=0)
            self.assertEqual(len(calls), 2)  # initial IC and restored warmup, not per step

    def test_first_actual_descent_is_rewarded_exactly_once(self):
        env, _ = warmup_fixture()
        env.reset(stagger=True, max_stagger_steps=4)
        state = env.state9_flat().clone(); previous = env.obr.prev_alt_log.clone()
        state[1::2, 2] += 10.
        env.obr.advance(state)
        reward = env.obr.compute_reward(state, torch.zeros(4, dtype=torch.bool), reward_mode=1)
        expected = 5.*(previous[env.obr.partner]-env.obr._altitude_log(state)[env.obr.partner])
        torch.testing.assert_close(reward, expected, atol=1e-12, rtol=0)
        second = env.obr.compute_reward(state, torch.zeros(4, dtype=torch.bool), reward_mode=1)
        torch.testing.assert_close(second, torch.zeros_like(second), atol=1e-12, rtol=0)

    def test_geometry_anchor_is_current_state_and_occupancy_is_not_zeroed(self):
        env, _ = warmup_fixture()
        env.reset(stagger=True, max_stagger_steps=4)
        state = env.state9_flat().clone()
        reference = BatchObsReward(4, device="cpu", enable_kernel=False)
        reference.restore(env.obr.clone_state()); reference.initialize_reward_state(state)
        torch.testing.assert_close(env.obr.prev_x, reference.prev_x, atol=0, rtol=0)
        frozen = env.obr.prev_x.clone()
        env.obr.advance(state)
        got = env.obr.compute_reward(state, torch.zeros(4, dtype=torch.bool), reward_mode=0)
        expected = frozen * MY_REWARD_CONFIG["geometry_episode_budget"] / MY_REWARD_CONFIG["geometry_reference_duration_sec"] * env.obr.dt
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=0)

    def test_hp_time_attitude_and_action_history_are_preserved(self):
        env, _ = warmup_fixture()
        env.reset(stagger=True, max_stagger_steps=4)
        torch.testing.assert_close(env.obr.t_sec, torch.arange(4, dtype=torch.float64)*env.obr.dt)
        self.assertEqual(env.obr.act_hist[2, 0, 3].item(), .8)
        self.assertTrue(bool(env.obr.prev_valid[2:].all()))
        self.assertEqual(env.obr.act_hist[:2].count_nonzero().item(), 0)

    def test_nonstagger_reset_keeps_initial_baseline(self):
        env, calls = warmup_fixture()
        env.reset(stagger=False)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(env.obr.prev_alt_log, env.obr._altitude_log(env.state9_flat()), atol=0, rtol=0)

    def test_reset_does_not_connect_previous_episode(self):
        env, _ = warmup_fixture()
        env.reset(stagger=True, max_stagger_steps=4)
        env.obr.prev_alt_log.fill_(999.); env.obr.prev_x.fill_(999.)
        env.reset(stagger=True, max_stagger_steps=4)
        got = env.obr.compute_reward(env.state9_flat(), torch.zeros(4, dtype=torch.bool), reward_mode=1)
        torch.testing.assert_close(got, torch.zeros_like(got), atol=1e-12, rtol=0)

    def test_fallback_after_uncaptured_departures_reanchors_current_ic(self):
        env, _ = warmup_fixture(delta=1000., phases=[0, 2, 3, 3])
        env.reset(stagger=True, max_stagger_steps=4)
        torch.testing.assert_close(env.obr.prev_alt_log, env.obr._altitude_log(env.state9_flat()), atol=0, rtol=0)
        got = env.obr.compute_reward(env.state9_flat(), torch.zeros(4, dtype=torch.bool), reward_mode=1)
        torch.testing.assert_close(got, torch.zeros_like(got), atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
