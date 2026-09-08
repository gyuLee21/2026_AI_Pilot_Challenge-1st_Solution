"""CPU-only reward contract, alternation and side-learner restoration regressions."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from GeoMathUtil import GeometryInfo
from claude_code import my_reward as MR
from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.reward_modes import (REWARD_CONTRACT, scheduled_exploiter_mode,
                                   log_altitude, validate_reward_mode)
from cuda_fdm.tests.pool_episode_val import stateful_trainer
from dogfight.sim.state_schema import StateIndex as S


def pair(own_alt=1000., target_alt=1000., own_hp=1., target_hp=1., time=0.):
    own, target = np.zeros(46), np.zeros(46)
    for state, altitude, hp in ((own, own_alt, own_hp), (target, target_alt, target_hp)):
        state[2] = -altitude
        state[6] = 200.
        state[S.ALT] = altitude
        state[S.HEALTH] = hp
        state[S.SIM_TIME] = time
    target[0] = 700.
    return own, target


def full_states(s9, hp, time):
    result = np.zeros((len(s9), 46))
    result[:, :9] = np.asarray(s9)
    result[:, S.ALT] = -result[:, 2]
    result[:, S.HEALTH] = hp
    result[:, S.SIM_TIME] = time
    return result


def cpu_transition(initial, current, mode=0, shaping=0., damage=(0., 0.), term=False, trunc=False, end=""):
    cfg = dict(MR.MY_REWARD_CONFIG, shaping_reward_scale=shaping, reward_mode=mode)
    geo = GeometryInfo()
    MR.reset_distance_tracker()
    MR.initialize_reward_episode(*initial, geo, cfg)
    return MR.compute_reward(*current, *damage, geo, {}, cfg, term, trunc, end)


class AltitudeRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_penalty_for_descent_or_old_1000ft_refund(self):
        for start, finish in ((3000., 500.), (610., 305.), (1001*.3048, 999*.3048)):
            reward, terms = cpu_transition(pair(start), pair(finish, time=.1))
            self.assertEqual(reward, 0.)
            self.assertEqual(terms["safety"], 0.)
            self.assertEqual(terms["altitude_hunt"], 0.)

    def test_remaining_hp_settlement_both_sides_and_last_damage_not_counted_twice(self):
        for hp in (0., .3, 1.):
            reward, terms = cpu_transition(pair(own_hp=hp), pair(299., own_hp=hp, time=.1),
                                           term=True, end=MR._OWNSHIP_ALT_END)
            self.assertAlmostEqual(reward, -10*hp)
            reward, terms = cpu_transition(pair(target_hp=hp), pair(target_alt=299., target_hp=hp, time=.1),
                                           term=True, end=MR._TARGET_ALT_END)
            self.assertAlmostEqual(reward, 10*hp)
        reward, terms = cpu_transition(pair(own_hp=.8), pair(299., own_hp=.6, time=.1),
                                      damage=(.2, 0.), term=True, end=MR._OWNSHIP_ALT_END)
        self.assertAlmostEqual(terms["damage"], -2.)
        self.assertAlmostEqual(terms["terminal"], -6.)
        self.assertAlmostEqual(reward, -8.)

    def test_hunter_frozen_target_descent_climb_and_stationary(self):
        for start, finish in ((1000., 500.), (500., 1000.), (700., 700.)):
            reward, terms = cpu_transition(pair(target_alt=start), pair(target_alt=finish, time=.1), mode=1)
            self.assertAlmostEqual(reward, 5*math.log(start/finish), places=12)
            self.assertEqual(terms["geometry"], 0.)
        reward, _ = cpu_transition(pair(3000.), pair(500., time=.1), mode=1)
        self.assertEqual(reward, 0.)  # own descent is not the hunter objective
        cfg = dict(MR.MY_REWARD_CONFIG, reward_mode=1)
        geo = GeometryInfo()
        initial = pair(target_alt=9000.)
        MR.initialize_reward_episode(*initial, geo, cfg)
        next_state = pair(target_alt=9000., time=.1)
        reward, _ = MR.compute_reward(*next_state, 0., 0., geo, {}, cfg, False, False, "")
        self.assertEqual(reward, 0.)  # no carry-over from previous test's episode

    def test_standard_geometry_remains_hunter_replaces_not_adds(self):
        standard, terms = cpu_transition(pair(), pair(time=.1), shaping=.0001)
        self.assertGreater(standard, 0.)
        self.assertLessEqual(abs(standard), .0025)
        self.assertEqual(standard, terms["geometry"])
        hunter, hterms = cpu_transition(pair(), pair(time=.1), mode=1, shaping=.0001)
        self.assertEqual(hunter, 0.)
        self.assertEqual(hterms["geometry"], 0.)

    def test_hp_kill_timeout_and_both_crash_priority(self):
        reward, _ = cpu_transition(pair(), pair(target_hp=0., time=.1), term=True)
        self.assertEqual(reward, 5.)
        for hp, expected in ((.8, 5.), (1., -4.)):
            reward, _ = cpu_transition(pair(time=199.9), pair(target_hp=hp, time=200.), trunc=True)
            self.assertEqual(reward, expected)
        own, tgt = pair(100., 100., .3, .8, .1)
        cfg = dict(MR.MY_REWARD_CONFIG, shaping_reward_scale=0.)
        bor = BatchObsReward(1, device="cpu", enable_kernel=False)
        s9 = torch.tensor(np.stack((own[:9], tgt[:9])))
        bor.hp.copy_(torch.tensor([.3, .8], dtype=torch.float64))
        reward = bor.compute_reward(s9, torch.ones(1, dtype=torch.bool), cfg)
        torch.testing.assert_close(reward, torch.tensor([-3., -8.], dtype=torch.float64), rtol=0, atol=1e-12)

    def test_cpu_and_tensor_reference_random_rewards_including_terminals(self):
        rng = np.random.default_rng(9401)
        for mode in (0, 1, 2, 3):
            for case in range(24):
                initial = np.stack(pair())
                initial[:, :2] = rng.uniform(-900, 900, (2, 2))
                initial[:, 3:6] = rng.uniform(-70, 70, (2, 3))
                initial[:, 2] = -rng.uniform(305, 8000, 2)
                initial[:, S.ALT] = -initial[:, 2]
                current = initial.copy()
                current[:, 2] += rng.uniform(-50, 50, 2)
                if case % 6 == 0:
                    current[0, 2] = -250.
                current[:, S.ALT] = -current[:, 2]
                current[:, S.SIM_TIME] = .1
                hp = rng.uniform(.1, 1., 2)
                loss = rng.uniform(0, .05, 2)
                current[:, S.HEALTH] = hp
                term = bool((current[:, S.ALT] < 300).any())
                end = MR._OWNSHIP_ALT_END if current[0, S.ALT] < 300 else MR._TARGET_ALT_END
                cfg = dict(MR.MY_REWARD_CONFIG, reward_mode=mode)
                reward, _ = cpu_transition(initial, current, mode=mode, shaping=.0001,
                    damage=loss, term=term, end=end)
                bor = BatchObsReward(1, device="cpu", enable_kernel=False)
                bor.initialize_reward_state(torch.tensor(initial[:, :9]), cfg=cfg)
                bor.hp.copy_(torch.tensor(hp)); bor.hp_loss.copy_(torch.tensor(loss)); bor.t_sec.fill_(.1)
                got = bor.compute_reward(torch.tensor(current[:, :9]), torch.tensor([term]),
                                         cfg=cfg, reward_mode=mode)
                self.assertAlmostEqual(float(got[0]), reward, places=10)
                self.assertTrue(bool(torch.isfinite(got).all()))

    def test_log_floor_is_finite_and_parameters_fail_closed(self):
        for alt in (0., -1., 1e-9, 300., 10000.):
            self.assertTrue(math.isfinite(log_altitude(alt)))
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                log_altitude(bad)
        for mode, coef in ((4, 5), (0, -1), (1, float("nan"))):
            with self.assertRaises(ValueError):
                validate_reward_mode(mode, coef)

    def test_hunter_each_three_is_main_iteration_based_to_20000(self):
        modes = [scheduled_exploiter_mode(i, 500) for i in range(500, 20001, 500)]
        self.assertEqual(modes, ([1, 0, 2, 1, 3, 0, 1, 2, 3]*5)[:40])
        self.assertEqual([500*(i+1) for i, mode in enumerate(modes) if mode == 1],
                         list(range(500, 20001, 1500)))
        self.assertEqual([mode for mode in modes if mode != 1], ([0, 2, 3]*9)[:26])
        self.assertEqual(scheduled_exploiter_mode(1500, 500, alternate=False), 0)
        self.assertEqual(scheduled_exploiter_mode(1499, 500), 0)
        self.assertEqual(scheduled_exploiter_mode(500, 0), 0)

    def test_actual_exploiter_loop_stops_at_75pct_ema_or_budget(self):
        # EMA starts at .5, alpha=.1. Perfect batches pass .70 on the fifth
        # update but do not pass .75 until the seventh: distinguish the thresholds.
        # One reset-censored batch plus two fresh qualifying EMA batches.
        for wins, budget, expected_steps in ((10., 1000, 9), (0., 3, 3)):
            trainer = stateful_trainer()
            trainer.cfg.exploiter_iters = budget
            trainer.cfg.exploiter_win_target = .75
            trainer.cfg.milestone_period = 500
            trainer.iteration = 500
            stats = dict(ep_count=10., win_sum=wins, loss_sum=10.-wins,
                         ret_sum=0., alt_loss_sum=0., opponent_alt_loss_sum=wins)
            updates = dict(pl=0., vl=0., ent=0., kl=0., cf=0., ev=0.)
            with patch.object(trainer, "collect_rollout", return_value=(None, None, stats)) as rollout, \
                    patch.object(trainer, "update", return_value=updates):
                score = trainer.train_exploiter()
            self.assertEqual(rollout.call_count, expected_steps)
            self.assertEqual(trainer.exploiter_history[-1]["iterations"], expected_steps)
            self.assertEqual(trainer.exploiter_history[-1]["reward_mode"], "altitude_hunt")
            self.assertEqual(score >= .75, wins > 0)
            self.assertEqual(trainer.pool.num_permanent(), 1)

    def test_exploiter_modes_restore_runtime_even_on_failure_and_resume(self):
        # Defense selection was disabled by the user; its slot falls back to standard.
        for milestone, expected_mode in ((500, 1), (1000, 1), (1500, 2), (2000, 1), (2500, 0)):
            for fail in (False, True):
                with self.subTest(milestone=milestone, fail=fail):
                    trainer = stateful_trainer()
                    trainer.cfg.milestone_period = 500
                    trainer.cfg.exploiter_win_target = .75
                    trainer.iteration = milestone
                    # Emulate a completed boundary, not an arbitrary in-flight
                    # iteration label. save() intentionally uses this counter.
                    trainer._committed_iteration = milestone
                    trainer.env.reward_mode, trainer.env.alt_hunt_coef = 0, 5.
                    before = trainer._runtime_state()
                    seen = []
                    def inner(**kwargs):
                        seen.append(trainer.env.reward_mode)
                        trainer.env.sim.states.fill_(7.)
                        if fail:
                            raise FloatingPointError("injected side-learner failure")
                        return .76
                    with patch.object(trainer, "_train_exploiter_inner", side_effect=inner):
                        if fail:
                            with self.assertRaises(FloatingPointError):
                                trainer.train_exploiter()
                        else:
                            self.assertEqual(trainer.train_exploiter(), .76)
                    self.assertEqual(seen, [expected_mode])
                    self.assertEqual(trainer.env.reward_mode, 0)
                    self.assertEqual(trainer.env.alt_hunt_coef, 5.)
                    torch.testing.assert_close(trainer.env.sim.states, before["sim"], rtol=0, atol=0)
                    with tempfile.TemporaryDirectory() as directory:
                        checkpoint = Path(directory) / "state.pt"
                        trainer.save(checkpoint)
                        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        self.assertEqual(data["reward_contract"], REWARD_CONTRACT)
                        trainer.load(checkpoint)
                        self.assertEqual(trainer.iteration, milestone)
                        resumed_seen = []
                        with patch.object(trainer, "_train_exploiter_inner",
                                          side_effect=lambda **kw: resumed_seen.append(trainer.env.reward_mode) or .76):
                            trainer.train_exploiter()
                        self.assertEqual(resumed_seen, seen)
                        bad = copy.deepcopy(data)
                        bad["reward_contract"] = "previous_low_altitude_shaping"
                        torch.save(bad, checkpoint)
                        with self.assertRaisesRegex(ValueError, "reward contract"):
                            trainer.load(checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
