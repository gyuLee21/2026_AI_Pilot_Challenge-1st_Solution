"""Main-only diverse exploiters: no damage weights leak into task terminals."""
import copy
import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from claude_code import my_reward as MR
from GeoMathUtil import GeometryInfo
from cuda_fdm.reward_modes import (damage_reward, scheduled_exploiter_mode,
    reward_mode_name, EXPLOITER_CYCLE)
from cuda_fdm.tests.altitude_reward_val import pair, cpu_transition
from cuda_fdm.tests.pool_episode_val import stateful_trainer
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal
from dogfight.sim.state_schema import StateIndex as S


class DamageStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_damage_changes_even_on_terminal_frames(self):
        cases = [
            (pair(time=.1), False, False, ""),
            (pair(299., own_hp=.6, time=.1), True, False, MR._OWNSHIP_ALT_END),
            (pair(target_alt=299., target_hp=.7, time=.1), True, False, MR._TARGET_ALT_END),
            (pair(target_hp=0., time=.1), True, False, ""),
            (pair(own_hp=0., time=.1), True, False, ""),
            (pair(own_hp=.6, target_hp=.7, time=200.), False, True, ""),
            (pair(299., own_hp=.6, time=200.), True, False, MR._OWNSHIP_ALT_END),
        ]
        for current, term, trunc, end in cases:
            initial = pair(time=199.9 if trunc or current[0][S.SIM_TIME] == 200 else 0.)
            baseline, original = cpu_transition(initial, current, shaping=.0001,
                damage=(.2, .1), term=term, trunc=trunc, end=end)
            for mode, expected_damage in ((2, 1.), (3, -2.)):
                reward, terms = cpu_transition(initial, current, mode=mode, shaping=.0001,
                    damage=(.2, .1), term=term, trunc=trunc, end=end)
                self.assertAlmostEqual(terms["damage"], expected_damage)
                for key in ("geometry", "terminal", "safety", "altitude_hunt"):
                    self.assertEqual(terms[key], original[key], (mode, key, end))
                self.assertAlmostEqual(reward-baseline, expected_damage-original["damage"])

    def test_altitude_settlement_is_not_removed_in_attack_or_defense(self):
        for mode in (0, 1, 2, 3):
            for end, current, expected in (
                (MR._OWNSHIP_ALT_END, pair(299., own_hp=.6, time=.1), -6.),
                (MR._TARGET_ALT_END, pair(target_alt=299., target_hp=.7, time=.1), 7.)):
                _, terms = cpu_transition(pair(), current, mode=mode, term=True, end=end)
                self.assertAlmostEqual(terms["terminal"], expected)

    def test_original_standard_and_hunter_cpu_outputs_bitwise(self):
        # Frozen upstream source remains an independent arithmetic reference.
        root = next(p for p in Path(__file__).resolve().parents
                    if (p / "scripts/continue_depth_final_width.py").is_file())
        spec = importlib.util.spec_from_file_location("upstream_reward_reference", root / "claude_code/my_reward.py")
        reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        rng = np.random.default_rng(9311)
        for mode in (0, 1):
            for _ in range(30):
                initial = pair(target_alt=float(rng.uniform(350, 6000)))
                current = pair(target_alt=float(rng.uniform(350, 6000)), time=.1)
                losses = tuple(rng.uniform(0, .2, 2))
                cfg = dict(MR.MY_REWARD_CONFIG, reward_mode=mode)
                geo = GeometryInfo()
                reference.reset_distance_tracker()
                reference.initialize_reward_episode(*initial, geo, cfg)
                expected, _ = reference.compute_reward(*current, *losses, geo, {}, cfg, False, False, "")
                actual, _ = cpu_transition(initial, current, mode=mode, shaping=.0001, damage=losses)
                self.assertEqual(actual, expected)

    def test_damage_scalar_and_tensor_exact_formulas(self):
        rng = np.random.default_rng(55)
        dealt, taken = (torch.tensor(x, dtype=torch.float64) for x in rng.uniform(0, 1, (2, 200)))
        for mode, expected in ((0, (dealt-taken)*10), (1, (dealt-taken)*10),
                               (2, dealt*10), (3, -taken*10)):
            torch.testing.assert_close(damage_reward(mode, dealt, taken, 10.), expected, rtol=0, atol=0)

    def test_cycle_resume_and_disabled_milestones(self):
        self.assertEqual(EXPLOITER_CYCLE, (1, 0, 2, 1, 3, 0, 1, 2, 3))
        events = [(i, reward_mode_name(scheduled_exploiter_mode(i, 500)))
                  for i in range(500, 20001, 500)]
        self.assertEqual(len(events), 40)  # unchanged work count
        self.assertEqual(sum(mode == "altitude_hunt" for _, mode in events), 14)
        for offset in range(1, 40):
            self.assertEqual(events[offset:], [(i, reward_mode_name(scheduled_exploiter_mode(i, 500)))
                for i in range(500*(offset+1), 20001, 500)])
        self.assertEqual(scheduled_exploiter_mode(0, 500), 0)
        self.assertEqual(scheduled_exploiter_mode(500, 500, False), 0)
        self.assertEqual(scheduled_exploiter_mode(1501, 500), 0)

    def test_all_main_context_restored_for_attack_defense_and_exception(self):
        for iteration, expected_mode in ((1500, 2), (2500, 3)):
            for fail in (False, True):
                trainer = stateful_trainer()
                trainer.iteration, trainer.cfg.milestone_period = iteration, 500
                trainer.env.reward_mode, trainer.env.alt_hunt_coef = 0, 5.
                before = trainer._runtime_state()
                learner = copy.deepcopy(trainer._snapshot_learner())
                def inner(**kwargs):
                    self.assertEqual(trainer.env.reward_mode, expected_mode)
                    trainer.env.sim.states.fill_(77.)
                    with torch.no_grad():
                        next(trainer.model.parameters()).fill_(8.)
                    np.random.random(); torch.rand(8); trainer.env.rng.random()
                    if fail:
                        raise FloatingPointError("injected style failure")
                    return .8
                with patch.object(trainer, "_train_exploiter_inner", side_effect=inner):
                    if fail:
                        with self.assertRaises(FloatingPointError):
                            trainer.train_exploiter()
                    else:
                        trainer.train_exploiter()
                after = trainer._runtime_state()
                assert_tree_equal(learner, trainer._snapshot_learner())
                for key in ("sim", "obr", "reward_mode", "alt_hunt_coef", "env_rng", "numpy_rng", "torch_rng"):
                    assert_tree_equal(before[key], after[key])
                for key in before["trainer"]:
                    if key != "opp_weights":
                        assert_tree_equal(before["trainer"][key], after["trainer"][key])
                self.assertEqual(trainer.env.reward_mode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
