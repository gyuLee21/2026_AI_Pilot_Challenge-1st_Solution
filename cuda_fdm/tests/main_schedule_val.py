"""CPU-only schedule boundaries, immutable curriculum, CLI and real resume checks."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.obs_reward import RW
from cuda_fdm.tests.pool_identity_audit import ToyEnv


def fixture(**overrides):
    settings = dict(device="cpu", architecture="mlp", hidden=(8, 8), num_bins=3,
                    rollout_steps=64, normalize_obs=False, lr=3e-4, ent_coef=1e-3,
                    sched_period=2000, sched_lr_decay=.5, sched_ent_decay=.5,
                    sched_lr_floor=5e-5, sched_ent_floor=1e-4,
                    milestone_period=0, exploiter_iters=0, seed=79)
    settings.update(overrides)
    env = ToyEnv(2)
    env.reward_cfg = copy.deepcopy(RW.MY_REWARD_CONFIG)
    return PPOGPUTrainer(env, PPOGPUConfig(**settings))


def snapshot(tr):
    return (tr.actor_opt.param_groups[0]["lr"], tr.critic_opt.param_groups[0]["lr"],
            tr.cfg.ent_coef, tr.cfg.rollout_steps,
            tr.env.reward_cfg["shaping_reward_scale"])


class MainScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def apply(self, tr, iteration):
        with contextlib.redirect_stdout(io.StringIO()):
            tr._apply_schedule(iteration)

    def test_exact_boundaries_and_twenty_thousand(self):
        tr = fixture()
        cases = ((1, 3e-4, 1e-3, 64, 1.), (2000, 3e-4, 1e-3, 64, 1.),
                 (2001, 1.5e-4, 5e-4, 72, .6), (4000, 1.5e-4, 5e-4, 72, .6),
                 (4001, 7.5e-5, 2.5e-4, 80, .32), (6000, 7.5e-5, 2.5e-4, 80, .32),
                 (6001, 5e-5, 1.25e-4, 88, .12), (8000, 5e-5, 1.25e-4, 88, .12),
                 (8001, 5e-5, 1e-4, 96, 0.), (18001, 5e-5, 1e-4, 136, 0.),
                 (20000, 5e-5, 1e-4, 136, 0.))
        for iteration, lr, ent, rollout, shaping in cases:
            with self.subTest(iteration=iteration):
                self.apply(tr, iteration)
                for actual in snapshot(tr)[:2]:
                    self.assertAlmostEqual(actual, lr, places=15)
                self.assertAlmostEqual(tr.cfg.ent_coef, ent, places=15)
                self.assertEqual(tr.cfg.rollout_steps, rollout)
                self.assertAlmostEqual(tr.env.reward_cfg["shaping_reward_scale"], 1e-4*shaping, places=15)

    def test_vnext_cap_freezes_at_the_largest_observed_stable_rollout(self):
        tr = fixture(sched_rollout_cap=104)
        for iteration in (10001, 18001, 20000, 20001, 40001, 100000):
            with self.subTest(iteration=iteration):
                self.apply(tr, iteration)
                self.assertEqual(tr.cfg.rollout_steps, 104)

    def test_floor_and_monotonicity_at_every_main_iteration(self):
        tr = fixture()
        before = (float("inf"),) * 3
        for iteration in range(1, 20001):
            self.apply(tr, iteration)
            actual = snapshot(tr)[:3]
            for current, old, floor in zip(actual, before, (5e-5, 5e-5, 1e-4)):
                self.assertGreaterEqual(current, floor)
                self.assertLessEqual(current, old)
            before = actual

    def test_unchanged_reward_aux_gamma_rollout_and_exploiter_settings(self):
        tr = fixture(exploiter_lr=1e-4, exploiter_ent_coef=1e-4)
        reward = copy.deepcopy(tr.env.reward_cfg)
        fixed_names = ("gamma", "gae_lambda", "aux_coef", "clip_coef", "vf_coef",
                       "target_kl", "update_epochs", "num_minibatches", "milestone_period",
                       "exploiter_lr", "exploiter_ent_coef", "exploiter_clip_coef",
                       "exploiter_win_target", "exploiter_alt_hunt_coef", "pool_sample_temp")
        fixed = {k: getattr(tr.cfg, k) for k in fixed_names}
        self.apply(tr, 20000)
        for key, value in reward.items():
            if key != "shaping_reward_scale":
                self.assertEqual(tr.env.reward_cfg[key], value, key)
        self.assertEqual(fixed, {k: getattr(tr.cfg, k) for k in fixed_names})
        self.assertEqual(tr.cfg.rollout_steps, 136)
        self.assertEqual(tr.env.reward_cfg["shaping_reward_scale"], 0.)

    def test_no_schedule_means_no_decay_floor_or_reward_change(self):
        tr = fixture(sched_period=0)
        before = snapshot(tr)
        self.apply(tr, 20000)
        self.assertEqual(snapshot(tr), before)
        self.assertIsNone(tr._schedule_contract())

    def test_zero_floors_preserve_old_decay(self):
        tr = fixture(sched_lr_decay=1/3, sched_ent_decay=1/3,
                     sched_lr_floor=0., sched_ent_floor=0.)
        self.apply(tr, 20000)
        self.assertAlmostEqual(snapshot(tr)[0], 3e-4*(1/3)**9, places=20)
        self.assertAlmostEqual(tr.cfg.ent_coef, 1e-3*(1/3)**9, places=20)

    def test_independent_actor_critic_bases_share_floor(self):
        tr = fixture(critic_lr=2e-4)
        self.apply(tr, 4001)
        self.assertAlmostEqual(snapshot(tr)[0], 7.5e-5, places=15)
        self.assertAlmostEqual(snapshot(tr)[1], 5e-5, places=15)
        self.apply(tr, 20000)
        self.assertEqual(snapshot(tr)[:2], (5e-5, 5e-5))

    def test_invalid_floors_rejected(self):
        for key in ("sched_lr_floor", "sched_ent_floor"):
            for value in (-1., float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    PPOGPUConfig(**{key: value})
        for kwargs in ({"sched_lr_floor": 1.}, {"sched_ent_floor": .01, "ent_coef": .001},
                       {"sched_lr_floor": 5e-5, "critic_lr": 1e-5}):
            with self.assertRaises(ValueError):
                PPOGPUConfig(**kwargs)

    def test_repeated_schedule_application_does_not_decay_twice(self):
        tr = fixture()
        self.apply(tr, 6001)
        expected = snapshot(tr)
        contract = tr._schedule_contract()
        for _ in range(5):
            self.apply(tr, 6001)
            self.assertEqual(snapshot(tr), expected)
            self.assertEqual(tr._schedule_contract(), contract)
        self.assertEqual(contract["base"]["ent"], 1e-3)
        self.assertEqual(contract["base"]["rollout"], 64)

    def test_real_checkpoint_resume_across_decay_and_floor_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="main-schedule-resume-") as tmp:
            for saved_it in (2000, 6000, 8000, 18000, 19999):
                with self.subTest(saved_it=saved_it):
                    tr = fixture()
                    for optimizer, parameters in ((tr.actor_opt, tr.model.actor_parameters()),
                                                   (tr.critic_opt, tr.model.critic_parameters())):
                        optimizer.zero_grad(set_to_none=True)
                        sum(p.square().sum() for p in parameters).backward()
                        optimizer.step()  # nonzero Adam moments, not empty-state round trip
                    self.apply(tr, saved_it)
                    tr.iteration = saved_it
                    tr._committed_iteration = saved_it
                    tr._iteration_inflight = False
                    path = Path(tmp) / f"iter{saved_it}.pt"
                    tr.save(path)
                    restored = fixture()
                    ckpt = restored.load(path)
                    self.assertEqual(ckpt["main_schedule"]["base"]["ent"], .001)
                    self.assertEqual(ckpt["main_schedule"]["base"]["rollout"], 64)
                    self.assertEqual(restored.iteration, saved_it)
                    for key, value in tr.model.state_dict().items():
                        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
                    for opt, other in ((tr.actor_opt, restored.actor_opt), (tr.critic_opt, restored.critic_opt)):
                        for key, values in opt.state_dict()["state"].items():
                            for name, value in values.items():
                                torch.testing.assert_close(value, other.state_dict()["state"][key][name], rtol=0, atol=0)
                    self.apply(tr, saved_it + 1)
                    self.apply(restored, saved_it + 1)
                    self.assertEqual(snapshot(tr), snapshot(restored))
                    self.assertEqual(tr._schedule_contract(), restored._schedule_contract())

    def test_changed_or_missing_schedule_refuses_resume_before_loading_weights(self):
        with tempfile.TemporaryDirectory(prefix="main-schedule-mismatch-") as tmp:
            tr = fixture(); self.apply(tr, 6001)
            path = Path(tmp) / "checkpoint.pt"; tr.save(path)
            for kwargs in ({"sched_lr_decay": 1/3}, {"sched_lr_floor": 3e-5},
                           {"sched_ent_floor": 5e-5}, {"sched_period": 4000},
                           {"ent_coef": 1.25e-4}, {"rollout_steps": 88}, {"lr": 2e-4}):
                restored = fixture(**kwargs)
                before = copy.deepcopy(restored.model.state_dict())
                with self.assertRaisesRegex(ValueError, "main schedule"):
                    restored.load(path)
                for key, value in before.items():
                    torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
            ckpt = torch.load(path, weights_only=False)
            del ckpt["main_schedule"]
            missing = Path(tmp) / "missing.pt"; torch.save(ckpt, missing)
            with self.assertRaisesRegex(ValueError, "main schedule"):
                fixture().load(missing)

    def test_main_command_only_explicitly_enables_approved_schedule(self):
        from cuda_fdm.mlp_size_search import MLPSearch, PROTOCOL
        search = MLPSearch.__new__(MLPSearch)
        search.folder = Path("unused-schedule-test")
        search.stop = search.folder / "STOP"
        main, _ = search.train_args("mlp512_d3", 0, 20000, final=True)
        compare, _ = search.train_args("mlp512_d3", 0, 800, final=False)
        expected = {"--sched-period": "2000", "--sched-lr-decay": "0.5",
                    "--sched-ent-decay": "0.5", "--sched-lr-floor": "0.00005",
                    "--sched-ent-floor": "0.0001", "--exploiter-lr": "0.0001",
                    "--exploiter-ent-coef": "0.0001", "--aux-coef": "0.1"}
        for flag, value in expected.items():
            self.assertEqual(main[main.index(flag)+1], value)
        self.assertEqual(compare[compare.index("--sched-period")+1], "0")
        self.assertNotIn("--sched-lr-floor", compare)
        self.assertNotIn("--sched-ent-floor", compare)
        self.assertIn("--no-wandb", compare)
        self.assertEqual(PROTOCOL["final"]["lr_floor"], 5e-5)

    def test_cli_parses_and_forwards_floor_arguments_without_gpu_or_wandb(self):
        from cuda_fdm import train_gpu
        captured = {}
        class CapturedConfig(Exception):
            pass
        def stop_at_trainer(env, cfg):
            captured.update(vars(cfg))
            raise CapturedConfig()
        args = ["train_gpu", "--device", "cpu", "--no-wandb", "--sched-lr-decay", ".5",
                "--sched-ent-decay", ".5", "--sched-lr-floor", "5e-5", "--sched-ent-floor", "1e-4"]
        with patch.object(sys, "argv", args), \
                patch.object(train_gpu, "GpuDogfightVecEnv", return_value=SimpleNamespace(reward_cfg={})), \
                patch.object(train_gpu, "PPOGPUTrainer", side_effect=stop_at_trainer):
            with self.assertRaises(CapturedConfig):
                train_gpu.main()
        self.assertEqual(captured["sched_lr_floor"], 5e-5)
        self.assertEqual(captured["sched_ent_floor"], 1e-4)
        self.assertEqual(captured["sched_lr_decay"], .5)
        self.assertEqual(captured["sched_ent_decay"], .5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_exploiter_keeps_its_own_rates_and_restores_main_schedule(self):
        from cuda_fdm.tests.pool_episode_val import stateful_trainer
        tr = stateful_trainer()
        tr.env.reward_cfg = copy.deepcopy(RW.MY_REWARD_CONFIG)
        tr.cfg.lr = 3e-4; tr.cfg.ent_coef = .001
        tr.cfg.sched_period = 2000
        tr.cfg.sched_lr_decay = tr.cfg.sched_ent_decay = .5
        tr.cfg.sched_lr_floor = 5e-5; tr.cfg.sched_ent_floor = 1e-4
        tr.cfg.exploiter_lr = tr.cfg.exploiter_ent_coef = 1e-4
        tr.cfg.exploiter_iters = 1; tr.cfg.milestone_period = 500
        tr.iteration = 6500; self.apply(tr, 6500)
        expected, contract = snapshot(tr), tr._schedule_contract()
        class ObservedExploiter(Exception):
            pass
        def observe(*args, **kwargs):
            self.assertEqual(tr.actor_opt.param_groups[0]["lr"], 1e-4)
            self.assertEqual(tr.critic_opt.param_groups[0]["lr"], 1e-4)
            self.assertEqual(tr.cfg.ent_coef, 1e-4)
            raise ObservedExploiter()
        with patch.object(tr, "collect_rollout", side_effect=observe), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ObservedExploiter):
                tr.train_exploiter()
        self.assertEqual(snapshot(tr), expected)
        self.assertEqual(tr._schedule_contract(), contract)


if __name__ == "__main__":
    unittest.main(verbosity=2)
