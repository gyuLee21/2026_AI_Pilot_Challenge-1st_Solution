"""Future-residual targets, shared heads, PPO integration and deployment (CPU).

The loop reference is Git origin/main2's formula, not another call to production
target construction. No CUDA, network access or historical-run modification.
"""
import copy
import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from cuda_fdm.future_aux import (AUX_CONTRACT, AUX_PROTOCOL, AUX_STATE_KEYS, AUX_METRIC_KEYS,
                                 build_future_labels, auxiliary_error, inference_state_dict)
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer, build_actor_critic
from cuda_fdm.finite_checks import require_finite
from cuda_fdm.tests.pool_episode_val import StatefulToy
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal


def git_loop_labels(features, starts, dt=.1, early_float=False):
    """Remote _capture_aux/_build_aux_labels (FP32 capture is optional).

    Git stores all features as float before the per-time-step calculations.
    Invalid label values are unspecified there: compare only the true mask.
    """
    x = features.float() if early_float else features
    tmax, n = x.shape[:2]
    labels = torch.zeros(tmax, n, 6, device=x.device)
    mask = torch.zeros(tmax, n, 2, dtype=torch.bool, device=x.device)
    for t in range(tmax):
        for slot, horizon, pos, vel in ((0, 5, 6, 9), (1, 10, 0, 3)):
            if t + horizon < tmax:
                valid = starts[t + 1:t + horizon + 1].sum(0) == 0
                baseline = x[t, :, pos:pos+3] + x[t, :, vel:vel+3] * (horizon * dt)
                residual = x[t+horizon, :, pos:pos+3] - baseline
                rotation = x[t, :, 12:21].reshape(n, 3, 3)
                labels[t, :, 3*slot:3*slot+3] = (rotation @ residual.unsqueeze(-1)).squeeze(-1) / 100.
                mask[t, :, slot] = valid
    return labels, mask


def feature_fixture(time=64, environments=4, device="cpu"):
    t = torch.arange(time, dtype=torch.float64, device=device).view(-1, 1, 1) * .1
    offset = torch.arange(environments, dtype=torch.float64, device=device).view(1, -1, 1) * 1300.
    x = torch.zeros(time, environments, 21, dtype=torch.float64, device=device)
    for pos, vel, speed, acceleration in ((0, 3, (220., 30., 1.), (2., 4., -3.)),
                                          (6, 9, (200., -20., 4.), (-6., 2., 1.))):
        v = torch.tensor(speed, device=device)
        a = torch.tensor(acceleration, device=device)
        x[..., pos:pos+3] = offset + v*t + .5*a*t*t
        x[..., vel:vel+3] = v + a*t
    x[..., 12:21] = torch.eye(3, dtype=torch.float64, device=device).flatten()
    return x


class AuxToy(StatefulToy):
    # The approved CUDA observation now includes the 30D own/target
    # acceleration block (214 -> 214).  Keep the toy aligned with the
    # production trainer contract so these tests exercise the real path.
    OBS_SIZE = 214

    def __init__(self, nenv=4):
        super().__init__(nenv)
        self.episode_steps = 23
        self.obr = SimpleNamespace(dt=.1, aux_features=None, enable_aux_capture=self.enable_aux,
                                   clone_state=self.clone_aux, restore=self.restore_aux)

    def enable_aux(self):
        self.obr.aux_features = torch.zeros(self.nenv, 21, dtype=torch.float64)

    def capture(self):
        if self.obr.aux_features is not None:
            self.obr.aux_features.copy_(feature_fixture(self.episode_step + 1, self.nenv)[-1])

    def reset(self, stagger=True):
        obs = super().reset(stagger)
        self.capture()
        return obs

    def step(self, controls):
        value = super().step(controls)
        self.capture()
        return value

    def clone_aux(self):
        data = dict(episode_step=self.episode_step)
        if self.obr.aux_features is not None:
            data["aux_features"] = self.obr.aux_features.clone()
        return data

    def restore_aux(self, data):
        self.episode_step = data["episode_step"]
        if self.obr.aux_features is not None:
            self.obr.aux_features.copy_(data["aux_features"])


def aux_trainer(enabled=True, **kwargs):
    config = dict(device="cpu", architecture="mlp", hidden=(16, 16, 16), num_bins=3,
                  aux_pred=enabled, rollout_steps=32, update_epochs=2, num_minibatches=4,
                  normalize_obs=True, opp_sample=False, seed=52, sched_period=0,
                  milestone_period=0, exploiter_iters=1, exploiter_win_target=1.0,
                  target_kl=None, total_iterations=2, save_runtime=True)
    config.update(kwargs)
    return PPOGPUTrainer(AuxToy(), PPOGPUConfig(**config))


def validate_cli_artifacts(folder):
    """Validate the actual bounded CLI checkpoint, scalar CSV and exported bundle."""
    from claude_code.action_provider import MLPActionProvider
    from cuda_fdm.train_gpu import csv_header
    folder = Path(folder)
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    assert checkpoint["iteration"] == 2 and checkpoint["training_protocol"] == AUX_PROTOCOL
    assert checkpoint["auxiliary_contract"] == AUX_CONTRACT
    for key, expected in dict(aux_pred=True, aux_coef=.1, architecture="mlp", gamma=.997,
                              gae_lambda=.95, rollout_steps=64, update_epochs=4,
                              num_minibatches=8, milestone_period=0, exploiter_iters=0).items():
        assert cfg[key] == expected, (key, cfg[key])
    require_finite(checkpoint, "actual CLI checkpoint")
    with (folder / "metrics.csv").open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == csv_header(True).split(",")
        rows = list(reader)
    assert len(rows) == 2
    for row in rows:
        assert all(math.isfinite(float(value)) for key, value in row.items() if key != "early_stop")
        assert int(row["gstep"]) == int(row["iter"]) * 4096 * 64
        assert all(0 <= float(row[key]) <= 1 for key in ("aux_opp_coverage", "aux_self_coverage"))
    assert not list(folder.glob("*.wandbid"))
    model = build_actor_critic(architecture="mlp", obs_dim=214, act_dim=4,
                               num_bins=cfg["num_bins"], hidden=cfg["hidden"],
                               activation=cfg["activation"], aux_pred=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    provider = MLPActionProvider(folder / "bundle", stochastic=False, debug_obs=False)
    assert not any("aux" in key for key in provider.model.state_dict())
    assert_tree_equal(inference_state_dict(model.state_dict(), True), provider.model.state_dict())
    x = torch.randn(512, 214, generator=torch.Generator().manual_seed(827))
    with torch.no_grad():
        before, after = model.actor_logits(x), provider.model.actor_logits(x)
    assert_tree_equal(before, after)
    data = dict(passed=True, iterations=2, environments=4096,
                 protocol=AUX_PROTOCOL, observations=214, action_channels=4,
                 auxiliary_columns=list(AUX_METRIC_KEYS), finite_csv_checkpoint=True,
                 no_wandb_id=True, exported_auxiliary_keys_absent=True,
                 checkpoint_bundle_cpu_logits_bitwise_equal=True,
                 iteration_seconds=[float(row["elapsed"]) for row in rows],
                 first_iteration_includes_lazy_initialization=True,
                 combat_skill_improvement_not_evaluated=True)
    (folder / "artifact_check.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return data


class FutureAuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_git_formula_and_masks_match_including_reset_at_horizon(self):
        x = feature_fixture()
        starts = torch.zeros(64, 4)
        starts[0] = 1
        starts[5, 0] = 1
        starts[10, 1] = 1
        starts[17:20, 2] = 1
        starts[63, 3] = 1
        labels, mask = build_future_labels(x, starts, .1)
        expected, expected_mask = git_loop_labels(x, starts)
        self.assertTrue(torch.equal(mask, expected_mask))
        valid = mask.repeat_interleave(3, -1)
        torch.testing.assert_close(labels[valid], expected[valid], atol=1e-7, rtol=1e-6)
        self.assertEqual(float(labels[~valid].abs().sum()), 0.)
        self.assertFalse(bool(mask[0, 0, 0]))  # endpoint after reset is not the old episode
        self.assertFalse(bool(mask[0, 1, 1]))
        self.assertTrue(bool(mask[5, 0, 0]))   # start flag at t itself is allowed
        self.assertFalse(bool(mask[-5:, :, 0].any()))
        self.assertFalse(bool(mask[-10:, :, 1].any()))

    def test_constant_velocity_zero_and_acceleration_units_main_body_frame(self):
        x = feature_fixture()
        starts = torch.zeros(64, 4)
        labels, _ = build_future_labels(x, starts, .1)
        expected = torch.tensor([-.0075, .0025, .00125, .01, .02, -.015])
        torch.testing.assert_close(labels[0, 0], expected, atol=2e-8, rtol=1e-6)
        rotation = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        x[..., 12:21] = rotation.flatten()
        labels, _ = build_future_labels(x, starts, .1)
        torch.testing.assert_close(labels[0, 0].view(2, 3), expected.view(2, 3) @ rotation.T)
        t = torch.arange(64, dtype=torch.float64).view(-1, 1, 1) * .1
        for pos, vel in ((0, 3), (6, 9)):
            x[..., vel:vel+3] = x[0, :, vel:vel+3].clone()
            x[..., pos:pos+3] = x[0, :, pos:pos+3].clone() + t*x[..., vel:vel+3]
        labels, _ = build_future_labels(x, starts, .1)
        self.assertLess(float(labels.abs().max()), 1e-12)

    def test_targets_detached_short_rollout_and_masked_loss(self):
        x = feature_fixture(4).requires_grad_()
        labels, mask = build_future_labels(x, torch.zeros(4, 4), .1)
        self.assertFalse(labels.requires_grad)
        self.assertFalse(bool(mask.any()))
        prediction = torch.ones(16, 6, requires_grad=True)
        loss, sums, counts = auxiliary_error(prediction, labels.flatten(0, 1), mask.flatten(0, 1))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0.)
        prediction = prediction.detach()
        prediction[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            require_finite(auxiliary_error(prediction, labels.flatten(0, 1), mask.flatten(0, 1)), "masked NaN")

    def test_remote_masked_mse_weights_not_six_independent_average_losses(self):
        prediction, target = torch.arange(24.).reshape(4, 6)/20., torch.zeros(4, 6)
        mask = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.bool)
        actual, _, count = auxiliary_error(prediction, target, mask)
        valid = mask.repeat_interleave(3, -1)
        expected = prediction[valid].square().mean()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(count.tolist(), [6., 6.])

    def test_head_creation_preserves_base_weights_rng_and_inference(self):
        kw = dict(architecture="mlp", obs_dim=214, act_dim=4, hidden=(16, 16, 16))
        torch.manual_seed(171)
        base = build_actor_critic(**kw)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(171)
        aux = build_actor_critic(**kw, aux_pred=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(base.state_dict(), inference_state_dict(aux.state_dict(), True))
        inputs = torch.randn(17, 214)
        actions = torch.randint(21, (17, 4))
        expected = base.evaluate_actions(inputs, actions)
        lp, entropy, prediction = aux.evaluate_actions_with_aux(inputs, actions)
        torch.testing.assert_close(lp, expected[0], atol=0, rtol=0)
        torch.testing.assert_close(entropy, expected[1], atol=0, rtol=0)
        torch.testing.assert_close(aux.value_with_aux(inputs)[0], base.get_value(inputs), atol=0, rtol=0)
        self.assertEqual(prediction.shape, (17, 6))
        with patch.object(aux.actor_aux_head, "forward", side_effect=AssertionError("inference called aux")), \
                patch.object(aux.critic_aux_head, "forward", side_effect=AssertionError("inference called aux")):
            torch.testing.assert_close(aux.act(inputs, sample=False)[0], base.act(inputs, sample=False)[0])
            torch.testing.assert_close(aux.get_value(inputs), base.get_value(inputs))

    def test_auxiliary_gradients_reach_trunk_without_actor_critic_sharing(self):
        model = build_actor_critic(architecture="mlp", obs_dim=214, hidden=(16, 16), aux_pred=True)
        actors, critics = model.actor_parameters(), model.critic_parameters()
        self.assertFalse({id(p) for p in actors} & {id(p) for p in critics})
        self.assertEqual(len({id(p) for p in actors + critics}), len(list(model.parameters())))
        x = torch.randn(11, 214)
        _, _, prediction = model.evaluate_actions_with_aux(x, torch.zeros(11, 4).long())
        ((prediction - 1)**2).mean().backward()
        self.assertGreater(float(model.actor_logits[0].weight.grad.abs().sum()), 0.)
        self.assertIsNone(model.actor_logits[-1].weight.grad)
        self.assertTrue(all(p.grad is None for p in critics))
        model.zero_grad(set_to_none=True)
        _, prediction = model.value_with_aux(x)
        (prediction - .3).square().mean().backward()
        self.assertGreater(float(model.critic[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in actors))
        cloned = copy.deepcopy(model)
        self.assertIs(cloned._actor_feature_layers[0], cloned.actor_logits[0])
        self.assertIsNot(cloned.actor_logits[0], model.actor_logits[0])

    def test_aux_on_off_initial_rollout_identical_and_ppo_off_unchanged(self):
        off = aux_trainer(False)
        old_result = off.collect_rollout()
        on = aux_trainer(True)
        new_result = on.collect_rollout()
        for key in ("b_obs", "b_act", "b_logp", "b_val", "b_rew", "b_done"):
            assert_tree_equal(getattr(off, key), getattr(on, key))
        assert_tree_equal(old_result, new_result)
        expected, mask = git_loop_labels(on.b_aux_features, on.b_done)
        self.assertTrue(torch.equal(mask, on.b_aux_mask))
        valid = mask.repeat_interleave(3, -1)
        torch.testing.assert_close(on.b_aux_labels[valid], expected[valid], atol=1e-7, rtol=1e-6)
        stats = on.update(*new_result[:2])
        require_finite(stats, "aux update")
        self.assertTrue(set(AUX_METRIC_KEYS) <= set(stats))
        self.assertEqual(stats["optimizer_steps"], 16)  # 2 epochs x 4 minibatches x 2 opts
        self.assertFalse(hasattr(off, "b_aux_features"))
        self.assertFalse(set(AUX_STATE_KEYS) & set(off.model.state_dict()))

    def test_checkpoint_exact_resume_and_protocol_mismatch_refusal(self):
        tr = aux_trainer()
        adv, ret, _ = tr.collect_rollout()
        tr.update(adv, ret)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aux.pt"
            tr.save(path)
            content = torch.load(path, weights_only=False)
            self.assertEqual(content["training_protocol"], AUX_PROTOCOL)
            self.assertEqual(content["auxiliary_contract"], AUX_CONTRACT)
            expected_rollout = tr.collect_rollout()
            labels, mask = tr.b_aux_labels.clone(), tr.b_aux_mask.clone()
            expected = tr.update(*expected_rollout[:2])
            state = tr._snapshot_learner()
            reloaded = aux_trainer()
            reloaded.load(path)
            actual_rollout = reloaded.collect_rollout()
            assert_tree_equal(expected_rollout, actual_rollout)
            assert_tree_equal(labels, reloaded.b_aux_labels)
            assert_tree_equal(mask, reloaded.b_aux_mask)
            actual = reloaded.update(*actual_rollout[:2])
            assert_tree_equal(expected, actual)
            assert_tree_equal(state, reloaded._snapshot_learner())
            for other in (aux_trainer(False), aux_trainer(True, aux_coef=.2)):
                with self.assertRaises(ValueError):
                    other.load(path)

    def test_nonfinite_features_rejected_even_if_all_targets_invalid(self):
        tr = aux_trainer(rollout_steps=1)
        tr.env.obr.aux_features[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            tr.collect_rollout()

    def test_auxiliary_contract_rejects_non_10hz_before_capture(self):
        for attribute, value in (("substeps", 3), ("dt", .05)):
            env = AuxToy()
            setattr(env if attribute == "substeps" else env.obr, attribute, value)
            with self.assertRaisesRegex(ValueError, "10Hz"):
                PPOGPUTrainer(env, PPOGPUConfig(device="cpu", aux_pred=True))
            self.assertIsNone(env.obr.aux_features)
        for coefficient in (float("nan"), float("inf"), -.1):
            with self.assertRaisesRegex(ValueError, "aux_coef"):
                aux_trainer(aux_coef=coefficient)

    def test_exploiter_restores_main_and_captures_independent_targets(self):
        tr = aux_trainer()
        tr.collect_rollout()
        before = copy.deepcopy(tr._snapshot_learner())
        runtime = tr._runtime_state()
        metrics = []
        with contextlib.redirect_stdout(io.StringIO()):
            tr.train_exploiter(metric_cb=lambda _, row: metrics.append(row))
        assert_tree_equal(before, tr._snapshot_learner())
        assert_tree_equal(runtime["obr"], tr._runtime_state()["obr"])
        self.assertEqual(len(metrics), 1)
        self.assertTrue(set(AUX_METRIC_KEYS) <= set(metrics[0]))

    def test_export_and_actual_submission_strip_aux_but_match_10hz_controls(self):
        from cuda_fdm import gpu_ckpt_to_bundle as exporter
        from cuda_fdm.search_eval import load_policy
        from cuda_fdm.tests.submission_contract_audit import CapturingProvider, _states, _context, _command_array
        from claude_code.model import make_obs_normalizer
        from claude_code import my_observation as mo
        from cuda_fdm.obs_reward import BatchObsReward
        from dogfight.unreal import ProviderCommandPolicy
        with tempfile.TemporaryDirectory() as directory:
            for depth in (2, 3, 4):
                tr = aux_trainer(hidden=(16,)*depth)
                adv, ret, _ = tr.collect_rollout()
                tr.update(adv, ret)
                path = Path(directory) / f"depth{depth}.pt"
                out = Path(directory) / f"bundle{depth}"
                tr.save(path)
                args = ["export", "--ckpt", str(path), "--output-dir", str(out)]
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    exporter.main()
                metadata = json.loads((out / "metadata.json").read_text())
                self.assertTrue(metadata["auxiliary_trained"])
                self.assertEqual(metadata["auxiliary_contract"], AUX_CONTRACT)
                policy_model, _ = load_policy(path, "cpu")
                self.assertFalse(any("aux" in k for k in policy_model.state_dict()))
                provider = CapturingProvider(out, stochastic=False, debug_obs=False)
                self.assertFalse(any("aux" in k for k in provider.model.state_dict()))
                normalization = make_obs_normalizer(provider.metadata["obs_normalization"])
                policy = ProviderCommandPolicy(provider, observation_mode=mo.OBSERVATION_MODE,
                                                observation_fn=mo.build_observation, action_repeat=6)
                policy.reset(None)
                reference = BatchObsReward(1, device="cpu", enable_kernel=False)
                last = None
                for k in range(12):
                    state = _states("maneuver", k)
                    if k:
                        reference.push_actions(torch.tensor(np.stack([last, last])))
                        reference.advance(torch.tensor(state))
                    expected_obs = reference.build_obs(torch.tensor(state))[0].numpy()
                    actual = _command_array(policy.compute_command(_context(state, k*6)))
                    np.testing.assert_allclose(provider.captured_obs, expected_obs, atol=3e-6, rtol=2e-5)
                    normalized = torch.from_numpy(normalization(expected_obs)).float().reshape(1, -1)
                    with torch.no_grad():
                        logits = tr.model.actor_logits(normalized).view(1, 4, tr.cfg.num_bins)
                        torch.testing.assert_close(policy_model.actor_logits(normalized),
                                                   tr.model.actor_logits(normalized), atol=0, rtol=0)
                        expected = torch.linspace(-1., 1., tr.cfg.num_bins)[logits.argmax(-1)][0].numpy()
                    expected[3] = expected[3]*.5 + .5
                    np.testing.assert_array_equal(actual, expected)
                    for frame in range(1, 6):
                        np.testing.assert_array_equal(
                            actual, _command_array(policy.compute_command(_context(state, k*6+frame))))
                    last = actual

    def test_strip_rejects_missing_and_unexpected_auxiliary_heads(self):
        sd = {key: torch.zeros(1) for key in AUX_STATE_KEYS}
        self.assertEqual(inference_state_dict(sd, True), {})
        with self.assertRaises(ValueError):
            inference_state_dict(sd, False)
        with self.assertRaises(ValueError):
            inference_state_dict({**sd, "actor_aux_head.surprise": torch.zeros(1)}, True)
        with self.assertRaises(ValueError):
            inference_state_dict({}, True)

    def test_csv_schema_explicit_opt_out_and_no_legacy_comparison_mix(self):
        from cuda_fdm import train_gpu
        from cuda_fdm.architecture_search import verify_training_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(train_gpu.csv_header(False) + "\n")
            train_gpu.validate_log_schema(path, False)
            with self.assertRaises(ValueError):
                train_gpu.validate_log_schema(path, True)
            checkpoint = Path(directory) / "aux.pt"
            torch.save({"cfg": {"aux_pred": True}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "auxiliary-OFF"):
                verify_training_config(checkpoint, {}, 0, False)
        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as vnext_dir:
                # 2026-09-03: this isolated 100K package always runs staged
                # vNext control from iteration 1 -- train_gpu.main() now
                # refuses to start without --active-league/--vnext-state-dir
                # regardless of what else is being exercised.
                argv = ["train_gpu", "--device", "cpu", "--no-wandb",
                       "--active-league", "--vnext-state-dir", vnext_dir,
                       "--league-dir", str(Path(vnext_dir) / "league")]
                if not enabled:
                    argv += ["--no-aux-pred"]
                fake = SimpleNamespace(train=lambda **kwargs: [])
                with patch.object(sys, "argv", argv), patch.object(train_gpu, "GpuDogfightVecEnv"), \
                        patch.object(train_gpu, "PPOGPUTrainer", return_value=fake) as factory, \
                        contextlib.redirect_stdout(io.StringIO()):
                    train_gpu.main()
                self.assertEqual(factory.call_args.args[1].aux_pred, enabled)


if __name__ == "__main__":
    unittest.main()
