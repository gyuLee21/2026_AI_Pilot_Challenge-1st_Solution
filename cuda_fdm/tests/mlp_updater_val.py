"""Independent flat-PPO reference and MLP-only updater regressions (CPU)."""
import copy
import contextlib
import inspect
import io
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cuda_fdm.ppo_gpu import (LEGACY_TRAINING_PROTOCOL, PPOGPUConfig,
                              PPOGPUTrainer, TRAINING_PROTOCOL)
from cuda_fdm.tests.pool_identity_audit import ToyEnv


def trainer_fixture(rollout=7, nenv=5, epochs=3, minibatches=4, architecture="mlp"):
    cfg = PPOGPUConfig(device="cpu", architecture=architecture, hidden=(16, 16, 16),
                       gru_size=4, rollout_steps=rollout, update_epochs=epochs,
                       num_minibatches=minibatches, normalize_obs=False, target_kl=None,
                       seed=127, ent_coef=.01, sched_period=0, milestone_period=0,
                       exploiter_iters=0)
    tr = PPOGPUTrainer(ToyEnv(nenv), cfg)
    torch.manual_seed(139)
    tr.b_obs.normal_()
    tr.b_act.random_(0, cfg.num_bins)
    with torch.no_grad():
        if architecture == "mlp":
            lp, _ = tr.model.evaluate_actions(tr.b_obs.flatten(0, 1), tr.b_act.flatten(0, 1))
            tr.b_logp.copy_(lp.reshape(rollout, nenv))
            tr.b_val.copy_(tr.model.get_value(tr.b_obs.flatten(0, 1)).view(rollout, nenv))
    return tr, torch.randn(rollout, nenv), torch.randn(rollout, nenv)


def flat_reference(tr, advantage, returns, permutations):
    """Test-only standard PPO, independent of production batching/evaluation API.

    The same explicit transition permutations allow a numerical correctness
    comparison, as opposed to comparing two different optimizer trajectories.
    """
    c = tr.cfg
    x, a = tr.b_obs.flatten(0, 1), tr.b_act.flatten(0, 1)
    old_lp, adv, target = tr.b_logp.flatten(), advantage.flatten(), returns.flatten()
    width = (len(x) + c.num_minibatches - 1) // c.num_minibatches
    last = {}
    for permutation in permutations:
        kl_values = []
        for ix in permutation.split(width):
            logits = tr.model.actor_logits(x[ix]).view(-1, tr.act_dim, c.num_bins)
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(a[ix]).sum(-1)
            entropy = distribution.entropy().sum(-1).mean()
            ratio = (logp - old_lp[ix]).exp()
            ga = adv[ix]
            if c.norm_adv and ga.numel() > 1:
                ga = (ga - ga.mean()) / (ga.std() + 1e-8)
            loss = torch.maximum(-ga * ratio, -ga * ratio.clamp(1-c.clip_coef, 1+c.clip_coef)).mean()
            tr.actor_opt.zero_grad(set_to_none=True)
            (loss - c.ent_coef * entropy).backward()
            ag = nn.utils.clip_grad_norm_(tr.model.actor_parameters(), c.max_grad_norm)
            tr.actor_opt.step()
            prediction = tr.model.get_value(x[ix])
            vloss = ((prediction - target[ix]) ** 2).mean() * .5
            tr.critic_opt.zero_grad(set_to_none=True)
            (vloss * c.vf_coef).backward()
            cg = nn.utils.clip_grad_norm_(tr.model.critic_parameters(), c.max_grad_norm)
            tr.critic_opt.step()
            kl_values.append(((ratio - 1) - (logp - old_lp[ix])).detach().mean())
            last = dict(pl=loss.detach(), vl=vloss.detach(), ent=entropy.detach(),
                        cf=((ratio.detach()-1).abs() > c.clip_coef).float().mean(),
                        actor_grad_norm=ag.detach(), critic_grad_norm=cg.detach())
        last["kl"] = torch.stack(kl_values).mean()
        if c.target_kl is not None and float(last["kl"]) > c.target_kl:
            break
    return last


class MLPUpdaterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_training_is_mlp_without_sequence_api_or_buffers(self):
        tr, _, _ = trainer_fixture()
        self.assertEqual(PPOGPUConfig().architecture, "mlp")
        self.assertEqual(LEGACY_TRAINING_PROTOCOL,
                         "cuda_mlp_flat_finite_horizon_diverse_h3_reset_v7")
        self.assertNotEqual(TRAINING_PROTOCOL, LEGACY_TRAINING_PROTOCOL)
        for name in ("b_actor_h", "b_critic_h", "_pack_recurrent_batch"):
            self.assertFalse(hasattr(tr, name), name)
        self.assertFalse(hasattr(tr.model, "evaluate_actions_packed_sequence"))
        self.assertFalse(hasattr(tr.model, "evaluate_values_packed_sequence"))
        source = inspect.getsource(PPOGPUTrainer.update)
        for text in (".cpu(", ".numpy(", "b_done", "_pack_", "hidden", "episode_start"):
            # Documentation mentions hidden-state removal; executable body must not.
            self.assertNotIn(text, source[source.index("        if self.model.is_recurrent:"):])

    def test_independent_reference_same_permutations_and_nonzero_adam_state(self):
        tr, adv, ret = trainer_fixture()
        tr.update(adv, ret)  # real nonzero optimizer moments, not just fresh Adam
        saved = tr._snapshot_learner()
        count = tr.b_logp.numel()
        torch.manual_seed(173)
        permutations = [torch.randperm(count) for _ in range(tr.cfg.update_epochs)]
        with patch("torch.randperm", side_effect=[p.clone() for p in permutations]):
            actual = tr.update(adv, ret)
        actual_weights = copy.deepcopy(tr.model.state_dict())
        actual_opts = [copy.deepcopy(o.state_dict()) for o in (tr.actor_opt, tr.critic_opt)]
        tr._restore_learner(copy.deepcopy(saved))
        expected = flat_reference(tr, adv, ret, permutations)
        for name, value in actual_weights.items():
            torch.testing.assert_close(value, tr.model.state_dict()[name], atol=2e-7, rtol=2e-6)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, atol=1e-7, rtol=1e-6)
        for old, new in zip(actual_opts, (tr.actor_opt.state_dict(), tr.critic_opt.state_dict())):
            for key, values in old["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, new["state"][key][name], atol=2e-7, rtol=2e-6)
        self.assertEqual(actual["optimizer_steps"], 24)  # 3 epochs x 4 minibatches x 2 opts

    def test_each_transition_once_per_epoch_and_shared_actor_critic_observation(self):
        tr, adv, ret = trainer_fixture()
        count = tr.b_logp.numel()
        tr.b_obs[:, :, 0] = torch.arange(count).view_as(adv)
        seen, actor_ptr, critic_ptr = [], [], []
        old_actor, old_value = tr.model.evaluate_actions, tr.model.get_value
        def actor(x, a):
            seen.append(x[:, 0].clone())
            actor_ptr.append(x.data_ptr())
            return old_actor(x, a)
        def critic(x):
            critic_ptr.append(x.data_ptr())
            return old_value(x)
        with patch.object(tr.model, "evaluate_actions", side_effect=actor), \
                patch.object(tr.model, "get_value", side_effect=critic):
            result = tr.update(adv, ret)
        self.assertEqual(actor_ptr, critic_ptr)
        for epoch in range(tr.cfg.update_epochs):
            got = torch.cat(seen[epoch*4:epoch*4+4]).long().sort().values
            torch.testing.assert_close(got, torch.arange(count), atol=0, rtol=0)
        self.assertEqual(result["optimizer_steps"], 24)  # 3 epochs x 4 minibatches x 2 opts

    def test_episode_markers_cannot_repack_flat_update(self):
        tr, adv, ret = trainer_fixture()
        saved = tr._snapshot_learner()
        for start in (0., 1.):
            tr._restore_learner(copy.deepcopy(saved))
            tr.b_done.fill_(start)
            torch.manual_seed(191)
            tr.update(adv, ret)
            if start == 0.:
                expected = copy.deepcopy(tr.model.state_dict())
            else:
                for key, value in tr.model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_kl_stop_remains_at_epoch_end(self):
        tr, adv, ret = trainer_fixture()
        tr.cfg.target_kl = -1.  # force the first epoch-end check to stop
        result = tr.update(adv, ret)
        self.assertTrue(result["early"])
        self.assertEqual(result["epochs"], 1)
        self.assertEqual(result["optimizer_steps"], 8)  # one epoch (4 minibatches x 2 opts), not one minibatch

    def test_singleton_and_partial_final_minibatches_are_finite(self):
        for rollout, nenv, mb in ((1, 1, 8), (3, 3, 4)):
            with self.subTest(shape=(rollout, nenv), minibatches=mb):
                tr, adv, ret = trainer_fixture(rollout, nenv, minibatches=mb)
                result = tr.update(adv, ret)
                self.assertTrue(all(bool(torch.isfinite(x).all()) for x in result.values()
                                    if torch.is_tensor(x)))
                self.assertTrue(all(bool(torch.isfinite(x).all()) for x in tr.model.state_dict().values()))

    def test_gru_cannot_silently_train_with_history_dropped(self):
        tr, adv, ret = trainer_fixture(architecture="gru")
        before = copy.deepcopy(tr.model.state_dict())
        with self.assertRaisesRegex(ValueError, "MLP-only"):
            tr.update(adv, ret)
        self.assertFalse(tr.actor_opt.state)
        for name, value in before.items():
            torch.testing.assert_close(value, tr.model.state_dict()[name], atol=0, rtol=0)

    def test_gru_training_cli_rejected_before_gpu_initialization(self):
        from cuda_fdm import train_gpu
        with patch.object(sys, "argv", ["train_gpu", "--architecture", "gru"]), \
                patch("torch.zeros") as allocate, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failed:
                train_gpu.main()
            self.assertEqual(failed.exception.code, 2)
            allocate.assert_not_called()

    def test_padded_v3_checkpoint_cannot_silently_resume_with_flat_shuffling(self):
        import tempfile
        from pathlib import Path
        tr, _, _ = trainer_fixture()
        with tempfile.TemporaryDirectory() as folder:
            current = Path(folder) / "current.pt"
            tr.save(current)
            data = torch.load(current, weights_only=False)
            data["training_protocol"] = "cuda_episode_opponent_finite_horizon_v3"
            old = Path(folder) / "old.pt"
            torch.save(data, old)
            before = copy.deepcopy(tr.model.state_dict())
            with self.assertRaisesRegex(ValueError, "minibatch"):
                tr.load(old)
            for name, value in before.items():
                torch.testing.assert_close(value, tr.model.state_dict()[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
