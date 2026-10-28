"""CPU regressions: stable episode opponents, PFSP, attribution and recovery.

No CUDA environment, long training, W&B or existing checkpoint mutation.
"""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cuda_fdm.ppo_gpu import OpponentPool, PPOGPUConfig, PPOGPUTrainer, build_actor_critic
from cuda_fdm.tests.pool_identity_audit import ToyEnv, toy_trainer, constant_action


def result_by_id(stats, key):
    return dict(zip(stats["opponent_ids"].tolist(), stats[key].tolist()))


class StatefulToy(ToyEnv):
    """Also implements the snapshot surfaces of GpuDogfightVecEnv."""
    def __init__(self, nenv=8):
        super().__init__(nenv)
        self.rng = np.random.default_rng(19)
        self.sim = SimpleNamespace(states=torch.zeros(self.nac, 101),
                                   obs=torch.zeros(self.nac, 17), actions=torch.zeros(self.nac, 4))
        self.ic_pool = torch.zeros(2, 2, 101)
        self.obr = SimpleNamespace(clone_state=lambda: {"episode_step": self.episode_step},
                                   restore=lambda s: setattr(self, "episode_step", s["episode_step"]))


def stateful_trainer(architecture="mlp"):
    cfg = PPOGPUConfig(device="cpu", architecture=architecture, hidden=(8, 8, 8),
                       gru_size=4, num_bins=3, rollout_steps=1, recurrent_seq_len=1,
                       update_epochs=1, num_minibatches=2, opp_sample=False,
                       normalize_obs=True, save_runtime=True, pool_evict_cap=1,
                       sched_period=0, milestone_period=0, exploiter_iters=1,
                       exploiter_win_target=1.0, selfplay_gate_threshold=1.1)
    return PPOGPUTrainer(StatefulToy(), cfg)


class PoolEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_eviction_preserves_all_live_actions_and_exact_result_identity(self):
        tr = toy_trainer(cap=1)
        first_id = tr.pool.entries[0]["id"]
        tr.collect_rollout()
        before = tr.opp_assign.clone()
        constant_action(tr.model, 2)
        second_id = tr.pool.add(tr.model, None, permanent=False)
        tr._refresh_weights()
        self.assertTrue(torch.equal(before, tr.opp_assign))
        self.assertEqual(tr.pool.size(), 1)
        self.assertEqual(tr.pool.resident_size(), 2)
        self.assertEqual(float(tr.opp_weights[0]), 0.)
        tr.collect_rollout()
        _, _, terminal = tr.collect_rollout()
        self.assertEqual(result_by_id(terminal, "win_by_opp"), {first_id: 8., second_id: 0.})
        self.assertTrue(all(torch.equal(a, tr.env.opponent_controls[0])
                            for a in tr.env.opponent_controls))
        tr.pool.update_emas(terminal["win_by_opp"], terminal["loss_by_opp"],
                            terminal["ep_by_opp"], 1., terminal["opponent_ids"])
        self.assertEqual(tr.pool.entries[0]["ema"], 1.)
        self.assertEqual(tr.pool.entries[1]["ema"], .5)
        tr._refresh_weights()
        self.assertEqual([e["id"] for e in tr.pool.entries], [second_id])
        tr.collect_rollout()
        self.assertTrue(bool((tr.opp_assign == second_id).all()))
        self.assertTrue(bool((tr.env.opponent_controls[-1] == 1.).all()))
        self.assertEqual(tr.env.reset_calls, 1)

    def test_append_and_weight_updates_do_not_reassign_live_games(self):
        tr = toy_trainer(cap=4, nenv=64)
        tr.collect_rollout()
        before = tr.opp_assign.clone()
        tr.pool.add(tr.model, None, permanent=True)
        tr._refresh_weights()
        self.assertTrue(torch.equal(before, tr.opp_assign))
        tr.pool.entries[0]["ema"], tr.pool.entries[1]["ema"] = .9, .1
        tr._refresh_weights()
        tr.collect_rollout()
        self.assertTrue(torch.equal(before, tr.opp_assign))

    def test_partial_done_and_repeated_fifo_keep_all_remaining_identities(self):
        tr = toy_trainer(cap=1)
        tr.collect_rollout()
        initial = tr.pool.entries[0]["id"]
        second = tr.pool.add(tr.model, None, permanent=False)
        tr._next_done[:4] = 1
        tr._refresh_weights()
        tr.collect_rollout()
        self.assertEqual(tr.opp_assign.tolist(), [second] * 4 + [initial] * 4)
        third = tr.pool.add(tr.model, None, permanent=False)
        tr._refresh_weights()
        self.assertEqual(tr.pool.size(), 1)
        self.assertEqual(tr.pool.resident_size(), 3)
        _, _, stats = tr.collect_rollout()
        self.assertEqual(result_by_id(stats, "ep_by_opp"), {initial: 4., second: 4., third: 0.})
        tr._refresh_weights()
        self.assertEqual([e["id"] for e in tr.pool.entries], [third])

    def test_original_pfsp_formula_and_gate_use_only_active_entries(self):
        tr = toy_trainer(cap=2)
        tr.pool.add(tr.model, None, permanent=True, ema=.1)
        tr.pool.add(tr.model, None, permanent=False, ema=.9)
        tr.pool.add(tr.model, None, permanent=False, ema=.8)
        active = tr.pool.active_entries()
        emas = np.array([e["ema"] for e in active])
        p = np.exp(-(emas - emas.min()) / .3)
        p = .5 / len(active) + .5 * p / p.sum()
        w = tr.pool.weights(.3, .5).numpy()
        np.testing.assert_allclose(w[[not e["retired"] for e in tr.pool.entries]], p, rtol=1e-6)
        self.assertEqual(w[0], 0.)
        tr.pool.entries[0]["ema"] = 0.  # retired weak opponent must not block gate
        self.assertTrue(tr.pool.gate_and_add(tr.model, None, .75))
        self.assertEqual(tr.pool.size(), 3)  # two evictable + one permanent
        self.assertEqual(tr.pool.num_permanent(), 1)

    def test_retired_gru_hidden_and_actions_match_uninterrupted_reference(self):
        kwargs = dict(obs_dim=4, act_dim=4, num_bins=3, architecture="gru",
                      hidden=(8, 8, 8), gru_size=4, encoder_depth=2)
        model = build_actor_critic(**kwargs)
        pool = OpponentPool(kwargs, "cpu", evict_cap=1, sample=False)
        identity = pool.add(model, None, permanent=False)
        obs, assign, starts = torch.randn(8, 4), torch.zeros(8, dtype=torch.long), torch.zeros(8)
        pool.act(obs, assign, starts)
        old = pool.entries[0]
        state = tuple(h.clone() for h in old["actor_state"])
        pool.add(model, None, permanent=False)
        starts[:4] = 1
        pool.refresh_residents(assign, starts)
        assign[:4] = pool.entries[1]["id"]
        with torch.no_grad():
            expected, hidden = old["net"].act(obs[4:], tuple(h[:, 4:] for h in state),
                                              starts[4:], sample=False)
        got = pool.act(obs, assign, starts)
        self.assertTrue(torch.equal(expected, got[4:]))
        for h, expected_h in zip(old["actor_state"], hidden):
            torch.testing.assert_close(h[:, 4:], expected_h, rtol=0, atol=0)
        self.assertEqual(old["id"], identity)

    def test_save_resume_with_retired_live_episodes_is_exact_for_mlp_and_gru(self):
        for architecture in ("mlp", "gru"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as tmp:
                tr = stateful_trainer(architecture)
                tr.collect_rollout()
                tr.pool.add(tr.model, tr.norm, permanent=False)
                tr._refresh_weights()
                path = Path(tmp) / "recovery.pt"
                tr.save(path)
                a, r, stats = tr.collect_rollout()
                expected = {n: getattr(tr, n).clone() for n in ("b_act", "b_obs", "b_rew", "opp_assign")}
                tr.load(path)
                aa, rr, actual_stats = tr.collect_rollout()
                for n, value in expected.items():
                    torch.testing.assert_close(value, getattr(tr, n), rtol=0, atol=0)
                torch.testing.assert_close(a, aa, rtol=0, atol=0)
                torch.testing.assert_close(r, rr, rtol=0, atol=0)
                self.assertEqual(result_by_id(stats, "ep_by_opp"), result_by_id(actual_stats, "ep_by_opp"))
                self.assertEqual(tr.pool.next_id, 2)

    def test_exploiter_restores_live_main_and_failure_also_restores(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                tr = stateful_trainer("mlp")
                tr.collect_rollout()
                learner = copy.deepcopy(tr.model.state_dict())
                before = tr._runtime_state()
                global_step = tr.global_step
                if fail:
                    def invalid(*args, **kwargs):
                        raise FloatingPointError("injected exploiter failure")
                    tr.update = invalid
                    with self.assertRaises(FloatingPointError):
                        tr.train_exploiter()
                else:
                    tr.train_exploiter()
                after = tr._runtime_state()
                for key in before["trainer"]:
                    if key != "opp_weights":  # new opponent changes future sampling, not live games
                        torch.testing.assert_close(before["trainer"][key], after["trainer"][key], rtol=0, atol=0)
                self.assertEqual(before["obr"], after["obr"])
                self.assertTrue(torch.equal(before["sim"], after["sim"]))
                for identity, state in before["pool_h"].items():
                    for x, y in zip(state, after["pool_h"][identity]):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
                self.assertEqual(tr.global_step, global_step)
                self.assertTrue(all(torch.equal(v, tr.model.state_dict()[k]) for k, v in learner.items()))
                self.assertEqual(tr.pool.num_permanent(), 0 if fail else 1)

    def test_invalid_checkpoint_cannot_overwrite_healthy_and_legacy_resume_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = stateful_trainer()
            path = Path(tmp) / "recovery.pt"
            tr.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with torch.no_grad():
                next(tr.model.parameters()).flatten()[0] = float("nan")
            with self.assertRaises(FloatingPointError):
                tr.save(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            healthy = torch.load(path, map_location="cpu", weights_only=False)
            healthy.pop("training_protocol")
            old_path = Path(tmp) / "legacy.pt"
            torch.save(healthy, old_path)
            with self.assertRaisesRegex(ValueError, "protocol"):
                tr.load(old_path)

    def test_corrupt_live_id_and_duplicate_serialized_ids_rejected(self):
        tr = stateful_trainer()
        runtime = tr._runtime_state()
        runtime["trainer"]["opp_assign"][0] = 100
        with self.assertRaisesRegex(ValueError, "missing opponent"):
            tr._restore_runtime(runtime)
        data = tr.pool.state_dicts()
        with self.assertRaisesRegex(ValueError, "unique stable"):
            tr.pool.load_state_dicts(data + data)


if __name__ == "__main__":
    unittest.main()
