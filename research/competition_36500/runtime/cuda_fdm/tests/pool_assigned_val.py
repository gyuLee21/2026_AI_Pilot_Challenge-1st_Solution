"""CPU checks and a frozen pre-subset reference for opponent inference.

The reference below is the old OpponentPool.act body, not a production fallback.
Integration/performance tests also use it to isolate only this implementation
change from reward, updater, PFSP and episode-identity changes.
"""
import copy
import unittest

import numpy as np
import torch

from cuda_fdm.ppo_gpu import OpponentPool, RunningNorm, build_actor_critic


@torch.no_grad()
def legacy_full_act(self, opp_obs, assign, episode_start):
    """Frozen full-active-batch reference; retirees already used fixed lanes."""
    outs = []
    for e in self.entries:
        if e.get("actor_state") is None or e["actor_state"][0].shape[1] != opp_obs.shape[0]:
            e["actor_state"] = e["net"].initial_state(opp_obs.shape[0], self.device)
        lanes = e.get("retired_lanes") if e["retired"] else None
        oo = opp_obs if lanes is None else opp_obs.index_select(0, lanes)
        on = e["norm"].normalize(oo) if e["norm"] is not None else oo
        hidden = e["actor_state"]
        starts = episode_start
        if lanes is not None:
            hidden = tuple(h.index_select(1, lanes) for h in hidden)
            starts = None if starts is None else starts.index_select(0, lanes)
        action, state = e["net"].act(
            on, state=hidden, episode_start=starts,
            sample=self.sample)
        if lanes is None:
            e["actor_state"] = tuple(x.detach() for x in state)
        else:
            for full, part in zip(e["actor_state"], state):
                full.index_copy_(1, lanes, part.detach())
            full_action = torch.zeros(opp_obs.shape[0], action.shape[-1],
                                      dtype=action.dtype, device=action.device)
            full_action.index_copy_(0, lanes, action)
            action = full_action
        outs.append(action)
    stacked = torch.stack(outs, 0)
    idx = self.rows_for_ids(assign).view(1, -1, 1).expand(1, opp_obs.shape[0], 4)
    return stacked.gather(0, idx).squeeze(0)


def make_pool(count=4, architecture="mlp", cap=None):
    torch.manual_seed(901)
    kwargs = dict(obs_dim=9, act_dim=4, num_bins=7, architecture=architecture,
                  hidden=(16, 16, 16), gru_size=8, encoder_depth=2)
    pool = OpponentPool(kwargs, "cpu", evict_cap=count if cap is None else cap)
    for i in range(count):
        net = build_actor_critic(**kwargs)
        norm = RunningNorm(9, "cpu") if i % 2 == 0 else None
        if norm is not None:
            norm.mean.fill_(i * .21)
            norm.var.fill_(1. + i * .13)
            norm.count.fill_(13. + i)
        pool.add(net, norm, permanent=False, ema=.1 + .1 * i)
    return pool


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=False)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right, (left, right)


class AssignedPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def compare(self, pool, obs, assignments):
        for sampled in (False, True):
            old, new = copy.deepcopy(pool), copy.deepcopy(pool)
            old.sample = new.sample = sampled
            for step, assign in enumerate(assignments):
                starts = None if step == 0 else (torch.arange(len(obs)) % 3 == 0).float()
                torch.manual_seed(910 + step)
                rng = torch.get_rng_state()
                expected = legacy_full_act(old, obs, assign, starts)
                end_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                actual = new.act(obs, assign, starts)
                assert_tree_equal(expected, actual)
                assert_tree_equal(end_rng, torch.get_rng_state())
                assert_tree_equal([e["actor_state"] for e in old.entries],
                                  [e["actor_state"] for e in new.entries])
            assert_tree_equal(old.state_dicts(), new.state_dicts())

    def test_multiple_counts_skew_zero_lanes_and_noncontiguous_observations(self):
        obs = torch.randn(37, 18)[:, ::2]
        self.assertFalse(obs.is_contiguous())
        for count in (1, 2, 4, 6):
            with self.subTest(count=count):
                a = torch.arange(len(obs)) % count
                skew = a.clone()
                skew[:33] = 0
                self.compare(make_pool(count), obs,
                             [a, a.roll(5), skew, torch.zeros_like(a)])

    def test_active_network_and_norm_see_only_assigned_rows(self):
        pool = make_pool(4)
        obs = torch.randn(37, 9)
        assign = torch.arange(len(obs)) % 3  # entry 3 has no assigned rows
        calls, handles = {}, []
        for e in pool.entries:
            def record(module, inputs, identity=e["id"]):
                calls.setdefault(identity, []).append(inputs[0].clone())
            handles.append(e["net"].actor_logits.register_forward_pre_hook(record))
        try:
            pool.act(obs, assign, torch.zeros(len(obs)))
        finally:
            for handle in handles:
                handle.remove()
        self.assertNotIn(3, calls)
        self.assertEqual(sum(v[0].shape[0] for v in calls.values()), len(obs))
        for e in pool.entries[:3]:
            expected = obs[assign == e["id"]]
            if e["norm"] is not None:
                expected = e["norm"].normalize(expected)
            assert_tree_equal(calls[e["id"]], [expected])

    def test_noncontiguous_stable_ids_and_retired_remaining_lanes(self):
        pool = make_pool(6, cap=2)
        # Retire 0..3, keep live retired ID 2, remove IDs 0/1/3.
        assign = torch.tensor([2, 4, 5] * 9)
        starts = torch.zeros(len(assign))
        pool.refresh_residents(assign, starts)
        self.assertEqual([e["id"] for e in pool.entries], [2, 4, 5])
        after_done = assign.clone()
        after_done[:6] = 5
        self.compare(pool, torch.randn(len(assign), 9), [assign, after_done])

    def test_logits_on_assigned_rows_match_full_inference(self):
        pool = make_pool(4)
        obs = torch.randn(37, 9)
        assign = torch.arange(len(obs)) % 4
        with torch.no_grad():
            for e in pool.entries:
                norm = e["norm"]
                full_obs = norm.normalize(obs) if norm else obs
                selected = torch.nonzero(assign == e["id"]).flatten()
                part_obs = obs.index_select(0, selected)
                part_obs = norm.normalize(part_obs) if norm else part_obs
                expected = e["net"].actor_logits(full_obs).index_select(0, selected)
                actual = e["net"].actor_logits(part_obs)
                torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)

    def test_legacy_gru_inference_hidden_and_sampling_unchanged(self):
        pool = make_pool(3, architecture="gru")
        assign = torch.arange(17) % 3
        self.compare(pool, torch.randn(17, 9), [assign, assign.roll(2), assign])

    def test_pfsp_metadata_assignments_and_normalization_remain_immutable(self):
        pool = make_pool(4)
        before = copy.deepcopy(pool.state_dicts())
        weights = pool.weights(.3, .5).clone()
        assign = torch.arange(37) % 4
        assigned_before = assign.clone()
        for _ in range(5):
            pool.act(torch.randn(37, 9), assign, torch.ones(37))
        assert_tree_equal(before, pool.state_dicts())
        assert_tree_equal(weights, pool.weights(.3, .5))
        assert_tree_equal(assign, assigned_before)

    def test_assigned_nonfinite_logits_fail_for_sampled_and_greedy(self):
        for sampled in (False, True):
            pool = make_pool(2)
            pool.sample = sampled
            obs = torch.zeros(8, 9)
            obs[0, 0] = float("nan")
            with self.assertRaises(ValueError):
                pool.act(obs, torch.arange(8) % 2, torch.zeros(8))


if __name__ == "__main__":
    unittest.main()
