"""CPU-only diagnostic of the 200-second game boundary; no learner edits/update.

Uses the real CUDA trainer's collect_rollout/GAE with a tiny CPU environment,
the actual reward references, and the actual termination predicate. The output
distinguishes successful reproduction from a correct finite-horizon objective.
It never resumes an experiment or launches CUDA work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from cuda_fdm import obs_reward as OR
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv
from dogfight.envs.termination import evaluate_termination, time_limit_reached
from dogfight.sim.state_schema import StateIndex


class BoundaryEnv:
    OBS_SIZE = 4
    min_altitude_m = 304.8

    def __init__(self, next_episode_reward=0.0, fresh_value=100.0):
        self.nenv = 5
        self.nac = 10
        self.next_episode_reward = float(next_episode_reward)
        self.fresh_value = float(fresh_value)

    def reset(self, stagger=True):
        del stagger
        self.calls = 0
        return torch.full((self.nenv, 2, self.OBS_SIZE), 2.0)

    def step(self, controls):
        del controls
        self.calls += 1
        first = self.calls == 1
        # Lanes 0..2: timeout win/loss/draw. Lane 3: destruction. Lane 4: live.
        done = torch.tensor([first, first, first, first, False])
        trunc = torch.tensor([first, first, first, False, False])
        term = done & ~trunc
        terminal_obs = torch.full((5, 2, 4), 10.0 if first else self.fresh_value)
        obs = terminal_obs.clone()
        obs[done] = self.fresh_value
        reward = torch.zeros(5, 2)
        if first:
            reward[:, 0] = torch.tensor([5.0, -5.0, -4.0, 5.0, 1.0])
        else:
            reward[:, 0] = self.next_episode_reward
        hp = torch.tensor([[1.0, 0.8], [0.8, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        return obs, reward, done, {
            "terminated": term,
            "truncated": trunc,
            "terminal_obs": terminal_obs,
            "terminal_hp": hp,
            "terminal_alt_m": torch.full((5, 2), 3000.0),
            "terminal_state_finite": torch.ones(5, dtype=torch.bool),
        }


def collect_case(architecture="mlp", rollout_steps=1, next_reward=0.0, fresh_value=100.0,
                 expect_legacy=False):
    cfg = PPOGPUConfig(
        device="cpu", architecture=architecture, hidden=(8, 8, 8),
        gru_size=4 if architecture == "gru" else 0, num_bins=3,
        normalize_obs=False, rollout_steps=rollout_steps,
        gamma=0.997, gae_lambda=0.95, sched_period=0,
        milestone_period=0, exploiter_iters=0, opp_sample=False,
    )
    tr = PPOGPUTrainer(BoundaryEnv(next_reward, fresh_value), cfg)

    def diagnostic_value(self, obs, state=None, episode_start=None):
        del episode_start
        return obs[:, 0].clone(), state

    # Critic predictions are deliberately known, while using the real collector
    # and GAE; no optimizer step and no replacement GAE implementation.
    tr.model.value_step = MethodType(diagnostic_value, tr.model)
    adv, ret, stats = tr.collect_rollout()
    actual = ret[0].tolist()
    expected_finite_horizon = [5.0, -5.0, -4.0, 5.0, 10.97]
    if rollout_steps == 1:
        expected = [14.97, 4.97, 5.97, 5.0, 10.97] if expect_legacy else expected_finite_horizon
        expected_rewards = [14.97, 4.97, 5.97, 5.0, 1.0] if expect_legacy else [5., -5., -4., 5., 1.]
        assert np.allclose(actual, expected, atol=2e-6)
        assert np.allclose(tr.b_rew[0].tolist(), expected_rewards, atol=2e-6)
        assert float(stats["ret_sum"]) == 1.0  # +5 -5 -4 +5, raw logger
    return {
        "architecture": architecture,
        "rollout_steps": rollout_steps,
        "gamma": cfg.gamma,
        "terminal_critic_prediction": 10.0,
        "fresh_episode_critic_prediction": fresh_value,
        "first_step_raw_rewards": [5.0, -5.0, -4.0, 5.0, 1.0],
        "first_step_learner_rewards": tr.b_rew[0].tolist(),
        "first_step_gae_targets": actual,
        "finite_horizon_expected_if_last_rollout_step": expected_finite_horizon,
        "first_step_advantages": adv[0].tolist(),
        "first_step_episode_start_masks": tr.b_done[0].tolist(),
        "timeout_target_error": [actual[i] - expected_finite_horizon[i] for i in range(3)],
    }


def reward_and_clock(expect_legacy=False):
    cfg = dict(OR.RW.MY_REWARD_CONFIG)
    cfg["shaping_reward_scale"] = 0.0  # isolate the actual terminal and damage terms
    hp_cases = [(1.0, .8), (.8, 1.0), (1.0, 1.0)]
    bor = OR.BatchObsReward(3, device="cpu", enable_kernel=False)
    s9 = torch.zeros(6, 9, dtype=torch.float64)
    s9[:, 2] = -3000.0
    s9[:, 6] = 250.0
    s9[1::2, 0] = 1000.0
    bor.hp.copy_(torch.tensor(hp_cases, dtype=torch.float64).flatten())
    bor.t_sec.fill_(200.1)
    term = torch.zeros(3, dtype=torch.bool)
    trunc = torch.ones(3, dtype=torch.bool)
    tensor_rewards = bor.compute_reward(s9, term, cfg, trunc).view(3, 2)
    assert np.allclose(tensor_rewards.tolist(), [[5., -5.], [-5., 5.], [-4., -4.]])
    cpu_rewards = []
    for own_hp, enemy_hp in hp_cases:
        own = np.zeros(46, dtype=np.float64)
        enemy = own.copy()
        own[StateIndex.ALT] = enemy[StateIndex.ALT] = 3000.0
        own[StateIndex.SIM_TIME] = enemy[StateIndex.SIM_TIME] = 200.1
        own[StateIndex.HEALTH], enemy[StateIndex.HEALTH] = own_hp, enemy_hp
        own[6] = enemy[6] = 250.0
        OR.RW.reset_distance_tracker()
        reward, parts = OR.RW.compute_reward(
            own, enemy, 0.0, 0.0, None, {}, cfg, False, True, "max time out")
        cpu_rewards.append({"own_hp": own_hp, "enemy_hp": enemy_hp,
                            "reward": reward, "terminal": parts["terminal"]})
    assert np.allclose([x["reward"] for x in cpu_rewards], [5., -5., -4.])

    bor.hp_loss[1] = 0.02  # final-step damage is still paid alongside terminal reward
    final_damage_rewards = bor.compute_reward(s9, term, cfg, trunc).view(3, 2)
    assert np.allclose(final_damage_rewards[0].tolist(), [5.2, -5.2])

    # Invoke the real CUDA-env termination method without constructing a GPU env.
    shell = GpuDogfightVecEnv.__new__(GpuDogfightVecEnv)
    shell.nenv = 1
    shell.min_altitude_m = 304.8
    shell.max_engage_time_s = 200.0
    shell.obr = SimpleNamespace(hp=torch.ones(2), t_sec=torch.zeros(1, dtype=torch.float64))
    own = np.zeros(46)
    enemy = own.copy()
    own[StateIndex.ALT] = enemy[StateIndex.ALT] = 3000.0
    own[StateIndex.HEALTH] = enemy[StateIndex.HEALTH] = 1.0
    own[StateIndex.FUEL] = enemy[StateIndex.FUEL] = 1.0
    sim = SimpleNamespace(fdm_update_success=True)
    clock = []
    for timestamp in (199.9, 200.0, 200.0000000001, 200.1):
        shell.obr.t_sec.fill_(timestamp)
        flags = shell._termination(s9[:2])
        own[StateIndex.SIM_TIME] = enemy[StateIndex.SIM_TIME] = timestamp
        cpu_flags = evaluate_termination(own, enemy, sim, sim, 200., 300., 0, None)
        clock.append({"time": timestamp,
                      "cuda_env_reference_terminated": bool(flags[0][0]),
                      "cuda_env_reference_truncated": bool(flags[1][0]),
                      "cpu_terminated": bool(cpu_flags[0]), "cpu_truncated": bool(cpu_flags[1]),
                      "reason": cpu_flags[2]})
    assert clock[1]["cpu_truncated"] is (not expect_legacy)
    assert clock[1]["cuda_env_reference_truncated"] is (not expect_legacy)
    assert clock[-1]["cpu_truncated"] and clock[-1]["cuda_env_reference_truncated"]

    duration = 0.0
    steps = 0
    while duration <= 200.0:
        duration += float(bor.dt)
        steps += 1
    fixed_duration = 0.0
    fixed_steps = 0
    while not time_limit_reached(fixed_duration, 200.0):
        fixed_duration += float(bor.dt)
        fixed_steps += 1
    return {
        "cpu_reward_cases": cpu_rewards,
        "torch_reference_rewards_both_sides": tensor_rewards.tolist(),
        "last_step_damage_plus_timeout_reward": final_damage_rewards[0].tolist(),
        "boundary_flags": clock,
        "dt": bor.dt,
        "double_accumulation_until_strict_greater": {"steps": steps, "time": duration},
        "current_inclusive_horizon": {"steps": fixed_steps, "time": fixed_duration},
        "geometry_theoretical_abs_bound_at_current_duration": 5.0 * fixed_duration / 200.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-legacy-defect", action="store_true",
                        help="Only for a restored historical implementation, never a passing current check")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing audit evidence; choose a new output: {args.output}")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    core_files = ("cuda_fdm/ppo_gpu.py", "cuda_fdm/rl_env.py", "cuda_fdm/gen/obs_kernel.cu",
                  "cuda_fdm/obs_reward.py", "claude_code/my_reward.py", "claude_code/ppo.py",
                  "src/dogfight/envs/termination.py")
    hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in core_files}
    cases = [collect_case("mlp", expect_legacy=args.expect_legacy_defect),
             collect_case("gru", expect_legacy=args.expect_legacy_defect)]
    normal_next_episode = collect_case(rollout_steps=2, next_reward=0., fresh_value=100.)
    extreme_next_episode = collect_case(rollout_steps=2, next_reward=500., fresh_value=999.)
    assert np.allclose(normal_next_episode["first_step_gae_targets"][:4],
                       extreme_next_episode["first_step_gae_targets"][:4], atol=2e-6)
    result = {
        "audit_completed": True,
        "finite_horizon_value_handling_correct": not args.expect_legacy_defect,
        "reproduced_unwanted_timeout_bootstrap": args.expect_legacy_defect,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
        "production_code_changed_by_audit": False,
        "cases": cases,
        "gae_does_not_cross_episode_reset": True,
        "reset_value_and_next_episode_reward_perturbation": {
            "baseline": normal_next_episode,
            "perturbed": extreme_next_episode,
        },
        "reward_and_clock": reward_and_clock(expect_legacy=args.expect_legacy_defect),
        "source_sha256": hashes,
        "interpretation": "The 200s match is a task terminal, not an artificial rollout cutoff. Its terminal target should exclude gamma*V(terminal_obs). Raw logged reward itself is correct. No training implementation was changed by this diagnostic.",
    }
    assert not torch.cuda.is_initialized()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("audit_completed", "finite_horizon_value_handling_correct",
                                            "reproduced_unwanted_timeout_bootstrap", "cuda_initialized")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
