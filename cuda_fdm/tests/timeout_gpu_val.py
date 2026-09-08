"""Bounded actual CUDA clock, terminal reward/GAE, and autoreset checks."""
import argparse
import gc
import json
from pathlib import Path
from types import MethodType

import torch

from cuda_fdm.finite_checks import require_finite
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer, TRAINING_PROTOCOL
from cuda_fdm.rl_env import GpuDogfightVecEnv


def make_env(nenv):
    env = GpuDogfightVecEnv(nenv, seed=416)
    # Static-clock fixture is outside all weapon ranges and safely above ground.
    env.distA_ft_choices = (20000.,)
    env.dist_headon_ft = 20000.
    env.alt_ft_range = (12000., 12000.)
    env.ic_pool_size = 32
    return env


@torch.no_grad()
def clock_check():
    env = make_env(4)
    env.reset(stagger=False)
    controls = torch.zeros(env.nac, 4, device="cuda", dtype=torch.float64)
    controls[:, 3] = .7
    terms = torch.empty(2000, env.nenv, device="cuda", dtype=torch.uint8)
    truncs = torch.empty_like(terms)
    total_reward = torch.zeros(env.nac, device="cuda", dtype=torch.float64)
    for step in range(2000):
        # Exercise the actual fused reconstruction/reward/termination kernel.
        # Physics is deliberately fixed to isolate accumulated-clock behavior.
        reward, term, trunc = env.obr.kernel_advance(env.sim.states, controls)
        terms[step] = term
        truncs[step] = trunc
        total_reward += reward
    assert not bool(terms.any())
    assert not bool(truncs[:-1].any())
    assert bool(truncs[-1].all())
    assert bool(((env.obr.t_sec - 200.).abs() < 1e-8).all())
    # No damage, no safety cost; remove the -4 timeout draw to recover geometry.
    geometry_total = total_reward + 4.
    assert bool((geometry_total.abs() <= 5. + 1e-9).all())
    require_finite((env.sim.states, total_reward, geometry_total), "clock2000")
    result = dict(control_steps=2000, first_timeout_step=2000,
                  final_time_sec=env.obr.t_sec.cpu().tolist(),
                  geometry_total=geometry_total.cpu().tolist(),
                  scope="2000 real fused reward/clock launches with fixed safe physical states")
    del env
    return result


@torch.no_grad()
def learner_check(architecture, output):
    cfg = PPOGPUConfig(device="cuda", architecture=architecture, hidden=(32, 32, 32),
                       gru_size=8 if architecture == "gru" else 0, rollout_steps=1,
                       update_epochs=1, num_minibatches=1, normalize_obs=True,
                       sched_period=0, milestone_period=0, exploiter_iters=0,
                       save_runtime=True, seed=718)
    env = make_env(4)
    trainer = PPOGPUTrainer(env, cfg)
    env.reset(stagger=False)
    env.obr.t_sec.fill_(199.9)
    env.obr.hp.copy_(torch.tensor([1., .8, .8, 1., 1., 1., 1., 0.], device="cuda"))
    obs = env.obr.kernel_build_obs(env.sim.states).view(4, 2, env.OBS_SIZE)
    trainer._next_obs = obs[:, 0].clone()
    trainer._next_opp_obs = obs[:, 1].clone()
    trainer._next_done.zero_()
    captured = {}
    original_step = env.step
    def capture_step(actions):
        new_obs, reward, done, info = original_step(actions)
        captured.update(reward=reward.clone(), done=done.clone(),
                        term=info["terminated"].clone(), trunc=info["truncated"].clone(),
                        terminal_hp=info["terminal_hp"].clone(),
                        terminal_finite=info["terminal_state_finite"].clone())
        return new_obs, reward, done, info
    env.step = capture_step
    calls = []
    original_value = trainer.model.value_step
    def known_value(self, obs, state=None, episode_start=None):
        calls.append(obs.shape[0])
        return torch.full((obs.shape[0],), 10., device=obs.device), state
    trainer.model.value_step = MethodType(known_value, trainer.model)
    adv, target, _ = trainer.collect_rollout()
    assert bool(captured["done"].all())
    assert captured["trunc"].cpu().tolist() == [True, True, True, False]
    assert captured["term"].cpu().tolist() == [False, False, False, True]
    assert bool(captured["terminal_finite"].all())
    raw = captured["reward"][:, 0].float()
    torch.testing.assert_close(target[0], raw, rtol=0, atol=2e-6)
    torch.testing.assert_close(trainer.b_rew[0], raw, rtol=0, atol=0)
    assert len(calls) == 2  # one current value plus live-rollout boundary value
    assert bool((env.obr.t_sec == 0).all())
    assert bool((env.obr.hp == 1.).all())
    torch.testing.assert_close(env.obr.prev_alt_log, env.obr._altitude_log(env.state9_flat()),
                               atol=1e-9, rtol=0)
    saved = dict(raw_rewards=raw.cpu().tolist(), targets=target[0].cpu().tolist(),
                 terminated=captured["term"].cpu().tolist(),
                 timed_out=captured["trunc"].cpu().tolist(), critic_calls=len(calls))
    trainer.model.value_step = original_value
    env.step = original_step
    with torch.enable_grad():
        stats = trainer.update(adv, target)
    require_finite((stats, trainer.model.state_dict()), "post-timeout update")
    checkpoint = output / f"{architecture}_timeout.pt"
    trainer.save(checkpoint)
    trainer.load(checkpoint)
    assert bool(trainer._next_done.bool().all())
    assert bool((env.obr.t_sec == 0).all())
    # The immediately following step belongs to fresh games, not another timeout.
    trainer.collect_rollout()
    assert not bool(trainer._next_done.any())
    require_finite((trainer.b_rew, trainer.b_val, env.sim.states), "fresh episode after timeout")
    result = dict(passed=True, architecture=architecture, **saved,
                  actual_physics_autoreset=True, update_save_resume=True,
                  fresh_episode_not_double_terminal=True, checkpoint=str(checkpoint))
    del trainer, env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Choose a fresh validation directory")
    torch.set_num_threads(1)
    result = dict(passed=True, training_protocol=TRAINING_PROTOCOL,
                  clock=clock_check(),
                  learner=[learner_check(a, args.output) for a in ("mlp",)])
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
