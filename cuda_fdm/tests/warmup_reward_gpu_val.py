"""Bounded actual-CUDA reset gate, run serially before the new main starts."""
import argparse
import json
from pathlib import Path

import torch

from cuda_fdm.finite_checks import require_finite, record_integrity_failure
from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.rl_env import GpuDogfightVecEnv
from claude_code.my_reward import MY_REWARD_CONFIG


@torch.no_grad()
def validate():
    torch.set_num_threads(1)
    env = GpuDogfightVecEnv(16, seed=947)
    env.ic_pool_size = 16
    initial_obs = env.reset(stagger=True, max_stagger_steps=16)
    require_finite((initial_obs, env.sim.states), "warmup.gpu.initial")
    state = env.state9_flat().clone()
    torch.testing.assert_close(env.obr.prev_alt_log, env.obr._altitude_log(state), atol=1e-9, rtol=0)
    assert bool((env.obr.t_sec > 0).any()), "test must cover nonzero warmup phases"
    reference = BatchObsReward(16, device="cuda", enable_kernel=False)
    reference.restore(env.obr.clone_state()); reference.initialize_reward_state(state)
    torch.testing.assert_close(env.obr.prev_x, reference.prev_x, atol=2e-7, rtol=0)
    # The init kernel must use the captured clock, also after phase1/2 boundaries.
    for elapsed in (0., 99.9, 100.1, 150.1, 199.8):
        env.obr.t_sec.fill_(elapsed)
        before = env.obr.clone_state()
        env.obr.kernel_init_reward_state(env.sim.states)
        reference.restore(before); reference.initialize_reward_state(state)
        torch.testing.assert_close(env.obr.prev_x, reference.prev_x, atol=2e-7, rtol=0)
        for key in before:
            if key not in ("prev_x", "prev_x_valid", "prev_alt_log"):
                torch.testing.assert_close(before[key], env.obr.clone_state()[key], atol=0, rtol=0)
    env.obr.t_sec.fill_(1.)
    env.obr.kernel_init_reward_state(env.sim.states)
    controls = torch.zeros(env.nac, 4, device="cuda", dtype=torch.float64)
    controls[:, 3] = .8
    reward, term, trunc = env.obr.kernel_advance(env.sim.states, controls,
        cfg=dict(MY_REWARD_CONFIG, damage_scale=0.), reward_mode=1)
    require_finite(reward, "warmup.gpu.first_reward")
    torch.testing.assert_close(reward, torch.zeros_like(reward), atol=1e-9, rtol=0)
    assert not bool(term.any()) and not bool(trunc.any())
    return dict(passed=True, nenv=16, nonzero_stagger_tested=True,
        no_false_first_step_hunter_reward=True, geometry_and_phase_anchor_correct=True,
        physical_and_reconstruction_state_preserved=True, maximum_false_reward=float(reward.abs().max()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    output = ap.parse_args().output
    if output.exists():
        raise FileExistsError("preserve prior validation output")
    try:
        result = validate()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result), flush=True)
    except Exception as exc:
        record_integrity_failure(output.parent, exc, "warmup_reward_gpu")
        raise


if __name__ == "__main__":
    main()
