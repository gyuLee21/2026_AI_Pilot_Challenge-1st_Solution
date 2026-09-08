"""Bounded GPU fault injection: autoreset cannot hide NaN or overwrite recovery."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from cuda_fdm import train_gpu
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv


def inject_nan_after_physics(env):
    original_step = env.sim.step

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        env.sim.states[0, 0] = float("nan")
        return result

    env.sim.step = step


class FaultTrainer(PPOGPUTrainer):
    def collect_rollout(self, *args, **kwargs):
        if not getattr(self, "_audit_fault_armed", False):
            inject_nan_after_physics(self.env)
            self._audit_fault_armed = True
        return super().collect_rollout(*args, **kwargs)


def run(output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = GpuDogfightVecEnv(4, seed=671)
    env.reset(stagger=False)
    inject_nan_after_physics(env)
    controls = torch.zeros(8, 4, device="cuda", dtype=torch.float64)
    controls[:, 3] = .7
    obs, _, done, info = env.step(controls)
    raw_flag = info["terminal_state_finite"].cpu().tolist()
    assert not raw_flag[0] and all(raw_flag[1:])
    assert bool(done[0]), "invalid physics must terminate"
    assert bool(torch.isfinite(obs).all()), "returned observation should demonstrate sanitization/autoreset"
    assert bool(torch.isfinite(env.sim.states).all()), "autoreset should have replaced the invalid physical state"
    original_rng = env.rng
    class WarmupRng:
        def integers(self, *args, **kwargs):
            if kwargs.get("size") == env.nenv:
                return np.ones(env.nenv, dtype=np.int64)
            return original_rng.integers(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(original_rng, name)
    env.rng = WarmupRng()
    warmup_error = None
    try:
        env._staggered_warmup(2, None)
    except FloatingPointError as exc:
        warmup_error = str(exc)
    assert warmup_error and "staggered initialization" in warmup_error
    del env

    cfg = PPOGPUConfig(architecture="mlp", hidden=(16, 16), gru_size=0,
                       total_iterations=1, rollout_steps=2, update_epochs=1,
                       num_minibatches=1, sched_period=0, milestone_period=0,
                       exploiter_iters=0, save_runtime=True)
    healthy = PPOGPUTrainer(GpuDogfightVecEnv(4, seed=0), cfg)
    checkpoint = output / "recovery.pt"
    healthy.save(checkpoint)
    original_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    del healthy
    torch.cuda.empty_cache()
    argv = ["train_gpu", "--nenv", "4", "--iters", "1", "--architecture", "mlp",
            "--hidden", "16,16", "--rollout", "2", "--epochs", "1",
            "--minibatches", "1", "--sched-period", "0", "--milestone-period", "0",
            "--exploiter-iters", "0", "--no-wandb", "--save-runtime", "--save-every", "1",
            "--save", str(checkpoint), "--log", str(output / "metrics.csv")]
    error = None
    with patch.object(sys, "argv", argv), patch.object(train_gpu, "PPOGPUTrainer", FaultTrainer):
        try:
            train_gpu.main()
        except FloatingPointError as exc:
            error = str(exc)
    assert error and "update refused" in error
    unchanged = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_digest
    assert unchanged, "failed training overwrote the healthy recovery checkpoint"
    marker = output / "INTEGRITY_FAILURE.json"
    assert marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["error_type"] == "FloatingPointError"
    result = dict(passed=True, injected="NaN ECEF position after physics before reward/autoreset",
                  raw_terminal_state_finite=raw_flag, post_autoreset_state_finite=True,
                  returned_observation_finite=True, training_error=error,
                  invalid_warmup_rejected=warmup_error,
                  healthy_recovery_unchanged=unchanged, failure_marker=str(marker),
                  note="This isolated test intentionally leaves its expected-failure marker; do not use this directory for training.")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.output)
