from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv


def run(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        env = GpuDogfightVecEnv(32, seed=97)
        cfg = PPOGPUConfig(
            total_iterations=1, device="cuda", architecture="mlp",
            hidden=(384, 384, 384), num_bins=21, normalize_obs=True,
            rollout_steps=2, sched_period=0, milestone_period=0,
            exploiter_iters=0, league_enabled=True,
            league_dir=str(Path(temp) / "league"), league_active_cap=24,
            league_payoff_games=64, league_admission_games=256,
            aux_pred=True, aux_coef=.1, seed=97, opp_sample=True)
        trainer = PPOGPUTrainer(env, cfg)
        init_sec = time.perf_counter() - started
        bundle = trainer._policy_bundle()
        measurements = []
        for label, games, block in (("payoff_cold", 64, 1),
                                    ("payoff_warm", 64, 2),
                                    ("admission_cold", 256, 3),
                                    ("admission_warm", 256, 4)):
            torch.cuda.synchronize()
            then = time.perf_counter()
            result = trainer._evaluate_pair(bundle, bundle, games, paired=True,
                                            seed_block=block)
            torch.cuda.synchronize()
            measurements.append({
                "label": label, "requested_games_per_order": games,
                "total_games": result["games"], "paired_blocks": result["paired_blocks"],
                "wall_sec": time.perf_counter() - then, "score": result["score"],
                "lcb95": result["lcb95"], "ucb95": result["ucb95"],
                "stochastic": result["stochastic"],
                "policy_rng_seed": result["policy_rng_seed"],
                "steps": result["steps"],
            })
        receipt = {
            "protocol": trainer._league_contract()["evaluator"],
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "trainer_init_sec": init_sec,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "measurements": measurements,
        }
    temp_path = output.with_suffix(output.suffix + ".tmp")
    temp_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp_path, output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
