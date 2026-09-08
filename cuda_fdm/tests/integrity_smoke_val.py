"""Bounded production-size updater smoke and GPU episode-identity recovery.

Two iterations per architecture/path are integration checks, never a learning
comparison or architecture-selection result. No W&B or historical run mutation.
"""
import argparse
import copy
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.finite_checks import require_finite, require_finite_training_stats, record_integrity_failure
from cuda_fdm.tests.kernel_stream_val import kernels


def same_parameters(left, right):
    worst = 0.
    for key, value in left.items():
        other = right[key].cpu()
        torch.testing.assert_close(value, other, atol=2e-6, rtol=2e-5)
        worst = max(worst, float((value - other).abs().max()))
    return worst


def production_smoke(output, nenv):
    rows, models = [], {}
    for architecture in ("mlp",):
        cfg = PPOGPUConfig(total_iterations=2, rollout_steps=64, update_epochs=4,
                           num_minibatches=8, architecture=architecture,
                           hidden=(672, 672, 672) if architecture == "mlp" else (512, 512, 512),
                           gru_size=0 if architecture == "mlp" else 256,
                           recurrent_seq_len=32, seed=842, save_runtime=True,
                           sched_period=0, milestone_period=0, exploiter_iters=0,
                           selfplay_gate_threshold=1.1)
        env = GpuDogfightVecEnv(nenv, seed=cfg.seed)
        tr = PPOGPUTrainer(env, cfg)
        learner, runtime = tr._snapshot_learner(), tr._runtime_state()
        original_build = env.obr.kernel_build_obs
        outcomes = {}
        for label, sync, full in (("sync_full", True, True), ("async_masked", False, False)):
            tr._restore_learner(learner)
            tr._restore_runtime(runtime)
            tr._refresh_weights()
            for kernel in kernels(env):
                kernel.force_sync = sync
            env.obr.kernel_build_obs = lambda states, env_mask=None: original_build(
                states, None if full else env_mask)
            base_rollout, base_update = tr.collect_rollout, tr.update
            phase_times = {}
            def timed(name, fn):
                def call(*args, **kwargs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    phase_times[name] = time.perf_counter() - start
                    return result
                return call
            tr.collect_rollout = timed("rollout_sec", base_rollout)
            tr.update = timed("update_sec", base_update)
            torch.cuda.reset_peak_memory_stats()
            def log(stats):
                require_finite_training_stats(stats)
                row = dict(architecture=architecture, path=label, iteration=stats.iteration,
                           transitions=stats.global_step, iteration_sec=stats.elapsed_sec,
                           **phase_times, kl=stats.approx_kl, clip_fraction=stats.clipfrac,
                           entropy=stats.entropy, value_loss=stats.value_loss,
                           explained_variance=stats.explained_variance,
                           completed_episodes=stats.completed_episodes,
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
                rows.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
            try:
                tr.train(on_iteration=log)
            finally:
                tr.collect_rollout, tr.update = base_rollout, base_update
                env.obr.kernel_build_obs = original_build
            require_finite(tr.model.state_dict(), f"{architecture}.{label}.parameters")
            require_finite(tr._runtime_state(), f"{architecture}.{label}.runtime")
            outcomes[label] = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
        error = same_parameters(outcomes["sync_full"], outcomes["async_masked"])
        checkpoint = output / f"{architecture}_smoke_iter2.pt"
        tr.save(checkpoint)
        models[architecture] = dict(parameter_max_abs=error, checkpoint=str(checkpoint),
                                    nenv=nenv, rollout_steps=64, iterations_per_path=2,
                                    same_initial_model_optimizer_normalizer_physics_rng=True)
        del tr, env, learner, runtime, outcomes
        gc.collect()
        torch.cuda.empty_cache()
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(models=models, metrics=rows,
                interpretation="integration and implementation equivalence only; gate/exploiter disabled in the two-iteration timing comparison")


def pool_gpu_check(output):
    cfg = PPOGPUConfig(architecture="mlp", hidden=(32, 32, 32), gru_size=0,
                       rollout_steps=1, update_epochs=1, num_minibatches=2, save_runtime=True,
                       pool_evict_cap=1, opp_sample=False, seed=819,
                       sched_period=0, milestone_period=0, exploiter_iters=1,
                       exploiter_win_target=1.0, selfplay_gate_threshold=1.1)
    env = GpuDogfightVecEnv(8, seed=cfg.seed)
    tr = PPOGPUTrainer(env, cfg)
    tr.collect_rollout()
    assert not bool(tr._next_done.any())
    before_id = tr.opp_assign.clone()
    second = tr.pool.add(tr.model, tr.norm, permanent=False)
    tr._refresh_weights()
    assert torch.equal(before_id, tr.opp_assign)
    env.obr.hp[1::4] = 0.  # terminate exactly four opponents; four games continue
    _, _, results = tr.collect_rollout()
    ids = results["opponent_ids"].cpu().tolist()
    eps = dict(zip(ids, results["ep_by_opp"].cpu().tolist()))
    wins = dict(zip(ids, results["win_by_opp"].cpu().tolist()))
    assert eps == {0: 4., second: 0.} and wins == eps, (eps, wins)
    tr.pool.update_emas(results["win_by_opp"], results["loss_by_opp"], results["ep_by_opp"],
                        cfg.selfplay_ema_alpha, results["opponent_ids"])
    tr._refresh_weights()
    assert tr.pool.resident_size() == 2
    checkpoint = output / "pool_live_retired.pt"
    tr.save(checkpoint)
    adv, ret, results = tr.collect_rollout()
    expected = {k: getattr(tr, k).clone() for k in ("b_obs", "b_act", "b_rew", "b_done", "opp_assign")}
    expected_physics = env.sim.states.clone()
    tr.load(checkpoint)
    adv2, ret2, results2 = tr.collect_rollout()
    for key, value in expected.items():
        torch.testing.assert_close(value, getattr(tr, key), atol=0, rtol=0)
    torch.testing.assert_close(adv, adv2, atol=0, rtol=0)
    torch.testing.assert_close(ret, ret2, atol=0, rtol=0)
    torch.testing.assert_close(expected_physics, env.sim.states, atol=0, rtol=0)
    assert tr.opp_assign.cpu().tolist() == [second, 0, second, 0, second, 0, second, 0]
    before = tr._runtime_state()
    params = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
    step = tr.global_step
    tr.train_exploiter()
    after = tr._runtime_state()
    for key, value in before["trainer"].items():
        if key != "opp_weights":
            torch.testing.assert_close(value, after["trainer"][key], atol=0, rtol=0)
    torch.testing.assert_close(before["sim"], after["sim"], atol=0, rtol=0)
    for key, value in before["obr"].items():
        torch.testing.assert_close(value, after["obr"][key], atol=0, rtol=0)
    for identity, state in before["pool_h"].items():
        for old, new in zip(state, after["pool_h"][identity]):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
    assert same_parameters(params, tr.model.state_dict()) == 0.
    assert tr.global_step == step and tr.pool.num_permanent() == 1
    tr.save(output / "pool_after_exploiter.pt")
    return dict(passed=True, completed_by_id=eps, wins_by_id=wins,
                live_assignment_unchanged_on_fifo=True, exact_mid_episode_resume=True,
                exact_main_runtime_and_parameters_after_exploiter=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--nenv", type=int, default=4096)
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = dict(passed=True, pool=pool_gpu_check(args.output),
                      production=production_smoke(args.output, args.nenv))
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"passed": result["passed"], "pool": result["pool"]}), flush=True)
    except Exception as exc:
        record_integrity_failure(args.output, exc, "bounded_integrity_smoke")
        raise
