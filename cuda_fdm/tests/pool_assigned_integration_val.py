"""Bounded real-FDM A/B of full vs assigned-lane MLP opponent inference.

Every replay starts from the same learner/Adam/RMS/physics/pool/RNG fixture.
No architecture search, legacy-run resume, W&B, or long training is performed.
Only test code can select the frozen full-inference reference.
"""
import argparse
import copy
import csv
import hashlib
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from cuda_fdm.finite_checks import require_finite, require_finite_training_stats, record_integrity_failure
from cuda_fdm.ppo_gpu import OpponentPool, PPOGPUConfig, PPOGPUTrainer, TRAINING_PROTOCOL
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.tests.integrity_smoke_val import pool_gpu_check
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal, legacy_full_act


def inference_path(name):
    return patch.object(OpponentPool, "act", legacy_full_act) if name == "full" else nullcontext()


def snapshot(tr):
    return dict(learner=copy.deepcopy(tr._snapshot_learner()),
                runtime=tr._runtime_state(), pool=copy.deepcopy(tr.pool.state_dicts()),
                next_id=tr.pool.next_id, iteration=getattr(tr, "iteration", 0))


def restore(tr, data):
    tr._restore_learner(copy.deepcopy(data["learner"]))
    tr.pool.load_state_dicts(copy.deepcopy(data["pool"]), next_id=data["next_id"])
    tr._restore_runtime(data["runtime"])
    tr.iteration = data["iteration"]
    tr._refresh_weights()


def phase_time(fn, *args, **kwargs):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return value, dict(wall_sec=time.perf_counter()-wall, stream_sec=start.elapsed_time(end)/1000.)


def capture_rollout(tr, result):
    adv, ret, stats = result
    return dict(buffers={k: getattr(tr, k).clone() for k in
                         ("b_obs", "b_act", "b_logp", "b_val", "b_rew", "b_done")},
                advantage=adv.clone(), returns=ret.clone(), stats=copy.deepcopy(stats),
                physics=tr.env.sim.states.clone(), assignments=tr.opp_assign.clone(),
                cuda_rng=torch.cuda.get_rng_state())


def add_fixture_opponents(tr, count):
    """Controlled benchmark snapshots, not new training-pool policies."""
    while tr.pool.resident_size() < count:
        i = tr.pool.resident_size()
        net = copy.deepcopy(tr.model)
        with torch.no_grad():
            for p in net.actor_parameters():
                p.add_(torch.randn_like(p) * (.0002 * i))
        norm = tr.norm.clone()
        norm.mean.add_(i * .015)
        norm.var.mul_(1. + .015 * i)
        tr.pool.add(net, norm, permanent=(i >= 4), ema=.5)
        del net
    # New benchmark episodes only; no mid-episode changes in the learner loop.
    tr._refresh_weights()
    tr._reset_env_state()


def rollout_benchmark(tr, fixture, count, repeats):
    for path in ("full", "assigned"):
        restore(tr, fixture)
        with inference_path(path):
            tr.collect_rollout()  # excluded warmup
    measurements, reference = [], None
    for repeat in range(repeats):
        order = ("full", "assigned") if repeat % 2 == 0 else ("assigned", "full")
        for path in order:
            restore(tr, fixture)
            with inference_path(path):
                result, timing = phase_time(tr.collect_rollout)
            current = capture_rollout(tr, result)  # checks outside timed interval
            require_finite(current, f"pool{count}.{path}.rollout")
            if reference is None:
                reference = current
            else:
                assert_tree_equal(reference, current)
            measurements.append(dict(path=path, repeat=repeat, **timing,
                                     completed_episodes=float(result[2]["ep_count"])))
            del current
    before = statistics.median(x["wall_sec"] for x in measurements if x["path"] == "full")
    after = statistics.median(x["wall_sec"] for x in measurements if x["path"] == "assigned")
    result = dict(opponents=count, full_rollout_sec=before, assigned_rollout_sec=after,
                  reduction_percent=100*(1-after/before), speedup=before/after,
                  transitions=tr.nenv*tr.cfg.rollout_steps,
                  rollout_buffers_advantage_returns_stats_physics_assignments_rng_exact=True,
                  measurements=measurements)
    print(json.dumps({k: v for k, v in result.items() if k != "measurements"}), flush=True)
    return result


def two_iterations(tr, path, trace=False):
    base_rollout, base_update, base_step = tr.collect_rollout, tr.update, tr.env.step
    phases, rows, traces, controls = {}, [], [], []

    def rollout(*args, **kwargs):
        result, phases["rollout"] = phase_time(base_rollout, *args, **kwargs)
        if trace:
            traces.append(capture_rollout(tr, result))
        return result

    def update(*args, **kwargs):
        result, phases["update"] = phase_time(base_update, *args, **kwargs)
        return result

    def step(action):
        if trace:
            controls.append(action.clone())
        return base_step(action)

    def log(s):
        require_finite_training_stats(s)
        rows.append(dict(path=path, iteration=s.iteration,
                         rollout_sec=phases["rollout"]["wall_sec"],
                         update_sec=phases["update"]["wall_sec"],
                         rollout_cuda_stream_sec=phases["rollout"]["stream_sec"],
                         update_cuda_stream_sec=phases["update"]["stream_sec"],
                         reported_iteration_sec=s.elapsed_sec,
                         completed_episodes=s.completed_episodes, score=s.win_rate,
                         mean_return=s.mean_return, kl=s.approx_kl, clipfrac=s.clipfrac,
                         entropy=s.entropy, value_loss=s.value_loss, explained_variance=s.explained_variance,
                         actor_grad_norm=s.extra["actor_grad_norm"],
                         critic_grad_norm=s.extra["critic_grad_norm"],
                         epochs=s.extra["epochs"], optimizer_steps=s.extra["optimizer_steps"]))

    tr.collect_rollout, tr.update = rollout, update
    if trace:
        tr.env.step = step
    start_iteration = int(getattr(tr, "iteration", 0)) + 1
    tr.cfg.total_iterations = start_iteration + 1
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall = time.perf_counter()
    try:
        with inference_path(path):
            tr.train(on_iteration=log, start_iteration=start_iteration)
    finally:
        tr.collect_rollout, tr.update, tr.env.step = base_rollout, base_update, base_step
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall
    require_finite((rows, tr.model.state_dict(), tr.actor_opt.state_dict(), tr.critic_opt.state_dict()),
                   f"{path}.two_iterations")
    evidence = None
    if trace:
        evidence = dict(rollouts=traces, controls=controls, final=snapshot(tr))
    return dict(path=path, iterations=2, loop_wall_sec=elapsed,
                seconds_per_iteration=elapsed/2, rows=rows,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20), evidence


def integrated_check_and_benchmark(tr, fixture, output, repeats):
    restore(tr, fixture)
    tr.save(output / "initial_pool6.pt")
    evidence = {}
    for path in ("full", "assigned"):
        restore(tr, fixture)
        _, evidence[path] = two_iterations(tr, path, trace=True)
    assert_tree_equal(evidence["full"], evidence["assigned"])
    del evidence
    # Exact continuation after a real two-update checkpoint, including live
    # games and multi-opponent stochastic inference.
    checkpoint = output / "after_pool6_iter2.pt"
    tr.save(checkpoint)
    expected = capture_rollout(tr, tr.collect_rollout())
    tr.load(checkpoint)
    actual = capture_rollout(tr, tr.collect_rollout())
    assert_tree_equal(expected, actual)
    del expected, actual
    runs = []
    for repeat in range(repeats):
        order = ("full", "assigned") if repeat % 2 == 0 else ("assigned", "full")
        for path in order:
            restore(tr, fixture)
            measurement, _ = two_iterations(tr, path, trace=False)
            measurement["repeat"] = repeat
            runs.append(measurement)
    before = statistics.median(x["seconds_per_iteration"] for x in runs if x["path"] == "full")
    after = statistics.median(x["seconds_per_iteration"] for x in runs if x["path"] == "assigned")
    return dict(passed=True, opponents=6, same_initial_model_nonzero_adam_rms_physics_pool_rng=True,
                all_main_opponent_actions_rollout_targets_optimizer_parameters_rng_exact=True,
                exact_save_resume=True, full_sec_per_iteration=before, assigned_sec_per_iteration=after,
                reduction_percent=100*(1-after/before), speedup=before/after, measurements=runs,
                timing_scope="Two-iteration train loop includes rollout, update, EMA and logging callback; checkpoint I/O excluded; phase timing barriers applied equally")


def run(output, nenv, repeats):
    cfg = PPOGPUConfig(architecture="mlp", hidden=(672, 672, 672), gru_size=0,
                       rollout_steps=64, update_epochs=4, num_minibatches=8,
                       gamma=.997, gae_lambda=.95, ent_coef=.001, target_kl=.03,
                       seed=916, save_runtime=True, pool_evict_cap=4,
                       sched_period=0, milestone_period=0, exploiter_iters=0,
                       selfplay_gate_threshold=1.1)
    env = GpuDogfightVecEnv(nenv, seed=cfg.seed)
    tr = PPOGPUTrainer(env, cfg)
    # Excluded warmup creates nonzero Adam state before controlled A/B replays.
    adv, ret, _ = tr.collect_rollout()
    tr.update(adv, ret)
    tr.iteration = 0
    rollouts = []
    for count in (1, 4, 6):
        add_fixture_opponents(tr, count)
        fixture = snapshot(tr)
        rollouts.append(rollout_benchmark(tr, fixture, count, repeats))
        restore(tr, fixture)
    integrated = integrated_check_and_benchmark(tr, fixture, output, repeats)
    print(json.dumps({k: v for k, v in integrated.items() if k != "measurements"}), flush=True)
    # Also exercise partial episode termination, FIFO live-retired identity,
    # exact stochastic-state restoration and exploiter main restoration on CUDA.
    lifecycle = pool_gpu_check(output)
    all_rows = [dict(repeat=x["repeat"], **row) for x in integrated["measurements"] for row in x["rows"]]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    sources = (Path(__file__), Path(__file__).with_name("pool_assigned_val.py"),
               Path(__file__).resolve().parents[1] / "ppo_gpu.py")
    result = dict(passed=True, protocol=TRAINING_PROTOCOL, nenv=nenv,
                  architecture="mlp672x3", rollout_steps=64, no_wandb=True,
                  no_long_training=True, rollout_benchmarks=rollouts,
                  integrated=integrated, pool_lifecycle=lifecycle,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  cuda_event_note="Stream spans include host launch gaps; not pure kernel durations")
    require_finite(result, "pool integration results")
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--nenv", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists() or (args.output / "INTEGRITY_FAILURE.json").exists():
        raise FileExistsError("Use a fresh bounded-validation directory")
    torch.set_num_threads(1)
    try:
        run(args.output, args.nenv, args.repeats)
    except Exception as exc:
        record_integrity_failure(args.output, exc, "assigned_pool_integration")
        raise
