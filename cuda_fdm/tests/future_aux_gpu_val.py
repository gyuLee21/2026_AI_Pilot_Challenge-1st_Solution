"""Bounded real-CUDA validation and future-aux overhead measurements.

No search, W&B, historical checkpoint continuation or performance claims from
short learning curves. CUDA events measure stream elapsed time, not isolated
kernel execution. The frozen pre-change kernel is used only as a test reference.
"""
import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from cuda_fdm.future_aux import build_future_labels, inference_state_dict, AUX_METRIC_KEYS
from cuda_fdm.finite_checks import require_finite, finite_max_abs_error, record_integrity_failure
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.obs_reward import ned_to_body, _mv
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.tests.future_aux_val import git_loop_labels
from cuda_fdm.tests.kernel_stream_val import make_env, action_sequence, kernels
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal
from cuda_fdm.tests.pool_assigned_integration_val import phase_time, snapshot, restore, capture_rollout
import cuda_rt

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def git_capture_features(env, early_float=False):
    """Git capture: duplicate state9/rotation/velocity work, for comparison only."""
    state = env.state9()
    main, opp = state[:, 0], state[:, 1]
    rm, ro = ned_to_body(main[:, 3:6]), ned_to_body(opp[:, 3:6])
    result = torch.cat((main[:, :3], _mv(rm.transpose(1, 2), main[:, 6:9]),
                        opp[:, :3], _mv(ro.transpose(1, 2), opp[:, 6:9]), rm.flatten(1)), 1)
    return result.float() if early_float else result


class OldObsABI:
    def __init__(self, kernel):
        self.kernel = kernel

    def launch(self, grid, block, args, tensors):
        self.kernel.launch(grid, block, args[:-1], tensors=tensors)

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def kernel_checks(before):
    env = make_env(32)
    env.obr.enable_aux_capture()
    env.obr.kernel_build_obs(env.sim.states)
    features = env.obr.aux_features
    initial = dict(states=env.sim.states.clone(), obr=env.obr.clone_state(),
                   rng=torch.cuda.get_rng_state())
    commands = action_sequence(env, 64)
    source = (before / "cuda_fdm__gen__obs_kernel.cu").read_text(encoding="utf-8")
    cc = torch.cuda.get_device_capability()
    ptx = cuda_rt.compile_ptx(source, f"compute_{cc[0]}{cc[1]}")
    old_kernel = cuda_rt.Kernel(ptx, "build_obs_kernel")
    new_kernel = env.obr._k_obs
    traces, records = {}, {}
    for name in ("before", "off", "on"):
        env.sim.states.copy_(initial["states"])
        env.obr.aux_features = features if name == "on" else None
        env.obr.restore(initial["obr"])
        torch.cuda.set_rng_state(initial["rng"])
        env.obr._k_obs = OldObsABI(old_kernel) if name == "before" else new_kernel
        env.obr.kernel_build_obs(env.sim.states)
        trace, captured, reference, starts = [], [], [], []
        start = torch.zeros(env.nenv, device="cuda")
        timeouts = 0
        before_counts = [(k.launch_count, k.context_sync_count) for k in kernels(env)]
        for t, controls in enumerate(commands):
            if name == "on":
                captured.append(features.clone())
                reference.append(git_capture_features(env))
                starts.append(start.clone())
            if t == 3:
                env.obr.hp[::8] = 0.
            if t == 7:
                env.obr.t_sec[1::8] = 199.9  # true 200-second terminal / autoreset
            obs, reward, done, info = env.step(controls)
            start = done.float()
            timeouts += int(info["truncated"].sum())
            trace.append(dict(obs=obs.clone(), reward=reward.clone(), done=done.clone(),
                               terminal=info["terminal_obs"].clone(), states=env.sim.states.clone(),
                               obr={k: v.clone() for k, v in env.obr.clone_state().items()
                                    if k != "aux_features"}))
        torch.cuda.synchronize()
        require_finite(trace, f"{name}.kernel_trace")
        traces[name] = trace
        records[name] = dict(timeouts=timeouts,
                             launches=sum(k.launch_count - c[0] for k, c in zip(kernels(env), before_counts)),
                             context_syncs=sum(k.context_sync_count - c[1] for k, c in zip(kernels(env), before_counts)))
        assert records[name]["launches"] == 320 and records[name]["context_syncs"] == 0
        assert timeouts > 0
    # The frozen historical path predates the official-origin/direct-MSL fix.
    # It remains a diagnostic reference, not an equality oracle for the new
    # coordinate contract.  The auxiliary ON/OFF comparison below is the actual
    # invariance gate and must stay bitwise exact.
    historical_contract_equal = True
    try:
        assert_tree_equal(traces["before"], traces["off"])
    except AssertionError:
        historical_contract_equal = False
    assert_tree_equal(traces["off"], traces["on"])
    captured, reference, starts = torch.stack(captured), torch.stack(reference), torch.stack(starts)
    feature_error = finite_max_abs_error(captured, reference, "CUDA cached vs state9 features")
    assert feature_error < 1e-7, feature_error
    labels, mask = build_future_labels(captured, starts, .1)
    reference_labels, ref_mask = git_loop_labels(reference, starts)
    assert_tree_equal(mask, ref_mask)
    valid = mask.repeat_interleave(3, -1)
    label_error = finite_max_abs_error(labels[valid], reference_labels[valid], "same-formula FP64 labels")
    assert label_error < 2e-7, label_error
    fp32_labels, fp32_mask = git_loop_labels(reference, starts, early_float=True)
    assert_tree_equal(mask, fp32_mask)
    remote_error = finite_max_abs_error(labels[valid], fp32_labels[valid], "Git early-FP32 labels")
    assert remote_error < 5e-4, remote_error  # 5cm component scale tolerance, actual error reported
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        env.obr.kernel_build_obs(env.sim.states)
        alternate = env.obr.aux_features.clone()
    torch.cuda.current_stream().wait_stream(side)
    side_error = finite_max_abs_error(alternate, git_capture_features(env), "aux capture other Torch stream")
    assert side_error < 1e-7
    return dict(passed=True, aux_off_on_raw_physics_obs_reward_history_bitwise_equal=True,
                historical_pre_datum_contract_equal=historical_contract_equal,
                historical_reference_note=(
                    "The frozen pre-datum kernel is diagnostic only; current AUX OFF/ON "
                    "is the bitwise invariance contract."),
                comparison_paths=records, capture_max_abs_error=feature_error,
                fp64_target_max_abs_error=label_error, git_fp32_target_max_abs_error=remote_error,
                git_fp32_max_position_difference_m=remote_error*100., masks_equal=True,
                other_stream_max_abs_error=side_error)


def production_trainer(enabled, nenv, width):
    env = GpuDogfightVecEnv(nenv, seed=73)
    # Standard 4096-entry IC pool, 200s episodes and production stagger are kept.
    cfg = PPOGPUConfig(device="cuda", architecture="mlp", hidden=(width,)*3,
                       aux_pred=enabled, aux_coef=.1, rollout_steps=64, update_epochs=4,
                       num_minibatches=8, gamma=.997, gae_lambda=.95, seed=73,
                       sched_period=0, milestone_period=0, exploiter_iters=0,
                       save_runtime=True, total_iterations=2)
    trainer = PPOGPUTrainer(env, cfg)
    # Lazy IC pool generation belongs to startup, not steady rollout. Otherwise
    # restoring an initial fixture with ic_pool=None rebuilds 4096 CPU ICs inside
    # EVERY timed rollout and obscures the real relative auxiliary overhead.
    env._ensure_ic_pool()
    return trainer


def diagnostic_metrics(update):
    return {key: float(value) for key, value in update.items()}


def measure_targets(tr, repeats):
    result = []
    funcs = dict(vectorized=lambda: build_future_labels(tr.b_aux_features, tr.b_done, .1),
                 git_loop=lambda: git_loop_labels(tr.b_aux_features, tr.b_done, early_float=True))
    for fn in funcs.values():
        fn()
    for repeat in range(repeats):
        order = funcs if repeat % 2 == 0 else reversed(funcs)
        for name in order:
            _, timing = phase_time(funcs[name])
            result.append(dict(path=name, repeat=repeat, **timing))
    def capture_loop(git):
        for t in range(tr.cfg.rollout_steps):
            value = git_capture_features(tr.env, early_float=True) if git else tr.env.obr.aux_features
            tr.b_aux_features[t].copy_(value)
    captures = []
    saved = tr.b_aux_features.clone()
    for git in (False, True):
        capture_loop(git)
        for repeat in range(repeats):
            _, timing = phase_time(capture_loop, git)
            captures.append(dict(path="git_state9" if git else "cached_copy", repeat=repeat, **timing))
    tr.b_aux_features.copy_(saved)
    return dict(targets=result, captures64=captures,
                note="Capture-only timing excludes FDM and observation work shared by both paths.")


def benchmark(output, nenv=4096, width=512, repeats=3):
    print("[future-aux] initialize production OFF fixture", flush=True)
    off = production_trainer(False, nenv, width)
    initial_off = snapshot(off)
    result_off = off.collect_rollout()
    trace_off = capture_rollout(off, result_off)
    fixture_off = snapshot(off)
    print("[future-aux] initialize production ON fixture", flush=True)
    on = production_trainer(True, nenv, width)
    initial_on = snapshot(on)
    result_on = on.collect_rollout()
    trace_on = capture_rollout(on, result_on)
    fixture_on = snapshot(on)
    assert_tree_equal(trace_off, trace_on)
    assert_tree_equal(off.model.state_dict(), inference_state_dict(on.model.state_dict(), True))
    assert_tree_equal(off.norm.state_dict(), on.norm.state_dict())
    assert_tree_equal(off.env.sim.states, on.env.sim.states)
    del trace_off, trace_on
    labels = measure_targets(on, repeats)
    inputs = dict(off=(off, result_off, fixture_off, initial_off),
                   on=(on, result_on, fixture_on, initial_on))
    # Warm both optimizer kernels; reset to identical base model and RNG fixtures
    # outside every timed interval. Auxiliary gradients intentionally differ.
    for tr, batch, fixture, _ in inputs.values():
        restore(tr, fixture)
        tr.update(*batch[:2])
    updates, rollouts = [], []
    for repeat in range(repeats):
        order = ("off", "on") if repeat % 2 == 0 else ("on", "off")
        for name in order:
            tr, batch, fixture, initial = inputs[name]
            restore(tr, fixture)
            torch.cuda.reset_peak_memory_stats()
            alloc_before = torch.cuda.memory_allocated()
            u, timing = phase_time(tr.update, *batch[:2])
            require_finite((u, tr.model.state_dict(), tr.actor_opt.state_dict(), tr.critic_opt.state_dict()), name)
            row = dict(path=name, repeat=repeat, **timing, metrics=diagnostic_metrics(u),
                        incremental_peak_allocated_mib=(torch.cuda.max_memory_allocated()-alloc_before)/2**20)
            updates.append(row)
            restore(tr, initial)
            batch2, timing = phase_time(tr.collect_rollout)
            # Every rollout, including stochastic actions and GAE, must be equal.
            assert_tree_equal(batch, batch2)
            rollouts.append(dict(path=name, repeat=repeat, **timing))
            print(f"[future-aux] {name} repeat{repeat}: rollout={timing['wall_sec']:.3f}s "
                  f"update={row['wall_sec']:.3f}s", flush=True)
    (output / "timings_before_resume_check.json").write_text(
        json.dumps(dict(updates=updates, rollouts=rollouts, targets=labels), indent=2, allow_nan=False),
        encoding="utf-8")
    # Real checkpoint/resume with learned head and Adam moments. Do not require
    # ON and OFF updated parameters to match: this is a new learning loss.
    restore(on, fixture_on)
    on.update(*result_on[:2])
    on.iteration = 1
    on.save(output / "aux_after_one_update.pt")
    expected_rollout = on.collect_rollout()
    expected_labels = on.b_aux_labels.clone()
    expected_update = on.update(*expected_rollout[:2])
    expected_state = on._snapshot_learner()
    on.load(output / "aux_after_one_update.pt")
    assert all(state["step"].device.type == "cpu"
               for opt in (on.actor_opt, on.critic_opt) for state in opt.state.values())
    actual_rollout = on.collect_rollout()
    assert_tree_equal(expected_rollout, actual_rollout)
    assert_tree_equal(expected_labels, on.b_aux_labels)
    actual_update = on.update(*actual_rollout[:2])
    assert_tree_equal(expected_update, actual_update)
    assert_tree_equal(expected_state, on._snapshot_learner())
    summary = {}
    for name in inputs:
        r = statistics.median(row["wall_sec"] for row in rollouts if row["path"] == name)
        u = statistics.median(row["wall_sec"] for row in updates if row["path"] == name)
        summary[name] = dict(rollout_sec=r, update_sec=u, sum_sec=r+u,
                             parameters=sum(p.numel() for p in inputs[name][0].model.parameters()))
    summary["added_seconds"] = summary["on"]["sum_sec"] - summary["off"]["sum_sec"]
    summary["overhead_percent"] = 100*(summary["on"]["sum_sec"]/summary["off"]["sum_sec"]-1)
    data = dict(passed=True, environments=nenv, rollout=64, architecture=[width]*3,
                 epochs=4, minibatch_splits=8, initial_parameters_and_rollout_identical=True,
                 cold_ic_pool_generation_excluded=True,
                 exact_gpu_save_resume=True, summary=summary, update_measurements=updates,
                 rollout_measurements=rollouts, label_and_capture=labels,
                 memory_note="Incremental update peak; both ON/OFF fixtures were resident for paired timing.",
                 time_note="Wall includes host overhead; CUDA Event values are stream elapsed, not pure kernel time.",
                 auxiliary_metrics_note="Training MSE aggregates the visited minibatches; not held-out skill evaluation.")
    print(json.dumps(summary), flush=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--nenv", type=int, default=4096)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--kernel-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        print("[future-aux] checking actual kernel capture and pre-change equivalence", flush=True)
        result = dict(kernel=kernel_checks(ROOT / "runs/change_validation/future_aux_v1/before"))
        print(json.dumps(result), flush=True)
        if not args.kernel_only:
            gc.collect()
            torch.cuda.empty_cache()
            result["production"] = benchmark(args.output, args.nenv, args.width, args.repeats)
        result["passed"] = True
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    except Exception as exc:
        record_integrity_failure(args.output, exc, "future_aux_gpu_validation")
        raise


if __name__ == "__main__":
    main()
