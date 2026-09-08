"""Bounded same-batch legacy-padding vs MLP-flat benchmark; no W&B.

The old updater is read from its immutable pre-edit backup, not retained as an
alternate production path. Old/flat shuffling intentionally differs. A separate
explicit-permutation PPO reference checks the new GPU arithmetic for correctness.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer, TRAINING_PROTOCOL
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.finite_checks import require_finite, record_integrity_failure
from cuda_fdm.tests.mlp_updater_val import flat_reference


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs/architecture_search/experiment_v1"


def import_baseline():
    path = EXPERIMENT / "performance_fixes_v1/before/cuda_fdm__ppo_gpu.py"
    spec = importlib.util.spec_from_file_location("_performance_baseline_v3", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.TRAINING_PROTOCOL != "cuda_episode_opponent_finite_horizon_v3":
        raise ValueError("The saved baseline is not the expected pre-MLP-update protocol")
    return module, path


@contextmanager
def phase_events(tr, legacy):
    """CUDA stream spans include host gaps; these are NOT pure kernel times."""
    names = ("evaluate_actions_packed_sequence", "evaluate_values_packed_sequence") if legacy else (
        "evaluate_actions", "get_value")
    pairs, old_methods = {"actor": [], "critic": []}, []
    for phase, name, optimizer in zip(("actor", "critic"), names, (tr.actor_opt, tr.critic_opt)):
        old_forward, old_step = getattr(tr.model, name), optimizer.step
        def forward(*args, _fn=old_forward, _phase=phase, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pairs[_phase].append((start, end))
            return _fn(*args, **kwargs)
        def step(*args, _fn=old_step, _phase=phase, **kwargs):
            result = _fn(*args, **kwargs)
            pairs[_phase][-1][1].record()
            return result
        had_forward = name in tr.model.__dict__
        had_step = "step" in optimizer.__dict__
        setattr(tr.model, name, forward)
        optimizer.step = step
        old_methods.append((name, old_forward, optimizer, old_step, had_forward, had_step))
    try:
        yield pairs
    finally:
        for name, forward, optimizer, step, had_forward, had_step in old_methods:
            if had_forward:
                setattr(tr.model, name, forward)
            else:
                delattr(tr.model, name)
            if had_step:
                optimizer.step = step
            else:
                delattr(optimizer, "step")


def timed_update(tr, adv, ret, snapshot, seed, legacy):
    # Adam.load_state_dict can share tensor storage; deepcopy is essential.
    tr._restore_learner(copy.deepcopy(snapshot))
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with phase_events(tr, legacy) as events:
        start.record()
        wall = time.perf_counter()
        result = tr.update(adv, ret)
        end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - wall
    require_finite((result, tr.model.state_dict(), tr.actor_opt.state_dict(), tr.critic_opt.state_dict()),
                   "measured MLP updater")
    stats = {k: float(v) if torch.is_tensor(v) else v for k, v in result.items()}
    return dict(update_sec=elapsed, cuda_stream_span_sec=start.elapsed_time(end)/1000.,
                actor_cuda_span_sec=sum(s.elapsed_time(e) for s, e in events["actor"])/1000.,
                critic_cuda_span_sec=sum(s.elapsed_time(e) for s, e in events["critic"])/1000.,
                actor_steps=len(events["actor"]), critic_steps=len(events["critic"]),
                transitions_per_sec=adv.numel()/elapsed,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, **stats)


def parameter_difference(left, right, prefix):
    keys = [k for k in left if k.startswith(prefix)]
    diff2 = sum(float(((left[k].double()-right[k].double())**2).sum()) for k in keys)
    base2 = sum(float((left[k].double()**2).sum()) for k in keys)
    return dict(max_abs=max(float((left[k]-right[k]).abs().max()) for k in keys),
                relative_l2=(diff2/max(base2, 1e-30))**.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Choose a fresh benchmark output; keep prior evidence")
    torch.set_num_threads(1)
    old_module, backup_path = import_baseline()
    checkpoint = EXPERIMENT / "timeout_fixes_v1/integrated_smoke/mlp_smoke_iter2.pt"
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = old_module.PPOGPUConfig(**data["cfg"])
    cfg.total_iterations = 1
    env = GpuDogfightVecEnv(4096, seed=cfg.seed)
    old = old_module.PPOGPUTrainer(env, cfg)
    old.load(checkpoint)
    adv, ret, rollout_stats = old.collect_rollout()  # one frozen, real 262144-transition batch
    snapshot = old._snapshot_learner()
    new = PPOGPUTrainer(env, PPOGPUConfig(**cfg.__dict__))
    for name in ("b_obs", "b_act", "b_logp", "b_val", "b_rew", "b_done"):
        setattr(new, name, getattr(old, name))
    # No further rollout from the shared env occurs in this fixed-batch test.
    batch_hash = hashlib.sha256(new.b_obs.cpu().numpy().tobytes()).hexdigest()
    packed = old._pack_recurrent_batch(adv, ret)
    padding = dict(valid_transitions=adv.numel(), padded_rows=packed["valid"].numel(),
                   sequences=packed["valid"].shape[0],
                   padding_fraction=1-adv.numel()/packed["valid"].numel())
    del packed
    rows, summaries, params = [], [], {}
    for mode, target_kl in (("normal_kl_gate", cfg.target_kl), ("fixed_four_epochs", None)):
        old.cfg.target_kl = new.cfg.target_kl = target_kl
        for name, tr in (("legacy_padded", old), ("mlp_flat", new)):
            timed_update(tr, adv, ret, snapshot, 1701, name == "legacy_padded")  # excluded warmup
        for repeat in range(args.repeats):
            order = (("legacy_padded", old), ("mlp_flat", new))
            if repeat % 2:
                order = order[::-1]
            for name, tr in order:
                row = dict(mode=mode, path=name, repeat=repeat,
                           **timed_update(tr, adv, ret, snapshot, 1701, name == "legacy_padded"))
                rows.append(row)
                params[(mode, name)] = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
                print(json.dumps(row, allow_nan=False), flush=True)
        selected = [r for r in rows if r["mode"] == mode]
        before = statistics.median(r["update_sec"] for r in selected if r["path"] == "legacy_padded")
        after = statistics.median(r["update_sec"] for r in selected if r["path"] == "mlp_flat")
        summaries.append(dict(mode=mode, old_sec=before, mlp_sec=after, speedup=before/after,
                              saved_sec=before-after, reduction_percent=100*(1-after/before),
                              actor_parameter_difference=parameter_difference(params[(mode, "legacy_padded")], params[(mode, "mlp_flat")], "actor"),
                              critic_parameter_difference=parameter_difference(params[(mode, "legacy_padded")], params[(mode, "mlp_flat")], "critic")))

    # Correctness is a different comparison: same flat membership AND permutation.
    new.cfg.target_kl = None
    new._restore_learner(copy.deepcopy(snapshot))
    torch.manual_seed(1717)
    permutations = [torch.randperm(adv.numel(), device="cuda") for _ in range(new.cfg.update_epochs)]
    with patch("torch.randperm", side_effect=[p.clone() for p in permutations]):
        actual = new.update(adv, ret)
    actual_params = {k: v.detach().cpu().clone() for k, v in new.model.state_dict().items()}
    actual_opt = [copy.deepcopy(opt.state_dict()) for opt in (new.actor_opt, new.critic_opt)]
    new._restore_learner(copy.deepcopy(snapshot))
    reference = flat_reference(new, adv, ret, permutations)
    require_finite((actual, reference, actual_params, new.model.state_dict()), "flat GPU reference")
    ref_error = 0.
    for key, value in actual_params.items():
        other = new.model.state_dict()[key].detach().cpu()
        torch.testing.assert_close(value, other, atol=2e-6, rtol=2e-5)
        ref_error = max(ref_error, float((value-other).abs().max()))
    for key, value in reference.items():
        torch.testing.assert_close(actual[key], value, atol=2e-6, rtol=2e-5)
    for expected, opt in zip(actual_opt, (new.actor_opt, new.critic_opt)):
        for key, values in expected["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, opt.state_dict()["state"][key][name], atol=2e-6, rtol=2e-5)
    result = dict(passed=True, protocol=TRAINING_PROTOCOL, device=torch.cuda.get_device_name(),
                  checkpoint=str(checkpoint), snapshot_has_nonzero_adam_moments=bool(snapshot["actor_opt"]["state"]),
                  observation_batch_sha256=batch_hash, padding=padding,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (backup_path, ROOT/"cuda_fdm/ppo_gpu.py", Path(__file__))},
                  reference_parameter_max_abs=ref_error, reference_optimizer_and_metrics_passed=True,
                  all_comparisons_same_initial_model_optimizer_rollout=True,
                  legacy_comparison_is_not_algorithm_equivalence=True,
                  timing_note="CUDA Events measure stream elapsed spans including host submission gaps, not pure kernel execution. fixed_four_epochs disables KL stopping ONLY for equal-work timing; normal_kl_gate and integration smoke retain target_kl=0.03.",
                  no_wandb=True, new_learning_runs=0, summaries=summaries, measurements=rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": True, "summaries": summaries, "reference_parameter_max_abs": ref_error}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Does not touch the experiment-root hold or healthy historical files.
        if "--output" in sys.argv:
            record_integrity_failure(Path(sys.argv[sys.argv.index("--output")+1]), exc, "MLP fixed-batch benchmark")
        raise
