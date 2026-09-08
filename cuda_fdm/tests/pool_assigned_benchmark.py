"""Production assigned-lane MLP inference vs the frozen full-batch reference.

Preserves stable opponent IDs and CUDA sampling RNG consumption. CPU grouping,
normalization, gather/scatter and action sampling are all inside measured time.
"""
import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from cuda_fdm.ppo_gpu import OpponentPool, RunningNorm, build_actor_critic
from cuda_fdm.finite_checks import require_finite
from cuda_fdm.tests.pool_assigned_val import legacy_full_act


@torch.no_grad()
def assigned_mlp_act(pool, obs, assign, starts):
    return pool.act(obs, assign, starts)


def measure(fn, pool, obs, assignments, steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    starts = torch.zeros(len(obs), device=obs.device)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    start.record()
    for step in range(steps):
        # Realistic reassignment frames, not a cached forever-constant partition.
        out = fn(pool, obs, assignments[step % len(assignments)], starts)
    end.record()
    torch.cuda.synchronize()
    return dict(wall_ms=(time.perf_counter()-wall)*1000/steps,
                cuda_stream_span_ms=start.elapsed_time(end)/steps, finite=bool(torch.isfinite(out).all()))


def compare(pool, obs, assignments):
    starts = torch.ones(len(obs), device=obs.device)
    for sample in (False, True):
        pool.sample = sample
        torch.manual_seed(239)
        rng = torch.cuda.get_rng_state()
        expected = [legacy_full_act(pool, obs, assign, starts) for assign in assignments]
        expected_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        actual = [assigned_mlp_act(pool, obs, assign, starts) for assign in assignments]
        actual_rng = torch.cuda.get_rng_state()
        assert torch.equal(expected_rng, actual_rng), "sampling consumes a different RNG stream"
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
    pool.sample = True
    return dict(greedy_and_sampled_actions_exact=True, cuda_rng_state_exact=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Use a new benchmark directory")
    torch.set_num_threads(1)
    torch.manual_seed(223)
    kwargs = dict(obs_dim=184, act_dim=4, num_bins=21, architecture="mlp", hidden=(672, 672, 672))
    net = build_actor_critic(**kwargs).cuda().eval()
    nenv = 4096
    obs = torch.randn(nenv, 184, device="cuda")
    norm = RunningNorm(184, "cuda")
    cases = []
    for count in (1, 2, 4, 6):
        pool = OpponentPool(kwargs, "cuda", evict_cap=count, sample=True)
        for i in range(count):
            # Distinct snapshots and RMS ensure a wrong row/norm is detectable.
            with torch.no_grad():
                for p in net.actor_parameters():
                    p.add_(torch.randn_like(p) * .0005)
            norm.mean.fill_(i*.11)
            norm.var.fill_(1+i*.07)
            pool.add(net, norm, permanent=False)
        for balance in (["balanced"] if count == 1 else ["balanced", "skewed"]):
            assignment = torch.arange(nenv, device="cuda") % count
            if balance == "skewed":
                assignment[:int(nenv*.9)] = 0
            assignments = [assignment, assignment.roll(33), assignment.roll(111)]
            correctness = compare(pool, obs, assignments)
            rows = []
            for name, fn in (("full", legacy_full_act), ("assigned", assigned_mlp_act)):
                measure(fn, pool, obs, assignments, 3)
            for repeat in range(args.repeats):
                order = [("full", legacy_full_act), ("assigned", assigned_mlp_act)]
                if repeat % 2:
                    order.reverse()
                for name, fn in order:
                    rows.append(dict(path=name, repeat=repeat,
                                     **measure(fn, pool, obs, assignments, args.steps)))
            before = statistics.median(r["wall_ms"] for r in rows if r["path"] == "full")
            after = statistics.median(r["wall_ms"] for r in rows if r["path"] == "assigned")
            row = dict(opponents=count, balance=balance, full_ms=before, assigned_ms=after,
                       speedup=before/after, estimated_saved_sec_per_64_steps=(before-after)*64/1000.,
                       **correctness, measurements=rows)
            cases.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "measurements"}), flush=True)
        # Include an active entry with zero lanes and an ID-retired entry.
        if count == 6:
            none_assigned = torch.zeros(nenv, dtype=torch.long, device="cuda")
            compare(pool, obs, [none_assigned])
            pool.evict_cap = 5
            pool.add(net, norm, permanent=False)
            live = torch.arange(nenv, device="cuda") % 6
            pool.refresh_residents(live, torch.zeros(nenv, device="cuda"))
            compare(pool, obs, [live])
        del pool
        gc.collect()
    require_finite(cases, "pool benchmark results")
    result = dict(passed=True, nenv=nenv, architecture="mlp672x3", cases=cases,
                  includes_partition_normalization_gather_scatter_sampling=True,
                  tests_zero_assigned_entries_and_live_retired_id=True,
                  production_pool_changed=True, no_wandb=True,
                  timing_scope="Opponent inference only on synthetic observations, not full FDM rollout or measured iteration speedup",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).resolve().parents[1]/"ppo_gpu.py")})
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": True, "production_pool_changed": True}), flush=True)


if __name__ == "__main__":
    main()
