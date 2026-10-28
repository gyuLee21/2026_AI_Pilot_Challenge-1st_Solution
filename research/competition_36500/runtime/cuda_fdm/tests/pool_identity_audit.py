"""CPU-only audit of pool refresh identity and existing search logs.

This diagnostic records current behavior (including fixed episode identity),
not a training job. Historical bug reports remain in audit_20260831. It creates only tiny CPU models,
uses a toy environment, and never imports the CUDA environment. Optional output
is an audit JSON; experiment logs and checkpoints are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from cuda_fdm.ppo_gpu import OpponentPool, PPOGPUConfig, PPOGPUTrainer, build_actor_critic


class ToyEnv:
    OBS_SIZE = 4
    min_altitude_m = 304.8

    def __init__(self, nenv=8, episode_steps=3):
        self.nenv, self.nac = nenv, 2 * nenv
        self.episode_steps = episode_steps
        self.reset_calls = 0
        self.opponent_controls = []

    def reset(self, stagger=True):
        del stagger
        self.reset_calls += 1
        self.episode_step = 0
        return torch.zeros(self.nenv, 2, self.OBS_SIZE)

    def step(self, controls):
        self.opponent_controls.append(controls.view(self.nenv, 2, 4)[:, 1].clone())
        self.episode_step += 1
        done = torch.full((self.nenv,), self.episode_step == self.episode_steps)
        terminal_obs = torch.full((self.nenv, 2, self.OBS_SIZE), float(self.episode_step))
        hp = torch.ones(self.nenv, 2)
        hp[done, 1] = 0.0
        obs = terminal_obs.clone()
        obs[done] = 0.0
        if bool(done.all()):
            self.episode_step = 0
        return obs, torch.zeros(self.nenv, 2), done, {
            "truncated": torch.zeros_like(done),
            "terminal_obs": terminal_obs,
            "terminal_hp": hp,
            "terminal_alt_m": torch.full((self.nenv, 2), 1000.0),
            "terminal_state_finite": torch.ones(self.nenv, dtype=torch.bool),
        }


@torch.no_grad()
def constant_action(model, index):
    for param in model.actor_parameters():
        param.zero_()
    model.actor_logits[-1].bias.view(model.act_dim, model.num_bins)[:, index] = 20.0


def toy_trainer(cap, nenv=8):
    cfg = PPOGPUConfig(device="cpu", architecture="mlp", hidden=(8, 8),
                       num_bins=3, gru_size=0, normalize_obs=False, opp_sample=False,
                       rollout_steps=1, pool_evict_cap=cap, pool_uniform_floor=1.0,
                       sched_period=0, milestone_period=0, exploiter_iters=0)
    tr = PPOGPUTrainer(ToyEnv(nenv), cfg)
    constant_action(tr.model, 0)
    constant_action(tr.pool.entries[0]["net"], 0)
    tr.pool.entries[0]["audit_name"] = "A"
    return tr


def assigned_names(tr):
    by_id = {e.get("id", i): e["audit_name"] for i, e in enumerate(tr.pool.entries)}
    return [by_id[i] for i in tr.opp_assign.tolist()]


def reproduce():
    fixed = hasattr(OpponentPool, "active_entries")
    tr = toy_trainer(cap=1)
    tr.collect_rollout()
    before = assigned_names(tr)
    lengths_before = tr.ep_len.tolist()
    constant_action(tr.model, 2)
    tr.pool.add(tr.model, tr.norm, permanent=False)
    tr.pool.entries[-1]["audit_name"] = "B"
    all_live = not bool(tr._next_done.any())
    tr._refresh_weights()
    after = assigned_names(tr)
    tr.collect_rollout()
    _, _, terminal = tr.collect_rollout()
    fifo = dict(all_lanes_live_at_refresh=all_live, assignment_before=before,
                assignment_after=after, episode_lengths_before=lengths_before,
                explicit_environment_resets=tr.env.reset_calls,
                lane0_opponent_controls=[x[0].tolist() for x in tr.env.opponent_controls],
                completed_episodes=int(terminal["ep_count"]),
                terminal_win_attribution={e["audit_name"]: int(terminal["win_by_opp"][i])
                                          for i, e in enumerate(tr.pool.entries)})
    assert all_live and tr.env.reset_calls == 1
    assert (before == after) if fixed else (before != after)
    credited = "A" if fixed else "B"
    assert fifo["completed_episodes"] == fifo["terminal_win_attribution"][credited] == tr.nenv

    tr = toy_trainer(cap=4, nenv=64)
    tr.collect_rollout()
    before = assigned_names(tr)
    constant_action(tr.model, 2)
    tr.pool.add(tr.model, tr.norm, permanent=True)
    tr.pool.entries[-1]["audit_name"] = "B"
    torch.manual_seed(717)
    tr._refresh_weights()
    after = assigned_names(tr)
    changed = sum(a != b for a, b in zip(before, after))
    append = dict(live_lanes=tr.nenv, changed_lanes=changed,
                  fifo_eviction=False, explicit_environment_resets=tr.env.reset_calls)
    assert (changed == 0 if fixed else changed > 0) and not bool(tr._next_done.any())
    old_assign = tr.opp_assign.clone()
    tr.pool.entries[0]["ema"], tr.pool.entries[1]["ema"] = 0.9, 0.1
    tr.opp_weights = tr._pool_weights()
    tr.collect_rollout()
    weights_only = dict(changed_lanes=int((old_assign != tr.opp_assign).sum()))
    assert weights_only["changed_lanes"] == 0

    pool = OpponentPool(tr._model_kwargs, "cpu", evict_cap=2, sample=False)
    for name, permanent in (("A", False), ("P", True), ("B", False)):
        pool.add(tr.model, None, permanent=permanent)
        pool.entries[-1]["audit_name"] = name
    old_indices = [0, 1, 2]
    old_names = [pool.entries[i]["audit_name"] for i in old_indices]
    pool.add(tr.model, None, permanent=False)
    pool.entries[-1]["audit_name"] = "C"
    new_names = [pool.entries[i]["audit_name"] for i in old_indices]
    index_shift = dict(unchanged_indices=old_indices, before=old_names, after=new_names)
    assert old_names == ["A", "P", "B"]
    assert new_names == (["A", "P", "B"] if fixed else ["P", "B", "C"])

    kwargs = dict(obs_dim=4, act_dim=4, num_bins=3, architecture="gru",
                  hidden=(8, 8, 8), gru_size=4, encoder_depth=2)
    model = build_actor_critic(**kwargs).to("cpu")
    pool = OpponentPool(kwargs, "cpu", evict_cap=2, sample=False)
    pool.add(model, None, permanent=False)
    obs, assign, starts = torch.randn(4, 4), torch.zeros(4, dtype=torch.long), torch.zeros(4)
    for _ in range(3):
        pool.act(obs, assign, starts)
    hidden_before = tuple(h.clone() for h in pool.entries[0]["actor_state"])
    pool.add(model, None, permanent=False)
    old_hidden_preserved = all(torch.equal(a, b) for a, b in
                               zip(hidden_before, pool.entries[0]["actor_state"]))
    new_hidden_absent = pool.entries[1]["actor_state"] is None
    pool.act(obs, assign, starts)
    with torch.no_grad():
        _, expected = pool.entries[1]["net"].act(
            obs, pool.entries[1]["net"].initial_state(4, "cpu"), starts, sample=False)
    new_hidden_matches_cold_step = all(torch.equal(a, b) for a, b in
                                      zip(expected, pool.entries[1]["actor_state"]))
    gru = dict(retained_snapshot_keeps_own_hidden=old_hidden_preserved,
               new_snapshot_has_no_hidden=new_hidden_absent,
               new_snapshot_hidden_equals_one_cold_step=new_hidden_matches_cold_step,
               unassigned_snapshot_also_advanced=True)
    assert old_hidden_preserved and new_hidden_absent and new_hidden_matches_cold_step
    return dict(episode_fixed_protocol=fixed, fifo_episode=fifo, insertion_without_eviction=append,
                weights_only_refresh=weights_only, removal_without_remap=index_shift,
                recurrent_state=gru)


def inspect_checkpoint(path):
    # Trusted, locally produced training checkpoint; force all tensors onto CPU.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    result = dict(path=str(path.resolve()), iteration=int(ckpt["iteration"]),
                  architecture=cfg.get("architecture"), rollout_steps=cfg["rollout_steps"],
                  save_runtime=cfg.get("save_runtime", False), sched_period=cfg.get("sched_period"),
                  pool_entry_count=len(ckpt.get("pool", [])),
                  pool_evict_cap=ckpt.get("pool_evict_cap"),
                  pool_serialized_keys=sorted(ckpt["pool"][0]) if ckpt.get("pool") else [])
    state = ckpt.get("runtime")
    if state:
        trainer = state["trainer"]
        done = trainer["_next_done"].bool()
        result["runtime"] = dict(
            nenv=done.numel(), done_lanes=int(done.sum()),
            live_lanes=int((~done).sum()), live_fraction=float((~done).float().mean()),
            assignment_counts=torch.bincount(trainer["opp_assign"]).tolist(),
            episode_length_quantiles=torch.quantile(trainer["ep_len"].float(),
                torch.tensor([0., .25, .5, .75, 1.])).tolist(),
            pool_hidden_count=len(state["pool_h"]),
            checkpoint_is_after_pool_event=None,
        )
    return result


def reproduce_nonfinite_scoring():
    """Replay OLD scoring arithmetic only; the new evaluator rejects the raw flag first."""
    cases = {}
    for name, altitude in (("nan_altitude", float("nan")), ("nan_velocity_finite_altitude", 1000.0)):
        hp_a, hp_b = torch.ones(1), torch.ones(1)
        alt_a, alt_b = torch.tensor([altitude]), torch.tensor([1000.0])
        state9 = torch.zeros(1, 2, 9)
        state9[0, 0, 2 if name == "nan_altitude" else 6] = float("nan")
        da = (hp_a <= 0) | (alt_a < 300.)
        db = (hp_b <= 0) | (alt_b < 300.)
        both_alive = ~da & ~db
        win = (db & ~da) | (both_alive & (hp_a > hp_b + 1e-9))
        lose = (da & ~db) | (both_alive & (hp_a < hp_b - 1e-9))
        score = win.float() + .5 * (~win & ~lose).float()
        first_hit = torch.full((1,), -1.)
        data = torch.stack((score, hp_a-hp_b, (alt_a < 300.).float(),
                            first_hit, (alt_b < 300.).float(), first_hit), 1)
        cases[name] = dict(semantics="legacy scoring arithmetic without the new raw-state gate",
                           raw_terminal_state_finite=bool(torch.isfinite(state9).all()),
                           score=float(score[0]), crash=bool(data[0, 2]),
                           legacy_transformed_data_check_accepts=bool(torch.isfinite(data).all()))
        assert not cases[name]["raw_terminal_state_finite"]
        assert cases[name]["score"] == .5 and cases[name]["legacy_transformed_data_check_accepts"]
    return cases


def inspect_logs(root, nenv, rollout_steps):
    result = []
    pattern = re.compile(r"^it\s+(\d+)\s+\|.*\[(gate|milestone)\]")
    for log in sorted((root / "training").glob("*/seed*/stdout.log")):
        rows = {}
        csv_path = log.with_name("metrics.csv")
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows[int(row["iter"])] = row
        events = []
        for lineno, line in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = pattern.search(line)
            if not match:
                continue
            iteration, kind = int(match[1]), match[2]
            event = dict(iteration=iteration, kind=kind, source_line=lineno)
            row = rows.get(iteration)
            if row:
                eps = float(row["eps"])
                event.update(completed_in_rollout=eps, mean_completed_length=float(row["mean_len"]),
                             live_boundary_fraction_lower_bound=max(0., 1. - eps / nenv),
                             live_boundary_fraction_uniform_timing_estimate=max(
                                 0., 1. - eps / (nenv * rollout_steps)))
            events.append(event)
        bounds = [e["live_boundary_fraction_lower_bound"] for e in events if
                  "live_boundary_fraction_lower_bound" in e]
        estimate = [e["live_boundary_fraction_uniform_timing_estimate"] for e in events if
                    "live_boundary_fraction_uniform_timing_estimate" in e]
        result.append(dict(
            model=log.parent.parent.name, seed=log.parent.name, log=str(log.resolve()),
            last_metrics_iteration=max(rows) if rows else None,
            gate_events=sum(e["kind"] == "gate" for e in events),
            milestone_events=sum(e["kind"] == "milestone" for e in events),
            event_counts_through_iteration={str(cut): sum(e["iteration"] <= cut for e in events)
                                             for cut in (400, 800, 1450)},
            duplicate_event_iterations=len(events) - len({(e["iteration"], e["kind"]) for e in events}),
            live_boundary_lower_bound_min=float(min(bounds)) if bounds else None,
            live_boundary_lower_bound_mean=float(np.mean(bounds)) if bounds else None,
            live_boundary_uniform_timing_estimate_mean=float(np.mean(estimate)) if estimate else None,
            events=events,
        ))
    return dict(nenv_assumption=nenv, rollout_steps_assumption=rollout_steps,
                interpretation={
                    "lower_bound": "At most eps lanes can finish on the final step; all other lanes remain live.",
                    "estimate": "1 - eps/(nenv*rollout); assumes approximately uniform terminal timing within a rollout, not a measured final-step fraction.",
                    "switch_fraction": "Not recoverable from aggregate logs; assignments before/after gate were not recorded.",
                }, candidates=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--nenv", type=int, default=4096)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = Path(__file__).resolve().parents[1] / "ppo_gpu.py"
    result = dict(scope="CPU-only current-behavior audit; no training, CUDA environment, or checkpoint mutation",
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reproduction=reproduce(), nonfinite_scoring_fixture=reproduce_nonfinite_scoring(),
                  logs=inspect_logs(args.experiment_root, args.nenv, args.rollout))
    if args.checkpoint:
        result["checkpoint"] = inspect_checkpoint(args.checkpoint)
        ckpt = result["checkpoint"]
        matched = next((row for row in result["logs"]["candidates"] if
                        row["model"] == args.checkpoint.parent.parent.name and
                        row["seed"] == args.checkpoint.parent.name), None)
        if matched and "runtime" in ckpt:
            events = [e for e in matched["events"] if e["iteration"] <= ckpt["iteration"]]
            last_event = max((e["iteration"] for e in events), default=None)
            ckpt["runtime"]["last_logged_pool_event_iteration"] = last_event
            ckpt["runtime"]["checkpoint_is_after_pool_event"] = last_event == ckpt["iteration"]
            if last_event is not None and not ckpt["sched_period"]:
                elapsed_steps = (ckpt["iteration"] - last_event) * ckpt["rollout_steps"]
                ckpt["runtime"]["steps_since_last_logged_pool_event"] = elapsed_steps
                ckpt["runtime"]["all_current_episodes_began_after_last_logged_pool_event"] = (
                    ckpt["runtime"]["episode_length_quantiles"][-1] <= elapsed_steps)
    result["cuda_initialized"] = torch.cuda.is_initialized()
    assert not result["cuda_initialized"], "CPU audit unexpectedly initialized CUDA"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"reproduction": result["reproduction"],
                      "nonfinite_scoring_fixture": result["nonfinite_scoring_fixture"],
                      "candidates": [{k: v for k, v in row.items() if k != "events"}
                                     for row in result["logs"]["candidates"]],
                      "checkpoint": result.get("checkpoint"),
                      "cuda_initialized": result["cuda_initialized"],
                      "output": str(args.output) if args.output else None}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
