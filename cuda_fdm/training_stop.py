"""Opt-in 3-9 plateau gate. Evaluation state is checkpoint-authoritative.

Uses ChampionManager in an evaluation-only namespace: never changes the league
champion pointer, archive eligibility, payoff graph or opponent sampling.
Intervals are paired stratified bootstrap estimates, NOT convergence proofs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .league_vnext.champion import ChampionManager, PromotionEvidence
from .league_vnext.contracts import ChampionConfig

PROTOCOL = "three_nine_plateau_v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def paired_difference(candidate, reference, key="paired_scores", alpha=.05,
                      bootstrap_samples=20000):
    """Equal-opponent stratified paired bootstrap on signed block differences.

Same opponent ordering, seeds, block counts and action mode are mandatory.
Keep a small finite-sample width floor; all observed differences being zero
does not prove two distinct policies have identical population performance.
No binomial/Jeffreys transformation is applied to signed differences.
"""
    if not candidate or len(candidate) != len(reference) or not 0 < alpha < 1:
        raise ValueError("incomplete paired comparison")
    rng = np.random.default_rng(170901)
    boot = np.zeros(bootstrap_samples)
    means, variances, widths, counts = [], [], [], []
    for new, old in zip(candidate, reference):
        for name in ("opponent_hash", "seed_block", "paired_blocks", "stochastic"):
            if new[name] != old[name]:
                raise ValueError(f"unpaired {name}")
        x, y = np.asarray(new[key], float), np.asarray(old[key], float)
        if (x.shape != y.shape or x.ndim != 1 or len(x) < 4
                or len(x) != new["paired_blocks"] or not new["stochastic"]
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or ((x < 0) | (x > 1) | (y < 0) | (y > 1)).any()):
            raise ValueError("invalid paired block observations")
        d = x - y
        means.append(float(d.mean()))
        variances.append(float(d.var(ddof=1)))
        counts.append(len(d))
        widths.append(1. / (len(d) + 1))
        # Chunking bounds temporary memory independently of evaluation budget.
        for start in range(0, bootstrap_samples, 1000):
            end = min(start + 1000, bootstrap_samples)
            ix = rng.integers(0, len(d), (end - start, len(d)))
            boot[start:end] += d[ix].mean(axis=1) / len(candidate)
    mean = float(np.mean(means))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    floor = float(np.mean(widths))
    se = math.sqrt(sum(v / n for v, n in zip(variances, counts))) / len(counts)
    return dict(mean=mean, lcb=max(-1., min(float(lo), mean - floor)),
                ucb=min(1., max(float(hi), mean + floor)),
                sd_diff=float(math.sqrt(np.mean(variances))), standard_error=se,
                ci_method="stratified_paired_percentile_bootstrap_with_width_floor",
                alpha=alpha, blocks_per_opponent=counts,
                # Planning diagnostic only. Never expands the current sample.
                approximate_blocks_for_1pp=max(4, math.ceil(
                    1.96 ** 2 * sum(variances) / (len(counts) ** 2 * .01 ** 2))))


def summarize(rows):
    if not rows:
        raise ValueError("empty suite")
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    summary = {group: {"score": float(np.mean([r["score"] for r in values])),
                       "crash": float(np.mean([r["left_alt_loss_rate"] for r in values])),
                       "games": sum(r["games"] for r in values)}
               for group, values in groups.items()}
    return {"score": float(np.mean([v["score"] for v in summary.values()])),
            "crash": float(np.mean([v["crash"] for v in summary.values()])),
            "worst_score": min(v["score"] for v in summary.values()),
            "worst_crash": max(v["crash"] for v in summary.values()),
            "groups": summary}


def compare(candidate, reference, alpha):
    group_names = sorted({r["group"] for r in reference})
    result = {"primary": paired_difference(candidate, reference, alpha=alpha),
              "crash": paired_difference(candidate, reference, "paired_crashes", alpha),
              "groups": {}}
    for group in group_names:
        new = [r for r in candidate if r["group"] == group]
        old = [r for r in reference if r["group"] == group]
        result["groups"][group] = {
            "score": paired_difference(new, old, alpha=alpha),
            "crash": paired_difference(new, old, "paired_crashes", alpha)}
    result["safe"] = all(v["score"]["lcb"] >= -.03 and v["crash"]["ucb"] <= .01
                         for v in result["groups"].values())
    return result


def counter_ucb(row, alpha):
    """Distribution-free Hoeffding bound for mirrored block score in [0,1]."""
    scores = np.asarray(row["paired_scores"], float)
    if len(scores) != row["paired_blocks"] or not np.isfinite(scores).all():
        raise ValueError("invalid counter blocks")
    return min(1., float(1 - scores.mean()) + math.sqrt(math.log(1 / alpha) / (2 * len(scores))))


def polish_value(iteration, start, length, initial=1e-4, final=3e-5):
    fraction = min(1., max(0., (iteration - start - 1) / max(1, length - 1)))
    return initial + fraction * (final - initial)


def finishing_profile(state, iteration):
    if state.get("phase") != "polish":
        return None
    return state.get("config", {}).get("finish_plan", {}).get("side_profiles", {}).get(str(iteration))


def finishing_value(state, iteration):
    config = state["config"]
    plan = config.get("finish_plan")
    if not plan:
        return polish_value(iteration, state["polish_start"], config["polish_iterations"])
    start = plan["decay_start"]
    end = config["normal_end"] + config["polish_iterations"]
    return polish_value(iteration, start - 1, end - start + 1,
                        initial=plan["initial_value"], final=plan["final_value"])


def validate_finish_plan(config, milestone_period):
    plan = config.get("finish_plan")
    if not plan:
        return
    start, end = config["normal_end"], config["normal_end"] + config["polish_iterations"]
    if config.get("early_stop_enabled", True):
        raise ValueError("absolute finishing schedule requires early stopping disabled")
    if not start < plan["decay_start"] < end:
        raise ValueError("finishing decay must be inside finishing budget")
    if not 0 < plan["final_value"] <= plan["initial_value"]:
        raise ValueError("invalid finishing learning-rate/entropy endpoints")
    if not 16 < plan["archive_audit_cap"] <= 128:
        raise ValueError("finishing archive audit cap must be in (16,128]")
    for iteration, profile in plan["side_profiles"].items():
        if (profile not in ("altitude_hunt", "standard", "attack")
                or not start < int(iteration) <= plan["decay_start"]
                or milestone_period <= 0 or int(iteration) % milestone_period):
            raise ValueError("invalid finishing side profile or milestone")


class TrainingStopController:
    def __init__(self, trainer, config, save_path, log_callback=None):
        self.trainer, self.config = trainer, copy.deepcopy(config)
        self.save_path = str(save_path)
        self.root = Path(save_path).parent / "training_stop"
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_callback = log_callback
        if config["protocol"] != PROTOCOL or trainer.env.scenario != "three_nine":
            raise ValueError("stop protocol/scenario mismatch")
        suite = config["suite"]
        groups = {r["group"] for r in suite}
        if (len(suite) != 12 or len(groups) != 4
                or any(sum(r["group"] == g for r in suite) != 3 for g in groups)
                or len({r["sha256"] for r in suite}) != 12):
            raise ValueError("requires twelve distinct frozen opponents in four equal groups")
        if (config["normal_end"] + config["polish_iterations"] != trainer.cfg.total_iterations
                or config["polish_iterations"] <= 0):
            raise ValueError("explicit total budget must reserve positive polish iterations")
        validate_finish_plan(config, getattr(trainer.cfg, "milestone_period", 500))
        for spec in suite:
            rec = trainer.archive.records[int(spec["id"])]
            if rec["sha256"] != spec["sha256"]:
                raise ValueError("frozen suite archive metadata changed")
            trainer.archive.load_policy(spec["id"])
        previous = getattr(trainer, "training_stop_state", None)
        if not isinstance(config.get("early_stop_enabled", True), bool):
            raise ValueError("early_stop_enabled must be boolean")
        early_stop_changed = False
        if previous and previous["config"] != config:
            old_config = copy.deepcopy(previous["config"])
            new_config = copy.deepcopy(config)
            old_enabled = old_config.pop("early_stop_enabled", True)
            new_enabled = new_config.pop("early_stop_enabled", True)
            if old_config != new_config or (old_enabled != new_enabled and previous["phase"] != "learn"):
                raise ValueError("stop config changed on resume; no silent reset of evidence")
            early_stop_changed = old_enabled != new_enabled
        self.state = copy.deepcopy(previous) if previous else {
            "config": config, "phase": "learn", "baseline_iteration": None,
            "window_start": None, "window_reference": None, "evaluations": {},
            "candidates": {}, "next_evaluation": None, "confirmations": [],
            "polish_start": None, "polish_end": None, "stop_reason": None,
            "champion": None, "evaluation_elapsed_seconds": 0., "last_tick": None}
        decision_protocol = self.state.get("decision_protocol")
        if decision_protocol not in (None,"paired_gate_practical_v2"):
            raise ValueError("unknown stopping decision protocol")
        if decision_protocol is None and self.state["confirmations"]:
            raise ValueError("decision protocol change requires review of existing confirmations")
        self.state["decision_protocol"] = "paired_gate_practical_v2"
        saved = self.state["champion"] or {}
        # Reuse the existing manager, but explicitly do not publish this pointer
        # to shadow.champion, which would change strategic query planning.
        self.champion = ChampionManager(
            ChampionConfig(primary_noninferiority_margin=.03,
                           safety_noninferiority_margin=.01,
                           worst_cluster_noninferiority_margin=.03), **saved)
        self.trainer.training_stop_state = copy.deepcopy(self.state)

        self.state["config"] = copy.deepcopy(config)
        self.trainer.training_stop_state = copy.deepcopy(self.state)
        if early_stop_changed:
            self.event("early_stop_policy_changed", enabled=config["early_stop_enabled"],
                       evaluation_preserved=True)
            self.persist()

    def persist(self):
        self.state["champion"] = self.champion.state_dict()
        self.trainer.training_stop_state = copy.deepcopy(self.state)
        self.trainer.save(self.save_path)
        # This is a human-readable mirror, never an alternate resume authority.
        json_save(self.root / "state.json", self.state)
        json_save(self.root / "selection.json", {
            "champion_iteration": self.champion.champion_id,
            "champion_artifact": self.state["candidates"].get(str(self.champion.champion_id)),
            "frontier": self.champion.state_dict()["frontier"],
            "routine_nominees_unconfirmed": self.nominees(-1, 5),
            "phase": self.state["phase"],
            "note": "routine rankings are candidates, not confirmed promotions"})

    def event(self, name, **values):
        record = {"event": name, "iteration": int(self.trainer.iteration), **values}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("[training-stop] " + json.dumps(record, allow_nan=False), flush=True)
        if self.log_callback:
            try:
                self.log_callback({"iteration": record["iteration"], **{
                    "training_stop/" + k: v for k, v in record.items()
                    if k != "iteration" and isinstance(v, (int, float, str))}})
            except Exception as exc:
                print(f"[training-stop] W&B log error: {exc}", flush=True)

    def capture(self):
        identity = int(self.trainer.iteration)
        key = str(identity)
        if key not in self.state["candidates"]:
            path = self.root / f"candidate_{identity}.pt"
            temp = path.with_suffix(".tmp")
            torch.save({**self.trainer._policy_bundle(), "iteration": identity,
                        "cfg": copy.deepcopy(vars(self.trainer.cfg)),
                        "training_stop_protocol": PROTOCOL}, temp)
            os.replace(temp, path)
            self.state["candidates"][key] = {"path": str(path), "sha256": digest(path)}
            self.persist()
        return identity

    def evaluate(self, identity, bank, blocks, opponents=None):
        spec = self.state["candidates"][str(identity)]
        if digest(spec["path"]) != spec["sha256"]:
            raise ValueError("evaluation candidate modified or missing")
        bundle = torch.load(spec["path"], map_location="cpu", weights_only=False)
        rows = []
        start = time.monotonic()
        for position, opponent in enumerate(opponents or self.config["suite"]):
            seed = int(bank) + position * 1009
            key = hashlib.sha256(json.dumps([spec["sha256"], opponent["sha256"],
                                            seed, blocks, PROTOCOL]).encode()).hexdigest()
            path = self.root / "evaluation_cache" / (key + ".json")
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                rec = self.trainer.archive.records[int(opponent["id"])]
                if rec["sha256"] != opponent["sha256"]:
                    raise ValueError("evaluation opponent changed")
                result = self.trainer._evaluate_pair(
                    bundle, self.trainer.archive.load_policy(opponent["id"]),
                    blocks, seed_block=seed)
                row = {**result, "opponent_id": opponent["id"],
                       "opponent_hash": opponent["sha256"], "candidate_hash": spec["sha256"],
                       "group": opponent["group"]}
                if (row["games"] != 2 * blocks or row["paired_blocks"] != blocks
                        or not row["stochastic"] or len(row["paired_crashes"]) != blocks):
                    raise ValueError("incomplete stop evaluation")
                json_save(path, row)
            if (row["candidate_hash"] != spec["sha256"]
                    or row["opponent_hash"] != opponent["sha256"]
                    or row["seed_block"] != seed or row["paired_blocks"] != blocks):
                raise ValueError("evaluation cache identity mismatch")
            rows.append(row)
        elapsed = time.monotonic() - start
        self.state["evaluation_elapsed_seconds"] += elapsed
        self.event("evaluation", candidate=identity, bank=bank, blocks=blocks,
                   elapsed_seconds=elapsed, score=summarize(rows)["score"],
                   worst_crash=summarize(rows)["worst_crash"])
        return rows

    def routine(self, identity):
        key = str(identity)
        if key not in self.state["evaluations"]:
            rows = self.evaluate(identity, 190700000, 64)
            self.state["evaluations"][key] = rows
            reference = self.state.get("window_reference")
            if reference is not None and reference != identity:
                diff = paired_difference(rows, self.state["evaluations"][str(reference)])
                self.state.setdefault("routine_differences", {})[key] = diff
                self.event("paired_difference", candidate=identity, reference=reference,
                           delta=diff["mean"], sd_diff=diff["sd_diff"],
                           standard_error=diff["standard_error"], lcb=diff["lcb"], ucb=diff["ucb"],
                           approximate_blocks_for_1pp=diff["approximate_blocks_for_1pp"])
            self.persist()
        return self.state["evaluations"][key]

    def recent_counters(self):
        history = self.trainer.exploiter_history[-4:]
        if len(history) != 4 or len({h.get("archive_id") for h in history}) != 4:
            raise ValueError("four recent completed counter artifacts required")
        return [{"id": h["archive_id"], "group": "recent_counter",
                 "sha256": self.trainer.archive.records[h["archive_id"]]["sha256"]}
                for h in history]

    def strategic_panel(self):
        """Six representatives; Nash mass selects coverage, not a skill score."""
        records = self.trainer.archive.records
        entries = [e for e in self.trainer.pool.entries
                   if e.get("role") in ("core", "challenger")
                   and e.get("archive_id") in records]
        mass = sorted(entries, key=lambda e: (-float(records[e["archive_id"]].get("nash_mass",0)),e["archive_id"]))
        hard = sorted(entries, key=lambda e: (float(e.get("ema",.5)),e["archive_id"]))
        active = {e["archive_id"] for e in entries}
        historical = sorted((r for r in records.values() if r["id"] not in active
                             and r.get("metrics",{}).get("last_historical_audit_iteration") is not None),
                            key=lambda r: (-int(r["metrics"]["last_historical_audit_iteration"]),r["id"]))
        picked = []
        for ids in ([e["archive_id"] for e in mass], [e["archive_id"] for e in hard],
                    [r["id"] for r in historical]):
            count=0
            for identity in ids:
                if identity not in picked:
                    picked.append(identity); count += 1
                    if count == 2: break
        for e in hard:
            if len(picked) >= 6: break
            if e["archive_id"] not in picked: picked.append(e["archive_id"])
        if len(picked) != 6:
            raise ValueError("six strategic/history opponents required; no silent pass")
        return [{"id": i,"sha256": records[i]["sha256"],"group":"strategic_panel"} for i in picked]

    def evidence(self, identity, rows, difference):
        s = summarize(rows)
        group_lcbs = [v["score"]["lcb"] for v in difference["groups"].values()]
        # The worst of per-group differences protects EVERY group, even when
        # the identity of the lowest-scoring group changes between policies.
        return PromotionEvidence(
            candidate_id=identity, evaluation_suite_version="champion_suite_v1",
            protocol_version="active_league_vnext_100k_v2",
            primary_difference_lcb=difference["primary"]["lcb"],
            safety_difference_lcb=-difference["crash"]["ucb"],
            worst_cluster_difference_lcb=min(group_lcbs),
            heldout_difference_lcb=difference["primary"]["lcb"],
            redteam_difference_lcb=difference.get("strategic",{"lcb":min(group_lcbs)})["lcb"],
            paired_blocks=min(r["paired_blocks"] for r in rows),
            primary_metric=s["score"], worst_cluster_metric=s["worst_score"],
            heldout_metric=s["score"], safety_metric=1 - s["worst_crash"],
            critical_regressions=() if difference["safe"] else ("group_noninferiority_failed",))

    def confirmation(self, identities, reference, bank, allow_plateau):
        for receipt in self.state["confirmations"]:
            if receipt["bank"] == bank:
                return receipt["plateau_confirmed"]
        # Practical per-look comparisons, not an anytime campaign certificate.
        # Correct for candidates actually compared. Requiring ALL safety gates
        # is an intersection decision; do not also divide alpha by every axis
        # and every future look, which made finite-sample NI nearly impossible.
        alpha = .05 / max(1,len(set(identities)-{reference}))
        panels = self.state.setdefault("confirmation_panels", {})
        if str(bank) not in panels:
            panels[str(bank)] = {"strategic": self.strategic_panel(),
                                 "counters": self.recent_counters()}
            self.persist()
        panel = panels[str(bank)]["strategic"]
        old = self.evaluate(reference, bank, 256)
        old_strategic = self.evaluate(reference, bank + 200000, 256, panel)
        comparisons, evaluated = {}, {reference: old}
        selected = reference
        best_selection_score = .75*summarize(old)["score"] + .25*summarize(old_strategic)["score"]
        for identity in identities:
            if identity == reference:
                continue
            rows = self.evaluate(identity, bank, 256)
            evaluated[identity] = rows
            d = compare(rows, old, alpha)
            dynamic = self.evaluate(identity, bank + 200000, 256, panel)
            d["strategic"] = paired_difference(dynamic, old_strategic, alpha=alpha)
            d["strategic_opponents"] = [
                {"score": paired_difference([n],[o],alpha=alpha),
                 "crash": paired_difference([n],[o],"paired_crashes",alpha)}
                for n,o in zip(dynamic,old_strategic)]
            d["safe"] = d["safe"] and all(
                item["score"]["lcb"] >= -.05 and item["crash"]["ucb"] <= .02
                for item in d["strategic_opponents"])
            comparisons[str(identity)] = d
            evidence = self.evidence(identity, rows, d)
            decision = self.champion.evaluate(evidence)
            # Only the winner of this fixed family may be committed; manager
            # decisions are recomputed after selection to avoid stale pointers.
            selection_score = .75*summarize(rows)["score"] + .25*summarize(dynamic)["score"]
            if decision.accepted and selection_score > best_selection_score:
                selected = identity
                best_selection_score = selection_score
            if decision.frontier_only and d["safe"]:
                self.champion.consider_for_frontier(evidence)
        counter_rows = self.evaluate(selected, bank + 500000, 256, panels[str(bank)]["counters"])
        counter_protection = None
        proposed_selected = selected
        if selected != reference:
            reference_counters = self.evaluate(reference, bank + 500000, 256,
                                               panels[str(bank)]["counters"])
            counter_protection = compare(counter_rows, reference_counters, alpha)
            if not counter_protection["safe"]:
                selected, counter_rows = reference, reference_counters
            else:
                d = comparisons[str(selected)]
                decision = self.champion.evaluate(self.evidence(selected, evaluated[selected], d))
                self.champion.commit(decision, iteration=self.trainer.iteration)
        counter_bounds = [counter_ucb(r, .05 / (6 * 4)) for r in counter_rows]
        counters_ok = all(value <= .60 for value in counter_bounds)
        # Not beating an old champion is not stagnation: a recovering learner
        # may still be improving rapidly below that high-water mark. Compare
        # the already-evaluated recent candidates too; no extra GPU matches.
        trend = []
        if allow_plateau and len(identities) == 3:
            chronological = sorted(identities)
            trend = [paired_difference(evaluated[c], evaluated[chronological[0]], alpha=alpha)
                     for c in chronological[1:]]
        # A missing window, missing comparison, or absent promotion is NEVER
        # itself plateau evidence. Candidate comparisons are real evaluations.
        plateau = (allow_plateau and len(comparisons) == 3
                   and len(trend) == 2 and all(d["ucb"] < .02 for d in trend)
                   and all(d["primary"]["ucb"] < .02 for d in comparisons.values())
                   and all(d["strategic"]["ucb"] < .02 for d in comparisons.values())
                   and (selected == reference or comparisons[str(selected)]["safe"])
                   and counters_ok)
        receipt = {"iteration": int(self.trainer.iteration), "bank": bank,
                   "reference": reference, "selected": selected, "comparisons": comparisons,
                   "proposed_selected": proposed_selected, "counter_protection": counter_protection,
                   "recent_candidate_trend": trend,
                   "counter_ucbs": counter_bounds, "counter_gate_passed": counters_ok,
                   "plateau_confirmed": bool(plateau), "per_comparison_alpha": alpha}
        self.state["confirmations"].append(receipt)
        self.event("confirmation", plateau_confirmed=bool(plateau), champion=selected,
                   counter_gate_passed=counters_ok, worst_counter_ucb=max(counter_bounds))
        self.persist()
        return bool(plateau)

    def begin_polish(self, reason):
        iteration = int(self.trainer.iteration)
        if self.config.get("finish_plan"):
            self.preserve_restart_point()
        self.state.update(phase="polish", polish_start=iteration,
                          polish_end=iteration + self.config["polish_iterations"],
                          stop_reason=reason, pre_polish_champion=self.champion.champion_id,
                          next_evaluation=iteration + 1000)
        # Continue the CURRENT complete PPO state, never splice old weights
        # into new optimizer/episodes. The frozen champion remains selectable.
        self.event("polish_started", reason=reason, end_iteration=self.state["polish_end"],
                   continuation="current_complete_ppo_state", champion=self.champion.champion_id)
        self.persist()


    def preserve_restart_point(self):
        """Immutable complete boundary checkpoint plus its external artifacts.

        A manifest is the commit marker; an interrupted copy is resumable.
        Never prune existing keep-iteration checkpoints or archive policies.
        """
        iteration = int(self.trainer.iteration)
        if iteration != self.config["normal_end"]:
            raise ValueError("finishing restart point must be at the configured boundary")
        destination = self.root.parent / "restart_points" / f"iter_{iteration}"
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for spec in manifest["files"]:
                if digest(destination / spec["relative_path"]) != spec["sha256"]:
                    raise ValueError("protected restart point integrity failure")
            return
        destination.mkdir(parents=True, exist_ok=True)
        self.persist()
        sources = [(Path(self.save_path), Path("checkpoint.pt"))]
        for name in sorted({r["file"] for r in self.trainer.archive.records.values()}):
            sources.append((self.trainer.archive.policy_dir / name,
                            Path("league/policies") / name))
        # Candidates/cache remain useful for replaying the saved stop controller.
        sources.extend((p, Path("training_stop") / p.relative_to(self.root))
                       for p in self.root.rglob("*") if p.is_file() and p.suffix != ".tmp")
        code_root = Path(__file__).parent
        sources.extend((p, Path("code/cuda_fdm") / p.relative_to(code_root))
                       for p in code_root.rglob("*.py"))
        files = []
        for source, relative in sources:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = digest(source)
            if not target.exists() or digest(target) != expected:
                temp = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(source, temp)
                os.replace(temp, target)
            if digest(target) != expected:
                raise ValueError("restart artifact copy mismatch")
            files.append({"relative_path": str(relative), "source_path": str(source),
                          "sha256": expected})
        json_save(destination / "config.json", self.config)
        json_save(manifest_path, {"iteration": iteration, "files": files,
                  "config": self.config, "complete": True,
                  "restore_note": "Full PPO checkpoint includes optimizer/RNG/runtime/league. "
                  "Restore dependency paths using source_path mapping in an isolated run; "
                  "do not overwrite a running campaign."})
        self.event("restart_point_preserved", path=str(destination), files=len(files))

    def tick(self):
        if not self.trainer.checkpoint_safe:
            raise RuntimeError("stop evaluation requires a committed iteration boundary")
        it = int(self.trainer.iteration)
        if self.state["phase"] == "done" or (self.state["last_tick"] == it
                                               and self.state.get("pending_tick") is None):
            return
        if self.state.get("pending_tick") not in (None,it):
            raise ValueError("pending decision must finish before another PPO iteration")
        self.state["pending_tick"] = it
        self.persist()
        self._tick()

    def _tick(self):
        if not self.trainer.checkpoint_safe:
            raise RuntimeError("stop evaluation requires a committed iteration boundary")
        it = int(self.trainer.iteration)
        if self.state["phase"] == "done" or self.state["last_tick"] == it:
            return
        if self.state["baseline_iteration"] is None:
            candidate = self.capture()
            self.routine(candidate)
            # Bootstrap initialization is NOT a promotion or convergence claim.
            self.champion.champion_id = candidate
            self.state.update(baseline_iteration=it, window_start=it,
                              window_reference=candidate, next_evaluation=it + 2000)
            self.event("baseline_initialized", candidate=candidate, first_plateau_check=it + 6000)
        elif self.state["phase"] == "learn":
            if (it >= self.state["next_evaluation"] or it >= self.config["normal_end"]
                    or self.state.get("pending_tick") == it):
                candidate = self.capture()
                self.routine(candidate)
                self.state["next_evaluation"] = it + 2000
                normal_looks = sum(200000000 <= r["bank"] < 300000000
                                   for r in self.state["confirmations"])
                if it - self.state["window_start"] >= 6000 and normal_looks < 4:
                    candidates = sorted(int(k) for k in self.state["evaluations"]
                                        if self.state["window_start"] < int(k) <= it)[-3:]
                    reference = self.state["window_reference"]
                    routine_ref = self.routine(reference)
                    diagnostics = [paired_difference(self.routine(c), routine_ref) for c in candidates]
                    recent_growth = (max(summarize(self.routine(c))["score"] for c in candidates[1:])
                                     - summarize(self.routine(candidates[0]))["score"]
                                     if len(candidates) == 3 else float("inf"))
                    plausible = (len(candidates) == 3 and all(d["mean"] < .01 for d in diagnostics)
                                 and recent_growth < .01)
                    self.event("window_diagnostics", max_sd_diff=max(d["sd_diff"] for d in diagnostics),
                               max_gain=max(d["mean"] for d in diagnostics), plausible=plausible,
                               recent_growth=recent_growth if math.isfinite(recent_growth) else None)
                    # Spend the four-model confirmation budget only when a
                    # plateau is plausible. Otherwise compare one nominee to
                    # the champion (roughly half the GPU evaluation work).
                    chosen = candidates if plausible else self.nominees(self.state["window_start"],1)
                    confirmed = self.confirmation(chosen, reference, 200000000 + it * 20, plausible)
                    self.state.update(window_start=it, window_reference=self.champion.champion_id)
                    if confirmed and self.config.get("early_stop_enabled", True):
                        self.begin_polish("paired_plateau_and_counter_gate")
                    elif confirmed:
                        self.event("plateau_observed_no_early_stop", early_stop_enabled=False)
                if self.state["phase"] == "learn" and it >= self.config["normal_end"]:
                    candidates = self.nominees(self.state["baseline_iteration"], 3)
                    self.confirmation(candidates, self.champion.champion_id,
                                      300000000, False)
                    self.begin_polish("normal_iteration_budget_not_convergence")
        elif self.state["phase"] == "polish":
            if it >= self.state["next_evaluation"]:
                self.routine(self.capture())
                self.state["next_evaluation"] = it + 1000
            if it >= self.state["polish_end"]:
                self.routine(self.capture())
                if self.config.get("finish_plan"):
                    # User will run a separate tournament including external
                    # harddeck/dive/4499 artifacts. Do not choose a final model
                    # or launch that tournament automatically at the budget.
                    self.state.update(phase="done", stop_reason="training_budget_complete_tournament_deferred",
                                      tournament_pending=True)
                    json_save(self.root / "training_completion.json", {
                        "iteration": it, "tournament_pending": True,
                        "checkpoint": self.save_path,
                        "champion_iteration": self.champion.champion_id,
                        "candidates": self.state["candidates"],
                        "note": "Training complete; champion is provisional, final tournament deferred."})
                    self.event("training_completed", end_iteration=it, tournament_pending=True)
                else:
                    candidates = self.nominees(self.state["polish_start"], 3)
                    self.confirmation(candidates, self.champion.champion_id, 400000000, False)
                    # Legacy campaigns retain their original final evaluation.
                    self.state["phase"] = "final"
                    self.state["final_candidate"] = self.champion.champion_id
                self.persist()
        if self.state["phase"] == "final":
            selected = self.state["final_candidate"]
            rows = self.evaluate(selected, 500000000, 256)
            counters = self.evaluate(selected, 501000000, 256, self.recent_counters())
            source = self.state["candidates"][str(selected)]
            target = self.root / "best_model.pt"
            temp = target.with_suffix(".tmp")
            shutil.copy2(source["path"], temp)
            os.replace(temp, target)
            if digest(target) != source["sha256"]:
                raise ValueError("best model copy hash mismatch")
            final = {"selected_iteration": selected, "stop_iteration": it,
                     "stop_reason": self.state["stop_reason"], "suite": summarize(rows),
                     "counter_scores": [r["score"] for r in counters],
                     "counter_ucbs": [counter_ucb(r, .05/4) for r in counters],
                     "final_counter_gate_passed": all(counter_ucb(r, .05/4) <= .60 for r in counters),
                     "artifact": str(target), "sha256": source["sha256"],
                     "note": "fixed-suite practical stop; not a Nash/exploitability certificate"}
            json_save(self.root / "final_report.json", final)
            self.state.update(phase="done", final_report=final)
            self.event("completed", selected_iteration=selected, reason=self.state["stop_reason"])
        self.state["last_tick"] = it
        self.state["pending_tick"] = None
        self.persist()

    def nominees(self, after, cap):
        candidates = [int(k) for k in self.state["evaluations"] if int(k) > after]
        candidates.sort(key=lambda c: (summarize(self.state["evaluations"][str(c)])["score"], c), reverse=True)
        return candidates[:cap]

    def due(self):
        return (self.state["phase"] in ("done", "final")
                or self.state.get("pending_tick") is not None
                or self.state["baseline_iteration"] is None
                or self.trainer.iteration >= self.state["next_evaluation"]
                or (self.state["phase"] == "learn"
                    and self.trainer.iteration >= self.config["normal_end"]))

    def before_iteration(self, iteration):
        return self.state["phase"] != "done" and self.state.get("pending_tick") is None
