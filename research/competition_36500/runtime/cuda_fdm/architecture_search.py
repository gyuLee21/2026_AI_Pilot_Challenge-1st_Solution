"""Resumable, single-GPU search followed by the approved fresh 10,000-iteration run.

Create ROOT/STOP to stop at a completed training iteration or between evaluation jobs.
All training checkpoints, evaluation specifications, results and decisions stay local.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "gru512_h256": dict(architecture="gru", hidden="512,512,512", gru_size=256, encoder_depth=2),
    "mlp672": dict(architecture="mlp", hidden="672,672,672", gru_size=0, encoder_depth=2),
    "mlp384": dict(architecture="mlp", hidden="384,384,384", gru_size=0, encoder_depth=2),
    "mlp512": dict(architecture="mlp", hidden="512,512,512", gru_size=0, encoder_depth=2),
    "mlp512_d2": dict(architecture="mlp", hidden="512,512", gru_size=0, encoder_depth=2),
    "mlp768": dict(architecture="mlp", hidden="768,768,768", gru_size=0, encoder_depth=2),
    "mlp672_d4": dict(architecture="mlp", hidden="672,672,672,672", gru_size=0, encoder_depth=2),
    "gru384_h128": dict(architecture="gru", hidden="384,384,384", gru_size=128, encoder_depth=2),
    "gru512_h128": dict(architecture="gru", hidden="512,512,512", gru_size=128, encoder_depth=2),
    "gru512_h256_d1": dict(architecture="gru", hidden="512,512,512", gru_size=256, encoder_depth=1),
}
KEEP = "100,200,400,500,800,1000,1500"
ENTITY = "leeai021213-ajou-university"
MLP_EXTENSION = {
    "version": 1,
    "enabled": True,
    "models": {name: MODELS[name] for name in ("mlp768", "mlp672_d4")},
    "preserved_top_two": ["mlp672", "mlp512_d2"],
    "comparison_baseline": "mlp672",
    "comparison_iterations": 800,
    "intermediate_snapshot": 400,
    "comparison_phase": "mlp_extension_800",
    "comparison_seed": 74001,
    "confirmation_iterations": 1500,
    "training_seeds": [0, 1, 2],
    "confirmation_phase": "confirmation_expanded_1500",
    "confirmation_seed": 94001,
    "curve_phase_prefix": "confirmation_expanded_curve",
    "games_per_scenario": 256,
    "scenario_weights": {"three_nine": .75, "headon": .25},
    "selection_weights": {"crossplay": .75, "frozen_references": .25},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def checkpoint_iteration(path):
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False)["iteration"])


def verify_training_config(path, model, seed, final):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
    if saved.get("aux_pred", False):
        raise ValueError("legacy architecture cohort is auxiliary-OFF; a new protocol is required")
    expected = dict(model, seed=seed, gamma=.997, gae_lambda=.95,
                    milestone_period=500 if final else 0, exploiter_iters=1000 if final else 0)
    expected["hidden"] = tuple(int(x) for x in model["hidden"].split(","))
    for key, value in expected.items():
        actual = saved.get(key)
        if key == "hidden":
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"checkpoint config mismatch {path}: {key}: {actual!r} != {value!r}")


def verify_evaluation_complete(result, spec):
    """A partial or incompatible suite must never produce a selection ranking."""
    if not isinstance(result, dict) or result.get("spec") != spec:
        raise FloatingPointError("evaluation result does not match its frozen specification")
    expected = {}
    for pair in spec["pairs"]:
        a, b = pair[:2]
        reset = bool(pair[2]) if len(pair) > 2 else False
        for scenario in ("three_nine", "headon"):
            expected[f"{a}|{b}|{scenario}|reset{int(reset)}"] = (a, b, scenario, reset)
    matches = result.get("matches")
    if len(expected) != 2 * len(spec["pairs"]) or not isinstance(matches, dict) or set(matches) != set(expected):
        raise FloatingPointError("evaluation suite is incomplete or has unexpected matches")
    for key, (a, b, scenario, reset) in expected.items():
        match = matches[key]
        if not isinstance(match, dict) or any(match.get(field) != value for field, value in
                (("a", a), ("b", b), ("scenario", scenario), ("reset_a", reset))):
            raise FloatingPointError(f"evaluation match metadata mismatch: {key}")
        records = match.get("records")
        if not isinstance(records, list) or len(records) != spec["games"]:
            raise FloatingPointError(f"evaluation game count mismatch: {key}")
        for record in records:
            if not isinstance(record, dict) or any(
                    not isinstance(record.get(field), (int, float)) or not math.isfinite(record[field])
                    for field in ("score", "damage_diff")):
                raise FloatingPointError(f"non-finite evaluation ranking inputs: {key}")


class Search:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stop = self.folder / "STOP"
        self.state = self.folder / "status.json"
        self.reference = ROOT / "runs/architecture_search/preserved_long_run/gru512_gru256_iter0100.pt"
        if not self.reference.exists():
            raise FileNotFoundError(self.reference)
        self.manifest_path = self.folder / "manifest.json"
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": MODELS,
            "architecture_iters": 800, "screen_iters": 400, "confirm_iters": 1500,
            "training_seeds": [0, 1, 2], "games_per_scenario": 256,
            "scenario_weights": {"three_nine": .75, "headon": .25},
            "selection_weights": {"crossplay": .75, "frozen_references": .25},
            "evaluation_seed": {"architecture": 72001, "screen": 73001, "confirm": 94001},
            "constraints": {"policy_hz": 10, "obs": 184, "gamma": .997,
                            "old_models_bt_mpc": False, "alphastar": False},
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/"cuda_fdm/ppo_gpu.py", ROOT/"cuda_fdm/train_gpu.py",
                                        ROOT/"cuda_fdm/search_eval.py", ROOT/"claude_code/model.py",
                                        ROOT/"cuda_fdm/search_validate.py", ROOT/"cuda_fdm/gpu_ckpt_to_bundle.py",
                                        ROOT/"cuda_fdm/architecture_search.py", ROOT/"cuda_fdm/rl_env.py",
                                        ROOT/"cuda_fdm/finite_checks.py",
                                        ROOT/"src/dogfight/envs/termination.py",
                                        ROOT/"cuda_fdm/gpu_env.py", ROOT/"cuda_fdm/tests/cuda_rt.py",
                                        ROOT/"cuda_fdm/obs_reward.py", ROOT/"claude_code/my_observation.py",
                                        ROOT/"claude_code/observation_contract.py",
                                        ROOT/"claude_code/action_provider.py",
                                        ROOT/"claude_code/altguard_provider.py",
                                        ROOT/"claude_code/altblend_provider.py",
                                        ROOT/"src/dogfight/unreal/policies.py",
                                        ROOT/"claude_code/my_reward.py",
                                        ROOT/"cuda_fdm/gen/fdm_kernel.cu",
                                        ROOT/"cuda_fdm/gen/fdm.cuh", ROOT/"cuda_fdm/gen/f16_gen.cuh",
                                        ROOT/"cuda_fdm/gen/obs_kernel.cu"]},
        }
        if not self.manifest_path.exists():
            write_json(self.manifest_path, self.manifest)
        self.lock_handle = open(self.folder / "manager.lock", "a+b")
        import msvcrt
        self.lock_handle.seek(0)
        if not self.lock_handle.read(1):
            self.lock_handle.write(b"0"); self.lock_handle.flush()
        self.lock_handle.seek(0)
        msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        self.update("initializing")

    def update(self, phase, **extra):
        data = dict(manager_pid=os.getpid(), phase=phase,
                    updated=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
        write_json(self.state, data)

    def check_integrity(self, child_markers=()):
        markers = [self.folder / "INTEGRITY_HOLD.json",
                   self.folder / "INTEGRITY_FAILURE.json", *child_markers]
        for marker in markers:
            if marker.exists():
                raise FloatingPointError(f"integrity marker blocks experiment: {marker}")

    def mlp_extension(self):
        extension = self.manifest.get("mlp_extension")
        if extension is not None and extension != MLP_EXTENSION:
            raise ValueError("mlp_extension must exactly match the frozen approved MLP_EXTENSION configuration")
        return extension

    @staticmethod
    def child_integrity_markers(args, folder):
        markers = {Path(folder) / "INTEGRITY_FAILURE.json"}
        for flag in ("--save", "--output", "--output-dir"):
            if flag not in args:
                continue
            destination = Path(args[args.index(flag) + 1])
            is_directory = flag == "--output-dir" or (flag == "--output" and not destination.suffix)
            markers.add((destination if is_directory else destination.parent) / "INTEGRITY_FAILURE.json")
        return markers

    def command(self, label, args, log_folder=None, retries=1, allow_failure=False):
        self.check_integrity()
        if self.stop.exists():
            raise InterruptedError("STOP file present")
        for relative, digest in self.manifest["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"source changed during controlled experiment: {relative}")
        folder = Path(log_folder or self.folder / "jobs" / label)
        folder.mkdir(parents=True, exist_ok=True)
        markers = self.child_integrity_markers(args, folder)
        for attempt in range(retries + 1):
            self.check_integrity(markers)
            env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, *args]
            with open(folder / "stdout.log", "a", encoding="utf-8") as out, \
                    open(folder / "stderr.log", "a", encoding="utf-8") as err:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                print(f"[manager] {label} pid={proc.pid} attempt={attempt+1}", flush=True)
                self.update(label, child_pid=proc.pid, command=command, attempt=attempt+1,
                            stdout=str(folder / "stdout.log"), stderr=str(folder / "stderr.log"))
                last_sample = 0.
                while proc.poll() is None:
                    now = time.time()
                    if now-last_sample >= 30:
                        last_sample = now
                        try:
                            telemetry = subprocess.check_output([
                                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
                            with open(self.folder / "gpu_telemetry.csv", "a", encoding="utf-8") as f:
                                f.write(f"{now},{label},{telemetry}\n")
                        except (OSError, subprocess.SubprocessError):
                            pass
                        self.update(label, child_pid=proc.pid, attempt=attempt+1,
                                    stdout=str(folder / "stdout.log"), stderr=str(folder / "stderr.log"))
                    time.sleep(3)
            self.check_integrity(markers)
            if label == "preflight" and proc.returncode != 0:
                raise FloatingPointError(f"preflight failed exit={proc.returncode}; inspect {folder}")
            if self.stop.exists():
                raise InterruptedError("STOP requested; completed current safe boundary")
            if proc.returncode == 0 or allow_failure:
                return proc.returncode
            print(f"[manager] {label} failed exit={proc.returncode}", flush=True)
        raise RuntimeError(f"{label} failed after {retries+1} attempts; inspect {folder}")

    def directory(self, name, seed):
        return self.folder / "training" / name / f"seed{seed}"

    def snapshot(self, name, seed, iteration):
        path = self.directory(name, seed) / f"iter_{iteration:05d}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)

    def train_to(self, name, seed, target, final=False):
        self.check_integrity()
        if final and self.mlp_extension() is not None and not getattr(self, "_expanded_confirmation_complete", False):
            raise RuntimeError("final training requires completed expanded confirmation in this manager run")
        folder = self.folder / "final_10000" if final else self.directory(name, seed)
        self.check_integrity([folder / "INTEGRITY_FAILURE.json"])
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "checkpoint.pt"
        if checkpoint.exists():
            verify_training_config(checkpoint, MODELS[name], seed, final)
        if checkpoint.exists() and checkpoint_iteration(checkpoint) >= target:
            return
        m = MODELS[name]
        # This held historical cohort predates auxiliary learning. Do not let a
        # new CLI default silently change either its comparisons or final run.
        args = ["-m", "cuda_fdm.train_gpu", "--no-aux-pred", "--architecture", m["architecture"],
                "--hidden", m["hidden"], "--seed", str(seed),
                "--iters", str(target), "--nenv", "4096", "--rollout", "64", "--epochs", "4",
                "--minibatches", "8", "--gamma", "0.997", "--gae-lambda", "0.95",
                "--lr", "0.0003", "--ent-coef", "0.001", "--clip", "0.2", "--target-kl", "0.03",
                "--substeps", "6", "--pool-evict-cap", "4",
                "--selfplay-gate-threshold", "0.6", "--selfplay-ema-alpha", "0.1",
                "--pool-sample-temp", "0.3", "--pool-uniform-floor", "0.5",
                "--save", str(checkpoint), "--save-every", "100", "--log", str(folder / "metrics.csv"),
                "--stop-file", str(self.stop)]
        if final:
            args += ["--wandb", "--wandb-entity", ENTITY,
                     "--wandb-expected-username", "leeai021213", "--wandb-project", "AIP contest",
                     "--wandb-group", "architecture-search-final",
                     "--wandb-run-name", f"final10000-{name}-s{seed}",
                     "--milestone-period", "500", "--exploiter-iters", "1000",
                     "--exploiter-win-target", ".7", "--sched-period", "2000",
                     "--keep-iterations", ",".join(map(str, range(500, 10001, 500)))]
        else:
            args += ["--no-wandb", "--milestone-period", "0", "--exploiter-iters", "0", "--sched-period", "0",
                     "--save-runtime", "--keep-iterations", KEEP]
        if checkpoint.exists():
            args += ["--resume", str(checkpoint)]
        # A failed run is resumed only from its last atomically saved complete iteration.
        for attempt in range(2):
            try:
                self.command(f"train-{name}-s{seed}-to{target}", args, folder, retries=0)
                if checkpoint_iteration(checkpoint) != target:
                    raise RuntimeError("training exited before requested iteration")
                return
            except RuntimeError:
                if attempt or self.stop.exists():
                    raise
                if checkpoint.exists() and "--resume" not in args:
                    args += ["--resume", str(checkpoint)]

    def evaluate(self, phase, candidates, refs, seed, extra_pairs=None):
        """candidates: name -> (checkpoint, family). References never influence training."""
        models = {k: v[0] for k, v in candidates.items()} | refs
        pairs = [[a, b] for a, b in itertools.combinations(candidates, 2)
                 if candidates[a][1] != candidates[b][1]]
        pairs += [[a, r] for a in candidates for r in refs]
        pairs += extra_pairs or []
        spec = dict(models=models, pairs=pairs, seed=seed, games=256,
                    candidates={k: v[1] for k, v in candidates.items()}, references=list(refs))
        folder = self.folder / "evaluation" / phase
        path = folder / "spec.json"
        if path.exists() and read_json(path) != spec:
            raise ValueError(f"evaluation specification changed: {path}")
        write_json(path, spec)
        self.command(f"eval-{phase}", ["-m", "cuda_fdm.search_eval", "--spec", str(path),
                                      "--output", str(folder / "results.json")], folder)
        try:
            result = read_json(folder / "results.json")
        except (OSError, ValueError) as exc:
            raise FloatingPointError(f"evaluation did not produce valid results: {folder}") from exc
        verify_evaluation_complete(result, spec)
        rows = []
        for name, (_, family) in candidates.items():
            cross, fixed, wins, damage = [], [], [], []
            for match in result["matches"].values():
                if match["reset_a"] or name not in (match["a"], match["b"]):
                    continue
                inverse = name == match["b"]
                other = match["a"] if inverse else match["b"]
                w = .75 if match["scenario"] == "three_nine" else .25
                records = match["records"]
                scores = [1-r["score"] if inverse else r["score"] for r in records]
                mean = sum(scores)/len(scores)
                (fixed if other in refs else cross).append((mean, w))
                wins.append((sum(s == 1 for s in scores)/len(scores), w))
                damage.append((sum(r["damage_diff"] for r in records)/len(records)*(-1 if inverse else 1), w))
            average = lambda values: sum(v*w for v,w in values)/sum(w for _,w in values) if values else .5
            cs, fs = average(cross), average(fixed)
            rows.append(dict(name=name, family=family, crossplay=cs, reference=fs,
                             score=.75*cs+.25*fs, win_rate=average(wins), damage_diff=average(damage)))
        groups = {}
        for row in rows:
            groups.setdefault(row["family"], []).append(row)
        ranking = []
        for family, group in groups.items():
            item = {k: sum(x[k] for x in group)/len(group)
                    for k in ("score", "crossplay", "reference", "win_rate", "damage_diff")}
            item.update(family=family, seed_scores=[x["score"] for x in group])
            if len(group) == 3:
                mean = item["score"]
                sd = math.sqrt(sum((x["score"]-mean)**2 for x in group)/2)
                radius = 4.30265273*sd/math.sqrt(3)
                item["seed_mean_ci95"] = [max(0., mean-radius), min(1., mean+radius)]
            else:
                item["seed_mean_ci95"] = None
            ranking.append(item)
        ranking.sort(key=lambda x: (x["score"], x["win_rate"], x["damage_diff"], x["reference"]), reverse=True)
        write_json(folder / "ranking.json", {"ranking": ranking, "per_seed": rows,
                    "note": "75% crossplay + 25% fixed references, scenario weighted 75/25. Per-match CI in results."})
        with open(folder / "ranking.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return ranking

    def run(self):
        self.check_integrity()
        extension = self.mlp_extension()
        self._expanded_confirmation_complete = False
        if extension is not None:
            architecture = read_json(self.folder / "architecture_decision.json")
            screen = read_json(self.folder / "screen_decision.json")
            if architecture["selected"] != "mlp" or screen["top_two"] != extension["preserved_top_two"]:
                raise ValueError("MLP extension requires the preserved MLP decision and original top two")
        self.command("preflight", ["-m", "cuda_fdm.search_validate", "--output", str(self.folder / "preflight")])
        try:
            passed = read_json(self.folder / "preflight/result.json")["passed"] is True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise FloatingPointError("preflight did not produce a valid passing result") from exc
        if not passed:
            raise FloatingPointError("preflight failed")
        bases = ["gru512_h256", "mlp672"]
        if extension is None:
            for name in bases:
                m = MODELS[name]
                folder = self.folder / "full_smoke" / name
                self.check_integrity([folder / "INTEGRITY_FAILURE.json"])
                if not (folder / "checkpoint.pt").exists():
                    self.command(f"smoke4096-{name}", ["-m", "cuda_fdm.train_gpu", "--no-aux-pred", "--architecture", m["architecture"],
                        "--hidden", m["hidden"], "--nenv", "4096",
                        "--rollout", "64", "--epochs", "4", "--minibatches", "8", "--iters", "2",
                        "--sched-period", "0", "--milestone-period", "0", "--exploiter-iters", "0",
                        "--save", str(folder / "checkpoint.pt"), "--log", str(folder / "metrics.csv"), "--no-wandb"], folder)
                if checkpoint_iteration(folder / "checkpoint.pt") != 2:
                    raise RuntimeError(f"4096-env smoke incomplete: {name}")
            for seed in range(3):
                for name in bases:
                    self.train_to(name, seed, 800)
            refs = {"reference_gru100": str(self.reference)}
            refs |= {f"reference_{name}_200": self.snapshot(name, 0, 200) for name in bases}
            candidates = {f"{name}_s{seed}": (self.snapshot(name, seed, 800), MODELS[name]["architecture"])
                          for name in bases for seed in range(3)}
            ranking = self.evaluate("architecture_800", candidates, refs, 72001)
            family = ranking[0]["family"]
            write_json(self.folder / "architecture_decision.json", {"selected": family, "ranking": ranking})
            names = ["mlp384", "mlp512", "mlp672", "mlp512_d2"] if family == "mlp" else [
                "gru384_h128", "gru512_h128", "gru512_h256", "gru512_h256_d1"]
            for name in names:
                self.train_to(name, 0, 400)
            screen = {f"{name}_s0": (self.snapshot(name, 0, 400), name) for name in names}
            ranking = self.evaluate("size_400", screen, refs, 73001)
            top = [r["family"] for r in ranking[:2]]
            write_json(self.folder / "screen_decision.json", {"top_two": top, "ranking": ranking})
        else:
            family = "mlp"
            names = ["mlp384", "mlp512", "mlp672", "mlp512_d2"]
            top = list(extension["preserved_top_two"])
            refs = {"reference_gru100": str(self.reference)}
            refs |= {f"reference_{name}_200": self.snapshot(name, 0, 200) for name in bases}
            new_names = list(extension["models"])
            for name in new_names:
                self.train_to(name, 0, extension["comparison_iterations"])
            compared = [extension["comparison_baseline"], *new_names]
            candidates = {f"{name}_s0": (self.snapshot(name, 0, extension["comparison_iterations"]), name)
                          for name in compared}
            ranking = self.evaluate(extension["comparison_phase"], candidates, refs, extension["comparison_seed"])
            selected_new = next(item["family"] for item in ranking if item["family"] in new_names)
            top.append(selected_new)
            decision = {"selected_new": selected_new, "preserved_top_two": extension["preserved_top_two"],
                        "confirmation_candidates": top, "ranking": ranking}
            path = self.folder / "mlp_extension_decision.json"
            if path.exists() and read_json(path) != decision:
                raise ValueError(f"MLP extension decision changed: {path}")
            if not path.exists():
                write_json(path, decision)
        for seed in range(3):
            for name in top:
                self.train_to(name, seed, 1500)
        fixed_final = refs | {f"reference_{name}_400": self.snapshot(name, 0, 400) for name in names}
        if extension is not None:
            fixed_final |= {f"reference_{name}_400": self.snapshot(name, 0, 400) for name in extension["models"]}
        confirmation = {f"{name}_s{seed}": (self.snapshot(name, seed, 1500), name)
                        for name in top for seed in range(3)}
        confirmation_phase = extension["confirmation_phase"] if extension is not None else "confirmation_1500"
        ranking = self.evaluate(confirmation_phase, confirmation, fixed_final, 94001)
        self._expanded_confirmation_complete = extension is not None
        # Learning curves use fixed reference sets and never affect the held-out final ranking.
        if extension is None:
            for iteration in (200, 400):
                curve = {f"{name}_s{seed}": (self.snapshot(name, seed, iteration), MODELS[name]["architecture"])
                         for name in bases for seed in range(3)}
                self.evaluate(f"architecture_curve_{iteration}", curve, {"reference_gru100": str(self.reference)}, 72001)
        for iteration in (500, 1000):
            curve = {f"{name}_s{seed}": (self.snapshot(name, seed, iteration), name)
                     for name in top for seed in range(3)}
            prefix = extension["curve_phase_prefix"] if extension is not None else "confirmation_curve"
            self.evaluate(f"{prefix}_{iteration}", curve, fixed_final, 73001)
        if family == "gru":
            for name in top:
                checkpoint = self.snapshot(name, 0, 1500)
                c = {name: (checkpoint, name)}
                self.evaluate(f"memory_{name}", c, refs, 94001,
                              extra_pairs=[[name, r, True] for r in refs])
        eligible = []
        for item in ranking:
            name = item["family"]
            checkpoint = self.snapshot(name, 0, 1500)
            bundle = self.folder / "bundles" / name
            self.command(f"bundle-{name}", ["-m", "cuda_fdm.gpu_ckpt_to_bundle", "--ckpt", checkpoint,
                                           "--output-dir", str(bundle)])
            output = self.folder / "latency" / f"{name}.json"
            self.command(f"latency-{name}", ["-m", "cuda_fdm.search_validate", "--checkpoint", checkpoint,
                                            "--output", str(output)], allow_failure=True)
            measured = read_json(output)
            if measured["passed"]:
                eligible.append(item)
        if not eligible:
            raise RuntimeError("no candidate passed two-agent local 10Hz latency gate")
        winner = eligible[0]["family"]
        decision = {"selected": winner, "model": MODELS[winner], "ranking": ranking,
                    "latency_eligible": [x["family"] for x in eligible],
                    "scope": "best among tested models/opponents; no claim against old strong policies",
                    "final_training": "fresh seed0, 10000 iterations, 10Hz, original PFSP/milestones/exploiters"}
        if extension is not None:
            decision["confirmation_phase"] = confirmation_phase
            decision["mlp_extension_selected_new"] = selected_new
        write_json(self.folder / "final_decision.json", decision)
        report = ["# Architecture search result", "", f"Selected: **{winner}**", "",
                  "|Model|Score|Cross-play|References|Win rate|Damage difference|", "|---|---:|---:|---:|---:|---:|"]
        for r in ranking:
            report.append(f"|{r['family']}|{r['score']:.4f}|{r['crossplay']:.4f}|{r['reference']:.4f}|{r['win_rate']:.4f}|{r['damage_diff']:.4f}|")
        report += ["", "See evaluation/*/ranking.csv and results.json for per-seed results and paired-bootstrap intervals.",
                   "Latency checks include two local control pipelines but exclude networking and server load.",
                   "No 1220/4499/BT/MPC opponents; 10Hz only. Final learner starts from scratch."]
        (self.folder / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        self.train_to(winner, 0, 10000, final=True)
        self.command("final-bundle", ["-m", "cuda_fdm.gpu_ckpt_to_bundle", "--ckpt",
                                      str(self.folder / "final_10000/checkpoint.pt"),
                                      "--output-dir", str(self.folder / "final_10000/bundle")])
        self.update("complete", selected=winner, final_checkpoint=str(self.folder / "final_10000/checkpoint.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/architecture_search/experiment_v1")
    args = ap.parse_args()
    search = Search(args.root)
    try:
        search.run()
    except InterruptedError as exc:
        search.update("paused", reason=str(exc))
        raise SystemExit(0)
    except FloatingPointError as exc:
        search.update("failed", reason=str(exc), integrity_failure=True)
        marker = search.folder / "INTEGRITY_FAILURE.json"
        if not marker.exists():
            write_json(marker, {"reason": str(exc), "manager_pid": os.getpid(),
                                "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        raise
    except Exception as exc:
        search.update("failed", reason=str(exc))
        raise


if __name__ == "__main__":
    main()
