"""Approved auxiliary-ON MLP search: depth, width, confirmation, fresh main.

This is a NEW reward cohort. It never migrates, unholds or resumes either old cohort.
The historical manager supplies only evaluation/ranking and path helpers.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cuda_fdm.architecture_search import Search, read_json, write_json, ROOT

DEFAULT_ROOT = ROOT / "runs/architecture_search/mlp_depth_width_aux_altmix_v2"
PRESERVED_COHORTS = ("experiment_v1", "mlp_depth_width_aux_v1")
DEPTHS = (2, 3, 4)
WIDTHS = (512, 768, 1024)
OPTIONAL_LOWER_WIDTH = 384
SEEDS = (0, 1, 2)
KEEP = (100, 200, 400, 500, 800, 1000, 1500)
PROTOCOL = {
    "name": "main_only_future_aux_diverse_h3_reset_schedule_v5",
    "plan_revision": "width512_768_1024_then384_v2",
    "training_protocol": "cuda_mlp_flat_finite_horizon_future_aux_diverse_h3_reset_v8",
    "depth": {"width": 512, "layers": list(DEPTHS), "seeds": list(SEEDS), "iterations": 800},
    "width": {"widths": list(WIDTHS), "seed": 0, "iterations": 800,
              "extension": "Test 384 once, only if 512 ranks first among 512/768/1024; no further expansion"},
    "confirmation": {"top": 2, "seeds": list(SEEDS), "iterations": 1500,
                     "curve_iterations": [500, 1000], "selection_iteration": 1500},
    "evaluation": {"games_per_match_per_scenario": 256,
                   "scenarios": {"three_nine": .75, "headon": .25},
                   "score": {"crossplay": .75, "fixed_references": .25},
                   "ties": ["win_rate", "damage_diff", "reference"],
                   "seeds": {"depth": 82001, "width": 83001, "curves": 83011, "confirm": 95001},
                   "references": "all three depth seed0 iter200; final adds ALL screened widths seed0 iter400",
                   "pairing": "128 identical initial-condition pairs with aircraft roles swapped; draws=0.5"},
    "training": {"architecture": "mlp", "nenv": 4096, "rollout": 64, "epochs": 4,
                 "minibatch_splits": 8, "obs": 184, "policy_hz": 10, "physics_hz": 60,
                 "substeps": 6, "gamma": .997, "gae_lambda": .95, "lr": .0003,
                 "entropy": .001, "clip": .2, "target_kl": .03,
                 "aux_pred": True, "aux_coef": .1,
                 "reward": "final-safe 50/20/20/10 trapezoid +/-5; no low-alt shaping; altitude exit = remaining HP x10",
                 "reward_contract": "remaining_hp_altitude_exit_damage_styles_hunt_each3_v2",
                 "initial_state_three_nine_headon": [3, 1],
                 "pool": "unchanged gated EMA-softmax; evict cap4, gate.6, EMA.1, temp.3, uniform.5",
                 "comparison_milestone_exploiter_schedule": False},
    "final": {"fresh_seed": 0, "iterations": 20000, "milestone_period": 500,
              "exploiter_max_iterations": 1000, "exploiter_target_ema": .75,
              "exploiter_reward": "one per500: hunt/standard/attack/hunt/defense/standard/hunt/attack/defense, repeat; hunter once per3",
              "exploiter_alt_hunt_coef": 5.0,
              "schedule_period": 2000, "checkpoint_every": 100,
              "lr_decay": .5, "entropy_decay": .5,
              "lr_floor": .00005, "entropy_floor": .0001,
              "schedule_contract": "bounded_main_schedule_v1",
              "shaping_ladder": [1.0, .6, .32, .12, 0.0], "rollout_increment": 8,
              "keep_milestones": 500, "save_runtime": True,
              "wandb_username": "leeai021213", "wandb_entity": "leeai021213-ajou-university",
              "wandb_project": "AIP contest", "checkpoint_upload": False},
    "main_only_runtime": True,
    "forbidden": ["comparison training in this runtime", "old-result mixing", "GRU training", "residual", "20Hz", "AlphaStar",
                  "BT", "MPC", "1220", "4499", "automatic coefficient tuning"],
    "latency": {"agents": 2, "cpu_threads": 1, "p99_ms_below": 50, "max_ms_below": 100,
                "scope": "actual local submission pipelines, excludes network/server"},
}


class IntegrityError(FloatingPointError):
    """Requires review: never retry, rehash, clear STOP, or downgrade a gate."""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_name(width, depth):
    return f"mlp{width}_d{depth}"


def model_spec(name):
    if not name.startswith("mlp") or "_d" not in name:
        raise IntegrityError(f"invalid candidate: {name}")
    w, d = map(int, name[3:].split("_d"))
    if w not in (*WIDTHS, OPTIONAL_LOWER_WIDTH) or d not in DEPTHS:
        raise IntegrityError(f"candidate outside approved bounds: {name}")
    return {"architecture": "mlp", "hidden": ",".join([str(w)] * d), "width": w, "depth": d}


def sources():
    paths = [
        "cuda_fdm/mlp_size_search.py", "cuda_fdm/MLP_SIZE_SEARCH.md",
        "cuda_fdm/architecture_search.py", "cuda_fdm/search_eval.py", "cuda_fdm/search_validate.py",
        "cuda_fdm/ppo_gpu.py", "cuda_fdm/train_gpu.py", "cuda_fdm/future_aux.py",
        "cuda_fdm/reward_modes.py", "cuda_fdm/tests/altitude_reward_val.py",
        "cuda_fdm/tests/altitude_reward_gpu_val.py",
        "cuda_fdm/finite_checks.py", "cuda_fdm/gpu_ckpt_to_bundle.py", "cuda_fdm/rl_env.py",
        "cuda_fdm/bundle_completion.py",
        "cuda_fdm/gpu_env.py", "cuda_fdm/ic.py", "cuda_fdm/obs_reward.py",
        "cuda_fdm/tests/cuda_rt.py", "cuda_fdm/tests/mlp_search_preflight.py",
        "cuda_fdm/tests/mlp_size_search_val.py", "cuda_fdm/tests/integrity_cpu_suite.py",
        "claude_code/model.py", "claude_code/my_reward.py", "claude_code/my_observation.py",
        "claude_code/observation_contract.py", "claude_code/action_provider.py",
        "claude_code/altguard_provider.py", "claude_code/altblend_provider.py",
        "src/dogfight/unreal/policies.py", "src/dogfight/envs/termination.py",
    ]
    paths += [str(p.relative_to(ROOT)).replace("\\", "/")
              for directory, pattern in (("cuda_fdm/gen", "*"), ("cuda_fdm/ref", "*.py"),
                                         ("cuda_fdm/tests", "*.py"))
              for p in (ROOT / directory).glob(pattern) if p.suffix in (".py", ".cu", ".cuh")]
    return {p: sha(ROOT / p) for p in sorted(set(paths))}


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = api.OpenProcess(0x1000, False, int(pid))
        if not handle:
            # Access denied is not evidence of absence.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def lock_file(path):
    handle = open(path, "a+b")
    try:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        handle.close()
        raise
    return handle


def freeze_json(path, data):
    path = Path(path)
    data = json.loads(json.dumps(data, allow_nan=False))
    if path.exists():
        if read_json(path) != data:
            raise IntegrityError(f"frozen record differs: {path}")
    else:
        write_json(path, data)


def check_metrics(path, live=False):
    """Finite CSV checks including aux/grad metrics; empty episode aggregates are allowed."""
    from cuda_fdm.train_gpu import csv_header
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if live and not content.endswith("\n"):
        lines = lines[:-1]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    if reader.fieldnames != csv_header(True).split(","):
        raise IntegrityError(f"auxiliary comparison CSV schema mismatch: {path}")
    rows = list(reader)
    for expected_it, row in enumerate(rows, 1):
        if None in row or any(v is None for v in row.values()):
            raise IntegrityError(f"partial/malformed completed CSV row: {path}")
        if int(row["iter"]) != expected_it:
            raise IntegrityError(f"missing/duplicate iteration in CSV: {path}")
        for key, value in row.items():
            if key == "early_stop":
                if value not in ("True", "False"):
                    raise IntegrityError(f"invalid early-stop flag: {path}")
                continue
            numeric = float(value)
            missing_episode = key in ("mean_ret", "mean_len", "win_rate") and float(row["eps"]) == 0
            if not math.isfinite(numeric) and not (missing_episode and math.isnan(numeric)):
                raise IntegrityError(f"non-finite {key} at iteration {expected_it}: {path}")
        for key in ("aux_opp_coverage", "aux_self_coverage", "clipfrac"):
            if not 0 <= float(row[key]) <= 1:
                raise IntegrityError(f"invalid metric range {key}: {path}")
    return rows


def expected_cfg(name, seed, final):
    from dataclasses import asdict
    from cuda_fdm.ppo_gpu import PPOGPUConfig
    m = model_spec(name)
    return asdict(PPOGPUConfig(
        architecture="mlp", hidden=tuple([m["width"]] * m["depth"]), gru_size=0,
        aux_pred=True, aux_coef=.1, seed=seed, save_runtime=True,
        rollout_steps=64, update_epochs=4, num_minibatches=8, gamma=.997, gae_lambda=.95,
        lr=.0003, ent_coef=.001, clip_coef=.2, target_kl=.03,
        sched_period=2000 if final else 0, milestone_period=500 if final else 0,
        exploiter_iters=1000 if final else 0, exploiter_lr=.0001,
        exploiter_win_target=.75, exploiter_alternate_altitude_hunt=True, exploiter_alt_hunt_coef=5.0,
        exploiter_ent_coef=.0001, exploiter_clip_coef=.2,
    ))


def verify_checkpoint(path, name, seed, final=False):
    import torch
    from cuda_fdm.future_aux import AUX_CONTRACT
    from cuda_fdm.reward_modes import REWARD_CONTRACT
    from cuda_fdm.finite_checks import require_finite
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("training_protocol") != PROTOCOL["training_protocol"] or ckpt.get("auxiliary_contract") != AUX_CONTRACT:
        raise IntegrityError(f"different checkpoint training/aux protocol: {path}")
    if ckpt.get("reward_contract") != REWARD_CONTRACT:
        raise IntegrityError(f"different checkpoint reward contract: {path}")
    iteration = int(ckpt["iteration"])
    expected = expected_cfg(name, seed, final)
    phase = (iteration - 1) // 2000 if final and iteration else 0
    expected["ent_coef"] *= (1. / 3.) ** phase
    expected["rollout_steps"] += 8 * phase
    for key, value in expected.items():
        if key == "total_iterations":
            continue  # only the finish line may extend, never the learning settings
        actual = ckpt["cfg"].get(key)
        if isinstance(value, tuple):
            actual = tuple(actual) if actual is not None else None
        if actual != value:
            raise IntegrityError(f"checkpoint config differs: {key}: {actual!r} != {value!r}: {path}")
    runtime = ckpt.get("runtime")
    if not runtime or tuple(runtime["trainer"]["opp_assign"].shape) != (4096,):
        raise IntegrityError(f"checkpoint has no matching 4096-environment runtime: {path}")
    if runtime.get("reward_mode") != 0 or runtime.get("alt_hunt_coef") != 5.0:
        raise IntegrityError(f"main runtime has unexpected reward mode: {path}")
    expected_steps = 4096 * sum(64 + (8 * ((i - 1) // 2000) if final else 0)
                                for i in range(1, iteration + 1))
    if ckpt["global_step"] != expected_steps:
        raise IntegrityError(f"checkpoint transition budget differs: {path}")
    require_finite(ckpt, f"checkpoint:{path}")
    return iteration


def validate_records(result, games=256):
    for key, match in result["matches"].items():
        records = match["records"]
        if len(records) != games:
            raise IntegrityError(f"wrong evaluation game count: {key}")
        for i, row in enumerate(records):
            if row["score"] not in (0., .5, 1.) or not -1.00001 <= row["damage_diff"] <= 1.00001:
                raise IntegrityError(f"invalid evaluation score: {key}")
            if row.get("ic_pair") != i % (games // 2) or row.get("role_swapped") is not (i >= games // 2):
                raise IntegrityError(f"invalid role pairing: {key}")
            duration = row.get("duration_sec", float("nan"))
            if not math.isfinite(duration) or not 0 < duration <= 200.4:
                raise IntegrityError(f"invalid episode duration: {key}")
            for field in ("first_hit_sec", "first_hit_b_sec"):
                v = row.get(field)
                if v is not None and (not math.isfinite(v) or not 0 < v <= duration):
                    raise IntegrityError(f"invalid hit time: {key}")


class MLPSearch(Search):
    def __init__(self, folder=DEFAULT_ROOT):
        self.folder = Path(folder).resolve()
        allowed = (ROOT / "runs/architecture_search").resolve()
        if not self.folder.is_relative_to(allowed) or self.folder == allowed or self.folder.name in PRESERVED_COHORTS:
            raise ValueError("new search must use a dedicated runs/architecture_search subdirectory, never a preserved cohort")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stop, self.state = self.folder / "STOP", self.folder / "status.json"
        self.manifest_path = self.folder / "manifest.json"
        self.stage, self._confirmation_complete = "initializing", False
        self.lock_handle = lock_file(allowed / "mlp_aux_gpu_manager.lock")
        try:
            previous = read_json(self.state) if self.state.exists() else {}
            for key in ("manager_pid", "child_pid"):
                pid = previous.get(key)
                if pid and pid != os.getpid() and pid_alive(pid):
                    raise RuntimeError(f"prior {key} {pid} is still alive; refusing duplicate GPU work")
            self.check_integrity()
            if self.manifest_path.exists():
                self.manifest = read_json(self.manifest_path)
                if self.manifest.get("protocol") != PROTOCOL:
                    raise IntegrityError("new search manifest does not match the frozen depth-first protocol")
            else:
                if (self.folder / "training").exists() or (self.folder / "final_20000").exists():
                    raise IntegrityError("pre-existing training without this cohort's manifest")
                self.manifest = dict(created_utc=now(), protocol=PROTOCOL, source_sha256=sources(),
                    authorization="2026-09-01 user: main ONLY; one hunter per THREE500-iter slots; other slots rotate balanced/attack/defense with ONLY Damage changed; preserve all comparison work; fresh main20000; EMA75%",
                    old_cohort_not_reused={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                        for cohort in PRESERVED_COHORTS
                        for name in ("manifest.json", "INTEGRITY_HOLD.json", "INTEGRITY_FAILURE.json", "STOP")
                        if (p := ROOT / "runs/architecture_search" / cohort / name).exists()})
                write_json(self.manifest_path, self.manifest)
            self.manifest_digest = sha(self.manifest_path)
            self.check_sources()
            self.update("initializing")
        except Exception:
            self.lock_handle.close()
            raise

    def close(self):
        if getattr(self, "lock_handle", None):
            self.lock_handle.close()

    def check_sources(self):
        if sha(self.manifest_path) != self.manifest_digest:
            raise IntegrityError("manifest changed while experiment was running")
        for relative, digest in self.manifest["source_sha256"].items():
            if sha(ROOT / relative) != digest:
                raise IntegrityError(f"source hash changed; not rebaselining: {relative}")
        for relative, digest in self.manifest.get("old_cohort_not_reused", {}).items():
            if not (ROOT / relative).exists() or sha(ROOT / relative) != digest:
                raise IntegrityError(f"preserved cohort guard changed: {relative}")

    def update(self, phase, **extra):
        write_json(self.state, dict(manager_pid=os.getpid(), phase=phase, stage=self.stage,
                    updated_utc=now(), protocol=PROTOCOL["name"], **extra))

    def event(self, **data):
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(time_utc=now(), **data), ensure_ascii=False, allow_nan=False) + "\n")

    def command(self, label, args, log_folder=None, retries=0, allow_failure=False):
        # Do not inherit the old manager's blind command retries. A heartbeat
        # may recover an ordinary interruption after checking for live children.
        del retries
        self.check_integrity()
        self.check_sources()
        if self.stop.exists():
            raise InterruptedError("STOP requested")
        folder = Path(log_folder or self.folder / "jobs" / label)
        folder.mkdir(parents=True, exist_ok=True)
        args = list(args)
        if "cuda_fdm.search_eval" in args:
            args += ["--stop-file", str(self.stop)]
        markers = self.child_integrity_markers(args, folder)
        self.check_integrity(markers)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_DISABLE_CODE="true", WANDB_SAVE_CODE="false")
        env["WANDB_MODE"] = "online" if "--wandb" in args else "disabled"
        command = [sys.executable, "-B", *args]
        metric_path = Path(args[args.index("--log") + 1]) if "--log" in args else None
        started = time.monotonic()
        pending = None
        with (folder / "stdout.log").open("a", encoding="utf-8") as out, \
                (folder / "stderr.log").open("a", encoding="utf-8") as err:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            context = dict(child_pid=proc.pid, command=command, stdout=str(folder / "stdout.log"),
                           stderr=str(folder / "stderr.log"), child_started_utc=now())
            print(f"[manager] {label} pid={proc.pid}", flush=True)
            self.event(event="child_start", label=label, **context)
            self.update(label, **context)
            sampled = 0.
            while proc.poll() is None:
                if time.monotonic() - sampled >= 30:
                    sampled = time.monotonic()
                    progress = {}
                    try:
                        self.check_integrity(markers)
                        self.check_sources()
                        if metric_path:
                            rows = check_metrics(metric_path, live=True)
                            if rows:
                                recent = rows[-50:] if len(rows) > 1 else rows
                                avg = statistics.mean(float(r["elapsed"]) for r in recent)
                                target = int(args[args.index("--iters") + 1])
                                progress = dict(iteration=int(rows[-1]["iter"]), target=target,
                                    transitions=int(rows[-1]["gstep"]), recent_sec_per_iter=avg,
                                    eta_current_training_sec=max(0, target-int(rows[-1]["iter"]))*avg,
                                    metrics_csv=str(metric_path), numeric_checks_passed=True,
                                    main_eta_excludes_future_exploiters="--wandb" in args)
                    except (FloatingPointError, ValueError, OSError) as exc:
                        pending = pending or IntegrityError(str(exc))
                        if not self.stop.exists():
                            write_json(self.stop, {"reason": str(exc), "created_by": "integrity_guard"})
                    try:
                        gpu = subprocess.check_output(["nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], text=True, timeout=5,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
                        with (self.folder / "gpu_telemetry.csv").open("a", encoding="utf-8") as f:
                            f.write(f"{now()},{label},{gpu}\n")
                        progress["gpu_sample"] = gpu
                    except (OSError, subprocess.SubprocessError):
                        pass
                    self.update(label, **context, **progress, stop_requested=self.stop.exists(),
                                child_wall_sec=time.monotonic()-started,
                                integrity_pending=str(pending) if pending else None)
                time.sleep(2)
        self.event(event="child_exit", label=label, child_pid=proc.pid,
                   returncode=proc.returncode, wall_sec=time.monotonic()-started)
        self.update(label, child_pid=None, returncode=proc.returncode)
        if pending:
            raise pending
        self.check_integrity(markers)
        self.check_sources()
        if self.stop.exists():
            raise InterruptedError("STOP applied at safe boundary")
        if proc.returncode and not (allow_failure and proc.returncode == 2):
            if label.startswith("preflight"):
                raise IntegrityError(f"preflight failed: {label}; inspect {folder}")
            raise RuntimeError(f"{label} failed exit={proc.returncode}; no automatic unsafe retry; inspect {folder}")
        return proc.returncode

    def train_args(self, name, seed, target, final=False):
        m = model_spec(name)
        folder = self.folder / "final_20000" if final else self.directory(name, seed)
        args = ["-m", "cuda_fdm.train_gpu", "--architecture", "mlp", "--hidden", m["hidden"],
            "--aux-pred", "--aux-coef", "0.1", "--seed", str(seed), "--iters", str(target),
            "--nenv", "4096", "--rollout", "64", "--epochs", "4", "--minibatches", "8",
            "--gamma", "0.997", "--gae-lambda", "0.95", "--lr", "0.0003", "--ent-coef", "0.001",
            "--clip", "0.2", "--target-kl", "0.03", "--substeps", "6", "--pool-evict-cap", "4",
            "--selfplay-gate-threshold", "0.6", "--selfplay-ema-alpha", "0.1",
            "--pool-sample-temp", "0.3", "--pool-uniform-floor", "0.5",
            "--exploiter-lr", "0.0001", "--exploiter-ent-coef", "0.0001", "--exploiter-clip-coef", "0.2",
            "--exploiter-win-target", "0.75", "--exploiter-alternate-altitude-hunt",
            "--exploiter-alt-hunt-coef", "5.0", "--save-runtime", "--save-every", "100",
            "--save", str(folder / "checkpoint.pt"), "--log", str(folder / "metrics.csv"),
            "--stop-file", str(self.stop)]
        if final:
            args += ["--wandb", "--wandb-required", "--wandb-entity", "leeai021213-ajou-university",
                "--wandb-expected-username", "leeai021213", "--wandb-project", "AIP contest",
                "--wandb-group", self.folder.name, "--wandb-run-name", f"main20000-aux-{name}-s{seed}",
                "--milestone-period", "500", "--exploiter-iters", "1000", "--sched-period", "2000",
                "--sched-lr-decay", "0.5", "--sched-ent-decay", "0.5",
                "--sched-lr-floor", "0.00005", "--sched-ent-floor", "0.0001",
                "--keep-iterations", ",".join(map(str, range(500, 20001, 500)))]
        else:
            args += ["--no-wandb", "--milestone-period", "0", "--exploiter-iters", "0",
                "--sched-period", "0", "--keep-iterations", ",".join(map(str, KEEP))]
        return args, folder

    def reconcile_metrics(self, folder, iteration):
        path = folder / "metrics.csv"
        rows = check_metrics(path)
        if len(rows) < iteration:
            raise IntegrityError(f"checkpoint is ahead of completed scalar log: {folder}")
        if len(rows) > iteration:
            # Recovery from a crash after the last atomic 100-iteration save:
            # retain the full old log, then replay only uncheckpointed work.
            backup = folder / f"metrics_before_resume_{time.time_ns()}.csv"
            shutil.copy2(path, backup)
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows[:iteration])
            os.replace(temporary, path)
            self.event(event="resume_log_archived", backup=str(backup), resume_iteration=iteration,
                       replay_iterations=len(rows)-iteration)

    def train_to(self, name, seed, target, final=False):
        self.check_integrity()
        self.check_sources()
        if self.stop.exists():
            raise InterruptedError("STOP requested")
        if final and not self._confirmation_complete:
            raise IntegrityError("main requires complete held-out evaluation and latency gates in this run")
        args, folder = self.train_args(name, seed, target, final)
        self.check_integrity([folder / "INTEGRITY_FAILURE.json"])
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "checkpoint.pt"
        freeze_json(folder / "run_config.json", dict(model=model_spec(name), seed=seed, final=final,
                    source_sha256=self.manifest["source_sha256"], training_protocol=PROTOCOL["training_protocol"],
                    base_cfg={k: v for k, v in expected_cfg(name, seed, final).items()
                              if k != "total_iterations"}))
        iteration = verify_checkpoint(checkpoint, name, seed, final) if checkpoint.exists() else 0
        self.reconcile_metrics(folder, iteration)
        if iteration >= target:
            if not final:
                self.snapshot(name, seed, target)
            return
        if checkpoint.exists():
            args += ["--resume", str(checkpoint)]
        self.command(f"train-{name}-s{seed}-to{target}", args, folder)
        if verify_checkpoint(checkpoint, name, seed, final) != target:
            raise IntegrityError(f"training exited at the wrong iteration: {folder}")
        rows = check_metrics(folder / "metrics.csv")
        if len(rows) != target:
            raise IntegrityError(f"training CSV not complete: {folder}")
        if not final and list(folder.glob("*.wandbid")):
            raise IntegrityError("comparison unexpectedly created a W&B run")
        if final and not checkpoint.with_suffix(".wandbid").exists():
            raise IntegrityError("main did not record its required W&B run id")
        self.event(event="training_complete", model=name, seed=seed, iteration=target, final=final,
                   mean_iter_sec=statistics.mean(float(r["elapsed"]) for r in rows),
                   transitions=int(rows[-1]["gstep"]))

    def evaluate(self, phase, candidates, refs, seed, extra_pairs=None):
        self.check_integrity()
        # Every evaluated checkpoint is immutable, source-matched, local to this
        # cohort, and saved at the declared iteration. No old reference import.
        fingerprints = {}
        for path in [*[v[0] for v in candidates.values()], *refs.values()]:
            p = Path(path).resolve()
            if not p.is_relative_to(self.folder / "training"):
                raise IntegrityError(f"foreign evaluation checkpoint: {path}")
            name, seed_dir = p.parent.parent.name, p.parent.name
            seed_number = int(seed_dir.removeprefix("seed"))
            iteration = verify_checkpoint(p, name, seed_number)
            if p.stem != f"iter_{iteration:05d}":
                raise IntegrityError(f"snapshot filename does not match checkpoint iteration: {p}")
            fingerprints[str(p)] = sha(p)
        folder = self.folder / "evaluation" / phase
        freeze_json(folder / "checkpoint_sha256.json", fingerprints)
        ranking = super().evaluate(phase, candidates, refs, seed, extra_pairs)
        validate_records(read_json(folder / "results.json"))
        for path, digest in fingerprints.items():
            if sha(path) != digest:
                raise IntegrityError(f"checkpoint changed during evaluation: {path}")
        self.event(event="evaluation_complete", phase=phase, ranking=ranking)
        return ranking

    def preflight(self):
        self.stage = "preflight"
        for name, module, filename in (
                ("cpu", "cuda_fdm.tests.integrity_cpu_suite", "result.json"),
                ("gpu", "cuda_fdm.tests.mlp_search_preflight", "gpu"),
                ("altitude_gpu", "cuda_fdm.tests.altitude_reward_gpu_val", "gpu"),
                ("timeout_gpu", "cuda_fdm.tests.timeout_gpu_val", "gpu")):
            destination = self.folder / "preflight" / name / filename
            result_path = destination / "result.json" if filename == "gpu" else destination
            folder = self.folder / "preflight" / name
            self.check_integrity([result_path.parent / "INTEGRITY_FAILURE.json"])
            freeze_json(folder / "source_sha256.json", self.manifest["source_sha256"])
            if not result_path.exists():
                self.command(f"preflight-{name}", ["-m", module, "--output", str(destination)], folder)
            if read_json(result_path).get("passed") is not True:
                raise IntegrityError(f"preflight did not pass: {result_path}")

    def latency_gate(self, name, checkpoint, label):
        folder = self.folder / "latency" / label
        output = folder / "result.json"
        self.check_integrity([folder / "INTEGRITY_FAILURE.json"])
        freeze_json(folder / "checkpoint.json", dict(path=str(checkpoint), sha256=sha(checkpoint)))
        if not output.exists():
            self.command(f"latency-{label}", ["-m", "cuda_fdm.search_validate",
                         "--checkpoint", str(checkpoint), "--output", str(output)], folder, allow_failure=True)
        measured = read_json(output)
        fields = ("p50_ms", "p95_ms", "p99_ms", "max_ms")
        if any(not isinstance(measured.get(k), (float, int)) or not math.isfinite(measured[k]) for k in fields):
            raise IntegrityError("latency result has non-finite or missing timings")
        expected_pass = measured["p99_ms"] < 50 and measured["max_ms"] < 100
        if measured.get("passed") is not expected_pass or measured.get("agents") != 2:
            raise IntegrityError("latency pass flag disagrees with measured bounds")
        self.event(event="latency_complete", model=name, **measured)
        return expected_pass

    def candidates(self, names, seeds, iteration):
        return {f"{name}_s{s}": (self.snapshot(name, s, iteration), name) for name in names for s in seeds}

    def run(self):
        raise RuntimeError("Main-only runtime: use the approved continue_search_diverse_main controller, never restart comparisons here")
        # Historical manager implementation below is retained only as reference.
        self.check_integrity()
        self.check_sources()
        self.preflight()
        self.stage = "depth"
        depths = [model_name(512, d) for d in DEPTHS]
        for seed in SEEDS:
            for name in depths:
                self.train_to(name, seed, 800)
        refs = {f"ref_{name}_200": self.snapshot(name, 0, 200) for name in depths}
        depth_rank = self.evaluate("depth_800", self.candidates(depths, SEEDS, 800), refs, 82001)
        depth = model_spec(depth_rank[0]["family"])["depth"]
        freeze_json(self.folder / "depth_decision.json", {"selected_depth": depth, "ranking": depth_rank,
            "scope": "best depth observed at width512, not proof of best depth at every width"})
        self.stage = "width"
        widths = [model_name(w, depth) for w in WIDTHS]
        for name in widths:
            self.train_to(name, 0, 800)
        width_rank = self.evaluate("width_800", self.candidates(widths, (0,), 800), refs, 83001)
        extension = None
        if width_rank[0]["family"] == model_name(512, depth):
            extension = model_name(OPTIONAL_LOWER_WIDTH, depth)
            self.train_to(extension, 0, 800)
            widths.append(extension)
            width_rank = self.evaluate("width_extended_800", self.candidates(widths, (0,), 800), refs, 83001)
        top = [row["family"] for row in width_rank[:2]]
        freeze_json(self.folder / "width_decision.json", dict(top_two=top, extension=extension, ranking=width_rank))
        self.stage = "confirmation"
        for seed in SEEDS:
            for name in top:
                self.train_to(name, seed, 1500)
        final_refs = refs | {f"ref_{name}_400": self.snapshot(name, 0, 400) for name in widths}
        for iteration in (500, 1000):
            self.evaluate(f"confirmation_curve_{iteration}", self.candidates(top, SEEDS, iteration), final_refs, 83011)
        ranking = self.evaluate("confirmation_1500", self.candidates(top, SEEDS, 1500), final_refs, 95001)
        self.stage = "latency"
        eligible = [r for r in ranking if self.latency_gate(r["family"], self.snapshot(r["family"], 0, 1500),
                                                          f"final_{r['family']}")]
        if not eligible:
            raise IntegrityError("no finalist passed the actual two-agent 10Hz submission latency check")
        winner = eligible[0]["family"]
        self._confirmation_complete = True
        decision = dict(selected=winner, model=model_spec(winner), depth_ranking=depth_rank,
                        width_ranking=width_rank, ranking=ranking,
                        latency_eligible=[r["family"] for r in eligible],
                        scope="best measured in this bounded sequential search, not a global optimum",
                        main="fresh seed0; 20000 main iterations plus scheduled exploiters; no search checkpoint resume")
        freeze_json(self.folder / "final_decision.json", decision)
        report = ["# MLP depth/width search result", "", f"Selected: {winner}", "",
                  "|Model|Score|Cross-play|References|Win rate|Damage difference|", "|---|---:|---:|---:|---:|---:|"]
        for row in ranking:
            report.append(f"|{row['family']}|{row['score']:.4f}|{row['crossplay']:.4f}|{row['reference']:.4f}|{row['win_rate']:.4f}|{row['damage_diff']:.4f}|")
        report += ["", "All comparisons use auxiliary coefficient 0.1. See per-seed CSV/paired game records and confidence intervals.",
                   "Depth was selected at width512; interactions at other widths were not exhaustively searched.",
                   "CPU gate measures the local two-agent submission pipeline, not network/server latency.",
                   "Main starts fresh. Milestone/exploiter wall time is additional to ordinary iteration time."]
        (self.folder / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        self.stage = "main"
        self.train_to(winner, 0, 20000, final=True)
        from cuda_fdm.bundle_completion import complete_final_bundle
        complete_final_bundle(self, winner)
        self.stage = "complete"
        self.update("complete", selected=winner, final_checkpoint=str(self.folder / "final_20000/checkpoint.pt"),
                    final_bundle=str(self.folder / "final_20000/bundle"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    search = None
    try:
        search = MLPSearch(args.root)
        search.run()
    except InterruptedError as exc:
        if search:
            search.update("paused", reason=str(exc))
        print(str(exc), flush=True)
    except (FloatingPointError, AssertionError, ValueError) as exc:
        if search:
            search.update("failed", reason=str(exc), integrity_failure=True)
            marker = search.folder / "INTEGRITY_FAILURE.json"
            if not marker.exists():
                write_json(marker, dict(reason=str(exc), manager_pid=os.getpid(), time_utc=now(), requires_review=True))
        raise
    except Exception as exc:
        if search:
            search.update("failed", reason=str(exc), integrity_failure=False)
        raise
    finally:
        if search:
            search.close()


if __name__ == "__main__":
    main()
