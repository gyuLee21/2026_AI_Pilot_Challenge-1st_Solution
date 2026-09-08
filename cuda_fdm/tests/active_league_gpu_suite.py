"""CUDA release gate for clean mirrored evaluation and side-learner restore."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Preserve previous validation; choose a new output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the active-league GPU release gate")
    torch.set_num_threads(1)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "cuda_fdm.tests.active_league_gpu_val")
    torch.cuda.reset_peak_memory_stats()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "passed": bool(result.wasSuccessful()),
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "tests": int(result.testsRun),
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "cuda_device": torch.cuda.get_device_name(0),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "contracts": [
            "clean_stratified_mirrored_initial_conditions",
            "one_complete_episode_per_lane",
            "policy_order_complement_and_nonvacuous_score",
            "official_direct_msl_altitude_boundary",
            "main_optimizer_normalizer_runtime_rng_and_pool_restore",
            "first_le_observes_previous_milestone_core",
            "fresh_milestone_sidelearner_payoff_save_reload",
        ],
        "output": log.getvalue(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "output"},
                     ensure_ascii=False), flush=True)
    if not report["passed"] or report["tests"] != 4:
        print(log.getvalue())
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
