"""Bounded CPU release gate for the active-league main runtime.

This intentionally excludes retired architecture-search and experiment_v1
scope tests whose fixtures do not exist inside a self-contained main runtime.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch


MODULES = (
    "cuda_fdm.tests.submission_contract_val",
    "cuda_fdm.tests.pool_episode_val",
    "cuda_fdm.tests.kernel_stream_val",
    "cuda_fdm.tests.kernel_contract_audit",
    "cuda_fdm.tests.finite_guard_val",
    "cuda_fdm.tests.rollout_finite_val",
    "cuda_fdm.tests.timeout_terminal_val",
    "cuda_fdm.tests.mlp_updater_val",
    "cuda_fdm.tests.pool_assigned_val",
    "cuda_fdm.tests.future_aux_val",
    "cuda_fdm.tests.altitude_reward_val",
    "cuda_fdm.tests.damage_styles_val",
    "cuda_fdm.tests.warmup_reward_val",
    "cuda_fdm.tests.bundle_completion_val",
    "cuda_fdm.tests.main_schedule_val",
    "cuda_fdm.tests.active_league_val",
    "cuda_fdm.tests.official_contract_val",
)


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Preserve previous validation; choose a new output: {output}")
    torch.set_num_threads(1)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "passed": bool(result.wasSuccessful() and not torch.cuda.is_initialized()),
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "tests": int(result.testsRun),
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "modules": list(MODULES),
        "excluded_retired_fixtures": [
            "cuda_fdm.tests.integrity_scope_val",
            "cuda_fdm.tests.architecture_search_manager_val",
            "cuda_fdm.tests.mlp_size_search_val",
        ],
        "output": log.getvalue(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "output"},
                     ensure_ascii=False), flush=True)
    if not report["passed"]:
        print(log.getvalue())
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
