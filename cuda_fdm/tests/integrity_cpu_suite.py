"""Persist the bounded CPU correctness suite result; no CUDA initialization."""
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
    "cuda_fdm.tests.integrity_scope_val",
    "cuda_fdm.tests.architecture_search_manager_val",
    "cuda_fdm.tests.timeout_terminal_val",
    "cuda_fdm.tests.mlp_updater_val",
    "cuda_fdm.tests.pool_assigned_val",
    "cuda_fdm.tests.future_aux_val",
    "cuda_fdm.tests.mlp_size_search_val",
    "cuda_fdm.tests.altitude_reward_val",
)


def run(output):
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
        result = unittest.TextTestRunner(stream=logs, verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful() and not torch.cuda.is_initialized(),
                  time_utc=datetime.now(timezone.utc).isoformat(), tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  cuda_initialized=torch.cuda.is_initialized(), modules=list(MODULES), output=logs.getvalue())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "output"}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        print(logs.getvalue())
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    run(ap.parse_args().output)
