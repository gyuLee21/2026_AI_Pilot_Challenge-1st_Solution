"""Bounded main-only suite. Original comparison-manager tests remain upstream."""
import argparse
import contextlib
import io
import json
import unittest
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
)


def run(output):
    if output.exists():
        raise FileExistsError("Preserve previous validation; use a fresh output")
    torch.set_num_threads(1)
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    report = dict(passed=result.wasSuccessful() and not torch.cuda.is_initialized(),
        tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        cuda_initialized=torch.cuda.is_initialized(), modules=list(MODULES), output=log.getvalue())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "output"}), flush=True)
    if not report["passed"]:
        print(log.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    run(ap.parse_args().output)
