"""CPU-only release suite and immutable receipt for the isolated 100K control."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import unittest

import torch


MODULES = (
    "cuda_fdm.tests.vnext_control_val",
    "cuda_fdm.tests.vnext_scalability_val",
    "cuda_fdm.tests.main_schedule_val",
    "cuda_fdm.tests.active_league_val",
    "cuda_fdm.tests.pool_episode_val",
    "cuda_fdm.tests.rollout_finite_val",
)


def run(output: Path | None = None) -> dict:
    if output is not None and output.exists():
        raise FileExistsError(f"preserve previous validation receipt: {output}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({
                "protocol": "active_league_100k_vnext_cpu_cleanup_in_progress_v1",
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "passed": False,
            }, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
    torch.set_num_threads(1)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "protocol": "active_league_100k_vnext_cpu_cleanup_v3",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(result.wasSuccessful() and not torch.cuda.is_initialized()),
        "tests": int(result.testsRun),
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "cuda_requested": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "modules": list(MODULES),
        "output": log.getvalue(),
    }
    if output is not None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key != "output"},
                     ensure_ascii=False), flush=True)
    if not report["passed"]:
        print(log.getvalue())
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
