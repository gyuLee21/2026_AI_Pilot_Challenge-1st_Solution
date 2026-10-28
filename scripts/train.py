"""Managed storage launcher; PPO behavior stays in cuda_fdm.train_gpu."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_FOLDERS = {"three_nine": "3-9", "headon": "headon", "mixed": "common"}
RUNTIMES = {"current": ROOT, "competition36500": ROOT / "research/competition_36500/runtime"}
MANAGED_OPTIONS = ("--save", "--log", "--league-dir", "--resume", "--scenario")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scenario", required=True, choices=SCENARIO_FOLDERS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runtime", choices=RUNTIMES, default="current",
                        help="Use the preserved 184D final 3-9 runtime explicitly; default retains the existing runtime")
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts/models/rl")
    parser.add_argument("--config", type=Path, help="JSON containing a list of PPO arguments")
    parser.add_argument("--resume", action="store_true", help="Resume this managed run's checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print command without creating files or starting CUDA")
    parser.add_argument("ppo_arguments", nargs=argparse.REMAINDER, help="PPO options after --")
    args = parser.parse_args()
    runtime_root = RUNTIMES[args.runtime]
    if args.runtime == "competition36500" and args.scenario != "three_nine":
        parser.error("competition36500 is the final 3-9 lineage; use current for other scenarios")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_name):
        parser.error("run-name must be a simple name, not a path")
    config_arguments = []
    if args.config:
        try:
            config_arguments = json.loads(args.config.read_text(encoding="utf-8"))["arguments"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid config: {error}")
        if not isinstance(config_arguments, list) or not all(isinstance(item, str) for item in config_arguments):
            parser.error("config arguments must be a list of strings")
    forwarded = args.ppo_arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    forwarded = config_arguments + forwarded
    for token in forwarded:
        option = token.split("=", 1)[0]
        # Also reject argparse abbreviations such as --sav and --resu.
        if option.startswith("--") and any(reserved.startswith(option) for reserved in MANAGED_OPTIONS):
            parser.error(f"{option} overrides managed storage/scenario; use the legacy CLI for custom paths")

    root = args.artifacts_root.resolve()
    run = root / SCENARIO_FOLDERS[args.scenario] / args.run_name
    if not run.resolve().is_relative_to(root):
        parser.error("run directory escapes artifacts root")
    checkpoint = run / "checkpoint.pt"
    if args.resume:
        if not checkpoint.is_file():
            parser.error(f"resume checkpoint does not exist: {checkpoint}")
    elif run.exists() and any(run.iterdir()):
        parser.error("run already contains data; choose another name or use --resume")

    command = [
        sys.executable, "-m", "cuda_fdm.train_gpu",
        "--save", str(checkpoint),
        "--log", str(run / "training.csv"),
        "--league-dir", str(run / "league"),
    ]
    if args.runtime == "current":
        command.extend(["--scenario", args.scenario])
    # The preserved final source predates --scenario and already fixes the
    # initial distribution to three-nine (scenario_b_prob=0). Do not pass a
    # modern flag to its historical parser or silently alter its source.
    if args.resume:
        command.extend(["--resume", str(checkpoint)])
    command.extend(forwarded)
    if args.dry_run:
        print(json.dumps({"cwd": str(runtime_root), "runtime": args.runtime,
                          "run_directory": str(run), "command": command}, indent=2))
        return 0
    run.mkdir(parents=True, exist_ok=True)
    import os
    environment = os.environ.copy()
    # A child interpreter isolates same-named historical modules; never mutate
    # this process's imports or transplant 214D weights into the 184D trainer.
    if args.runtime != "current":
        environment['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'src'), str(ROOT)))
    return subprocess.run(command, cwd=runtime_root, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
