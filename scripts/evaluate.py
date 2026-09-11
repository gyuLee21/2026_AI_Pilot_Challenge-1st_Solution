"""Stable entrypoint for the existing read-only GPU tournament."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    raise SystemExit(subprocess.run(
        [sys.executable, str(ROOT / "evaluation/tournament.py"), *sys.argv[1:]],
        check=False,
    ).returncode)
