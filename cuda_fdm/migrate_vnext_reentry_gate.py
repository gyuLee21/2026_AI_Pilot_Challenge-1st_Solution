"""One-time, guarded migration of the vNext historical re-entry gate.

The gate is part of the serialized VNextConfig, so changing its source default
without migrating a recovery checkpoint would make resume fail closed.  This
tool changes only that one scalar in the embedded controller state and the
materialized shadow state, then proves the checkpoint is otherwise identical
to an immutable pre-migration backup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .league_vnext.decision_log import DecisionLog


CHECKPOINT_GATE_PATH = (
    "vnext_control", "config", "active_game",
    "historical_counter_main_ucb_max",
)
SHADOW_GATE_PATH = (
    "config", "active_game", "historical_counter_main_ucb_max",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_path(value: dict, path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _set_path(value: dict, path: tuple[str, ...], replacement: Any) -> None:
    current: Any = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _atomic_torch_save(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_same_except(left: Any, right: Any, *, skip: tuple[str, ...],
                        path: tuple[Any, ...] = ()) -> None:
    if path == skip:
        return
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype
                and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left, right)):
            raise RuntimeError(f"tensor changed at {path}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and np.array_equal(left, right)):
            raise RuntimeError(f"array changed at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError(f"mapping type changed at {path}")
        if set(left) != set(right):
            raise RuntimeError(f"mapping keys changed at {path}")
        for key in left:
            _assert_same_except(
                left[key], right[key], skip=skip, path=path + (key,))
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"sequence changed at {path}")
        for index, (before, after) in enumerate(zip(left, right)):
            _assert_same_except(
                before, after, skip=skip, path=path + (index,))
        return
    if isinstance(left, float) and isinstance(right, float):
        if (math.isnan(left) and math.isnan(right)) or left == right:
            return
        raise RuntimeError(f"float changed at {path}: {left} -> {right}")
    if type(left) is not type(right) or left != right:
        raise RuntimeError(f"value changed at {path}: {left!r} -> {right!r}")


def migrate(checkpoint: Path, backup: Path, shadow_state_path: Path, *,
            old: float, new: float) -> dict:
    checkpoint = checkpoint.resolve()
    backup = backup.resolve()
    shadow_state_path = shadow_state_path.resolve()
    if checkpoint == backup:
        raise ValueError("checkpoint and backup must be different files")
    for path in (checkpoint, backup, shadow_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(checkpoint) != _sha256(backup):
        raise RuntimeError("checkpoint no longer matches the approved pre-migration backup")

    baseline = torch.load(backup, map_location="cpu", weights_only=False)
    boundary = baseline.get("checkpoint_boundary") or {}
    if (boundary.get("protocol") != "clean_main_iteration_boundary_v1"
            or boundary.get("state") != "committed"
            or int(boundary.get("iteration", -1)) != int(baseline.get("iteration", -2))):
        raise RuntimeError("refusing to migrate a non-clean checkpoint")
    embedded_before = float(_get_path(baseline, CHECKPOINT_GATE_PATH))
    if embedded_before != float(old):
        raise RuntimeError(
            f"unexpected checkpoint gate {embedded_before}; expected {old}")

    shadow_before = json.loads(shadow_state_path.read_text(encoding="utf-8"))
    materialized_before = float(_get_path(shadow_before, SHADOW_GATE_PATH))
    if materialized_before != float(old):
        raise RuntimeError(
            f"unexpected shadow-state gate {materialized_before}; expected {old}")
    # shadow_state is persisted during the milestone commit, while the clean
    # checkpoint also captures the later side-learner roster refresh and final
    # PPO-health events. Therefore equality is not expected. Both hashes must
    # belong to the same valid chain, and the clean checkpoint must name its
    # current tail.
    decision_log = DecisionLog(shadow_state_path.with_name("decisions.jsonl"))
    checkpoint_decision_hash = baseline["vnext_control"].get(
        "decision_log_sha256")
    shadow_decision_hash = shadow_before.get("decision_log_sha256")
    if checkpoint_decision_hash != decision_log.last_hash:
        raise RuntimeError("checkpoint does not reference the decision-log tail")
    if not decision_log.contains_hash(shadow_decision_hash):
        raise RuntimeError("shadow-state decision hash is absent from the valid chain")

    migrated = baseline
    _set_path(migrated, CHECKPOINT_GATE_PATH, float(new))
    shadow_after = json.loads(json.dumps(shadow_before))
    _set_path(shadow_after, SHADOW_GATE_PATH, float(new))

    _atomic_json(shadow_state_path, shadow_after)
    _atomic_torch_save(checkpoint, migrated)

    written_checkpoint = torch.load(
        checkpoint, map_location="cpu", weights_only=False)
    written_shadow = json.loads(shadow_state_path.read_text(encoding="utf-8"))
    _assert_same_except(
        torch.load(backup, map_location="cpu", weights_only=False),
        written_checkpoint, skip=CHECKPOINT_GATE_PATH)
    _assert_same_except(
        shadow_before, written_shadow, skip=SHADOW_GATE_PATH)
    if float(_get_path(written_checkpoint, CHECKPOINT_GATE_PATH)) != float(new):
        raise RuntimeError("checkpoint gate migration did not persist")
    if float(_get_path(written_shadow, SHADOW_GATE_PATH)) != float(new):
        raise RuntimeError("shadow-state gate migration did not persist")

    return {
        "iteration": int(written_checkpoint["iteration"]),
        "old_main_ucb95_max": float(old),
        "new_main_ucb95_max": float(new),
        "comparison": "<=",
        "checkpoint_sha256_before": _sha256(backup),
        "checkpoint_sha256_after": _sha256(checkpoint),
        "all_other_checkpoint_state_equal": True,
        "all_other_shadow_state_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--shadow-state", required=True, type=Path)
    parser.add_argument("--old", required=True, type=float)
    parser.add_argument("--new", required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(migrate(
        args.checkpoint, args.backup, args.shadow_state,
        old=args.old, new=args.new), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
