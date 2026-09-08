"""Append-only, hash-chained decision evidence for league control."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


GENESIS = "0" * 64


def _unsorted_canonical(value: dict) -> bytes:
    """Insertion-order serialization some existing records were hashed with.

    2026-09-04 (diagnosed from two live resume failures): a subset of records
    on disk verify only when re-serialized WITHOUT `sort_keys`. Their recorded
    digest was produced from insertion order, so any record holding a dict
    whose insertion order differs from its sorted order -- `nash`, keyed by
    archive id, e.g. {'5': .., '10': ..} written from solver_eligible_ids
    [10, 5] -- hashes differently under the sorted canonicalization that
    verify() uses, and fails its own integrity check on the next process
    start. That blocked every resume while leaving the file itself intact.
    Records are accepted under either form; new ones are written sorted.
    """
    return json.dumps(value, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class DecisionLog:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self.verify()

    def _verify_lines(self, lines: list[str]) -> tuple[str | None, int | None]:
        """Returns (last_valid_hash, broken_line_number). Exactly one is None."""
        previous = GENESIS
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                recorded = event.pop("event_sha256", None)
                if event.get("previous_sha256") != previous:
                    raise ValueError("chain break")
                if recorded not in (hashlib.sha256(_canonical(event)).hexdigest(),
                                    hashlib.sha256(_unsorted_canonical(event)).hexdigest()):
                    raise ValueError("hash mismatch")
            except (json.JSONDecodeError, ValueError):
                return None, line_number
            # Chain on the *recorded* hash: that is what the next record's
            # previous_sha256 was written against, in either scheme.
            previous = recorded
        return previous, None

    def verify(self) -> str:
        if not self.path.exists():
            return GENESIS
        lines = self.path.read_text(encoding="utf-8").splitlines()
        last_hash, broken_at = self._verify_lines(lines)
        if broken_at is None:
            return last_hash
        non_blank = [index for index, line in enumerate(lines, 1) if line.strip()]
        # 2026-09-04 (found via a real crash): a process killed mid-append can
        # leave the *final* record torn -- its previous_sha256 link and every
        # earlier record are fine, only that one entry's own bytes never
        # finished syncing. That is a recoverable crash artifact, not a
        # tampered log: repair it exactly once by dropping that one trailing
        # record and re-verifying. A break anywhere else (or a second broken
        # line after repair) is a real integrity problem and still raises.
        if not non_blank or broken_at != non_blank[-1]:
            raise RuntimeError(f"decision-log corrupted at line {broken_at}")
        repaired = lines[:broken_at - 1]
        last_hash, broken_at_2 = self._verify_lines(repaired)
        if broken_at_2 is not None:
            raise RuntimeError(
                f"decision-log corrupted at line {broken_at_2} "
                "(after dropping a torn trailing record)")
        self.path.write_text(
            "".join(line + "\n" for line in repaired), encoding="utf-8")
        return last_hash if last_hash is not None else GENESIS

    def contains_hash(self, wanted: str) -> bool:
        if wanted == GENESIS:
            return True
        if not self.path.exists():
            return False
        with self.path.open("r", encoding="utf-8") as stream:
            return any(json.loads(line).get("event_sha256") == wanted
                       for line in stream if line.strip())

    def append(self, kind: str, payload: dict, *, iteration: int | None = None) -> str:
        event = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind),
            "iteration": None if iteration is None else int(iteration),
            "payload": payload,
            "previous_sha256": self._last_hash,
        }
        event_hash = hashlib.sha256(_canonical(event)).hexdigest()
        stored = dict(event, event_sha256=event_hash)
        encoded = json.dumps(stored, sort_keys=True, ensure_ascii=False,
                             allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._last_hash = event_hash
        return event_hash

    @property
    def last_hash(self) -> str:
        return self._last_hash
