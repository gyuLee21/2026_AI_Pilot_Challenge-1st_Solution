"""Rebuildable O(1) operational index over the append-only cold archive."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


INDEX_PROTOCOL = "active_league_archive_materialized_index_v1"


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _role(kind: str) -> str:
    kind = str(kind)
    if kind == "altitude_sentinel":
        return "altitude_sentinel"
    if kind.startswith("recent"):
        return "main_recent"
    if kind.startswith("milestone"):
        return "main_milestone"
    if "eie" in kind:
        return "eie"
    if "ere" in kind:
        return "ere"
    if kind.startswith("exploiter_le") or kind == "le":
        return "le"
    if "champion" in kind:
        return "champion"
    if "heldout" in kind or "rejected" in kind:
        return "heldout"
    return "main_milestone"


class ArchiveMaterializedIndex:
    """Keeps derived metadata out of the immutable raw policy ledger.

    The index can always be rebuilt from the archive manifest and sparse graph.
    A damaged index never becomes training evidence silently.
    """

    def __init__(self, root, *, state: dict | None = None):
        self.root = Path(root).resolve()
        self.path = self.root / "archive_index.json"
        self.events = self.root / "archive_index_events.jsonl"
        self.records: dict[int, dict] = {}
        self.anchor_ids: list[int] = []
        self.source_manifest_sha256: str | None = None
        self.epoch = 0
        if state is not None:
            self.load_state_dict(state)
        elif self.path.is_file():
            self.load_state_dict(json.loads(self.path.read_text(encoding="utf-8")))

    @staticmethod
    def _anchors(records: dict[int, dict], cap: int = 32) -> list[int]:
        eligible = [identity for identity, record in records.items()
                    if record.get("payoff_eligible", True)
                    and record.get("admitted", True)
                    and not record.get("invalid", False)]
        eligible.sort(key=lambda identity: (
            float(records[identity].get("nash_mass", 0.0)),
            float(records[identity].get("regression", 0.0)),
            int(records[identity].get("iteration", -1)), identity), reverse=True)
        return sorted(eligible[:int(cap)])

    def rebuild(self, archive_state: dict, graph, *, iteration: int,
                source_manifest_sha256: str | None = None,
                model_schema_hash: str = "unknown",
                observation_schema_hash: str = "obs184_official_v1",
                action_schema_hash: str = "factorized_categorical_4x21_v1",
                normalizer_hash: str = "unknown",
                persist: bool = True) -> dict:
        raw = {int(record["id"]): dict(record)
               for record in archive_state.get("records", [])}
        if len(raw) != len(archive_state.get("records", [])):
            raise ValueError("archive materialized index saw duplicate policy IDs")
        anchors = self._anchors(raw)
        enriched = {}
        for identity in sorted(raw):
            record = raw[identity]
            fingerprint = [graph.conservative_value(identity, anchor)
                           if identity != anchor else 0.5 for anchor in anchors]
            admitted = bool(record.get("admitted", True))
            safety = "quarantined" if record.get("invalid", False) else "valid"
            value = dict(record)
            value.update({
                "archive_id": identity,
                "content_sha256": str(record.get("sha256", "")),
                "protocol_version": str(archive_state.get("protocol", "unknown")),
                "source_checkpoint_id": str(record.get("source_checkpoint_id", "migration_seed")),
                "creation_main_iter": int(record.get("iteration", iteration)),
                "creation_wallclock": record.get("creation_wallclock"),
                "policy_role": _role(record.get("kind", "")),
                "parent_archive_ids": list(record.get("parent_archive_ids", [])),
                "lineage_id": record.get("lineage_id"),
                "model_schema_hash": str(record.get("model_schema_hash", model_schema_hash)),
                "observation_schema_hash": str(record.get(
                    "observation_schema_hash", observation_schema_hash)),
                "action_schema_hash": str(record.get("action_schema_hash", action_schema_hash)),
                "normalizer_hash": str(record.get("normalizer_hash", normalizer_hash)),
                "reward_profile": record.get("profile"),
                "training_target_descriptor": record.get("training_target_descriptor"),
                "admission_status": str(record.get(
                    "admission_status", "admitted" if admitted else "heldout")),
                "safety_status": safety,
                "behavior_descriptor_version": record.get("behavior_descriptor_version"),
                "behavior_descriptor": record.get("behavior_descriptor"),
                "payoff_fingerprint_version": "sparse_conservative_anchor_v1",
                "payoff_fingerprint_anchor_ids": list(anchors),
                "payoff_fingerprint": fingerprint,
                "notes": str(record.get("notes", "")),
            })
            enriched[identity] = value

        previous_hash = _canonical_hash(self.state_dict()) if self.records else None
        self.records = enriched
        self.anchor_ids = anchors
        self.source_manifest_sha256 = source_manifest_sha256
        self.epoch += 1
        event = {
            "protocol": INDEX_PROTOCOL, "epoch": self.epoch,
            "iteration": int(iteration), "record_count": len(enriched),
            "anchor_ids": anchors, "previous_index_sha256": previous_hash,
            "index_sha256": _canonical_hash(self.state_dict()),
        }
        if persist:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            self.persist()
        return enriched

    def lookup(self, archive_id: int) -> dict:
        return dict(self.records[int(archive_id)])

    def state_dict(self) -> dict:
        return {
            "protocol": INDEX_PROTOCOL, "epoch": self.epoch,
            "source_manifest_sha256": self.source_manifest_sha256,
            "anchor_ids": list(self.anchor_ids),
            "records": [self.records[key] for key in sorted(self.records)],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("protocol") != INDEX_PROTOCOL:
            raise ValueError("archive materialized index protocol mismatch")
        records = {}
        for value in state.get("records", []):
            record = dict(value)
            # policy_role is derived metadata. Recompute it on load so old
            # checkpoints/index files carrying the pre-fix sentinel fallback
            # are repaired immediately without mutating the immutable archive.
            record["policy_role"] = _role(record.get("kind", ""))
            records[int(record["archive_id"])] = record
        if len(records) != len(state.get("records", [])):
            raise ValueError("duplicate archive ID in materialized index")
        self.records = records
        self.epoch = int(state.get("epoch", 0))
        self.anchor_ids = [int(value) for value in state.get("anchor_ids", [])]
        self.source_manifest_sha256 = state.get("source_manifest_sha256")

    def persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True,
                                        allow_nan=False), encoding="utf-8")
        os.replace(temporary, self.path)


__all__ = ["INDEX_PROTOCOL", "ArchiveMaterializedIndex"]
