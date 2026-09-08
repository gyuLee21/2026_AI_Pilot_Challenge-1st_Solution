"""Persistent, time-sliced side-learner lineage metadata and scheduling."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path


LINEAGE_ROLES = ("ME-EIE", "LE", "ME-ERE")
RESET_REASONS = {
    "checkpoint_corruption", "long_stagnation", "vanishing_gradient",
    "vanishing_kl", "support_contraction", "material_target_shift",
    "protocol_change",
}


@dataclass
class LineageRecord:
    lineage_id: str
    role: str
    parent_policy_ids: list[int]
    model_file: str
    optimizer_file: str
    normalizer_file: str
    rng_file: str
    policy_archive_id: int | None = None
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    protocol_version: str = "active_league_vnext_100k_v2"
    current_target_version: str = "solver_epoch_0"
    scheduler_state: dict = field(default_factory=dict)
    target_distribution: dict = field(default_factory=dict)
    behavior_history: list[dict] = field(default_factory=list)
    last_useful_iteration: int = 0
    last_trained_iteration: int = 0
    slices: int = 0
    total_side_iterations: int = 0
    total_side_transitions: int = 0
    status: str = "idle"
    plasticity: dict = field(default_factory=dict)
    reset_history: list[dict] = field(default_factory=list)

    def validate(self) -> None:
        if self.role not in LINEAGE_ROLES:
            raise ValueError("unknown side-learner role")
        if self.status not in {"idle", "active", "retired", "blocked"}:
            raise ValueError("invalid lineage status")
        if not all((self.model_file, self.optimizer_file,
                    self.normalizer_file, self.rng_file)):
            raise ValueError("persistent lineage checkpoint components are required")
        if not self.protocol_version or not self.current_target_version:
            raise ValueError("lineage protocol and target versions are required")
        for name, value in self.artifact_sha256.items():
            if name not in {"model", "optimizer", "normalizer", "rng"}:
                raise ValueError("unknown lineage artifact hash")
            if len(str(value)) != 64:
                raise ValueError("lineage artifact SHA-256 is invalid")


class LineageManager:
    def __init__(self, state: dict | None = None):
        self.records: dict[str, LineageRecord] = {}
        self.active_lineage_id: str | None = None
        self.role_cursor = 0
        self.candidate_queue: list[dict] = []
        if state is not None:
            self.load_state_dict(state)

    def register(self, record: LineageRecord) -> None:
        record.validate()
        if record.lineage_id in self.records:
            raise RuntimeError("lineage id already exists")
        self.records[record.lineage_id] = record

    def queue_candidate(self, archive_id: int, *, role: str, reason: str,
                        iteration: int, evidence: dict) -> None:
        if role not in LINEAGE_ROLES:
            raise ValueError("unknown candidate lineage role")
        if not evidence:
            raise ValueError("queued lineage candidate requires evidence")
        archive_id = int(archive_id)
        existing = next((item for item in self.candidate_queue
                         if int(item["archive_id"]) == archive_id), None)
        value = {"archive_id": archive_id, "role": role,
                 "reason": str(reason), "iteration": int(iteration),
                 "evidence": dict(evidence), "status": "queued"}
        if existing is None:
            self.candidate_queue.append(value)
        else:
            if existing.get("role") != role or existing.get("reason") != str(reason):
                raise RuntimeError("queued lineage candidate changed role or reason")
            history = existing.setdefault("evidence_history", [dict(existing["evidence"])])
            if value["evidence"] not in history:
                history.append(value["evidence"])
            existing["last_observed_iteration"] = int(iteration)

    def choose_next(self, *, iteration: int, role_pressure: dict[str, float] | None = None) -> str:
        if self.active_lineage_id is not None:
            raise RuntimeError("a GPU side learner is already active")
        pressure = {role: float((role_pressure or {}).get(role, 0.0)) for role in LINEAGE_ROLES}
        if any(not math.isfinite(value) for value in pressure.values()):
            raise FloatingPointError("non-finite lineage role pressure")
        available = [record for record in self.records.values() if record.status == "idle"]
        if not available:
            raise RuntimeError("no persistent side lineage is available")
        fallback_role = LINEAGE_ROLES[self.role_cursor % len(LINEAGE_ROLES)]

        def priority(record: LineageRecord):
            staleness = max(0, int(iteration) - int(record.last_trained_iteration))
            fallback = 1.0 if record.role == fallback_role else 0.0
            return (pressure[record.role], staleness, fallback, -record.slices)

        selected = max(available, key=priority)
        selected.status = "active"
        self.active_lineage_id = selected.lineage_id
        self.role_cursor = (LINEAGE_ROLES.index(selected.role) + 1) % len(LINEAGE_ROLES)
        return selected.lineage_id

    def finish_slice(self, lineage_id: str, *, iteration: int,
                     useful: bool, plasticity: dict,
                     side_iterations: int = 0, side_transitions: int = 0,
                     target_version: str | None = None) -> None:
        if self.active_lineage_id != lineage_id:
            raise RuntimeError("lineage is not the active GPU learner")
        record = self.records[lineage_id]
        record.last_trained_iteration = int(iteration)
        record.slices += 1
        if int(side_iterations) < 0 or int(side_transitions) < 0:
            raise ValueError("lineage experience counters cannot be negative")
        record.total_side_iterations += int(side_iterations)
        record.total_side_transitions += int(side_transitions)
        if target_version is not None:
            record.current_target_version = str(target_version)
        record.plasticity = dict(plasticity)
        if useful:
            record.last_useful_iteration = int(iteration)
        record.status = "idle"
        self.active_lineage_id = None

    def reset(self, lineage_id: str, *, reason: str, iteration: int,
              evidence: dict) -> None:
        if reason not in RESET_REASONS:
            raise ValueError("unapproved lineage reset reason")
        if not evidence:
            raise ValueError("lineage reset requires evidence")
        record = self.records[lineage_id]
        if record.status == "active":
            raise RuntimeError("active lineage cannot be reset")
        record.reset_history.append({"iteration": int(iteration),
                                     "reason": reason, "evidence": dict(evidence)})
        record.slices = 0
        record.last_trained_iteration = int(iteration)
        record.plasticity = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def verify_artifacts(self, lineage_id: str, *, root,
                         expected_protocol: str,
                         expected_target_version: str | None = None) -> None:
        record = self.records[lineage_id]
        record.validate()
        if record.protocol_version != str(expected_protocol):
            raise ValueError("lineage protocol mismatch")
        if (expected_target_version is not None
                and record.current_target_version != str(expected_target_version)):
            raise ValueError("lineage target version mismatch")
        base = Path(root).resolve()
        refs = {"model": record.model_file, "optimizer": record.optimizer_file,
                "normalizer": record.normalizer_file, "rng": record.rng_file}
        for name, reference in refs.items():
            path_text = str(reference).split("#", 1)[0]
            path = (base / path_text).resolve()
            if base not in path.parents and path != base:
                raise ValueError("lineage artifact escapes its runtime root")
            expected = record.artifact_sha256.get(name)
            if not path.is_file() or expected is None or self._sha256(path) != expected:
                raise RuntimeError(f"lineage {name} artifact is missing or changed")

    def state_dict(self) -> dict:
        return {
            "records": [asdict(self.records[key]) for key in sorted(self.records)],
            "active_lineage_id": self.active_lineage_id,
            "role_cursor": self.role_cursor,
            "candidate_queue": list(self.candidate_queue),
        }

    def load_state_dict(self, state: dict) -> None:
        self.records = {}
        for value in state.get("records", []):
            record = LineageRecord(**value)
            record.validate()
            if record.lineage_id in self.records:
                raise ValueError("duplicate lineage in saved state")
            self.records[record.lineage_id] = record
        self.active_lineage_id = state.get("active_lineage_id")
        self.role_cursor = int(state.get("role_cursor", 0))
        self.candidate_queue = [dict(value) for value in state.get("candidate_queue", [])]
        candidate_ids = [int(value["archive_id"]) for value in self.candidate_queue]
        if (len(candidate_ids) != len(set(candidate_ids))
                or any(value.get("status") not in {"queued", "claimed", "complete"}
                       for value in self.candidate_queue)):
            raise ValueError("invalid persistent-lineage candidate queue")
        active = [key for key, record in self.records.items() if record.status == "active"]
        if (len(active) > 1 or (active and active[0] != self.active_lineage_id)
                or (self.active_lineage_id is not None and not active)):
            raise ValueError("saved lineage GPU ownership is inconsistent")


__all__ = ["LINEAGE_ROLES", "RESET_REASONS", "LineageRecord", "LineageManager"]
