"""Sparse, uncertainty-aware payoff evidence with resumable transactions.

Screening evidence is deliberately separated from confirmatory/solver
evidence. Partial evaluator output remains in the transaction journal and is
not visible to the meta-solver until a complete paired seed block commits.
Intervals are computed once at commit time and cached.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from cuda_fdm.league import score_block_interval


EDGE_PROTOCOL = "sparse_paired_payoff_graph_v2"
TX_STATES = {"planned", "running", "partial", "committed", "invalid"}
EVIDENCE_PHASES = {"screening", "confirmatory", "solver", "migrated"}
SOLVER_PHASES = {"confirmatory", "solver", "migrated"}


def _pair(left: int, right: int) -> tuple[int, int, bool]:
    left, right = int(left), int(right)
    if left == right:
        raise ValueError("self payoff edges are implicit and must not be stored")
    return (left, right, False) if left < right else (right, left, True)


def _key(left: int, right: int) -> str:
    low, high, _ = _pair(left, right)
    return f"{low}:{high}"


@dataclass(frozen=True)
class EdgeEstimate:
    left: int
    right: int
    known: bool
    posterior_mean: float | None
    lcb: float | None
    ucb: float | None
    paired_blocks: int
    paired_games: int
    last_iteration: int | None
    width: float
    wins: float = 0.0
    draws: float = 0.0
    losses: float = 0.0
    edge_status: str = "unknown"


class SparsePayoffGraph:
    """Stores evaluated edges without coercing missing evidence to a draw."""

    def __init__(self, *, confidence: float = 0.95, state: dict | None = None):
        self.confidence = float(confidence)
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        self.edges: dict[str, dict] = {}
        self.transactions: dict[str, dict] = {}
        if state is not None:
            self.load_state_dict(state)

    @staticmethod
    def transaction_id(left: int, right: int, seed_block: str | int, *,
                       phase: str = "solver", evaluator_protocol: str = "unknown",
                       scenario_bank_version: str = "unknown") -> str:
        low, high, _ = _pair(left, right)
        raw = (f"{low}:{high}:{phase}:{evaluator_protocol}:"
               f"{scenario_bank_version}:{seed_block}").encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def plan(self, left: int, right: int, seed_block: str | int, *, reason: str,
             iteration: int, phase: str = "solver",
             evaluator_protocol: str = "unknown",
             scenario_bank_version: str = "unknown") -> str:
        if phase not in EVIDENCE_PHASES:
            raise ValueError(f"unsupported payoff evidence phase: {phase}")
        if not evaluator_protocol or not scenario_bank_version:
            raise ValueError("evaluator protocol and scenario-bank version are required")
        low, high, _ = _pair(left, right)
        txid = self.transaction_id(
            low, high, seed_block, phase=phase,
            evaluator_protocol=evaluator_protocol,
            scenario_bank_version=scenario_bank_version)
        existing = self.transactions.get(txid)
        if existing is not None:
            if existing["state"] == "committed":
                raise RuntimeError("payoff seed block was already committed")
            if existing["state"] != "invalid":
                return txid
        self.transactions[txid] = {
            "id": txid, "left": low, "right": high,
            "seed_block": str(seed_block), "reason": str(reason),
            "phase": phase, "evaluator_protocol": str(evaluator_protocol),
            "scenario_bank_version": str(scenario_bank_version),
            "planned_iteration": int(iteration), "state": "planned",
            "pending": [], "pending_wdl": {"wins": 0.0, "draws": 0.0, "losses": 0.0},
        }
        return txid

    def start(self, txid: str) -> None:
        tx = self.transactions[txid]
        if tx["state"] == "running":
            # A process may die after journalling ``running`` but before any
            # result is committed. Re-entering the same deterministic block is
            # safe and must not require deleting transaction evidence.
            return
        if tx["state"] not in {"planned", "partial"}:
            raise RuntimeError(f"cannot start payoff transaction in state {tx['state']}")
        tx["state"] = "running"

    @staticmethod
    def _clean_scores(paired_scores) -> np.ndarray:
        scores = np.asarray(paired_scores, dtype=np.float64).reshape(-1)
        if (scores.size == 0 or np.any(~np.isfinite(scores))
                or np.any((scores < 0.0) | (scores > 1.0))):
            raise ValueError("paired payoff scores must be finite values in [0, 1]")
        return scores

    @staticmethod
    def _clean_wdl(wins, draws, losses) -> tuple[float, float, float]:
        if wins is None and draws is None and losses is None:
            return 0.0, 0.0, 0.0
        values = np.asarray([wins, draws, losses], dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("W/D/L evidence must be finite and non-negative")
        return tuple(float(value) for value in values)

    def commit(self, txid: str, paired_scores, *, iteration: int,
               complete: bool = True, paired_game_ids=None,
               wins=None, draws=None, losses=None,
               metadata: dict | None = None,
               refresh_cache: bool = True) -> EdgeEstimate:
        tx = self.transactions[txid]
        if tx["state"] not in {"planned", "running", "partial"}:
            raise RuntimeError(f"cannot commit payoff transaction in state {tx['state']}")
        scores = self._clean_scores(paired_scores)
        if paired_game_ids is None:
            offset = len(tx.get("pending", []))
            paired_game_ids = [f"{tx['seed_block']}:{offset + i}" for i in range(scores.size)]
        ids = [str(value) for value in paired_game_ids]
        if len(ids) != scores.size or len(set(ids)) != len(ids):
            raise ValueError("paired game IDs must be unique and match paired scores")

        pending = {str(item["id"]): float(item["score"])
                   for item in tx.get("pending", [])}
        overlap = {game_id for game_id in ids if game_id in pending}
        # A complete retry of an already journalled fragment is idempotent.
        # A partly-overlapping fragment is ambiguous because aggregate W/D/L
        # cannot be split safely between its old and new game IDs.
        replay_only = bool(ids) and len(overlap) == len(ids)
        if overlap and not replay_only:
            raise RuntimeError("partial payoff replay overlaps pending game IDs")
        for game_id, score in zip(ids, scores):
            if game_id in pending and not math.isclose(
                    pending[game_id], float(score), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("idempotent payoff replay changed a paired result")
            pending[game_id] = float(score)
        tx["pending"] = [{"id": key, "score": pending[key]} for key in sorted(pending)]

        add_w, add_d, add_l = self._clean_wdl(wins, draws, losses)
        wdl = tx.setdefault("pending_wdl", {"wins": 0.0, "draws": 0.0, "losses": 0.0})
        if not replay_only:
            wdl["wins"] += add_w
            wdl["draws"] += add_d
            wdl["losses"] += add_l

        if not complete:
            tx["state"] = "partial"
            tx["last_partial_iteration"] = int(iteration)
            return self.estimate(tx["left"], tx["right"])

        key = _key(tx["left"], tx["right"])
        edge = self.edges.setdefault(key, {
            "left": tx["left"], "right": tx["right"], "blocks": [], "cache": {}})
        identity = (tx["phase"], tx["evaluator_protocol"],
                    tx["scenario_bank_version"], tx["seed_block"])
        if any((block["phase"], block["evaluator_protocol"],
                block["scenario_bank_version"], block["seed_block"]) == identity
               for block in edge["blocks"]):
            raise RuntimeError("duplicate payoff evidence commit")
        committed_scores = [float(item["score"]) for item in tx["pending"]]
        clean_metadata = json.loads(json.dumps(metadata or {}, sort_keys=True,
                                               allow_nan=False))
        edge["blocks"].append({
            "seed_block": tx["seed_block"], "scores": committed_scores,
            "paired_game_ids": [item["id"] for item in tx["pending"]],
            "iteration": int(iteration), "transaction_id": txid,
            "phase": tx["phase"], "evaluator_protocol": tx["evaluator_protocol"],
            "scenario_bank_version": tx["scenario_bank_version"],
            "wins": float(wdl["wins"]), "draws": float(wdl["draws"]),
            "losses": float(wdl["losses"]), "status": "committed",
            "metadata": clean_metadata,
        })
        tx["state"] = "committed"
        tx["committed_iteration"] = int(iteration)
        tx["paired_blocks"] = len(committed_scores)
        tx.pop("pending", None)
        tx.pop("pending_wdl", None)
        if refresh_cache:
            self._refresh_cache(edge)
        return self.estimate(tx["left"], tx["right"])

    def invalidate(self, txid: str, reason: str) -> None:
        tx = self.transactions[txid]
        if tx["state"] == "committed":
            raise RuntimeError("committed payoff evidence cannot be invalidated in place")
        tx["state"] = "invalid"
        tx["invalid_reason"] = str(reason)

    @staticmethod
    def _phase_blocks(edge: dict, phases: set[str]) -> list[dict]:
        return [block for block in edge.get("blocks", []) if block.get("phase") in phases]

    def _cache_from_blocks(self, edge: dict, blocks: list[dict], cache_name: str):
        scores = np.asarray([score for block in blocks for score in block["scores"]],
                            dtype=np.float64)
        if scores.size == 0:
            return None
        successes = float(scores.sum())
        posterior = (successes + 0.5) / (float(scores.size) + 1.0)
        seed_raw = f"{edge['left']}:{edge['right']}:{cache_name}".encode("utf-8")
        seed = int(hashlib.sha256(seed_raw).hexdigest()[:12], 16)
        low, high = score_block_interval(
            scores, confidence=self.confidence, samples=4096, seed=seed)
        return {
            "posterior_mean": float(posterior), "lcb": float(low), "ucb": float(high),
            "paired_blocks": int(scores.size), "paired_games": int(scores.size * 2),
            "last_iteration": max(int(block["iteration"]) for block in blocks),
            "wins": sum(float(block.get("wins", 0.0)) for block in blocks),
            "draws": sum(float(block.get("draws", 0.0)) for block in blocks),
            "losses": sum(float(block.get("losses", 0.0)) for block in blocks),
        }

    def _refresh_cache(self, edge: dict) -> None:
        cache_root = edge.setdefault("cache", {})
        for key in list(cache_root):
            if key.startswith("slice:"):
                cache_root.pop(key, None)
        for cache_name, phases in (("solver", SOLVER_PHASES), ("all", EVIDENCE_PHASES)):
            blocks = self._phase_blocks(edge, phases)
            cache = self._cache_from_blocks(edge, blocks, cache_name)
            if cache is None:
                cache_root.pop(cache_name, None)
                continue
            cache_root[cache_name] = cache

    @staticmethod
    def _estimate_from_cache(left: int, right: int, reverse: bool,
                             cache: dict | None) -> EdgeEstimate:
        if cache is None:
            return EdgeEstimate(int(left), int(right), False, None, None, None,
                                0, 0, None, 1.0)
        posterior = float(cache["posterior_mean"])
        lcb, ucb = float(cache["lcb"]), float(cache["ucb"])
        wins, draws, losses = (float(cache["wins"]), float(cache["draws"]),
                               float(cache["losses"]))
        if reverse:
            posterior, lcb, ucb = 1.0 - posterior, 1.0 - ucb, 1.0 - lcb
            wins, losses = losses, wins
        status = "uncertain" if lcb <= 0.5 <= ucb else "solver_ready"
        return EdgeEstimate(
            int(left), int(right), True, posterior, lcb, ucb,
            int(cache["paired_blocks"]), int(cache["paired_games"]),
            int(cache["last_iteration"]), max(0.0, ucb - lcb),
            wins, draws, losses, status)

    def estimate(self, left: int, right: int, *, include_screening: bool = False) -> EdgeEstimate:
        low, high, reverse = _pair(left, right)
        edge = self.edges.get(f"{low}:{high}")
        cache_name = "all" if include_screening else "solver"
        cache = None if edge is None else edge.get("cache", {}).get(cache_name)
        return self._estimate_from_cache(left, right, reverse, cache)

    def estimate_solver_slice(self, left: int, right: int, *,
                              evaluator_protocol: str,
                              scenario_bank_versions,
                              iteration: int | None = None,
                              phases=SOLVER_PHASES) -> EdgeEstimate:
        low, high, reverse = _pair(left, right)
        edge = self.edges.get(f"{low}:{high}")
        if edge is None:
            return self._estimate_from_cache(left, right, reverse, None)
        banks = tuple(sorted(set(map(str, scenario_bank_versions))))
        phase_names = tuple(sorted(set(map(str, phases))))
        key_payload = json.dumps({
            "protocol": str(evaluator_protocol), "banks": banks,
            "phases": phase_names, "iteration": iteration,
        }, sort_keys=True, separators=(",", ":"))
        cache_name = "slice:" + hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:20]
        cache_root = edge.setdefault("cache", {})
        cache = cache_root.get(cache_name)
        if cache is None:
            blocks = [
                block for block in edge.get("blocks", [])
                if block.get("phase") in phase_names
                and str(block.get("evaluator_protocol")) == str(evaluator_protocol)
                and str(block.get("scenario_bank_version")) in banks
                and (iteration is None or int(block.get("iteration", -1)) == int(iteration))
            ]
            cache = self._cache_from_blocks(edge, blocks, cache_name)
            if cache is not None:
                cache_root[cache_name] = cache
        return self._estimate_from_cache(left, right, reverse, cache)

    def phase_blocks(self, left: int, right: int, phase: str) -> int:
        edge = self.edges.get(_key(left, right))
        if edge is None:
            return 0
        return sum(len(block.get("scores", [])) for block in edge.get("blocks", [])
                   if block.get("phase") == phase)

    def query_eligible(self, left: int, right: int, *, phase: str,
                       required_blocks: int, maximum_blocks: int,
                       iteration: int, stale_after_iterations: int) -> bool:
        if phase not in EVIDENCE_PHASES:
            raise ValueError("unknown payoff query phase")
        count = self.phase_blocks(left, right, phase)
        if count < int(required_blocks):
            return True
        if count >= int(maximum_blocks):
            return False
        edge = self.edges.get(_key(left, right), {})
        phase_iterations = [int(block["iteration"]) for block in edge.get("blocks", [])
                            if block.get("phase") == phase]
        return bool(phase_iterations
                    and int(iteration) - max(phase_iterations)
                    >= int(stale_after_iterations))

    def needs_more(self, left: int, right: int, *, minimum_blocks: int,
                   maximum_blocks: int, decision_boundary: float = 0.5) -> bool:
        """Compatibility helper for fixed-size designs; CI peeking is forbidden."""
        del decision_boundary
        count = self.estimate(left, right).paired_blocks
        return count < min(int(minimum_blocks), int(maximum_blocks))

    def conservative_value(self, left: int, right: int) -> float:
        estimate = self.estimate(left, right)
        if (not estimate.known or estimate.lcb is None or estimate.ucb is None
                or estimate.lcb <= 0.5 <= estimate.ucb):
            return 0.5
        return float(estimate.posterior_mean)

    def solver_edge_eligible(self, left: int, right: int, *,
                             minimum_blocks: int = 1,
                             evaluator_protocol: str | None = None,
                             scenario_bank_versions=None) -> bool:
        estimate = (self.estimate(left, right)
                    if evaluator_protocol is None else self.estimate_solver_slice(
                        left, right, evaluator_protocol=evaluator_protocol,
                        scenario_bank_versions=scenario_bank_versions or ()))
        return bool(estimate.known
                    and estimate.paired_blocks >= int(minimum_blocks))

    def missing_solver_pairs(self, policy_ids, *,
                             minimum_blocks: int = 1,
                             evaluator_protocol: str | None = None,
                             scenario_bank_versions=None) -> list[tuple[int, int]]:
        # Queries and committed paired scores use canonical low-ID perspective.
        ids = sorted(set(int(value) for value in policy_ids))
        return [(left, right)
                for index, left in enumerate(ids)
                for right in ids[index + 1:]
                if not self.solver_edge_eligible(
                    left, right, minimum_blocks=minimum_blocks,
                    evaluator_protocol=evaluator_protocol,
                    scenario_bank_versions=scenario_bank_versions)]

    def assert_complete_solver_subgame(self, policy_ids, *,
                                       minimum_blocks: int = 1,
                                       evaluator_protocol: str | None = None,
                                       scenario_bank_versions=None) -> None:
        missing = self.missing_solver_pairs(
            policy_ids, minimum_blocks=minimum_blocks,
            evaluator_protocol=evaluator_protocol,
            scenario_bank_versions=scenario_bank_versions)
        if missing:
            preview = ", ".join(f"{left}:{right}" for left, right in missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise RuntimeError(
                f"Nash solver received {len(missing)} missing payoff edges "
                f"(minimum_blocks={int(minimum_blocks)}): {preview}{suffix}")

    def conservative_nash(self, policy_ids, *, iterations: int = 4000,
                          minimum_blocks: int = 1,
                          evaluator_protocol: str | None = None,
                          scenario_bank_versions=None) -> dict[int, float]:
        ids = list(dict.fromkeys(int(value) for value in policy_ids))
        if not ids:
            return {}
        if len(ids) == 1:
            return {ids[0]: 1.0}
        # Confidence-crossing measured edges remain neutral. Missing edges are
        # categorically different and must never acquire Nash mass as a draw.
        self.assert_complete_solver_subgame(
            ids, minimum_blocks=minimum_blocks,
            evaluator_protocol=evaluator_protocol,
            scenario_bank_versions=scenario_bank_versions)
        matrix = np.zeros((len(ids), len(ids)), dtype=np.float64)
        for i, left in enumerate(ids):
            for j in range(i + 1, len(ids)):
                estimate = (self.estimate(left, ids[j])
                            if evaluator_protocol is None else self.estimate_solver_slice(
                                left, ids[j], evaluator_protocol=evaluator_protocol,
                                scenario_bank_versions=scenario_bank_versions or ()))
                value01 = (0.5 if estimate.lcb <= 0.5 <= estimate.ucb
                           else float(estimate.posterior_mean))
                value = 2.0 * value01 - 1.0
                matrix[i, j], matrix[j, i] = value, -value
        weights = np.ones(len(ids), dtype=np.float64)
        average = np.zeros(len(ids), dtype=np.float64)
        eta = math.sqrt(2.0 * math.log(max(len(ids), 2)) / max(int(iterations), 1))
        for _ in range(max(int(iterations), 1)):
            distribution = weights / weights.sum()
            average += distribution
            weights *= np.exp(np.clip(eta * (matrix @ distribution), -20.0, 20.0))
            weights = np.maximum(weights, 1e-300)
        average /= average.sum()
        return {identity: float(average[index]) for index, identity in enumerate(ids)}

    def confident_cycle_members(self, policy_ids) -> tuple[int, ...]:
        """Return policies in a decisive directed three-cycle.

        An orientation is used only when its complete 95% interval lies on one
        side of 0.5. Unknown and confidence-crossing edges cannot manufacture a
        cycle sentinel.
        """
        ids = set(map(int, policy_ids))
        beats = {identity: set() for identity in ids}
        beaten_by = {identity: set() for identity in ids}
        for edge in self.edges.values():
            left, right = int(edge["left"]), int(edge["right"])
            if left not in ids or right not in ids:
                continue
            estimate = self.estimate(left, right)
            if not estimate.known:
                continue
            if float(estimate.lcb) > 0.5:
                winner, loser = left, right
            elif float(estimate.ucb) < 0.5:
                winner, loser = right, left
            else:
                continue
            beats[winner].add(loser)
            beaten_by[loser].add(winner)
        members = set()
        for first in ids:
            for second in beats[first]:
                thirds = beats[second].intersection(beaten_by[first])
                for third in thirds:
                    members.update((first, second, third))
        return tuple(sorted(members))

    def known_neighbors(self, identity: int) -> list[int]:
        identity = int(identity)
        result = []
        for edge in self.edges.values():
            if edge["left"] == identity:
                result.append(int(edge["right"]))
            elif edge["right"] == identity:
                result.append(int(edge["left"]))
        return sorted(set(result))

    def rebuild_caches(self) -> None:
        for edge in self.edges.values():
            self._refresh_cache(edge)

    def state_dict(self) -> dict:
        return {"protocol": EDGE_PROTOCOL, "confidence": self.confidence,
                "edges": self.edges, "transactions": self.transactions}

    def load_state_dict(self, state: dict) -> None:
        if state.get("protocol") != EDGE_PROTOCOL:
            raise ValueError("sparse payoff graph protocol mismatch")
        if not math.isclose(float(state.get("confidence")), self.confidence,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("payoff confidence changed across resume")
        self.edges = json.loads(json.dumps(state.get("edges", {}), allow_nan=False))
        self.transactions = json.loads(json.dumps(state.get("transactions", {}), allow_nan=False))
        for tx in self.transactions.values():
            if tx.get("state") not in TX_STATES or tx.get("phase") not in EVIDENCE_PHASES:
                raise ValueError("invalid payoff transaction state")
        for key, edge in self.edges.items():
            if key != _key(edge["left"], edge["right"]):
                raise ValueError("payoff edge key mismatch")
            seen = set()
            for block in edge.get("blocks", []):
                identity = (block.get("phase"), block.get("evaluator_protocol"),
                            block.get("scenario_bank_version"), block.get("seed_block"))
                if identity in seen or block.get("phase") not in EVIDENCE_PHASES:
                    raise ValueError("duplicate or invalid payoff block in saved graph")
                seen.add(identity)
                self._clean_scores(block.get("scores", []))
            self._refresh_cache(edge)


__all__ = ["EDGE_PROTOCOL", "EVIDENCE_PHASES", "SOLVER_PHASES",
           "EdgeEstimate", "SparsePayoffGraph"]
