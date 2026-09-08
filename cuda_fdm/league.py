"""Bounded active league and disk-backed policy archive.

The archive is append-only and may grow for the whole run.  Only policies
selected into the active league are materialised on the training GPU.  This
module deliberately contains no environment code so selection, checkpoint and
probability contracts can be tested on CPU.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch


LEGACY_LEAGUE_PROTOCOL = "cold_archive_active24_official_msl_stochastic_confident_v5"
LEAGUE_PROTOCOL = "cold_archive_active24_vnext_100k_conservative_v6"
POLICY_PROTOCOLS = {LEGACY_LEAGUE_PROTOCOL, LEAGUE_PROTOCOL}
ROLE_MIXTURE = {
    "latest": 0.20,
    "recent": 0.15,
    "hard": 0.35,
    "variance": 0.15,
    "forgotten": 0.10,
    "coverage": 0.05,
}
ACTIVE_LAYOUT = {"latest": 1, "recent": 4, "core": 16, "challenger": 3}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_content_hash(digest, value) -> None:
    """Deterministically hash tensors, normalizer state and schema metadata."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0" + str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            _update_content_hash(digest, str(key))
            _update_content_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _update_content_hash(digest, item)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
    else:
        raise TypeError(f"unsupported policy content-hash value: {type(value).__name__}")


def policy_content_sha256(model_state, norm_state) -> str:
    digest = hashlib.sha256()
    _update_content_hash(digest, {"model": model_state, "norm": norm_state})
    return digest.hexdigest()


def model_schema_sha256(model_state) -> str:
    schema = {str(key): {"shape": list(value.shape), "dtype": str(value.dtype)}
              for key, value in sorted(model_state.items())}
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def count_aware_alpha(games: float, half_life_games: float = 512.0) -> float:
    """EMA coefficient whose half life is expressed in completed games."""
    games = float(games)
    half_life_games = float(half_life_games)
    if not math.isfinite(games) or games < 0.0:
        raise ValueError("games must be finite and non-negative")
    if not math.isfinite(half_life_games) or half_life_games <= 0.0:
        raise ValueError("half_life_games must be finite and positive")
    return 1.0 - 2.0 ** (-games / half_life_games)


def wilson_lower(score: float, games: float, confidence: float = 0.95) -> float:
    """Wilson lower bound; draws are represented as half a success."""
    score, games = float(score), float(games)
    if games <= 0.0:
        return 0.0
    if not (math.isfinite(score) and 0.0 <= score <= 1.0 and math.isfinite(games)):
        raise ValueError("invalid score/games")
    z = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    den = 1.0 + z * z / games
    centre = score + z * z / (2.0 * games)
    radius = z * math.sqrt(score * (1.0 - score) / games + z * z / (4.0 * games * games))
    return max(0.0, min(1.0, (centre - radius) / den))


def multinomial_score_interval(wins: float, draws: float, losses: float,
                               confidence: float = 0.95, samples: int = 4096,
                               seed: int = 0) -> tuple[float, float]:
    """Deterministic Dirichlet-multinomial interval for W/D/L score.

    A draw is not treated as a fractional Bernoulli observation.  Jeffreys
    pseudo-counts keep all-win/all-loss samples from claiming zero uncertainty.
    """
    counts = np.asarray([wins, draws, losses], dtype=np.float64)
    if (np.any(~np.isfinite(counts)) or np.any(counts < 0.0)
            or float(counts.sum()) <= 0.0):
        raise ValueError("invalid W/D/L counts")
    if not (0.0 < float(confidence) < 1.0) or int(samples) < 128:
        raise ValueError("invalid interval configuration")
    rng = np.random.default_rng(int(seed) % (2**63 - 1))
    posterior = rng.dirichlet(counts + 0.5, size=int(samples))
    scores = posterior[:, 0] + 0.5 * posterior[:, 1]
    tail = 0.5 * (1.0 - float(confidence))
    low, high = np.quantile(scores, [tail, 1.0 - tail])
    return float(low), float(high)


def _bootstrap_jeffreys_interval(block_scores, confidence: float,
                                  samples: int, seed: int) -> tuple[float, float]:
    """Conservative interval for independent bounded paired-block scores.

    The ordinary nonparametric bootstrap collapses to zero width when every
    observed block is 0 or 1.  We retain that bootstrap, but envelope it with a
    paired-block Jeffreys beta posterior using fractional success/failure
    counts.  Finite all-win/all-loss and constant-draw samples therefore keep
    non-zero uncertainty without treating the two role-swapped games as
    independent observations.
    """
    scores = np.asarray(block_scores, dtype=np.float64).reshape(-1)
    if (scores.size == 0 or np.any(~np.isfinite(scores))
            or np.any((scores < 0.0) | (scores > 1.0))):
        raise ValueError("invalid mirrored block scores")
    if not (0.0 < float(confidence) < 1.0) or int(samples) < 128:
        raise ValueError("invalid block interval configuration")
    # Canonicalise the upper half so score-complement evaluations receive
    # exactly complementary bounds rather than Monte-Carlo-near complements.
    if float(scores.mean()) > 0.5:
        low, high = _bootstrap_jeffreys_interval(
            1.0 - scores, confidence, samples, seed)
        return 1.0 - high, 1.0 - low
    rng = np.random.default_rng(int(seed) % (2**63 - 1))
    indices = rng.integers(scores.size, size=(int(samples), scores.size))
    bootstrap_means = scores[indices].mean(1)
    successes = float(scores.sum())
    failures = float(scores.size) - successes
    jeffreys_means = rng.beta(successes + 0.5, failures + 0.5, size=int(samples))
    tail = 0.5 * (1.0 - float(confidence))
    boot_low, boot_high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    prior_low, prior_high = np.quantile(jeffreys_means, [tail, 1.0 - tail])
    return float(min(boot_low, prior_low)), float(max(boot_high, prior_high))


def paired_score_interval(first_scores, swapped_scores, confidence: float = 0.95,
                          samples: int = 4096, seed: int = 0) -> tuple[float, float]:
    """Interval over mirrored IC blocks, not over individual games.

    ``first_scores[k]`` and ``swapped_scores[k]`` are the same physical initial
    condition with policy slots exchanged.  The paired block mean is therefore
    the independent sampling unit used by admission/gating confidence bounds.
    """
    first = np.asarray(first_scores, dtype=np.float64).reshape(-1)
    swapped = np.asarray(swapped_scores, dtype=np.float64).reshape(-1)
    if (first.size == 0 or first.shape != swapped.shape
            or np.any(~np.isfinite(first)) or np.any(~np.isfinite(swapped))
            or np.any((first < 0.0) | (first > 1.0))
            or np.any((swapped < 0.0) | (swapped > 1.0))):
        raise ValueError("invalid paired evaluation scores")
    blocks = 0.5 * (first + swapped)
    return _bootstrap_jeffreys_interval(blocks, confidence, samples, seed)


def score_block_interval(block_scores, confidence: float = 0.95,
                         samples: int = 4096, seed: int = 0) -> tuple[float, float]:
    """Finite-sample interval accumulated over mirrored-IC audit blocks."""
    return _bootstrap_jeffreys_interval(block_scores, confidence, samples, seed)


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full(values.size, 1.0 / max(values.size, 1))


def _capped_normalise(values: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Water-fill a probability vector while respecting feasible per-row caps."""
    values = _normalise(values)
    caps = np.asarray(caps, dtype=np.float64)
    if values.size != caps.size or np.any(~np.isfinite(caps)) or np.any(caps <= 0.0):
        raise ValueError("invalid probability caps")
    if float(caps.sum()) < 1.0 - 1e-12:
        # Early league: a strict 12% cap is infeasible with only a few policies.
        caps = np.maximum(caps, 1.0 / values.size)
    result = np.zeros_like(values)
    free = np.ones(values.size, dtype=bool)
    remaining = 1.0
    raw = values.copy()
    while free.any() and remaining > 1e-12:
        proposal = _normalise(raw[free]) * remaining
        indices = np.flatnonzero(free)
        clipped = proposal > caps[indices] + 1e-15
        if not clipped.any():
            result[indices] = proposal
            remaining = 0.0
            break
        fixed = indices[clipped]
        result[fixed] = caps[fixed]
        remaining -= float(caps[fixed].sum())
        free[fixed] = False
    if remaining > 1e-9:
        result += remaining * _normalise(caps - result)
    return result / result.sum()


def role_mixture(entries: list[dict], non_latest_cap: float = 0.12) -> np.ndarray:
    """20/15/35/15/10/5 role mixture over active, non-retired entries.

    `latest` is explicitly a 20% singleton channel, so the 10--12% anti-monopoly
    cap applies to every *other* opponent. Missing channels are redistributed
    over the channels that currently have eligible policies.
    """
    active = [e for e in entries if not e.get("retired", False)]
    if not active:
        raise ValueError("active league is empty")
    q = np.clip(np.asarray([float(e.get("ema", 0.5)) for e in active]), 0.0, 1.0)
    regression = np.maximum(0.0, np.asarray([float(e.get("past_best", x)) for e, x in zip(active, q)]) - q)
    roles = [str(e.get("role", "core")) for e in active]
    coverage = np.asarray([bool(e.get("coverage", False)) for e in active])
    challenger = np.asarray([r == "challenger" for r in roles])
    core = np.asarray([r in ("core", "challenger") for r in roles])
    # Explicit warm-up schedule. Before a strategic core exists there are no
    # policies eligible for four of the six declared league channels. Generic
    # channel renormalisation plus probability caps otherwise produces an
    # opaque result that changes merely because another recent snapshot was
    # added. Keep half the games on the lagged latest clone and half on the
    # chronological recent ring until a prior milestone enters core.
    if not core.any():
        latest = np.asarray([r == "latest" for r in roles])
        recent = np.asarray([r == "recent" for r in roles])
        if latest.any() and recent.any():
            result = np.zeros(len(active), dtype=np.float64)
            result[latest] = 0.5 / int(latest.sum())
            result[recent] = 0.5 / int(recent.sum())
            return result
        eligible_early = latest | recent
        if eligible_early.any():
            result = np.zeros(len(active), dtype=np.float64)
            result[eligible_early] = 1.0 / int(eligible_early.sum())
            return result
    eligible = {
        "latest": np.asarray([r == "latest" for r in roles]),
        "recent": np.asarray([r == "recent" for r in roles]),
        "hard": core,
        "variance": core,
        "forgotten": core & ((regression > 0.0) | challenger),
        "coverage": core & coverage,
    }
    available_mass = sum(ROLE_MIXTURE[name] for name, mask in eligible.items() if mask.any())
    if available_mass <= 0.0:
        return np.full(len(active), 1.0 / len(active))
    result = np.zeros(len(active), dtype=np.float64)
    for name, mask in eligible.items():
        if not mask.any():
            continue
        mass = ROLE_MIXTURE[name] / available_mass
        if name == "hard":
            raw = np.square(1.0 - q[mask])
        elif name == "variance":
            raw = q[mask] * (1.0 - q[mask])
        elif name == "forgotten":
            raw = regression[mask] + 0.05 * challenger[mask].astype(np.float64)
        else:
            raw = np.ones(int(mask.sum()), dtype=np.float64)
        result[np.flatnonzero(mask)] += mass * _normalise(raw)
    caps = np.full(len(active), float(non_latest_cap), dtype=np.float64)
    caps[np.asarray([r == "latest" for r in roles])] = 0.20
    return _capped_normalise(result, caps)


def payoff_posterior_mean(entry: dict | None) -> float:
    """Estimate an edge without turning an uncertain edge into an exact draw.

    Production edges store one score per mirrored initial-condition block.
    Their fractional wins receive a Jeffreys Beta(1/2, 1/2) posterior mean.
    Legacy/synthetic edges fall back to cumulative W/D/L or score/game count.
    Confidence intervals are deliberately excluded here: they are used by the
    payoff-refresh scheduler to decide which edge needs more evaluation.
    """
    if not entry:
        return 0.5

    paired_scores: list[float] = []
    blocks = entry.get("blocks", [])
    if isinstance(blocks, list) and blocks:
        complete = all(isinstance(block, dict) and "paired_scores" in block
                       for block in blocks)
        if complete:
            paired_scores = [float(value) for block in blocks
                             for value in block["paired_scores"]]
            expected = entry.get("paired_blocks")
            if expected is not None and len(paired_scores) != int(expected):
                raise ValueError("payoff paired-block evidence count mismatch")

    if paired_scores:
        evidence = np.asarray(paired_scores, dtype=np.float64)
        if (np.any(~np.isfinite(evidence))
                or np.any((evidence < 0.0) | (evidence > 1.0))):
            raise ValueError("invalid payoff paired-block evidence")
        successes = float(evidence.sum())
        count = float(evidence.size)
    elif all(name in entry for name in ("wins", "draws", "losses")):
        wins = float(entry["wins"])
        draws = float(entry["draws"])
        losses = float(entry["losses"])
        evidence = np.asarray([wins, draws, losses], dtype=np.float64)
        if np.any(~np.isfinite(evidence)) or np.any(evidence < 0.0):
            raise ValueError("invalid payoff W/D/L evidence")
        successes = wins + 0.5 * draws
        count = wins + draws + losses
    else:
        score = float(entry.get("score", 0.5))
        count = float(entry.get("games", 0.0))
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            raise ValueError("invalid payoff score")
        if not math.isfinite(count) or count < 0.0:
            raise ValueError("invalid payoff game count")
        if count == 0.0:
            return score
        successes = score * count

    if not math.isfinite(count) or count <= 0.0:
        return 0.5
    posterior = (successes + 0.5) / (count + 1.0)
    if not math.isfinite(posterior) or not (0.0 <= posterior <= 1.0):
        raise ValueError("invalid payoff posterior mean")
    return float(posterior)


def empirical_nash(policy_ids: list[int], payoff: dict, iterations: int = 4000) -> dict[int, float]:
    """Approximate the current conservative symmetric meta-game.

    Unknown edges and measured edges whose fixed 95% interval crosses 0.5 are
    neutral in the mainline solver. Posterior means remain available to the
    shadow comparison, but cannot steer the production mixture prematurely.
    """
    ids = list(dict.fromkeys(int(x) for x in policy_ids))
    n = len(ids)
    if not n:
        return {}
    if n == 1:
        return {ids[0]: 1.0}
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, left in enumerate(ids):
        for j, right in enumerate(ids):
            if i == j:
                continue
            entry = payoff.get(str(left), {}).get(str(right), {})
            if entry:
                low = entry.get("lcb95")
                high = entry.get("ucb95")
                if (low is not None and high is not None
                        and not (float(low) <= 0.5 <= float(high))):
                    matrix[i, j] = 2.0 * payoff_posterior_mean(entry) - 1.0
    matrix = 0.5 * (matrix - matrix.T)
    weights = np.ones(n, dtype=np.float64)
    average = np.zeros(n, dtype=np.float64)
    eta = math.sqrt(2.0 * math.log(max(n, 2)) / max(int(iterations), 1))
    for _ in range(max(int(iterations), 1)):
        p = weights / weights.sum()
        average += p
        utility = matrix @ p
        weights *= np.exp(np.clip(eta * utility, -20.0, 20.0))
        weights = np.maximum(weights, 1e-300)
    average = _normalise(average)
    return {identity: float(average[i]) for i, identity in enumerate(ids)}


class LeagueArchive:
    """Append-only policy files plus compact metadata/payoff matrix."""

    def __init__(self, root, state=None):
        self.root = Path(root).resolve()
        self.policy_dir = self.root / "policies"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[int, dict] = {}
        self.payoff: dict[str, dict] = {}
        self.next_id = 0
        if state is not None:
            self.load_state_dict(state)

    def __len__(self):
        return len(self.records)

    def _path(self, identity: int) -> Path:
        return self.policy_dir / f"policy_{int(identity):05d}.pt"

    def _content_path(self, content_hash: str) -> Path:
        return self.policy_dir / f"policy_sha256_{str(content_hash)}.pt"

    def add(self, model_state, norm_state, *, kind, iteration, profile="standard",
            admitted=True, payoff_eligible=True, metrics=None) -> int:
        identity = self.next_id
        self.next_id += 1
        cpu_model = {k: v.detach().cpu() for k, v in model_state.items()}
        content_hash = policy_content_sha256(cpu_model, norm_state)
        # Legacy payloads embed one archive_id and cannot be shared with a new
        # vNext identity even when their model bytes are equal. The first vNext
        # occurrence materializes a content-addressed v6 object; later v6
        # identities may safely reuse that object.
        duplicate = next((record for record in self.records.values()
                          if record.get("content_sha256") == content_hash
                          and record.get("policy_protocol") == LEAGUE_PROTOCOL), None)
        if duplicate is None:
            payload = {"protocol": LEAGUE_PROTOCOL, "content_sha256": content_hash,
                       "model": cpu_model, "norm": norm_state}
            path = self._content_path(content_hash)
            if path.exists():
                # A crash can leave an immutable policy object ahead of the
                # manifest/checkpoint boundary. Reuse it only after validating
                # its semantic identity; never overwrite or delete it.
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if (existing.get("protocol") != LEAGUE_PROTOCOL
                        or existing.get("content_sha256") != content_hash
                        or policy_content_sha256(existing.get("model", {}),
                                                 existing.get("norm")) != content_hash):
                    raise RuntimeError("orphaned content-addressed policy failed validation")
            else:
                temp = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, temp)
                os.replace(temp, path)
            file_name, file_hash = path.name, _sha256(path)
            duplicate_of = None
        else:
            path = self.policy_dir / duplicate["file"]
            if not path.is_file() or _sha256(path) != duplicate["sha256"]:
                raise RuntimeError("deduplicated archive source policy changed")
            file_name, file_hash = duplicate["file"], duplicate["sha256"]
            duplicate_of = int(duplicate["id"])
        record = {
            "id": identity, "file": file_name, "sha256": file_hash,
            "size_bytes": int(path.stat().st_size),
            "content_sha256": content_hash,
            "duplicate_of_archive_id": duplicate_of,
            "policy_protocol": LEAGUE_PROTOCOL,
            "kind": str(kind), "iteration": int(iteration), "profile": str(profile),
            "admitted": bool(admitted), "payoff_eligible": bool(payoff_eligible),
            "current_score": 0.5, "past_best": 0.5, "regression": 0.0,
            # P2 fix (audit 2026-09-02, A7): "current_score" is written by two
            # different measurements -- the pool entry's live training EMA
            # (_sync_active_scores_to_archive) and, when a real evaluator edge
            # against the current milestone exists, a clean paired-evaluation
            # score (refresh_meta) that then overwrites it. Both feed the same
            # hard/near/variance core-selection ordering with no way to tell
            # which produced a given value. Track the two measurements and
            # provenance separately; "current_score" keeps its existing
            # meaning ("best available estimate") so no selection-logic
            # consumer changes behavior.
            "online_ema_score": None, "clean_paired_score": None,
            "current_score_source": "default",
            "nash_mass": 0.0, "metrics": dict(metrics or {}),
            "creation_wallclock": datetime.now(timezone.utc).isoformat(),
            "model_schema_hash": model_schema_sha256(cpu_model),
            "observation_schema_hash": "obs214_official_v1",
            "action_schema_hash": "factorized_categorical_4x21_v1",
            "normalizer_hash": policy_content_sha256({}, norm_state),
            "parent_archive_ids": [], "lineage_id": None,
            "behavior_descriptor_version": None, "behavior_descriptor": None,
            "payoff_fingerprint_version": None,
            "safety_status": "valid",
            "admission_status": "admitted" if admitted else "heldout",
        }
        self.records[identity] = record
        self.persist()
        return identity

    def load_policy(self, identity: int):
        identity = int(identity)
        record = self.records[identity]
        path = self.policy_dir / record["file"]
        if not path.exists() or _sha256(path) != record["sha256"]:
            raise FloatingPointError(f"archive policy missing or changed: {path}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        protocol = str(record.get("policy_protocol", value.get("protocol", "")))
        if protocol not in POLICY_PROTOCOLS or value.get("protocol") != protocol:
            raise ValueError(f"archive policy contract mismatch: {path}")
        if protocol == LEGACY_LEAGUE_PROTOCOL and value.get("archive_id") != identity:
            # A migrated v5 record always owns its original identity. v6
            # content-addressed records may intentionally share a weight file.
            raise ValueError(f"legacy archive identity mismatch: {path}")
        if (protocol == LEAGUE_PROTOCOL
                and value.get("content_sha256") != record.get("content_sha256")):
            raise ValueError(f"vNext archive content identity mismatch: {path}")
        return value

    def set_payoff(self, left: int, right: int, score: float, games: int) -> None:
        left, right, score, games = int(left), int(right), float(score), int(games)
        if left not in self.records or right not in self.records:
            raise KeyError("payoff policy is not archived")
        if not (math.isfinite(score) and 0.0 <= score <= 1.0 and games > 0):
            raise ValueError("invalid payoff observation")
        # Compatibility helper for synthetic/unit-test payoffs.  Production
        # evaluator observations use accumulate_payoff() with explicit W/D/L.
        wins = score * games
        losses = games - wins
        self._store_payoff(left, right, wins=wins, draws=0.0, losses=losses,
                           blocks=[{"seed_block": None, "wins": wins,
                                    "draws": 0.0, "losses": losses}])

    @staticmethod
    def _entry_counts(entry):
        if entry is None:
            return 0.0, 0.0, 0.0, []
        if all(name in entry for name in ("wins", "draws", "losses")):
            return (float(entry["wins"]), float(entry["draws"]), float(entry["losses"]),
                    list(entry.get("blocks", [])))
        games = float(entry.get("games", 0.0))
        score = float(entry.get("score", 0.5))
        return score * games, 0.0, (1.0 - score) * games, []

    def _store_payoff(self, left, right, *, wins, draws, losses, blocks):
        games = float(wins + draws + losses)
        score = float((wins + 0.5 * draws) / games)
        paired_complete = bool(blocks) and all("paired_scores" in block for block in blocks)
        paired_scores = ([float(value) for block in blocks
                          for value in block["paired_scores"]]
                         if paired_complete else [])
        interval_seed = int(left) * 1_000_003 + int(right) * 97_409 + int(games)
        if paired_scores:
            low, high = score_block_interval(paired_scores, seed=interval_seed)
        else:
            low, high = multinomial_score_interval(
                wins, draws, losses, seed=interval_seed)
        forward = {"score": score, "games": games, "wins": float(wins),
                   "draws": float(draws), "losses": float(losses),
                   "lcb95": low, "ucb95": high, "blocks": list(blocks)}
        if paired_scores:
            forward["paired_blocks"] = len(paired_scores)
        reverse = {"score": 1.0 - score, "games": games, "wins": float(losses),
                   "draws": float(draws), "losses": float(wins),
                   "lcb95": 1.0 - high, "ucb95": 1.0 - low,
                "blocks": [{
                        "seed_block": block.get("seed_block"),
                        "wins": block.get("losses", 0.0),
                        "draws": block.get("draws", 0.0),
                        "losses": block.get("wins", 0.0),
                        # 2026-09-04 FIX (P1): carry the evaluation provenance
                        # through the mirror. Omitting it made legacy
                        # protocol/bank slices silently miss the reverse
                        # perspective (forward blocks keep them verbatim).
                        # Only keys present on the source block are copied, so
                        # synthetic blocks without provenance stay unchanged.
                        **{key: block[key] for key in (
                            "evaluator_protocol", "scenario_bank_version",
                            "evidence_phase") if key in block},
                       "left_alt_loss_rate": (block.get("right_alt_loss_rate")
                                              if block.get("altitude_loss_measured") is True
                                              else None),
                       "right_alt_loss_rate": (block.get("left_alt_loss_rate")
                                               if block.get("altitude_loss_measured") is True
                                               else None),
                       "left_alt_loss_games": (block.get("right_alt_loss_games")
                                               if block.get("altitude_loss_measured") is True
                                               else None),
                       "right_alt_loss_games": (block.get("left_alt_loss_games")
                                                if block.get("altitude_loss_measured") is True
                                                else None),
                       "altitude_loss_measured": bool(
                           block.get("altitude_loss_measured") is True),
                       **({"paired_blocks": block.get("paired_blocks"),
                           "paired_scores": [1.0 - float(value)
                                             for value in block["paired_scores"]]}
                          if "paired_scores" in block else {}),
                   } for block in blocks]}
        if paired_scores:
            reverse["paired_blocks"] = len(paired_scores)
        self.payoff.setdefault(str(left), {})[str(right)] = forward
        self.payoff.setdefault(str(right), {})[str(left)] = reverse
        for identity in (left, right):
            self.payoff.setdefault(str(identity), {})[str(identity)] = {
                "score": 0.5, "games": games, "wins": 0.0, "draws": games,
                "losses": 0.0, "lcb95": 0.5, "ucb95": 0.5, "blocks": []}

    def accumulate_payoff(self, left: int, right: int, *, wins: float, draws: float,
                          losses: float, seed_block=None, lcb95=None, ucb95=None,
                          paired_blocks=None, paired_scores=None,
                          metadata: dict | None = None) -> None:
        left, right = int(left), int(right)
        values = np.asarray([wins, draws, losses], dtype=np.float64)
        if left not in self.records or right not in self.records:
            raise KeyError("payoff policy is not archived")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0) or values.sum() <= 0.0:
            raise ValueError("invalid payoff W/D/L observation")
        old_w, old_d, old_l, blocks = self._entry_counts(
            self.payoff.get(str(left), {}).get(str(right)))
        # 2026-09-04 FIX (P2): upsert by seed_block. Retrying an already
        # committed seed (e.g. resume after a crash between evaluation and
        # checkpoint) used to append a second block and double-count the edge,
        # while the graph path raises on duplicates. Replacing keeps a forced
        # refresh working and a retry idempotent. seed_block=None (synthetic)
        # keeps the old append behaviour.
        if seed_block is not None:
            _seed = int(seed_block)
            _kept = []
            for _b in blocks:
                if _b.get("seed_block") == _seed:
                    old_w -= float(_b.get("wins", 0.0))
                    old_d -= float(_b.get("draws", 0.0))
                    old_l -= float(_b.get("losses", 0.0))
                else:
                    _kept.append(_b)
            blocks = _kept
        block = {"seed_block": None if seed_block is None else int(seed_block),
                 "wins": float(wins), "draws": float(draws), "losses": float(losses)}
        if lcb95 is not None and ucb95 is not None:
            block.update(lcb95=float(lcb95), ucb95=float(ucb95))
        if paired_blocks is not None:
            block["paired_blocks"] = int(paired_blocks)
        if paired_scores is not None:
            paired_values = np.asarray(paired_scores, dtype=np.float64).reshape(-1)
            if (paired_values.size == 0 or np.any(~np.isfinite(paired_values))
                    or np.any((paired_values < 0.0) | (paired_values > 1.0))):
                raise ValueError("invalid paired payoff block scores")
            if paired_blocks is not None and paired_values.size != int(paired_blocks):
                raise ValueError("paired payoff block count mismatch")
            block["paired_scores"] = paired_values.tolist()
        if metadata:
            allowed = {"left_alt_loss_rate", "right_alt_loss_rate",
                       "left_alt_loss_games", "right_alt_loss_games",
                       "altitude_loss_measured", "evaluator_protocol",
                       "scenario_bank_version", "evidence_phase"}
            unknown = set(metadata) - allowed
            if unknown:
                raise ValueError(f"unsupported payoff metadata fields: {sorted(unknown)}")
            block.update(dict(metadata))
        blocks.append(block)
        self._store_payoff(left, right, wins=old_w + float(wins),
                           draws=old_d + float(draws), losses=old_l + float(losses),
                           blocks=blocks)

    def payoff_uncertainty(self, left: int, right: int) -> float:
        entry = self.payoff.get(str(int(left)), {}).get(str(int(right)))
        if entry is None:
            return 1.0
        if "lcb95" in entry and "ucb95" in entry:
            return max(0.0, float(entry["ucb95"]) - float(entry["lcb95"]))
        wins, draws, losses, _ = self._entry_counts(entry)
        low, high = multinomial_score_interval(wins, draws, losses,
                                                seed=int(left) * 1009 + int(right))
        return high - low

    def strategic_ids(self, include_heldout=False) -> list[int]:
        return [identity for identity, r in sorted(self.records.items())
                if r["payoff_eligible"] and (r["admitted"] or include_heldout)]

    def refresh_meta(self, current_id: int | None = None) -> dict[int, float]:
        ids = self.strategic_ids()
        nash = empirical_nash(ids, self.payoff)
        for identity, record in self.records.items():
            record["nash_mass"] = float(nash.get(identity, 0.0))
            if current_id is not None:
                item = self.payoff.get(str(current_id), {}).get(str(identity))
                if item is not None:
                    q = float(item["score"])
                    record["current_score"] = q
                    record["clean_paired_score"] = q
                    record["current_score_source"] = "clean_paired"
                    record["past_best"] = max(float(record.get("past_best", 0.5)), q)
                    record["regression"] = max(0.0, record["past_best"] - q)
        self.persist()
        return nash

    def _fingerprint(self, identity: int, anchors: list[int]) -> np.ndarray:
        row = self.payoff.get(str(identity), {})
        return np.asarray([float(row.get(str(a), {}).get("score", 0.5)) for a in anchors])

    def select_core(self, current_id: int | None, *, limit=16, exclude=()) -> tuple[list[int], list[int]]:
        excluded = {int(x) for x in exclude}
        candidates = [x for x in self.strategic_ids() if x not in excluded and x != current_id]
        if not candidates:
            return [], []
        anchors = self.strategic_ids()
        rows = {x: self.records[x] for x in candidates}
        chosen: list[int] = []

        def add(order, count):
            for identity in order:
                if identity not in chosen:
                    chosen.append(identity)
                if len(chosen) >= min(int(limit), count):
                    break

        # Quotas are cumulative: 4 Nash, then 6 hard, 3 near-peer, 1 forgotten.
        # 2026-09-02 (user decision): hard raised 4->6 to weight the core
        # toward opponents main currently struggles against (this is also
        # where a strong exploiter naturally lands, via current_score).
        # forgotten trimmed 3->1 to compensate -- it overlaps in role with
        # hard/Nash (an old, under-tested policy that is now easy again is
        # rarely also a hard/high-nash-mass one) -- so the loss there is
        # cheaper than losing the farthest-fingerprint coverage pass below.
        # Quotas now sum to 4+6+3+1=14, leaving the coverage pass its usual
        # 2 guaranteed diversity slots to fill core up to limit=16.
        add(sorted(candidates, key=lambda x: rows[x].get("nash_mass", 0.0), reverse=True), 4)
        hard = sorted(candidates, key=lambda x: rows[x].get("current_score", 0.5))
        target = min(limit, len(chosen) + 6)
        for x in hard:
            if x not in chosen:
                chosen.append(x)
            if len(chosen) >= target:
                break
        near = [x for x in candidates if 0.3 <= rows[x].get("current_score", 0.5) <= 0.7]
        near.sort(key=lambda x: abs(rows[x].get("current_score", 0.5) - 0.5))
        target = min(limit, len(chosen) + 3)
        for x in near:
            if x not in chosen:
                chosen.append(x)
            if len(chosen) >= target:
                break
        forgotten = sorted(candidates, key=lambda x: rows[x].get("regression", 0.0), reverse=True)
        target = min(limit, len(chosen) + 1)
        for x in forgotten:
            if x not in chosen:
                chosen.append(x)
            if len(chosen) >= target:
                break

        # Two farthest payoff fingerprints provide response-space coverage.
        coverage: list[int] = []
        while len(coverage) < 2:
            remaining = [x for x in candidates if x not in chosen and x not in coverage]
            if not remaining:
                break
            basis = chosen + coverage
            if not basis:
                pick = remaining[0]
            else:
                pick = max(remaining, key=lambda x: min(
                    float(np.linalg.norm(self._fingerprint(x, anchors) - self._fingerprint(y, anchors)))
                    for y in basis))
            coverage.append(pick)
            if len(chosen) < limit:
                chosen.append(pick)

        fill = sorted(candidates, key=lambda x: (
            rows[x].get("nash_mass", 0.0) + rows[x].get("regression", 0.0)
            + 1.0 - rows[x].get("current_score", 0.5)), reverse=True)
        for x in fill:
            if len(chosen) >= limit:
                break
            if x not in chosen:
                chosen.append(x)
        return chosen[:limit], [x for x in coverage if x in chosen[:limit]]

    def state_dict(self):
        return {"protocol": LEAGUE_PROTOCOL, "root": str(self.root), "next_id": self.next_id,
                "records": [self.records[x] for x in sorted(self.records)], "payoff": self.payoff}

    def load_state_dict(self, state) -> None:
        if state.get("protocol") != LEAGUE_PROTOCOL:
            raise ValueError("league archive protocol mismatch")
        self.next_id = int(state["next_id"])
        self.records = {int(r["id"]): dict(r) for r in state.get("records", [])}
        self.payoff = dict(state.get("payoff", {}))
        if self.records and self.next_id <= max(self.records):
            raise ValueError("archive next_id would reuse an existing identity")
        # Migration performs a full cryptographic audit. Normal restart checks
        # the materialized manifest and file sizes here, then verifies SHA and
        # payload lazily when a policy enters solver/active use. This keeps
        # recovery bounded as the immutable cold ledger grows.
        for record in self.records.values():
            path = self.policy_dir / record["file"]
            if (not path.is_file() or (record.get("size_bytes") is not None
                    and int(path.stat().st_size) != int(record["size_bytes"]))):
                raise FloatingPointError(f"archive policy missing or truncated: {path}")
        # A process may fail after atomically writing a policy but before the
        # next recovery checkpoint.  Preserve such files as cold orphans and
        # never reuse/overwrite their identities.
        disk_ids = []
        for path in self.policy_dir.glob("policy_*.pt"):
            try:
                disk_ids.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        if disk_ids:
            self.next_id = max(self.next_id, max(disk_ids) + 1)

    def persist(self) -> None:
        _atomic_json(self.root / "archive_manifest.json", self.state_dict())


__all__ = ["LEGACY_LEAGUE_PROTOCOL", "LEAGUE_PROTOCOL", "POLICY_PROTOCOLS",
           "ROLE_MIXTURE", "ACTIVE_LAYOUT", "LeagueArchive",
           "count_aware_alpha", "wilson_lower", "multinomial_score_interval",
           "paired_score_interval", "score_block_interval",
           "role_mixture", "payoff_posterior_mean", "empirical_nash",
           "policy_content_sha256", "model_schema_sha256"]
