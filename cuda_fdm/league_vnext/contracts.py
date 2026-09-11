"""Validated configuration and staged-release contracts for 100K+ training.

The values in this module are release contracts, not convenient defaults.  In
particular, the strategic population is bounded independently from the cold
archive and payoff stopping uses a fixed screening/confirmatory design.  A
future confidence-sequence implementation must use a new protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
import math


class VNextMode(StrEnum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    STAGED = "staged"


class VNextStage(IntEnum):
    PREPARED = 0
    SPARSE_LEAGUE = 1
    PERSISTENT_LINEAGES = 2
    ADAPTIVE_CURRICULUM = 3
    FROZEN_FINAL = 4


@dataclass(frozen=True)
class StrategicIndexConfig:
    # The authoritative SPEC requires a measured 48/64/96 sweep.  Sixty-four
    # is only the prepared shadow candidate; it is not silently promoted to a
    # calibrated production value.
    soft_budget: int = 64
    minimum_budget: int = 48
    maximum_budget: int = 96
    recent_champions: int = 8
    nash_support: int = 16
    hard_opponents: int = 16
    regression_sentinels: int = 12
    behavior_representatives: int = 16
    payoff_representatives: int = 16
    redteam_representatives: int = 8
    scenario_representatives: int = 8
    embedding_distance_epsilon: float = 0.02
    payoff_distance_epsilon: float = 0.03
    behavior_distance_epsilon: float = 0.05
    recent_protection_main_iters: int = 2000
    cold_audit_per_epoch: int = 2

    def validate(self) -> None:
        if not (1 <= self.minimum_budget <= self.soft_budget <= self.maximum_budget):
            raise ValueError("strategic-index budgets must be ordered and positive")
        if self.maximum_budget > 96:
            raise ValueError("authoritative SPEC caps the provisional sweep at 96")
        for name, value in asdict(self).items():
            if name.endswith("epsilon"):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
            elif not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class PayoffGraphConfig:
    screening_paired_blocks: int = 16
    confirmatory_paired_blocks: int = 128
    # Preserve the current clean-evaluator resolution: 64 mirrored pairs are
    # 128 complete games.  Adaptive 32->64->128 peeking remains forbidden.
    solver_paired_blocks: int = 64
    maximum_paired_blocks: int = 128
    confidence: float = 0.95
    # A release may lower this after p95 measurement. Four prevents the
    # prepared implementation from recreating a full 24-policy row by default.
    query_budget_per_milestone: int = 4
    stale_after_iterations: int = 2000
    fingerprint_neighbors: int = 4
    behavior_neighbors: int = 4
    fixed_two_stage: bool = True
    adaptive_early_stop: bool = False
    screening_point_score_min: float = 0.50
    evaluator_protocol: str = "step_only_full_episode_stratified_mirrored_stochastic_v3"
    screening_scenario_bank: str = "screening_v1"
    confirmatory_scenario_bank: str = "confirmatory_v1"
    solver_scenario_bank: str = "solver_v1"
    # The migrated v16 payoff used the same clean evaluator under the legacy
    # bank name. Only these explicitly compatible banks may complete Nash.
    solver_compatible_scenario_banks: tuple[str, ...] = (
        "solver_v1", "legacy_clean_bank_v1")

    def validate(self) -> None:
        if not (1 <= self.screening_paired_blocks <= self.maximum_paired_blocks):
            raise ValueError("invalid screening payoff block limits")
        if not (1 <= self.solver_paired_blocks <= self.maximum_paired_blocks):
            raise ValueError("invalid solver payoff block limits")
        if not (1 <= self.confirmatory_paired_blocks <= self.maximum_paired_blocks):
            raise ValueError("invalid confirmatory payoff block limits")
        if self.query_budget_per_milestone < 1:
            raise ValueError("payoff query budget must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("payoff confidence must be in (0, 1)")
        if not 0.0 <= self.screening_point_score_min <= 1.0:
            raise ValueError("screening point-score threshold must be in [0, 1]")
        if self.stale_after_iterations < 1:
            raise ValueError("payoff staleness horizon must be positive")
        if self.fixed_two_stage is not True or self.adaptive_early_stop:
            raise ValueError("v2 mainline requires fixed screening and fresh confirmatory seeds")
        if self.screening_scenario_bank == self.confirmatory_scenario_bank:
            raise ValueError("screening and confirmatory scenario banks must be distinct")
        if len({self.screening_scenario_bank, self.confirmatory_scenario_bank,
                self.solver_scenario_bank}) != 3:
            raise ValueError("screening, confirmatory and solver banks must be distinct")
        if not self.evaluator_protocol:
            raise ValueError("payoff evaluator protocol is required")
        compatible = tuple(map(str, self.solver_compatible_scenario_banks))
        if (not compatible or len(compatible) != len(set(compatible))
                or self.solver_scenario_bank not in compatible):
            raise ValueError("solver-compatible scenario banks are invalid")


@dataclass(frozen=True)
class ActiveGameConfig:
    """Small complete empirical game plus bounded historical regression audit."""

    # latest/current + 16 core + 3 challengers. Recent snapshots remain a
    # curriculum ring and never enter the Nash game merely because they are
    # GPU-resident.
    solver_policy_cap: int = 20
    # Normal milestones need at most the new current row (19 edges). One new
    # challenger row also fits; further challengers stay pending for a later
    # milestone instead of creating an unbounded pause.
    completion_edge_cap: int = 48
    # A challenger must actually be sampled by completed rollout episodes
    # before probation can finish and it can compete for a core seat.
    challenger_min_exposure_games: int = 32
    # Head-on 2026-09-09: eight audits per 500-iter milestone, 1:1:6 ratio.
    # Active/solver caps and recovery thresholds are unchanged; overlap and
    # empty categories backfill from the same oldest queue.
    cold_cycle_quota: int = 1
    cold_regression_quota: int = 1
    cold_stale_quota: int = 6
    cold_rotation_quota: int = 0
    # The complete 95% interval must keep current Main at or below this score.
    # 2026-09-04 user decision: raised 0.50 -> 0.65. Waiting until the interval
    # falls below a draw reacts too late to useful forgetting. A 0.65 ceiling
    # returns a moderately competitive historical opponent sooner, but only as
    # a probationary challenger; 32 live exposure games, a complete solver row
    # and Nash ranking are still required before it can take a core seat. The
    # 0.30 gap to the 0.95 stale-retirement bar prevents threshold oscillation.
    historical_counter_main_ucb_max: float = 0.65

    @property
    def cold_audit_total(self) -> int:
        return (self.cold_cycle_quota + self.cold_regression_quota
                + self.cold_stale_quota + self.cold_rotation_quota)

    def validate(self) -> None:
        if self.solver_policy_cap != 20:
            raise ValueError("solver game must remain current + core16 + challenger3")
        if not 1 <= self.completion_edge_cap <= 64:
            raise ValueError("active-game completion edge cap must be in 1..64")
        if self.challenger_min_exposure_games < 1:
            raise ValueError("challenger exposure gate must be positive")
        for name in ("cold_cycle_quota", "cold_regression_quota",
                     "cold_stale_quota", "cold_rotation_quota"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 1 <= self.cold_audit_total <= 16:
            raise ValueError("structured cold audit must contain 1..16 policies")
        if not 0.0 < self.historical_counter_main_ucb_max <= 0.65:
            raise ValueError("historical-counter gate must not exceed 0.65")


@dataclass(frozen=True)
class ChampionConfig:
    primary_noninferiority_margin: float = 0.0
    safety_noninferiority_margin: float = 0.0
    worst_cluster_noninferiority_margin: float = 0.0
    minimum_robustness_improvement: float = 0.0
    minimum_paired_blocks: int = 128
    frontier_cap: int = 5

    def validate(self) -> None:
        for name in ("primary_noninferiority_margin", "safety_noninferiority_margin",
                     "worst_cluster_noninferiority_margin",
                     "minimum_robustness_improvement"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_paired_blocks < 1:
            raise ValueError("minimum_paired_blocks must be positive")
        if not 1 <= self.frontier_cap <= 8:
            raise ValueError("champion frontier cap must remain in the SPEC shadow range 1..8")


@dataclass(frozen=True)
class HealthConfig:
    observe_only: bool = True
    log_period: int = 20
    review_period: int = 5000
    maximum_policy_kl: float = 0.08
    maximum_clip_fraction: float = 0.60
    minimum_support_retention: float = 0.20
    maximum_policy_lag: int = 40
    minimum_fresh_fraction: float = 0.95
    # P1 fix (audit 2026-09-02, B4): milestones can silently run for a very
    # long time without admitting a single new challenger -- e.g. if the
    # staged evaluation path stalls, or every candidate keeps failing the
    # same gate. Nothing previously surfaced that as a distinguishable
    # condition from "the league is healthy and stable"; only a raw decision
    # log scan would show it. 6 consecutive milestones (3,000 iterations at
    # the default 500-iteration cadence) without any admission raises a
    # visible alert instead.
    maximum_milestones_without_admission: int = 6

    def validate(self) -> None:
        if self.observe_only is not True:
            raise ValueError("automatic PPO intervention is not approved for the first vNext stage")
        if self.log_period < 1 or self.review_period < 1:
            raise ValueError("health log/review periods must be positive")
        if self.maximum_milestones_without_admission < 1:
            raise ValueError("maximum_milestones_without_admission must be positive")
        for name in ("maximum_policy_kl", "maximum_clip_fraction",
                     "minimum_support_retention", "minimum_fresh_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class VNextConfig:
    protocol: str = "active_league_100k_vnext_control_v2"
    mode: VNextMode = VNextMode.SHADOW
    stage: VNextStage = VNextStage.PREPARED
    source_iteration: int = 20000
    target_iteration: int = 100000
    active_cap: int = 24
    maximum_gpu_learners: int = 1
    immutable_cold_archive: bool = True
    frozen_holdout: bool = True
    decision_freeze_on_anomaly: bool = True
    strategic_index: StrategicIndexConfig = field(default_factory=StrategicIndexConfig)
    payoff_graph: PayoffGraphConfig = field(default_factory=PayoffGraphConfig)
    active_game: ActiveGameConfig = field(default_factory=ActiveGameConfig)
    champion: ChampionConfig = field(default_factory=ChampionConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    def validate(self) -> None:
        if self.protocol != "active_league_100k_vnext_control_v2":
            raise ValueError("unexpected vNext protocol")
        # 2026-09-03 (user decision): this 100K package was originally built
        # to migrate a trained final-20000 checkpoint forward. The user has
        # since decided 100K trains from scratch instead -- 20K was a
        # separate problem-finding testbed, not a seed for 100K's weights.
        # source_iteration=0 is the legitimate "no pre-run archive baseline"
        # value for that case: shadow.py seeds its iteration cursor from it,
        # and active_game.py's historical-counter audit treats records at or
        # before it as pre-migration legacy entries -- for a fresh run there
        # is no such legacy archive, so 0 correctly yields none. 20000
        # remains valid for anyone who still wants the original migration
        # path.
        if (self.source_iteration not in (0, 20000)
                or self.target_iteration < 1
                or (self.source_iteration == 20000 and self.target_iteration < 100000)):
            raise ValueError(
                "vNext must either start fresh with a positive target (source_iteration=0) or "
                "migrate from final-20000, toward at least 100K")
        if self.active_cap != 24:
            raise ValueError("active GPU league cap must remain 24 until an ablation approves a change")
        if self.maximum_gpu_learners != 1:
            raise ValueError("only one GPU learner may run at a time")
        if not self.immutable_cold_archive or not self.frozen_holdout:
            raise ValueError("cold archive and holdout must remain protected")
        if self.mode == VNextMode.STAGED and self.stage == VNextStage.PREPARED:
            raise ValueError("prepared stage cannot make live decisions")
        self.strategic_index.validate()
        self.payoff_graph.validate()
        self.active_game.validate()
        self.champion.validate()
        self.health.validate()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["stage"] = int(self.stage)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "VNextConfig":
        data = dict(value)
        data["mode"] = VNextMode(data.get("mode", VNextMode.SHADOW))
        data["stage"] = VNextStage(int(data.get("stage", VNextStage.PREPARED)))
        for key, kind in (("strategic_index", StrategicIndexConfig),
                          ("payoff_graph", PayoffGraphConfig),
                          ("active_game", ActiveGameConfig),
                          ("champion", ChampionConfig),
                          ("health", HealthConfig)):
            if isinstance(data.get(key), dict):
                if key == "payoff_graph" and isinstance(
                        data[key].get("solver_compatible_scenario_banks"), list):
                    data[key]["solver_compatible_scenario_banks"] = tuple(
                        data[key]["solver_compatible_scenario_banks"])
                data[key] = kind(**data[key])
        result = cls(**data)
        result.validate()
        return result
