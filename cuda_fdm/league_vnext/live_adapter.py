"""Release-gated P1 adapter for bounded sparse league mutations.

The adapter is intentionally limited to the P1 sparse-solver stage. Persistent
GPU lineage execution remains a separate P2 release gate, while altitude
red-team candidates are durably queued for that lineage instead of being
discarded or placed directly into the active roster.
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace

from .active_game import (
    HISTORICAL_AUDIT_PROTOCOL, select_historical_audits,
    strategic_active_ids)
from .active_roster import (
    LAYOUT, ActiveRosterSelector, ordered_probationary_ids,
    rank_roster_candidates)
from .contracts import VNextMode, VNextStage
from .payoff_graph import SOLVER_PHASES
from .profile_stats import EXPLOITER_ROLES


LIVE_ADAPTER_PROTOCOL = "active_league_100k_live_adapter_release_v2"
SUCCESSOR_ADMISSION_PROTOCOL = "role_profile_successor_admission_v1"
# Per-candidate evaluation events kept on the record. Bounded so a policy that
# is re-measured for tens of thousands of iterations cannot grow the archive
# without limit; the decisive event is pinned separately.
EVAL_HISTORY_LIMIT = 32
# 2026-09-04: a seat is only worth holding while the opponent still teaches
# Main something. Members enter core/challenger by beating the Main of their
# admission moment, but Main keeps improving and nothing removed them once it
# had: at iteration 6,900 of the live 3-9 run every one of the 7 core and 3
# challenger members sat at 0.916-0.996 win rate *for Main*, so 59.6% of
# Main's games were spent on opponents it already beat >=90% of the time.
# Eviction only ever happened by losing a rank contest, and with 7 of 16 core
# seats filled there was no contest to lose. Retire the ones that have gone
# genuinely dead, on an absolute bar rather than a relative one.
# 2026-09-04: lowered 0.98 -> 0.95. At 0.98 the bar only caught the completely
# dead members; the live roster showed a dense band at 0.95-0.98 that Main also
# had nothing left to learn from. Re-entry is the counterweight: a policy
# retired here can return after a historical audit meets Main UCB95 <= 0.65.
# 2026-09-05 user decision: graduate at Main EMA >= 90%, still requiring
# 2,000 games. Historical return remains the independent UCB95 <= 65% gate.
STALE_MEMBER_MAIN_WINRATE = 0.90
# Never judge on a thin sample: a freshly seated member starts at ema 0.5 and
# needs real exposure before its number means anything.
STALE_MEMBER_MIN_GAMES = 2000
# 2026-09-04: raised 1 -> 2. Live showed solved members queuing faster than
# eviction drains: most-solved-first ordering is right, but with challengers
# solved in ~500 iters each, one seat per 500-iter milestone lets 0.95+
# members (e.g. 56k-game 0.977) linger and dilute the hard channel with easy
# games. Two still drains gradually; promotion stays at one.
STALE_MEMBER_EVICTIONS_PER_MILESTONE = 2


def _seed_integer(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def _optional_finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class VNextMilestoneAdapter:
    """Executes a bounded proposal against the production clean evaluator."""

    def __init__(self, controller, *, release_receipt: dict):
        self.controller = controller
        receipt = dict(release_receipt)
        if (receipt.get("protocol") != LIVE_ADAPTER_PROTOCOL
                or receipt.get("approved") is not True):
            raise RuntimeError("live sparse adapter has no valid release receipt")
        if (controller.config.mode != VNextMode.STAGED
                or controller.config.stage != VNextStage.SPARSE_LEAGUE):
            raise RuntimeError("this adapter is released only for staged P1 sparse league")
        if int(receipt.get("stage", -1)) != int(controller.config.stage):
            raise RuntimeError("live adapter receipt stage mismatch")
        self.release_receipt = receipt

    @staticmethod
    def _candidate_interval(result: dict, *, candidate_id: int,
                            left_id: int) -> tuple[float, float, float, float | None]:
        if int(candidate_id) == int(left_id):
            score = float(result["score"])
            lcb = float(result["lcb95"])
            ucb = float(result["ucb95"])
            target_alt = _optional_finite(result.get("right_alt_loss_rate"))
        else:
            score = 1.0 - float(result["score"])
            lcb = 1.0 - float(result["ucb95"])
            ucb = 1.0 - float(result["lcb95"])
            target_alt = _optional_finite(result.get("left_alt_loss_rate"))
        return score, lcb, ucb, target_alt

    @staticmethod
    def _candidate_perspective(result: dict, *, candidate_id: int,
                               left_id: int) -> tuple[float, float, float | None]:
        score, lcb, _, target_alt = VNextMilestoneAdapter._candidate_interval(
            result, candidate_id=candidate_id, left_id=left_id)
        return score, lcb, target_alt

    @staticmethod
    def _strategy_bucket(record: dict) -> tuple[str, str] | None:
        metrics = record.get("metrics", {})
        role = metrics.get("role")
        if role not in EXPLOITER_ROLES:
            kind = str(record.get("kind", "")).lower()
            if "eie" in kind:
                role = "ME-EIE"
            elif "ere" in kind:
                role = "ME-ERE"
            elif kind.startswith("exploiter_le") or kind.endswith("_le"):
                role = "LE"
        if role not in EXPLOITER_ROLES:
            return None
        return str(role), str(record.get("profile", "standard"))

    @classmethod
    def _assign_successor_incumbents(cls, trainer, candidate_ids: list[int]) -> None:
        active = []
        for entry in trainer.pool.active_entries():
            if entry.get("role") not in {"core", "challenger"}:
                continue
            identity = entry.get("archive_id")
            record = trainer.archive.records.get(identity)
            if identity is None or record is None or record.get("admitted") is not True:
                continue
            active.append((int(identity), record))
        for candidate_id in candidate_ids:
            candidate = trainer.archive.records[candidate_id]
            bucket = cls._strategy_bucket(candidate)
            matches = [(identity, record) for identity, record in active
                       if bucket is not None and cls._strategy_bucket(record) == bucket
                       and identity != candidate_id]
            if not matches:
                candidate.pop("successor_incumbent_id", None)
                candidate.setdefault("metrics", {}).pop("successor_incumbent_id", None)
                continue
            # current_score is the main's score against the policy. Use the
            # strongest active incumbent only as a cheap pre-selection; both
            # policies are evaluated fresh against the same frozen main below.
            incumbent, _ = min(
                matches, key=lambda item: (
                    float(item[1].get("current_score", 0.5)),
                    -int(item[1].get("iteration", -1)), -item[0]))
            candidate["successor_incumbent_id"] = int(incumbent)
            candidate.setdefault("metrics", {})["successor_incumbent_id"] = int(incumbent)

    @staticmethod
    def _successor_evidence(metrics: dict, phase: str) -> dict:
        root = metrics.setdefault("successor_admission", {
            "protocol": SUCCESSOR_ADMISSION_PROTOCOL})
        return root.setdefault(str(phase), {})

    @classmethod
    def _successor_evaluation_context(cls, trainer, query: dict, *,
                                      candidate_ids: set[int], current_id: int,
                                      iteration: int) -> dict | None:
        """Return one shared IC seed for a role/profile successor comparison.

        The payoff transaction remains pair-specific.  Only the evaluator IC
        seed is shared, so a candidate and its incumbent face the same frozen
        main on the same scenario bank and paired initial conditions.
        """
        pair = {int(query["left"]), int(query["right"])}
        if int(current_id) not in pair:
            return None
        decision_candidate = next(
            (identity for identity in pair if identity in candidate_ids), None)
        if (decision_candidate is None
                and "successor_incumbent_baseline" in str(query.get("reason", ""))):
            incumbent_id = next(identity for identity in pair
                                if identity != int(current_id))
            matches = [
                int(candidate) for candidate in candidate_ids
                if int(trainer.archive.records.get(candidate, {}).get(
                    "successor_incumbent_id", -1)) == int(incumbent_id)
            ]
            decision_candidate = max(
                matches,
                key=lambda identity: (
                    int(trainer.archive.records.get(identity, {}).get(
                        "iteration", -1)), identity),
                default=None)
        if decision_candidate is None:
            return None
        bucket = cls._strategy_bucket(
            trainer.archive.records.get(int(decision_candidate), {}))
        if bucket is None:
            return None
        role, profile = bucket
        seed_text = (
            f"successor:{query['phase']}:{query['scenario_bank_version']}:"
            f"{int(iteration)}:{role}:{profile}:main{int(current_id)}")
        return {
            "candidate_id": int(decision_candidate),
            "role": role,
            "profile": profile,
            "seed_text": seed_text,
            "seed_block": _seed_integer(seed_text),
        }

    def _resolve_successor_admission(self, trainer, candidate: int,
                                     phase: str) -> None:
        record = trainer.archive.records[candidate]
        metrics = record.setdefault("metrics", {})
        evidence = self._successor_evidence(metrics, phase)
        candidate_result = evidence.get("candidate")
        incumbent_result = evidence.get("incumbent")
        if not candidate_result or not incumbent_result:
            evidence["status"] = "awaiting_both_results"
            return
        if (candidate_result.get("current_id") != incumbent_result.get("current_id")
                or candidate_result.get("iteration") != incumbent_result.get("iteration")):
            evidence["status"] = "awaiting_same_main_results"
            return
        if (candidate_result.get("scenario_bank_version") is None
                or candidate_result.get("evaluation_seed_block") is None
                or (candidate_result.get("scenario_bank_version")
                    != incumbent_result.get("scenario_bank_version"))
                or (candidate_result.get("evaluation_seed_block")
                    != incumbent_result.get("evaluation_seed_block"))):
            evidence["status"] = "awaiting_same_scenario_seed_results"
            return
        incumbent_id = int(metrics["successor_incumbent_id"])
        if int(incumbent_result.get("archive_id", -1)) != incumbent_id:
            evidence["status"] = "incumbent_identity_mismatch"
            return
        if trainer.archive.records.get(incumbent_id) is None:
            evidence["status"] = "incumbent_missing"
            return
        if phase == "screening":
            passed = float(candidate_result["score"]) > float(incumbent_result["score"])
            evidence.update({"passed": bool(passed),
                             "status": "passed" if passed else "failed"})
            if passed and metrics.get("screening_status") != "passed":
                metrics["screening_status"] = "passed"
                record["screening_status"] = "passed"
            return
        if phase != "confirmatory" or metrics.get("direct_admission_passed") is True:
            return
        passed = float(candidate_result["lcb95"]) > float(incumbent_result["ucb95"])
        evidence.update({"passed": bool(passed),
                         "status": "passed" if passed else "failed"})
        if not passed:
            return
        record["admitted"] = True
        record["admission_status"] = "probationary"
        if str(record.get("kind", "")).startswith("heldout"):
            record["kind"] = "exploiter_probationary"
        metrics.update({
            "confirmatory_status": "passed_successor_admission",
            "successor_admission_passed": True,
            "successor_of_archive_id": incumbent_id,
        })
        self._sync_admission_history(trainer, candidate)

    def _queue_altitude_redteam(self, archive_id: int, *, iteration: int,
                                score: float, lcb: float,
                                target_alt_loss_rate: float) -> None:
        record = self.controller.archive_index.records.get(int(archive_id), {})
        evidence = {
            "candidate_score": float(score), "candidate_lcb95": float(lcb),
            "target_main_alt_loss_rate": float(target_alt_loss_rate),
            "source": "fresh_vnext_clean_paired_evaluator",
        }
        self.controller.lineages.queue_candidate(
            int(archive_id), role="LE", reason="heldout_altitude_redteam",
            iteration=int(iteration), evidence=evidence)
        # The materialized record is rebuilt later; keep this local update only
        # for the live-decision receipt emitted in the same milestone.
        record["persistent_lineage_candidate"] = True

    def _update_candidate_status(self, trainer, query: dict, result: dict, *,
                                 candidate_ids: set[int], current_id: int,
                                 iteration: int) -> None:
        candidate = next((identity for identity in (int(query["left"]), int(query["right"]))
                          if identity in candidate_ids), None)
        if candidate is None or current_id not in (int(query["left"]), int(query["right"])):
            return
        record = trainer.archive.records[candidate]
        if record.get("admitted") is True:
            # on_milestone() freezes candidate_id_set before running its
            # queries, so a policy admitted by an early confirmatory in this
            # same milestone is still in that set for every later query. Solver
            # traffic then overwrote the fresh_candidate_* fields that recorded
            # why it was admitted -- id=11 ended up displaying the *solver*
            # lcb (0.490, below the 0.50 bar) instead of the confirmatory lcb
            # (0.532) that actually passed it, which reads exactly like an
            # admission bug. Admission evidence is written once.
            return
        metrics = record.setdefault("metrics", {})
        score, lcb, ucb, target_alt = self._candidate_interval(
            result, candidate_id=candidate, left_id=int(query["left"]))
        measured = bool(result.get("altitude_loss_measured", False)) and target_alt is not None
        altitude_redteam = bool(
            measured and target_alt >= float(trainer.cfg.league_altitude_redteam_threshold))
        metrics.update({
            "fresh_candidate_score": score,
            "fresh_candidate_lcb95": lcb,
            "fresh_candidate_ucb95": ucb,
            "target_main_alt_loss_rate": target_alt if measured else None,
            "altitude_loss_measured": measured,
            "altitude_redteam": altitude_redteam,
        })
        evidence = self._successor_evidence(metrics, query["phase"])
        evidence["candidate"] = {
            "score": score, "lcb95": lcb, "ucb95": ucb,
            "current_id": int(current_id), "iteration": int(iteration),
            "scenario_bank_version": result.get("scenario_bank_version"),
            "evaluation_seed_block": result.get("evaluation_seed_block"),
        }
        # Append-only evidence. Every phase overwrote one shared set of
        # fresh_candidate_* fields, so which measurement actually drove a
        # decision was unrecoverable afterwards. Keep the decisive events.
        history = metrics.setdefault("eval_history", [])
        history.append({
            "phase": str(query["phase"]), "iteration": int(iteration),
            "opponent_id": int(current_id), "score": score,
            "lcb95": lcb, "ucb95": ucb,
            "scenario_bank_version": result.get("scenario_bank_version"),
            "evaluation_seed_block": result.get("evaluation_seed_block"),
        })
        del history[:-EVAL_HISTORY_LIMIT]
        if query["phase"] == "screening":
            passed = score >= self.controller.config.payoff_graph.screening_point_score_min
            metrics["screening_status"] = "passed" if passed else "failed"
            record["screening_status"] = metrics["screening_status"]
            if not passed and ucb < float(trainer.cfg.league_admission_score):
                # Even the optimistic end of this candidate's interval sits
                # below the point threshold confirmatory would demand, so no
                # confirmatory run against this Main could admit it. Without
                # this it stays in candidate_ids and its admission_target edge
                # -- worth +100 mandatory priority -- reserves query budget at
                # every remaining milestone, forever. Retirement is not
                # permanent: the historical audit still re-tests it and can
                # re-enrol it if Main later regresses.
                record["admission_status"] = "dormant_rejected"
                metrics["dormant_rejected_at_iteration"] = int(iteration)
                metrics["dormant_rejected_reason"] = "screening_ucb95_below_admission_score"
        elif query["phase"] == "confirmatory":
            passed = (score >= float(trainer.cfg.league_admission_score)
                      and lcb >= float(trainer.cfg.league_admission_lcb))
            metrics["direct_admission_passed"] = bool(passed)
            metrics["confirmatory_status"] = "passed" if passed else "failed"
            # Pin the exact measurement that decided this, so the record stays
            # auditable even as later evidence lands.
            metrics["admission_decision_eval"] = dict(history[-1])
            if passed:
                record["admitted"] = True
                record["admission_status"] = "probationary"
                if str(record.get("kind", "")).startswith("heldout"):
                    record["kind"] = "exploiter_probationary"
            else:
                record["admitted"] = False
                record["admission_status"] = (
                    "heldout_altitude_redteam" if altitude_redteam else "heldout")
        self._resolve_successor_admission(
            trainer, candidate, str(query["phase"]))
        self._sync_admission_history(trainer, candidate)

    def _update_successor_incumbent_status(self, trainer, query: dict, result: dict, *,
                                           candidate_ids: set[int], current_id: int,
                                           iteration: int) -> None:
        if ("successor_incumbent_baseline" not in str(query.get("reason", ""))
                or current_id not in (int(query["left"]), int(query["right"]))):
            return
        incumbent_id = (int(query["right"]) if int(query["left"]) == current_id
                        else int(query["left"]))
        score, lcb, ucb, _ = self._candidate_interval(
            result, candidate_id=incumbent_id, left_id=int(query["left"]))
        for candidate in candidate_ids:
            record = trainer.archive.records.get(candidate, {})
            if int(record.get("successor_incumbent_id", -1)) != incumbent_id:
                continue
            metrics = record.setdefault("metrics", {})
            evidence = self._successor_evidence(metrics, query["phase"])
            evidence["incumbent"] = {
                "archive_id": incumbent_id,
                "score": score, "lcb95": lcb, "ucb95": ucb,
                "current_id": int(current_id), "iteration": int(iteration),
                "scenario_bank_version": result.get("scenario_bank_version"),
                "evaluation_seed_block": result.get("evaluation_seed_block"),
            }
            self._resolve_successor_admission(
                trainer, candidate, str(query["phase"]))

    def _queue_pending_altitude_redteams(self, trainer, candidate_ids: set[int], *,
                                         iteration: int) -> None:
        for candidate in candidate_ids:
            record = trainer.archive.records.get(candidate, {})
            metrics = record.get("metrics", {})
            target_alt = metrics.get("target_main_alt_loss_rate")
            if (record.get("admitted") is True or metrics.get("altitude_redteam") is not True
                    or target_alt is None):
                continue
            record["kind"] = "heldout_altitude_redteam"
            metrics["persistent_lineage_candidate"] = True
            self._queue_altitude_redteam(
                candidate, iteration=iteration,
                score=float(metrics["fresh_candidate_score"]),
                lcb=float(metrics["fresh_candidate_lcb95"]),
                target_alt_loss_rate=float(target_alt))

    def _execute_query(self, trainer, query: dict, *, iteration: int,
                       candidate_ids: set[int], current_id: int) -> dict | None:
        graph = self.controller.graph
        seed_text = (f"{query['phase']}:{query['scenario_bank_version']}:"
                     f"{iteration}:{query['left']}:{query['right']}:"
                     f"{query.get('reason', 'unspecified')}")
        successor_context = self._successor_evaluation_context(
            trainer, query, candidate_ids=candidate_ids,
            current_id=current_id, iteration=iteration)
        evaluation_seed_block = (
            int(successor_context["seed_block"])
            if successor_context is not None else _seed_integer(seed_text))
        txid = graph.transaction_id(
            query["left"], query["right"], seed_text, phase=query["phase"],
            evaluator_protocol=self.controller.config.payoff_graph.evaluator_protocol,
            scenario_bank_version=query["scenario_bank_version"])
        if graph.transactions.get(txid, {}).get("state") == "committed":
            return None
        txid = graph.plan(
            query["left"], query["right"], seed_text,
            reason=query["reason"], iteration=iteration, phase=query["phase"],
            evaluator_protocol=self.controller.config.payoff_graph.evaluator_protocol,
            scenario_bank_version=query["scenario_bank_version"])
        graph.start(txid)
        result = trainer._evaluate_pair(
            trainer.archive.load_policy(query["left"]),
            trainer.archive.load_policy(query["right"]),
            int(query["minimum_blocks"]), paired=True,
            seed_block=evaluation_seed_block)
        result = dict(result)
        result["scenario_bank_version"] = str(query["scenario_bank_version"])
        result["evaluation_seed_block"] = int(evaluation_seed_block)
        if successor_context is not None:
            result["successor_decision_candidate_id"] = int(
                successor_context["candidate_id"])
        paired_scores = list(result["paired_scores"])
        game_ids = [f"{seed_text}:{index}" for index in range(len(paired_scores))]
        metadata = {
            "altitude_loss_measured": bool(result.get("altitude_loss_measured", False)),
            "left_alt_loss_rate": _optional_finite(result.get("left_alt_loss_rate")),
            "right_alt_loss_rate": _optional_finite(result.get("right_alt_loss_rate")),
            "left_alt_loss_games": result.get("left_alt_loss_games"),
            "right_alt_loss_games": result.get("right_alt_loss_games"),
        }
        graph.commit(
            txid, paired_scores, iteration=iteration, complete=True,
            paired_game_ids=game_ids, wins=result.get("wins"),
            draws=result.get("draws"), losses=result.get("losses"),
            metadata=metadata)
        if query["phase"] in SOLVER_PHASES:
            trainer._record_payoff_result(
                query["left"], query["right"], result,
                seed_block=evaluation_seed_block,
                evidence_phase=query["phase"],
                scenario_bank_version=query["scenario_bank_version"],
                evaluator_protocol=self.controller.config.payoff_graph.evaluator_protocol)
        self._update_candidate_status(
            trainer, query, result, candidate_ids=candidate_ids,
            current_id=current_id, iteration=iteration)
        self._update_successor_incumbent_status(
            trainer, query, result, candidate_ids=candidate_ids,
            current_id=current_id, iteration=iteration)
        return result

    def _complete_active_solver_game(self, trainer, *, policy_ids: list[int],
                                     iteration: int, current_id: int,
                                     candidate_ids: set[int]) -> list[dict]:
        """Fill only missing edges in the bounded active empirical game."""
        graph = self.controller.graph
        payoff = self.controller.config.payoff_graph
        active_game = self.controller.config.active_game
        missing = graph.missing_solver_pairs(
            policy_ids, minimum_blocks=payoff.solver_paired_blocks,
            evaluator_protocol=payoff.evaluator_protocol,
            scenario_bank_versions=payoff.solver_compatible_scenario_banks)
        if len(missing) > active_game.completion_edge_cap:
            raise RuntimeError(
                f"active solver completion needs {len(missing)} edges; bounded cap is "
                f"{active_game.completion_edge_cap}")
        executed = []
        for left, right in missing:
            observed = graph.estimate_solver_slice(
                left, right, evaluator_protocol=payoff.evaluator_protocol,
                scenario_bank_versions=payoff.solver_compatible_scenario_banks).paired_blocks
            query = {
                "left": int(left), "right": int(right),
                "priority": 0.0, "reason": "active_solver_completion",
                "minimum_blocks": max(1, payoff.solver_paired_blocks - observed),
                "maximum_blocks": payoff.maximum_paired_blocks,
                "phase": "solver",
                "scenario_bank_version": payoff.solver_scenario_bank,
            }
            self._execute_query(
                trainer, query, iteration=iteration,
                candidate_ids=candidate_ids, current_id=current_id)
            executed.append(query)
        graph.assert_complete_solver_subgame(
            policy_ids, minimum_blocks=payoff.solver_paired_blocks,
            evaluator_protocol=payoff.evaluator_protocol,
            scenario_bank_versions=payoff.solver_compatible_scenario_banks)
        return executed

    def _record_historical_audit(self, trainer, *, archive_id: int,
                                 current_id: int, iteration: int, reason: str,
                                 result: dict | None = None) -> bool:
        graph = self.controller.graph
        payoff = self.controller.config.payoff_graph
        estimate = graph.estimate_solver_slice(
            current_id, archive_id,
            evaluator_protocol=payoff.evaluator_protocol,
            scenario_bank_versions=(payoff.solver_scenario_bank,),
            iteration=iteration, phases={"solver"})
        if (not estimate.known
                or estimate.paired_blocks < payoff.solver_paired_blocks):
            raise RuntimeError("historical audit has no complete solver evidence")
        record = trainer.archive.records[int(archive_id)]
        metrics = record.setdefault("metrics", {})
        score = float(estimate.posterior_mean)
        record["current_score"] = score
        # A4-adjacent fix (audit 2026-09-02, A7 follow-up): this is a third
        # writer of "current_score" that the earlier A7 fix
        # (_sync_active_scores_to_archive / LeagueArchive.refresh_meta) did
        # not know about. estimate_solver_slice() here is a real solver-phase
        # evaluator measurement, so it is "clean_paired" evidence exactly like
        # refresh_meta()'s -- tag it the same way so provenance stays accurate
        # regardless of which of the three call sites last touched a record.
        record["clean_paired_score"] = score
        record["current_score_source"] = "clean_paired"
        record["past_best"] = max(float(record.get("past_best", 0.5)), score)
        record["regression"] = max(0.0, float(record["past_best"]) - score)
        metrics.update({
            "historical_audit_protocol": HISTORICAL_AUDIT_PROTOCOL,
            "last_historical_audit_iteration": int(iteration),
            "last_historical_audit_reason": str(reason),
            "last_historical_audit_current_id": int(current_id),
            "last_historical_audit_main_score": score,
            "last_historical_audit_main_lcb95": float(estimate.lcb),
            "last_historical_audit_main_ucb95": float(estimate.ucb),
            "last_historical_audit_paired_blocks": int(estimate.paired_blocks),
            "last_historical_audit_games": int(estimate.paired_games),
        })
        if result is not None:
            current_is_left = int(current_id) < int(archive_id)
            metrics.update({
                "last_historical_audit_altitude_loss_measured": bool(
                    result.get("altitude_loss_measured", False)),
                "last_historical_audit_main_alt_loss_rate": _optional_finite(
                    result.get("left_alt_loss_rate" if current_is_left
                               else "right_alt_loss_rate")),
                "last_historical_audit_counter_alt_loss_rate": _optional_finite(
                    result.get("right_alt_loss_rate" if current_is_left
                               else "left_alt_loss_rate")),
            })
        threshold = self.controller.config.active_game.historical_counter_main_ucb_max
        if float(estimate.ucb) > threshold:
            return False
        # Re-enrolment preserves the old policy bytes and archive identity. It
        # grants only a challenger probation slot; core membership still needs
        # a complete bounded solver row at a later milestone.
        record["admitted"] = True
        record["payoff_eligible"] = True
        record["admission_status"] = "probationary"
        metrics.update({
            "historical_counter_reactivated": True,
            "historical_counter_reactivated_at_iteration": int(iteration),
            "historical_counter_gate_main_ucb_max": float(threshold),
            "historical_counter_reactivation_count": int(metrics.get(
                "historical_counter_reactivation_count", 0)) + 1,
        })
        return True

    def _audit_historical_archive(self, trainer, *, iteration: int,
                                  current_id: int, active_ids: list[int],
                                  cycle_ids, candidate_ids: set[int]) -> tuple[list[dict], list[int]]:
        config = self.controller.config
        stop_state = getattr(trainer, "training_stop_state", None) or {}
        finish_plan = stop_state.get("config", {}).get("finish_plan", {})
        expanded = stop_state.get("phase") == "polish" and bool(finish_plan)
        audit_blocks = max(config.payoff_graph.solver_paired_blocks,
                           int(finish_plan.get("archive_audit_paired_blocks", 64))) if expanded else config.payoff_graph.solver_paired_blocks
        audit_config = config.active_game
        if expanded:
            # Pure oldest-first selection gives every valid archived policy a
            # turn before rechecking already covered policies. Pool residents
            # already receive online exposure and are excluded from cold audits.
            active_ids = list(set(active_ids) | {
                int(e["archive_id"]) for e in trainer.pool.active_entries()
                if e.get("archive_id") is not None})
            audit_config = replace(audit_config, cold_cycle_quota=0,
                                   cold_regression_quota=0,
                                   cold_stale_quota=finish_plan["archive_audit_cap"],
                                   cold_rotation_quota=0)
            all_targets, _ = select_historical_audits(
                trainer.archive.records, source_iteration=config.source_iteration,
                active_ids=active_ids, cursor=0,
                config=replace(audit_config, cold_stale_quota=len(trainer.archive.records)),
                include_all=True)
            sweep_start = stop_state["config"]["normal_end"]
            uncovered = sum(int((trainer.archive.records[t.archive_id].get("metrics") or {}).get(
                "last_historical_audit_iteration") or -1) < sweep_start for t in all_targets)
            if uncovered:
                audit_config = replace(audit_config, cold_stale_quota=min(
                    finish_plan["archive_audit_cap"], uncovered))
            else:
                risk_quota = finish_plan["archive_audit_cap"] // 4
                audit_config = replace(audit_config, cold_regression_quota=risk_quota,
                                       cold_stale_quota=finish_plan["archive_audit_cap"]-risk_quota)
        targets, next_cursor = select_historical_audits(
            trainer.archive.records, source_iteration=config.source_iteration,
            active_ids=active_ids, cycle_ids=cycle_ids,
            cursor=self.controller.historical_audit_cursor,
            config=audit_config, include_all=expanded)
        self.controller.historical_audit_cursor = int(next_cursor)
        queries, reactivated = [], []
        for target in targets:
            result = None
            estimate = self.controller.graph.estimate_solver_slice(
                current_id, target.archive_id,
                evaluator_protocol=config.payoff_graph.evaluator_protocol,
                scenario_bank_versions=(config.payoff_graph.solver_scenario_bank,),
                iteration=iteration, phases={"solver"})
            if estimate.paired_blocks < audit_blocks:
                left, right = sorted((int(current_id), int(target.archive_id)))
                query = {
                    "left": left, "right": right,
                    "priority": 0.0,
                    "reason": (f"historical_{target.reason}" +
                               (f"_blocks{audit_blocks}" if audit_blocks > config.payoff_graph.solver_paired_blocks else "")),
                    "minimum_blocks": max(
                        1, audit_blocks
                        - estimate.paired_blocks),
                    "maximum_blocks": max(config.payoff_graph.maximum_paired_blocks, audit_blocks),
                    "phase": "solver",
                    "scenario_bank_version": config.payoff_graph.solver_scenario_bank,
                }
                result = self._execute_query(
                    trainer, query, iteration=iteration,
                    candidate_ids=candidate_ids, current_id=current_id)
                queries.append(query)
            if audit_blocks > config.payoff_graph.solver_paired_blocks:
                verified = self.controller.graph.estimate_solver_slice(
                    current_id, target.archive_id,
                    evaluator_protocol=config.payoff_graph.evaluator_protocol,
                    scenario_bank_versions=(config.payoff_graph.solver_scenario_bank,),
                    iteration=iteration, phases={"solver"})
                if verified.paired_blocks < audit_blocks:
                    raise RuntimeError("finishing archive audit lacks required paired evidence")
            if self._record_historical_audit(
                    trainer, archive_id=target.archive_id,
                    current_id=current_id, iteration=iteration,
                    reason=target.reason, result=result):
                reactivated.append(int(target.archive_id))
        if expanded:
            self.controller.log.append("finishing_archive_audit", {
                "eligible_inactive": len(all_targets), "uncovered_before": uncovered,
                "uncovered_after": max(0, uncovered-len(targets)) if uncovered else 0,
                "selected": len(targets), "queries": len(queries),
                "reactivated_ids": reactivated, "cap": finish_plan["archive_audit_cap"],
                "required_paired_blocks": audit_blocks,
                "mode": "coverage" if uncovered else "risk_and_stale",
            }, iteration=iteration)
        return queries, reactivated

    @staticmethod
    def _ordered_probationary_challengers(trainer) -> list[int]:
        return ordered_probationary_ids(trainer.archive.records)

    def _select_solver_challengers(self, trainer, *, current_id: int,
                                   incumbent_entries,
                                   iteration: int | None = None) -> tuple[list[int], list[int]]:
        """Admit only challenger rows that fit the bounded completion pause."""
        config = self.controller.config
        payoff = config.payoff_graph
        accepted: list[int] = []
        pending: list[int] = []
        incumbent_ids = {
            int(entry["archive_id"]) for entry in incumbent_entries
            if entry.get("archive_id") is not None}
        incumbent_probationary = sum(
            trainer.archive.records.get(identity, {}).get("admission_status")
            == "probationary" for identity in incumbent_ids)
        available_slots = max(
            0, min(19 - len(incumbent_entries), 3 - incumbent_probationary))
        for identity in self._ordered_probationary_challengers(trainer):
            if int(identity) in incumbent_ids:
                continue
            if len(accepted) >= min(3, available_slots):
                pending.append(int(identity))
                continue
            entries = [*incumbent_entries, *(
                {"role": "challenger", "archive_id": value}
                for value in [*accepted, int(identity)])]
            ids = strategic_active_ids(
                entries, current_id=current_id,
                cap=config.active_game.solver_policy_cap)
            missing = self.controller.graph.missing_solver_pairs(
                ids, minimum_blocks=payoff.solver_paired_blocks,
                evaluator_protocol=payoff.evaluator_protocol,
                scenario_bank_versions=payoff.solver_compatible_scenario_banks)
            if len(missing) <= config.active_game.completion_edge_cap:
                accepted.append(int(identity))
            else:
                pending.append(int(identity))
        if iteration is not None:
            for identity in accepted:
                trainer.archive.records[identity].setdefault("metrics", {}).pop(
                    "solver_pending_since_iteration", None)
            for identity in pending:
                metrics = trainer.archive.records[identity].setdefault("metrics", {})
                metrics.setdefault("solver_pending_since_iteration", int(iteration))
                metrics.setdefault("solver_pending_reason", "solver_capacity_or_edge_budget")
                metrics["solver_status"] = "pending"
                metrics["active_role"] = None
        return accepted, pending

    def _promote_milestone_main_candidates(self, trainer, *, iteration: int,
                                           current_id: int, incumbent_entries,
                                           reserved_challenger_ids=(),
                                           limit: int = 1) -> list[int]:
        """Let Main's own past snapshots compete for core seats.

        Only policies that came through exploiter admission ever received an
        admission_status, so `milestone_main` archives could never appear in
        _ordered_probationary_challengers(), never reached solver_ids, and were
        therefore never rankable for a core seat. With admission rare that left
        all 16 core seats permanently empty, which in turn keeps
        `role_mixture` on its "no core exists" warm-up split -- the hard,
        variance, forgotten and coverage channels stay dark and the strategic
        machinery never actually runs.

        A never-seated past self needs no adversarial vetting, so it is promoted
        straight to "solver_eligible" (the status a matured challenger reaches)
        rather than through the 3 challenger seats, which belong to unproven
        policies. A retired past self is different: it may return only through
        the historical UCB gate and challenger probation. Only
        `limit` per milestone and only while the solver game still completes
        inside completion_edge_cap: each new member adds pairwise edges that
        cost real evaluator games, so this stays on the same budget discipline
        _select_solver_challengers() already uses. Promotion grants candidacy,
        never a seat -- rank_roster_candidates() still orders by nash_mass, so
        a snapshot Main dominates simply never gets seated (or loses its seat).
        """
        config = self.controller.config
        payoff = config.payoff_graph
        # _select_solver_challengers() fills challenger seats up to
        # 19 - len(incumbents), so current Main plus incumbents plus those
        # challengers can already sit exactly on solver_policy_cap. Count the
        # challengers this milestone actually took, or strategic_active_ids()
        # raises once a promotion is stacked on a full population.
        seated = 1 + len(incumbent_entries) + len(reserved_challenger_ids or ())
        if seated >= config.active_game.solver_policy_cap:
            return []
        limit = min(int(limit), config.active_game.solver_policy_cap - seated)
        records = trainer.archive.records
        promoted_iterations = [
            int(record.get("iteration", 0)) for record in records.values()
            if record.get("kind") == "milestone_main"
            and record.get("admission_status") == "solver_eligible"]
        available = [
            int(identity) for identity, record in records.items()
            if record.get("kind") == "milestone_main"
            # 2026-09-04 FIX (A2): milestone_main records are archived with
            # admitted=True (ppo_gpu._archive_current / league.add), so the old
            # `admission_status is None` filter matched nothing and core stayed
            # 0/16 forever. Accept only never-promoted past selves (None or
            # "admitted"). `archive_only` means the policy was deliberately
            # removed from active matchmaking; it may return exclusively via
            # _record_historical_audit(), which sets "probationary" and sends it
            # through challenger exposure before any later core seat. Including
            # archive_only here bypassed that gate (archive 4 did exactly that at
            # live iteration 9500 despite a Main UCB95 of 1.0). The stale marker
            # is an extra fail-closed guard against malformed legacy metadata.
            and record.get("admission_status") in (None, "admitted")
            and record.get("metrics", {}).get("stale_retired_at_iteration") is None
            and int(identity) != int(current_id)]
        if not available:
            return []

        def spread_key(identity: int):
            created = int(records[identity].get("iteration", 0))
            if not promoted_iterations:
                # Cold start: the newest past self is the strongest sparring
                # partner available, so seed core with it rather than with a
                # near-random snapshot from the first few hundred iterations.
                return (created, identity)
            # Afterwards prefer the widest hole in the promoted history: the
            # newest snapshots are already covered by the recent ring, so a
            # core seat buys the most diversity furthest from what is held.
            gap = min(abs(created - value) for value in promoted_iterations)
            return (gap, created, identity)

        # 2026-09-03 (found via user-requested re-audit): StrategicIndexSelector
        # already ranks candidates by hard/regression(forgotten)/payoff_novelty/
        # nash_support each milestone (strategic_index.py) -- that IS the
        # "strong/weak/diverse" targeting this pool is supposed to have. But
        # observe_milestone() only ever used its output (`last_index`) to plan
        # payoff queries; refresh_roster() computed its own, unrelated
        # candidate set from whatever was already seated. The index's result
        # never reached a roster decision, so milestone_main promotion fell
        # back to pure temporal spacing -- a real but behaviour-blind proxy
        # for diversity. Prefer whichever available snapshot the index
        # actually flagged this milestone; spread_key stays the tie-break and
        # the sole ordering when the index hasn't (yet) selected any
        # milestone_main this round, so promotion never stalls waiting on it.
        signal_reasons = {"hard", "regression", "payoff_novelty",
                          "behavior_novelty", "nash_support", "redteam", "scenario"}
        last_index = getattr(self.controller, "last_index", None) or {}
        selected_ids = {int(value) for value in last_index.get("selected_ids", ())}
        reasons_by_id = last_index.get("reasons", {})

        def has_strategic_signal(identity: int) -> bool:
            if identity not in selected_ids:
                return False
            tags = reasons_by_id.get(str(identity), ())
            return bool(signal_reasons.intersection(tags))

        def priority_key(identity: int):
            return (has_strategic_signal(identity), spread_key(identity))

        promoted: list[int] = []
        # Seed with the challengers this milestone accepted so the missing-pair
        # estimate is measured against the population that will actually be
        # solved, not an understated one.
        entries = [*incumbent_entries, *(
            {"role": "challenger", "archive_id": int(identity)}
            for identity in (reserved_challenger_ids or ()))]
        for identity in sorted(available, key=priority_key, reverse=True)[:int(limit)]:
            trial = [*entries, {"role": "core", "archive_id": int(identity)}]
            ids = strategic_active_ids(
                trial, current_id=current_id,
                cap=config.active_game.solver_policy_cap)
            missing = self.controller.graph.missing_solver_pairs(
                ids, minimum_blocks=payoff.solver_paired_blocks,
                evaluator_protocol=payoff.evaluator_protocol,
                scenario_bank_versions=payoff.solver_compatible_scenario_banks)
            if len(missing) > config.active_game.completion_edge_cap:
                continue
            record = records[identity]
            record["admission_status"] = "solver_eligible"
            record.setdefault("metrics", {})[
                "core_promoted_at_iteration"] = int(iteration)
            promoted.append(int(identity))
            entries = trial
        return promoted

    def _retire_stale_members(self, trainer, incumbent_entries, *,
                              iteration: int) -> tuple[list[dict], list[int]]:
        """Drop seats held by opponents Main has comprehensively solved.

        Returns the surviving entries plus the retired archive ids. Retirement
        is not deletion: the record stays in the archive at "archive_only", so
        the historical audit can still re-enrol it if Main later regresses
        against it (_record_historical_audit).
        """
        scored = []
        for entry in incumbent_entries:
            identity = entry.get("archive_id")
            if identity is None:
                continue
            games = float(entry.get("games", 0.0))
            main_winrate = float(entry.get("ema", 0.5))
            if (games >= STALE_MEMBER_MIN_GAMES
                    and main_winrate >= STALE_MEMBER_MAIN_WINRATE):
                scored.append((main_winrate, int(identity), entry))
        if not scored:
            return list(incumbent_entries), []
        # Most thoroughly solved first.
        scored.sort(key=lambda item: (-item[0], item[1]))
        retired_ids = []
        for main_winrate, identity, _entry in scored[:STALE_MEMBER_EVICTIONS_PER_MILESTONE]:
            record = trainer.archive.records.get(identity)
            if record is None:
                continue
            record["admission_status"] = "archive_only"
            metrics = record.setdefault("metrics", {})
            metrics["stale_retired_at_iteration"] = int(iteration)
            metrics["stale_retired_main_winrate"] = main_winrate
            retired_ids.append(identity)
        survivors = [entry for entry in incumbent_entries
                     if entry.get("archive_id") not in set(retired_ids)]
        return survivors, retired_ids

    def _mature_completed_challengers(self, trainer, *, iteration: int) -> None:
        previous_solver = set(map(int, getattr(
            self.controller, "last_solver_ids", ())))
        for entry in trainer.pool.active_entries():
            if entry.get("role") != "challenger" or entry.get("archive_id") is None:
                continue
            identity = int(entry["archive_id"])
            record = trainer.archive.records.get(identity)
            if (record is None or identity not in previous_solver
                    or record.get("admission_status") != "probationary"
                    or float(entry.get("games", 0.0))
                    < self.controller.config.active_game.challenger_min_exposure_games):
                continue
            record["admission_status"] = "solver_eligible"
            record.setdefault("metrics", {})[
                "probation_completed_at_iteration"] = int(iteration)

    @staticmethod
    def _rank_incumbent_entries(trainer, entries) -> list[dict]:
        return sorted(entries, key=lambda entry: (
            trainer.archive.records[int(entry["archive_id"])].get(
                "admission_status") == "probationary",
            float(trainer.archive.records[int(entry["archive_id"])].get(
                "nash_mass", 0.0))
            + float(trainer.archive.records[int(entry["archive_id"])].get(
                "regression", 0.0)),
            bool(entry.get("coverage", False)),
            int(trainer.archive.records[int(entry["archive_id"])].get(
                "iteration", -1)),
            int(entry["archive_id"])), reverse=True)

    def _maybe_run_immediate_confirmatory(self, trainer, query: dict, *,
                                          iteration: int, candidate_ids: set[int],
                                          current_id: int) -> dict | None:
        """2026-09-03 (user decision): a standard league runs confirmatory the
        moment screening clears, not on the query planner's next milestone
        cycle. The planner only replans once per on_milestone call, so
        without this a candidate whose screening just passed would sit at
        "screening_status=passed, not yet admitted" for a full extra
        milestone before confirmatory is even planned -- on top of the
        confirmatory evaluation itself. Left/right are carried over
        unchanged from the screening query so the candidate's score/lcb
        perspective in _update_candidate_status stays consistent with what
        screening measured.
        """
        if query["phase"] != "screening":
            return None
        admission_candidate = next(
            (identity for identity in (int(query["left"]), int(query["right"]))
             if identity in candidate_ids), None)
        if admission_candidate is None:
            return None
        if trainer.archive.records.get(
                admission_candidate, {}).get("screening_status") != "passed":
            return None
        payoff_config = self.controller.config.payoff_graph
        confirmatory_query = dict(query)
        confirmatory_query.update(
            phase="confirmatory",
            minimum_blocks=payoff_config.confirmatory_paired_blocks,
            scenario_bank_version=payoff_config.confirmatory_scenario_bank)
        return self._execute_query(
            trainer, confirmatory_query, iteration=iteration,
            candidate_ids=candidate_ids, current_id=current_id)

    def screen_fresh_candidate(self, trainer, *, archive_id: int,
                               iteration: int, current_id: int) -> bool:
        """Screen a just-trained exploiter before Main moves on.

        The side learner runs *after* on_milestone(), so its candidate could
        not be planned until the next milestone -- 500 iterations later,
        against a Main that had meanwhile kept training. That delay measured
        two different things at once: whether the exploiter found a real
        weakness, and whether the weakness survived 500 iterations of Main
        improvement. Screening here answers the first question against the
        exact target the exploiter trained on; the later milestone re-test
        then cleanly answers the second.
        """
        record = trainer.archive.records.get(int(archive_id))
        if record is None or record.get("admitted") is True:
            return False
        if int(archive_id) == int(current_id):
            return False
        payoff = self.controller.config.payoff_graph
        left, right = sorted((int(archive_id), int(current_id)))
        query = {
            "left": left, "right": right, "phase": "screening",
            "reason": "admission_target+fresh_candidate",
            "minimum_blocks": payoff.screening_paired_blocks,
            "maximum_blocks": payoff.maximum_paired_blocks,
            "scenario_bank_version": payoff.screening_scenario_bank,
            "priority": 0.0,
        }
        candidate_ids = {int(archive_id)}
        if self._execute_query(trainer, query, iteration=iteration,
                               candidate_ids=candidate_ids,
                               current_id=current_id) is None:
            return False
        self._maybe_run_immediate_confirmatory(
            trainer, query, iteration=iteration, candidate_ids=candidate_ids,
            current_id=current_id)
        return bool(record.get("admitted") is True)

    def activate_post_side_candidate(self, trainer, *, archive_id: int,
                                     iteration: int, current_id: int,
                                     recovery: bool = False) -> bool:
        """Seat a same-milestone admission before the next Main rollout.

        The side learner runs after :meth:`on_milestone`, so its archive id is
        absent from ``last_solver_ids``.  Calling ``refresh_roster`` alone then
        silently leaves an admitted policy in the archive until the following
        500-iteration milestone.  Reserve one challenger seat, complete the one
        new solver row, recompute Nash, and only then publish the new roster.

        A fresh candidate is mandatory for this transaction.  At most two
        other probationary incumbents are retained so the validated three-seat
        challenger quota cannot strand the policy that was just admitted.
        """
        identity = int(archive_id)
        current_id = int(current_id)
        record = trainer.archive.records.get(identity)
        if record is not None:
            self._sync_admission_history(trainer, identity)
        if (record is None or record.get("admitted") is not True
                or record.get("admission_status") != "probationary"
                or record.get("safety_status", "valid") != "valid"):
            return False

        active_with_candidate = [
            entry for entry in trainer.pool.active_entries()
            if entry.get("role") in {"core", "challenger"}
            and entry.get("archive_id") is not None]
        if (any(int(entry["archive_id"]) == identity
                for entry in active_with_candidate)
                and identity in set(map(int, self.controller.last_solver_ids))):
            return False
        active = [entry for entry in active_with_candidate
                  if int(entry["archive_id"]) != identity]
        active_ids = {int(entry["archive_id"]) for entry in active}

        # The new admission consumes one of three challenger seats. Preserve
        # no more than two incumbent probationary policies, in the same fair
        # queue order used by milestone admission; extra probationaries stay
        # admitted and receive a solver-pending marker in refresh_roster().
        probationary_order = [
            value for value in self._ordered_probationary_challengers(trainer)
            if value != identity and value in active_ids]
        retained_probationary = probationary_order[:max(0, LAYOUT["challenger"] - 1)]
        ranked = self._rank_incumbent_entries(trainer, active)
        ranked_non_probationary = [
            entry for entry in ranked
            if trainer.archive.records[int(entry["archive_id"])].get(
                "admission_status") != "probationary"]
        incumbent_limit = self.controller.config.active_game.solver_policy_cap - 2
        survivors = [
            next(entry for entry in active
                 if int(entry["archive_id"]) == value)
            for value in retained_probationary]
        survivors.extend(ranked_non_probationary[:max(
            0, incumbent_limit - len(survivors))])
        prospective_entries = [
            *survivors,
            {"role": "challenger", "archive_id": identity},
        ]
        solver_ids = strategic_active_ids(
            prospective_entries, current_id=current_id,
            cap=self.controller.config.active_game.solver_policy_cap)

        payoff = self.controller.config.payoff_graph
        missing = self.controller.graph.missing_solver_pairs(
            solver_ids, minimum_blocks=payoff.solver_paired_blocks,
            evaluator_protocol=payoff.evaluator_protocol,
            scenario_bank_versions=payoff.solver_compatible_scenario_banks)
        edge_cap = self.controller.config.active_game.completion_edge_cap
        if len(missing) > edge_cap:
            metrics = record.setdefault("metrics", {})
            metrics.setdefault("solver_pending_since_iteration", int(iteration))
            metrics["post_side_activation_blocked_edges"] = int(len(missing))
            metrics["solver_pending_reason"] = "completion_edge_cap_exceeded"
            metrics["solver_status"] = "pending"
            metrics["active_role"] = None
            self.controller.log.append(
                "post_side_candidate_pending",
                {"archive_id": identity, "missing_edges": len(missing),
                 "completion_edge_cap": edge_cap,
                 "recovery": bool(recovery)},
                iteration=int(iteration))
            trainer.archive.persist()
            self.controller.persist()
            trainer.vnext_control_state = self.controller.state_dict()
            return False

        try:
            completion_queries = self._complete_active_solver_game(
                trainer, policy_ids=solver_ids, iteration=int(iteration),
                current_id=current_id, candidate_ids={identity})
        except Exception as exc:
            self._mark_post_side_pending(
                trainer, identity=identity, iteration=int(iteration),
                reason="payoff_completion_failed", error=exc,
                recovery=recovery)
            return False
        remaining_missing = self.controller.graph.missing_solver_pairs(
            solver_ids, minimum_blocks=payoff.solver_paired_blocks,
            evaluator_protocol=payoff.evaluator_protocol,
            scenario_bank_versions=payoff.solver_compatible_scenario_banks)
        if remaining_missing:
            self._mark_post_side_pending(
                trainer, identity=identity, iteration=int(iteration),
                reason="payoff_graph_incomplete_after_completion",
                error=RuntimeError(f"{len(remaining_missing)} missing edges"),
                recovery=recovery)
            return False
        try:
            solution = self.controller.graph.conservative_nash(
                solver_ids, minimum_blocks=payoff.solver_paired_blocks,
                evaluator_protocol=payoff.evaluator_protocol,
                scenario_bank_versions=payoff.solver_compatible_scenario_banks)
        except Exception as exc:
            self._mark_post_side_pending(
                trainer, identity=identity, iteration=int(iteration),
                reason="nash_solver_failed", error=exc, recovery=recovery)
            return False
        previous_ids = {
            int(entry["archive_id"]) for entry in active}
        displaced = sorted(previous_ids - set(map(int, solver_ids)))
        pool_snapshot = self._pool_transaction_snapshot(trainer.pool)
        record_snapshot = self._record_transaction_snapshot(
            trainer.archive.records)
        old_solver_ids = list(self.controller.last_solver_ids)
        old_last_index = copy.deepcopy(self.controller.last_index)
        old_cadence = int(
            self.controller.consecutive_milestones_without_admission)
        record_metrics = record.get("metrics", {})
        if record_metrics.get("manual_safety_reactivation"):
            route = "historical_reactivation"
        elif record_metrics.get(
                "admission_rule") == "altitude_sentinel_direct_entry_v1":
            route = "direct_altitude"
        else:
            route = "confirmatory"
        try:
            # refresh_roster consumes the proposed solver/Nash values directly;
            # neither controller state nor archive Nash metadata is published
            # until the forced challenger has been loaded and validated.
            self.refresh_roster(
                trainer, iteration=int(iteration), current_id=current_id,
                forced_challenger_ids=(identity,), selected_ids=solver_ids,
                nash_override=solution)
            seated = any(
                entry.get("role") == "challenger"
                and entry.get("archive_id") is not None
                and int(entry["archive_id"]) == identity
                for entry in trainer.pool.active_entries())
            if not seated:
                raise RuntimeError(
                    f"post-side candidate {identity} was solved but not seated")

            for archive_record in trainer.archive.records.values():
                archive_record["nash_mass"] = 0.0
            for value, mass in solution.items():
                trainer.archive.records[int(value)]["nash_mass"] = float(mass)
            self.controller.last_solver_ids = list(solver_ids)
            if self.controller.last_index is None:
                self.controller.last_index = {"reasons": {}}
            self.controller.last_index["solver_eligible_ids"] = list(solver_ids)
            metrics = record.setdefault("metrics", {})
            metrics.pop("solver_pending_since_iteration", None)
            metrics.pop("solver_pending_reason", None)
            metrics["solver_status"] = "active"
            metrics["active_role"] = "challenger"
            metrics["post_side_activated_at_iteration"] = int(iteration)
            metrics["post_side_activation_recovery"] = bool(recovery)
            metrics["post_side_entry_route"] = route
            # Direct altitude entry bypasses confirmatory admission by design.
            # It is a league entry, but must not make the confirmatory pipeline
            # look healthy or reset its no-admission counter.
            metrics["league_entry_recorded"] = True
            if (route == "confirmatory"
                    and not metrics.get(
                        "post_side_confirmatory_cadence_recorded", False)):
                self.controller.observe_admission_cadence(
                    int(iteration), admitted_count=1)
                metrics["post_side_confirmatory_cadence_recorded"] = True
            self.controller.archive_index.rebuild(
                trainer.archive.state_dict(), self.controller.graph,
                iteration=int(iteration))
        except Exception as exc:
            self._restore_pool_transaction(trainer.pool, pool_snapshot)
            self._restore_record_transaction(
                trainer.archive.records, record_snapshot)
            self.controller.last_solver_ids = old_solver_ids
            self.controller.last_index = old_last_index
            self.controller.consecutive_milestones_without_admission = old_cadence
            if hasattr(trainer, "_refresh_weights"):
                try:
                    trainer._refresh_weights()
                except Exception:
                    # Preserve the original commit failure as the diagnostic;
                    # the resident/record snapshots have already been restored.
                    pass
            self._mark_post_side_pending(
                trainer, identity=identity, iteration=int(iteration),
                reason="roster_commit_failed", error=exc,
                recovery=recovery)
            return False

        self.controller.log.append(
            "post_side_candidate_activated",
            {"archive_id": identity, "solver_eligible_ids": list(solver_ids),
             "displaced_ids": displaced,
             "completion_query_count": len(completion_queries),
             "recovery": bool(recovery), "entry_route": route},
            iteration=int(iteration))
        trainer.archive.persist()
        self.controller.persist()
        trainer.vnext_control_state = self.controller.state_dict()
        return True

    def _mark_post_side_pending(self, trainer, *, identity: int,
                                iteration: int, reason: str,
                                error: Exception | None = None,
                                recovery: bool = False) -> None:
        """Fail closed while preserving ``admission_status=probationary``."""
        record = trainer.archive.records[int(identity)]
        metrics = record.setdefault("metrics", {})
        metrics.setdefault("solver_pending_since_iteration", int(iteration))
        metrics["solver_pending_reason"] = str(reason)
        metrics["solver_status"] = "pending"
        metrics["active_role"] = None
        payload = {
            "archive_id": int(identity), "pending_reason": str(reason),
            "recovery": bool(recovery),
        }
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"
        self.controller.log.append(
            "post_side_candidate_pending", payload,
            iteration=int(iteration))
        trainer.archive.persist()
        self.controller.persist()
        trainer.vnext_control_state = self.controller.state_dict()

    def stranded_probationary_ids(self, trainer) -> list[int]:
        """Return admitted policies with no active, solver, pending or exit state."""
        active_ids = {
            int(entry["archive_id"]) for entry in trainer.pool.active_entries()
            if entry.get("archive_id") is not None}
        solver_ids = set(map(int, self.controller.last_solver_ids))
        stranded = []
        for identity, candidate in trainer.archive.records.items():
            metrics = candidate.get("metrics") or {}
            if (candidate.get("admitted") is True
                    and candidate.get("admission_status") == "probationary"
                    and int(identity) not in active_ids
                    and int(identity) not in solver_ids
                    and metrics.get("solver_pending_since_iteration") is None
                    and metrics.get("active_evicted_at_iteration") is None
                    and metrics.get("stale_retired_at_iteration") is None):
                stranded.append(int(identity))
        return sorted(stranded, key=lambda value: (
            int(trainer.archive.records[value].get("iteration", -1)), value))

    def assert_no_stranded_probationary(self, trainer) -> None:
        stranded = self.stranded_probationary_ids(trainer)
        if stranded:
            raise RuntimeError(
                "admitted probationary policies lack active/pending/exit state: "
                f"{stranded}")

    def reconcile_stranded_post_side_candidate(self, trainer, *, iteration: int,
                                               current_id: int) -> tuple[int, ...]:
        """Recover stranded and transiently failed post-side admissions.

        Capacity/edge-cap pending is an intentional scheduling result and is
        not retried on every launch. Transaction failures are different:
        after a code or infrastructure fix, resume is the safest place to
        retry them through the same atomic activation path.
        """
        recovered = []
        retryable_reasons = {
            "payoff_completion_failed",
            "payoff_graph_incomplete_after_completion",
            "nash_solver_failed",
            "roster_commit_failed",
        }
        retryable_pending = [
            int(identity) for identity, candidate in trainer.archive.records.items()
            if candidate.get("admitted") is True
            and candidate.get("admission_status") == "probationary"
            and (candidate.get("metrics") or {}).get("solver_pending_reason")
            in retryable_reasons]
        candidates = list(dict.fromkeys([
            *self.stranded_probationary_ids(trainer),
            *sorted(retryable_pending, key=lambda value: (
                int(trainer.archive.records[value].get("iteration", -1)), value)),
        ]))
        for identity in candidates:
            if self.activate_post_side_candidate(
                    trainer, archive_id=identity, iteration=int(iteration),
                    current_id=int(current_id), recovery=True):
                trainer.archive.records[identity].setdefault("metrics", {})[
                    "post_side_activation_recovered_at_iteration"] = int(iteration)
                trainer.archive.persist()
                recovered.append(identity)
        self.assert_no_stranded_probationary(trainer)
        return tuple(recovered)

    def on_milestone(self, trainer, *, iteration: int, current_id: int) -> dict:
        if not self.controller.may_mutate_training:
            raise RuntimeError("vNext live decisions are frozen by health or release state")
        if hasattr(trainer, "_sync_active_scores_to_archive"):
            trainer._sync_active_scores_to_archive()
        # 2026-09-04 FIX (A3a): snapshot pre-call admissions so newly_admitted
        # counts everything this call admits (not just the pre-snapshot
        # candidate_ids, which exclude same-call admissions by construction).
        _admitted_before = {int(identity) for identity, record in
                            trainer.archive.records.items()
                            if record.get("admitted") is True}
        active_ids = strategic_active_ids(
            trainer.pool.active_entries(), current_id=current_id,
            cap=self.controller.config.active_game.solver_policy_cap)
        self._mature_completed_challengers(trainer, iteration=iteration)
        records = trainer.archive.records
        historical_ids = [
            int(identity) for identity, record in records.items()
            if int(record.get("iteration", self.controller.config.source_iteration))
            <= self.controller.config.source_iteration
            and record.get("payoff_eligible", True)
            and not record.get("invalid", False)]
        all_cycle_ids = self.controller.graph.confident_cycle_members(historical_ids)
        cycle_sentinels = sorted(all_cycle_ids, key=lambda identity: (
            float(records[identity].get("regression", 0.0)),
            -float(records[identity].get("current_score", 0.5)), identity),
            reverse=True)[:12]
        sentinel_candidates = [
            identity for identity, record in records.items()
            if record.get("payoff_eligible", True)
            and not record.get("invalid", False)]
        sentinels = sorted(sentinel_candidates, key=lambda identity: (
            float(records[identity].get("regression", 0.0)), identity), reverse=True)[:12]
        candidate_ids = [identity for identity, record in records.items()
                         if not record.get("admitted", True)
                         and record.get("payoff_eligible", True)
                         and record.get("admission_status") != "dormant_rejected"
                         and int(record.get("iteration", 0))
                         > self.controller.config.source_iteration]
        candidate_ids = candidate_ids[-8:]
        self._assign_successor_incumbents(trainer, candidate_ids)
        proposal = self.controller.observe_milestone(
            iteration, trainer.archive.state_dict(), current_id=current_id,
            active_ids=active_ids, sentinel_ids=sentinels,
            cycle_ids=cycle_sentinels,
            candidate_ids=candidate_ids, commit=False)
        if self.controller.freeze_reasons:
            raise RuntimeError("vNext proposal is frozen and cannot mutate training")
        candidate_id_set = set(candidate_ids)
        for query in proposal["payoff_queries"]:
            self._execute_query(
                trainer, query, iteration=iteration,
                candidate_ids=candidate_id_set, current_id=current_id)
            self._maybe_run_immediate_confirmatory(
                trainer, query, iteration=iteration,
                candidate_ids=candidate_id_set, current_id=current_id)
        self._queue_pending_altitude_redteams(
            trainer, set(candidate_ids), iteration=iteration)

        historical_queries, reactivated = self._audit_historical_archive(
            trainer, iteration=iteration, current_id=current_id,
            active_ids=active_ids, cycle_ids=all_cycle_ids,
            candidate_ids=set(candidate_ids))
        # Admission and historical re-enrolment happen before completion. The
        # exact challengers that will enter rollout matchmaking must therefore
        # receive their complete row now, never one milestone later.
        incumbent_entries = [
            entry for entry in trainer.pool.active_entries()
            if entry.get("role") in {"core", "challenger"}
            and entry.get("archive_id") is not None]
        # Reserve one of the 19 opponent slots when a new probationary row is
        # waiting. This bounds churn and lets a completed challenger compete
        # with the old core at the following milestone.
        incumbent_entries, retired_stale_ids = self._retire_stale_members(
            trainer, incumbent_entries, iteration=iteration)
        incumbent_ids = {int(entry["archive_id"]) for entry in incumbent_entries}
        active_probationary = sum(
            trainer.archive.records.get(identity, {}).get("admission_status")
            == "probationary" for identity in incumbent_ids)
        waiting_probationary = any(
            identity not in incumbent_ids
            for identity in self._ordered_probationary_challengers(trainer))
        reserve = 1 if waiting_probationary and active_probationary < 3 else 0
        incumbent_entries = self._rank_incumbent_entries(
            trainer, incumbent_entries)[:19 - reserve]
        prospective_challengers, pending_challengers = self._select_solver_challengers(
            trainer, current_id=current_id, incumbent_entries=incumbent_entries,
            iteration=iteration)
        prospective_core = self._promote_milestone_main_candidates(
            trainer, iteration=iteration, current_id=current_id,
            incumbent_entries=incumbent_entries,
            reserved_challenger_ids=prospective_challengers)
        prospective_entries = [*incumbent_entries, *(
            {"role": "challenger", "archive_id": identity}
            for identity in prospective_challengers), *(
            {"role": "core", "archive_id": identity}
            for identity in prospective_core)]
        solver_ids = strategic_active_ids(
            prospective_entries, current_id=current_id,
            cap=self.controller.config.active_game.solver_policy_cap)
        completion_queries = self._complete_active_solver_game(
            trainer, policy_ids=solver_ids, iteration=iteration,
            current_id=current_id, candidate_ids=set(candidate_ids))
        payoff_config = self.controller.config.payoff_graph
        solution = self.controller.graph.conservative_nash(
            solver_ids, minimum_blocks=payoff_config.solver_paired_blocks,
            evaluator_protocol=payoff_config.evaluator_protocol,
            scenario_bank_versions=payoff_config.solver_compatible_scenario_banks)
        for record in trainer.archive.records.values():
            record["nash_mass"] = 0.0
        for identity, mass in solution.items():
            trainer.archive.records[identity]["nash_mass"] = float(mass)
        # 2026-09-04 FIX (A3a): count every admission this call produced
        # (diff vs pre-call snapshot), not just pre-snapshot candidates.
        # current_id (milestone_main, admitted before this call) excluded.
        newly_admitted = [identity for identity, record in
                          trainer.archive.records.items()
                          if record.get("admitted") is True
                          and int(identity) not in _admitted_before
                          and int(identity) != int(current_id)]
        self.controller.observe_admission_cadence(
            iteration, admitted_count=len(newly_admitted))
        self.controller.last_solver_ids = list(solver_ids)
        self.controller.last_index["solver_eligible_ids"] = list(solver_ids)
        self.controller.archive_index.rebuild(
            trainer.archive.state_dict(), self.controller.graph,
            iteration=iteration)
        self.refresh_roster(trainer, iteration=iteration, current_id=current_id)
        trainer.archive.persist()
        self.controller.log.append(
            "milestone_live_sparse_commit",
            {"adapter_protocol": LIVE_ADAPTER_PROTOCOL,
             "solver_eligible_ids": solver_ids, "nash": solution,
             "decision_query_count": len(proposal["payoff_queries"]),
             "active_completion_query_count": len(completion_queries),
             "historical_audit_query_count": len(historical_queries),
             "historical_counter_reactivated_ids": reactivated,
             "pending_challenger_ids": pending_challengers,
             "core_promoted_ids": prospective_core,
             "stale_retired_ids": retired_stale_ids,
             "candidate_queue": list(self.controller.lineages.candidate_queue),
             "newly_admitted_ids": newly_admitted,
             "consecutive_milestones_without_admission":
                 self.controller.consecutive_milestones_without_admission},
            iteration=iteration)
        self.controller.persist()
        trainer.vnext_control_state = self.controller.state_dict()
        proposal["training_mutation_applied"] = True
        return proposal

    @staticmethod
    def _pool_transaction_snapshot(pool):
        has_entries = hasattr(pool, "entries")
        entries = list(pool.entries if has_entries else pool.active_entries())
        return {
            "has_entries": has_entries,
            "entries": entries,
            "entry_values": [(entry, dict(entry)) for entry in entries],
            "next_id": int(getattr(pool, "next_id", 0)),
        }

    @staticmethod
    def _restore_pool_transaction(pool, snapshot) -> None:
        for entry, values in snapshot["entry_values"]:
            entry.clear()
            entry.update(values)
        if snapshot["has_entries"]:
            pool.entries = list(snapshot["entries"])
        if hasattr(pool, "next_id"):
            pool.next_id = int(snapshot["next_id"])
        if hasattr(pool, "_reindex"):
            pool._reindex()
        if hasattr(pool, "_validate_layout"):
            pool._validate_layout()

    @staticmethod
    def _record_transaction_snapshot(records):
        return {
            int(identity): {
                "admission_status": record.get("admission_status"),
                "nash_mass": record.get("nash_mass", 0.0),
                "metrics": copy.deepcopy(record.get("metrics") or {}),
            }
            for identity, record in records.items()}

    @staticmethod
    def _restore_record_transaction(records, snapshot) -> None:
        for identity, values in snapshot.items():
            record = records.get(identity)
            if record is None:
                continue
            record["admission_status"] = values["admission_status"]
            record["nash_mass"] = values["nash_mass"]
            record["metrics"] = values["metrics"]

    @staticmethod
    def _validate_roster_proposal(proposal, *, selected, current_id: int,
                                  forced_challenger_ids=(),
                                  expected_strategic_count: int | None = None,
                                  strict_solver=False) -> None:
        all_ids = list(proposal.all_ids)
        if len(proposal.latest) != 1 or int(proposal.latest[0]) != int(current_id):
            raise RuntimeError("target roster must contain exactly the current latest")
        if len(proposal.recent) > LAYOUT["recent"]:
            raise RuntimeError("target recent roster exceeds quota")
        if len(proposal.core) > LAYOUT["core"]:
            raise RuntimeError("target core roster exceeds quota")
        if len(proposal.challenger) > LAYOUT["challenger"]:
            raise RuntimeError("target challenger roster exceeds quota")
        if len(all_ids) > sum(LAYOUT.values()) or len(all_ids) != len(set(all_ids)):
            raise RuntimeError("target roster is oversized or contains duplicates")
        selected_set = set(map(int, selected))
        if strict_solver and (
                len(selected_set) > 20 or int(current_id) not in selected_set):
            raise RuntimeError("target solver violates the 20-policy/current-main invariant")
        for identity in map(int, forced_challenger_ids):
            if identity not in selected_set or identity not in proposal.challenger:
                raise RuntimeError(
                    f"forced challenger {identity} is absent from solver/target roster")
        if expected_strategic_count is not None:
            expected = max(0, min(
                LAYOUT["core"] + LAYOUT["challenger"],
                int(expected_strategic_count)))
            expected_challengers = min(LAYOUT["challenger"], expected)
            expected_core = min(
                LAYOUT["core"], expected - expected_challengers)
            realized = (len(proposal.core), len(proposal.challenger))
            if realized != (expected_core, expected_challengers):
                raise RuntimeError(
                    "target roster left avoidable strategic vacancies: "
                    f"expected core/challenger {expected_core}/{expected_challengers}, "
                    f"got {realized[0]}/{realized[1]}")

    def refresh_roster(self, trainer, *, iteration: int, current_id: int,
                       forced_challenger_ids=(), selected_ids=None,
                       nash_override=None) -> None:
        if self.controller.last_index is None:
            raise RuntimeError("cannot refresh vNext roster before strategic selection")
        selected = [int(value) for value in (
            selected_ids if selected_ids is not None else getattr(
                self.controller, "last_solver_ids", ()))]
        if not selected:
            selected = [int(value) for value in self.controller.last_index.get(
                "solver_eligible_ids", ())]
        if not selected:
            raise RuntimeError("cannot refresh roster without a complete solver population")
        active_before = list(trainer.pool.active_entries())
        recent_ids = {int(entry["archive_id"])
                      for entry in active_before
                      if entry.get("role") == "recent"
                      and entry.get("archive_id") is not None}
        # ``latest`` is the mutable live network and has no archive id in the
        # resident pool. At a milestone current_id is also frozen into the
        # newest ``recent`` slot. RosterProposal represents latest by
        # current_id and therefore deduplicates that recent id, even though
        # sync_archive_roles() changes only core/challenger and the resident
        # pool correctly keeps all four recent snapshots. Validate the fixed
        # resident roles and proposed strategic roles separately.
        fixed_role_counts_before = {
            role: sum(entry.get("role") == role for entry in active_before)
            for role in ("latest", "recent")}
        previous_strategic_ids = {
            int(entry["archive_id"]) for entry in trainer.pool.active_entries()
            if entry.get("role") in {"core", "challenger"}
            and entry.get("archive_id") is not None}
        previous_core_ids = {
            int(entry["archive_id"]) for entry in trainer.pool.active_entries()
            if entry.get("role") == "core"
            and entry.get("archive_id") is not None}
        solver_set = set(selected)
        # Audit 2026-09-02: rank here, allocate in ActiveRosterSelector --
        # the same two steps the shadow proposal runs, so a P0 observation of
        # "active_roster" now predicts what this staged path actually applies.
        # propose() resolves a policy that qualifies for both roles in favour
        # of the challenger seat (probation before a core seat).
        strategic_order, challenger_order = rank_roster_candidates(
            trainer.archive.records, solver_ids=selected,
            recent_ids=recent_ids, current_id=current_id,
            nash_override=nash_override)
        forced = [int(value) for value in forced_challenger_ids
                  if int(value) in set(selected)]
        challenger_order = [
            *forced,
            *(identity for identity in challenger_order if identity not in set(forced)),
        ]
        # propose() is stateless; build it here rather than reaching into the
        # controller so this path shares the allocator, not an instance.
        proposal = ActiveRosterSelector(cap=sum(LAYOUT.values())).propose(
            latest_id=int(current_id), recent_ids=sorted(recent_ids),
            strategic_ids=strategic_order, challenger_ids=challenger_order)
        self._validate_roster_proposal(
            proposal, selected=selected, current_id=int(current_id),
            forced_challenger_ids=forced,
            expected_strategic_count=len(strategic_order),
            strict_solver=(selected_ids is not None or bool(forced)))
        core = list(proposal.core)
        challengers = list(proposal.challenger)
        reasons = self.controller.last_index.get("reasons", {})
        coverage = [identity for identity in core
                    if any("novelty" in reason or "audit" in reason
                           for reason in reasons.get(str(identity), reasons.get(identity, [])))]

        # The final target is completely calculated and validated before the
        # first mutation.  sync_archive_roles() may load networks and can fail;
        # restore the exact resident set and record lifecycle metadata if any
        # part of the diff-commit fails.
        pool_snapshot = self._pool_transaction_snapshot(trainer.pool)
        record_snapshot = self._record_transaction_snapshot(
            trainer.archive.records)
        try:
            trainer.pool.sync_archive_roles(
                trainer.archive, core, challengers, coverage_ids=coverage)
            if hasattr(trainer.pool, "entries"):
                active_after = list(trainer.pool.active_entries())
                realized = {
                    role: [int(entry["archive_id"]) for entry in active_after
                           if entry.get("role") == role
                           and entry.get("archive_id") is not None]
                    for role in LAYOUT}
                if (set(realized["core"]) != set(core)
                        or set(realized["challenger"]) != set(challengers)):
                    raise RuntimeError("realized strategic roster differs from target")
                if len(active_after) > sum(LAYOUT.values()):
                    raise RuntimeError("realized active roster exceeds cap")
                archive_ids_after = [
                    int(entry["archive_id"]) for entry in active_after
                    if entry.get("archive_id") is not None]
                if len(archive_ids_after) != len(set(archive_ids_after)):
                    raise RuntimeError(
                        "realized active roster contains duplicate archive ids")
                realized_counts = {
                    role: sum(entry.get("role") == role
                              for entry in active_after)
                    for role in LAYOUT}
                if any(realized_counts[role] != fixed_role_counts_before[role]
                       for role in ("latest", "recent")):
                    raise RuntimeError(
                        "strategic refresh changed fixed latest/recent residents")
                expected_active = (
                    sum(fixed_role_counts_before.values())
                    + len(core) + len(challengers))
                if len(active_after) != expected_active:
                    raise RuntimeError(
                        "realized active roster contains an unexplained vacancy or extra member")
        except Exception:
            self._restore_pool_transaction(trainer.pool, pool_snapshot)
            self._restore_record_transaction(
                trainer.archive.records, record_snapshot)
            raise
        # 2026-09-03 (churn observability, user request): with core recomputed
        # from a full re-rank every milestone (propose() has no incumbent
        # protection), a noisy nash_mass estimate near the 16th seat could in
        # principle flip a policy in and out repeatedly. take() only truncates
        # once solver_eligible core candidates exceed 16, so this cannot fire
        # yet with core underfull -- but log enter/exit/tenure/occupancy from
        # the start so the first real contention is visible in the record
        # rather than needing to be reconstructed after the fact.
        core_entered = [identity for identity in core if identity not in previous_core_ids]
        core_exited = [identity for identity in previous_core_ids if identity not in core]
        for identity in core_entered:
            trainer.archive.records[identity].setdefault(
                "metrics", {})["core_entered_at_iteration"] = int(iteration)
        for identity, record in trainer.archive.records.items():
            if identity in core:
                record["admission_status"] = "core"
            elif identity in challengers:
                if record.get("admission_status") != "probationary":
                    record["admission_status"] = "solver_eligible"
            elif (identity in solver_set and identity != int(current_id)
                  and identity not in challengers
                  and record.get("admitted", True)):
                record["admission_status"] = "archive_only"
        for identity in previous_strategic_ids - set(core) - set(challengers):
            record = trainer.archive.records.get(identity)
            if record is None:
                continue
            metrics = record.setdefault("metrics", {})
            metrics["active_evicted_at_iteration"] = int(iteration)
            entered_at = metrics.get("core_entered_at_iteration")
            if identity in previous_core_ids and entered_at is not None:
                metrics["core_tenure_milestones"] = max(
                    0, (int(iteration) - int(entered_at))
                    // max(1, trainer.cfg.milestone_period))
            if record.get("admission_status") == "probationary":
                metrics.setdefault("solver_pending_since_iteration", int(iteration))
                metrics["solver_pending_reason"] = "challenger_capacity_displacement"
                metrics["active_eviction_reason"] = "challenger_capacity_displacement"
            else:
                record["admission_status"] = "archive_only"
                metrics["active_eviction_reason"] = (
                    "weak_policy" if metrics.get("stale_retired_at_iteration") == int(iteration)
                    else "strategic_capacity_displacement")
        for identity in challengers:
            challenger_record = trainer.archive.records.get(identity)
            if (challenger_record is None
                    or challenger_record.get("admission_status") != "probationary"):
                continue
            metrics = challenger_record.setdefault("metrics", {})
            metrics.pop("solver_pending_since_iteration", None)
            metrics.pop("solver_pending_reason", None)
            if int(metrics.get("last_challenger_activation_iteration", -1)) != int(iteration):
                metrics["challenger_activation_count"] = int(
                    metrics.get("challenger_activation_count", 0)) + 1
                metrics["last_challenger_activation_iteration"] = int(iteration)
        self.sync_membership_metadata(
            trainer, solver_ids=selected, nash_override=nash_override)
        self.controller.log.append(
            "core_roster_diff",
            {"core_occupancy": f"{len(core)}/{LAYOUT['core']}",
             "core_entered": [
                 {"archive_id": identity,
                  "nash_mass": float(nash_override.get(identity, 0.0)
                                     if nash_override is not None
                                     else trainer.archive.records[identity].get("nash_mass", 0.0)),
                  "rank": core.index(identity) + 1}
                 for identity in core_entered],
             "core_exited": [
                 {"archive_id": identity,
                  "nash_mass": float(trainer.archive.records.get(
                      identity, {}).get("nash_mass", 0.0)),
                  "core_tenure_milestones": trainer.archive.records.get(
                      identity, {}).get("metrics", {}).get(
                      "core_tenure_milestones"),
                  "exit_reason": ("role_changed_to_challenger" if identity in challengers
                                  else trainer.archive.records.get(
                                      identity, {}).get("metrics", {}).get(
                                      "active_eviction_reason"))}
                 for identity in core_exited]},
            iteration=iteration)
        trainer._refresh_weights()

    @staticmethod
    def _sync_admission_history(trainer, identity: int) -> None:
        """Finalize observational receipts without rewriting admission evidence."""
        record = trainer.archive.records[int(identity)]
        metrics = record.setdefault("metrics", {})
        decided = (record.get("admitted") is True
                   or metrics.get("confirmatory_status") in {"passed", "failed", "passed_successor_admission"}
                   or metrics.get("screening_status") == "failed")
        if not decided:
            return
        metrics["vnext_fresh_evaluation_pending"] = False
        for receipt in getattr(trainer, "exploiter_history", ()):
            if receipt.get("archive_id") != int(identity):
                continue
            receipt["accepted"] = record.get("admitted") is True
            admission = receipt.setdefault("admission", {})
            admission["vnext_fresh_evaluation_pending"] = False
            admission["final_admitted"] = record.get("admitted") is True
            admission["final_admission_status"] = record.get("admission_status")
            if "admission_decision_eval" in metrics:
                admission["admission_decision_eval"] = copy.deepcopy(metrics["admission_decision_eval"])

    def sync_membership_metadata(self, trainer, *, solver_ids=None, nash_override=None) -> None:
        """Derive membership from the committed (or validated proposed) roster.

        Admission lifecycle, membership and historical exit events are separate.
        Also serves as an idempotent resume/checkpoint repair for old receipts.
        This does not change admission, roster, weights, payoff or any RNG.
        """
        solver = set(map(int, self.controller.last_solver_ids if solver_ids is None else solver_ids))
        roles = {}
        for entry in trainer.pool.active_entries():
            identity = entry.get("archive_id")
            if identity is None:
                continue
            identity = int(identity)
            roles[identity] = entry.get("role")
            if entry.get("role") in {"core", "challenger"}:
                entry["nash_mass"] = float(
                    nash_override.get(identity, 0.0) if nash_override is not None
                    else trainer.archive.records[identity].get("nash_mass", 0.0))
        for identity, record in trainer.archive.records.items():
            metrics = record.setdefault("metrics", {})
            metrics["active_role"] = roles.get(int(identity))
            if int(identity) in solver:
                metrics["solver_status"] = "active"
                metrics.pop("solver_pending_since_iteration", None)
                metrics.pop("solver_pending_reason", None)
            elif metrics.get("solver_pending_since_iteration") is not None:
                metrics["solver_status"] = "pending"
                metrics.setdefault("solver_pending_reason", "solver_capacity_or_edge_budget")
            else:
                metrics["solver_status"] = "archive_only"
            # Repair only an unambiguous historical stale exit. Do not label
            # a later capacity displacement using an older stale event.
            stale_at = metrics.get("stale_retired_at_iteration")
            if stale_at is not None and stale_at == metrics.get("active_evicted_at_iteration"):
                metrics["active_eviction_reason"] = "weak_policy"
            self._sync_admission_history(trainer, int(identity))


__all__ = [
    "LIVE_ADAPTER_PROTOCOL", "SUCCESSOR_ADMISSION_PROTOCOL",
    "VNextMilestoneAdapter",
]
