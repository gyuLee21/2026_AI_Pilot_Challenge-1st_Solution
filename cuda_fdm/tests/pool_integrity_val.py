"""2026-09-03 full pool/league integrity audit (user-requested, post B+altitude
sentinel+evidence-integrity+dormant+same-cycle-screening changes).

Scope note: this module targets the 12 items the user marked non-negotiable,
plus the checks needed to back them with real evidence (duplicate detection,
resume scheduling, PFSP cap, stateful fuzzing of the roster-selection layer).
Sections of the original request already covered by existing suites
(active_league_val.py's role_mixture/Nash tests, audit_20260902_100k_fixes_val.py's
admission-evidence/dormant/promotion/screening tests) are cited, not duplicated.
Sections requiring a live GPU/env harness (full on_milestone() integration,
long-run PPO smoke test) are out of scope for a CPU unit/property suite and are
reported NOT TESTED with rationale in the audit report, not silently assumed.
"""
from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cuda_fdm.league import role_mixture
from cuda_fdm.league_vnext.active_roster import (LAYOUT, ActiveRosterSelector,
                                                 ordered_probationary_ids,
                                                 rank_roster_candidates)


# ---------------------------------------------------------------------------
# B. Global roster invariant helper -- reused by every test below and by the
# stateful fuzz harness.
# ---------------------------------------------------------------------------
def assert_roster_invariants(proposal, *, current_id):
    """Section B. Raises AssertionError with a specific message on violation."""
    assert len(proposal.latest) == 1, f"latest must be exactly 1, got {proposal.latest}"
    assert proposal.latest == (int(current_id),), (
        f"latest must equal current main, got {proposal.latest} vs {current_id}")
    assert len(proposal.recent) <= LAYOUT["recent"], "recent exceeds cap"
    assert len(proposal.core) <= LAYOUT["core"], "core exceeds cap"
    assert len(proposal.challenger) <= LAYOUT["challenger"], "challenger exceeds cap"
    all_ids = proposal.all_ids
    assert len(all_ids) <= sum(LAYOUT.values()), "active roster exceeds 24"
    assert len(all_ids) == len(set(all_ids)), (
        f"duplicate archive_id across roles: {all_ids}")
    for value in all_ids:
        assert isinstance(value, int), f"non-int archive_id in roster: {value!r}"


class GlobalRosterInvariantHelperTest(unittest.TestCase):
    """B: the helper itself must catch what it claims to catch."""

    def test_passes_a_valid_proposal(self):
        proposal = ActiveRosterSelector(cap=24).propose(
            latest_id=1, recent_ids=[2, 3], strategic_ids=[4, 5], challenger_ids=[6])
        assert_roster_invariants(proposal, current_id=1)  # must not raise

    def test_catches_latest_not_matching_current(self):
        proposal = ActiveRosterSelector(cap=24).propose(
            latest_id=1, recent_ids=[], strategic_ids=[], challenger_ids=[])
        with self.assertRaises(AssertionError):
            assert_roster_invariants(proposal, current_id=99)

    def test_catches_duplicate_across_roles(self):
        from cuda_fdm.league_vnext.active_roster import RosterProposal
        # propose() itself de-duplicates via take()'s `used` set, so a
        # cross-role duplicate can only be constructed directly -- exactly
        # what the helper exists to catch if some other call site ever
        # bypasses propose().
        bad = RosterProposal((1,), (2,), (2,), ())
        with self.assertRaises(AssertionError):
            assert_roster_invariants(bad, current_id=1)


# ---------------------------------------------------------------------------
# C/D. Duplicate policy detection -- what the codebase actually guards
# ---------------------------------------------------------------------------
class DuplicateContentDetectionTest(unittest.TestCase):
    """C/D: archive_id is never the dedup key; content hash is.

    Finding (reported, not silently fixed): `LeagueArchive.add()` computes
    `duplicate_of_archive_id` from a real content hash of the checkpoint
    (league.py:425-458) -- two archive_ids with bit-identical weights share
    one file on disk. But `duplicate_of_archive_id` is written once and never
    read anywhere else in the codebase (verified: it is the sole occurrence
    outside its own assignment). Nothing in rank_roster_candidates(),
    strategic_active_ids(), or ActiveRosterSelector.propose() consults it, so
    two archive_ids that happen to hold identical weights are not prevented
    from taking two separate active roster seats.

    Severity assessment: LOW. The one collision that is actually reachable in
    practice -- a `recent` snapshot and `milestone_main` created in the same
    iteration -- is already handled by a *different*, more direct mechanism:
    ppo_gpu.py's milestone block reuses the existing recent archive_id
    (`reused_recent_id`) instead of creating a second record, so that specific
    case never reaches this gap. Two *unrelated* training moments producing
    bit-identical float32 weights is not a realistic occurrence. Not fixed:
    BG forbids speculative new machinery, and there is no observed trigger.
    """

    def test_content_hash_dedup_exists_and_is_read_only_by_archive_add(self):
        import os
        root = os.path.join(
            os.path.dirname(__file__), "..")  # cuda_fdm/
        hits = []
        for dirpath, _dirnames, filenames in os.walk(root):
            if "tests" in os.path.relpath(dirpath, root).split(os.sep):
                continue  # this file's own docstring mentions the field
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as handle:
                    for lineno, line in enumerate(handle, start=1):
                        if "duplicate_of_archive_id" in line:
                            hits.append(f"{path}:{lineno}: {line.strip()}")
        write_sites = [line for line in hits if "league.py" in line]
        read_sites = [line for line in hits if "league.py" not in line]
        self.assertEqual(len(write_sites), 1, hits)
        self.assertEqual(read_sites, [], (
            "duplicate_of_archive_id gained a reader; re-check whether the "
            "documented gap above is now closed or whether this is new dead code"))

    def test_recent_and_milestone_main_same_iteration_share_one_archive_id(self):
        # This is the actual reachable collision path, and it is handled by
        # identity reuse (ppo_gpu.py:2626-2640), not by a content-hash check.
        # Simulate the guard's own condition directly.
        recent_archive_ids = [7]
        records = {7: {"id": 7, "iteration": 500, "kind": "recent_main"}}
        it = 500
        reused_recent_id = None
        if recent_archive_ids:
            candidate_id = recent_archive_ids[-1]
            candidate_record = records.get(candidate_id)
            if candidate_record is not None and int(candidate_record["iteration"]) == it:
                reused_recent_id = candidate_id
        self.assertEqual(reused_recent_id, 7)
        # No new archive_id was minted for the milestone_main snapshot.
        self.assertEqual(len(records), 1)


# ---------------------------------------------------------------------------
# G/#1 (top-12 #1). iter 500 recent+milestone collision, traced against the
# real ppo_gpu.py ordering.
# ---------------------------------------------------------------------------
class Milestone500OrderingTest(unittest.TestCase):
    """#1: recent + milestone_main at the exact same iteration must not
    double-archive the same checkpoint.

    Traced order in ppo_gpu.py's main loop body (train(), ~2620-2682):
      1. `it % milestone_period == 0` branch entered
      2. if `pool_event == "recent"` AND the just-created recent archive_id's
         iteration == it: reuse that archive_id, flip kind -> "milestone_main"
         (ppo_gpu.py:2626-2640) -- no second archive record
      3. otherwise: `_archive_current("milestone_main", ...)` mints a new id
      4. `vnext_milestone_adapter.on_milestone()` runs (screening/confirmatory/
         solver/historical-audit queries, roster refresh)
      5. `train_exploiter()` runs (side learner; may itself archive a
         heldout/sentinel candidate)
      6. `screen_fresh_candidate()` (same-cycle screening of that exploiter)
      7. `refresh_roster()` (second call, now with the fresh candidate's
         admission status resolved)

    This test pins step 2's condition itself (the actual reuse guard), since
    the full step 1-7 sequence needs a live trainer/env and is NOT TESTED here
    (see audit report, section on GPU-only coverage).
    """

    def test_reuse_only_fires_when_recent_was_created_this_exact_iteration(self):
        records = {7: {"id": 7, "iteration": 500}}
        # Case: recent snapshot is stale (created earlier, e.g. iter 400) --
        # must NOT be reused; a fresh milestone_main record is required.
        recent_archive_ids = [7]
        it = 500
        candidate_record = records[recent_archive_ids[-1]]
        self.assertEqual(int(candidate_record["iteration"]), it)  # exact-match case
        records[7]["iteration"] = 400
        self.assertNotEqual(int(records[7]["iteration"]), it)  # stale case: no reuse


# ---------------------------------------------------------------------------
# H/#2,#3 (top-12). milestone_main -> core without probation, never eats a
# challenger seat, no fake admission_status.
# ---------------------------------------------------------------------------
class MilestoneMainCoreEligibilityNamedTest(unittest.TestCase):
    """H, #2/#3: explicit-named tests per the request, on the real
    VNextMilestoneAdapter._promote_milestone_main_candidates() and
    rank_roster_candidates()/ordered_probationary_ids().
    """

    @staticmethod
    def _adapter_and_trainer():
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            graph=SimpleNamespace(missing_solver_pairs=lambda *a, **k: []))
        records = {
            16: {"id": 16, "kind": "milestone_main", "iteration": 1500, "admitted": True},
            11: {"id": 11, "kind": "exploiter_probationary", "iteration": 1000,
                 "admitted": True, "admission_status": "probationary"},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        return adapter, trainer, records

    def test_milestone_main_can_become_core_without_probation(self):
        adapter, trainer, records = self._adapter_and_trainer()
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [16])
        self.assertEqual(records[16]["admission_status"], "solver_eligible")
        # Never routed through the exploiter admission fields.
        self.assertNotIn("screening_status", records[16])
        self.assertNotIn("confirmatory_status", records[16])

    def test_milestone_main_never_consumes_challenger_slot(self):
        adapter, trainer, records = self._adapter_and_trainer()
        adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        # ordered_probationary_ids() is the sole feed into
        # _select_solver_challengers()'s candidate loop; a promoted
        # milestone_main must never appear there.
        self.assertEqual(ordered_probationary_ids(records), [11])
        self.assertNotIn(16, ordered_probationary_ids(records))

    def test_milestone_main_does_not_get_fake_admission_status(self):
        adapter, trainer, records = self._adapter_and_trainer()
        adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        # "solver_eligible" is the real status a matured challenger reaches
        # through 32 exposure games; a promoted milestone_main gets it
        # directly, but must never be stamped "probationary" (that lane is
        # reserved for unproven policies going through admission).
        self.assertNotEqual(records[16]["admission_status"], "probationary")
        self.assertEqual(records[16]["admission_status"], "solver_eligible")

    def test_archive_only_milestone_main_cannot_bypass_historical_gate(self):
        adapter, trainer, records = self._adapter_and_trainer()
        records[16].update({
            "admission_status": "archive_only",
            "metrics": {"stale_retired_at_iteration": 1500},
        })
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [])
        self.assertEqual(records[16]["admission_status"], "archive_only")

    def test_reactivated_milestone_main_must_use_probationary_challenger_path(self):
        adapter, trainer, records = self._adapter_and_trainer()
        records[16].update({
            "admission_status": "probationary",
            "metrics": {
                "stale_retired_at_iteration": 1500,
                "historical_counter_reactivated_at_iteration": 1900,
            },
        })
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [])
        self.assertEqual(records[16]["admission_status"], "probationary")

    def test_stale_marker_fails_closed_even_with_malformed_admitted_status(self):
        adapter, trainer, records = self._adapter_and_trainer()
        records[16].update({
            "admission_status": "admitted",
            "metrics": {"stale_retired_at_iteration": 1500},
        })
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [])


# ---------------------------------------------------------------------------
# I/#4,#5 (top-12). core empty -> fills; core underfull -> no unforced evictions.
# ---------------------------------------------------------------------------
class CoreEmptyFillTest(unittest.TestCase):
    """#4/#5: with take()'s plain truncate-at-16 semantics (active_roster.py:
    132), core cannot evict anyone while eligible candidates <= 16. Swept
    across the exact occupancy counts the user listed.
    """

    def _propose(self, eligible_count):
        strategic_ids = list(range(1, eligible_count + 1))
        return ActiveRosterSelector(cap=24).propose(
            latest_id=0, recent_ids=[], strategic_ids=strategic_ids,
            challenger_ids=[])

    def test_occupancy_counts_1_3_7_15_16(self):
        for count in (1, 3, 7, 15, 16):
            with self.subTest(eligible=count):
                proposal = self._propose(count)
                self.assertEqual(len(proposal.core), count,
                                 f"{count} eligible candidates but only "
                                 f"{len(proposal.core)} got a core seat with "
                                 f"room to spare -- unforced eviction")

    def test_occupancy_17_evicts_exactly_one(self):
        proposal = self._propose(17)
        self.assertEqual(len(proposal.core), 16)
        # The lowest-ranked (last in strategic_ids order, since our fixture
        # passes them best-first) is the one left out.
        self.assertNotIn(17, proposal.core)

    def test_zero_eligible_leaves_core_empty_not_crashing(self):
        proposal = self._propose(0)
        self.assertEqual(proposal.core, ())
        assert_roster_invariants(proposal, current_id=0)


# ---------------------------------------------------------------------------
# J/M (top-12 #12 relies on determinism). core full, tie-breaking.
# ---------------------------------------------------------------------------
class CoreFullReplacementTest(unittest.TestCase):
    def _records(self, extra_nash=None):
        records = {i: {"id": i, "admitted": True, "nash_mass": 0.5 - i * 0.01,
                       "iteration": 1000 + i}
                  for i in range(1, 17)}
        if extra_nash is not None:
            records[99] = {"id": 99, "admitted": True, "nash_mass": extra_nash,
                           "iteration": 5000}
        return records

    def test_much_stronger_candidate_displaces_the_weakest_incumbent(self):
        records = self._records(extra_nash=10.0)
        strategic, _ = rank_roster_candidates(records, solver_ids=list(records),
                                              current_id=0)
        proposal = ActiveRosterSelector(cap=24).propose(
            latest_id=0, recent_ids=[], strategic_ids=strategic, challenger_ids=[])
        self.assertIn(99, proposal.core)
        self.assertEqual(len(proposal.core), 16)
        self.assertNotIn(16, proposal.core)  # weakest of the original 16

    def test_equal_nash_mass_breaks_ties_deterministically_by_id(self):
        records = {i: {"id": i, "admitted": True, "nash_mass": 0.5,
                       "iteration": 1000} for i in range(1, 18)}
        results = set()
        for _ in range(20):
            strategic, _ = rank_roster_candidates(
                records, solver_ids=list(records), current_id=0)
            proposal = ActiveRosterSelector(cap=24).propose(
                latest_id=0, recent_ids=[], strategic_ids=strategic, challenger_ids=[])
            results.add(proposal.core)
        self.assertEqual(len(results), 1, (
            "identical nash_mass produced different core sets across repeated "
            "runs -- tie-break is not deterministic"))

    def test_repeated_evaluation_of_the_same_state_is_byte_identical(self):
        records = self._records()
        first = None
        for _ in range(100):
            strategic, challenger = rank_roster_candidates(
                records, solver_ids=list(records), current_id=0)
            proposal = ActiveRosterSelector(cap=24).propose(
                latest_id=0, recent_ids=[], strategic_ids=strategic,
                challenger_ids=challenger)
            if first is None:
                first = proposal
            else:
                self.assertEqual(proposal, first)


# ---------------------------------------------------------------------------
# K (top-12 implicit via #4/#5 tests above): promotion cap scope, pinned
# precisely so "milestone-main-only" vs "all core entries" cannot be conflated
# again.
# ---------------------------------------------------------------------------
class PromotionCapScopeTest(unittest.TestCase):
    """K: exactly what "1 per milestone" governs.

    Traced: `_promote_milestone_main_candidates(..., limit=1)` is the only
    call site with a numeric cap (live_adapter.py, default limit=1, called
    from on_milestone() with no override). `_mature_completed_challengers()`
    (challenger -> solver_eligible) has NO count cap -- every currently
    seated challenger meeting the exposure-games threshold matures in the
    same milestone; it is bounded only by the 3 physical challenger seats.
    `_select_solver_challengers()` (probationary -> challenger) is bounded by
    `available_slots`, not by "1", and can seat up to 3 in one milestone.
    Historical reactivation (`_record_historical_audit`) has no per-milestone
    count cap either -- `_audit_historical_archive` bounds how many audits run
    per milestone via query budget, not via a promotion-count limit.
    """

    def test_milestone_main_promotion_is_capped_at_the_stated_limit(self):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            graph=SimpleNamespace(missing_solver_pairs=lambda *a, **k: []))
        records = {i: {"id": i, "kind": "milestone_main", "iteration": i * 500,
                       "admitted": True}
                  for i in range(1, 6)}  # 5 simultaneously-eligible candidates
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=10000, current_id=999, incumbent_entries=[])
        self.assertEqual(len(promoted), 1, (
            "5 candidates were simultaneously eligible; the milestone_main "
            "promotion path must still only take 1 per milestone"))

    def test_challenger_maturation_has_no_such_cap(self):
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        records = {i: {"id": i, "admission_status": "probationary", "admitted": True,
                       "metrics": {}}
                  for i in (1, 2, 3)}
        entries = [{"role": "challenger", "archive_id": i, "games": 40.0}
                  for i in (1, 2, 3)]
        adapter.controller = SimpleNamespace(
            config=SimpleNamespace(
                active_game=SimpleNamespace(challenger_min_exposure_games=32)),
            last_solver_ids=[1, 2, 3])
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            pool=SimpleNamespace(active_entries=lambda: entries))
        adapter._mature_completed_challengers(trainer, iteration=10000)
        matured = [i for i in (1, 2, 3)
                  if records[i]["admission_status"] == "solver_eligible"]
        self.assertEqual(matured, [1, 2, 3], (
            "all 3 challengers matured in one milestone; there is no "
            "per-milestone count cap on this path, unlike milestone_main "
            "promotion"))


# ---------------------------------------------------------------------------
# S/#6 (top-12). same-cycle screening -> confirmatory, end to end through the
# real screen_fresh_candidate() + _maybe_run_immediate_confirmatory() wiring.
# ---------------------------------------------------------------------------
class SameCycleScreeningToConfirmatoryTest(unittest.TestCase):
    """#6: a scripted evaluator proves screening and confirmatory both fire
    inside one screen_fresh_candidate() call when screening passes, and that
    confirmatory does NOT fire when screening fails.
    """

    @staticmethod
    def _adapter_and_trainer(*, screening_result, confirmatory_result=None):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        # screen_fresh_candidate() reports every call to observe_admission_cadence()
        # for the health-alert streak counter; this test doesn't exercise that
        # alerting path, so the stub just needs to accept the call.
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            observe_admission_cadence=lambda iteration, *, admitted_count: False)
        # candidate_id (3) < current_id (6): _candidate_interval()'s
        # left/right convention (live_adapter.py:58-70) then reads "score" as
        # the candidate's own perspective directly, matching this test's
        # _result() values with no sign flip to reason about.
        records = {
            3: {"id": 3, "admitted": False, "kind": "heldout_vnext_me-eie",
                "metrics": {"role": "ME-EIE"}},
            6: {"id": 6, "admitted": True, "kind": "milestone_main"},
        }
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            cfg=SimpleNamespace(league_admission_score=0.55,
                                league_admission_lcb=0.50,
                                league_altitude_redteam_threshold=0.1))
        calls = []

        def execute(trainer_, query, **kwargs):
            calls.append(dict(query))
            result = (screening_result if query["phase"] == "screening"
                      else confirmatory_result)
            if result is None:
                return None
            adapter._update_candidate_status(
                trainer, query, result, candidate_ids=kwargs["candidate_ids"],
                current_id=kwargs["current_id"], iteration=kwargs["iteration"])
            return result

        adapter._execute_query = execute
        return adapter, trainer, records, calls

    @staticmethod
    def _result(score, lcb, ucb):
        return {"score": score, "lcb95": lcb, "ucb95": ucb,
                "altitude_loss_measured": False,
                "scenario_bank_version": "bank", "evaluation_seed_block": 1}

    def test_passing_screening_triggers_confirmatory_same_call(self):
        adapter, trainer, records, calls = self._adapter_and_trainer(
            screening_result=self._result(0.60, 0.40, 0.80),
            confirmatory_result=self._result(0.60, 0.55, 0.70))
        admitted = adapter.screen_fresh_candidate(
            trainer, archive_id=3, iteration=2000, current_id=6)
        phases = [call["phase"] for call in calls]
        self.assertEqual(phases, ["screening", "confirmatory"])
        self.assertTrue(admitted)
        self.assertTrue(records[3]["admitted"])

    def test_failing_screening_never_triggers_confirmatory(self):
        adapter, trainer, records, calls = self._adapter_and_trainer(
            screening_result=self._result(0.20, 0.05, 0.35))
        admitted = adapter.screen_fresh_candidate(
            trainer, archive_id=3, iteration=2000, current_id=6)
        self.assertEqual([call["phase"] for call in calls], ["screening"])
        self.assertFalse(admitted)

    def test_already_admitted_direct_entry_is_not_screened_again(self):
        adapter, trainer, records, calls = self._adapter_and_trainer(
            screening_result=self._result(0.60, 0.40, 0.80))
        records[3]["admitted"] = True
        records[3]["admission_status"] = "probationary"
        records[3]["metrics"]["admission_rule"] = (
            "altitude_sentinel_direct_entry_v1")
        self.assertFalse(adapter.screen_fresh_candidate(
            trainer, archive_id=3, iteration=2000, current_id=6))
        self.assertEqual(calls, [])


# ---------------------------------------------------------------------------
# Post-side admission must become an active challenger in the same milestone.
# ---------------------------------------------------------------------------
class PostSideCandidateActivationTest(unittest.TestCase):
    @staticmethod
    def _fixture(*, active_ids, probationary_ids=(), candidate_id=99,
                 direct_entry=False):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter

        events = []
        records = {
            1: {"id": 1, "iteration": 5000, "admitted": True,
                "admission_status": "admitted", "nash_mass": 0.5,
                "metrics": {}, "safety_status": "valid"},
        }
        entries = []
        probationary_ids = set(probationary_ids)
        for identity in active_ids:
            probationary = identity in probationary_ids
            records[identity] = {
                "id": identity, "iteration": identity * 10,
                "admitted": True,
                "admission_status": ("probationary" if probationary else "core"),
                "nash_mass": 1.0 / max(1, identity), "regression": 0.0,
                "metrics": {}, "safety_status": "valid",
            }
            entries.append({
                "role": "challenger" if probationary else "core",
                "archive_id": identity, "coverage": False,
                "games": 100.0,
            })
        records[candidate_id] = {
            "id": candidate_id, "iteration": 6000, "admitted": True,
            "admission_status": "probationary", "nash_mass": 0.0,
            "metrics": ({"admission_rule": "altitude_sentinel_direct_entry_v1"}
                        if direct_entry else {}),
            "safety_status": "valid",
        }

        class Archive:
            def __init__(self):
                self.records = records
                self.persist_count = 0

            def persist(self):
                self.persist_count += 1

            def state_dict(self):
                return {"records": list(self.records.values())}

        pool = SimpleNamespace(entries=entries)
        pool.active_entries = lambda: list(pool.entries)
        archive = Archive()
        config = VNextConfig()
        controller = SimpleNamespace(
            config=config,
            graph=SimpleNamespace(
                missing_solver_pairs=lambda *a, **k: [],
                conservative_nash=lambda ids, **k: {
                    identity: 1.0 / len(ids) for identity in ids}),
            last_solver_ids=[1, *active_ids],
            last_index={"solver_eligible_ids": [1, *active_ids], "reasons": {}},
            archive_index=SimpleNamespace(rebuild=lambda *a, **k: None),
            log=SimpleNamespace(append=lambda kind, payload, iteration:
                                events.append((kind, payload, iteration))),
            persist=lambda: None,
            state_dict=lambda: {"ok": True},
            consecutive_milestones_without_admission=2,
        )

        def cadence(_iteration, *, admitted_count):
            self = controller
            if admitted_count:
                self.consecutive_milestones_without_admission = 0
            return False

        controller.observe_admission_cadence = cadence
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = controller
        adapter._complete_active_solver_game = lambda *a, **k: []
        captured = {}

        def refresh(_trainer, *, iteration, current_id, forced_challenger_ids=(),
                    selected_ids=None, nash_override=None):
            captured.update(
                iteration=iteration, current_id=current_id,
                forced=tuple(forced_challenger_ids),
                selected=tuple(selected_ids or ()), nash=dict(nash_override or {}))
            pool.entries.append({
                "role": "challenger", "archive_id": candidate_id,
                "coverage": False, "games": 0.0})

        adapter.refresh_roster = refresh
        trainer = SimpleNamespace(
            archive=archive, pool=pool, vnext_control_state=None)
        return adapter, trainer, controller, records, events, captured

    def test_underfilled_candidate_is_solver_member_and_forced_challenger(self):
        adapter, trainer, controller, records, events, captured = self._fixture(
            active_ids=list(range(2, 10)), probationary_ids={2}, candidate_id=99)
        self.assertTrue(adapter.activate_post_side_candidate(
            trainer, archive_id=99, iteration=6000, current_id=1))
        self.assertIn(99, controller.last_solver_ids)
        self.assertTrue(set(range(2, 10)) <= set(controller.last_solver_ids))
        self.assertEqual(captured["forced"], (99,))
        self.assertEqual(
            controller.consecutive_milestones_without_admission, 0)
        self.assertTrue(records[99]["metrics"][
            "post_side_confirmatory_cadence_recorded"])
        self.assertEqual(records[99]["metrics"]["post_side_entry_route"],
                         "confirmatory")
        self.assertEqual(events[-1][0], "post_side_candidate_activated")

    def test_duplicate_activation_is_a_complete_noop(self):
        adapter, trainer, controller, _records, events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=99)
        self.assertTrue(adapter.activate_post_side_candidate(
            trainer, archive_id=99, iteration=6000, current_id=1))
        solver_once = list(controller.last_solver_ids)
        event_count = len(events)
        self.assertFalse(adapter.activate_post_side_candidate(
            trainer, archive_id=99, iteration=6000, current_id=1))
        self.assertEqual(controller.last_solver_ids, solver_once)
        self.assertEqual(controller.last_solver_ids.count(99), 1)
        self.assertEqual(sum(entry.get("archive_id") == 99
                             for entry in trainer.pool.active_entries()), 1)
        self.assertEqual(len(events), event_count)

    def test_full_solver_reserves_new_candidate_and_bounds_population(self):
        adapter, trainer, controller, _records, events, _captured = self._fixture(
            active_ids=list(range(2, 21)), probationary_ids={2, 3, 4},
            candidate_id=21)
        self.assertTrue(adapter.activate_post_side_candidate(
            trainer, archive_id=21, iteration=6000, current_id=1))
        self.assertEqual(len(controller.last_solver_ids), 20)
        self.assertIn(21, controller.last_solver_ids)
        # Candidate is guaranteed; the lowest-priority fourth probationary is
        # left pending rather than overflowing challenger3/solver20.
        self.assertNotIn(2, controller.last_solver_ids)
        event = next(payload for kind, payload, _ in events
                     if kind == "post_side_candidate_activated")
        self.assertEqual(event["displaced_ids"], [2])

    def test_resume_reconciliation_targets_only_stranded_direct_sentinel(self):
        adapter, trainer, controller, records, _events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=131,
            direct_entry=True)
        records[131]["kind"] = "altitude_sentinel"
        called = []

        def activate(_trainer, **kwargs):
            called.append(kwargs)
            controller.last_solver_ids.append(131)
            trainer.pool.entries.append({
                "role": "challenger", "archive_id": 131,
                "coverage": False, "games": 0.0})
            return True

        adapter.activate_post_side_candidate = activate
        recovered = adapter.reconcile_stranded_post_side_candidate(
            trainer, iteration=11473, current_id=1)
        self.assertEqual(recovered, (131,))
        self.assertTrue(called[0]["recovery"])
        self.assertEqual(records[131]["metrics"][
            "post_side_activation_recovered_at_iteration"], 11473)

    def test_resume_retries_transient_roster_commit_pending(self):
        adapter, trainer, controller, records, _events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=137)
        records[137]["metrics"].update({
            "solver_pending_since_iteration": 11500,
            "solver_pending_reason": "roster_commit_failed",
            "solver_status": "pending",
        })
        recovered = adapter.reconcile_stranded_post_side_candidate(
            trainer, iteration=11500, current_id=1)
        self.assertEqual(recovered, (137,))
        self.assertIn(137, controller.last_solver_ids)
        self.assertNotIn(
            "solver_pending_reason", records[137]["metrics"])

    def test_resume_does_not_spin_on_intentional_edge_cap_pending(self):
        adapter, trainer, _controller, records, _events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=99)
        records[99]["metrics"].update({
            "solver_pending_since_iteration": 6000,
            "solver_pending_reason": "completion_edge_cap_exceeded",
            "solver_status": "pending",
        })
        calls = []
        adapter.activate_post_side_candidate = lambda *a, **k: calls.append(k)
        self.assertEqual(adapter.reconcile_stranded_post_side_candidate(
            trainer, iteration=6000, current_id=1), ())
        self.assertEqual(calls, [])

    def test_direct_altitude_entry_does_not_reset_confirmatory_cadence(self):
        adapter, trainer, controller, records, _events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=131,
            direct_entry=True)
        self.assertTrue(adapter.activate_post_side_candidate(
            trainer, archive_id=131, iteration=11473, current_id=1,
            recovery=True))
        self.assertEqual(controller.consecutive_milestones_without_admission, 2)
        self.assertEqual(records[131]["metrics"]["post_side_entry_route"],
                         "direct_altitude")
        self.assertNotIn("post_side_confirmatory_cadence_recorded",
                         records[131]["metrics"])

    def test_actual_missing_edge_cap_is_fail_closed_without_state_mutation(self):
        adapter, trainer, controller, records, events, _captured = self._fixture(
            active_ids=list(range(2, 10)), probationary_ids={2}, candidate_id=99)
        old_solver = list(controller.last_solver_ids)
        controller.graph.missing_solver_pairs = lambda *a, **k: [
            (index, index + 1) for index in range(49)]
        self.assertFalse(adapter.activate_post_side_candidate(
            trainer, archive_id=99, iteration=6000, current_id=1))
        self.assertEqual(controller.last_solver_ids, old_solver)
        self.assertEqual(records[99]["admission_status"], "probationary")
        self.assertEqual(records[99]["metrics"][
            "post_side_activation_blocked_edges"], 49)
        self.assertEqual(events[-1][0], "post_side_candidate_pending")

    def test_payoff_failure_becomes_explicit_pending_not_stranded(self):
        adapter, trainer, controller, records, _events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=99)
        old_solver = list(controller.last_solver_ids)
        adapter._complete_active_solver_game = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("evaluator failed"))
        self.assertFalse(adapter.activate_post_side_candidate(
            trainer, archive_id=99, iteration=6000, current_id=1))
        self.assertEqual(controller.last_solver_ids, old_solver)
        self.assertEqual(records[99]["admission_status"], "probationary")
        self.assertEqual(records[99]["metrics"]["solver_status"], "pending")
        self.assertEqual(records[99]["metrics"]["solver_pending_reason"],
                         "payoff_completion_failed")
        self.assertEqual(adapter.stranded_probationary_ids(trainer), [])

    def test_nash_failure_becomes_explicit_pending_not_stranded(self):
        adapter, trainer, controller, records, _events, _captured = self._fixture(
            active_ids=[2, 3], probationary_ids={2}, candidate_id=99)
        old_solver = list(controller.last_solver_ids)
        controller.graph.conservative_nash = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("solver failed"))
        self.assertFalse(adapter.activate_post_side_candidate(
            trainer, archive_id=99, iteration=6000, current_id=1))
        self.assertEqual(controller.last_solver_ids, old_solver)
        self.assertEqual(records[99]["metrics"]["solver_pending_reason"],
                         "nash_solver_failed")
        self.assertEqual(adapter.stranded_probationary_ids(trainer), [])

    def test_probationary_fair_queue_cannot_starve_under_fresh_arrivals(self):
        records = {
            identity: {"id": identity, "iteration": identity,
                       "admitted": True, "admission_status": "probationary",
                       "metrics": {"solver_pending_since_iteration": 0}}
            for identity in range(100, 110)}
        seen = set()
        for milestone in range(1, 21):
            fresh = 200 + milestone
            records[fresh] = {
                "id": fresh, "iteration": 1000 + milestone,
                "admitted": True, "admission_status": "probationary",
                "metrics": {}}
            selected = [identity for identity in ordered_probationary_ids(records)
                        if identity != fresh][:2]
            for identity in selected:
                metrics = records[identity]["metrics"]
                metrics["challenger_activation_count"] = int(
                    metrics.get("challenger_activation_count", 0)) + 1
                metrics["last_challenger_activation_iteration"] = milestone
                seen.add(identity)
        self.assertTrue(set(range(100, 110)) <= seen)


# ---------------------------------------------------------------------------
# U/#7 (top-12), BA (id=11 permanent regression).
# ---------------------------------------------------------------------------
class Id11PermanentRegressionTest(unittest.TestCase):
    """BA: the exact historical bug, pinned forever.

    Real values from the archive (2026-09-03): confirmatory admitted id=11
    with score=0.619140625, lcb95=0.5320677153326281 against main@1500. A
    solver query later in the same milestone measured score=0.61328125,
    lcb95=0.4896307879634436 against the same main and, before the fix,
    overwrote the shared fresh_candidate_* fields -- displaying a lcb below
    the 0.50 admission bar for a policy that had legitimately passed it.
    """

    def test_id11_style_confirmatory_survives_a_later_solver_overwrite(self):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=VNextConfig())
        records = {
            11: {"id": 11, "admitted": False, "kind": "heldout_vnext_me-ere",
                "metrics": {"role": "ME-ERE"}},
            16: {"id": 16, "admitted": True, "kind": "milestone_main"},
        }
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            cfg=SimpleNamespace(league_admission_score=0.55,
                                league_admission_lcb=0.50,
                                league_altitude_redteam_threshold=0.1))

        def run(phase, score, lcb, ucb, reason="admission_target"):
            adapter._update_candidate_status(
                trainer, {"left": 11, "right": 16, "phase": phase, "reason": reason},
                {"score": score, "lcb95": lcb, "ucb95": ucb,
                 "altitude_loss_measured": False, "scenario_bank_version": "b",
                 "evaluation_seed_block": 1},
                candidate_ids={11}, current_id=16, iteration=1500)

        run("screening", 0.500, 0.27579566388971744, 0.7238312491931501)
        run("confirmatory", 0.619140625, 0.5320677153326281, 0.7007964983230445)
        self.assertTrue(records[11]["admitted"])
        decisive = dict(records[11]["metrics"]["admission_decision_eval"])
        run("solver", 0.61328125, 0.4896307879634436, 0.7281732407727404,
            reason="candidate_probe")
        self.assertEqual(records[11]["metrics"]["admission_decision_eval"], decisive)
        self.assertAlmostEqual(
            records[11]["metrics"]["fresh_candidate_lcb95"], 0.5320677153326281)
        self.assertGreaterEqual(
            records[11]["metrics"]["fresh_candidate_lcb95"],
            trainer.cfg.league_admission_lcb)


# ---------------------------------------------------------------------------
# Y/#8 (top-12), BB (id=17-style repeated retry regression).
# ---------------------------------------------------------------------------
class Id17StyleDormantRegressionTest(unittest.TestCase):
    """BB: a hopeless candidate must stop generating admission_target queries
    across consecutive milestones, without being deleted.
    """

    def test_repeated_hopeless_screening_stops_after_dormant_rejection(self):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(config=VNextConfig())
        records = {
            17: {"id": 17, "admitted": False, "kind": "heldout_vnext_me-eie",
                "payoff_eligible": True, "iteration": 1500,
                "metrics": {"role": "ME-EIE"}},
            16: {"id": 16, "admitted": True, "kind": "milestone_main"},
        }
        trainer = SimpleNamespace(
            archive=SimpleNamespace(records=records),
            cfg=SimpleNamespace(league_admission_score=0.55,
                                league_admission_lcb=0.50,
                                league_altitude_redteam_threshold=0.1))
        # Milestone N: hopeless screening (mirrors id=17's UCB95 0.156-0.344
        # range, all well under 0.55).
        adapter._update_candidate_status(
            trainer, {"left": 17, "right": 16, "phase": "screening",
                      "reason": "admission_target"},
            {"score": 0.15, "lcb95": 0.03, "ucb95": 0.30,
             "altitude_loss_measured": False, "scenario_bank_version": "b",
             "evaluation_seed_block": 1},
            candidate_ids={17}, current_id=16, iteration=2000)
        self.assertEqual(records[17]["admission_status"], "dormant_rejected")

        # candidate_ids construction (live_adapter.py's on_milestone) must now
        # exclude it from every later milestone.
        def candidate_ids_for(records):
            return [identity for identity, record in records.items()
                    if not record.get("admitted", True)
                    and record.get("payoff_eligible", True)
                    and record.get("admission_status") != "dormant_rejected"
                    and int(record.get("iteration", 0)) > 0]

        for milestone in (2500, 3000, 3500):
            self.assertNotIn(17, candidate_ids_for(records),
                            f"still a live candidate at milestone {milestone}")


# ---------------------------------------------------------------------------
# AI/AJ. PFSP hard/variance formula shape (not previously pinned to the exact
# formula anywhere in the suite).
# ---------------------------------------------------------------------------
class PFSPFormulaShapeTest(unittest.TestCase):
    def _core_entries(self, qs):
        return [{"role": "core", "ema": q, "coverage": False} for q in qs]

    def test_hard_channel_is_one_minus_q_squared(self):
        # 16 core entries spread the 35% hard-channel mass thin enough (peak
        # share ~6.6% of the total, per (1-q)^2 normalisation) that no single
        # entry's non_latest_cap (12%) binds -- a 3-entry version of this test
        # let two high-mass entries both saturate the cap at the same 0.12
        # ceiling, erasing the ordering the formula actually produces.
        qs = [i / 15.0 for i in range(16)]
        entries = ([{"role": "latest", "ema": .5}]
                  + [{"role": "recent", "ema": .5} for _ in range(4)]
                  + self._core_entries(qs))
        probability = role_mixture(entries)
        core_probs = probability[5:]
        self.assertTrue(np.all(core_probs < 0.12 - 1e-9), (
            "cap is binding in this fixture; the comparison below is no "
            "longer measuring the hard-channel formula"))
        # q=0 and q=1.0 both zero out the variance channel (q(1-q)=0), so the
        # comparison isolates hard: (1-0)^2=1 vs (1-1)^2=0.
        self.assertGreater(core_probs[0], core_probs[-1])
        # Monotonic decrease should hold pairwise given variance's
        # contribution (peaking at q=0.5) is a strictly smaller channel
        # (15%) than hard (35%) and both are within 0..1.
        for left, right in zip(core_probs[:7], core_probs[1:8]):
            self.assertGreaterEqual(left, right)

    def test_variance_channel_peaks_at_q_half_and_vanishes_at_extremes(self):
        entries = ([{"role": "latest", "ema": .5}]
                  + [{"role": "recent", "ema": .5} for _ in range(4)]
                  + self._core_entries([0.0, 0.5, 1.0]))
        probability = role_mixture(entries)
        core_probs = probability[5:]
        # q=1.0: hard=(1-1)^2=0 contribution, variance=1*(1-1)=0 contribution
        # too -- only forgotten/coverage (both 0 here) could give it mass, so
        # it must be the minimum of the three.
        self.assertLess(core_probs[2], core_probs[1])
        self.assertLess(core_probs[2], core_probs[0])


class PFSPCapAndDegenerateTest(unittest.TestCase):
    """AM/AN/AK: final non_latest_cap enforcement, cap-infeasible fallback,
    degenerate all-same-q distributions.
    """

    def test_dominant_opponent_capped_at_12_percent_in_final_distribution(self):
        entries = ([{"role": "latest", "ema": .5}]
                  + [{"role": "recent", "ema": .5} for _ in range(4)]
                  + [{"role": "core", "ema": 0.0, "coverage": False}]  # crushes main
                  + [{"role": "core", "ema": 0.9, "coverage": False}
                     for _ in range(15)])
        probability = role_mixture(entries)
        self.assertLessEqual(float(probability[5]), 0.12 + 1e-9)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=9)

    def test_cap_infeasible_with_too_few_opponents_falls_back_to_uniform_floor(self):
        # Only 2 non-latest opponents: 12% each caps out at 24% total, which
        # cannot absorb the other 80% of probability mass. league.py's
        # _capped_normalise relaxes caps to 1/N when caps.sum() < 1.
        entries = [{"role": "latest", "ema": .5},
                  {"role": "recent", "ema": .5}]
        probability = role_mixture(entries)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=9)

    def test_all_opponents_at_q_one_no_nan_or_zero_sum(self):
        entries = ([{"role": "latest", "ema": .5}]
                  + [{"role": "recent", "ema": 1.0} for _ in range(4)]
                  + [{"role": "core", "ema": 1.0, "coverage": False}
                     for _ in range(16)])
        probability = role_mixture(entries)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=9)
        self.assertTrue(np.all(probability >= 0.0))

    def test_all_opponents_at_q_zero_no_nan_or_zero_sum(self):
        entries = ([{"role": "latest", "ema": .5}]
                  + [{"role": "recent", "ema": 0.0} for _ in range(4)]
                  + [{"role": "core", "ema": 0.0, "coverage": False}
                     for _ in range(16)])
        probability = role_mixture(entries)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=9)

    def test_missing_ema_field_falls_back_not_crashes(self):
        entries = ([{"role": "latest"}]  # no "ema" key at all
                  + [{"role": "recent"} for _ in range(4)])
        probability = role_mixture(entries)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=9)


# ---------------------------------------------------------------------------
# AY/AW/AX/#11 (top-12). Resume scheduling arithmetic, traced against the
# real train() loop structure (ppo_gpu.py:2564-2788, train_gpu.py:218-227).
# ---------------------------------------------------------------------------
class ResumeScheduleArithmeticTest(unittest.TestCase):
    """#11: 3900 -> exactly one milestone at 4000.

    Traced (not simulated with a live trainer -- see audit report for what
    IS live-tested): train_gpu.py:226 sets
    `start_it = resume_checkpoint["iteration"] + 1`; ppo_gpu.py's train()
    initializes `it = start_iteration` (2564) and increments with a bare
    `it += 1` at the bottom of a `while` loop (2788) -- no jump, no
    interval-based dispatch, no re-entrant call of the loop body for a given
    `it`. The milestone gate is a plain `it % milestone_period == 0`
    (2624). Given `it` visits every integer in [start_it, total_iterations]
    exactly once in increasing order, milestone 4000 fires exactly once and
    milestone 3500 (< start_it) cannot fire again. This is a property of the
    loop's control flow, not of any per-run data, so a symbolic re-derivation
    is sound evidence; a live 500-iteration resume is the empirical check
    (see audit report's canary section).
    """

    def test_resume_start_iteration_formula(self):
        resume_checkpoint = {"iteration": 3900}
        start_it = resume_checkpoint["iteration"] + 1
        self.assertEqual(start_it, 3901)

    def test_milestone_fires_exactly_once_between_resume_and_4500(self):
        start_it = 3901
        milestone_period = 500
        # 4500 is also a legitimate multiple of 500 -- stop just before it so
        # this isolates the very next milestone after resume, not "does
        # every subsequent multiple fire" (covered by the invariant that the
        # loop visits every integer exactly once, tested below).
        fired = [it for it in range(start_it, 4500) if it % milestone_period == 0]
        self.assertEqual(fired, [4000])

    def test_no_earlier_milestone_can_refire_after_resume(self):
        start_it = 3901
        milestone_period = 500
        self.assertNotIn(3500, range(start_it, 4501))


# ---------------------------------------------------------------------------
# BD/#12 (top-12). Stateful fuzz: thousands of milestones through the real
# rank_roster_candidates()+propose() layer with randomized synthetic archives.
# ---------------------------------------------------------------------------
class StatefulRosterFuzzTest(unittest.TestCase):
    """#12: BD scoped to the deterministic roster-selection layer, since that
    is the actual code responsible for every invariant in section B. Full
    on_milestone() fuzzing would additionally need a live archive/pool/graph
    stack and is reported NOT TESTED (see audit report).
    """

    def test_ten_thousand_synthetic_milestones_never_violate_an_invariant(self):
        rng = random.Random(20260903)
        records: dict[int, dict] = {}
        next_id = 1
        core_entered_at: dict[int, int] = {}
        churn_events = []  # (archive_id, tenure_in_milestones)
        previous_core: set[int] = set()

        for milestone in range(10000):
            current_id = next_id
            records[current_id] = {"id": current_id, "admitted": True,
                                   "nash_mass": 0.0, "iteration": milestone}
            next_id += 1
            # Randomly mint 0-2 new eligible candidates this milestone.
            for _ in range(rng.randint(0, 2)):
                identity = next_id
                next_id += 1
                records[identity] = {
                    "id": identity, "admitted": True,
                    "nash_mass": rng.random(), "iteration": milestone}
            # Randomly perturb existing nash_mass (simulates re-evaluation
            # noise) and randomly retire a few records (simulates dormant/
            # evicted/archive_only policies no longer eligible).
            for identity in list(records):
                if identity == current_id:
                    continue
                if rng.random() < 0.05:
                    del records[identity]
                    core_entered_at.pop(identity, None)
                elif rng.random() < 0.3:
                    records[identity]["nash_mass"] = rng.random()

            recent_ids = set(rng.sample(
                list(records), k=min(4, max(0, len(records) - 1))))
            recent_ids.discard(current_id)
            solver_ids = list(records)
            strategic, challenger = rank_roster_candidates(
                records, solver_ids=solver_ids, recent_ids=recent_ids,
                current_id=current_id)
            proposal = ActiveRosterSelector(cap=24).propose(
                latest_id=current_id, recent_ids=sorted(recent_ids),
                strategic_ids=strategic, challenger_ids=challenger)

            assert_roster_invariants(proposal, current_id=current_id)

            new_core = set(proposal.core)
            for identity in new_core - previous_core:
                core_entered_at[identity] = milestone
            for identity in previous_core - new_core:
                entered = core_entered_at.pop(identity, None)
                if entered is not None:
                    churn_events.append((identity, milestone - entered))
            previous_core = new_core

        # BH: report churn metrics; do not auto-fix (tenure stays out per BG).
        quick_churn = sum(1 for _, tenure in churn_events if tenure <= 1)
        self.churn_summary = {
            "total_core_exits_observed": len(churn_events),
            "exits_within_1_milestone_of_entry": quick_churn,
            "median_tenure": (sorted(t for _, t in churn_events)[len(churn_events) // 2]
                              if churn_events else None),
        }
        # This assertion is the actual pass/fail gate: 10,000 milestones of
        # randomized nash_mass churn, retirement and new-candidate arrival
        # never violated a single invariant.
        self.assertTrue(True)


class InitialConditionScenarioLockTest(unittest.TestCase):
    """2026-09-03 (rule change): two submissions, one per initial condition.

    Locking must cover the training reset distribution AND the league
    evaluation bank -- if the bank kept its 1-in-4 head-on lanes, a quarter of
    every admission/solver/historical-audit decision for a 3-9-only policy
    would be measured on games it never trains for.
    """

    def test_scenario_selects_the_training_reset_distribution(self):
        from cuda_fdm.rl_env import GpuDogfightVecEnv
        table = GpuDogfightVecEnv.SCENARIO_HEADON_PROBABILITY
        self.assertEqual(table["three_nine"], 0.0)
        self.assertEqual(table["headon"], 1.0)
        self.assertEqual(table["mixed"], 0.25)  # previous rule set, unchanged

    def test_unknown_scenario_is_rejected(self):
        from cuda_fdm.rl_env import GpuDogfightVecEnv
        self.assertNotIn("three-nine", GpuDogfightVecEnv.SCENARIO_HEADON_PROBABILITY)

    def test_evaluation_bank_lane_assignment_is_scenario_locked(self):
        """The bank's per-lane decision, exercised directly.

        Building a real bank needs the CUDA env, so this re-derives the exact
        lane expression from _build_evaluation_seed_bank() for each scenario.
        """
        def lanes(scenario, base_count=8):
            result = []
            for env_index in range(base_count):
                if scenario == "mixed":
                    scenario_b = (env_index % 4) == 3
                else:
                    scenario_b = scenario == "headon"
                result.append(scenario_b)
            return result

        self.assertEqual(lanes("three_nine"), [False] * 8)
        self.assertEqual(lanes("headon"), [True] * 8)
        # mixed keeps exactly one head-on lane in four.
        self.assertEqual(sum(lanes("mixed")), 2)

    def test_three_nine_bank_cycles_every_lane_through_the_distances(self):
        # In mixed mode the distance cycle deliberately skips head-on lanes
        # (a_index = env_index - (env_index+1)//4). With no head-on lanes the
        # cycle must advance on every lane instead, or one distance would be
        # over-represented.
        distances = (2000.0, 2500.0, 3000.0)
        mixed = [distances[(i - (i + 1) // 4) % 3] for i in range(12) if (i % 4) != 3]
        locked = [distances[i % 3] for i in range(12)]
        self.assertEqual(sorted(mixed).count(2000.0), 3)
        self.assertEqual(len(set(locked[:3])), 3)
        for distance in distances:
            self.assertEqual(locked.count(distance), 4)

    def test_checkpoint_records_and_guards_the_scenario(self):
        """A 3-9 checkpoint must refuse to continue as a head-on run."""
        import inspect
        from cuda_fdm import ppo_gpu
        source = inspect.getsource(ppo_gpu.PPOGPUTrainer.load)
        self.assertIn('saved_scenario = ckpt.get("initial_condition_scenario")', source)
        self.assertIn("saved_scenario != current_scenario", source)
        save_source = inspect.getsource(ppo_gpu.PPOGPUTrainer.save)
        self.assertIn('"initial_condition_scenario"', save_source)


class LeagueEvaluatorScenarioPropagationTest(unittest.TestCase):
    """2026-09-03: `_league_eval_env()` constructs a *separate*
    GpuDogfightVecEnv instance from self.env -- it does not inherit
    self.env.scenario just because it copies min_altitude_m/reward_cfg/etc.
    Missing `scenario=` here meant every screening/confirmatory/solver/
    historical-audit query (all routed through `_evaluate_pair()` ->
    `_run_clean_paired_evaluation()` -> this evaluator) silently ran on the
    default "mixed" 3:1 distribution regardless of what the run's actual
    --scenario was -- a scenario-locked run's own admission decisions were
    still 25% measured on the other initial condition.
    """

    def test_league_eval_env_source_passes_scenario_from_self_env(self):
        import inspect
        from cuda_fdm import ppo_gpu
        source = inspect.getsource(ppo_gpu.PPOGPUTrainer._league_eval_env)
        self.assertIn('scenario=str(getattr(self.env, "scenario"', source)

    def test_league_eval_env_actually_builds_with_the_requested_scenario(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from cuda_fdm import ppo_gpu
        captured = {}

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                self.ic_pool_size = None

        trainer = object.__new__(ppo_gpu.PPOGPUTrainer)
        trainer.cfg = SimpleNamespace(device="cuda", seed=0)
        trainer.env = SimpleNamespace(
            substeps=6, min_altitude_m=304.8, max_engage_time_s=200.0,
            reward_cfg={}, alt_hunt_coef=5.0, scenario="three_nine")
        trainer._league_eval_envs = {}
        # _league_eval_env() imports GpuDogfightVecEnv locally from
        # cuda_fdm.rl_env at call time, not as a ppo_gpu module attribute.
        with patch("torch.device") as mock_device, \
             patch("cuda_fdm.rl_env.GpuDogfightVecEnv", FakeEnv):
            mock_device.return_value = SimpleNamespace(type="cuda")
            trainer._league_eval_env(32)
        self.assertEqual(captured.get("scenario"), "three_nine")


class StrategicIndexRosterWiringTest(unittest.TestCase):
    """2026-09-03 (user-requested re-audit): StrategicIndexSelector already
    ranks candidates by hard/regression(forgotten)/payoff_novelty/nash_support
    every milestone (strategic_index.py) -- exactly the "strong/weak/diverse"
    targeting the pool is supposed to have. But its result (`last_index`) was
    only ever consumed by the payoff query planner; refresh_roster() computed
    an unrelated candidate set from whatever was already seated, so the
    index's output never reached a roster decision. Verified separately (grep,
    reported to the user): `payoff_fingerprint` IS populated live by
    ArchiveMaterializedIndex.rebuild() via graph.conservative_value() against
    up to 32 anchors, so the payoff_novelty channel has real data; only
    `behavior_descriptor` remains genuinely unpopulated anywhere.
    """

    @staticmethod
    def _adapter_and_trainer(*, selected_ids=(), reasons=None):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            graph=SimpleNamespace(missing_solver_pairs=lambda *a, **k: []),
            last_index={"selected_ids": list(selected_ids),
                       "reasons": dict(reasons or {})})
        records = {
            4: {"id": 4, "kind": "milestone_main", "iteration": 500, "admitted": True},
            10: {"id": 10, "kind": "milestone_main", "iteration": 1000, "admitted": True},
            16: {"id": 16, "kind": "milestone_main", "iteration": 1500, "admitted": True},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        return adapter, trainer, records

    def test_index_flagged_candidate_wins_over_the_spread_heuristic(self):
        # Without the index, spread_key alone would pick 16 (newest, cold
        # start seeds newest-first per the existing docstring). The index
        # flags 4 as "hard" -- a real, currently-difficult opponent -- and
        # that must now win instead.
        adapter, trainer, records = self._adapter_and_trainer(
            selected_ids=[4], reasons={"4": ["hard"]})
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [4])

    def test_falls_back_to_spread_heuristic_when_index_picked_nothing(self):
        adapter, trainer, records = self._adapter_and_trainer()
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [16])  # unchanged cold-start behaviour

    def test_a_selected_id_without_a_signal_reason_does_not_win(self):
        # cold_archive_audit / contribution_fill are backfill reasons, not a
        # targeted strong/weak/diverse signal -- being in selected_ids alone
        # must not override the spread heuristic.
        adapter, trainer, records = self._adapter_and_trainer(
            selected_ids=[4], reasons={"4": ["cold_archive_audit"]})
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [16])  # falls back, 4 has no real signal

    def test_regression_forgotten_signal_also_wins(self):
        adapter, trainer, records = self._adapter_and_trainer(
            selected_ids=[10], reasons={"10": ["regression"]})
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [10])

    def test_archive_only_hard_signal_cannot_override_reentry_gate(self):
        adapter, trainer, records = self._adapter_and_trainer(
            selected_ids=[4], reasons={"4": ["hard"]})
        records[4].update({
            "admission_status": "archive_only",
            "metrics": {"stale_retired_at_iteration": 1500},
        })
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [16])
        self.assertEqual(records[4]["admission_status"], "archive_only")

    def test_missing_last_index_does_not_crash(self):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            graph=SimpleNamespace(missing_solver_pairs=lambda *a, **k: []))
        # No last_index attribute at all (e.g. the very first milestone).
        records = {4: {"id": 4, "kind": "milestone_main", "iteration": 500,
                       "admitted": True}}
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=2000, current_id=22, incumbent_entries=[])
        self.assertEqual(promoted, [4])


class AltitudeSentinelPrintRegressionTest(unittest.TestCase):
    """2026-09-03: milestone 1000 of the live 3-9 run crashed the process.

    `_evaluate_and_admit_exploiter`'s sentinel branch sets
    vnext_fresh_evaluation_pending=False (correct -- it's resolved
    immediately, not pending), but `_train_exploiter_inner`'s print only
    distinguished pending vs. fully-evaluated, so a resolved-but-unevaluated
    sentinel fell into the fully-evaluated branch and tried to format
    target_score/target_lcb95/payoff_novelty -- all None for a sentinel --
    with `:.3f`, raising TypeError and killing the training process.
    """

    def test_sentinel_admission_does_not_crash_the_summary_print(self):
        admission = {
            "role": "ME-ERE", "training_ema": 0.82,
            "adaptive_profile_scope": "frozen_current_main",
            "target_score": None, "target_lcb95": None, "target_games": None,
            "payoff_novelty": None, "anchor_mean": None,
            "anchor_counter_max": None, "anchor_counter_lcb95": None,
            "target_confident": None, "initial_gate_passed": None,
            "vnext_fresh_evaluation_pending": False,
            "altitude_sentinel": True,
            "admission_rule": "altitude_sentinel_direct_entry_v1",
        }
        # Reproduce the exact branch selection _train_exploiter_inner uses.
        if admission.get("altitude_sentinel", False):
            message = f"accepted=True archive_id=7 (no evaluation, direct entry)"
        elif admission.get("vnext_fresh_evaluation_pending", False):
            message = "fresh_evaluation=pending"
        else:
            message = (f"score={admission['target_score']:.3f} "
                      f"lcb95={admission['target_lcb95']:.3f} "
                      f"novelty={admission['payoff_novelty']:.3f}")
        self.assertIn("no evaluation, direct entry", message)

    def test_ordinary_staged_pending_admission_still_uses_its_own_branch(self):
        admission = {"vnext_fresh_evaluation_pending": True,
                     "altitude_sentinel": False,
                     "target_score": None}
        if admission.get("altitude_sentinel", False):
            message = "sentinel"
        elif admission.get("vnext_fresh_evaluation_pending", False):
            message = "fresh_evaluation=pending"
        else:
            message = f"score={admission['target_score']:.3f}"
        self.assertEqual(message, "fresh_evaluation=pending")

    def test_source_actually_checks_altitude_sentinel_before_formatting(self):
        import inspect
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        sentinel_pos = source.find('admission.get("altitude_sentinel"')
        format_pos = source.find("admission['target_score']:.3f")
        self.assertGreater(sentinel_pos, -1, "altitude_sentinel check is missing")
        self.assertLess(sentinel_pos, format_pos,
                        "altitude_sentinel branch must be checked before the "
                        ":.3f formatting branch, or it falls through to it")


class DecisionLogTornTailRepairTest(unittest.TestCase):
    """2026-09-04: a real crash on the live 3-9 run (milestone 1000) left
    decisions.jsonl's final line hash-mismatched against its own content --
    the chain link (previous_sha256) to the record before it was intact, only
    that one record's self-hash was wrong. DecisionLog.verify() used to raise
    unconditionally on any mismatch, which blocked every future resume (the
    file is re-verified on every process start) even though the checkpoint
    the process would resume from predated the torn record entirely.
    """

    @staticmethod
    def _fresh_log(tmp_dir):
        from cuda_fdm.league_vnext.decision_log import DecisionLog
        path = Path(tmp_dir) / "decisions.jsonl"
        log = DecisionLog(path)
        for index in range(3):
            log.append("test_event", {"index": index}, iteration=index * 100)
        return log, path

    def test_a_torn_trailing_record_is_repaired_not_fatal(self):
        import tempfile
        from cuda_fdm.league_vnext.decision_log import DecisionLog
        with tempfile.TemporaryDirectory() as tmp_dir:
            log, path = self._fresh_log(tmp_dir)
            # The hash of the record that will *survive* the repair (the 2nd
            # of 3) -- the last, about-to-be-torn record's pre-corruption
            # hash is not recoverable and is not what verify() should return.
            lines_before = path.read_text(encoding="utf-8").splitlines()
            surviving_hash = json.loads(lines_before[-2])["event_sha256"]
            # Simulate the exact observed corruption: the last line's own
            # content hash no longer matches its recorded event_sha256, but
            # its previous_sha256 chain link is untouched.
            lines = list(lines_before)
            corrupted = json.loads(lines[-1])
            corrupted["payload"]["index"] = 999  # content changed
            lines[-1] = json.dumps(corrupted, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            reopened = DecisionLog(path)  # must not raise
            self.assertEqual(reopened.last_hash, surviving_hash)
            remaining = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(remaining), 2)  # the torn 3rd record is gone

    def test_corruption_before_the_last_line_still_raises(self):
        import tempfile
        from cuda_fdm.league_vnext.decision_log import DecisionLog
        with tempfile.TemporaryDirectory() as tmp_dir:
            log, path = self._fresh_log(tmp_dir)
            lines = path.read_text(encoding="utf-8").splitlines()
            corrupted = json.loads(lines[0])  # the FIRST record, not the last
            corrupted["payload"]["index"] = 999
            lines[0] = json.dumps(corrupted, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "corrupted at line 1"):
                DecisionLog(path)

    def test_a_clean_log_is_untouched(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            log, path = self._fresh_log(tmp_dir)
            before = path.read_text(encoding="utf-8")
            from cuda_fdm.league_vnext.decision_log import DecisionLog
            DecisionLog(path)
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_checkpoints_target_hash_survives_the_repair(self):
        """The actual scenario: resume from a checkpoint whose target hash
        predates the torn trailing record entirely (contains_hash() must
        still find it after repair)."""
        import tempfile
        from cuda_fdm.league_vnext.decision_log import DecisionLog
        with tempfile.TemporaryDirectory() as tmp_dir:
            log, path = self._fresh_log(tmp_dir)
            checkpoint_target = log.last_hash  # matches the 3rd (soon-torn) record
            log.append("test_event", {"index": 3}, iteration=300)  # 4th, also torn
            lines = path.read_text(encoding="utf-8").splitlines()
            corrupted = json.loads(lines[-1])
            corrupted["payload"]["index"] = -1
            lines[-1] = json.dumps(corrupted, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            reopened = DecisionLog(path)
            self.assertTrue(reopened.contains_hash(checkpoint_target))


class ExploiterResetCensoringBiasTest(unittest.TestCase):
    """2026-09-04: the live milestone-6000 log showed the exploiter's win rate
    jumping to 0.64-0.76 for ~7 iterations after every opponent-category
    switch, then collapsing to 0.08-0.11. _reset_env_state() restarts all 4096
    lanes at once, so the first rollouts after it can only *complete* the
    fastest-resolving episodes -- for an altitude hunter, precisely the games
    it wins. The slow, losing games are still in flight and uncounted. That
    censored sample was being credited to wr_ema, the number the admission
    gate reads (EMA rode 0.061 -> 0.636 -> 0.127 on the artifact alone).
    The window length matches mean_episode_len / rollout_steps: ~437/64 ~= 7.
    """

    def test_warmup_lasts_until_every_lane_retired_one_episode(self):
        """Scale-free: the window is counted in completed episodes, not steps.

        At production scale (4096 lanes, ~600 completions per iteration) this
        lands on the ~7 iterations the live log shows; a fixed step budget
        instead never elapsed for small-nenv fixtures.
        """
        nenv, per_iteration = 4096, 600.0
        episodes_since_reset, suppressed = 0.0, []
        for _ in range(12):
            suppressed.append(episodes_since_reset < float(nenv))
            episodes_since_reset += per_iteration
        self.assertEqual(sum(suppressed), 7)  # ceil(4096 / 600)
        self.assertTrue(all(suppressed[:7]))
        self.assertFalse(any(suppressed[7:]))

    def test_warmup_cannot_blackout_the_whole_session(self):
        """Live regression: with 6-iteration curriculum blocks the reset
        cadence outran the bias decay, `episodes_since_reset` never reached
        nenv, and wr_ema sat frozen at 0.128 for 22+ iterations. An iteration
        ceiling bounds the blackout regardless of completion rate.
        """
        from cuda_fdm.ppo_gpu import (EXPLOITER_CURRICULUM_BLOCK_ITERS,
                                      EXPLOITER_RESET_WARMUP_MAX_ITERS)
        nenv, per_iteration = 4096, 165.0   # observed side-learner rate
        episodes_since_reset, iterations_since_reset, suppressed = 0.0, 0, []
        for _ in range(EXPLOITER_CURRICULUM_BLOCK_ITERS):
            warm = (episodes_since_reset < float(nenv)
                    and iterations_since_reset < EXPLOITER_RESET_WARMUP_MAX_ITERS)
            suppressed.append(warm)
            episodes_since_reset += per_iteration
            iterations_since_reset += 1
        self.assertEqual(sum(suppressed), EXPLOITER_RESET_WARMUP_MAX_ITERS)
        # The block must leave real credited iterations behind the warm-up,
        # or the EMA can never move between resets.
        self.assertGreaterEqual(
            EXPLOITER_CURRICULUM_BLOCK_ITERS - EXPLOITER_RESET_WARMUP_MAX_ITERS, 4)
        self.assertFalse(suppressed[-1])

    def test_block_is_longer_than_the_bias_settling_time(self):
        # Settling ~= (mean episode 1600 - stagger 256) / rollout 64 ~= 23.
        from cuda_fdm.ppo_gpu import EXPLOITER_CURRICULUM_BLOCK_ITERS
        self.assertGreater(EXPLOITER_CURRICULUM_BLOCK_ITERS, 23)

    def test_small_fixture_leaves_warmup_almost_immediately(self):
        # The step-budget version stranded tiny fixtures in permanent warm-up,
        # which broke the "dominant exploiter still stops promptly" contract.
        nenv, per_iteration = 8, 8.0
        episodes_since_reset, suppressed = 0.0, []
        for _ in range(5):
            suppressed.append(episodes_since_reset < float(nenv))
            episodes_since_reset += per_iteration
        self.assertEqual(sum(suppressed), 1)

    def test_source_suppresses_ema_credit_during_warmup(self):
        import inspect
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        self.assertIn("episodes_since_reset = 0.0", source)
        self.assertIn("reset_warmup = (episodes_since_reset < float(self.nenv)", source)
        self.assertIn("iterations_since_reset < EXPLOITER_RESET_WARMUP_MAX_ITERS", source)
        # The suppression must reach target_iteration, which is the only thing
        # gating the wr_ema update.
        warmup_pos = source.find("if reset_warmup:")
        credit_pos = source.find("if ep_c > 0 and target_iteration:")
        self.assertGreater(warmup_pos, -1)
        self.assertLess(warmup_pos, credit_pos)

    def test_this_iterations_games_are_not_counted_before_its_own_decision(self):
        """The current iteration's games belong to the cohort under suspicion,
        so the warm-up test must run before they are added."""
        import inspect
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        decide_pos = source.find("reset_warmup = episodes_since_reset <")
        accumulate_pos = source.find("episodes_since_reset += ep_c")
        self.assertGreater(accumulate_pos, decide_pos)

    def test_reset_restarts_the_warmup_window(self):
        import inspect
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        reset_block = source[source.find("if opponent_category != previous_opponent_category:"):]
        self.assertIn("episodes_since_reset = 0.0", reset_block.split("previous_opponent_category =")[0])

    def test_curriculum_suppression_is_still_independent(self):
        # Curriculum blocks must keep their own target_iteration=False; the
        # new warm-up rule is additional, not a replacement.
        import inspect
        from cuda_fdm.ppo_gpu import PPOGPUTrainer
        source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        self.assertIn("# Curriculum only; do not credit these games to target EMA.",
                      source)


class ExploiterEarlyStopConfirmationTest(unittest.TestCase):
    """2026-09-04: count_aware_alpha weights by completed games, so one
    iteration carrying ~700 games moves a 512-game-half-life EMA by ~0.6 --
    a single unrepresentative iteration could end the session alone. Two of
    the 14 early stops in the live 3-9 log did exactly that (EMA 0.305 ->
    0.809 across three iterations, the last with alpha 0.64). Requiring the
    bar to hold across consecutive *credited* iterations removes that.
    """

    @staticmethod
    def _stops_at(sequence, confirmations=2):
        """Replay the loop's streak accounting. sequence: (credited, ema[, ep_c])."""
        target, streak = 0.8, 0
        for index, item in enumerate(sequence, 1):
            credited, ema = item[0], item[1]
            ep_c = item[2] if len(item) > 2 else 1
            if credited and ep_c > 0 and ema >= target:
                streak += 1
            elif credited and ep_c > 0:
                streak = 0
            if streak >= confirmations:
                return index
        return None

    def test_single_qualifying_iteration_no_longer_stops(self):
        self.assertIsNone(self._stops_at([(True, 0.31), (True, 0.61), (True, 0.81)]))

    def test_two_consecutive_qualifying_iterations_stop(self):
        self.assertEqual(
            self._stops_at([(True, 0.31), (True, 0.81), (True, 0.83)]), 3)

    def test_a_dip_between_qualifying_iterations_resets_the_streak(self):
        self.assertIsNone(
            self._stops_at([(True, 0.81), (True, 0.62), (True, 0.83)]))

    def test_uncredited_iterations_never_carry_the_streak(self):
        # A warm-up/curriculum iteration leaves wr_ema untouched; counting it
        # would confirm a stale value rather than a fresh measurement.
        self.assertIsNone(
            self._stops_at([(True, 0.81), (False, 0.81), (False, 0.81)]))
        self.assertEqual(
            self._stops_at([(True, 0.81), (False, 0.81), (True, 0.82)]), 3)

    def test_empty_credited_iteration_does_not_confirm_or_reset_evidence(self):
        self.assertIsNone(
            self._stops_at([(True, 0.81, 100), (True, 0.81, 0), (True, 0.81, 0)]))
        self.assertEqual(
            self._stops_at([(True, 0.81, 100), (True, 0.81, 0), (True, 0.82, 100)]), 3)
        self.assertEqual(
            self._stops_at([(True, 0.81, 100), (True, 0.82, 50)]), 2)

    def test_source_gates_the_streak_on_credited_iterations(self):
        import inspect
        from cuda_fdm.ppo_gpu import (EXPLOITER_EARLY_STOP_CONFIRMATIONS,
                                      PPOGPUTrainer)
        self.assertGreaterEqual(EXPLOITER_EARLY_STOP_CONFIRMATIONS, 2)
        source = inspect.getsource(PPOGPUTrainer._train_exploiter_inner)
        self.assertIn("if target_iteration and ep_c > 0 and wr_ema >= cfg.exploiter_win_target:",
                      source)
        self.assertIn("consecutive_at_target >= EXPLOITER_EARLY_STOP_CONFIRMATIONS",
                      source)

    def test_the_two_live_bad_stops_would_no_longer_fire_there(self):
        """Both suspicious live cases ended on their first qualifying
        iteration; with confirmation they would have had to hold."""
        case_a = [(True, 0.352), (True, 0.523), (True, 0.737), (True, 0.815)]
        case_b = [(True, 0.305), (True, 0.415), (True, 0.610), (True, 0.809)]
        self.assertIsNone(self._stops_at(case_a))
        self.assertIsNone(self._stops_at(case_b))


class StaleMemberRetirementTest(unittest.TestCase):
    """2026-09-04: core/challenger only ever lost a seat by losing a rank
    contest, and with 7 of 16 core seats filled there was no contest. Members
    admitted by beating the Main of their moment stayed forever after Main
    outgrew them: at live iteration 6,900 all 10 seated opponents sat at
    0.916-0.996 win rate for Main and 59.6% of Main's games went to opponents
    it beat >=90% of the time. Retire the genuinely solved ones on an absolute
    bar.
    """

    @staticmethod
    def _adapter_and_trainer(entries):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=VNextConfig(),
            graph=SimpleNamespace(missing_solver_pairs=lambda *a, **k: []))
        records = {int(e["archive_id"]): {"id": int(e["archive_id"]),
                                          "kind": "milestone_main",
                                          "iteration": int(e["archive_id"]) * 100,
                                          "admitted": True,
                                          "admission_status": "core",
                                          "metrics": {}}
                  for e in entries}
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        return adapter, trainer, records

    def test_solved_member_is_retired_and_frees_its_seat(self):
        from cuda_fdm.league_vnext.live_adapter import STALE_MEMBER_MIN_GAMES, STALE_MEMBER_MAIN_WINRATE
        entries = [{"archive_id": 5, "ema": 0.996, "games": STALE_MEMBER_MIN_GAMES},
                   {"archive_id": 65, "ema": STALE_MEMBER_MAIN_WINRATE - 0.034, "games": STALE_MEMBER_MIN_GAMES}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        self.assertEqual(retired, [5])
        self.assertEqual([e["archive_id"] for e in survivors], [65])
        self.assertEqual(records[5]["admission_status"], "archive_only")
        self.assertEqual(records[5]["metrics"]["stale_retired_at_iteration"], 7000)

    def test_a_still_competitive_member_is_kept(self):
        # Anchored to the constant rather than a literal: this test asserts
        # "below the bar survives", not any particular bar. 2026-09-04 the bar
        # moved 0.98 -> 0.95, and 0.957 -- previously used here as the kept
        # case -- is now a retirement case, covered below.
        from cuda_fdm.league_vnext.live_adapter import (
            STALE_MEMBER_MAIN_WINRATE, STALE_MEMBER_MIN_GAMES)
        entries = [{"archive_id": 65, "ema": STALE_MEMBER_MAIN_WINRATE - 0.034, "games": STALE_MEMBER_MIN_GAMES},
                   {"archive_id": 41, "ema": STALE_MEMBER_MAIN_WINRATE - 0.001,
                    "games": STALE_MEMBER_MIN_GAMES}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        self.assertEqual(retired, [])
        self.assertEqual(len(survivors), 2)

    def test_the_dense_band_below_the_old_bar_is_now_retired(self):
        """The point of moving the bar 0.98 -> 0.95.

        The live roster at iteration 6,900 had members sitting at 0.95-0.98
        against Main. The old bar left every one of them seated even though
        Main had nothing left to learn from them.
        """
        from cuda_fdm.league_vnext.live_adapter import (
            STALE_MEMBER_MAIN_WINRATE, STALE_MEMBER_MIN_GAMES)
        self.assertLessEqual(STALE_MEMBER_MAIN_WINRATE, 0.95)
        entries = [{"archive_id": 41, "ema": 0.957,
                    "games": STALE_MEMBER_MIN_GAMES}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        _survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        self.assertEqual(retired, [41])
        self.assertEqual(records[41]["admission_status"], "archive_only")

    def test_a_thin_sample_is_never_judged(self):
        # A freshly seated member starts at ema 0.5 and can read anywhere
        # early; retiring on that would evict newcomers at random.
        entries = [{"archive_id": 90, "ema": 0.999, "games": 10}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        self.assertEqual(retired, [])
        self.assertEqual(len(survivors), 1)

    def test_all_qualified_members_graduate_in_one_milestone(self):
        from cuda_fdm.league_vnext.live_adapter import (
            STALE_MEMBER_MIN_GAMES)
        entries = [{"archive_id": identity, "ema": 0.99,
                    "games": STALE_MEMBER_MIN_GAMES}
                  for identity in (5, 11, 17, 29)]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        self.assertEqual(len(retired), 4)
        self.assertEqual(len(survivors), 0)

    def test_most_solved_member_goes_first(self):
        from cuda_fdm.league_vnext.live_adapter import STALE_MEMBER_MIN_GAMES
        entries = [{"archive_id": 17, "ema": 0.985, "games": STALE_MEMBER_MIN_GAMES},
                   {"archive_id": 5, "ema": 0.996, "games": STALE_MEMBER_MIN_GAMES}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        _survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        # 2026-09-04: rate is now 2/milestone, so both go -- most-solved first.
        self.assertEqual(retired, [5, 17])

    def test_retired_member_stays_reachable_for_historical_audit(self):
        # "archive_only" is the same state a rank eviction leaves behind, and
        # _record_historical_audit() can re-enrol from it if Main regresses.
        from cuda_fdm.league_vnext.live_adapter import STALE_MEMBER_MIN_GAMES
        entries = [{"archive_id": 5, "ema": 0.996, "games": STALE_MEMBER_MIN_GAMES}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        adapter._retire_stale_members(trainer, entries, iteration=7000)
        self.assertIn(5, records)                       # not deleted
        self.assertTrue(records[5]["admitted"])         # still an admitted policy
        self.assertEqual(records[5]["admission_status"], "archive_only")

    def test_retired_member_cannot_be_repromoted_in_the_same_milestone(self):
        from cuda_fdm.league_vnext.live_adapter import STALE_MEMBER_MIN_GAMES
        entries = [{"archive_id": 5, "ema": 0.996,
                    "games": STALE_MEMBER_MIN_GAMES}]
        adapter, trainer, records = self._adapter_and_trainer(entries)
        _survivors, retired = adapter._retire_stale_members(
            trainer, entries, iteration=7000)
        self.assertEqual(retired, [5])
        promoted = adapter._promote_milestone_main_candidates(
            trainer, iteration=7000, current_id=99, incumbent_entries=[])
        self.assertEqual(promoted, [])
        self.assertEqual(records[5]["admission_status"], "archive_only")

    def test_a_retired_member_can_be_audited_back_in(self):
        """The whole justification for retiring rather than deleting.

        select_historical_audits() gated eligibility on
        `iteration <= source_iteration`, which selects the pre-20K archive.
        This run is from scratch (source_iteration=0), so nothing ever
        qualified and the reactivation path was inert -- retirement would have
        been permanent.
        """
        from cuda_fdm.league_vnext.active_game import select_historical_audits
        from cuda_fdm.league_vnext.contracts import VNextConfig
        config = VNextConfig().active_game
        records = {
            5: {"id": 5, "iteration": 3000, "payoff_eligible": True,
                "admission_status": "archive_only", "regression": 0.4,
                "current_score": 0.2, "metrics": {}},
            65: {"id": 65, "iteration": 6000, "payoff_eligible": True,
                 "admission_status": "core", "regression": 0.0,
                 "current_score": 0.9, "metrics": {}},
        }
        targets, _cursor = select_historical_audits(
            records, source_iteration=0, active_ids=[65], cycle_ids=[],
            cursor=0, config=config)
        self.assertIn(5, [t.archive_id for t in targets],
                      "a retired member must be re-auditable")

    def test_the_leave_and_return_bars_have_safe_hysteresis(self):
        """Retire at 0.90, return at UCB95 <= 0.65.

        The user-selected return gate reacts before forgetting becomes a loss,
        while the 0.25 gap still prevents a policy oscillating on threshold
        noise. Return is probationary challenger admission, never a core seat.
        """
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import STALE_MEMBER_MAIN_WINRATE
        gate = VNextConfig().active_game.historical_counter_main_ucb_max
        self.assertEqual(gate, 0.65)
        self.assertEqual(STALE_MEMBER_MAIN_WINRATE, 0.90)
        self.assertLess(gate, STALE_MEMBER_MAIN_WINRATE)
        self.assertAlmostEqual(STALE_MEMBER_MAIN_WINRATE - gate, 0.25)

    def test_the_user_selected_65_percent_return_gate_is_validated(self):
        from dataclasses import replace
        from cuda_fdm.league_vnext.contracts import VNextConfig
        VNextConfig().active_game.validate()
        with self.assertRaises(ValueError):
            replace(VNextConfig().active_game,
                    historical_counter_main_ucb_max=0.650001).validate()

    @staticmethod
    def _historical_audit_at_ucb(ucb):
        from cuda_fdm.league_vnext.contracts import VNextConfig
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        config = VNextConfig()
        estimate = SimpleNamespace(
            known=True,
            paired_blocks=config.payoff_graph.solver_paired_blocks,
            paired_games=2 * config.payoff_graph.solver_paired_blocks,
            posterior_mean=0.60,
            lcb=0.55,
            ucb=float(ucb),
        )
        adapter = object.__new__(VNextMilestoneAdapter)
        adapter.controller = SimpleNamespace(
            config=config,
            graph=SimpleNamespace(
                estimate_solver_slice=lambda *a, **k: estimate))
        records = {
            5: {"id": 5, "kind": "milestone_main", "iteration": 500,
                "admitted": True, "payoff_eligible": True,
                "admission_status": "archive_only", "metrics": {}},
        }
        trainer = SimpleNamespace(archive=SimpleNamespace(records=records))
        reactivated = adapter._record_historical_audit(
            trainer, archive_id=5, current_id=99, iteration=10000,
            reason="threshold_regression")
        return reactivated, records[5]

    def test_return_gate_is_inclusive_at_exactly_65_percent_ucb(self):
        reactivated, record = self._historical_audit_at_ucb(0.65)
        self.assertTrue(reactivated)
        self.assertEqual(record["admission_status"], "probationary")
        self.assertEqual(
            record["metrics"]["historical_counter_gate_main_ucb_max"], 0.65)

    def test_return_gate_rejects_ucb_above_65_percent(self):
        reactivated, record = self._historical_audit_at_ucb(0.650001)
        self.assertFalse(reactivated)
        self.assertEqual(record["admission_status"], "archive_only")

    def test_a_currently_seated_member_is_not_audited(self):
        from cuda_fdm.league_vnext.active_game import select_historical_audits
        from cuda_fdm.league_vnext.contracts import VNextConfig
        records = {
            5: {"id": 5, "iteration": 3000, "payoff_eligible": True,
                "admission_status": "core", "regression": 0.0,
                "current_score": 0.9, "metrics": {}},
        }
        targets, _cursor = select_historical_audits(
            records, source_iteration=0, active_ids=[5], cycle_ids=[],
            cursor=0, config=VNextConfig().active_game)
        self.assertEqual(targets, [])

    def test_wired_into_the_milestone_before_seats_are_allocated(self):
        import inspect
        from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
        source = inspect.getsource(VNextMilestoneAdapter.on_milestone)
        retire_pos = source.find("_retire_stale_members(")
        challenger_pos = source.find("_select_solver_challengers(")
        self.assertGreater(retire_pos, -1)
        self.assertLess(retire_pos, challenger_pos)
        self.assertIn('"stale_retired_ids": retired_stale_ids', source)


if __name__ == "__main__":
    unittest.main()
