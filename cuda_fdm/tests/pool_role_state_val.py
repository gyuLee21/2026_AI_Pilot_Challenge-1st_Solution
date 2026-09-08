"""CPU production-path regressions: stable role changes and derived metadata."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter
from cuda_fdm.league_vnext.contracts import VNextConfig
from cuda_fdm.tests.audit_20260902_100k_fixes_val import _league_trainer


class PoolRoleStateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pool-role-state-")
        self.addCleanup(self.temp.cleanup)
        self.trainer = _league_trainer(self.temp.name)
        self.pool = self.trainer.pool
        self.adapter = object.__new__(VNextMilestoneAdapter)
        self.adapter.controller = SimpleNamespace(last_solver_ids=[])

    def add(self, role="challenger"):
        trainer = self.trainer
        bundle = trainer._policy_bundle()
        identity = trainer.archive.add(bundle["model"], bundle["norm"],
            kind="exploiter", iteration=500, admitted=True, payoff_eligible=True)
        trainer.archive.records[identity]["admission_status"] = "probationary"
        self.pool.add_archive_policy(trainer.archive, identity, role)
        entry = next(e for e in self.pool.active_entries() if e.get("archive_id") == identity)
        entry.update(ema=.94, past_best=.98, games=3000.)
        return identity, entry

    def test_role_change_preserves_identity_network_rng_and_online_evidence(self):
        identity, entry = self.add()
        old = dict(entry)
        rng = entry["rng"].get_state().clone()
        next_id = self.pool.next_id
        for core, challenger in (([identity], []), ([], [identity]), ([identity], [])):
            self.pool.sync_archive_roles(self.trainer.archive, core, challenger)
            self.assertIs(next(e for e in self.pool.active_entries() if e.get("archive_id") == identity), entry)
            for key in ("id", "ema", "games", "past_best"):
                self.assertEqual(entry[key], old[key])
            for key in ("net", "norm", "rng", "actor_state"):
                self.assertIs(entry[key], old[key])
            torch.testing.assert_close(entry["rng"].get_state(), rng, rtol=0, atol=0)
            self.assertFalse(entry["retired"])
            self.assertEqual(self.pool.next_id, next_id)

    def test_role_change_does_not_delay_95pct_retirement(self):
        identity, entry = self.add()
        entry["ema"] = .96
        self.pool.sync_archive_roles(self.trainer.archive, [identity], [])
        _, retired = self.adapter._retire_stale_members(self.trainer, [entry], iteration=1000)
        self.assertEqual(retired, [identity])

    def test_full_strategic_role_swap_preserves_all_residents(self):
        core = [self.add("core")[0] for _ in range(16)]
        challenger = [self.add()[0] for _ in range(3)]
        old = {e["archive_id"]: e for e in self.pool.active_entries() if e.get("archive_id") is not None}
        self.pool.sync_archive_roles(self.trainer.archive,
            core[1:] + [challenger[0]], challenger[1:] + [core[0]])
        self.assertEqual(self.pool.role_counts()["core"], 16)
        self.assertEqual(self.pool.role_counts()["challenger"], 3)
        for entry in self.pool.active_entries():
            if entry.get("archive_id") is not None:
                self.assertIs(entry, old[entry["archive_id"]])

    def test_load_failure_rolls_back_role_and_retirement_and_next_id(self):
        identity, entry = self.add()
        old_entries = list(self.pool.entries)
        old_next = self.pool.next_id
        with patch.object(self.trainer.archive, "load_policy", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.pool.sync_archive_roles(self.trainer.archive, [identity, 999], [])
        self.assertEqual(entry["role"], "challenger")
        self.assertFalse(entry["retired"])
        self.assertEqual(self.pool.next_id, old_next)
        self.assertEqual([id(e) for e in self.pool.entries], [id(e) for e in old_entries])

    def test_duplicate_target_rejected_without_mutation(self):
        identity, entry = self.add()
        with self.assertRaises(ValueError):
            self.pool.sync_archive_roles(self.trainer.archive, [identity], [identity])
        self.assertEqual(entry["role"], "challenger")

    def test_real_exit_retains_resident_until_references_end(self):
        identity, entry = self.add()
        self.pool.sync_archive_roles(self.trainer.archive, [], [])
        self.assertTrue(entry["retired"])
        self.assertIn(entry["id"], [e["id"] for e in self.pool.entries])
        self.assertNotIn(identity, [e.get("archive_id") for e in self.pool.active_entries()])

    def test_true_reentry_still_starts_new_exposure_session(self):
        identity, old = self.add()
        self.pool.sync_archive_roles(self.trainer.archive, [], [])
        self.pool.sync_archive_roles(self.trainer.archive, [], [identity])
        new = next(e for e in self.pool.active_entries() if e.get("archive_id") == identity)
        self.assertNotEqual(new["id"], old["id"])
        self.assertEqual(new["games"], 0.)

    def test_online_sync_preserves_clean_admission_evidence(self):
        identity, entry = self.add()
        record = self.trainer.archive.records[identity]
        record.update(clean_paired_score=.123, clean_paired_lcb95=.1)
        record["metrics"]["admission_decision_eval"] = {"score": .75, "phase": "confirmatory"}
        self.trainer._sync_active_scores_to_archive()
        self.assertEqual(record["current_score"], .94)
        self.assertEqual(record["online_games"], 3000.)
        self.assertEqual(record["online_resident_id"], entry["id"])
        self.assertAlmostEqual(record["regression"], .04)
        self.assertEqual(record["clean_paired_score"], .123)
        self.assertEqual(record["metrics"]["admission_decision_eval"]["score"], .75)

    def test_membership_nash_and_pending_are_derived_from_target(self):
        identity, entry = self.add("core")
        record = self.trainer.archive.records[identity]
        record["metrics"].update(active_role="challenger", solver_status="pending",
            solver_pending_since_iteration=500, solver_pending_reason="old")
        self.adapter.sync_membership_metadata(self.trainer, solver_ids=[identity], nash_override={identity: .73})
        self.assertEqual(entry["nash_mass"], .73)
        self.assertEqual(record["metrics"]["active_role"], "core")
        self.assertEqual(record["metrics"]["solver_status"], "active")
        self.assertNotIn("solver_pending_reason", record["metrics"])
        self.assertEqual(record["admission_status"], "probationary")

    def test_pending_probationary_does_not_lose_admission_lifecycle(self):
        identity, entry = self.add()
        self.pool.sync_archive_roles(self.trainer.archive, [], [])
        record = self.trainer.archive.records[identity]
        record["metrics"]["solver_pending_since_iteration"] = 500
        self.adapter.sync_membership_metadata(self.trainer)
        self.assertEqual(record["admission_status"], "probationary")
        self.assertEqual(record["metrics"]["solver_status"], "pending")
        self.assertIsNone(record["metrics"]["active_role"])
        self.assertIn(identity, self.adapter._ordered_probationary_challengers(self.trainer))

    def test_weak_exit_repair_does_not_overwrite_later_capacity_exit(self):
        identity, entry = self.add()
        metrics = self.trainer.archive.records[identity]["metrics"]
        metrics.update(stale_retired_at_iteration=500, active_evicted_at_iteration=500,
            active_eviction_reason="bounded_solver_refresh")
        self.adapter.sync_membership_metadata(self.trainer)
        self.assertEqual(metrics["active_eviction_reason"], "weak_policy")
        metrics.update(active_evicted_at_iteration=1000, active_eviction_reason="strategic_capacity_displacement")
        self.adapter.sync_membership_metadata(self.trainer)
        self.assertEqual(metrics["active_eviction_reason"], "strategic_capacity_displacement")

    def test_final_admission_receipt_is_idempotent_and_preserves_decision(self):
        identity, entry = self.add()
        record = self.trainer.archive.records[identity]
        decision = {"score": .7, "lcb95": .6, "phase": "confirmatory"}
        record["metrics"].update(vnext_fresh_evaluation_pending=True, admission_decision_eval=decision)
        self.trainer.exploiter_history = [{"archive_id": identity, "accepted": False,
            "admission": {"vnext_fresh_evaluation_pending": True, "training_value": .85}}]
        self.adapter.sync_membership_metadata(self.trainer)
        receipt = self.trainer.exploiter_history[0]
        self.assertTrue(receipt["accepted"])
        self.assertFalse(receipt["admission"]["vnext_fresh_evaluation_pending"])
        self.assertEqual(receipt["admission"]["admission_decision_eval"], decision)
        self.assertEqual(receipt["admission"]["training_value"], .85)
        before = copy.deepcopy((self.trainer.archive.records, self.trainer.exploiter_history))
        self.adapter.sync_membership_metadata(self.trainer)
        self.assertEqual(before, (self.trainer.archive.records, self.trainer.exploiter_history))

    def test_save_reload_preserves_online_statistics_and_immutable_main(self):
        identity, entry = self.add()
        self.trainer._committed_iteration = self.trainer.iteration = 500
        self.adapter.controller.last_solver_ids = [identity]
        self.trainer.vnext_milestone_adapter = self.adapter
        model = copy.deepcopy(self.trainer.model.state_dict())
        rng = torch.get_rng_state().clone()
        path = Path(self.temp.name) / "checkpoint.pt"
        self.trainer.save(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(data["iteration"], 500)
        record = next(r for r in data["league_archive"]["records"] if r["id"] == identity)
        self.assertEqual(record["online_games"], 3000.)
        self.assertEqual(record["metrics"]["active_role"], "challenger")
        manifest = json.loads((Path(self.temp.name) / "league" / "archive_manifest.json").read_text())
        saved_record = next(r for r in manifest["records"] if r["id"] == identity)
        self.assertEqual(saved_record["online_games"], record["online_games"])
        self.assertEqual(saved_record["metrics"], record["metrics"])
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in model.items():
            torch.testing.assert_close(data["model"][key], value, rtol=0, atol=0)
        self.trainer.load(path)
        restored = next(e for e in self.pool.active_entries() if e.get("archive_id") == identity)
        self.assertEqual(restored["games"], 3000.)
        self.assertEqual(restored["ema"], .94)

    def activation_fixture(self, core_count=16):
        trainer = self.trainer
        bundle = trainer._policy_bundle()
        recent = []
        for i in range(4):
            identity = trainer.archive.add(bundle["model"], bundle["norm"],
                kind="milestone_main", iteration=i * 100, admitted=True, payoff_eligible=True)
            self.pool.add_recent_bundle(trainer.model, trainer.norm, identity, i * 100)
            recent.append(identity)
        core = [self.add("core")[0] for _ in range(core_count)]
        challenger = [self.add()[0] for _ in range(3)]
        for identity in core + challenger:
            trainer.archive.records[identity]["admission_status"] = "solver_eligible"
        candidate = trainer.archive.add(bundle["model"], bundle["norm"],
            kind="exploiter", iteration=6000, admitted=True, payoff_eligible=True)
        trainer.archive.records[candidate]["admission_status"] = "probationary"
        trainer.archive.records[candidate]["metrics"]["admission_rule"] = "altitude_sentinel_direct_entry_v1"
        controller = SimpleNamespace(
            last_solver_ids=[recent[-1], *core, *challenger],
            last_index={"reasons": {}}, config=VNextConfig(),
            graph=SimpleNamespace(missing_solver_pairs=lambda *a, **kw: [],
                conservative_nash=lambda ids, **kw: {x: 1 / len(ids) for x in ids}),
            archive_index=SimpleNamespace(rebuild=lambda *a, **kw: None),
            consecutive_milestones_without_admission=5,
            log=SimpleNamespace(append=lambda *a, **kw: None),
            persist=lambda: None, state_dict=lambda: {})
        self.adapter.controller = controller
        self.adapter._complete_active_solver_game = lambda *a, **kw: []
        trainer.cfg.milestone_period = 500
        return recent[-1], core, challenger, candidate

    def test_real_full_activation_preserves_retained_stats_and_publishes_nash(self):
        current, core, challenger, candidate = self.activation_fixture()
        old = {e.get("archive_id"): dict(e) for e in self.pool.active_entries()}
        self.assertTrue(self.adapter.activate_post_side_candidate(self.trainer,
            archive_id=candidate, current_id=current, iteration=6000))
        self.assertEqual(self.pool.role_counts(), {"latest": 1, "recent": 4, "core": 16, "challenger": 3})
        self.assertEqual(len(self.adapter.controller.last_solver_ids), 20)
        for entry in self.pool.active_entries():
            identity = entry.get("archive_id")
            if identity in old:
                self.assertEqual((entry["id"], entry["games"], entry["ema"]),
                    (old[identity]["id"], old[identity]["games"], old[identity]["ema"]))
            if entry["role"] in {"core", "challenger"}:
                self.assertAlmostEqual(entry["nash_mass"], 1 / 20)
                self.assertEqual(self.trainer.archive.records[identity]["metrics"]["active_role"], entry["role"])
        self.assertEqual(self.adapter.controller.consecutive_milestones_without_admission, 5)

    def test_underfilled_activation_keeps_every_incumbent_when_eligible(self):
        current, core, challenger, candidate = self.activation_fixture(core_count=14)
        self.assertTrue(self.adapter.activate_post_side_candidate(self.trainer,
            archive_id=candidate, current_id=current, iteration=6000))
        active = {e.get("archive_id") for e in self.pool.active_entries()}
        self.assertTrue(set(core + challenger + [candidate]) <= active)
        self.assertEqual(self.pool.role_counts(), {"latest": 1, "recent": 4, "core": 15, "challenger": 3})

    def test_real_activation_refresh_failure_restores_resident_stats(self):
        current, core, challenger, candidate = self.activation_fixture()
        old = {e["id"]: (e["role"], e["games"], e["ema"], e["nash_mass"]) for e in self.pool.entries}
        old_solver = list(self.adapter.controller.last_solver_ids)
        with patch.object(self.trainer, "_refresh_weights", side_effect=RuntimeError("injected refresh failure")):
            self.assertFalse(self.adapter.activate_post_side_candidate(self.trainer,
                archive_id=candidate, current_id=current, iteration=6000))
        self.assertEqual(old_solver, self.adapter.controller.last_solver_ids)
        self.assertEqual(old, {e["id"]: (e["role"], e["games"], e["ema"], e["nash_mass"]) for e in self.pool.entries})
        self.assertEqual(self.trainer.archive.records[candidate]["metrics"]["solver_pending_reason"], "roster_commit_failed")


if __name__ == "__main__":
    unittest.main()
