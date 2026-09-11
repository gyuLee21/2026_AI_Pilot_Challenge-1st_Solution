"""One-shot external opponents, using existing payoff/Nash/atomic roster primitives.

No new pool lifecycle: these are user-authorized probationary challengers.
Up to three small payoff proposals are evaluated first; only the final roster commits.
Main weights, optimizers and rollout state are never loaded from these policies.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["scenario"] != "headon" or int(plan["iteration"]) < 1 or not 1 <= len(plan["models"]) <= 3:
        raise ValueError("unexpected scheduled legacy import contract")
    if len({m["sha256"] for m in plan["models"]}) != len(plan["models"]):
        raise ValueError("duplicate legacy policies")
    for model in plan["models"]:
        model["path"] = str((path.parent / model["path"]).resolve())
        if hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("legacy policy hash mismatch")
    return plan


def proposals(current, cores, old_challengers, imported):
    if len(cores) > 16 or not 1 <= len(imported) <= 3:
        raise ValueError("invalid batch layout")
    result = []
    for count in range(1, len(imported) + 1):
        ids = [current, *cores, *imported[:count], *old_challengers[:3-count]]
        if len(ids) > 20 or len(set(ids)) != len(ids):
            raise ValueError("invalid proposed solver membership")
        result.append(ids)
    return result


def completed(trainer, plan):
    return sum(r.get("metrics", {}).get("legacy_import_batch_completed") == plan["batch_id"]
               for r in trainer.archive.records.values()) == len(plan["models"])


def apply_if_due(trainer, plan, save_path):
    iteration = int(trainer.iteration)
    if iteration < plan["iteration"] or completed(trainer, plan):
        return False
    if trainer.env.scenario != "headon" or trainer.env.scenario_b_prob != 1.0:
        raise ValueError("scheduled import requires a head-on-only environment")
    if not trainer.checkpoint_safe:
        raise ValueError("legacy import requires a committed boundary")
    adapter = trainer.vnext_milestone_adapter
    controller = adapter.controller
    if not controller.may_mutate_training:
        raise ValueError("legacy import cannot bypass a health/release freeze")
    # Always preserve a usable checkpoint before beginning the external transaction.
    trainer.save(save_path)
    backup = Path(save_path).with_name(f"checkpoint_pre_legacy_import_{plan['iteration']}.pt")
    if not backup.exists():
        shutil.copy2(save_path, backup)
    start = time.perf_counter()
    current = int(trainer._last_milestone_archive_id)
    active = trainer.pool.active_entries()
    cores = [int(e["archive_id"]) for e in active if e["role"] == "core"]
    old_challengers = [int(e["archive_id"]) for e in active if e["role"] == "challenger"]
    imported = []
    for model in plan["models"]:
        matches = [int(r["id"]) for r in trainer.archive.records.values()
                   if r.get("metrics", {}).get("legacy_import_sha256") == model["sha256"]]
        if len(matches) > 1:
            raise ValueError("duplicate external archive entry")
        if matches:
            imported.extend(matches)
            continue
        if hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("legacy policy changed before admission")
        bundle = torch.load(model["path"], map_location="cpu", weights_only=False)
        identity = trainer.archive.add(bundle["model"], bundle["norm"],
            kind="imported_legacy_main", iteration=iteration, profile="standard",
            admitted=True, payoff_eligible=True, metrics={
                "admission_rule": "user_authorized_legacy_import_v1", "admission_route": "external_legacy",
                "legacy_import_sha256": model["sha256"], "legacy_source_iteration": model["iteration"],
                "legacy_source_provenance": bundle["provenance"],
                "solver_status": "pending", "solver_pending_since_iteration": iteration,
                "solver_pending_reason": "external_import_in_progress", "active_role": None})
        trainer.archive.records[identity]["admission_status"] = "probationary"
        imported.append(identity)
    solver_proposals = proposals(current, cores, old_challengers, imported)
    query_counts = []
    # Each proposal uses the existing <=48-edge check; no intermediate roster mutation.
    for selected in solver_proposals:
        queries = adapter._complete_active_solver_game(trainer, policy_ids=selected,
            iteration=iteration, current_id=current, candidate_ids=set(imported))
        query_counts.append(len(queries))
    selected = solver_proposals[-1]
    p = controller.config.payoff_graph
    solution = controller.graph.conservative_nash(selected, minimum_blocks=p.solver_paired_blocks,
        evaluator_protocol=p.evaluator_protocol, scenario_bank_versions=p.solver_compatible_scenario_banks)
    pool_snapshot = adapter._pool_transaction_snapshot(trainer.pool)
    record_snapshot = adapter._record_transaction_snapshot(trainer.archive.records)
    old_solver, old_index = list(controller.last_solver_ids), copy.deepcopy(controller.last_index)
    try:
        adapter.refresh_roster(trainer, iteration=iteration, current_id=current,
            forced_challenger_ids=tuple(imported), selected_ids=selected, nash_override=solution)
        for record in trainer.archive.records.values():
            record["nash_mass"] = 0.0
        for identity, mass in solution.items():
            trainer.archive.records[int(identity)]["nash_mass"] = float(mass)
        controller.last_solver_ids = list(selected)
        controller.last_index["solver_eligible_ids"] = list(selected)
        for identity in imported:
            trainer.archive.records[identity]["metrics"].update(
                legacy_import_batch_completed=plan["batch_id"], league_entry_recorded=True,
                post_side_entry_route="external_legacy", post_side_activated_at_iteration=iteration)
        adapter.sync_membership_metadata(trainer, solver_ids=selected, nash_override=solution)
        if not set(imported).issubset({int(e["archive_id"]) for e in trainer.pool.active_entries() if e["role"] == "challenger"}):
            raise ValueError("all requested legacy challengers were not seated")
        if set(cores) != {int(e["archive_id"]) for e in trainer.pool.active_entries() if e["role"] == "core"}:
            raise ValueError("external import unexpectedly evicted a core")
        if adapter.stranded_probationary_ids(trainer):
            raise ValueError("stranded admission after external import")
        trainer._refresh_weights()
        controller.archive_index.rebuild(trainer.archive.state_dict(), controller.graph, iteration=iteration)
        controller.budget.add("evaluation", time.perf_counter() - start)
        controller.log.append("user_authorized_legacy_challenger_import", {
            "batch_id": plan["batch_id"], "archive_ids": imported, "solver_ids": selected,
            "payoff_queries_per_proposal": query_counts, "source_scenario": plan.get("source_scenario", "mixed"),
            "evaluation_scenario": "headon"}, iteration=iteration)
        trainer.archive.persist()
        controller.persist()
        trainer.vnext_control_state = controller.state_dict()
        trainer.save(save_path)
        imported_checkpoint = Path(save_path).with_name(f"checkpoint_post_legacy_import_{plan['iteration']}.pt")
        if not imported_checkpoint.exists():
            shutil.copy2(save_path, imported_checkpoint)
    except Exception:
        adapter._restore_pool_transaction(trainer.pool, pool_snapshot)
        adapter._restore_record_transaction(trainer.archive.records, record_snapshot)
        controller.last_solver_ids, controller.last_index = old_solver, old_index
        trainer._refresh_weights()
        raise
    print(f"[legacy-import] main={iteration} challengers={imported} scenario=headon queries={query_counts}", flush=True)
    return True
