"""Explicit distance identity; existing admission/solver rules remain unchanged."""
from dataclasses import replace
import math
import time

def finish_pending_rebaseline(trainer, save_path):
    transition = getattr(trainer,'headon_distance_transition',None)
    if not transition or transition.get('status') != 'pending':
        return False
    if not save_path:
        raise ValueError('distance rebaseline requires a checkpoint save path')
    adapter = trainer.vnext_milestone_adapter
    controller = adapter.controller
    ids = list(controller.last_solver_ids)
    if not ids: raise ValueError('distance migration has no saved solver population')
    before = [(e.get('archive_id'),e.get('role')) for e in trainer.pool.entries]
    started = time.perf_counter()
    # All necessary pairs, not merely a new row; existing 48-edge bound applies.
    queries = adapter._complete_active_solver_game(trainer,policy_ids=ids,
        iteration=trainer.iteration,current_id=ids[0],candidate_ids=set())
    p = controller.config.payoff_graph
    solution = controller.graph.conservative_nash(ids,minimum_blocks=p.solver_paired_blocks,
        evaluator_protocol=p.evaluator_protocol,scenario_bank_versions=p.solver_compatible_scenario_banks)
    if before != [(e.get('archive_id'),e.get('role')) for e in trainer.pool.entries]:
        raise RuntimeError('distance rebaseline changed pool membership')
    for identity,record in trainer.archive.records.items():
        record['nash_mass'] = float(solution.get(identity,0.))
    for entry in trainer.pool.entries:
        entry['nash_mass'] = float(solution.get(entry.get('archive_id'),0.))
    trainer._refresh_weights()
    controller.budget.add('evaluation',time.perf_counter()-started)
    transition.update(status='complete',completed_iteration=trainer.iteration,
                      rebaseline_edges=len(queries),solver_ids=ids)
    controller.log.append('headon_distance_changed',dict(transition),iteration=trainer.iteration)
    trainer.vnext_control_state=controller.state_dict()
    try:
        trainer.save(save_path)
    except BaseException:
        transition['status']='pending'
        raise
    controller.persist()
    print(f'[headon-distance] rebaseline complete: {transition}',flush=True)
    return True

def validate_distance(checkpoint, distance):
    saved = float(checkpoint.get('headon_distance_m',3048.))
    if not math.isclose(saved,float(distance),rel_tol=0.,abs_tol=1e-6):
        raise ValueError(f'head-on distance changed {saved}->{distance}; explicit checkpoint/cache migration required')

def distance_payoff_config(config, distance):
    tag = f'headon_{float(distance):g}m_v1'
    return replace(config,screening_scenario_bank='screening_'+tag,
        confirmatory_scenario_bank='confirmatory_'+tag,
        solver_scenario_bank='solver_'+tag,
        solver_compatible_scenario_banks=('solver_'+tag,))
