"""Opt-in original-20k opponent migration, independent of Main warm start."""
import copy
import hashlib
import os
import time
import shutil
from pathlib import Path
import torch

MAPPING = torch.cat((torch.arange(164), torch.arange(194,214)))

def convert_bundle(bundle):
    out = copy.deepcopy(bundle)
    for key in ('actor_logits.0.weight','critic.0.weight'):
        value = bundle['model'].get(key)
        if value is None or value.ndim != 2 or value.shape[1] != 184:
            raise ValueError('expected original 184D MLP input layers')
        widened = value.new_zeros((value.shape[0],214))
        widened[:,MAPPING] = value
        out['model'][key] = widened
    for key, neutral in (('mean',0.),('var',1.)):
        value = bundle['norm'][key]
        if value.shape != (184,): raise ValueError('expected 184D normalization')
        widened = value.new_full((214,),neutral)
        widened[MAPPING] = value
        out['norm'][key] = widened
    if not all(torch.isfinite(v).all() for v in out['model'].values()):
        raise ValueError('nonfinite legacy model')
    return out

def equal_tree(a,b):
    if torch.is_tensor(a): return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def finish_pending(trainer,save_path,transition):
    trainer.legacy_league_transition = copy.deepcopy(transition)
    if not transition or transition.get('status')!='pending': return False
    if not trainer.checkpoint_safe: raise RuntimeError('requires committed iteration')
    path=Path(transition['artifact'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=transition['sha256']:
        raise RuntimeError('legacy migration artifact changed')
    artifact=torch.load(path,map_location='cpu',weights_only=False)
    if artifact['iteration']!=trainer.iteration: raise RuntimeError('migration iteration mismatch')
    adapter=trainer.vnext_milestone_adapter; controller=adapter.controller
    if not controller.may_mutate_training: raise RuntimeError('league is frozen')
    started=time.perf_counter()
    main=copy.deepcopy(trainer.model.state_dict())
    norm=copy.deepcopy(trainer.norm.state_dict())
    actor=copy.deepcopy(trainer.actor_opt.state_dict()); critic=copy.deepcopy(trainer.critic_opt.state_dict())
    # No checkpoint is published until the whole replacement is valid. On an
    # exception startup aborts; the original pending checkpoint remains usable.
    trainer.archive.load_state_dict(artifact['archive'])
    clean=copy.deepcopy(artifact['controller'])
    clean['decision_log_sha256']=controller.log.last_hash
    clean['wallclock_budget']=controller.budget.state_dict()
    controller.load_state_dict(clean)
    trainer.exploiter_history=copy.deepcopy(artifact['exploiter_history'])
    from .league_vnext.profile_stats import empty_profile_stats
    trainer._profile_bandit=empty_profile_stats()
    trainer._heldout_audit_cursor=0
    trainer._vnext_probe_observations=None
    trainer._vnext_probe_version=None
    trainer._vnext_previous_probe_probs=None
    current=trainer._archive_current('milestone_main',metrics={'legacy_league_transplant_main':True})
    strategic=[e['archive_id'] for e in artifact['pool'] if e['role'] in ('core','challenger')]
    selected=[current,*strategic]
    if len(selected)!=20 or len(set(selected))!=20: raise RuntimeError('expected 20 unique solver policies')
    # Bootstrap only: bounded complete prefixes, not the ordinary admission
    # path. Never raises the 48-edge admission cap and never publishes prefixes.
    queries=0
    for size in range(2,len(selected)+1):
        completed=adapter._complete_active_solver_game(trainer,policy_ids=selected[:size],
            iteration=trainer.iteration,current_id=current,candidate_ids=set())
        queries+=len(completed)
        print(f'[legacy-league] payoff {queries}/190 complete; prefix={size}/20',flush=True)
    p=controller.config.payoff_graph
    solution=controller.graph.conservative_nash(selected,minimum_blocks=p.solver_paired_blocks,
        evaluator_protocol=p.evaluator_protocol,scenario_bank_versions=p.solver_compatible_scenario_banks)
    pool=copy.deepcopy(artifact['pool'])
    latest=next(e for e in pool if e['role']=='latest')
    latest.update(model=main,norm=norm,created_iteration=trainer.iteration,archive_id=None)
    trainer.pool.load_state_dicts(pool,next_id=artifact['pool_next_id'])
    trainer._latest_clone_iteration=trainer.iteration
    trainer._last_milestone_archive_id=current
    trainer._recent_archive_ids=[e['archive_id'] for e in pool if e['role']=='recent']
    controller.last_solver_ids=selected
    controller.last_index={'solver_eligible_ids':selected,'selected_ids':selected}
    for identity,record in trainer.archive.records.items(): record['nash_mass']=float(solution.get(identity,0.))
    adapter.sync_membership_metadata(trainer,solver_ids=selected,nash_override=solution)
    adapter.assert_no_stranded_probationary(trainer)
    assert len(trainer.pool.active_entries())==24
    assert {e['archive_id'] for e in trainer.pool.active_entries() if e['role']!='latest'}==set(artifact['active_ids'])
    for a,b in ((main,trainer.model.state_dict()),(norm,trainer.norm.state_dict()),
                (actor,trainer.actor_opt.state_dict()),(critic,trainer.critic_opt.state_dict())):
        if not equal_tree(a,b): raise RuntimeError('migration changed the learner')
    # Reset episodes whose stable opponent IDs belong to the discarded league.
    trainer._refresh_weights()
    trainer._reset_env_state()
    trainer._refresh_weights()
    controller.archive_index.rebuild(trainer.archive.state_dict(),controller.graph,iteration=trainer.iteration)
    controller.budget.add('evaluation',time.perf_counter()-started)
    trainer.legacy_league_transition.update(status='complete',new_main_archive_id=current,
        solver_ids=selected,payoff_queries=queries,source_archive_count=258,active_count=24,
        learner_and_optimizer_unchanged=True)
    controller.log.append('legacy_league_transplant_completed',trainer.legacy_league_transition,iteration=trainer.iteration)
    trainer.vnext_control_state=controller.state_dict()
    trainer.save(save_path)
    shutil.copy2(save_path,path.parent/'committed_before_rollout.pt')
    controller.persist()
    print('[legacy-league] complete: original archive/pool restored, Main/Adam preserved',flush=True)
    return True
