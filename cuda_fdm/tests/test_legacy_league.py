import torch
import pytest
from pathlib import Path
from cuda_fdm.legacy_league import convert_bundle, MAPPING

def test_semantic_mapping_and_nonmutation():
    torch.manual_seed(4)
    model = {k: torch.randn(8,184) for k in ('actor_logits.0.weight','critic.0.weight')}
    model['actor_logits.0.bias']=torch.randn(8)
    norm={'mean':torch.randn(184),'var':torch.rand(184)+.1,'count':torch.tensor(10.)}
    out=convert_bundle({'model':model,'norm':norm})
    x=torch.randn(30,184); y=torch.randn(30,214); y[:,MAPPING]=x
    for k in ('actor_logits.0.weight','critic.0.weight'):
        assert out['model'][k].shape==(8,214)
        assert torch.equal(out['model'][k][:,MAPPING],model[k])
        assert torch.count_nonzero(out['model'][k][:,164:194])==0
        torch.testing.assert_close(x@model[k].T,y@out['model'][k].T)
        assert model[k].shape==(8,184)
    assert torch.equal(out['norm']['mean'][MAPPING],norm['mean'])
    assert torch.equal(out['norm']['var'][MAPPING],norm['var'])
    assert torch.all(out['norm']['var'][164:194]==1)

def test_reject_double_conversion():
    with pytest.raises(ValueError):
        convert_bundle({'model':{'actor_logits.0.weight':torch.zeros(8,214)},'norm':{}})

def test_completed_transition_noop():
    from types import SimpleNamespace
    from cuda_fdm.legacy_league import finish_pending
    trainer=SimpleNamespace()
    assert finish_pending(trainer,'unused',{'status':'complete'}) is False

@pytest.mark.parametrize('fail_at',[None,4])
def test_bootstrap_no_partial_pool_commit(tmp_path,fail_at):
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    import hashlib
    from cuda_fdm.legacy_league import finish_pending
    pool=[dict(role='latest',archive_id=None)]
    pool += [dict(role='recent',archive_id=i) for i in range(1,5)]
    pool += [dict(role='core',archive_id=i) for i in range(5,21)]
    pool += [dict(role='challenger',archive_id=i) for i in range(21,24)]
    model=torch.nn.Linear(1,1)
    actor=torch.optim.Adam(model.parameters()); critic=torch.optim.Adam(model.parameters())
    graph=NS(conservative_nash=Mock(return_value={i:1/20 for i in range(5,25)}))
    ctl=NS(may_mutate_training=True,log=NS(last_hash=None,append=Mock()),
        budget=NS(state_dict=lambda:{},add=Mock()),load_state_dict=Mock(),graph=graph,
        config=NS(payoff_graph=NS(solver_paired_blocks=64,evaluator_protocol='p',solver_compatible_scenario_banks=('b',))),
        archive_index=NS(rebuild=Mock()),state_dict=lambda:{},persist=Mock())
    entry_pool=NS(load_state_dicts=Mock(),active_entries=lambda:pool)
    archive=NS(load_state_dict=Mock(),records={i:{'metrics':{}} for i in range(1,25)},state_dict=lambda:{})
    trainer=NS(checkpoint_safe=True,iteration=10,model=model,norm=model,actor_opt=actor,critic_opt=critic,
        archive=archive,pool=entry_pool,_archive_current=lambda *a,**k:24,
        _refresh_weights=Mock(),_reset_env_state=Mock())
    seen=set();sizes=[]
    def complete(*a,policy_ids,**kwargs):
        assert not entry_pool.load_state_dicts.called
        if len(policy_ids)==fail_at: raise RuntimeError('injected evaluation failure')
        import itertools
        missing=set(itertools.combinations(policy_ids,2))-seen
        assert len(missing)<=48
        seen.update(missing);sizes.append(len(missing)); return list(missing)
    trainer.vnext_milestone_adapter=NS(controller=ctl,_complete_active_solver_game=complete,
        sync_membership_metadata=Mock(),assert_no_stranded_probationary=Mock())
    artifact={'iteration':10,'archive':{},'controller':{},'exploiter_history':[],
        'pool':pool,'pool_next_id':30,'active_ids':list(range(1,24))}
    path=tmp_path/'artifact.pt';torch.save(artifact,path)
    transition={'status':'pending','artifact':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    target=tmp_path/'checkpoint.pt';trainer.save=Mock(side_effect=lambda p:Path(p).write_bytes(b'checkpoint'))
    if fail_at:
        with pytest.raises(RuntimeError,match='injected'): finish_pending(trainer,target,transition)
        assert not trainer.save.called and not entry_pool.load_state_dicts.called
        assert trainer.legacy_league_transition['status']=='pending'
    else:
        assert finish_pending(trainer,target,transition)
        assert sizes==list(range(1,20)) and len(seen)==190
        assert trainer.legacy_league_transition['status']=='complete'
        assert trainer.legacy_league_transition['learner_and_optimizer_unchanged']
        assert trainer.save.call_count==1
