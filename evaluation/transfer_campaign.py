"""Frozen 36k transfer evaluation. Training remains stopped; no trainer mutation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'evaluation')]
OUT = ROOT/'artifacts/evaluations/transfer_36000_20260913'
OLD = ROOT/'artifacts/evaluations/final_selection_10hz_reduced_20260913'


def seed_rows():
    from native_matchups import independent_seed_bank
    return [dict(game_index=i, seed=int(s), swapped=bool(i%2))
            for i,s in enumerate(independent_seed_bank(26091317,100))]


def policy_obs(obs, width):
    import torch
    if width==214: return obs
    if width==184: return torch.cat((obs[:,:164],obs[:,194:]),dim=1)
    raise ValueError('unsupported observation width')


def initial_bank(rows):
    import os
    import native_baselines as native
    cwd=os.getcwd()
    native.setup()
    from claude_code.env_utils import make_env
    env=make_env(overrides=dict(target_mode='fixed',ownship_control_mode='rl',
        step_ratio=6,max_engage_time=200.,episode_step_limit=2001,min_altitude=304.8,
        randomize_start_side=False,scenario_b_prob=0.),runner_index='transfer_ic')
    result=[]
    try:
        for row in rows:
            env._apply_start_side(row['swapped'],row['swapped'])
            env.reset(seed=row['seed'])
            assert env._initial_scenario_kind=='A'
            requested=[dict(n=s._init_pos_n,e=s._init_pos_e,d=s._init_pos_d,
                heading=s._init_heading,speed=s._init_speed,roll=s._init_roll,pitch=s._init_pitch)
                for s in (env._sim,env._target_sim)]
            assert all(abs(s['roll'])<1e-8 and abs(s['pitch'])<1e-8 for s in requested)
            result.append(dict(row,requested=requested,cpu_own=env._sim.get_state()[:9].tolist(),
                               cpu_opp=env._target_sim.get_state()[:9].tolist()))
    finally:
        env.close();os.chdir(cwd)
    return result


def gpu(smoke=False):
    import numpy as np
    import torch
    from tournament import save_json,sha
    from cuda_fdm.ic import build_seed_vector
    from cuda_fdm.rl_env import GpuDogfightVecEnv
    from cuda_fdm.search_eval import load_policy
    from cuda_fdm.ppo_gpu import action_to_env
    from junhwa_policy import load_junhwa_policy
    from policy_io import load_bundle_policy
    torch.set_num_threads(1)
    out=OUT/('gpu_smoke' if smoke else 'gpu')
    rows=initial_bank(seed_rows()[:2] if smoke else seed_rows())
    spec=make_spec();n=len(rows)
    env=GpuDogfightVecEnv(n,scenario='three_nine',seed=26091317)
    env.max_engage_time_s=2. if smoke else 200.
    seeds=[]
    for row in rows:
        for s in row['requested']:
            seeds.append(build_seed_vector(**env._ic_dict(s['n'],s['e'],s['d'],s['heading'],s['speed'])))
    bank=torch.tensor(np.asarray(seeds),device='cuda',dtype=torch.float64)
    save_json(out/'manifest.json',dict(spec=spec,initial_conditions=rows,smoke=smoke,
        runner_sha256=sha(__file__),note='Same requested IC; DLL may expose one initialized physics tick.'))
    models={}
    for m in spec['models']:
        if m['kind']=='junhwa_actor': pair=load_junhwa_policy(m['path'],'cuda')
        elif m['kind']=='legacy184_bundle': pair=load_bundle_policy(m['path'],'cuda',m['legacy_min_altitude_m'])
        else: pair=load_policy(m['path'],'cuda')
        models[m['name']]=(pair,m['kind'])
    pairs=spec['gpu_pairs']
    if smoke: pairs=[p for p in pairs if p[0]=='gylee_20000']+[['gylee_36000','junhwa_final']]
    with torch.inference_mode():
        for a,b in pairs:
            dest=out/f'{a}__{b}.json'
            if dest.exists(): continue
            env.reset(stagger=False)
            env.sim.load_seed(bank);env.obr.reset_all();env.obr.kernel_init_reward_state(env.sim.states)
            obs=env.obr.kernel_build_obs(env.sim.states).view(n,2,214)
            initial=env.state9().cpu().numpy()
            for i,row in enumerate(rows):
                for side in (0,1):
                    s=row['requested'][side]
                    np.testing.assert_allclose(initial[i,side,:3],[s['n'],s['e'],s['d']],atol=.1,rtol=0)
                    assert abs(np.linalg.norm(initial[i,side,6:9])-s['speed'])<1.
            selected=[models[a],models[b]]
            hidden=[p[0][0].initial_state(n,'cuda') for p in selected]
            starts=torch.ones(n,device='cuda');finished=torch.zeros(n,dtype=torch.bool,device='cuda')
            records=[None]*n
            for step in range(1,(20 if smoke else 2000)+4):
                controls=[]
                for side,((model,norm),kind) in enumerate(selected):
                    x=obs[:,side]
                    if kind=='legacy184_bundle':
                        x=x.clone()
                        # CPU legacy provider stores pre-mapping throttle history.
                        for age in range(min(step-1,5)): x[:,194+age*4+3]=2*x[:,194+age*4+3]-1
                    if norm is not None: x=norm.normalize(policy_obs(x,norm.mean.numel()))
                    actions,hidden[side]=model.act(x,hidden[side],starts,sample=False)
                    controls.append(action_to_env(actions,model.num_bins))
                obs,reward,done,info=env.step(torch.stack(controls,dim=1))
                hp=info['terminal_hp'];alt=info['terminal_alt_m']
                active=~finished
                assert bool(info['terminal_state_finite'][active].all())
                assert bool(torch.isfinite(hp[active]).all() & torch.isfinite(alt[active]).all())
                for i in (done & active).nonzero().flatten().cpu().tolist():
                    h0,h1=hp[i].cpu().tolist();z0,z1=alt[i].cpu().tolist()
                    d0=h0<=0 or z0<304.8;d1=h1<=0 or z1<304.8
                    score=float(d1) if d0!=d1 else .5 if d0 else 1. if h0>h1+1e-9 else 0. if h0<h1-1e-9 else .5
                    records[i]=dict(rows[i],score=score,damage_diff=h0-h1,hp=h0,target_hp=h1,
                        crash=z0<304.8,crash_b=z1<304.8,destroyed=h0<=0,destroyed_b=h1<=0,
                        duration_sec=step*.1,initial_gpu=initial[i].tolist())
                finished |= done;starts=done.float()
                if bool(finished.all()): break
            assert all(r is not None for r in records),'unfinished GPU games'
            save_json(dest,dict(left=a,right=b,complete=True,records=records,
                wins=sum(r['score']==1 for r in records),losses=sum(r['score']==0 for r in records),
                draws=sum(r['score']==.5 for r in records)))
            print(f'GPU {a} vs {b}: complete {n}',flush=True)


def make_spec():
    old=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    candidates=['gylee_20000','gylee_23000','gylee_36000']
    opponents=['junhwa_final','junhwa_grid_3L749_gru_last','junhwa_wide1_v2',
        'junhwa_exploiter_survive_v1','junhwa_exploiter_survive_15500',
        'submission4499','cutoff','cutoff_10hz','hard_deck_dive','gylee_67000','gylee_45000']
    models=[dict(m) for m in old['manifest']['models']
            if m['name'] in candidates+opponents]
    models.append(dict(name='gylee_36000',kind='checkpoint',
        path=str(ROOT.parent/'mixed20k_resume_20260911/main/iter_36000.pt')))
    for iteration in (67000,45000):
        models.append(dict(name=f'gylee_{iteration}',kind='checkpoint',path=str(
            ROOT/f'artifacts/models/rl/3-9/completed_70k/three_nine_iter_{iteration}.pt')))
    for m in models: m['decision_hz']=10
    cpu=[[a,b] for a in candidates for b in opponents]
    gpu=[[a,b] for a,b in cpu if b not in ('cutoff','cutoff_10hz','hard_deck_dive')]
    return dict(scenario='three_nine',games_per_pair=100,seed=26091317,
        action_mode='deterministic',models=models,cpu_pairs=cpu,gpu_pairs=gpu,
        seed_protocol='100_unique_seeds_balanced_sides_no_mirroring',rl_hz=10)


def cpu(smoke=False):
    import final_selection as runner
    from tournament import save_json,sha
    spec=make_spec()
    runner.CANDIDATES=list(dict.fromkeys(a for a,b in spec['cpu_pairs']))
    runner.OPPONENTS=list(dict.fromkeys(b for a,b in spec['cpu_pairs']))
    runner.make_spec=lambda:spec
    out=OUT/('cpu_smoke' if smoke else 'cpu')
    # Reuse only unchanged code/protocol/model identities. Original results stay intact.
    if not smoke:
        old=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
        assert old['rl_hz']==10 and old['games']==100 and old['manifest']['seed']==spec['seed']
        assert all(sha(ROOT/p)==digest for p,digest in old['hashes'].items())
        old_models={m['name']:m for m in old['manifest']['models']}
        for a,b in spec['cpu_pairs']:
            if [a,b] not in old['pairs']: continue
            for name in (a,b):
                m=next((m for m in spec['models'] if m['name']==name),None)
                if m is not None:
                    source=Path(m['path'])
                    if source.is_dir(): source=source/'policy_weights.pkl.gz'
                    assert sha(source)==old_models[name]['sha256']
            source=OLD/f'{a}__{b}'
            if source.exists(): shutil.copytree(source,out/source.name,dirs_exist_ok=True)
        save_json(OUT/'cpu_reuse.json',dict(source=str(OLD),protocol_verified=True))
    sys.argv=['final_selection','--output',str(out),'--workers','7']+(['--smoke'] if smoke else [])
    runner.main()


def prepare():
    import torch
    from tournament import save_json,sha,prepare as manifest
    checkpoint=ROOT.parent/'mixed20k_resume_20260911/main/iter_36000.pt'
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    emas={str(p['archive_id']):float(p['ema']) for p in c['pool'] if p.get('archive_id') in (439,440)}
    assert c['iteration']==36000 and len(emas)==2 and min(emas.values())>=.5
    save_json(OUT/'spec.json',make_spec())
    save_json(OUT/'receipt.json',dict(iteration=36000,ema=emas,
        ema_source='post-milestone checkpoint (may incorporate league evaluation)',
        checkpoint_sha256=sha(checkpoint),training_stopped=True,
        manifest=manifest(OUT/'spec.json'),seeds=seed_rows()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','cpu','gpu'])
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='cpu': cpu(args.smoke)
    else: gpu(args.smoke)
