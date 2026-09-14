"""Paired head-on rate comparison using the actual submission provider."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'evaluation')]


def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def play(task):
    left,right,seed,block,output,seconds=task[:6]
    options=task[6] if len(task)>6 else {}
    scenario=options.get('scenario','headon')
    import native_baselines as native
    native.setup()
    import numpy as np
    from claude_code.env_utils import make_env
    from submission.decision_rate import DecisionRateProvider
    from submission.frozen_policy import load
    a=DecisionRateProvider(load(left['path']),None,left['hz'])
    b=DecisionRateProvider(load(right['path']),None,right['hz'])
    env=make_env(overrides=dict(target_mode='fixed',ownship_control_mode='rl',
        step_ratio=6,max_engage_time=seconds,episode_step_limit=2001,
        min_altitude=304.8,randomize_start_side=False,scenario_b_prob=1. if scenario=='headon' else 0.,
        start_headon_distance_ft=5539/.3048,artifacts_dir=str(Path(output)/'sim')),
        runner_index=f'rate_{os.getpid()}')
    env._ownship_action_provider=a; env._target_action_provider=b
    records=[]
    try:
        for swapped in options.get('sides',(False,True)):
            env._apply_start_side(swapped,swapped);env.reset(seed=seed)
            if env._initial_scenario_kind!=('B' if scenario=='headon' else 'A'): raise ValueError('wrong scenario')
            initial_a=env._sim.get_state()[:9].tolist();initial_b=env._target_sim.get_state()[:9].tolist()
            if scenario=='headon':
                np.testing.assert_allclose(np.linalg.norm(np.asarray(initial_a[:3])-initial_b[:3]),5539,atol=.03,rtol=0)
            if swapped and records:
                np.testing.assert_allclose(initial_a,records[0]['initial_b'],atol=1e-8,rtol=0)
                np.testing.assert_allclose(initial_b,records[0]['initial_a'],atol=1e-8,rtol=0)
            began=time.perf_counter()
            for step in range(1,2002):
                _,_,term,trunc,info=env.step(np.zeros(4,dtype=np.float32))
                if term or trunc:break
            else: raise RuntimeError('episode timeout missing')
            hp=float(info['ownship_health']);hp_b=float(info['target_health'])
            alt=-float(env._sim.get_state()[2]);alt_b=-float(env._target_sim.get_state()[2])
            if not np.isfinite([hp,hp_b,alt,alt_b]).all(): raise ValueError('invalid terminal')
            da=hp<=0 or alt<304.8;db=hp_b<=0 or alt_b<304.8
            score=float(db) if da!=db else .5 if da else 1. if hp>hp_b+1e-9 else 0. if hp<hp_b-1e-9 else .5
            record=dict(block=block,seed=seed,swapped=swapped,score=score,hp=hp,hp_b=hp_b,
                hp_diff=hp-hp_b,crash=alt<304.8,crash_b=alt_b<304.8,initial_a=initial_a,initial_b=initial_b,
                duration_sec=step*.1,wall_sec=time.perf_counter()-began,
                frames_a=a.calls,frames_b=b.calls,decisions_a=a.decisions,decisions_b=b.decisions,
                inference_p99_ms_a=float(np.percentile(a.latencies,99)*1000),
                inference_p99_ms_b=float(np.percentile(b.latencies,99)*1000))
            for p in (a,b):
                if p.calls!=step*6 or p.decisions!=(p.calls-1)//(60//p.hz)+1:
                    raise ValueError('actual inference cadence mismatch')
            records.append(record)
        save(Path(output)/f'block_{block:03}.json',dict(records=records,complete=True))
        return records
    finally: env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=7)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve()
    import numpy as np
    import torch
    from submission.frozen_policy import export_checkpoint
    torch.set_num_threads(1)
    paths={'46k':ROOT/'artifacts/models/rl/headon/completed_20k_resume/headon_iter_46000.pt',
           '20k':ROOT.parent/'mixed20k_resume_20260911/main/initial_20000.pt'}
    model_meta={}
    for name,path in paths.items():
        q=out/'models'/f'{name}.pt'
        payload=export_checkpoint(path,q)
        model_meta[name]=dict(path=str(q),iteration=payload['iteration'],source_sha256=payload['source_sha256'])
    pairs=[('46k',10,'46k',60),('20k',60,'46k',10),('20k',10,'46k',10)]
    manifest=dict(models=model_meta,pairs=pairs,blocks=1 if args.smoke else 50,seed=260912,
        seconds=2 if args.smoke else 200,headon_distance_m=5539,action_mode='argmax',
        observation='60Hz physical integration; derivatives/history over 0.1s',
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['submission/decision_rate.py','submission/frozen_policy.py','evaluation/decision_rate_match.py','claude_code/my_observation.py']})
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):
        raise ValueError('manifest changed; use new output directory')
    save(out/'manifest.json',manifest)
    tasks=[]
    seeds=np.random.default_rng(260912).integers(0,2**31-1,size=manifest['blocks'])
    for an,ah,bn,bh in pairs:
        folder=out/f'{an}_{ah}hz__{bn}_{bh}hz'
        for block,seed in enumerate(seeds):
            if not (folder/f'block_{block:03}.json').exists():
                tasks.append((dict(model_meta[an],hz=ah),dict(model_meta[bn],hz=bh),int(seed),block,str(folder),manifest['seconds']))
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        pending=[executor.submit(play,t) for t in tasks]
        for n,f in enumerate(as_completed(pending),1):
            f.result(); print(f'completed blocks {n}/{len(tasks)}',flush=True)
            save(out/'progress.json',dict(completed_new_blocks=n,total_new_blocks=len(tasks)))
    report=[]
    for an,ah,bn,bh in pairs:
        rows=[]
        for block in range(manifest['blocks']):
            rows+=json.loads((out/f'{an}_{ah}hz__{bn}_{bh}hz'/f'block_{block:03}.json').read_text(encoding='utf-8'))['records']
        scores=np.array([r['score'] for r in rows]); blocks=scores.reshape(-1,2).mean(1)
        boot=np.random.default_rng(42).choice(blocks,(10000,len(blocks))).mean(1)
        report.append(dict(left=f'{an}_{ah}hz',right=f'{bn}_{bh}hz',games=len(rows),
            wins=int((scores==1).sum()),losses=int((scores==0).sum()),draws=int((scores==.5).sum()),
            score=float(scores.mean()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
            crash=sum(r['crash'] for r in rows),crash_b=sum(r['crash_b'] for r in rows),
            hp_diff=float(np.mean([r['hp_diff'] for r in rows])),records=rows))
    save(out/'report.json',dict(complete=True,matches=report))
    print(json.dumps([{k:v for k,v in r.items() if k!='records'} for r in report]),flush=True)


if __name__=='__main__':main()
