"""Manual-only mirrored GPU round robin using the existing search evaluator.

No training hooks, model selection or automatic scheduling. Each manifest lists
the desired frozen checkpoints explicitly. Completed pairs survive restart.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def prepare(spec_path, runtime=RUNTIME):
    spec_path=Path(spec_path).resolve()
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    scenario=spec.get('scenario','three_nine')
    if scenario not in ('three_nine','headon'):
        raise ValueError('scenario must be three_nine or headon; mixed is not a tournament')
    distance=float(spec.get('headon_distance_m',5539.))
    if not math.isfinite(distance) or distance<=0:
        raise ValueError('headon_distance_m must be finite and positive')
    if spec.get('action_mode','deterministic') != 'deterministic':
        raise ValueError('this evaluator uses argmax actions; stochastic comparison requires a separate protocol')
    games=spec.get('games_per_pair',50)
    if not isinstance(games,int) or games < 2 or games%2:
        raise ValueError('games_per_pair must be positive and even: 50 = 25 mirrored blocks')
    models=spec.get('models',[])
    if not 2 <= len(models) <= 100:
        raise ValueError('supply 2..100 frozen models; no models are selected automatically')
    participants=[]; names=set(); hashes=set()
    for model in models:
        name=model['name']
        if not isinstance(name,str) or not name.strip() or name in names:
            raise ValueError('model names must be nonempty and unique')
        kind=model.get('kind','checkpoint')
        if kind not in ('checkpoint','legacy184_bundle','junhwa_actor'):
            raise ValueError(f'{name}: a policy adapter is required for non-checkpoint models')
        path=(spec_path.parent/model['path']).resolve()
        source_hash=sha(path/'policy_weights.pkl.gz') if kind=='legacy184_bundle' else sha(path)
        metadata_hash=sha(path/'metadata.json') if kind=='legacy184_bundle' else None
        legacy_floor=float(model['legacy_min_altitude_m']) if kind=='legacy184_bundle' else None
        if legacy_floor is not None and not math.isfinite(legacy_floor):
            raise ValueError('legacy observation altitude floor must be finite')
        config_from=(spec_path.parent/model['config_from']).resolve() if model.get('config_from') else None
        config_hash=sha(config_from) if config_from else None
        identity=(source_hash,config_hash,metadata_hash,legacy_floor)
        if identity in hashes: raise ValueError('duplicate model artifacts in roster')
        names.add(name); hashes.add(identity)
        participants.append(dict(name=name,path=str(path),sha256=source_hash,kind=kind,
                                 metadata_sha256=metadata_hash,legacy_min_altitude_m=legacy_floor,
                                 config_from=str(config_from) if config_from else None,
                                 config_sha256=config_hash))
        if spec.get('cross_family_only',False):
            family=model.get('family')
            if not isinstance(family,str) or not family.strip():
                raise ValueError('cross-family evaluation requires every model family')
            participants[-1]['family']=family
    return dict(protocol='scenario_manual_round_robin_v2',scenario=scenario,
                headon_distance_m=distance,
                action_mode='deterministic',games_per_pair=games,seed=int(spec.get('seed',710901)),
                cross_family_only=bool(spec.get('cross_family_only',False)),
                models=participants,substeps=6,max_engage_time_s=200.,min_altitude_m=304.8,
                evaluator_sha256=sha(runtime/'cuda_fdm/search_eval.py'),
                runner_sha256=sha(Path(__file__)),
                policy_adapter_sha256=sha(Path(__file__).with_name('policy_io.py')),
                junhwa_adapter_sha256=sha(Path(__file__).with_name('junhwa_policy.py')),
                environment_sha256=sha(runtime/'cuda_fdm/rl_env.py'))

def selected_pairs(manifest):
    models=manifest['models']
    return [(i,j) for i in range(len(models)) for j in range(i+1,len(models))
            if not manifest.get('cross_family_only',False)
            or models[i]['family']!=models[j]['family']]

def rank(manifest,pairs):
    n=len(manifest['models'])
    rows=[dict(name=m['name'],wins=0,draws=0,losses=0,games=0,crashes=0,opponents=0,
               opponent_scores=[]) for m in manifest['models']]
    matrix=[[None]*n for _ in range(n)]
    for pair in pairs:
        i,j=pair['left'],pair['right']; records=pair['result']['records']
        score=sum(r['score'] for r in records)/len(records)
        matrix[i][j]=score; matrix[j][i]=1-score
        for index,flip in ((i,False),(j,True)):
            row=rows[index]; row['opponents']+=1
            row['opponent_scores'].append(1-score if flip else score)
            for r in records:
                value=1-r['score'] if flip else r['score']
                row['wins']+=int(value==1); row['draws']+=int(value==.5); row['losses']+=int(value==0)
                row['games']+=1; row['crashes']+=int(r['crash_b'] if flip else r['crash'])
    for row in rows:
        row['score']=(row['wins']+.5*row['draws'])/row['games'] if row['games'] else None
        row['crash_rate']=row['crashes']/row['games'] if row['games'] else None
        row['worst_opponent_score']=min(row['opponent_scores'],default=None)
        del row['opponent_scores']
    rows.sort(key=lambda r: (-(r['score'] if r['score'] is not None else -1),r['name']))
    return rows,matrix

def run_round_robin(manifest,output,pair_evaluator,stop_file=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('existing results belong to a different roster/configuration/code')
    else: save_json(manifest_path,manifest)
    n=len(manifest['models']); pairs=[]; allowed=set(selected_pairs(manifest)); expected=len(allowed)
    for i in range(n):
        for j in range(i+1,n):
            if (i,j) not in allowed: continue
            if stop_file and Path(stop_file).exists():
                raise InterruptedError('stopped at complete-pair boundary; rerun to resume')
            path=output/'pairs'/f'{i:03d}_{j:03d}.json'
            seed=manifest['seed'] # common mirrored IC bank for all policy pairs
            if path.exists():
                pair=json.loads(path.read_text(encoding='utf-8'))
            else:
                result=pair_evaluator(i,j,seed)
                pair=dict(left=i,right=j,seed=seed,result=result)
            if (pair['left'],pair['right'],pair['seed']) != (i,j,seed):
                raise ValueError('pair identity mismatch')
            records=pair['result']['records']; half=manifest['games_per_pair']//2
            if len(records)!=2*half: raise ValueError('incorrect game count')
            for k,r in enumerate(records):
                if (r['score'] not in (0,.5,1) or r['ic_pair']!=k%half
                    or bool(r['role_swapped'])!=(k>=half)
                    or not isinstance(r['crash'],bool) or not isinstance(r['crash_b'],bool)):
                    raise ValueError('invalid mirrored outcomes')
            if not path.exists(): save_json(path,pair)
            pairs.append(pair)
            save_json(output/'progress.json',dict(completed_pairs=len(pairs),total_pairs=expected,
                                                 scored_games=len(pairs)*2*half,complete=False))
            print(f'[tournament] {len(pairs)}/{expected}: {manifest["models"][i]["name"]} vs {manifest["models"][j]["name"]}',flush=True)
    rankings,matrix=rank(manifest,pairs)
    report=dict(complete=True,rankings=rankings,score_matrix=matrix,
                total_pairs=expected,total_games=expected*manifest['games_per_pair'],
                note='Observed average-score ranking over this roster; cyclic matchups and finite-sample uncertainty remain.')
    if manifest.get('cross_family_only',False):
        families={m['name']:m['family'] for m in manifest['models']}
        report['rankings_by_family']={family:[r for r in rankings if families[r['name']]==family]
                                     for family in sorted(set(families.values()))}
        report['note']='Compare rankings within each family: families face different opponent sets. Excluded matches are unmeasured, not draws.'
    save_json(output/'report.json',report)
    with (output/'rankings.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rankings[0])); writer.writeheader(); writer.writerows(rankings)
    with (output/'score_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['model']+[m['name'] for m in manifest['models']])
        writer.writerows([[m['name']]+matrix[i] for i,m in enumerate(manifest['models'])])
    save_json(output/'progress.json',dict(completed_pairs=expected,total_pairs=expected,
                                         scored_games=report['total_games'],complete=True))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--runtime',type=Path,default=RUNTIME)
    parser.add_argument('--validate-only',action='store_true'); parser.add_argument('--stop-file')
    args=parser.parse_args(); manifest=prepare(args.spec,args.runtime)
    print(json.dumps({'models':len(manifest['models']),'games_per_pair':manifest['games_per_pair'],
                      'total_games':len(selected_pairs(manifest))*manifest['games_per_pair']}))
    if args.validate_only: return
    sys.path.insert(0,str(args.runtime))
    import torch
    from cuda_fdm.search_eval import load_policy,evaluate_pair
    from cuda_fdm.rl_env import GpuDogfightVecEnv
    from policy_io import load_bundle_policy
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.set_num_threads(1)
    output=Path(args.output)
    # Verify immutable run identity BEFORE creating any resolved artifacts.
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text(encoding='utf-8'))!=manifest:
        raise ValueError('output directory belongs to a different tournament')
    policies=[]
    for index,m in enumerate(manifest['models']):
        path=Path(m['path'])
        if m['kind']=='junhwa_actor':
            from junhwa_policy import load_junhwa_policy
            policies.append(load_junhwa_policy(path,'cuda'))
            continue
        if m['kind']=='legacy184_bundle':
            policies.append(load_bundle_policy(path,'cuda',m['legacy_min_altitude_m']))
            continue
        if m['config_from']:
            bundle=torch.load(path,map_location='cpu',weights_only=False)
            template=torch.load(m['config_from'],map_location='cpu',weights_only=False,mmap=True)
            bundle=dict(bundle,cfg=template['cfg'])
            path=output/'resolved_models'/f'{index:03d}.pt'; path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(bundle,path)
        policies.append(load_policy(path,'cuda'))
    env=GpuDogfightVecEnv(manifest['games_per_pair'],scenario=manifest['scenario'],
                          headon_distance_m=manifest['headon_distance_m'],
                          substeps=6,seed=manifest['seed'],device='cuda',
                          min_altitude_m=304.8,max_engage_time_s=200.)
    def evaluate(i,j,seed):
        return evaluate_pair(env,policies[i],policies[j],manifest['scenario'],seed)
    run_round_robin(manifest,output,evaluate,args.stop_file)

if __name__=='__main__': main()
