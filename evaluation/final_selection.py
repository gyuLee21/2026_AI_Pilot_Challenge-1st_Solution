"""CPU submission shortlist: independent seeds, balanced sides, resumable chunks.

The established 10 Hz RL provider and corrected native FDM are reused.
No training hooks. An incomplete/failed game never becomes a draw or exclusion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys

import native_baselines as native
from native_matchups import play, independent_seed_bank
from tournament import prepare, save_json, sha

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ['gylee_20000','gylee_23000','gylee_26000','gylee_30000','gylee_32000']
OPPONENTS = ['junhwa_final','junhwa_grid_3L749_gru_last','junhwa_wide1_v2',
             'junhwa_exploiter_survive_v1','junhwa_exploiter_survive_15500',
             'submission4499','cutoff','cutoff_10hz','hard_deck_dive']


def make_spec():
    folder = ROOT/'configs/evaluation'
    original = json.loads((folder/'local_20k_30k_verified.local.json').read_text())
    extra = json.loads((folder/'local_continuation5.local.json').read_text())
    models = {m['name']:m for m in original['models']+extra['models']}
    selected = CANDIDATES+[n for n in OPPONENTS if n not in native.BASELINES]
    return dict(scenario='three_nine',games_per_pair=100,seed=26091317,
        action_mode='deterministic',models=[models[n] for n in selected],
        cpu_pairs=[[a,b] for a in CANDIDATES for b in OPPONENTS],
        seed_protocol='100_unique_seeds_balanced_sides_no_mirroring',
        rl_hz=10,cutoff_hz={'cutoff':60,'cutoff_10hz':10})


def failure_reason(row):
    if row['score'] != 0: return None
    if row['crash'] and row['destroyed']: return 'crash_and_destroyed'
    if row['crash']: return 'crash'
    if row['destroyed']: return 'destroyed'
    if row.get('end') in ('max time out','episode step limit'): return 'timeout_hp'
    return 'other_hp_or_rule'


def summarize(rows):
    from collections import Counter
    n=len(rows)
    wins=sum(r['score']==1 for r in rows)
    losses=sum(r['score']==0 for r in rows)
    reasons=Counter(failure_reason(r) for r in rows if r['score']==0)
    assert sum(reasons.values())==losses
    return dict(games=n,wins=wins,losses=losses,draws=n-wins-losses,
        win_rate=wins/n,score=sum(r['score'] for r in rows)/n,
        mean_hp_diff=sum(r['damage_diff'] for r in rows)/n,
        own_crashes=sum(r['crash'] for r in rows),opponent_crashes=sum(r['crash_b'] for r in rows),
        loss_reasons=dict(reasons),
        side_wins=[sum(r['score']==1 and r['role_swapped']==s for r in rows) for s in (False,True)],
        side_games=[sum(r['role_swapped']==s for r in rows) for s in (False,True)])


def collect(out, pairs, games, chunk):
    results=[]
    for a,b in pairs:
        rows=[]
        for start in range(0,games,chunk):
            path=out/f'{a}__{b}'/f'chunk_{start:03d}'/'result.json'
            if not path.exists(): continue
            result=json.loads(path.read_text(encoding='utf-8'))
            batch=result['records']
            if (result['left'],result['right'],result['complete'])!=(a,b,True):
                raise ValueError('result identity mismatch')
            if [r['game_index'] for r in batch]!=list(range(start,min(start+chunk,games))):
                raise ValueError('chunk game identity mismatch')
            rows.extend(batch)
        if rows:
            if len({r['seed'] for r in rows})!=len(rows): raise ValueError('repeated seed')
            bank=independent_seed_bank(26091317,games)
            for r in rows:
                if r['seed']!=int(bank[r['game_index']]) or r['role_swapped']!=bool(r['game_index']%2):
                    raise ValueError('seed or starting-side mismatch')
            results.append(dict(left=a,right=b,complete=len(rows)==games,
                                summary=summarize(rows),records=rows))
    complete=len(results)==len(pairs) and all(r['complete'] for r in results)
    rankings=[]
    if complete:
        for name in dict.fromkeys(a for a,b in pairs):
            own=[r for r in results if r['left']==name]
            rows=[row for r in own for row in r['records']]
            rankings.append(dict(name=name,**summarize(rows),
                worst_opponent_win_rate=min(r['summary']['win_rate'] for r in own)))
        rankings.sort(key=lambda r:(r['win_rate'],r['score'],r['worst_opponent_win_rate']),reverse=True)
    report=dict(complete=complete,total_games=sum(len(r['records']) for r in results),
        expected_games=len(pairs)*games,rankings=rankings,results=results,
        note='Observed ranking on this opponent suite, not proof of universal superiority. All RL argmax 10Hz; GRU hidden advances every decision.')
    save_json(out/('report.json' if complete else 'partial_report.json'),report)
    return report


def render(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    lookup={(r['left'],r['right']):r['summary'] for r in report['results']}
    candidates=list(dict.fromkeys(r['left'] for r in report['results']))
    opponents=list(dict.fromkeys(r['right'] for r in report['results']))
    rates=np.array([[lookup[a,b]['win_rate']*100 for b in opponents] for a in candidates])
    fig,ax=plt.subplots(figsize=(23,9))
    im=ax.imshow(rates,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    labels={'junhwa_final':'Junhwa final','junhwa_grid_3L749_gru_last':'Junhwa GRU',
        'junhwa_wide1_v2':'Junhwa wide v2','junhwa_exploiter_survive_v1':'Junhwa survive',
        'junhwa_exploiter_survive_15500':'Junhwa 15500','submission4499':'4499',
        'cutoff':'Cutoff 60Hz','cutoff_10hz':'Cutoff 10Hz','hard_deck_dive':'Hard-deck dive','release_mpc':'Release MPC'}
    ax.set_xticks(range(len(opponents)),[labels.get(n,n) for n in opponents],rotation=25,ha='right')
    ax.set_yticks(range(len(candidates)),[n.replace('gylee_','') for n in candidates])
    for i,a in enumerate(candidates):
        for j,b in enumerate(opponents):
            s=lookup[a,b]
            ax.text(j,i,f"{s['win_rate']:.0%}\n{s['wins']}/{s['losses']}/{s['draws']}\nHP {s['mean_hp_diff']:+.3f}",
                ha='center',va='center',color='white' if rates[i,j]>65 else 'black',fontsize=10)
    ax.set_title('3-9 최종 후보 평가 — RL 10Hz argmax / 대진마다 독립 100시드, 좌우 각 50판',pad=20)
    fig.colorbar(im,ax=ax,label='승률 W/N (%)')
    fig.tight_layout();fig.savefig(out/'final_heatmap.png',dpi=170);plt.close(fig)
    lines=['3-9 CPU FINAL | ALL RL 10Hz argmax | independent seeds | W/L/D | HP = own - opponent',
           'Losses: crash / destroyed / both / timeout HP / other. Both is counted separately.']
    lines+=['='*130,'OVERALL (equal weight per opponent; inspect worst matchup and crash losses too)']
    for r in report['rankings']:
        lines.append(f"{r['name']:16} {r['wins']}/{r['losses']}/{r['draws']}  win={r['win_rate']:.2%}  "
                     f"worst={r['worst_opponent_win_rate']:.2%}  HP={r['mean_hp_diff']:+.4f}  "
                     f"own crashes={r['own_crashes']}/{r['games']}")
    for a in candidates:
        lines+=['='*130,a,'target                              W/L/D       win%      HPdiff   crashL shotL bothL timeL otherL']
        for b in opponents:
            s=lookup[a,b];r=s['loss_reasons']
            lines.append(f"{b:36} {s['wins']:3}/{s['losses']:3}/{s['draws']:3}   {100*s['win_rate']:6.1f}   {s['mean_hp_diff']:+.4f}   "
                         + ' '.join(f'{r.get(k,0):5}' for k in ['crash','destroyed','crash_and_destroyed','timeout_hp','other_hp_or_rule']))
    (out/'final_results.txt').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=7);p.add_argument('--smoke',action='store_true')
    p.add_argument('--report-only',action='store_true');args=p.parse_args()
    out=Path(args.output).resolve()
    spec=make_spec()
    specpath=ROOT/'configs/evaluation/final_selection_100seeds.local.json'
    if not specpath.exists() or json.loads(specpath.read_text())!=spec: save_json(specpath,spec)
    manifest=prepare(specpath)
    models={m['name']:m for m in manifest['models']}
    for model in models.values(): model['decision_hz']=spec['rl_hz']
    models.update({n:dict(name=n,kind='native') for n in native.BASELINES})
    pairs=spec['cpu_pairs']
    if args.smoke:
        pairs=[[CANDIDATES[0],b] for b in OPPONENTS]+[[a,OPPONENTS[0]] for a in CANDIDATES[1:]]
    games=2 if args.smoke else 100;chunk=2 if args.smoke else 10
    files=[Path(__file__),ROOT/'evaluation/native_matchups.py',ROOT/'evaluation/native_baselines.py',
           native.RUNTIME/'FighterSim.py',native.RUNTIME/'JSBSimAIPLib.dll',ROOT/'claude_code/my_observation.py',
           native.RUNTIME/'src/dogfight/envs/single_agent_env.py',
           native.RUNTIME/'src/dogfight/envs/termination.py',
           ROOT/'evaluation/high_rate_provider.py',ROOT/'submission/decision_rate.py',
           native.RUNTIME/'src/dogfight/ai/bt_action_provider.py',native.RUNTIME/'cutoff_udp_provider.py']
    for base in ('artifacts/models/bt','artifacts/models/mpc/Release_MPC_team_share'):
        files.extend(p for p in (ROOT/base).rglob('*') if p.suffix.lower() in ('.dll','.exe','.xml','.yaml','.py'))
    identity=dict(manifest=manifest,protocol=spec['seed_protocol'],rl_hz=spec['rl_hz'],pairs=pairs,games=games,chunk=chunk,
                  smoke=args.smoke,hashes={str(f.relative_to(ROOT)):sha(f) for f in files})
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=identity:
        raise ValueError('immutable manifest mismatch: use a new output path')
    save_json(out/'manifest.json',identity)
    if not args.report_only:
        tasks=[]
        for a,b in pairs:
            for start in range(0,games,chunk):
                folder=out/f'{a}__{b}'/f'chunk_{start:03d}'
                if not (folder/'result.json').exists():
                    tasks.append((models[a],models[b],str(folder),'three_nine',5539.,min(chunk,games-start),
                        2. if args.smoke else 200.,spec['seed'],
                        dict(independent_seeds=True,total_games=games,game_offset=start)))
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
        # Spread each seed batch over candidates before advancing to later batches.
        tasks.sort(key=lambda t:(t[8]['game_offset'],OPPONENTS.index(t[1]['name']),CANDIDATES.index(t[0]['name'])))
        print(f'START: {len(tasks)} chunks remaining, {len(pairs)*games} games, {args.workers} workers',flush=True)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 max_tasks_per_child=1) as pool:
            futures=[pool.submit(play,t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                try: f.result()
                except BaseException as e:
                    save_json(out/'failure.json',dict(error=repr(e),completed_new_chunks=i-1))
                    for pending in futures: pending.cancel()
                    raise
                save_json(out/'progress.json',dict(completed_new_chunks=i,total_new_chunks=len(tasks)))
                print(f'CHUNKS {i}/{len(tasks)}',flush=True)
    report=collect(out,pairs,games,chunk)
    if report['complete'] and not args.smoke: render(out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__': main()
