"""Merge approved 50-game evidence into per-ownship power-test text tables.

No backend columns in the presentation. Source paths and execution backend remain
in JSON provenance. Missing matches are pending, never fabricated as draws.
"""
from itertools import combinations
import json
from pathlib import Path
import math

from tournament import prepare, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/evaluation/final_power_tests'
ALIASES = {'original20k_20000':'gylee_original20k',
           'three_nine_iter_69000':'gylee_69000', 'headon_iter_46000':'gylee_headon46000'}
LABELS = {'gylee_original20k':'gylee 20000', 'gylee_69000':'gylee 69000',
          'gylee_headon46000':'gylee headon 46000', 'submission4499':'4499',
          'release_mpc':'release MPC', 'hard_deck_dive':'hard deck dive', 'cutoff':'cutoff (10Hz)',
          'junhwa_grid_3L749_gru_last':'junhwa GRU', 'junhwa_exploiter_survive_v1':'junhwa survive',
          'junhwa_wide1_v2':'junhwa wide', 'junhwa_final':'junhwa final', 'taemin_headon':'taemin MLP'}


def normalize(rows, flip=False):
    if len(rows) != 50:
        raise ValueError('only complete 50-game evidence is reusable')
    result = []
    for r in rows:
        hp = r.get('damage_diff')
        if hp is None:
            hp = r['hp']-r['target_hp']
        score = r['score']
        if score not in (0,.5,1) or not math.isfinite(hp):
            raise ValueError('invalid outcome')
        result.append(dict(score=1-score if flip else score, hp_diff=-hp if flip else hp))
    return result


def counts(rows):
    w = sum(r['score']==1 for r in rows)
    l = sum(r['score']==0 for r in rows)
    d = sum(r['score']==.5 for r in rows)
    return w,l,d,w/len(rows),w/(w+l) if w+l else None,sum(r['hp_diff'] for r in rows)/len(rows)


def line(name, rows):
    w,l,d,all_rate,decisive,hp = counts(rows)
    decisive_text = f'{decisive:.3f}' if decisive is not None else 'N/A'
    return f'{name:<28} {w:>3}/{l:<3}/{d:<3} {all_rate:>12.3f} {decisive_text:>12} {hp:>+9.3f} {len(rows):>7} {0:>7}'


def collect(scenario):
    config = ('junhwa_gylee_20k_69k.local.json' if scenario=='three_nine' else 'power_headon_available.local.json')
    expected = prepare(ROOT/'configs/evaluation'/config)
    roster = [m['name'] for m in expected['models']]
    if scenario=='headon' and 'taemin_headon' not in roster:
        roster.insert(0,'taemin_headon')
    roster += ['release_mpc','cutoff'] if scenario=='headon' else ['release_mpc','hard_deck_dive','cutoff']
    models = {m['name']:m for m in expected['models']}
    excluded = lambda a,b: a.startswith('junhwa_') and b.startswith('junhwa_')
    required = {tuple(sorted((a,b))) for a,b in combinations(roster,2) if not excluded(a,b)}
    found = {}
    def add(a,b,records,path,backend):
        key=tuple(sorted((a,b)))
        if key not in required or key in found:
            return
        found[key]=dict(left=key[0],right=key[1],records=normalize(records,a!=key[0]),
                        source=str(path),backend=backend)
    gpu_dirs = ([ROOT/'artifacts/evaluation/junhwa_gylee_20k_69k_gpu'] if scenario=='three_nine' else [])
    gpu_dirs += [ROOT.parent/'evaluation_results/league_20260911_separated'/scenario]
    if scenario=='headon' and (ROOT/'artifacts/evaluation/taemin_headon_gpu/manifest.json').exists():
        gpu_dirs.append(ROOT/'artifacts/evaluation/taemin_headon_gpu')
    for folder in gpu_dirs:
        m=json.loads((folder/'manifest.json').read_text())
        for k in ('scenario','headon_distance_m','action_mode','games_per_pair','seed','substeps','max_engage_time_s','min_altitude_m','environment_sha256','policy_adapter_sha256'):
            if m[k]!=expected[k]:
                raise ValueError(f'protocol mismatch {folder}: {k}')
        # Legacy difference verified: only eager environment import became lazy;
        # evaluate_pair, load_policy, summarize bodies are identical.
        if m['evaluator_sha256'] not in (expected['evaluator_sha256'],'ac983271eac991f31929d172ed8f2133fc0c9d566ca738dc1971daf855b0c543'):
            raise ValueError('unreviewed evaluator')
        accepted={}
        for i,actor in enumerate(m['models']):
            name=ALIASES.get(actor['name'],actor['name'])
            if name not in models:
                continue
            target=models[name]
            if any(actor.get(k)!=target.get(k) for k in ('sha256','kind','metadata_sha256','legacy_min_altitude_m')):
                raise ValueError(f'model identity mismatch {name}')
            accepted[i]=name
        for path in sorted((folder/'pairs').glob('*.json')):
            pair=json.loads(path.read_text())
            i,j=pair['left'],pair['right']
            if i in accepted and j in accepted:
                add(accepted[i],accepted[j],pair['result']['records'],path,'gpu')
    if scenario=='three_nine':
        folder=ROOT/'artifacts/evaluation/junhwa_gylee_20k_69k_cpu'
        identity=json.loads((folder/'manifest.json').read_text())
        if identity['games']!=50 or identity['seconds']!=200 or identity['rl_manifest']!=expected:
            raise ValueError('existing CPU protocol mismatch')
        for path in folder.glob('*/*/result.json'):
            r=json.loads(path.read_text())
            if not r['complete']: continue
            add(r['model'],r['baseline'],r['records'],path,'cpu')
    folder=ROOT/'artifacts/evaluation'/f'power_{scenario}_cpu'
    for path in folder.glob('*/result.json'):
        r=json.loads(path.read_text())
        if r['complete'] and r['scenario']==scenario:
            add(r['left'],r['right'],r['records'],path,'cpu')
    pending=sorted(required-set(found))
    complete=not pending
    texts=[f'{scenario}: {"최종" if complete else "진행 중"} 파워 테스트 — {len(found)}/{len(required)} 대전 완료',
           '승률(D포함)=W/(W+L+D), 승률(D제외)=W/(W+L), HP차=종료 시 own HP - target HP.',
           '50판=25 초기조건의 양쪽 자리 교대. 미실행은 대기이며 제외 판수로 세지 않음.',
           '신경망은 모든 고도에서 사용. 모델마다 상대 집합이 다를 수 있으므로 전체 평균의 단순 순위에 주의.', '']
    for own in roster:
        texts += ['='*105, f'최종 파워 테스트 결과   ownship = {LABELS.get(own,own)}', '='*105,
                  f'{"target":28} {"W/L/D":11} {"승률(D포함)":>12} {"승률(D제외)":>12} {"HP차":>9} {"판수":>7} {"제외":>7}', '-'*105]
        total=[]
        for opp in roster:
            if own==opp or excluded(own,opp): continue
            key=tuple(sorted((own,opp)))
            if key not in found:
                texts.append(f'{LABELS.get(opp,opp):28} 대기')
                continue
            entry=found[key]
            rows=entry['records']
            if own!=entry['left']:
                rows=[dict(score=1-r['score'],hp_diff=-r['hp_diff']) for r in rows]
            texts.append(line(LABELS.get(opp,opp),rows))
            total.extend(rows)
        if total: texts.append(line('TOTAL',total))
        texts += ['='*105,'']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{scenario}.txt').write_text('\n'.join(texts),encoding='utf-8')
    save_json(OUT/f'{scenario}.json',dict(complete=complete,completed_pairs=len(found),
        total_pairs=len(required),pending=pending,matches=list(found.values())))
    return dict(scenario=scenario,complete=complete,completed_pairs=len(found),total_pairs=len(required),pending=pending)


if __name__=='__main__':
    for scenario in ('three_nine','headon'):
        print(json.dumps(collect(scenario),ensure_ascii=False))
