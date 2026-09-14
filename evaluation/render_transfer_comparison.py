"""Render completed opponent tests only; exclude decision-rate self matches."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'artifacts/evaluations/transfer_36000_20260913'
MODELS=['gylee_20000','gylee_23000','gylee_36000']
LABELS={
    'junhwa_final':'Junhwa final','junhwa_grid_3L749_gru_last':'Junhwa GRU',
    'junhwa_wide1_v2':'Junhwa wide v2','junhwa_exploiter_survive_v1':'Junhwa survive',
    'junhwa_exploiter_survive_15500':'Junhwa 15500','submission4499':'4499',
    'gylee_67000':'3-9 67k','gylee_45000':'3-9 45k','headon_iter_46000':'Head-on 46k',
    'cutoff':'Cutoff 60Hz','cutoff_10hz':'Cutoff 10Hz','hard_deck_dive':'Hard-deck dive'}
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False


def render(name, lookup):
    opponents=[b for b in LABELS if any((a,b) in lookup for a in MODELS)]
    values=np.array([[lookup[a,b]['wins'] for a in MODELS] for b in opponents])
    fig,ax=plt.subplots(figsize=(8.4, .66*len(opponents)+2))
    im=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(3),['20k','23k','36k'],fontsize=13)
    ax.set_yticks(range(len(opponents)),[LABELS[b] for b in opponents],fontsize=11)
    for i,b in enumerate(opponents):
        for j,a in enumerate(MODELS):
            s=lookup[a,b]
            assert s['wins']+s['losses']+s['draws']==100
            ax.text(j,i,f"{s['wins']}%\n{s['wins']}/{s['losses']}/{s['draws']}",ha='center',va='center',
                    color='white' if s['wins']>62 else 'black',fontsize=12)
    ax.set_title(f'{name} · 3-9 상대별 평가\n칸: 후보 승률 W/N · 승/패/무 · 대진당 100판',pad=16)
    ax.set_xlabel('후보 모델 · RL 모두 10Hz argmax',fontsize=11)
    fig.colorbar(im,ax=ax,label='후보 승률 (%)',fraction=.035,pad=.04)
    fig.tight_layout()
    path=ROOT/f'{name.lower()}_opponents_heatmap.png'
    fig.savefig(path,dpi=150);plt.close(fig);print(path)


def main():
    gpu={}
    for folder in [ROOT/'gpu',ROOT/'headon46k_extension/gpu']:
        for path in folder.glob('gylee_*__*.json'):
            r=json.loads(path.read_text());assert r['complete'] and len(r['records'])==100
            gpu[r['left'],r['right']]=r
    report=json.loads((ROOT/'cpu/report.json').read_text());assert report['complete']
    cpu={(r['left'],r['right']):r['summary'] for r in report['results']}
    render('GPU',gpu);render('CPU',cpu)


if __name__=='__main__':main()
