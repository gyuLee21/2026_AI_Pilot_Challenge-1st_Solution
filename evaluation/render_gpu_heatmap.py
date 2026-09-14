import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'artifacts/evaluations/transfer_36000_20260913/gpu'
OUT=SRC/'gpu_heatmap.png'

files=sorted(SRC.glob('gylee_*__*.json'))
rows=sorted({json.loads(p.read_text())['left'] for p in files})
cols=sorted({json.loads(p.read_text())['right'] for p in files})
lookup={}
for p in files:
    j=json.loads(p.read_text())
    rec=j['records']
    assert j['complete'] and len(rec)==100
    assert len({r['seed'] for r in rec})==100
    lookup[(j['left'],j['right'])]=dict(rate=j['wins']/100,score=(j['wins']+.5*j['draws'])/100,
        wins=j['wins'],losses=j['losses'],draws=j['draws'],crash=sum(r['crash'] for r in rec),
        hp=float(np.mean([r['damage_diff'] for r in rec])))
assert len(lookup)==24
z=np.array([[lookup[(a,b)]['rate']*100 for b in cols] for a in rows])
fig,ax=plt.subplots(figsize=(17,5.8))
im=ax.imshow(z,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(len(cols)),[x.replace('junhwa_','J:').replace('gylee_','G:').replace('submission','4499') for x in cols],rotation=30,ha='right')
ax.set_yticks(range(len(rows)),[x.replace('gylee_','') for x in rows])
for i,a in enumerate(rows):
    for k,b in enumerate(cols):
        q=lookup[(a,b)]
        ax.text(k,i,f"{q['rate']:.0%}\n{q['wins']}/{q['losses']}/{q['draws']}",ha='center',va='center',fontsize=9,color='white' if z[i,k]>62 else 'black')
ax.set_xlabel('상대 (GPU, 3-9, 10Hz argmax)')
ax.set_ylabel('우리 후보')
ax.set_title('36,000 후보 GPU 평가 히트맵 — 조합별 100판 (승률 기준)')
fig.colorbar(im,ax=ax,label='승률 (%)')
fig.tight_layout()
fig.savefig(OUT,dpi=180)
print(OUT)
for a in rows:
    qs=[lookup[(a,b)] for b in cols]
    print(a, f"mean_win={np.mean([q['rate'] for q in qs]):.3f}", f"mean_score={np.mean([q['score'] for q in qs]):.3f}", f"crash={sum(q['crash'] for q in qs)}/{len(qs)*100}")
