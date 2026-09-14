"""100 independent 3-9 seeds, actual submission rate provider, CPU workers 7."""
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import decision_rate_match as runner


def main():
    out=runner.ROOT/'artifacts/evaluations/transfer_36000_20260913/cpu_36k_10hz_vs_60hz'
    from submission.frozen_policy import export_checkpoint
    import torch
    torch.set_num_threads(1)
    source=runner.ROOT.parent/'mixed20k_resume_20260911/main/iter_36000.pt'
    frozen=out/'model.pt'
    payload=export_checkpoint(source,frozen)
    gpu=out.parent/'gpu_36k_10hz_vs_60hz'
    manifest=json.loads((gpu/'manifest.json').read_text())
    assert payload['source_sha256']==manifest['sha256']
    rows=manifest['initial_conditions']
    runner.save(out/'manifest.json',dict(source_sha256=payload['source_sha256'],scenario='three_nine',
        games=100,workers=7,rates=[10,60],initial_conditions=rows,
        observation='DecisionRateProvider: 60Hz integration and 0.1s history',
        side_rule='GPU swapped XOR GPU slow_slot; CPU ownship is always 10Hz'))
    tasks=[]
    for i,row in enumerate(rows):
        if (out/f'block_{i:03}.json').exists(): continue
        swapped=bool(row['swapped']) ^ bool((i//2)%2)
        tasks.append((dict(path=str(frozen),hz=10),dict(path=str(frozen),hz=60),row['seed'],i,str(out),200.,
                      dict(scenario='three_nine',sides=[swapped])))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=7,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(runner.play,t) for t in tasks]
        for k,f in enumerate(as_completed(futures),1):
            f.result()
            runner.save(out/'progress.json',dict(completed_new_games=k,total_new_games=len(tasks)))
            print(f'completed {k}/{len(tasks)}',flush=True)
    games=[json.loads((out/f'block_{i:03}.json').read_text())['records'][0] for i in range(100)]
    result=dict(complete=True,wins_10hz=sum(r['score']==1 for r in games),
        wins_60hz=sum(r['score']==0 for r in games),draws=sum(r['score']==.5 for r in games),records=games)
    runner.save(out/'report.json',result)
    print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__': main()
