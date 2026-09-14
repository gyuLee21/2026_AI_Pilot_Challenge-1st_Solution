"""One-shot, fail-closed GPU evaluation after a verified clean training stop."""
import argparse
import json
from pathlib import Path
from tournament import prepare, selected_pairs, save_json, run_round_robin


def verify_completion(run, load):
    run=Path(run)
    for name in ('status.json','stop_at_2000_receipt.json'):
        if json.loads((run/name).read_text(encoding='utf-8'))['additional_iterations']!=2000:
            raise ValueError('training did not stop at additional iteration 2000')
    for name in ('checkpoint.pt','additional_2000.pt'):
        if load(run/name)['iteration']!=72000:
            raise ValueError('checkpoint must be league iteration 72000 / policy iteration 69500')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    # Exclusive file prevents accidental duplicate launches, including after failures.
    with (output/'launch.lock').open('x') as lock:
        import os
        lock.write(str(os.getpid()))
    try:
        import torch
        verify_completion(args.run_dir,lambda p:torch.load(p,map_location='cpu',weights_only=False,mmap=True))
        manifest=prepare(args.spec)
        if manifest['scenario']!='three_nine' or len(selected_pairs(manifest))!=24:
            raise ValueError('expected 24 cross-family three-nine pairings')
        save_json(output/'launch_status.json',dict(state='gpu_preflight',manifest=manifest))
        from parallel_tournament import initialize, evaluate
        # Use every policy in full-length mirrored smoke matches, without intra-family games.
        smoke_pairs=[(i,4) for i in range(4)]+[(0,j) for j in range(5,9)]
        initialize(dict(manifest,games_per_pair=2))
        smoke=[]
        for i,j in smoke_pairs:
            print(f'[preflight] {manifest["models"][i]["name"]} vs {manifest["models"][j]["name"]}',flush=True)
            result=evaluate((i,j,manifest['seed']))
            if len(result['records'])!=2:
                raise ValueError('incomplete GPU smoke match')
            smoke.append(dict(left=i,right=j,result=result))
            save_json(output/'gpu_preflight.json',dict(complete=False,matches=smoke))
        save_json(output/'gpu_preflight.json',dict(complete=True,matches=smoke))
        # Reinitialize all recurrent state and environments; smoke never enters scoring.
        initialize(manifest)
        save_json(output/'launch_status.json',dict(state='evaluating'))
        run_round_robin(manifest,output,lambda i,j,s:evaluate((i,j,s)),output/'STOP')
        save_json(output/'launch_status.json',dict(state='complete'))
    except BaseException as exc:
        save_json(output/'launch_status.json',dict(state='failed',error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__': main()
