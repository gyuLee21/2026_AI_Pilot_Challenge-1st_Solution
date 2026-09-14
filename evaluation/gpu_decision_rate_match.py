"""Same frozen 36k policy, 10Hz versus 60Hz, GPU physics and inference."""
from collections import deque
import json
import numpy as np
import torch
import transfer_campaign as campaign
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.ic import build_seed_vector
from cuda_fdm.obs_reward import ned_to_body, log_so3
from cuda_fdm.search_eval import load_policy
from cuda_fdm.ppo_gpu import action_to_env
from tournament import save_json, sha


def update_history(obr, states, commands):
    obr.pqr.zero_(); obr.accel.zero_(); obr.act_hist.zero_()
    if len(states) == 7:
        old, new = states[0], states[-1]
        r0 = ned_to_body(old[:, 3:6]).transpose(1, 2)
        r1 = ned_to_body(new[:, 3:6]).transpose(1, 2)
        obr.pqr.copy_(log_so3(r0.transpose(1, 2).bmm(r1)) / .1)
        obr.accel.copy_((r1.bmm(new[:, 6:9, None]) - r0.bmm(old[:, 6:9, None])).squeeze(-1) / .1)
    for lag in range(1, 6):
        if len(commands) >= lag * 6:
            obr.act_hist[:, lag-1].copy_(commands[-lag*6])


def main():
    torch.set_num_threads(1)
    out = campaign.OUT / 'gpu_36k_10hz_vs_60hz'
    if (out/'result.json').exists():
        print((out/'result.json').read_text()); return
    # Reuse exactly the existing GPU campaign's requested initial conditions.
    rows = json.loads((campaign.OUT/'gpu/manifest.json').read_text())['initial_conditions']
    n = len(rows)
    assert n == 100
    env = GpuDogfightVecEnv(n, substeps=1, scenario='three_nine', seed=26091317)
    env.obr.dt = 1/60
    assert abs(env.obr.dt - 1/60) < 1e-12
    checkpoint = campaign.ROOT.parent/'mixed20k_resume_20260911/main/iter_36000.pt'
    model, norm = load_policy(str(checkpoint), 'cuda')
    assert model.is_recurrent is False
    seeds = [build_seed_vector(**env._ic_dict(s['n'],s['e'],s['d'],s['heading'],s['speed']))
             for row in rows for s in row['requested']]
    env.reset(stagger=False)
    env.sim.load_seed(torch.tensor(np.asarray(seeds),device='cuda',dtype=torch.float64))
    env.obr.reset_all(); env.obr.kernel_init_reward_state(env.sim.states)
    # Alternate which policy rate controls plane slot 0; each rate gets 50 games per slot.
    slow_slots = [(i//2)%2 for i in range(n)]
    assert sum(slow_slots) == 50
    assert sum(bool(row['swapped']) ^ bool(slot) for row,slot in zip(rows,slow_slots)) == 50
    slow = torch.tensor([[slot == 0, slot == 1] for slot in slow_slots],device='cuda').flatten()
    hidden = model.initial_state(2*n, 'cuda')
    done_before = torch.zeros(n,dtype=torch.bool,device='cuda')
    cached = torch.zeros((2*n,4),device='cuda')
    states, commands = deque(maxlen=7), deque(maxlen=30)
    records = [None]*n
    save_json(out/'manifest.json',dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),
        runner_sha256=sha(__file__),scenario='three_nine',games=100,physics_hz=60,
        rates=[10,60],history_seconds=.1,action_mode='argmax',initial_conditions=rows))
    with torch.inference_mode():
        for frame in range(12004):
            states.append(env.state9_flat().clone())
            update_history(env.obr, states, commands)
            obs = env.obr.kernel_build_obs(env.sim.states)
            if norm is not None: obs = norm.normalize(campaign.policy_obs(obs,norm.mean.numel()))
            # This checkpoint is feed-forward: unused 10Hz predictions must have no hidden-state effects.
            indices, hidden = model.act(obs,hidden,torch.full((2*n,),float(frame==0),device='cuda'),sample=False)
            proposed = action_to_env(indices,model.num_bins)
            update = (~slow) | (frame%6 == 0)
            cached = torch.where(update[:,None],proposed,cached)
            commands.append(cached.clone())
            _,_,done,info = env.step(cached,capture_terminal_obs=False)
            if frame == 5:
                live = ~done
                assert bool(torch.allclose(env.obr.t_sec[live], torch.full_like(env.obr.t_sec[live], .1), atol=1e-10))
            assert bool(info['terminal_state_finite'][~done_before].all())
            for i in (done & ~done_before).nonzero().flatten().cpu().tolist():
                hp = info['terminal_hp'][i].cpu().tolist()
                alt = info['terminal_alt_m'][i].cpu().tolist()
                dead = [hp[j]<=0 or alt[j]<304.8 for j in (0,1)]
                score = float(dead[1]) if dead[0]!=dead[1] else .5 if dead[0] else 1. if hp[0]>hp[1]+1e-9 else 0. if hp[0]<hp[1]-1e-9 else .5
                slow_slot = slow_slots[i]
                records[i]=dict(rows[i],slow_slot=slow_slot,score_10hz=score if slow_slot==0 else 1-score,
                    hp=hp,alt=alt,duration_sec=(frame+1)/60)
            done_before |= done
            if frame%1200 == 0: print(f'frame={frame} completed={int(done_before.sum())}/100',flush=True)
            if bool(done_before.all()): break
    assert all(r is not None for r in records)
    result=dict(complete=True,wins_10hz=sum(r['score_10hz']==1 for r in records),
        wins_60hz=sum(r['score_10hz']==0 for r in records),draws=sum(r['score_10hz']==.5 for r in records),records=records)
    save_json(out/'result.json',result)
    print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__ == '__main__': main()
