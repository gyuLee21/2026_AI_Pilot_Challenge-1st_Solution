"""Read-only two-episode diagnostic of the assumed Taemin inference contract."""
import sys
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from junhwa_policy import load_junhwa_policy
from policy_io import load_bundle_policy
from cuda_fdm.search_eval import evaluate_pair
from cuda_fdm.rl_env import GpuDogfightVecEnv

torch.set_num_threads(1)
actor, _ = load_junhwa_policy(ROOT/'artifacts/models/external/headon/taemin/mlp_eval.pt')
stats = {'steps': 0, 'actions': torch.zeros(4,21,device='cuda'), 'sat': [], 'obs_max': []}
def pre_hook(module, inputs, output):
    if len(stats['sat']) < 100:
        stats['sat'].append((output.abs()>3).float().mean().item())
        stats['obs_max'].append(inputs[0].abs().max().item())
actor.actor_body[0].register_forward_hook(pre_hook)
original_act = actor.act
def traced_act(obs, state, episode_start, sample=False):
    actions, state = original_act(obs,state,episode_start,sample)
    stats['steps'] += 1
    for i in range(4):
        stats['actions'][i] += torch.bincount(actions[:,i],minlength=21)
    return actions,state
actor.act = traced_act
opponent = load_bundle_policy(ROOT/'artifacts/models/rl/common/submission4499','cuda',300)
env = GpuDogfightVecEnv(2,scenario='headon',headon_distance_m=5539,
                      substeps=6,seed=260911,device='cuda',min_altitude_m=304.8,max_engage_time_s=200.)
result = evaluate_pair(env,(actor,None),opponent,'headon',260911)
stats['actions'] = stats['actions'].cpu().int().tolist()
stats['first_layer_saturation_mean'] = sum(stats.pop('sat'))/100
stats['obs_max_range'] = [min(stats['obs_max']),max(stats.pop('obs_max'))]
print(json.dumps({'diagnostics':stats,'result':result},indent=2))
