"""Frozen-only 214D MLPs. Explicitly not compatible learner initializers."""
import torch
from torch import nn

MARKER = '_external214'

class ExternalActor(nn.Module):
    obs_dim = 214
    act_dim = 4
    num_bins = 21

    def __init__(self, state):
        super().__init__()
        self.register_buffer(MARKER, torch.tensor(1))
        layers=[]; width=214; i=0
        while f'body.{i}.weight' in state:
            w=state[f'body.{i}.weight']
            if w.shape[1]!=width: raise ValueError('inconsistent external actor width')
            layers.extend([nn.Linear(width,w.shape[0]), nn.Tanh()]); width=w.shape[0]; i+=2
        if not layers or state['head.weight'].shape != (84,width):
            raise ValueError('external actor must have 214 input and 4x21 action bins')
        self.body=nn.Sequential(*layers); self.head=nn.Linear(width,84)
        self.load_state_dict(state,strict=True)
        self.requires_grad_(False); self.eval()

    def actor_logits(self, obs):
        if obs.shape[-1]!=214: raise ValueError('external actor requires real 214D observations')
        return self.head(self.body(obs))

    def initial_state(self, batch_size, device=None):
        z=torch.zeros(1,batch_size,1,device=device)
        return z,z.clone()

    def act(self, obs, state=None, episode_start=None, sample=True):
        logits=self.actor_logits(obs).view(-1,4,21)
        action=torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(-1)
        return action, state if state is not None else self.initial_state(obs.shape[0],obs.device)

def convert_checkpoint(checkpoint):
    cfg=checkpoint['cfg']; source=checkpoint['model']
    if cfg.get('activation','tanh')!='tanh' or any('gru' in k.lower() for k in source):
        raise ValueError('only explicitly supported feedforward tanh actors can enter this league')
    state={MARKER:torch.tensor(1)}
    if 'actor_body.0.weight' in source:
        for k,v in source.items():
            if k.startswith('actor_body.'): state[k.replace('actor_body.','body.',1)]=v.clone()
        for k in ('weight','bias'): state['head.'+k]=source['actor_logits.'+k].clone()
    else:
        indices=sorted(int(k.split('.')[1]) for k in source if k.startswith('actor_logits.') and k.endswith('.weight'))
        if not indices: raise ValueError('missing supported actor')
        for i in indices:
            for k in ('weight','bias'):
                state[('head.' if i==indices[-1] else f'body.{i}.')+k]=source[f'actor_logits.{i}.{k}'].clone()
    if any(not torch.isfinite(v).all() for v in state.values()): raise ValueError('nonfinite external actor')
    ExternalActor(state)
    norm=checkpoint.get('norm')
    if norm is not None:
        if norm['mean'].shape!=(214,) or norm['var'].shape!=(214,) or (norm['var']<0).any():
            raise ValueError('invalid external normalization')
    return {'model':state,'norm':norm}

def is_external(bundle):
    return MARKER in bundle['model']

def frozen_external(bundle, device):
    from cuda_fdm.ppo_gpu import RunningNorm
    net=ExternalActor(bundle['model']).to(device)
    norm=None
    if bundle.get('norm') is not None:
        norm=RunningNorm(214,device)
        norm.load_state_dict(bundle['norm'])
    return net,norm,{'state':None}
