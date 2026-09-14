"""Actor-only checkpoint with exact frozen normalization; no PPO imports."""
import torch
from torch import nn


class FrozenPolicy(nn.Module):
    def __init__(self, payload):
        super().__init__()
        self.num_bins=int(payload['num_bins'])
        weights=payload['actor']
        self.obs_dim=weights['0.weight'].shape[1]
        if self.obs_dim not in (184,214): raise ValueError('unsupported feature layout')
        layers=[]
        count=len([k for k in weights if k.endswith('.weight')])
        for i in range(count):
            w=weights[f'{i*2}.weight']
            layers.append(nn.Linear(w.shape[1],w.shape[0]))
            if i<count-1: layers.append(nn.Tanh())
        self.actor=nn.Sequential(*layers)
        self.actor.load_state_dict(weights,strict=True)
        self.register_buffer('mean',payload['mean'])
        self.register_buffer('var',payload['var'])
        self.requires_grad_(False); self.eval()

    def initial_state(self,batch_size,device): return None

    @torch.inference_mode()
    def logits(self,obs):
        if self.obs_dim==184 and obs.shape[-1]==214:
            obs=torch.cat((obs[:,:164],obs[:,194:]),dim=1)
        if obs.shape[-1]!=self.obs_dim: raise ValueError('wrong observation width')
        x=((obs.to(self.mean.dtype)-self.mean)/torch.sqrt(self.var+1e-8)).clamp(-10,10).float()
        return self.actor(x).reshape(-1,4,self.num_bins)

    def act(self,obs,state,episode_start,sample=False):
        if sample: raise ValueError('submission is deterministic')
        return self.logits(obs).argmax(-1),None


def export_checkpoint(path,destination):
    from pathlib import Path
    import hashlib
    ck=torch.load(path,map_location='cpu',weights_only=False)
    cfg=ck['cfg']; sd=ck['model']
    if cfg.get('architecture','mlp')!='mlp' or cfg.get('activation','tanh')!='tanh':
        raise ValueError('only verified tanh MLP export supported')
    actor={}
    if 'actor_body.0.weight' in sd:
        for k,v in sd.items():
            if k.startswith('actor_body.'): actor[k[len('actor_body.'):]]=v
        last=2*len(cfg['hidden'])
        actor.update({f'{last}.weight':sd['actor_logits.weight'],f'{last}.bias':sd['actor_logits.bias']})
    else:
        actor={k[len('actor_logits.'):]:v for k,v in sd.items() if k.startswith('actor_logits.')}
    payload=dict(actor=actor,mean=ck['norm']['mean'],var=ck['norm']['var'],
        num_bins=cfg['num_bins'],iteration=ck['iteration'],source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    model=FrozenPolicy(payload)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()) or (model.var<0).any():
        raise ValueError('invalid checkpoint values')
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,destination)
    return payload


def load(path):
    return FrozenPolicy(torch.load(path,map_location='cpu',weights_only=True))
