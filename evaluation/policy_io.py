"""Read-only adapters for frozen submission bundles; no training imports/state changes."""
import math
import torch
from claude_code.model import load_bundle


class Legacy184Bundle:
    """Project 214D observations by feature identity, then use submission inference.

    Legacy184 shares current features 0:164 and command history 194:214.
    The intervening 30 acceleration features were added after this contract.
    Normalization remains float64 with epsilon 1e-8 and clip 10, exactly as
    make_obs_normalizer; weights are neither expanded nor modified.
    """
    def __init__(self, model, metadata, device, legacy_min_altitude_m=300.):
        if metadata['observation_size'] != 184 or metadata['model']['type'] != 'mlp_discrete_actor_critic':
            raise ValueError('only verified legacy184 discrete MLP bundles are supported')
        self.model=model.eval()
        self.num_bins=model.num_bins
        self.indices=torch.tensor([*range(164),*range(194,214)],device=device)
        self.margin_offset=math.tanh((304.8-float(legacy_min_altitude_m))/300.)
        norm=metadata.get('obs_normalization')
        self.mean=self.inv_std=None
        if norm:
            self.mean=torch.tensor(norm['mean'],dtype=torch.float64,device=device)
            var=torch.tensor(norm['var'],dtype=torch.float64,device=device)
            if self.mean.shape!=(184,) or var.shape!=(184,) or not torch.isfinite(self.mean).all() or not torch.isfinite(var).all() or (var<0).any():
                raise ValueError('invalid bundle normalization')
            self.inv_std=1. / torch.sqrt(var+1e-8)
        for p in self.model.parameters():
            if not torch.isfinite(p).all(): raise ValueError('nonfinite bundle weights')
            p.requires_grad_(False)

    def initial_state(self,batch_size,device):
        return None

    def preprocess(self,obs):
        if obs.shape[-1]!=214: raise ValueError('bundle bridge requires current 214D input')
        x=obs.index_select(-1,self.indices)
        # tanh addition avoids unstable atanh near saturation at +/-1.
        t=x[:,4].double()
        x[:,4]=((t+self.margin_offset)/(1+t*self.margin_offset)).float()
        if self.mean is not None:
            x=((x.double()-self.mean)*self.inv_std).clamp(-10,10).float()
        return x

    def act(self,obs,state,episode_start,sample=False):
        if sample: raise ValueError('tournament bundle bridge is deterministic only')
        return self.model.act_deterministic(self.preprocess(obs)),None


def load_bundle_policy(path,device='cuda',legacy_min_altitude_m=300.):
    model,metadata=load_bundle(path,device=device)
    return Legacy184Bundle(model,metadata,device,legacy_min_altitude_m),None
