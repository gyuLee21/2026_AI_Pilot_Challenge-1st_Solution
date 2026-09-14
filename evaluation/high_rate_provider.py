"""Use the submission 60Hz observation clock with the existing frozen adapters.

GRU advances once per decision (60Hz); hidden state resets per episode.
Legacy bundles retain raw throttle history; padding remains zero.
"""
from submission.decision_rate import DecisionRateProvider


def make_provider(actor, norm, kind, hz=60):
    class AdaptedActor:
        num_bins=actor.num_bins

        def initial_state(self,batch_size,device):
            self.frames=0
            return actor.initial_state(batch_size,device)

        def act(self,x,state,episode_start,sample=False):
            if kind=='legacy184_bundle':
                x=x.clone()
                valid=min(self.frames//6,5)
                history=x[:,-20:].reshape(-1,5,4)
                history[:,:valid,3]=history[:,:valid,3]*2-1
            if norm is not None:
                if norm.mean.numel()==184:
                    import torch
                    x=torch.cat((x[:,:164],x[:,194:]),dim=1)
                x=norm.normalize(x)
            self.frames+=60//hz
            return actor.act(x,state,episode_start,sample=sample)

    return DecisionRateProvider(AdaptedActor(),None,hz)
