import unittest
from types import SimpleNamespace
import numpy as np
import torch
import native_baselines as native

native.setup()
from high_rate_provider import make_provider


class Actor:
    num_bins=21
    def initial_state(self,*args):
        self.observations=[]; self.starts=[]
        return None
    def act(self,x,state,start,sample=False):
        assert not sample
        self.observations.append(x.clone());self.starts.append(bool(start[0]))
        return torch.tensor([[10,10,10,5]]),None


def context():
    a=np.zeros(51);a[2]=-4000;a[6]=300
    b=a.copy();b[0]=600
    return SimpleNamespace(ownship_state=a,target_state=b)


class HighRateTests(unittest.TestCase):
    def test_every_frame_decides_physical_time_not_six_times_faster(self):
        a=Actor();p=make_provider(a,None,'checkpoint')
        for _ in range(61): p.compute_action(context())
        self.assertEqual((p.calls,p.decisions),(61,61))
        self.assertAlmostEqual(p.observation.rec.t_sec,1.)
        self.assertAlmostEqual(a.observations[-1][0,-17].item(),.25)
        self.assertEqual(sum(a.starts),1)
        p.reset();p.compute_action(context())
        self.assertEqual(p.decisions,1)
        self.assertEqual(p.observation.rec.t_sec,0.)
        self.assertEqual(a.observations[0][0,-20:].abs().sum(),0)

    def test_legacy_throttle_preserves_zero_padding(self):
        a=Actor();p=make_provider(a,None,'legacy184_bundle')
        for _ in range(7):p.compute_action(context())
        h=a.observations[-1][0,-20:].reshape(5,4)
        self.assertAlmostEqual(h[0,3].item(),-.5)
        self.assertEqual(h[1:].abs().sum(),0)

    def test_184_norm_gets_semantic_projection(self):
        norm=SimpleNamespace(mean=torch.zeros(184),normalize=lambda x:x)
        a=Actor();p=make_provider(a,norm,'checkpoint');p.compute_action(context())
        self.assertEqual(a.observations[-1].shape,(1,184))


if __name__=='__main__':unittest.main()
