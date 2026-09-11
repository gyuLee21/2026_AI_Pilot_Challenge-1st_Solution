"""Bundle bridge parity against the original submission inference contract."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claude_code.model import make_actor_critic, save_bundle, make_obs_normalizer


class BundleTest(unittest.TestCase):
    def test_legacy_bridge_preserves_normalization_and_argmax(self):
        import policy_io
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model=make_actor_critic(obs_dim=184,act_dim=4,hidden=(16,),num_bins=21)
        norm=dict(mean=[.3]*184,var=[1.7]*184,count=10)
        with tempfile.TemporaryDirectory() as root:
            save_bundle(model,root,obs_norm=norm)
            adapted,normalizer=policy_io.load_bundle_policy(root,'cpu')
            x=torch.randn(64,214)*20
            x[:,4]=torch.linspace(-.9,.9,64)
            legacy=torch.cat((x[:,:164],x[:,194:]),dim=1)
            legacy[:,4]=torch.tanh(torch.atanh(x[:,4].double())+(304.8-300.)/300.).float()
            expected=model.act_deterministic(torch.from_numpy(make_obs_normalizer(norm)(legacy.numpy())))
            torch.testing.assert_close(adapted.preprocess(x),torch.from_numpy(make_obs_normalizer(norm)(legacy.numpy())),atol=1e-6,rtol=1e-6)
            actual,_=adapted.act(x,None,torch.ones(64),sample=False)
            self.assertTrue(torch.equal(expected,actual))
            self.assertIsNone(normalizer)
            with self.assertRaises(ValueError): adapted.act(x[:,:184],None,None,sample=False)

if __name__=='__main__': unittest.main()
