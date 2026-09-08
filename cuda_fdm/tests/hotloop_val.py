"""Exact unused-work removal contracts; no CUDA or production writes."""
import unittest
from unittest.mock import patch
import torch
from cuda_fdm.ppo_gpu import AuxiliaryMLPActorCritic
from cuda_fdm.rl_env import GpuDogfightVecEnv

class HotLoopContractTest(unittest.TestCase):
    def test_action_and_rollout_keep_samples_logprob_and_rng(self):
        model=AuxiliaryMLPActorCritic(obs_dim=214,hidden=(16,16))
        for n in (1,16,128):
            obs=torch.randn(n,214)
            rng=torch.get_rng_state()
            expected,lp,entropy,state=model.actor_step(obs)
            end=torch.get_rng_state()
            torch.set_rng_state(rng)
            actual,new_lp,unused,_=model.actor_rollout_step(obs)
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(lp,new_lp))
            self.assertIsNone(unused)
            self.assertTrue(torch.equal(end,torch.get_rng_state()))
            torch.set_rng_state(rng)
            sampled,_=model.act(obs)
            self.assertTrue(torch.equal(expected.long(),sampled))
            self.assertTrue(torch.equal(end,torch.get_rng_state()))

    def test_inference_does_not_compute_discarded_distribution_statistics(self):
        model=AuxiliaryMLPActorCritic(obs_dim=214,hidden=(16,16))
        obs=torch.zeros(3,214)
        with patch.object(torch.distributions.Categorical,'entropy',side_effect=AssertionError('unused entropy')):
            model.actor_rollout_step(obs)
            with patch.object(torch.distributions.Categorical,'log_prob',side_effect=AssertionError('unused log-prob')):
                model.act(obs)
                model.act(obs,sample=False)

    def test_step_option_restored_after_success_and_exception(self):
        env=object.__new__(GpuDogfightVecEnv)
        env.step=lambda actions:getattr(env,'_capture_terminal_obs',True)
        self.assertFalse(env.step_training(None))
        self.assertTrue(env.step(None))
        def fail(actions):
            self.assertFalse(env._capture_terminal_obs)
            raise RuntimeError('physics failure')
        env.step=fail
        with self.assertRaises(RuntimeError):env.step_training(None)
        self.assertTrue(env._capture_terminal_obs)

if __name__=='__main__':unittest.main()
