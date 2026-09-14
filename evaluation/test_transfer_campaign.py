import importlib.util
from pathlib import Path
import unittest
import numpy as np


class CampaignTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).with_name('transfer_campaign.py')
        self.assertTrue(path.exists(), 'independent CPU/GPU campaign is not implemented')
        spec = importlib.util.spec_from_file_location('transfer_campaign', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_seed_identity_not_mirrored(self):
        m = self.module()
        rows = m.seed_rows()
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r['seed'] for r in rows}), 100)
        self.assertEqual(sum(r['swapped'] for r in rows), 50)
        self.assertEqual([r['swapped'] for r in rows[:4]], [False, True, False, True])

    def test_candidate_set_and_cpu_gpu_common_pairs(self):
        m = self.module()
        spec = m.make_spec()
        self.assertEqual(set(a for a,b in spec['cpu_pairs']),
                         {'gylee_20000','gylee_23000','gylee_36000'})
        self.assertEqual(len(spec['cpu_pairs']), 33)
        self.assertEqual(len(spec['gpu_pairs']), 24)
        self.assertTrue(set(map(tuple,spec['gpu_pairs'])) <= set(map(tuple,spec['cpu_pairs'])))
        self.assertFalse(any('cutoff' in b or b=='hard_deck_dive' for a,b in spec['gpu_pairs']))

    def test_action_input_width_mapping(self):
        m = self.module()
        import torch
        x = torch.arange(214).view(1,214)
        y = m.policy_obs(x,184)
        self.assertEqual(y.shape,(1,184))
        self.assertEqual(y[0,164:].tolist(),list(range(194,214)))
        self.assertTrue(torch.equal(m.policy_obs(x,214),x))
        with self.assertRaises(ValueError): m.policy_obs(x,200)

    def test_initial_bank_captures_real_native_reset(self):
        m=self.module()
        self.assertTrue(hasattr(m,'initial_bank'),'native initial condition export missing')
        bank=m.initial_bank(m.seed_rows()[:2])
        self.assertEqual(len(bank),2)
        self.assertNotEqual(bank[0]['seed'],bank[1]['seed'])
        for row in bank:
            self.assertGreater(np.linalg.norm(row['cpu_own'][6:9]),190)
            self.assertAlmostEqual(row['requested'][0]['speed'],row['requested'][1]['speed'])
            self.assertGreater(row['requested'][0]['speed'],199)


if __name__=='__main__': unittest.main()
