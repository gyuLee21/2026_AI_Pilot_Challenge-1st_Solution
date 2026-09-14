import unittest
from native_matchups import independent_seed_bank
from final_selection import failure_reason, summarize, make_spec


class FinalSelectionTests(unittest.TestCase):
    def test_100_independent_seeds_and_balanced_chunked_sides(self):
        bank=independent_seed_bank(26091317,100)
        self.assertEqual(len(set(bank)),100)
        ids=[i for start in range(0,100,10) for i in range(start,start+10)]
        self.assertEqual(ids,list(range(100)))
        self.assertEqual(sum(bool(i%2) for i in ids),50)
        self.assertTrue(all(bank==independent_seed_bank(26091317,100)))
        self.assertTrue(set(bank[::2]).isdisjoint(bank[1::2]))

    def test_losses_partition_without_double_counting(self):
        rows=[]
        for crash,destroyed,end in [(True,False,'ownship altitude below min'),
            (False,True,'ownship destroyed'),(True,True,'ownship altitude below min'),
            (False,False,'max time out'),(False,False,'fuel fail')]:
            rows.append(dict(score=0,crash=crash,destroyed=destroyed,end=end,
                crash_b=False,damage_diff=-.2,role_swapped=False))
        result=summarize(rows)
        self.assertEqual(result['losses'],5)
        self.assertEqual(sum(result['loss_reasons'].values()),5)
        self.assertEqual(result['loss_reasons']['crash_and_destroyed'],1)
        self.assertIsNone(failure_reason(dict(rows[0],score=1)))
        self.assertAlmostEqual(result['mean_hp_diff'],-.2)

    def test_requested_roster(self):
        spec=make_spec()
        self.assertEqual(len(spec['cpu_pairs']),45)
        self.assertEqual(spec['games_per_pair'],100)
        self.assertEqual(spec['rl_hz'],10)
        self.assertEqual(len({tuple(p) for p in spec['cpu_pairs']}),45)
        self.assertFalse(any(b in ('release_mpc','Shin_BT_best','Shin_BT_def') for a,b in spec['cpu_pairs']))
        self.assertIn(['gylee_32000','hard_deck_dive'],spec['cpu_pairs'])
        self.assertIn(['gylee_32000','cutoff_10hz'],spec['cpu_pairs'])
        self.assertNotIn('gylee_69000',[m['name'] for m in spec['models']])
        self.assertFalse(any('gylee_69000' in pair for pair in spec['cpu_pairs']))


if __name__=='__main__': unittest.main()
