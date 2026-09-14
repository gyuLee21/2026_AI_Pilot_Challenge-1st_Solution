import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from tournament import run_round_robin, prepare

def manifest(n):
    return dict(models=[dict(name=f'm{i}') for i in range(n)],games_per_pair=50,seed=1)

def evaluation(i,j,seed):
    return {'records':[dict(score=1.,crash=False,crash_b=True,ic_pair=k%25,
                            role_swapped=k>=25) for k in range(50)]}

class TournamentTest(unittest.TestCase):
    def test_cross_family_has_24_pairs_and_resumes_without_internal_games(self):
        m=manifest(9); m['cross_family_only']=True
        for i,model in enumerate(m['models']):
            model['family']='gylee' if i<4 else 'junhwa' if i<8 else 'baseline'
        calls=[]
        def play(i,j,s):
            self.assertNotEqual(m['models'][i]['family'],m['models'][j]['family'])
            calls.append((i,j)); return evaluation(i,j,s)
        with tempfile.TemporaryDirectory() as root:
            result=run_round_robin(m,root,play)
            self.assertEqual(len(calls),24)
            self.assertEqual(result['total_games'],1200)
            self.assertIsNone(result['score_matrix'][0][1])
            self.assertEqual(len(result['rankings_by_family']['gylee']),4)
            run_round_robin(m,root,play)
            self.assertEqual(len(calls),24)

    def test_scenario_and_distance_are_part_of_immutable_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'cuda_fdm').mkdir()
            for name in ('search_eval.py','rl_env.py'):
                (root/'cuda_fdm'/name).write_text('# fixture')
            for name in ('a.pt','b.pt'):
                (root/name).write_text(name)
            spec=dict(models=[dict(name=n,path=n+'.pt') for n in ('a','b')])
            path=root/'spec.json'
            path.write_text(json.dumps(spec))
            old=prepare(path,root)
            self.assertEqual(old['scenario'],'three_nine')
            spec.update(scenario='headon',headon_distance_m=5539.)
            path.write_text(json.dumps(spec))
            new=prepare(path,root)
            self.assertEqual(new['scenario'],'headon')
            self.assertEqual(new['headon_distance_m'],5539.)
            self.assertNotEqual(old,new)
            (root/'legacy').mkdir()
            (root/'legacy/metadata.json').write_text('{}')
            (root/'legacy/policy_weights.pkl.gz').write_bytes(b'weights')
            spec['models'][0]=dict(name='4499',path='legacy',kind='legacy184_bundle',legacy_min_altitude_m=300.)
            path.write_text(json.dumps(spec))
            bundled=prepare(path,root)
            (root/'legacy/metadata.json').write_text('{"changed":true}')
            self.assertNotEqual(bundled,prepare(path,root))
            for distance in (0,-1,float('nan')):
                spec['headon_distance_m']=distance
                path.write_text(json.dumps(spec))
                with self.assertRaises(ValueError): prepare(path,root)

    def test_40_models_exact_39000_and_reverse_scores(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            result=run_round_robin(manifest(40),root,evaluation)
            self.assertEqual(result['total_games'],39000)
            self.assertEqual(result['total_pairs'],780)
            self.assertEqual(result['rankings'][0]['name'],'m0')
            self.assertEqual(result['rankings'][0]['games'],1950)
            self.assertEqual(result['rankings'][-1]['crash_rate'],1)
            self.assertEqual(result['score_matrix'][0][39],1)
            self.assertEqual(result['score_matrix'][39][0],0)
            def forbidden(*a): raise AssertionError('cached pair was replayed')
            self.assertEqual(run_round_robin(manifest(40),root,forbidden),result)

    def test_interrupted_pair_resume_and_roster_mismatch(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            def fail(i,j,seed):
                if (i,j)==(0,2): raise RuntimeError('injected failure')
                return evaluation(i,j,seed)
            with self.assertRaises(RuntimeError): run_round_robin(manifest(3),root,fail)
            calls=[]
            def resume(i,j,seed): calls.append((i,j)); return evaluation(i,j,seed)
            run_round_robin(manifest(3),root,resume)
            self.assertEqual(calls,[(0,2),(1,2)])
            with self.assertRaises(ValueError): run_round_robin(manifest(4),root,evaluation)

    def test_incomplete_games_rejected_before_commit(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            def bad(i,j,s):
                result=evaluation(i,j,s); result['records'].pop(); return result
            with self.assertRaises(ValueError): run_round_robin(manifest(2),root,bad)
            self.assertFalse((Path(root)/'pairs/000_001.json').exists())

if __name__=='__main__': unittest.main()
