"""CPU-only target/deadline/lease and real CLI regressions."""
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from cuda_fdm.training_control import DeadlineGuard, RunLease, resolve_deadline, retarget_resume_state
from cuda_fdm.league_vnext.budget import WallclockBudgetTracker
from cuda_fdm.league_vnext.contracts import VNextConfig
from cuda_fdm.league_vnext.shadow import VNextShadowController
from cuda_fdm.tests.audit_20260902_100k_fixes_val import _league_trainer
from cuda_fdm.tests.pool_episode_val import StatefulToy
from cuda_fdm.tests.active_league_gpu_val import assert_nested_equal
from cuda_fdm import train_gpu


class TrainingControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_fresh_short_horizon_and_legacy_migration_gate(self):
        for n in (1,499,20000,100000): VNextConfig(source_iteration=0,target_iteration=n).validate()
        with self.assertRaises(ValueError): VNextConfig(source_iteration=0,target_iteration=0).validate()
        with self.assertRaises(ValueError): VNextConfig(source_iteration=20000,target_iteration=20000).validate()

    def state(self):
        return dict(config=VNextConfig(source_iteration=0).to_dict(),
                    wallclock_budget=WallclockBudgetTracker().state_dict(),
                    payoff_graph={'evidence':[1,2,3]},last_solver_ids=[1,2])

    def test_target_change_requires_explicit_approval(self):
        with self.assertRaisesRegex(ValueError,'accept-target-change'):
            retarget_resume_state(self.state(),target=20000,completed=500)

    def test_only_target_fields_change(self):
        original=self.state(); before=copy.deepcopy(original)
        after,receipt=retarget_resume_state(original,target=20000,completed=500,allow_change=True)
        self.assertEqual(original,before)
        self.assertEqual(after['config']['target_iteration'],20000)
        self.assertEqual(after['wallclock_budget']['target_iteration'],20000)
        after['config']['target_iteration']=100000
        after['wallclock_budget']['target_iteration']=100000
        self.assertEqual(after,before)
        self.assertEqual(receipt['target_iteration'],20000)

    def test_no_change_is_identity(self):
        state=self.state()
        result,receipt=retarget_resume_state(state,target=100000,completed=14500)
        self.assertIs(result,state); self.assertIsNone(receipt)

    def test_target_cannot_precede_checkpoint(self):
        with self.assertRaises(ValueError):
            retarget_resume_state(self.state(),target=100,completed=500,allow_change=True)

    def test_controller_retarget_save_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller=VNextShadowController(tmp,VNextConfig(source_iteration=0))
            controller.budget.add('main',2.5); controller.budget.add('side',600.)
            state,_=retarget_resume_state(controller.state_dict(),target=20000,completed=0,allow_change=True)
            short=VNextShadowController(tmp,VNextConfig(source_iteration=0,target_iteration=20000),state=state)
            again=VNextShadowController(tmp,short.config,state=short.state_dict())
            self.assertEqual(short.state_dict(),again.state_dict())
            self.assertEqual(list(again.budget.samples['side']),[600.])

    def test_eta_counts_actual_future_milestones(self):
        budget=WallclockBudgetTracker(target_iteration=999)
        budget.add('main',2.);budget.add('evaluation',10.);budget.add('side',20.)
        self.assertEqual(budget.report(iteration=500)['rolling_eta_seconds'],998.)
        self.assertEqual(budget.report(iteration=500)['remaining_milestones'],0)
        budget.target_iteration=1000
        self.assertEqual(budget.report(iteration=500)['rolling_eta_seconds'],1030.)
        self.assertEqual(budget.report(iteration=500,milestone_period=0)['rolling_eta_seconds'],1000.)

    def guard(self): return DeadlineGuard('2026-09-13T16:00:00+09:00')

    def test_timezone_and_invalid_safety_rejected(self):
        with self.assertRaises(ValueError): DeadlineGuard('2026-09-13T16:00:00')
        with self.assertRaises(ValueError): DeadlineGuard(self.guard().deadline,safety_factor=float('nan'))
        with self.assertRaises(ValueError): DeadlineGuard(self.guard().deadline,safety_factor=.5)
        self.assertEqual(self.guard().epoch,datetime(2026,9,13,7,tzinfo=timezone.utc).timestamp())

    def test_milestone_refused_before_it_starts(self):
        guard=self.guard();budget=WallclockBudgetTracker()
        kw=dict(target_iteration=100000,milestone_period=500,budget=budget,now=guard.epoch-4000)
        self.assertTrue(guard.check(next_iteration=499,**kw)['allow'])
        self.assertFalse(guard.check(next_iteration=500,**kw)['allow'])
        self.assertFalse(guard.check(next_iteration=499,**{**kw,'now':guard.epoch})['allow'])

    def test_observed_slow_milestone_increases_reserve(self):
        guard=self.guard();budget=WallclockBudgetTracker()
        budget.add('evaluation',3000);budget.add('side',3000)
        report=guard.check(next_iteration=500,target_iteration=1000,milestone_period=500,
                           budget=budget,now=guard.epoch-7000)
        self.assertFalse(report['allow']);self.assertGreater(report['estimated_next_seconds'],9000)

    def test_deadline_inherited_and_explicitly_clearable(self):
        self.assertIsNone(resolve_deadline())
        saved=self.guard().state_dict()
        self.assertEqual(resolve_deadline(saved=saved).state_dict(),saved)
        self.assertIsNone(resolve_deadline(saved=saved,clear=True))
        with self.assertRaises(ValueError): resolve_deadline(reserve=20.)

    def test_gate_leaves_runtime_and_learner_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            trainer=_league_trainer(tmp,total_iterations=10)
            before=trainer._runtime_state();learner=trainer._snapshot_learner()
            self.assertEqual(trainer.train(before_iteration=lambda it:False),[])
            assert_nested_equal(self,before,trainer._runtime_state())
            assert_nested_equal(self,learner,trainer._snapshot_learner())
            self.assertEqual(trainer.iteration,0);self.assertTrue(trainer.checkpoint_safe)

    def test_gate_before_milestone_preserves_committed_iteration(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            trainer=_league_trainer(tmp,total_iterations=3,milestone_period=2)
            trainer.train_exploiter=lambda **kw:self.fail('must not start milestone')
            history=trainer.train(before_iteration=lambda it:it<2)
            self.assertEqual([s.iteration for s in history],[1])
            trainer.training_deadline=self.guard().state_dict()
            path=Path(tmp)/'checkpoint.pt';trainer.save(path)
            trainer.training_deadline=None;trainer.load(path)
            self.assertEqual(trainer.training_deadline,self.guard().state_dict())
            self.assertEqual(trainer.iteration,1)

    def test_run_lease_excludes_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunLease(tmp,resume=False):
                with self.assertRaises(RuntimeError):
                    with RunLease(tmp,resume=True): pass
            with RunLease(tmp,resume=True): pass

    def test_fresh_and_stop_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';path.touch()
            with self.assertRaises(ValueError):
                with RunLease(tmp,resume=False): pass
            (Path(tmp)/'STOP').touch()
            with self.assertRaisesRegex(ValueError,'STOP'):
                with RunLease(tmp,resume=True): pass

    def cli(self,root,n,extra=()):
        root=Path(root)
        def env(nenv,**kw):
            value=StatefulToy(nenv);value.scenario=kw['scenario'];value.scenario_b_prob=1.
            return value
        argv=['train_gpu','--nenv','8','--iters',str(n),'--rollout','1','--epochs','1',
              '--minibatches','1','--device','cpu','--hidden','8,8','--num-bins','3',
              '--no-aux-pred','--scenario','headon','--sched-period','0','--milestone-period','500',
              '--exploiter-iters','0','--active-league','--league-dir',str(root/'league'),
              '--vnext-mode','staged','--vnext-stage','1','--vnext-state-dir',str(root/'control'),
              '--save',str(root/'checkpoint.pt'),'--log',str(root/'metrics.csv'),
              '--save-runtime','--no-wandb',*extra]
        with patch.object(sys,'argv',argv),patch.object(train_gpu,'GpuDogfightVecEnv',env),contextlib.redirect_stdout(io.StringIO()):
            train_gpu.main()
        return torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)

    def test_real_cli_short_run_and_explicit_retarget(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.cli(tmp,3)
            self.assertEqual(d['iteration'],3)
            self.assertEqual(d['vnext_control']['config']['target_iteration'],3)
            self.assertNotIn('training_deadline',d)
            resume=['--resume',str(Path(tmp)/'checkpoint.pt')]
            before=hashlib.sha256((Path(tmp)/'checkpoint.pt').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'accept-target-change'):self.cli(tmp,4,resume)
            self.assertEqual(before,hashlib.sha256((Path(tmp)/'checkpoint.pt').read_bytes()).hexdigest())
            d=self.cli(tmp,4,[*resume,'--accept-target-change'])
            self.assertEqual(d['iteration'],4)
            self.assertEqual(d['vnext_control']['wallclock_budget']['target_iteration'],4)

    def test_real_cli_deadline_stops_before_milestone_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            deadline=datetime.fromtimestamp(time.time()+2100,timezone.utc).isoformat()
            d=self.cli(tmp,4,['--deadline',deadline,'--milestone-period','2'])
            self.assertEqual(d['iteration'],1)
            self.assertEqual(d['training_deadline']['deadline'],deadline)
            before=d['runtime']
            d=self.cli(tmp,4,['--resume',str(Path(tmp)/'checkpoint.pt'),'--milestone-period','2'])
            self.assertEqual(d['iteration'],1)
            assert_nested_equal(self,before,d['runtime'])


if __name__=='__main__':unittest.main(verbosity=2)
