"""Physical 60 Hz observations, 0.1-second history, independently timed decisions.

Both evaluation rates consume the same reconstruction. This deliberately fixes
the old adapter's 0.1s integration on 60Hz packets. It does not claim bitwise
equivalence with that older reconstructed-observation evaluator.
"""
from collections import deque
import numpy as np
from claude_code import my_observation as MO
from dogfight.ai.action_provider import ActionProvider, ActionResult


class RateObservation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rec = MO.StateReconstructor(dt_per_step=1/60)
        self.states = deque(maxlen=7)
        self.commands = deque(maxlen=30)
        self.frames = 0

    def observe(self, own, opponent):
        own = np.asarray(own, dtype=np.float64)
        opponent = np.asarray(opponent, dtype=np.float64)
        if own.size < 9 or opponent.size < 9 or not np.isfinite(own[:9]).all() or not np.isfinite(opponent[:9]).all():
            raise ValueError('nonfinite or incomplete aircraft state')
        if self.frames:
            if len(self.commands) != min(self.frames,30):
                raise RuntimeError('one applied command required per physical frame')
            self.rec.advance(own, opponent)
        self.states.append((own.copy(),opponent.copy()))
        # Keep derivatives measured over the training 0.1-second window.
        if len(self.states) == 7:
            old_own, old_opp = self.states[0]
            for prefix, old, current in [('own',old_own,own),('tgt',old_opp,opponent)]:
                setattr(self.rec,prefix+'_pqr_est',MO._estimate_pqr(old[3:6],current[3:6],.1))
                old_v=MO._ned_to_body_matrix(*old[3:6]).T @ old[6:9]
                new_v=MO._ned_to_body_matrix(*current[3:6]).T @ current[6:9]
                setattr(self.rec,prefix+'_accel_est',(new_v-old_v)/.1)
        else:
            self.rec.own_pqr_est.fill(0); self.rec.tgt_pqr_est.fill(0)
            self.rec.own_accel_est.fill(0); self.rec.tgt_accel_est.fill(0)
        self.rec.action_history.fill(0)
        for lag in range(1,6):
            if len(self.commands)>=lag*6:
                self.rec.action_history[lag-1]=self.commands[-lag*6]
        obs=MO.build_observation(own,opponent,self.rec._geo,reconstructor=self.rec)
        if not np.isfinite(obs).all():
            raise FloatingPointError('nonfinite reconstructed observation')
        self.frames+=1
        return obs

    def record_command(self, command):
        a=np.asarray(command,dtype=np.float32)
        if a.shape!=(4,) or not np.isfinite(a).all() or (np.abs(a[:3])>1.000001).any() or not -1e-6<=a[3]<=1.000001:
            raise ValueError('invalid applied command')
        self.commands.append(a.copy())


class DecisionRateProvider(ActionProvider):
    requires_observation=False
    builds_own_observation=True
    required_action_repeat=1

    def __init__(self, actor, norm, hz):
        if hz not in (10,60): raise ValueError('only 10Hz and 60Hz are supported')
        self.actor,self.norm,self.hz=actor,norm,hz
        self.reset()

    def reset(self,context=None):
        self.observation=RateObservation()
        self.state=self.actor.initial_state(1,'cpu')
        self.calls=0; self.decisions=0; self.cached=None
        self.latencies=[]

    def compute_action(self,context):
        import time
        import torch
        began=time.perf_counter()
        raw=self.observation.observe(context.ownship_state,context.target_state)
        updated=self.calls%(60//self.hz)==0
        if updated:
            with torch.inference_mode():
                x=torch.from_numpy(raw).reshape(1,-1)
                if self.norm is not None: x=self.norm.normalize(x)
                indices,self.state=self.actor.act(x,self.state,torch.tensor([self.decisions==0]),sample=False)
                # Legacy discrete actors return integer-valued float tensors.
                # Preserve argmax categories; never truncate a continuous action.
                if (not torch.isfinite(indices).all() or
                    not torch.equal(indices,indices.round()) or
                    (indices<0).any() or (indices>=self.actor.num_bins).any()):
                    raise ValueError('actor must return valid discrete category indices')
                indices=indices.long()
                levels=torch.linspace(-1.,1.,self.actor.num_bins)
                cmd=levels[indices[0]].numpy().copy()
                cmd[3]=cmd[3]*.5+.5
            self.cached=cmd
            self.decisions+=1
        self.observation.record_command(self.cached)
        self.calls+=1
        self.latencies.append(time.perf_counter()-began)
        return ActionResult(self.cached.copy(),f'neural_{self.hz}hz',info={'policy_updated':updated})
