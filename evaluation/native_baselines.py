"""CPU paired evaluation of frozen RL actors against native BT/MPC/UDP policies.

Each spawned process owns its simulator, observation history, GRU state and
native opponent. Results are resumable per complete 50-game matchup.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'artifacts/runtimes/cpu_baselines'
BASELINES=('hard_deck_dive','release_mpc','cutoff','cutoff_10hz','Shin_BT_best','Shin_BT_def')


def setup():
    # Use the current 214D observation implementation with the official CPU FDM.
    sys.path.insert(0,str(ROOT))
    import claude_code
    claude_code.__path__.append(str(RUNTIME/'claude_code'))
    sys.path.extend([str(RUNTIME),str(RUNTIME/'src')])
    os.environ['AIP_RULE_XML']=str(ROOT/'artifacts/models/bt/vertical_deck_dive_ver01/Rule_vertical_deck_dive_ver01.xml')
    os.chdir(RUNTIME)
    import torch
    torch.set_num_threads(1)


def load_actor(model):
    import torch
    from cuda_fdm.search_eval import load_policy
    from junhwa_policy import load_junhwa_policy
    if model['kind']=='junhwa_actor':
        return load_junhwa_policy(model['path'],'cpu')
    if model.get('config_from'):
        checkpoint=torch.load(model['path'],map_location='cpu',weights_only=False,mmap=True)
        template=torch.load(model['config_from'],map_location='cpu',weights_only=False,mmap=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            torch.save(dict(checkpoint,cfg=template['cfg']),path)
            return load_policy(path,'cpu')
    return load_policy(model['path'],'cpu')


def rl_provider(model):
    import numpy as np
    import torch
    from dogfight.ai.action_provider import ActionProvider,ActionResult
    from claude_code.my_observation import StateReconstructor,build_observation,OBSERVATION_SIZE
    from GeoMathUtil import GeometryInfo
    from cuda_fdm.ppo_gpu import action_to_env
    actor,norm=load_actor(model)
    if OBSERVATION_SIZE!=214: raise ValueError('CPU observation must be 214D')
    if model.get('decision_hz')==60 or model.get('submission_clock',False):
        from high_rate_provider import make_provider
        return make_provider(actor,norm,model.get('kind','checkpoint'),hz=model.get('decision_hz',60))

    class Provider(ActionProvider):
        requires_observation=False
        def __init__(self):
            self.geo=GeometryInfo(); self.reset()
        def reset(self,context=None):
            self.recon=StateReconstructor()
            self.state=actor.initial_state(1,'cpu')
            self.tick=0; self.cached=None
        @torch.inference_mode()
        def compute_action(self,context):
            if self.tick%6==0:
                own=np.asarray(context.ownship_state,dtype=np.float64)
                opp=np.asarray(context.target_state,dtype=np.float64)
                if self.tick:
                    self.recon.advance(own,opp)
                obs=build_observation(own,opp,self.geo,None,reconstructor=self.recon)
                if not np.isfinite(obs).all(): raise FloatingPointError('nonfinite observation')
                x=torch.as_tensor(obs,dtype=torch.float32).reshape(1,214)
                if norm is not None:
                    if norm.mean.numel()==184:
                        x=torch.cat((x[:,:164],x[:,194:]),dim=1)
                    x=norm.normalize(x)
                actions,self.state=actor.act(x,self.state,torch.tensor([float(self.tick==0)]),sample=False)
                raw=(actions[0].numpy().astype(np.float32)/(actor.num_bins-1))*2-1
                command=action_to_env(actions,actor.num_bins)[0].numpy()
                # GPU rl_env stores applied controls, including throttle [0,1].
                # Legacy CPU submission bundles used pre-mapping [-1,1] history.
                self.recon.push_action(raw if model.get('kind')=='legacy184_bundle' else command)
                if not np.isfinite(command).all(): raise FloatingPointError('nonfinite command')
                self.cached=ActionResult(action=command,source='frozen_rl',confidence=1.)
            self.tick+=1
            return self.cached
    return Provider()


def baseline_provider(name,output):
    if name in ('Shin_BT_best','Shin_BT_def','hard_deck_dive'):
        # Load the audited evaluation provider explicitly. The application
        # provider's second positional argument is ai_pilot, not rule_xml.
        import importlib.util
        path=RUNTIME/'src/dogfight/ai/bt_action_provider.py'
        spec=importlib.util.spec_from_file_location('evaluation_native_bt_provider',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BTActionProvider=module.BTActionProvider
    if name in ('Shin_BT_best','Shin_BT_def'):
        package=ROOT/'artifacts/models/bt'/name
        return BTActionProvider(str(package/(name+'.dll')),rule_xml=str(package/(name+'.xml')))
    if name=='hard_deck_dive':
        package=ROOT/'artifacts/models/bt/vertical_deck_dive_ver01'
        return BTActionProvider(str(package/'AIP_vertical_deck_dive_ver01_release_target.dll'),
                                rule_xml=str(package/'Rule_vertical_deck_dive_ver01.xml'))
    if name=='release_mpc':
        package=ROOT/'artifacts/models/mpc/Release_MPC_team_share'
        sys.path.append(str(package/'src'))
        from mpc.config import load_config
        from mpc.provider import MPCActionProvider
        return MPCActionProvider(package,load_config(package/'configs/mpc.yaml'))
    if name in ('cutoff','cutoff_10hz'):
        from cutoff_udp_provider import CutoffUDPActionProvider
        return CutoffUDPActionProvider(ROOT/'artifacts/models/bt/cutoff/unreal_bt_client.exe',
                                       output/'client',os.getpid(),action_repeat=6 if name=='cutoff_10hz' else 1)
    raise ValueError(name)


def audited_provider(inner):
    import numpy as np
    class Audited:
        requires_observation=False
        def reset(self,context=None):
            self.calls=0; self.fallbacks=0; self.updates=0; self.actions=set()
            inner.reset(context)
        def compute_action(self,context):
            result=inner.compute_action(context)
            if not np.isfinite(result.action).all(): raise FloatingPointError('native policy action')
            self.calls+=1
            self.fallbacks+=int('fallback' in result.source)
            self.updates+=int(result.info.get('policy_updated',True))
            self.actions.add(tuple(np.round(result.action,4)))
            return result
        def close(self): inner.close()
    return Audited()


def play(task):
    model,baseline,output,games,seconds,seed=task
    output=Path(output)
    setup()
    import numpy as np
    from claude_code.env_utils import make_env
    from tournament import save_json
    own=rl_provider(model)
    opponent=audited_provider(baseline_provider(baseline,output))
    env=make_env(overrides=dict(target_mode='fixed',ownship_control_mode='rl',
        step_ratio=6,max_engage_time=seconds,episode_step_limit=2001,
        min_altitude=304.8,randomize_start_side=False,scenario_b_prob=0.,
        artifacts_dir=str(output/'trajectories')),runner_index=f'native_{os.getpid()}')
    env._ownship_action_provider=own
    env._target_action_provider=opponent
    seeds=np.random.default_rng(seed).integers(0,2**31-1,size=games//2)
    rows=[]; started=time.perf_counter()
    try:
        for k in range(games):
            block=k//2; swapped=bool(k%2)
            env._apply_start_side(swapped,swapped)
            _,info=env.reset(seed=int(seeds[block]))
            if env._initial_scenario_kind!='A': raise ValueError('not three-nine')
            initial_own=env._sim.get_state()[:7].tolist()
            initial_opp=env._target_sim.get_state()[:7].tolist()
            if swapped:
                np.testing.assert_allclose(initial_own,rows[-1]['initial_opp'],rtol=0,atol=1e-8)
                np.testing.assert_allclose(initial_opp,rows[-1]['initial_own'],rtol=0,atol=1e-8)
            game_start=time.perf_counter()
            for steps in range(1,2002):
                _,_,terminated,truncated,info=env.step(np.zeros(4,dtype=np.float32))
                if terminated or truncated: break
            else: raise RuntimeError('episode did not finish')
            hp=float(info['ownship_health']); target_hp=float(info['target_health'])
            state=env._sim.get_state(); target=env._target_sim.get_state()
            alt=-float(state[2]); target_alt=-float(target[2])
            if not np.isfinite([hp,target_hp,alt,target_alt]).all(): raise FloatingPointError('terminal state')
            if opponent.calls<steps or opponent.fallbacks>max(12,opponent.calls*.2):
                raise RuntimeError(f'native policy not healthy: calls={opponent.calls} fallback={opponent.fallbacks}')
            dead=hp<=0 or alt<304.8; target_dead=target_hp<=0 or target_alt<304.8
            score=(float(target_dead) if dead!=target_dead else .5 if dead else
                   1. if hp>target_hp+1e-9 else 0. if hp<target_hp-1e-9 else .5)
            rows.append(dict(score=score,crash=alt<304.8,crash_b=target_alt<304.8,
                hp=hp,target_hp=target_hp,ic_pair=block,role_swapped=swapped,
                seed=int(seeds[block]),duration_sec=steps*.1,wall_sec=time.perf_counter()-game_start,
                initial_own=initial_own,initial_opp=initial_opp,end=info.get('end_condition')))
            rows[-1].update(native_calls=opponent.calls,native_updates=opponent.updates,
                            native_fallbacks=opponent.fallbacks,native_distinct_actions=len(opponent.actions),
                            final_own_state=np.asarray(state).tolist(),final_target_state=np.asarray(target).tolist())
            save_json(output/'progress.json',dict(model=model['name'],baseline=baseline,
                completed_games=len(rows),total_games=games,records=rows,complete=False))
            print(f'[{baseline} vs {model["name"]}] {k+1}/{games} RLscore={score} wall={rows[-1]["wall_sec"]:.1f}s',flush=True)
        result=dict(model=model['name'],family=model.get('family'),baseline=baseline,
                    records=rows,wall_sec=time.perf_counter()-started,complete=True,
                    score=sum(r['score'] for r in rows)/games)
        save_json(output/'result.json',result)
        return result
    finally:
        opponent.close(); own.close(); env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--baseline',choices=BASELINES)
    args=parser.parse_args()
    from tournament import prepare,save_json,sha
    manifest=prepare(args.spec)
    models=[m for m in manifest['models'] if m.get('family') in ('gylee','junhwa')]
    if not models: raise ValueError('spec must include Gylee or Junhwa RL actors')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    baselines=[args.baseline] if args.baseline else BASELINES
    identity=dict(rl_manifest=manifest,baselines=list(baselines),cpu_runner_sha256=sha(__file__),
                  games=2 if args.smoke else 50,seconds=2. if args.smoke else 200.)
    sources=[ROOT/'claude_code/my_observation.py',RUNTIME/'JSBSimAIPLib.dll',
             RUNTIME/'claude_code/env_utils.py',RUNTIME/'src/dogfight/ai/bt_action_provider.py',
             RUNTIME/'cutoff_udp_provider.py',
             ROOT/'artifacts/models/bt/vertical_deck_dive_ver01/AIP_vertical_deck_dive_ver01_release_target.dll',
             ROOT/'artifacts/models/bt/vertical_deck_dive_ver01/Rule_vertical_deck_dive_ver01.xml',
             ROOT/'artifacts/models/bt/cutoff/unreal_bt_client.exe']
    mpc=ROOT/'artifacts/models/mpc/Release_MPC_team_share'
    sources.extend([mpc/'configs/mpc.yaml',mpc/'runtime/predictor/Release/MPCJSBSim.dll'])
    sources.extend(sorted((mpc/'src/mpc').glob('*.py')))
    identity['source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in sources}
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=identity: raise ValueError('manifest changed')
    save_json(path,identity)
    tasks=[]; results=[]
    for baseline in baselines:
        for model in models:
            folder=out/baseline/model['name']
            if (folder/'result.json').exists():
                results.append(json.loads((folder/'result.json').read_text())); continue
            tasks.append((model,baseline,str(folder),identity['games'],identity['seconds'],260911))
    total=len(results)+len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        futures=[executor.submit(play,t) for t in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            save_json(out/'progress.json',dict(completed_pairs=len(results),total_pairs=total))
    save_json(out/'report.json',dict(complete=True,results=results,total_games=sum(len(r['records']) for r in results)))
    save_json(out/'progress.json',dict(complete=True,completed_pairs=total,total_pairs=total,
                                     total_games=sum(len(r['records']) for r in results)))


if __name__=='__main__': main()
