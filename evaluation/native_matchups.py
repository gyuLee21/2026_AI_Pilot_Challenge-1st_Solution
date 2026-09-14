"""Explicit CPU matchups, including baseline versus baseline, in either scenario.

Uses the established native providers and identical 3-9 seed/termination rules.
Each result preserves both perspectives; no training or low-altitude takeover.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

import native_baselines as native
from tournament import prepare, save_json, sha


def independent_seed_bank(seed, games):
    import numpy as np
    if games < 2 or games % 2:
        raise ValueError('balanced side evaluation requires an even game count')
    return np.random.default_rng(seed).choice(2**31-1, size=games, replace=False)


def provider(model, output):
    if model.get('kind') == 'native':
        return native.audited_provider(native.baseline_provider(model['name'], output))
    # Keep the established actor observation/action path; extend only bundle loading.
    original = native.load_actor
    if model.get('kind') == 'legacy184_bundle':
        from policy_io import load_bundle_policy
        native.load_actor = lambda m: load_bundle_policy(m['path'], 'cpu', m['legacy_min_altitude_m'])
    try:
        return native.rl_provider(model)
    finally:
        native.load_actor = original


def play(task):
    a, b, folder, scenario, distance, games, seconds, seed = task[:8]
    options = task[8] if len(task)>8 else {}
    folder = Path(folder)
    native.setup()
    import numpy as np
    from claude_code.env_utils import make_env
    own = provider(a, folder/'left')
    opponent = provider(b, folder/'right')
    env = make_env(overrides=dict(target_mode='fixed', ownship_control_mode='rl',
        step_ratio=6, max_engage_time=seconds, episode_step_limit=2001,
        min_altitude=304.8, randomize_start_side=False,
        scenario_b_prob=float(scenario == 'headon'), start_headon_distance_ft=distance/.3048,
        artifacts_dir=str(folder/'trajectories')), runner_index=f'power_{os.getpid()}')
    env._ownship_action_provider = own
    env._target_action_provider = opponent
    independent = options.get('independent_seeds', False)
    total_games = options.get('total_games', games)
    offset = options.get('game_offset', 0)
    if independent:
        # Unique within each matchup; the same bank across candidates permits
        # fair candidate-to-candidate comparisons without mirrored IC reuse.
        seeds = independent_seed_bank(seed, total_games)
    else:
        seeds = np.random.default_rng(seed).integers(0, 2**31-1, size=games//2)
    rows = []
    try:
        for k in range(games):
            game_index = offset+k
            swapped = bool(game_index % 2)
            episode_seed = int(seeds[game_index if independent else k//2])
            env._apply_start_side(swapped, swapped)
            env.reset(seed=episode_seed)
            if env._initial_scenario_kind != ('B' if scenario == 'headon' else 'A'):
                raise ValueError('wrong initial scenario')
            ia = env._sim.get_state()[:7].tolist()
            ib = env._target_sim.get_state()[:7].tolist()
            if scenario == 'headon':
                requested = [np.array([s._init_pos_n,s._init_pos_e,s._init_pos_d])
                             for s in (env._sim,env._target_sim)]
                np.testing.assert_allclose(np.linalg.norm(requested[0]-requested[1]), distance, atol=.03, rtol=0)
                # DLL initialization already advances ~one 60Hz tick. Check
                # requested separation exactly; bound measured-state motion.
                travel = sum(s._init_speed*float(s.get_state()[41])
                             for s in (env._sim,env._target_sim))
                np.testing.assert_allclose(np.linalg.norm(np.array(ia[:3])-ib[:3]), distance, atol=travel+.5, rtol=0)
                np.testing.assert_allclose([ia[5] % 360, ib[5] % 360], [0,180] if swapped else [180,0], atol=.01)
            if swapped and not independent:
                np.testing.assert_allclose(ia, rows[-1]['initial_opp'], rtol=0, atol=1e-8)
                np.testing.assert_allclose(ib, rows[-1]['initial_own'], rtol=0, atol=1e-8)
            started = time.perf_counter()
            for steps in range(1, 2002):
                _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                if terminated or truncated:
                    break
            else:
                raise RuntimeError('episode did not finish')
            hp, target_hp = float(info['ownship_health']), float(info['target_health'])
            state, target = env._sim.get_state(), env._target_sim.get_state()
            alt, target_alt = -float(state[2]), -float(target[2])
            if not np.isfinite([hp, target_hp, alt, target_alt]).all():
                raise FloatingPointError('terminal state')
            audit = {}
            for label, p in (('left',own), ('right',opponent)):
                if hasattr(p, 'decisions'):
                    if p.calls != steps*6 or p.decisions != (p.calls-1)//(60//p.hz)+1:
                        raise RuntimeError(f'{label} {p.hz}Hz decision count mismatch')
                    audit[label] = dict(physical_frames=p.calls,decisions=p.decisions,hz=p.hz)
                if hasattr(p, 'tick'):
                    if p.tick != steps*6:
                        raise RuntimeError(f'{label} RL physical frame count mismatch')
                    audit[label] = dict(physical_frames=p.tick,decisions=(p.tick+5)//6,hz=10)
                if hasattr(p, 'fallbacks'):
                    if p.calls < steps or p.fallbacks > max(12,p.calls*.2):
                        raise RuntimeError(f'{label} native provider unhealthy')
                    audit[label] = dict(calls=p.calls, fallbacks=p.fallbacks, updates=p.updates)
            dead, target_dead = hp <= 0 or alt < 304.8, target_hp <= 0 or target_alt < 304.8
            score = (float(target_dead) if dead != target_dead else .5 if dead else
                     1. if hp > target_hp+1e-9 else 0. if hp < target_hp-1e-9 else .5)
            rows.append(dict(score=score, damage_diff=hp-target_hp, hp=hp, target_hp=target_hp,
                crash=alt<304.8, crash_b=target_alt<304.8, ic_pair=None if independent else k//2,
                game_index=game_index, role_swapped=swapped, end=info.get('end_condition'),
                destroyed=hp<=0, destroyed_b=target_hp<=0, altitude_m=alt, target_altitude_m=target_alt,
                seed=episode_seed, initial_own=ia, initial_opp=ib, duration_sec=steps*.1,
                wall_sec=time.perf_counter()-started, native_audit=audit))
            save_json(folder/'progress.json', dict(left=a['name'],right=b['name'],records=rows,
                completed_games=len(rows),total_games=games,complete=False))
            print(f'{scenario}: {a["name"]} vs {b["name"]}: {k+1}/{games}', flush=True)
        result = dict(left=a['name'],right=b['name'],scenario=scenario,records=rows,complete=True)
        save_json(folder/'result.json', result)
        return result
    finally:
        opponent.close()
        own.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    specpath = Path(args.spec).resolve()
    spec = json.loads(specpath.read_text())
    manifest = prepare(specpath)
    models = {m['name']:m for m in manifest['models']}
    models.update({n:dict(name=n,kind='native') for n in native.BASELINES})
    out = Path(args.output).resolve()
    identity = dict(manifest=manifest, pairs=spec['cpu_pairs'], smoke=args.smoke,
                    runner_sha256=sha(__file__), native_sha256=sha(native.__file__),
                    fdm_wrapper_sha256=sha(native.RUNTIME/'FighterSim.py'),
                    cutoff_bridge_sha256=sha(native.RUNTIME/'cutoff_udp_provider.py'),
                    fdm_dll_sha256=sha(native.RUNTIME/'JSBSimAIPLib.dll'),
                    observation_sha256=sha(native.ROOT/'claude_code/my_observation.py'))
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != identity:
        raise ValueError('immutable manifest mismatch')
    save_json(out/'manifest.json', identity)
    tasks, results = [], []
    for left,right in spec['cpu_pairs']:
        folder=out/f'{left}__{right}'
        if (folder/'result.json').exists():
            results.append(json.loads((folder/'result.json').read_text()))
        else:
            tasks.append((models[left],models[right],str(folder),manifest['scenario'],
                manifest['headon_distance_m'],2 if args.smoke else manifest['games_per_pair'],
                2. if args.smoke else 200.,manifest['seed']))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        for future in as_completed([executor.submit(play,t) for t in tasks]):
            results.append(future.result())
            save_json(out/'progress.json',dict(completed_pairs=len(results),total_pairs=len(spec['cpu_pairs'])))
    save_json(out/'report.json',dict(complete=True,results=results))


if __name__ == '__main__':
    main()
