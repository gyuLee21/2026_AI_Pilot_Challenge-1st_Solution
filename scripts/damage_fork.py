"""Explicit 67500 policy fork with the 70000 league and 3000 extra updates."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cuda_fdm.warm_start import prepare_checkpoint, finish_pending

RUN = ROOT/'artifacts/models/rl/3-9/damage10_from67500_2500'
MANIFEST = ROOT.parent/'evaluation_results/league_20260911_separated/three_nine/manifest.json'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding='utf-8')


def main_lr(additional, pulse):
    if pulse and pulse['start_additional'] <= additional <= pulse['end_additional']:
        return float(pulse['lr'])
    return 5e-5


def prepare():
    if RUN.exists():
        raise RuntimeError('fork directory already exists; use --run to continue prepared fork')
    models = json.loads(MANIFEST.read_text())['models']
    source = next(m for m in models if m['name']=='three_nine_archive_67500')
    donor = next(m for m in models if m['name']=='three_nine_iter_70000')
    for entry in (source, donor):
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    c = torch.load(donor['path'], map_location='cpu', weights_only=False)
    b = torch.load(source['path'], map_location='cpu', weights_only=False)
    b['provenance'] = dict(policy_source_iteration=67500, league_clock_origin=70000,
                           source_global_step=c['global_step'], policy_sha256=source['sha256'],
                           donor_sha256=donor['sha256'], optimizer='reset')
    fork = prepare_checkpoint(c, b)
    fork['training_stop'] = None
    fork.pop('training_deadline', None)
    RUN.mkdir(parents=True)
    league = RUN/'league'
    (league/'policies').mkdir(parents=True)
    old = Path(c['league_archive']['root'])
    for r in c['league_archive']['records']:
        src, dst = old/'policies'/r['file'], league/'policies'/r['file']
        if not dst.exists():
            shutil.copy2(src, dst)
        assert dst.stat().st_size==src.stat().st_size
    (RUN/'vnext_control').mkdir()
    shutil.copy2(Path(donor['path']).parent/'vnext_control/decisions.jsonl', RUN/'vnext_control/decisions.jsonl')
    fork['cfg']['league_dir'] = str(league)
    fork['league_archive']['root'] = str(league)
    assert not fork['actor_opt']['state'] and not fork['critic_opt']['state']
    assert all(torch.equal(v, fork['model'][k]) for k,v in b['model'].items())
    assert all(torch.equal(v, fork['norm'][k]) for k,v in b['norm'].items())
    torch.save(fork, RUN/'start.pt')
    write(RUN/'experiment.json', dict(**b['provenance'], damage_scale=2,
          altitude_settlement_scale=2, actor_lr=5e-5, critic_lr=5e-5,
          entropy_coef=1e-4, rollout=96, additional_iterations=3000,
          source_policy_iteration=67500, displayed_final_iteration=70500,
          internal_league_start=70000, internal_league_end=73000,
          reason_for_internal_clock='Preserve dates of imported 70000 league evidence',
          new_exploiters=True, exploiter_at_additional_iterations=[1500,2000],
          source_checkpoint_unchanged=True))
    print('[prepare] verified policy, normalization, fresh optimizers, copied league', flush=True)


def run():
    from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
    from cuda_fdm.rl_env import GpuDogfightVecEnv
    from cuda_fdm.league_vnext.contracts import VNextConfig
    from cuda_fdm.league_vnext.shadow import VNextShadowController
    from cuda_fdm.league_vnext.live_adapter import VNextMilestoneAdapter, LIVE_ADAPTER_PROTOCOL
    from cuda_fdm.training_control import RunLease
    import wandb
    torch.set_num_threads(1)
    save = RUN/'checkpoint.pt'
    source = save if save.exists() else RUN/'start.pt'
    c = torch.load(source, map_location='cpu', weights_only=False)
    pulse = json.loads((RUN/'experiment.json').read_text()).get('lr_pulse')
    cfg = dict(c['cfg'])
    cfg.update(total_iterations=73000, rollout_steps=64, lr=3e-4, critic_lr=3e-4,
               ent_coef=.001, sched_lr_floor=5e-5, sched_ent_floor=1e-4,
               sched_rollout_cap=96, exploiter_iters=1000, exploiter_period=500,
               league_dir=str(RUN/'league'))
    with RunLease(RUN, resume=True):
        env = GpuDogfightVecEnv(4096, scenario='three_nine', substeps=6, seed=0, device='cuda')
        env.reward_cfg.update(c['training_objective']['reward'])
        env.reward_cfg.update(damage_scale=2., altitude_settlement_scale=2., altitude_terminal_mode='result_remaining_hp')
        trainer = PPOGPUTrainer(env, PPOGPUConfig(**cfg))
        trainer.load(source, allow_schedule_change=True, allow_objective_change=True)
        state = copy.deepcopy(c['vnext_control'])
        state['config']['target_iteration'] = 73000
        state['wallclock_budget']['target_iteration'] = 73000
        controller = VNextShadowController(RUN/'vnext_control',
                       config=VNextConfig.from_dict(state['config']), state=state)
        adapter = VNextMilestoneAdapter(controller, release_receipt={
            'protocol':LIVE_ADAPTER_PROTOCOL, 'approved':True, 'stage':1})
        trainer.vnext_milestone_adapter = adapter
        trainer.vnext_control_state = controller.state_dict()
        wb = wandb.init(project='AIP contest', entity='leeai021213-ajou-university',
             id='d10f67500-20260911', resume='allow', name='3-9-67500-plus3000-damage2-from926',
             dir=str(RUN), config=json.loads((RUN/'experiment.json').read_text()),allow_val_change=True)
        print('[wandb]', wb.url, flush=True)
        for metric in ('win_rate','mean_return','lr','entropy_coef','own_crash_rate','approx_kl','pool_size','policy_iteration'):
            wb.define_metric(metric,step_metric='additional_iteration')
        try:
            finish_pending(trainer, save)
            if not (RUN/'opponents_ready.json').exists():
                # 70000 already occupies recent; explicitly seat 68000/69000
                # through the same payoff completion/Nash/roster selector.
                late = [791, 801]
                assert [trainer.archive.records[i]['iteration'] for i in late]==[68000,69000]
                active = trainer.pool.active_entries()
                assert any(e.get('archive_id')==811 for e in active)
                current = trainer._last_milestone_archive_id
                cores = [e['archive_id'] for e in active if e['role']=='core']
                incumbent = [e['archive_id'] for e in active if e['role']=='challenger'][:1]
                selected = [current, *cores, *incumbent, *late]
                assert len(selected)<=20 and len(selected)==len(set(selected))
                for identity in late:
                    r=trainer.archive.records[identity]
                    r.update(admitted=True, admission_status='probationary')
                    r.setdefault('metrics', {})['manual_fork_opponent']=True
                # Bounded bootstrap prefixes, as in the existing league import.
                for size in range(2,len(selected)+1):
                    adapter._complete_active_solver_game(trainer, policy_ids=selected[:size],
                        iteration=trainer.iteration,current_id=current,candidate_ids=set())
                p=controller.config.payoff_graph
                solution=controller.graph.conservative_nash(selected,minimum_blocks=p.solver_paired_blocks,
                    evaluator_protocol=p.evaluator_protocol,scenario_bank_versions=p.solver_compatible_scenario_banks)
                adapter.refresh_roster(trainer,iteration=trainer.iteration,current_id=current,
                    forced_challenger_ids=late,selected_ids=selected,nash_override=solution)
                controller.last_solver_ids=selected
                controller.last_index['solver_eligible_ids']=selected
                adapter.sync_membership_metadata(trainer,solver_ids=selected,nash_override=solution)
                trainer._refresh_weights()
                trainer._reset_env_state()
                adapter.assert_no_stranded_probationary(trainer)
                ids=[e.get('archive_id') for e in trainer.pool.active_entries()]
                assert all(i in ids for i in [791,801,811])
                trainer.vnext_control_state=controller.state_dict()
                trainer.save(save)
                write(RUN/'opponents_ready.json',dict(active_ids=ids,solver_ids=selected))
                print('[pool] 68000 / 69000 / 70000 active',flush=True)
            def callback(s):
                assert abs(s.extra['lr']-main_lr(trainer.iteration-70000,pulse))<1e-10 and abs(s.extra['ent_coef']-1e-4)<1e-10
                assert trainer.env.reward_cfg['damage_scale']==2 and trainer.env.reward_cfg['altitude_settlement_scale']==2 and trainer.cfg.rollout_steps==96
                row=asdict(s)
                additional=trainer.iteration-70000
                row.update(additional_iteration=additional,policy_iteration=67500+additional)
                with (RUN/'metrics.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row,default=str,allow_nan=False)+'\n')
                health=dict(s.extra, approx_kl=s.approx_kl, clipfrac=s.clipfrac,
                    entropy=s.entropy, fresh_fraction=1., policy_lag=0,
                    elapsed_sec=s.elapsed_sec, explained_variance=s.explained_variance)
                controller.observe_iteration(trainer.iteration,health)
                trainer.vnext_control_state=controller.state_dict()
                wb.log(dict(additional_iteration=additional,policy_iteration=67500+additional,
                    win_rate=s.win_rate,mean_return=s.mean_return,lr=s.extra['lr'],
                    entropy_coef=s.extra['ent_coef'],own_crash_rate=s.extra['own_alt_event_rate_v2'],
                    approx_kl=row.get('approx_kl'),pool_size=trainer.pool.size()))
                print(f'[main] +{additional}/3000 policy_iter={67500+additional} wr={s.win_rate:.3f} crash={s.extra["own_alt_event_rate_v2"]:.3f} kl={s.approx_kl:.5f} lr={s.extra["lr"]} ent_coef={s.extra["ent_coef"]}',flush=True)
                if additional%100==0:
                    trainer.save(save)
                    shutil.copy2(save,RUN/f'additional_{additional:04d}.pt')
                if pulse and additional == pulse['end_additional']:
                    trainer.save(save)
                    shutil.copy2(save,RUN/'after_lr_pulse.pt')
                    print('[lr-pulse] 100 updates complete; base LR resumes next iteration',flush=True)
            def before(i):
                if (RUN/'STOP').exists():
                    return False
                trainer.cfg.sched_lr_floor=main_lr(i-70000,pulse)
                trainer.cfg.exploiter_iters=1000 if i-70000 in (1500,2000) else 0
                return True
            def side(ms,i,metrics):
                with (RUN/'exploiter_metrics.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(main_iteration=ms,side_iteration=i,metrics=metrics),default=str)+'\n')
                wb.log({f'exploiter/{key}':value for key,value in metrics.items() if isinstance(value,(int,float))})
            trainer.train(start_iteration=trainer.iteration+1,on_iteration=callback,
                          before_iteration=before,on_exploiter_iter=side)
            trainer.save(save)
            write(RUN/'status.json',dict(completed=trainer.iteration==73000,
                  additional_iterations=trainer.iteration-70000,wandb_url=wb.url))
        finally:
            wb.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()
