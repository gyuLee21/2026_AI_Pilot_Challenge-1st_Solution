"""Process-isolated payoff evaluations; archive writes belong to the parent only.

Each worker calls the same stochastic evaluator with the same per-match seed
and lane count. No batched-policy kernels, changed sampling, or shared RNG.
Workers exist only for one row, releasing GPU memory before PPO continues.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from types import SimpleNamespace

import torch


def evaluator_spec(trainer):
    return dict(
        cfg={k: getattr(trainer.cfg, k) for k in ('device', 'seed', 'num_bins')},
        model_kwargs=dict(trainer._model_kwargs), obs_dim=trainer.obs_dim,
        act_dim=trainer.act_dim,
        env={k: getattr(trainer.env, k) for k in (
            'substeps', 'min_altitude_m', 'max_engage_time_s', 'reward_cfg', 'alt_hunt_coef')},
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled())


def initialize(spec, left_bundle):
    global _trainer, _left
    from cuda_fdm.ppo_gpu import PPOGPUTrainer, make_action_grid
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision(spec['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = spec['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = spec['cudnn_tf32']
    torch.use_deterministic_algorithms(spec['deterministic'])
    _trainer = PPOGPUTrainer.__new__(PPOGPUTrainer)
    _trainer.cfg = SimpleNamespace(**spec['cfg'])
    _trainer.env = SimpleNamespace(**spec['env'])
    _trainer.obs_dim, _trainer.act_dim = spec['obs_dim'], spec['act_dim']
    _trainer._model_kwargs = spec['model_kwargs']
    _trainer._league_eval_envs = {}
    _trainer.grid = make_action_grid(_trainer.cfg.num_bins, device=_trainer.cfg.device)
    _left = _trainer._frozen_policy(left_bundle)


def evaluate(task):
    identity, right_bundle, blocks, seed = task
    right = _trainer._frozen_policy(right_bundle)
    result = _trainer._run_clean_paired_evaluation(_left, right, blocks, seed)
    return identity, result


def evaluate_row(trainer, left_bundle, identities, *, seed_block, workers):
    """Return a complete row or raise, without changing archive or main runtime."""
    if not 2 <= workers <= 4:
        raise ValueError('parallel payoff workers must be between 2 and 4')
    if not identities:
        return {}
    # Parent load_policy retains SHA/protocol/identity validation before dispatch.
    # At most one outstanding task per worker bounds CPU shared-memory handles.
    saved = trainer._runtime_state()
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(identities)),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,
                initargs=(evaluator_spec(trainer), left_bundle)) as executor:
            pending = []
            for identity in identities:
                if len(pending) >= workers:
                    key, result = pending.pop(0).result()
                    results[key] = result
                task = (identity, trainer.archive.load_policy(identity),
                        trainer.cfg.league_payoff_games, seed_block)
                pending.append(executor.submit(evaluate, task))
            for future in pending:
                key, result = future.result()
                results[key] = result
        if set(results) != set(identities):
            raise RuntimeError('parallel payoff row is incomplete')
        return results
    finally:
        trainer._restore_runtime(saved)
