"""Process-isolated scheduling of the unchanged mirrored GPU evaluator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time

from tournament import RUNTIME, prepare, run_round_robin, save_json, sha, selected_pairs


def initialize(manifest):
    global _manifest, _env, _policies, _scratch
    sys.path.insert(0, str(RUNTIME))
    import torch
    from cuda_fdm.rl_env import GpuDogfightVecEnv
    torch.set_num_threads(1)
    _manifest, _policies = manifest, {}
    _scratch = tempfile.TemporaryDirectory(prefix='league-worker-')
    _env = GpuDogfightVecEnv(
        manifest['games_per_pair'], scenario=manifest['scenario'],
        headon_distance_m=manifest['headon_distance_m'], substeps=6,
        seed=manifest['seed'], device='cuda', min_altitude_m=304.8,
        max_engage_time_s=200.)


def policy(index):
    import torch
    from cuda_fdm.search_eval import load_policy
    from policy_io import load_bundle_policy
    if index not in _policies:
        model = _manifest['models'][index]
        path = Path(model['path'])
        if model['kind'] == 'legacy184_bundle':
            result = load_bundle_policy(path, 'cuda', model['legacy_min_altitude_m'])
        elif model['kind'] == 'junhwa_actor':
            from junhwa_policy import load_junhwa_policy
            result = load_junhwa_policy(path, 'cuda')
        else:
            if model['config_from']:
                bundle = torch.load(path, map_location='cpu', weights_only=False)
                template = torch.load(model['config_from'], map_location='cpu', weights_only=False, mmap=True)
                path = Path(_scratch.name) / f'{index}.pt'
                torch.save(dict(bundle, cfg=template['cfg']), path)
            result = load_policy(path, 'cuda')
        _policies[index] = result
    return _policies[index]


def evaluate(task):
    from cuda_fdm.search_eval import evaluate_pair
    i, j, seed = task
    return evaluate_pair(_env, policy(i), policy(j), _manifest['scenario'], seed)


def pool(manifest, workers):
    return ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize, initargs=(manifest,))


class PrefetchedPairs:
    """Bound outstanding work; only the parent writes validated pair results."""
    def __init__(self, executor, tasks, width):
        self.executor, self.tasks = executor, iter(tasks)
        self.pending = {}
        for _ in range(width):
            self.advance()

    def advance(self):
        task = next(self.tasks, None)
        if task is not None:
            self.pending[task] = self.executor.submit(evaluate, task)

    def __call__(self, i, j, seed):
        result = self.pending.pop((i, j, seed)).result()
        self.advance()
        return result


def equivalent(first, second):
    # Timing is the only permitted difference, including every per-game field.
    return first['records'] == second['records'] and first['summary'] == second['summary']


def choose_workers(trials):
    valid = [row for row in trials if row['exact_match']]
    best = min(valid, key=lambda row: row['seconds'])
    baseline = next(row for row in trials if row['workers'] == 1)
    return best['workers'] if best['seconds'] < baseline['seconds'] * .9 else 1


def benchmark(manifest, output):
    n = len(manifest['models'])
    pairs = selected_pairs(manifest)[:4]
    tasks = [(i, j, manifest['seed']) for i, j in pairs]
    baseline = None
    trials = []
    for workers in (1, 2, 4):
        with pool(manifest, workers) as executor:
            # Warm imports, CUDA kernels and all trial policies in each process.
            list(executor.map(evaluate, tasks))
            started = time.perf_counter()
            results = list(executor.map(evaluate, tasks))
            elapsed = time.perf_counter() - started
        if baseline is None:
            baseline = results
        matches = all(equivalent(a, b) for a, b in zip(baseline, results))
        trials.append(dict(workers=workers, seconds=elapsed, exact_match=matches))
        save_json(Path(output)/f'benchmark_{workers}.json', dict(results=results, trial=trials[-1]))
        print(json.dumps(trials[-1]), flush=True)
    # Require a useful measured improvement before enabling concurrency.
    selected = choose_workers(trials)
    report = dict(trials=trials, selected_workers=selected,
                  manifest=manifest, parallel_runner_sha256=sha(__file__))
    save_json(Path(output)/'benchmark.json', report)
    return report


def run(manifest, output, workers, stop_file=None):
    output = Path(output)
    manifest = dict(manifest, parallel_runner_sha256=sha(__file__), workers=workers)
    existing = output/'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')) != manifest:
        raise ValueError('output belongs to a different tournament')
    tasks = [(i, j, manifest['seed']) for i,j in selected_pairs(manifest)
             if not (output/'pairs'/f'{i:03d}_{j:03d}.json').exists()]
    with pool(manifest, workers) as executor:
        evaluator = PrefetchedPairs(executor, tasks, workers)
        return run_round_robin(manifest, output, evaluator, stop_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--benchmark-result')
    parser.add_argument('--stop-file')
    args = parser.parse_args()
    manifest = prepare(args.spec)
    if args.benchmark_only:
        benchmark(manifest, args.output)
        return
    if not args.benchmark_result:
        parser.error('--benchmark-result is required before running the league')
    report = json.loads(Path(args.benchmark_result).read_text(encoding='utf-8'))
    if report['manifest'] != manifest or report['parallel_runner_sha256'] != sha(__file__):
        raise ValueError('benchmark does not match current models/code/protocol')
    run(manifest, args.output, report['selected_workers'], args.stop_file)


if __name__ == '__main__':
    main()
