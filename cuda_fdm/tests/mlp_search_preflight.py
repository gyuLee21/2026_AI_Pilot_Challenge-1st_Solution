"""Small CUDA integrity gate for the newly approved auxiliary-ON MLP search."""
import argparse
import gc
from pathlib import Path

import torch

from cuda_fdm.finite_checks import require_finite, record_integrity_failure
from cuda_fdm.future_aux import inference_state_dict, AUX_PROTOCOL
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.search_eval import atomic_json, evaluate_pair, load_policy
from cuda_fdm.search_validate import convert_bundle
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal


def run(output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    for width, depth in ((512, 2), (512, 3), (512, 4), (1024, 4)):
        name = f"mlp{width}_d{depth}"
        print(f"[preflight] {name}: auxiliary update/save/resume/export", flush=True)
        cfg = PPOGPUConfig(architecture="mlp", hidden=(width,)*depth, gru_size=0,
            aux_pred=True, aux_coef=.1, rollout_steps=64, update_epochs=4, num_minibatches=8,
            ent_coef=.001, save_runtime=True, total_iterations=1,
            sched_period=0, milestone_period=0, exploiter_iters=0, seed=61)
        env = GpuDogfightVecEnv(32, substeps=6, seed=cfg.seed)
        tr = PPOGPUTrainer(env, cfg)
        batch = tr.collect_rollout()
        stats = tr.update(*batch[:2])
        require_finite((stats, tr.model.state_dict(), env.sim.states), f"{name}.first_update")
        tr.iteration = 1
        path = output / f"{name}.pt"
        tr.save(path)
        expected_batch = tr.collect_rollout()
        expected_labels = tr.b_aux_labels.clone()
        expected_mask = tr.b_aux_mask.clone()
        expected_stats = tr.update(*expected_batch[:2])
        expected_state = tr._snapshot_learner()
        tr.load(path)
        actual_batch = tr.collect_rollout()
        assert_tree_equal(expected_batch, actual_batch)
        assert_tree_equal(expected_labels, tr.b_aux_labels)
        assert_tree_equal(expected_mask, tr.b_aux_mask)
        actual_stats = tr.update(*actual_batch[:2])
        assert_tree_equal(expected_stats, actual_stats)
        assert_tree_equal(expected_state, tr._snapshot_learner())
        tr.iteration = 2
        tr.save(path)
        bundle = output / f"{name}_bundle"
        convert_bundle(path, bundle)
        from claude_code.action_provider import MLPActionProvider
        provider = MLPActionProvider(bundle, stochastic=False, debug_obs=False)
        x = torch.randn(128, 184, generator=torch.Generator().manual_seed(4121))
        with torch.no_grad():
            gpu = tr.model.actor_logits(x.cuda()).cpu()
            cpu = provider.model.actor_logits(x)
        require_finite((gpu, cpu), f"{name}.bundle_logits")
        torch.testing.assert_close(cpu, gpu, atol=2e-5, rtol=2e-4)
        assert torch.equal(cpu.reshape(-1, 4, 21).argmax(-1), gpu.reshape(-1, 4, 21).argmax(-1))
        assert not any("aux" in k for k in provider.model.state_dict())
        assert_tree_equal({k: v.cpu() for k, v in inference_state_dict(tr.model.state_dict(), True).items()},
                          provider.model.state_dict())
        checks.append(dict(model=name, passed=True, exact_rollout_update_resume=True,
            bundle_controls_match=True, parameters=sum(p.numel() for p in tr.model.parameters()),
            stats={k: float(v) for k, v in actual_stats.items()}))
        del env, tr, batch, expected_batch, actual_batch, expected_state, provider, cpu, gpu, stats
        gc.collect(); torch.cuda.empty_cache()
    policy = load_policy(output / "mlp512_d2.pt")
    env = GpuDogfightVecEnv(8, substeps=6, seed=9137)
    for scenario in ("three_nine", "headon"):
        match = evaluate_pair(env, policy, policy, scenario, 9137)
        scores = [r["score"] for r in match["records"]]
        assert all(scores[i] + scores[i+4] == 1 for i in range(4))
        atomic_json(output / f"paired_{scenario}.json", match)
    result = dict(passed=True, checks=checks, paired_full_episode_evaluation=True,
                  training_protocol=AUX_PROTOCOL,
                  no_wandb=True, note="N=32 correctness only; not a ranking or large-batch speed benchmark")
    atomic_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        print(run(args.output), flush=True)
    except Exception as exc:
        record_integrity_failure(args.output, exc, "mlp_search_preflight")
        raise
