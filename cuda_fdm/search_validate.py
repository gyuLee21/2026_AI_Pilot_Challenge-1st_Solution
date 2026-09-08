"""Regression gates and two-agent local CPU latency checks for architecture search."""
import argparse
import json
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch

from claude_code.model import MLPDiscreteActorCritic, GRUDiscreteActorCritic, load_bundle
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer, build_actor_critic
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.search_eval import atomic_json, load_policy, evaluate_pair
from cuda_fdm.finite_checks import require_finite, require_finite_training_stats, record_integrity_failure


def convert_bundle(checkpoint, destination):
    from cuda_fdm.gpu_ckpt_to_bundle import main
    old = sys.argv
    try:
        sys.argv = ["gpu_ckpt_to_bundle", "--ckpt", str(checkpoint), "--output-dir", str(destination)]
        main()
    finally:
        sys.argv = old


@torch.inference_mode()
def latency(checkpoint, output, count=2000):
    """Two serial agent pipelines: reconstruction, observation, normalization, policy, JSON.

    Excludes transport/server/network latency; p99<50ms reserves half the 10Hz budget.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from claude_code.my_observation import build_observation, OBSERVATION_MODE
    from claude_code.action_provider import MLPActionProvider
    from dogfight.unreal import ProviderCommandPolicy
    from dogfight.unreal.client import RemoteClientContext, PlaneSnapshot
    from dogfight.unreal.protocol import PlaneInfo, Rotation3D, Vector3D
    torch.set_num_threads(1)
    durations = []
    own = np.array([3500., -350., -2500., 0., 0., 90., 230., 0., 0.])
    enemy = np.array([3500., 350., -2500., 0., 0., 270., 230., 0., 0.])
    def plane(s, identity, frame):
        info = PlaneInfo(index=frame, plane_id=identity,
                         position=Vector3D(float(s[0]), float(s[1]), float(-s[2])),
                         rotation=Rotation3D(*map(float, s[3:6])),
                         velocity=Vector3D(*map(float, s[6:9])))
        return PlaneSnapshot(True, identity, frame, info)
    # Include the actual bundle loader/provider/60Hz packet cache, not a parallel
    # hand-written path that could benchmark the wrong reconstruction contract.
    with tempfile.TemporaryDirectory(prefix="cuda_submission_latency_") as tmp:
        convert_bundle(checkpoint, tmp)
        policies = [ProviderCommandPolicy(MLPActionProvider(tmp, stochastic=True, debug_obs=False),
                    observation_mode=OBSERVATION_MODE, observation_fn=build_observation,
                    action_repeat=6) for _ in range(2)]
        for k in range(count + 100):
            own[3] = 15 * np.sin(k / 30.)
            enemy[3] = -own[3]
            if k % 500 == 0:
                for policy in policies:
                    policy.reset(None)
            contexts = [RemoteClientContext(plane_id=j + 1, frame_index=k * 6,
                        own_plane=plane(a, j + 1, k * 6), enemy_plane=plane(b, 2 - j, k * 6))
                        for j, (a, b) in enumerate(((own, enemy), (enemy, own)))]
            start = time.perf_counter_ns()
            for policy, context in zip(policies, contexts):
                cmd = policy.compute_command(context)
                json.dumps([cmd.roll_cmd, cmd.pitch_cmd, cmd.yaw_cmd, cmd.throttle_cmd], allow_nan=False)
            if k >= 100:
                durations.append((time.perf_counter_ns() - start) / 1e6)
            # Five intervening packets exercise the real action_repeat cache but
            # are excluded from decision-frame latency (no policy inference).
            for frame in range(1, 6):
                for policy, context in zip(policies, contexts):
                    context.frame_index = k * 6 + frame
                    policy.compute_command(context)
    values = np.asarray(durations)
    result = dict(checkpoint=str(checkpoint), calls=count, agents=2, cpu_threads=1,
                  stochastic=True,
                  p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                  p99_ms=float(np.percentile(values, 99)), max_ms=float(values.max()),
                  passed=bool(np.percentile(values, 99) < 50 and values.max() < 100),
                  observation_contract=policies[0].action_provider.metadata.get("observation_contract"),
                  scope="actual exported bundle + ProviderCommandPolicy + private reconstructor + two decision-frame inferences + JSON; excludes network/server")
    atomic_json(output, result)
    return result


def preflight(output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    # Default GRU must retain the same initialization/state dictionary exactly.
    torch.manual_seed(741)
    old = GRUDiscreteActorCritic(184, hidden=(64, 64, 64), gru_size=16)
    torch.manual_seed(741)
    new = build_actor_critic(obs_dim=184, hidden=(64, 64, 64), gru_size=16)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    for architecture, depth in (("gru", 2), ("gru", 1), ("mlp", 2)):
        label = f"{architecture}_depth{depth}"
        cfg = PPOGPUConfig(total_iterations=1, rollout_steps=8, update_epochs=1,
                           num_minibatches=2, architecture=architecture,
                           hidden=(64, 64, 64) if architecture == "gru" else (64, 64),
                           gru_size=16 if architecture == "gru" else 0,
                           encoder_depth=depth, save_runtime=True, sched_period=0,
                           milestone_period=0, exploiter_iters=0, selfplay_gate_threshold=1.1)
        env = GpuDogfightVecEnv(8, seed=cfg.seed)
        tr = PPOGPUTrainer(env, cfg)
        if architecture == "mlp":
            reference = MLPDiscreteActorCritic(184, hidden=cfg.hidden).cuda()
            reference.load_state_dict(tr.model.state_dict())
            x = torch.randn(4, 184, device="cuda")
            assert torch.equal(reference.act_deterministic(x).long(), tr.model.act(x, sample=False)[0])
        if architecture == "mlp":
            stats = tr.train()
            require_finite_training_stats(stats[0])
        else:
            # Historical GRU bundles remain evaluable, not trainable by the
            # newly selected MLP-only updater.
            adv, ret, _ = tr.collect_rollout()
            require_finite((adv, ret), f"{label}.evaluation_rollout")
        require_finite(env.sim.states, f"{label}.physical_state")
        require_finite(tr.model.state_dict(), f"{label}.parameters")
        ckpt = output / f"{label}.pt"
        tr.save(ckpt)
        adv, ret, _ = tr.collect_rollout()
        expected = {k: getattr(tr, k).clone() for k in ("b_obs", "b_act", "b_rew", "b_done")}
        tr.load(ckpt)
        adv2, ret2, _ = tr.collect_rollout()
        require_finite((expected, adv, ret, adv2, ret2), f"{label}.resume_rollout")
        for key, value in expected.items():
            require_finite(getattr(tr, key), f"{label}.restored.{key}")
            torch.testing.assert_close(value, getattr(tr, key), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(adv, adv2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ret, ret2, atol=1e-5, rtol=1e-5)
        bundle = output / f"{label}_bundle"
        convert_bundle(ckpt, bundle)
        cpu, meta = load_bundle(bundle)
        gpu, _ = load_policy(ckpt)
        x = torch.randn(16, 184)
        if architecture == "gru":
            ch = cpu.initial_state(16)
            gh = gpu.initial_state(16, "cuda")
            for _ in range(3):
                cl, ch = cpu._step_logits(x, ch)
                gl, gh = gpu._step_logits(x.cuda(), gh)
                require_finite((cl, ch, gl, gh), f"{label}.bundle_output")
                torch.testing.assert_close(cl, gl.cpu(), atol=2e-5, rtol=2e-4)
                assert torch.equal(cl.argmax(-1), gl.cpu().argmax(-1))
            starts = torch.ones(16)
            cl, _ = cpu._step_logits(x, ch, starts)
            zero, _ = cpu._step_logits(x, cpu.initial_state(16), starts)
            torch.testing.assert_close(cl, zero, atol=0, rtol=0)
        else:
            cpu_logits, gpu_logits = cpu.actor_logits(x), gpu.actor_logits(x.cuda()).cpu()
            require_finite((cpu_logits, gpu_logits), f"{label}.bundle_output")
            torch.testing.assert_close(cpu_logits, gpu_logits, atol=2e-5, rtol=2e-4)
        results.append(dict(model=label, passed=True, parameters=sum(p.numel() for p in cpu.parameters()),
                            updater_tested=architecture == "mlp"))
        del tr, env, gpu, cpu
        torch.cuda.empty_cache()
    policy = load_policy(output / "mlp_depth2.pt")
    eval_env = GpuDogfightVecEnv(8, seed=891)
    for scenario in ("three_nine", "headon"):
        match = evaluate_pair(eval_env, policy, policy, scenario, 891)
        scores = [r["score"] for r in match["records"]]
        assert all(scores[i] + scores[i+4] == 1 for i in range(4)), scores
        assert match["summary"]["games"] == 8
        atomic_json(output / f"paired_{scenario}.json", match)
    atomic_json(output / "result.json", {"passed": True, "checks": results,
                                          "paired_full_episode_evaluation": True})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint")
    args = ap.parse_args()
    try:
        if args.checkpoint:
            result = latency(args.checkpoint, args.output)
            print(result, flush=True)
            if not result["passed"]:
                raise SystemExit(2)
        else:
            print(preflight(args.output), flush=True)
    except (FloatingPointError, AssertionError, ValueError) as exc:
        folder = Path(args.output).parent if args.checkpoint else Path(args.output)
        record_integrity_failure(folder, exc, "validation")
        raise
