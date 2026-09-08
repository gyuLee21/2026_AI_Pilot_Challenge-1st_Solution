"""Fused four-mode parity, unchanged upstream kernel, reset and main restoration."""
import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch

from claude_code import my_reward as MR
from cuda_fdm.finite_checks import require_finite, record_integrity_failure
from cuda_fdm.ic import build_seed_vector
from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.ppo_gpu import PPOGPUConfig, PPOGPUTrainer
from cuda_fdm.reward_modes import REWARD_CONTRACT
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.tests.altitude_reward_val import cpu_transition, full_states
from cuda_fdm.tests.pool_assigned_val import assert_tree_equal


def seeds_for(env, heights):
    return np.stack([build_seed_vector(**env._ic_dict(
        0. if a % 2 == 0 else 700., 0., -altitude, 0., 200.))
        for a, altitude in enumerate(heights)])


@torch.no_grad()
def fused_parity():
    nenv = 8
    env = GpuDogfightVecEnv(nenv, seed=541)
    # Compile the frozen pre-datum-fix kernel IN MEMORY, never edit either source.
    # Reward formulas must remain numerically equivalent away from the boundary,
    # but bitwise identity is no longer a valid contract: the official direct-MSL
    # altitude intentionally changes the state supplied to those same formulas.
    import cuda_rt
    upstream = next(p for p in Path(__file__).resolve().parents
                    if (p / "scripts/continue_depth_final_width.py").is_file())
    cap = torch.cuda.get_device_capability()
    baseline_kernel = cuda_rt.Kernel(cuda_rt.compile_ptx(
        (upstream / "cuda_fdm/gen/obs_kernel.cu").read_text(encoding="utf-8"),
        f"compute_{cap[0]}{cap[1]}"), "advance_kernel")
    baseline = BatchObsReward(nenv)
    class LegacyAdvanceABI:
        def launch(self, grid, block, args, **kwargs):
            # Frozen upstream also predates obs214 acceleration buffers.
            legacy = args[:8] + args[10:-3]
            baseline_kernel.launch(grid, block, legacy[:-9] + legacy[-8:], **kwargs)
    baseline._k_adv = LegacyAdvanceABI()
    initial_heights = np.array([[1000., 1000.]] * nenv).flatten()
    final_heights = np.array([[299., 1000.], [1000., 299.], [299., 299.],
                             [1000., 1000.], [1000., 500.], [1000., 2000.],
                             [305., 305.], [1000., 1000.]]).flatten()
    initial_seeds, final_seeds = seeds_for(env, initial_heights), seeds_for(env, final_heights)
    hp = torch.tensor([.3, .8, .3, .8, .4, .7, 0., .8,
                       .6, .7, .7, .4, .3, .2, .8, .6], dtype=torch.float64, device="cuda")
    controls = torch.zeros(nenv*2, 4, dtype=torch.float64, device="cuda")
    controls[:, 3] = .8
    results = []
    legacy_reward_max_abs = 0.0
    for mode in (0, 1, 2, 3):
        for shaping in (0., .0001):
            cfg = dict(MR.MY_REWARD_CONFIG, shaping_reward_scale=shaping)
            env.sim.load_seed(initial_seeds)
            initial_s9 = env.state9_flat().clone()
            env.obr.reset_all()
            env.obr.hp.copy_(hp)
            env.obr.kernel_init_reward_state(env.sim.states)
            ref = BatchObsReward(nenv, enable_kernel=False)
            ref.hp.copy_(hp)
            ref.initialize_reward_state(initial_s9, cfg=cfg)
            env.obr.t_sec[-1] = ref.t_sec[-1] = 199.9
            if mode in (0, 1):
                baseline.restore(env.obr.clone_state())
            env.sim.load_seed(final_seeds)
            current_s9 = env.state9_flat().clone()
            ref.push_actions(controls); ref.advance(current_s9)
            got, term, trunc = env.obr.kernel_advance(env.sim.states, controls,
                cfg=cfg, reward_mode=mode, alt_hunt_coef=5.)
            if mode in (0, 1):
                old_reward, old_term, old_trunc = baseline.kernel_advance(env.sim.states, controls,
                    cfg=cfg, reward_mode=mode, alt_hunt_coef=5.)
                legacy_reward_max_abs = max(
                    legacy_reward_max_abs, float((got - old_reward).abs().max()))
                # These fixtures are deliberately away from the disputed cutoff.
                # Direct MSL intentionally changes altitude-derived shaping, so
                # the historical reward is diagnostic only.  Current CUDA is
                # checked against independent Torch and CPU formulas below.
                require_finite(old_reward, "legacy datum diagnostic reward")
                torch.testing.assert_close(term, old_term, atol=0, rtol=0)
                torch.testing.assert_close(trunc, old_trunc, atol=0, rtol=0)
            expected = ref.compute_reward(current_s9, term.bool(), cfg=cfg,
                truncated_env=trunc.bool(), reward_mode=mode, alt_hunt_coef=5.)
            require_finite((got, expected, env.sim.states), "altmix CUDA/Torch")
            torch.testing.assert_close(got, expected, atol=2e-7, rtol=2e-7)
            torch.testing.assert_close(env.obr.hp, ref.hp, atol=1e-9, rtol=0)
            torch.testing.assert_close(env.obr.prev_alt_log, ref.prev_alt_log, atol=1e-9, rtol=0)
            assert term.cpu().tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
            assert trunc.cpu().tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
            # Independent CPU full-state contract for each perspective.
            errors = []
            for a in range(2*nenv):
                e, p = a//2, a ^ 1
                t0 = 199.9 if e == nenv-1 else 0.
                initial_full = full_states(initial_s9[[a, p]].cpu(), hp[[a, p]].cpu(), t0)
                current_full = full_states(current_s9[[a, p]].cpu(), ref.hp[[a, p]].cpu(), t0+.1)
                end = (MR._OWNSHIP_ALT_END if current_s9[a, 2] > -300 else
                       MR._TARGET_ALT_END if current_s9[p, 2] > -300 else "")
                expected_cpu, _ = cpu_transition(initial_full, current_full, mode, shaping,
                    tuple(ref.hp_loss[[a, p]].cpu().tolist()), bool(term[e]), bool(trunc[e]), end)
                errors.append(abs(expected_cpu-float(got[a])))
            assert max(errors) < 2e-7, errors
            if mode == 0 and shaping == 0:
                # No firing on the sloped first two pairs: remaining HP only.
                torch.testing.assert_close(got[:4], torch.tensor([-3., 3., 8., -8.],
                    device="cuda", dtype=torch.float64), atol=1e-9, rtol=0)
            results.append(dict(mode=mode, shaping=shaping, max_cpu_error=max(errors),
                max_torch_error=float((got-expected).abs().max()), rewards=got.cpu().tolist()))

    # Actual autoreset must seed the NEW target's altitude, with no cross-episode delta.
    env.ic_pool_size = 8
    env._ensure_ic_pool()
    env.sim.load_seed(final_seeds)
    env.reward_mode = 1
    env.obr.reset_all(); env.obr.kernel_init_reward_state(env.sim.states)
    obs, reward, done, info = env.step(controls)
    assert bool(done[:3].all())
    new_logs = env.obr._altitude_log(env.state9_flat())
    torch.testing.assert_close(env.obr.prev_alt_log, new_logs, atol=1e-9, rtol=0)
    # Freeze current post-reset states and evaluate the next interval without physics.
    new_reward, _, _ = env.obr.kernel_advance(env.sim.states, controls,
        cfg=dict(MR.MY_REWARD_CONFIG, damage_scale=0.), reward_mode=1)
    torch.testing.assert_close(new_reward[done.repeat_interleave(2)],
        torch.zeros_like(new_reward[done.repeat_interleave(2)]), atol=1e-9, rtol=0)
    require_finite((obs, reward, new_reward, info["terminal_obs"]), "altmix autoreset")
    return dict(passed=True, cases=results, direct_msl_cpu_torch_parity=True,
                legacy_reward_formula_max_abs=legacy_reward_max_abs,
                legacy_bitwise_identity_expected=False,
                reset_log_seed_correct=True, no_reset_impulse=True)


def side_learner_smoke(output):
    cfg = PPOGPUConfig(hidden=(32, 32), gru_size=0, aux_pred=True, aux_coef=.1,
        rollout_steps=32, update_epochs=1, num_minibatches=2, save_runtime=True,
        ent_coef=.001, sched_period=0, milestone_period=500, exploiter_iters=2,
        exploiter_win_target=.75, seed=123)
    env = GpuDogfightVecEnv(16, seed=123)
    env.ic_pool_size = 16
    trainer = PPOGPUTrainer(env, cfg)
    trainer.update(*trainer.collect_rollout()[:2])
    records = []
    for milestone, expected_mode in ((500, 1), (1000, 0), (1500, 2), (2000, 1), (2500, 0)):
        trainer.iteration = milestone
        before = trainer._runtime_state()
        learner = copy.deepcopy(trainer._snapshot_learner())
        metrics = []
        score = trainer.train_exploiter(metric_cb=lambda i, m: metrics.append((i, m)))
        assert len(metrics) == 2
        assert all(m["reward_mode"] == expected_mode for _, m in metrics)
        assert trainer.env.reward_mode == 0
        assert_tree_equal(learner, trainer._snapshot_learner())
        after = trainer._runtime_state()
        for key in ("sim", "obr", "reward_mode", "alt_hunt_coef", "env_rng", "numpy_rng", "torch_rng", "cuda_rng"):
            assert_tree_equal(before[key], after[key])
        for key in before["trainer"]:
            if key != "opp_weights":
                assert_tree_equal(before["trainer"][key], after["trainer"][key])
        records.append(dict(main_iteration=milestone, mode=expected_mode, win_rate_ema=score,
                            iterations=len(metrics), main_exactly_restored=True))
    assert trainer.pool.num_permanent() == 5
    assert [x["reward_mode"] for x in trainer.exploiter_history] == ["altitude_hunt", "standard", "attack", "altitude_hunt", "standard"]
    checkpoint = output / "alternating_smoke.pt"
    trainer.save(checkpoint)
    history = copy.deepcopy(trainer.exploiter_history)
    trainer.load(checkpoint)
    assert trainer.exploiter_history == history
    assert trainer.env.reward_mode == 0
    trainer.update(*trainer.collect_rollout()[:2])
    require_finite((trainer.model.state_dict(), env.sim.states), "altmix resumed main update")
    return dict(passed=True, records=records, history=history, save_resume_main_update=True)


def run(output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Use a fresh validation output directory")
    parity = fused_parity()
    gc.collect(); torch.cuda.empty_cache()
    smoke = side_learner_smoke(output)
    report = dict(passed=True, reward_contract=REWARD_CONTRACT, parity=parity, smoke=smoke, no_wandb=True)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        run(args.output)
    except Exception as exc:
        record_integrity_failure(args.output, exc, "altitude_reward_gpu_validation")
        raise
