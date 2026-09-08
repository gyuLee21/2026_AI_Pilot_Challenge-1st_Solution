"""Bounded same-input kernel ABI, stream/lifetime and throughput regressions.

No model learning or W&B. Reference: context synchronization + full post-reset
observation rebuild. Candidate: current Torch stream + reset-only rebuild.
CUDA event durations include stream idle gaps; they are not pure kernel time.
"""
import argparse
import ctypes
import gc
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.rl_env import GpuDogfightVecEnv
from cuda_fdm.finite_checks import require_finite, finite_max_abs_error
import cuda_rt


class KernelABITests(unittest.TestCase):
    def test_fp32_and_wrong_shapes_rejected_before_pointer_launch(self):
        obr = BatchObsReward(1, device="cpu", enable_kernel=False)
        for method in (obr.kernel_build_obs, obr.kernel_init_reward_state, obr.kernel_advance):
            with self.subTest(method=method.__name__):
                args = (torch.zeros(2, 101),)
                if method == obr.kernel_advance:
                    args += (torch.zeros(2, 4),)
                with self.assertRaisesRegex(TypeError, "dtype"):
                    method(*args)
        with self.assertRaisesRegex(ValueError, "shape"):
            obr.kernel_build_obs(torch.zeros(1, 101, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            obr.kernel_build_obs(torch.zeros(2, 101, dtype=torch.float64))

    def test_current_stream_and_tensor_lifetime_contract_without_cuda(self):
        seen = {"sync": 0, "record": []}
        stream = SimpleNamespace(cuda_stream=123456)
        def launch(*args):
            seen["stream"] = args[8].value
            return 0
        def sync():
            seen["sync"] += 1
            return 0
        tensor = SimpleNamespace(device=torch.device("cuda", 0),
                                  record_stream=lambda s: seen["record"].append(s))
        kernel = cuda_rt.Kernel.__new__(cuda_rt.Kernel)
        kernel.device, kernel.func = 0, ctypes.c_void_p(1)
        kernel.force_sync = False
        kernel.launch_count = kernel.context_sync_count = 0
        with patch.object(cuda_rt, "_cu", SimpleNamespace(cuLaunchKernel=launch, cuCtxSynchronize=sync)), \
                patch.object(torch.cuda, "current_device", return_value=0), \
                patch.object(torch.cuda, "current_stream", return_value=stream):
            kernel.launch((1, 1, 1), (1, 1, 1), [ctypes.c_int(1)], tensors=(tensor,))
            self.assertEqual(seen["stream"], 123456)
            self.assertEqual(seen["sync"], 0)
            self.assertEqual(seen["record"], [stream])
            kernel.launch((1, 1, 1), (1, 1, 1), [ctypes.c_int(1)])
            self.assertEqual(seen["sync"], 1)  # legacy raw-pointer safety fallback
            kernel.force_sync = True
            kernel.launch((1, 1, 1), (1, 1, 1), [ctypes.c_int(1)], tensors=(tensor,))
            self.assertEqual(seen["sync"], 2)


def kernels(env):
    return (env.sim.kern, env.obr._k_adv, env.obr._k_init_reward, env.obr._k_obs)


def snapshot(env):
    return dict(states=env.sim.states.clone(), obr=env.obr.clone_state(),
                rng=torch.cuda.get_rng_state(), obs=env.obr.obs_buf.clone())


def restore(env, state):
    env.sim.states.copy_(state["states"])
    env.obr.restore(state["obr"])
    env.obr.obs_buf.copy_(state["obs"])
    torch.cuda.set_rng_state(state["rng"])


def run_trace(env, initial, controls, sync, full_obs, capture=True, forced_endings=True):
    for kernel in kernels(env):
        kernel.force_sync = sync
    restore(env, initial)
    build = env.obr.kernel_build_obs
    env.obr.kernel_build_obs = lambda states, env_mask=None: build(states, None if full_obs else env_mask)
    trace = []
    starts = [(k.launch_count, k.context_sync_count) for k in kernels(env)]
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    begin.record()
    try:
        for i, commands in enumerate(controls):
            if forced_endings and i == 3:
                env.obr.hp[::8] = 0.  # a subset dies, others continue
            if forced_endings and i == 7:
                env.obr.t_sec[1::8] = 201.  # distinct subset truncates
            obs, reward, done, info = env.step(commands)
            if capture:
                values = [obs.reshape(-1).double(), reward.reshape(-1), done.double(),
                          info["terminal_obs"].reshape(-1).double(), info["terminal_hp"].reshape(-1),
                          info["terminal_state_finite"].double(), env.sim.states.reshape(-1)]
                values += [v.reshape(-1).double() for v in env.obr.clone_state().values()]
                trace.append(torch.cat(values))
        end.record()
        end.synchronize()
    finally:
        env.obr.kernel_build_obs = build
    result = dict(wall_sec=time.perf_counter() - t0, cuda_stream_elapsed_ms=begin.elapsed_time(end),
                  launches=sum(k.launch_count - old[0] for k, old in zip(kernels(env), starts)),
                  context_syncs=sum(k.context_sync_count - old[1] for k, old in zip(kernels(env), starts)))
    return torch.stack(trace) if capture else None, result


def make_env(nenv):
    env = GpuDogfightVecEnv(nenv, seed=271)
    env.ic_pool_size = 128  # validation-only: bounded CPU fixture initialization
    env.reset(stagger=False)
    env._ensure_ic_pool()
    return env


def action_sequence(env, count):
    generator = torch.Generator(device="cuda").manual_seed(81)
    actions = torch.rand(count, env.nac, 4, device="cuda", dtype=torch.float64, generator=generator)
    actions[:, :, :3] = (actions[:, :, :3] - .5) * .2
    actions[:, :, 3] = .6 + .2 * actions[:, :, 3]
    return actions


def gpu_checks(benchmark=False):
    torch.set_num_threads(1)
    env = make_env(32)
    controls = action_sequence(env, 64)
    initial = snapshot(env)
    reference, ref_perf = run_trace(env, initial, controls, sync=True, full_obs=True)
    optimized, opt_perf = run_trace(env, initial, controls, sync=False, full_obs=False)
    error = finite_max_abs_error(reference, optimized, "sync-full vs async-masked trace")
    assert error == 0., error
    assert ref_perf["context_syncs"] == 320 and opt_perf["context_syncs"] == 0
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        side_trace, side_perf = run_trace(env, initial, controls, sync=False, full_obs=False)
    side.synchronize()
    side_error = finite_max_abs_error(reference, side_trace, "nondefault stream trace")
    assert side_error == 0., side_error

    for tensor, label in ((env.sim.states.float(), "states"), (controls[0].float(), "actions")):
        try:
            env.obr.kernel_advance(tensor if label == "states" else env.sim.states,
                                   tensor if label == "actions" else controls[0])
        except TypeError:
            pass
        else:
            raise AssertionError(f"FP32 {label} was not rejected")

    # Borrow a default-stream allocation on a delayed side stream, delete the
    # Python owner, and pressure the original allocator before the kernel reads it.
    restore(env, initial)
    expected = env.obr.kernel_build_obs(env.sim.states).clone()
    default = torch.cuda.current_stream()
    for _ in range(20):
        borrowed = env.sim.states.clone()
        side.wait_stream(default)
        with torch.cuda.stream(side):
            torch.cuda._sleep(1000000)
            actual = env.obr.kernel_build_obs(borrowed).clone()
        del borrowed
        for _ in range(4):
            pressure = torch.empty_like(env.sim.states).fill_(-12345.)
        side.synchronize()
        assert finite_max_abs_error(actual, expected, "cross-stream borrowed allocation") == 0.
    del pressure

    # Contiguous copies made inside the raw-pointer wrapper must also stay alive.
    wide = torch.empty(env.nac, 202, device="cuda", dtype=torch.float64)
    wide[:, ::2] = env.sim.states
    strided = env.obr.kernel_build_obs(wide[:, ::2]).clone()
    assert finite_max_abs_error(strided, expected, "strided input copy") == 0.
    result = dict(passed=True, nenv=32, steps=64, forced_death_and_timeout=True,
                  full_trajectory_max_abs=error, nondefault_stream_max_abs=side_error,
                  cross_stream_allocator_repetitions=20, fp32_state_and_action_rejected=True,
                  reference=ref_perf, optimized=opt_perf, nondefault_stream=side_perf)
    del env, reference, optimized, side_trace, initial, controls
    gc.collect()
    if benchmark:
        env = make_env(4096)
        controls = action_sequence(env, 64)
        initial = snapshot(env)
        # Four paths isolate synchronization removal from masked observation reuse.
        rows = {label: [] for label in ("sync_full", "sync_masked", "async_full", "async_masked")}
        modes = [("sync_full", True, True), ("sync_masked", True, False),
                 ("async_full", False, True), ("async_masked", False, False)]
        run_trace(env, initial, controls[:4], False, False, capture=False)
        for repeat in range(3):
            order = modes if repeat % 2 == 0 else list(reversed(modes))
            for label, sync, full in order:
                _, perf = run_trace(env, initial, controls, sync, full, capture=False)
                rows[label].append(perf)
                print(json.dumps({"benchmark": label, "repeat": repeat, **perf}), flush=True)
        result["benchmark"] = {label: dict(samples=samples,
             median_wall_sec=float(np.median([s["wall_sec"] for s in samples])),
             transitions_per_sec=4096 * 64 / float(np.median([s["wall_sec"] for s in samples])))
                                for label, samples in rows.items()}
        result["benchmark_scope"] = "4096 env x 64 steps; physics/obs/reward/autoreset only; no policy inference or PPO update"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    result = gpu_checks(args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
