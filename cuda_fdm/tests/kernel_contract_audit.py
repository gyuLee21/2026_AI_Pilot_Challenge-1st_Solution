"""Audit CUDA wrapper contracts using CPU tensors and mocked driver calls only.

Run with ``python -B -m cuda_fdm.tests.kernel_contract_audit``. JSON is printed
to stdout; this script does not write training files or initialize CUDA. It
checks pointer forwarding and launch contracts, not GPU numerical correctness
or runtime performance. In particular, no out-of-bounds pointer is dereferenced.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def launch(self, grid, block, args, **kwargs):
        self.calls.append({"grid": grid, "block": block, "args": args})


def _method(source, class_name, method_name):
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == method_name)


def _static_report():
    paths = ["cuda_fdm/gpu_env.py", "cuda_fdm/rl_env.py",
             "cuda_fdm/obs_reward.py", "cuda_fdm/tests/cuda_rt.py",
             "cuda_fdm/gen/obs_kernel.cu", "cuda_fdm/train_gpu.py",
             "cuda_fdm/architecture_search.py", "cuda_fdm/ppo_gpu.py"]
    sources = {p: (ROOT / p).read_text(encoding="utf-8") for p in paths}
    step = _method(sources["cuda_fdm/rl_env.py"], "GpuDogfightVecEnv", "step")
    targets = {"self.sim.step", "self.obr.kernel_advance",
               "self.obr.kernel_init_reward_state", "self.obr.kernel_build_obs"}
    calls = sorted((n for n in ast.walk(step) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) in targets), key=lambda n: n.lineno)
    return {
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in paths},
        "step_launches": [{"call": ast.unparse(n.func), "line": n.lineno}
                          for n in calls],
        "launches_per_step": len(calls),
        "launches_per_rollout64": 64 * len(calls),
        "train_cli_exposes_precision": "--precision" in sources["cuda_fdm/train_gpu.py"],
    }


def _driver_report(cuda_rt):
    observed = {"launch_count": 0, "context_sync_count": 0}

    def launch(*args):
        observed["launch_count"] += 1
        observed["stream_handle"] = args[8].value or 0
        return 0

    def sync():
        observed["context_sync_count"] += 1
        return 0

    fake_driver = SimpleNamespace(cuLaunchKernel=launch, cuCtxSynchronize=sync)
    kernel = cuda_rt.Kernel.__new__(cuda_rt.Kernel)
    kernel.func = ctypes.c_void_p(123)
    kernel.device = 0
    kernel.force_sync = False
    kernel.launch_count = kernel.context_sync_count = 0
    with patch.object(cuda_rt, "_cu", fake_driver), \
            patch.object(cuda_rt.torch.cuda, "current_device", return_value=0), \
            patch.object(cuda_rt.torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=456)):
        kernel.launch((1, 1, 1), (1, 1, 1), [ctypes.c_int(1)])
        observed["legacy_sync_count"] = observed["context_sync_count"]
        kernel.launch((1, 1, 1), (1, 1, 1), [ctypes.c_int(1)], tensors=())
        observed["tensor_path_additional_syncs"] = observed["context_sync_count"] - observed["legacy_sync_count"]
    return observed


def _pointer_report(torch, BatchObsReward, obs_size):
    obr = BatchObsReward(1, device="cpu", enable_kernel=False)
    obr.block = 128
    obr._origin_args = lambda: []  # Recorded arguments are never executed.
    obr._all_env_mask = torch.ones(1, dtype=torch.uint8)
    obr.obs_buf = torch.zeros(2, obs_size, dtype=torch.float32)
    obr.reward_buf = torch.zeros(2, dtype=torch.float64)
    obr.term_buf = torch.zeros(1, dtype=torch.uint8)
    obr.trunc_buf = torch.zeros(1, dtype=torch.uint8)
    rows = []
    for dtype in (torch.float32, torch.float64):
        states = torch.zeros(2, 101, dtype=dtype)
        actions = torch.zeros(2, 4, dtype=dtype)
        for method, field, inputs in [
            ("kernel_init_reward_state", "_k_init_reward", (states,)),
            ("kernel_advance", "_k_adv", (states, actions)),
            ("kernel_build_obs", "_k_obs", (states,)),
        ]:
            recorder = LaunchRecorder()
            setattr(obr, field, recorder)
            row = {"method": method, "input_dtype": str(dtype)}
            try:
                getattr(obr, method)(*inputs)
            except (TypeError, ValueError, AssertionError) as exc:
                row.update(rejected=True, rejection=str(exc))
            else:
                row["rejected"] = False
                row["launches"] = len(recorder.calls)
                args = recorder.calls[-1]["args"]
                row["original_state_pointer_forwarded"] = args[0].value == states.data_ptr()
                if len(inputs) == 2:
                    row["original_action_pointer_forwarded"] = args[1].value == actions.data_ptr()
            rows.append(row)
    return {
        "kernel_calls": rows,
        "fp32_two_plane_state_bytes": 2 * 101 * 4,
        "double_pointer_second_plane_offset_bytes": 101 * 8,
        "fp32_two_plane_action_bytes": 2 * 4 * 4,
        "double_pointer_second_plane_action_offset_bytes": 4 * 8,
        "interpretation": (
            "For nenv=1, the double-pointer offset for the second aircraft is "
            "already one past each FP32 allocation. Contiguous does not cast dtype. "
            "The CPU audit does not dereference these pointers or execute CUDA."
        ),
    }


def _guard_report(env_module):
    class ReachedSimulator(Exception):
        pass

    rows = []
    for precision in ("fp32", "fp64"):
        calls = []

        def simulator(*args, **kwargs):
            calls.append(kwargs)
            raise ReachedSimulator

        with patch.object(env_module, "GpuDogfight", simulator):
            try:
                env_module.GpuDogfightVecEnv(1, precision=precision)
            except ReachedSimulator:
                result = "reached_simulator_constructor"
            except (TypeError, ValueError, AssertionError) as exc:
                result = f"rejected: {exc}"
            else:
                raise AssertionError("Expected rejection or mocked simulator construction")
        rows.append({"precision": precision, "result": result,
                     "simulator_constructor_calls": len(calls)})
    return rows


class RLPrecisionGuardTests(unittest.TestCase):
    """Check rejection before the simulator can touch CUDA or allocate state."""

    def test_fp32_rejected_before_simulator_construction(self):
        env_module = importlib.import_module("cuda_fdm.rl_env")
        with patch.object(env_module, "GpuDogfight") as simulator:
            with self.assertRaisesRegex(ValueError, "precision='fp64' only"):
                env_module.GpuDogfightVecEnv(1, precision="fp32")
            simulator.assert_not_called()

    def test_fp64_default_and_explicit_keep_simulator_contract(self):
        env_module = importlib.import_module("cuda_fdm.rl_env")

        class ReachedSimulator(Exception):
            pass

        for kwargs in ({}, {"precision": "fp64"}):
            with self.subTest(kwargs=kwargs):
                with patch.object(env_module, "GpuDogfight", side_effect=ReachedSimulator) as simulator:
                    with self.assertRaises(ReachedSimulator):
                        env_module.GpuDogfightVecEnv(1, **kwargs)
                    simulator.assert_called_once_with(
                        1, substeps=6, planes_per_env=2, precision="fp64", block=128)


def main():
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Run this audit in a fresh CPU-only process")

    def forbid_cuda(*args, **kwargs):
        raise AssertionError("CPU-only audit attempted CUDA initialization")

    with patch.object(torch.cuda, "_lazy_init", forbid_cuda):
        env_module = importlib.import_module("cuda_fdm.rl_env")
        obs_module = importlib.import_module("cuda_fdm.obs_reward")
        cuda_rt = importlib.import_module("cuda_rt")
        report = {
            "audit": "cpu_only_cuda_kernel_contract",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "static": _static_report(),
            "mock_driver": _driver_report(cuda_rt),
            "cpu_pointer_forwarding": _pointer_report(
                torch, obs_module.BatchObsReward, obs_module.OBS_SIZE),
            "rl_constructor_guard": _guard_report(env_module),
            "gpu_performance_measured": False,
            "gpu_numerical_correctness_tested": False,
        }
    report["cuda_initialized"] = torch.cuda.is_initialized()
    if report["cuda_initialized"]:
        raise AssertionError("Audit must not initialize CUDA")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
