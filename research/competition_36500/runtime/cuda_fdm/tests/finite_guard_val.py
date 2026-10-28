"""CPU-only tests proving non-finite regression trajectories cannot report PASS."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cuda_fdm.finite_checks import finite_max_abs_error, require_finite, require_finite_training_stats
from cuda_fdm.tests import obs_reward_val


def nan_kernel_regression(module):
    """Run the real regression function with all-NaN physics, without a GPU."""
    original_zeros, original_tensor = torch.zeros, torch.tensor

    def cpu_zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    class Sim:
        def __init__(self, *args, **kwargs):
            self.states = torch.full((2, 101), float("nan"), dtype=torch.float64)

        def load_seed(self, seeds):
            pass

        def step(self, *args, **kwargs):
            pass

    class Env:
        def __init__(self, *args, **kwargs):
            pass

        def _build_all_seeds(self):
            return None

    class ObsReward:
        def __init__(self, *args, **kwargs):
            self.hp = torch.full((2,), float("nan"))
            self.pqr = torch.full((2, 3), float("nan"))

        def initialize_reward_state(self, *args, **kwargs):
            pass

        kernel_init_reward_state = initialize_reward_state
        push_actions = initialize_reward_state
        advance = initialize_reward_state

        def build_obs(self, *args, **kwargs):
            return torch.full((2, 184), float("nan"))

        kernel_build_obs = build_obs

        def compute_reward(self, *args, **kwargs):
            return torch.full((2,), float("nan"))

        def kernel_advance(self, *args, **kwargs):
            return self.compute_reward(), original_zeros(1, dtype=torch.bool), original_zeros(1, dtype=torch.bool)

    def kinematics(_):
        return tuple(torch.full(shape, float("nan"), dtype=torch.float64)
                     for shape in ((2, 3), (2, 3), (2, 3), (2,)))

    def ned(*_):
        return tuple(torch.full((2,), float("nan"), dtype=torch.float64) for _ in range(3))

    with patch("cuda_fdm.gpu_env.GpuDogfight", Sim), \
         patch.object(module, "GpuDogfightVecEnv", Env), \
         patch.object(module.OR, "BatchObsReward", ObsReward), \
         patch("cuda_fdm.rl_env.kinematics_state", kinematics), \
         patch("cuda_fdm.rl_env.ned_from_ecef_altasl", ned), \
         patch.object(torch, "zeros", cpu_zeros), patch.object(torch, "tensor", cpu_tensor):
        return module.test_kernel_vs_torch(nenv=1, K=2)


class FiniteGuardTests(unittest.TestCase):
    def test_matches_finite_maximum(self):
        self.assertEqual(finite_max_abs_error(np.array([1., 3.]), np.array([1., 2.])), 1.)

    def test_matching_nan_and_infinity_are_failures(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for kind in (np.array, torch.tensor):
                with self.subTest(bad=bad, kind=kind):
                    with self.assertRaises(FloatingPointError):
                        finite_max_abs_error(kind([bad]), kind([bad]))

    def test_single_bad_input_is_failure(self):
        with self.assertRaises(FloatingPointError):
            finite_max_abs_error(np.array([0., np.nan]), np.zeros(2))

    def test_overflowing_difference_is_failure(self):
        with self.assertRaises(FloatingPointError):
            finite_max_abs_error(np.array([1e308]), np.array([-1e308]))

    def test_shape_mismatch_not_broadcast(self):
        with self.assertRaises(ValueError):
            finite_max_abs_error(np.zeros((1, 2)), np.zeros(2))

    def test_nested_state_nonfinite_is_failure(self):
        with self.assertRaises(FloatingPointError):
            require_finite({"norm": {"var": torch.tensor([1., float("nan")])}})

    def test_real_regression_rejects_nan_physics_without_cuda(self):
        with self.assertRaisesRegex(FloatingPointError, "physical_state"):
            nan_kernel_regression(obs_reward_val)

    def test_missing_episode_statistics_only_allowed_without_completions(self):
        stats = SimpleNamespace(iteration=1, policy_loss=0., value_loss=0., entropy=1.,
                                approx_kl=0., clipfrac=0., explained_variance=0.,
                                steps_per_sec=1., elapsed_sec=1., completed_episodes=0,
                                mean_return=float("nan"), mean_length=float("nan"), win_rate=float("nan"))
        require_finite_training_stats(stats)
        stats.completed_episodes = 1
        with self.assertRaises(FloatingPointError):
            require_finite_training_stats(stats)


def reproduce_before_after():
    root = Path(__file__).resolve().parents[2]
    audit = root / "runs/architecture_search/experiment_v1/audit_20260831"
    before = audit / "before/cuda_fdm__tests__obs_reward_val.py"
    spec = importlib.util.spec_from_file_location("legacy_obs_reward_regression_audit", before)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        legacy_passed = nan_kernel_regression(legacy)
    after_error = None
    try:
        nan_kernel_regression(obs_reward_val)
    except FloatingPointError as exc:
        after_error = str(exc)
    result = dict(legacy_reported_pass=bool(legacy_passed), legacy_stdout=capture.getvalue(),
                  fixed_rejected=after_error is not None, fixed_error=after_error,
                  injected_physical_state="all NaN", cuda_initialized=torch.cuda.is_initialized())
    (audit / "finite_regression_before_after.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = reproduce_before_after()
    if not result["legacy_reported_pass"] or not result["fixed_rejected"]:
        raise SystemExit("NaN regression before/after reproduction failed")
    unittest.main()
