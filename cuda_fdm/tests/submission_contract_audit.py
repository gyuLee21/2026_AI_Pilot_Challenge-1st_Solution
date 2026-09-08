"""CPU-only replay of the actual UDP submission observation contract.

This is an audit, not a flight simulation or a GPU parity test. It runs the real
ProviderCommandPolicy -> MLPActionProvider -> loaded bundle chain on synthetic
PlaneInfo packets, and compares its inputs with the CUDA reconstructor's Torch
reference on CPU. The CUDA reference receives the same executed commands, so
observation differences are not caused by divergent flight trajectories/actions.

No production behavior is patched. A private CPU reconstructor also demonstrates
the schedule needed by a CUDA-bundle-specific adapter: no initial advance;
advance current post-step state before each subsequent observation; record the
applied command, including throttle in [0, 1].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from claude_code import my_observation as MO
from claude_code.action_provider import MLPActionProvider
from claude_code.model import (
    discrete_indices_to_continuous,
    make_obs_normalizer,
    policy_action_to_command,
)
from cuda_fdm.obs_reward import BatchObsReward
from cuda_fdm.ppo_gpu import RunningNorm, action_to_env
from cuda_fdm.finite_checks import require_finite
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import ProviderCommandPolicy
from dogfight.unreal.client import PlaneSnapshot, RemoteClientContext
from dogfight.unreal.protocol import PlaneInfo, Rotation3D, Vector3D
from dogfight.unreal.policies import plane_info_to_state


THROTTLE_DIMS = np.array([197, 201, 205, 209, 213])  # 214D: history 194~213, throttle = ch3 of each step
RECON_DIMS = np.array([6, 7, 8, 43, 44, 45, 46] + list(range(128, 164)))


class CapturingProvider(MLPActionProvider):
    def _prepare_observation(self, context):
        observation = super()._prepare_observation(context)
        self.captured_obs = np.asarray(observation, dtype=np.float32).copy()
        return observation


class ShadowActor:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state = (
            self.model.initial_state(1, "cpu")
            if getattr(self.model, "is_recurrent", False) else None
        )

    @torch.inference_mode()
    def predict(self, normalized):
        x = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
        if getattr(self.model, "is_recurrent", False):
            logits, self.state = self.model._step_logits(x, self.state)
        else:
            logits = self.model.actor_logits(x).view(1, 4, self.model.num_bins)
        idx = logits.argmax(-1)[0].numpy()
        command = policy_action_to_command(
            discrete_indices_to_continuous(idx, self.model.num_bins)
        )
        return idx, command, logits[0].numpy()


def _plane(state, frame, plane_id):
    return PlaneInfo(
        index=frame, plane_id=plane_id,
        position=Vector3D(float(state[0]), float(state[1]), float(-state[2])),
        rotation=Rotation3D(*map(float, state[3:6])),
        velocity=Vector3D(*map(float, state[6:9])),
    )


def _context(states, frame):
    own = _plane(states[0], frame, 1)
    target = _plane(states[1], frame, 2)
    return RemoteClientContext(
        plane_id=1, frame_index=frame,
        own_plane=PlaneSnapshot(True, 1, frame, own),
        enemy_plane=PlaneSnapshot(True, 2, frame, target),
    )


def _states(scenario, k):
    own = np.array([3500., -350., -2500., 0., 0., 0., 230., 0., 0.])
    target = np.array([6500., -350., -2500., 0., 0., 180., 230., 0., 0.])
    if scenario == "maneuver":
        own[3:6] = [35 * np.sin(.17 * k), 8 * np.sin(.11 * k),
                    (80 + 25 * np.sin(.07 * k)) % 360]
        target[3:6] = [-25 * np.sin(.13 * k), 6 * np.sin(.09 * k),
                       (260 + 20 * np.sin(.12 * k)) % 360]
        own[6:9] = [230 + 12 * np.sin(.05 * k), 2 * np.sin(.17 * k),
                    4 * np.sin(.11 * k)]
        target[6:9] = [240 + 10 * np.sin(.08 * k), -2 * np.sin(.13 * k),
                       3 * np.sin(.09 * k)]
        target[2] += 40 * np.sin(.04 * k)
    elif scenario == "cone_transition":
        target[0] = own[0] + 500
        # Deliberately leave/enter the firing cone on different sides, including
        # an initial no-damage frame. Short replay avoids HP depletion.
        own[5] = 0. if k % 4 == 1 else 5.
        target[5] = 180. if k % 5 == 2 else 185.
        own[6] += k
        target[6] -= .5 * k
    return np.stack([own, target])


def _command_array(command):
    return np.array([command.roll_cmd, command.pitch_cmd, command.yaw_cmd,
                     command.throttle_cmd], dtype=np.float32)


def _diff_stats(left, right):
    require_finite(left, "submission.comparison.left")
    require_finite(right, "submission.comparison.right")
    diff = np.abs(np.asarray(left) - np.asarray(right))
    return {
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean(diff ** 2))),
        "dimensions_gt_1e-6": np.flatnonzero(diff.max(axis=0) > 1e-6).tolist(),
        "dimension_max_abs": diff.max(axis=0).tolist(),
    }


def throttle_grid_probe(normalize):
    rows = []
    for throttle_idx in (0, 10, 20):
        idx = np.array([10, 10, 10, throttle_idx])
        raw = discrete_indices_to_continuous(idx, 21)
        command = action_to_env(torch.tensor(idx)[None], 21)[0].numpy()
        np.testing.assert_allclose(command, policy_action_to_command(raw), atol=1e-7)
        batch = BatchObsReward(1, device="cpu", enable_kernel=False)
        rec = MO.StateReconstructor()
        for _ in range(MO.ACTION_HISTORY_LEN):
            batch.push_actions(torch.tensor(np.stack([command, command])))
            rec.push_action(raw)
        states = _states("stationary", 0)
        reference = batch.build_obs(torch.tensor(states))[0].numpy()
        submitted = MO.build_observation(states[0], states[1], rec._geo,
                                         reconstructor=rec)
        rows.append({
            "policy_throttle": float(raw[3]), "command_throttle": float(command[3]),
            "history_delta_cuda_minus_cpu": (reference - submitted)[THROTTLE_DIMS].tolist(),
            "normalized_history_delta": (normalize(reference) - normalize(submitted))[
                THROTTLE_DIMS].tolist(),
            "differing_dimensions": np.flatnonzero(np.abs(reference - submitted) > 1e-6).tolist(),
        })
    return rows


@torch.inference_mode()
def audit(bundle_dir, count=96):
    torch.set_num_threads(1)
    provider = CapturingProvider(bundle_dir, device="cpu", stochastic=False, debug_obs=False)
    assert provider.metadata.get("observation_module") == "claude_code.my_observation"
    hook = load_observation_hook(provider.metadata["observation_module"])
    policy = ProviderCommandPolicy(provider, observation_mode=hook["mode"],
                                   observation_fn=hook["build_observation"], action_repeat=6)
    normalize = make_obs_normalizer(provider.metadata.get("obs_normalization"))
    def provider_rec():
        return (provider._cuda_observation.rec if provider._cuda_observation is not None
                else MO.get_reconstructor())
    gpu_normalize = RunningNorm(MO.OBSERVATION_SIZE, "cpu")
    norm_data = provider.metadata.get("obs_normalization")
    if norm_data:
        gpu_normalize.load_state_dict({
            k: torch.tensor(norm_data[k], dtype=torch.float32)
            for k in ("mean", "var", "count")
        })

    output = {
        "bundle": str(Path(bundle_dir).resolve()),
        "selected_iteration": provider.metadata.get("selected_iteration"),
        "model": provider.metadata.get("model"),
        "scope": "CPU synthetic packet replay, actual UDP policy/provider and bundle; no flight or GPU jobs",
        "teacher_forcing": "CUDA reference and private CPU adapter probe receive the actual submitted commands",
        "throttle_grid_probe": throttle_grid_probe(normalize),
        "cases": [],
    }
    for scenario, frames in (("stationary", count), ("maneuver", count),
                             ("cone_transition", min(count, 24))):
        batch = BatchObsReward(1, device="cpu", enable_kernel=False)
        adapter_rec = MO.StateReconstructor()
        shadows = {name: ShadowActor(provider.model) for name in (
            "cuda_reference", "actual_submission", "history_only_repaired",
            "timing_only_repaired", "private_cpu_adapter")}
        policy.reset(None)
        observed, reference, adapter_obs = [], [], []
        normalized_cpu, normalized_gpu, normalized_adapter = [], [], []
        per_variant = {name: {"decisions_different": 0, "channels_different": [0] * 4,
                              "max_command_difference": 0., "max_logit_difference": 0.}
                       for name in shadows if name != "cuda_reference"}
        trace, previous_command = [], None
        for k in range(frames):
            context = _context(_states(scenario, k), k * 6)
            states = np.stack([plane_info_to_state(context.own_plane.plane_info)[:9],
                               plane_info_to_state(context.enemy_plane.plane_info)[:9]])
            state_t = torch.tensor(states, dtype=torch.float64)
            if k:
                commands = torch.tensor(np.stack([previous_command, previous_command]))
                batch.push_actions(commands)
                batch.advance(state_t)
                adapter_rec.push_action(previous_command)
                adapter_rec.advance(states[0], states[1])
            cuda_obs = batch.build_obs(state_t)[0].numpy()
            candidate_obs = MO.build_observation(states[0], states[1], adapter_rec._geo,
                                                reconstructor=adapter_rec)
            actual_command = _command_array(policy.compute_command(context))
            actual_obs = provider.captured_obs.copy()
            history_fixed = actual_obs.copy()
            history_fixed[THROTTLE_DIMS] = cuda_obs[THROTTLE_DIMS]
            timing_fixed = actual_obs.copy()
            timing_fixed[RECON_DIMS] = cuda_obs[RECON_DIMS]
            norm_ref = (gpu_normalize.normalize(torch.from_numpy(cuda_obs)).numpy()
                        if norm_data else cuda_obs)
            variants = {
                "cuda_reference": norm_ref,
                "actual_submission": normalize(actual_obs),
                "history_only_repaired": normalize(history_fixed),
                "timing_only_repaired": normalize(timing_fixed),
                "private_cpu_adapter": normalize(candidate_obs),
            }
            predictions = {name: shadows[name].predict(obs) for name, obs in variants.items()}
            np.testing.assert_array_equal(predictions["actual_submission"][1], actual_command)
            ref_idx, ref_cmd, ref_logits = predictions["cuda_reference"]
            for name, stats in per_variant.items():
                idx, cmd, logits = predictions[name]
                changed = idx != ref_idx
                stats["decisions_different"] += int(changed.any())
                stats["channels_different"] = (
                    np.asarray(stats["channels_different"]) + changed).tolist()
                stats["max_command_difference"] = max(stats["max_command_difference"],
                                                       float(np.max(np.abs(cmd - ref_cmd))))
                stats["max_logit_difference"] = max(stats["max_logit_difference"],
                                                     float(np.max(np.abs(logits - ref_logits))))
            observed.append(actual_obs)
            reference.append(cuda_obs)
            adapter_obs.append(candidate_obs)
            normalized_cpu.append(variants["actual_submission"])
            normalized_gpu.append(norm_ref)
            normalized_adapter.append(variants["private_cpu_adapter"])
            if k < 4 or (len(trace) < 8 and np.any(predictions["actual_submission"][0] != ref_idx)):
                trace.append({
                    "decision": k,
                    "cpu_clock_after_compute": provider_rec().t_sec,
                    "observed_time_norm_cpu_cuda": [float(actual_obs[26]), float(cuda_obs[26])],
                    "hp_cpu_cuda": [actual_obs[6:9].tolist(), cuda_obs[6:9].tolist()],
                    "damage_rate_cpu_cuda": [actual_obs[45:47].tolist(), cuda_obs[45:47].tolist()],
                    "body_omega_cpu_cuda": [actual_obs[131:134].tolist(), cuda_obs[131:134].tolist()],
                    "newest_throttle_cpu_cuda": [float(actual_obs[167]), float(cuda_obs[167])],
                    "greedy_command_cpu_cuda": [actual_command.tolist(), ref_cmd.tolist()],
                })
            # Exercise the real 6:1 command-repeat contract: no observation,
            # recurrent state, reconstruction time, or action-history update.
            rec_time = provider_rec().t_sec
            rec_hist = provider_rec().action_history.copy()
            for repeat in range(1, 6):
                cached = _command_array(policy.compute_command(_context(states, k * 6 + repeat)))
                np.testing.assert_array_equal(cached, actual_command)
                assert provider_rec().t_sec == rec_time
                np.testing.assert_array_equal(provider_rec().action_history, rec_hist)
            previous_command = actual_command

        case = {"scenario": scenario, "decisions": frames,
                "raw_submission_vs_cuda": _diff_stats(observed, reference),
                "normalized_submission_vs_cuda": _diff_stats(normalized_cpu, normalized_gpu),
                "raw_private_adapter_vs_cuda": _diff_stats(adapter_obs, reference),
                "normalized_private_adapter_vs_cuda": _diff_stats(normalized_adapter, normalized_gpu),
                "greedy_comparison": per_variant, "trace": trace}
        for stats in per_variant.values():
            stats["decision_difference_rate"] = stats["decisions_different"] / frames
        output["cases"].append(case)
        # The proposed compatibility schedule must reproduce the formulas
        # independently; tolerances accommodate float32 storage and rotations.
        assert case["raw_private_adapter_vs_cuda"]["max_abs"] < 1e-5
        assert per_variant["private_cpu_adapter"]["decisions_different"] == 0
        if provider._cuda_observation is not None:
            assert case["raw_submission_vs_cuda"]["max_abs"] < 1e-5
            assert per_variant["actual_submission"]["decisions_different"] == 0

    output["features"] = list(MO.FEATURE_NAMES)
    output["device"] = "cpu"
    output["cuda_initialized"] = torch.cuda.is_initialized()
    assert not output["cuda_initialized"]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=96)
    args = parser.parse_args()
    result = audit(args.bundle, args.count)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "cuda_initialized": result["cuda_initialized"],
                      "cases": [{"scenario": case["scenario"], "decisions": case["decisions"],
                                 "raw_max": case["raw_submission_vs_cuda"]["max_abs"],
                                 "normalized_max": case["normalized_submission_vs_cuda"]["max_abs"],
                                 "dimensions": case["raw_submission_vs_cuda"]["dimensions_gt_1e-6"],
                                 "greedy": case["greedy_comparison"],
                                 "adapter_raw_max": case["raw_private_adapter_vs_cuda"]["max_abs"]}
                                for case in result["cases"]]}, indent=2))


if __name__ == "__main__":
    main()
