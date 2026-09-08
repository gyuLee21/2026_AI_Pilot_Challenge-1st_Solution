"""Check the fix scope and fail-closed CLI lifecycle without running training."""
import ast
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cuda_fdm import train_gpu

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs/architecture_search/experiment_v1"
BEFORE = EXPERIMENT / "integrity_fixes_v1/before"
ALTITUDE_BEFORE = ROOT / "runs/change_validation/altitude_hp_altmix_v2/before"


def definition(source, name, parent=None):
    nodes = ast.parse(source).body
    if parent:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent).body
    return ast.dump(next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name))


def cuda_function(source, name):
    """Extract a named CUDA definition, including its signature (not call sites)."""
    match = re.search(r"(?:extern \"C\" )?__(?:device|global)__[^\n]*\b" + name + r"\s*\(", source)
    if match is None:
        raise AssertionError(f"missing CUDA definition: {name}")
    opening = source.index("{", match.start())
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[match.start():cursor]


class IntegrityScopeTests(unittest.TestCase):
    def test_model_hyperparameters_updater_and_schedule_unchanged(self):
        old = (BEFORE / "cuda_fdm__ppo_gpu.py").read_text(encoding="utf-8")
        new = (ROOT / "cuda_fdm/ppo_gpu.py").read_text(encoding="utf-8")
        for name in ("RunningNorm", "ActorCritic",
                     "action_to_env", "make_action_grid"):
            self.assertEqual(definition(old, name), definition(new, name), name)
        # User explicitly selected MLP and removed recurrent training. All PPO
        # hyperparameters except the explicitly approved architecture, auxiliary
        # additions, alternating modes and 70->75% target must still match.
        tree = ast.parse(old)
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PPOGPUConfig")
        next(n for n in config.body if isinstance(n, ast.AnnAssign) and n.target.id == "architecture").value = ast.Constant("mlp")
        next(n for n in config.body if isinstance(n, ast.AnnAssign) and n.target.id == "exploiter_win_target").value = ast.Constant(.75)
        current_config = next(n for n in ast.parse(new).body if isinstance(n, ast.ClassDef) and n.name == "PPOGPUConfig")
        auxiliary = {n.target.id: ast.literal_eval(n.value) for n in current_config.body
                     if isinstance(n, ast.AnnAssign) and n.target.id in
                     ("aux_pred", "aux_coef", "exploiter_alternate_altitude_hunt", "exploiter_alt_hunt_coef")}
        self.assertEqual(auxiliary, {"aux_pred": False, "aux_coef": .1,
                                    "exploiter_alternate_altitude_hunt": True, "exploiter_alt_hunt_coef": 5.0})
        current_config.body = [n for n in current_config.body
                               if not (isinstance(n, ast.AnnAssign) and n.target.id in auxiliary)]
        self.assertEqual(ast.dump(config), ast.dump(current_config))
        mlp_old = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLPActorCritic")
        mlp_new = next(n for n in ast.parse(new).body if isinstance(n, ast.ClassDef) and n.name == "MLPActorCritic")
        removed = {"evaluate_actions_packed_sequence", "evaluate_values_packed_sequence"}
        for name in [n.name for n in mlp_old.body if isinstance(n, ast.FunctionDef) and n.name not in removed]:
            self.assertEqual(definition(old, name, "MLPActorCritic"), definition(new, name, "MLPActorCritic"), name)
        self.assertFalse(removed & {n.name for n in mlp_new.body if isinstance(n, ast.FunctionDef)})
        self.assertNotIn("def _pack_recurrent_batch", new)
        for name in ("_apply_schedule",):
            self.assertEqual(definition(old, name, "PPOGPUTrainer"),
                             definition(new, name, "PPOGPUTrainer"), name)

    def test_performance_changes_preserve_pool_selection_and_episode_identity(self):
        before = EXPERIMENT / "performance_fixes_v1/before/cuda_fdm__ppo_gpu.py"
        old = before.read_text(encoding="utf-8")
        new = (ROOT / "cuda_fdm/ppo_gpu.py").read_text(encoding="utf-8")
        old_pool = next(n for n in ast.parse(old).body if isinstance(n, ast.ClassDef) and n.name == "OpponentPool")
        new_pool = next(n for n in ast.parse(new).body if isinstance(n, ast.ClassDef) and n.name == "OpponentPool")
        # User explicitly approved subset MLP inference. Every other method,
        # including PFSP, stable identities, retirement and serialization stays
        # AST-identical; the different inference path has CPU/GPU A/B checks.
        old_pool.body = [n for n in old_pool.body if not isinstance(n, ast.FunctionDef) or n.name != "act"]
        new_pool.body = [n for n in new_pool.body if not isinstance(n, ast.FunctionDef) or n.name != "act"]
        self.assertEqual(ast.dump(old_pool), ast.dump(new_pool))
        reference = (ROOT / "cuda_fdm/tests/pool_assigned_val.py").read_text(encoding="utf-8")
        old_act = next(n for n in ast.parse(old).body if isinstance(n, ast.ClassDef) and n.name == "OpponentPool")
        old_act = next(n for n in old_act.body if isinstance(n, ast.FunctionDef) and n.name == "act")
        ref_act = next(n for n in ast.parse(reference).body if isinstance(n, ast.FunctionDef) and n.name == "legacy_full_act")
        # Different names/docstrings only; the reference computation is frozen.
        old_act.name = ref_act.name
        old_act.body = old_act.body[1:]
        ref_act.body = ref_act.body[1:]
        self.assertEqual(ast.dump(old_act), ast.dump(ref_act))
        for name in ("_sample_opp", "_pool_weights", "_refresh_weights"):
            self.assertEqual(definition(old, name, "PPOGPUTrainer"), definition(new, name, "PPOGPUTrainer"))

    def test_observation_geometry_and_physics_unchanged_outside_approved_altitude_block(self):
        old = (BEFORE / "cuda_fdm__obs_reward.py").read_text(encoding="utf-8")
        new = (ROOT / "cuda_fdm/obs_reward.py").read_text(encoding="utf-8")
        functions = [n.name for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)]
        for name in functions:
            self.assertEqual(definition(old, name), definition(new, name), name)
        for name in ("build_obs", "advance"):
            self.assertEqual(definition(old, name, "BatchObsReward"),
                             definition(new, name, "BatchObsReward"), name)
        old_kernel = (BEFORE / "cuda_fdm__gen__obs_kernel.cu").read_text(encoding="utf-8")
        new_kernel = (ROOT / "cuda_fdm/gen/obs_kernel.cu").read_text(encoding="utf-8")
        # Approved training-only label capture reuses already-computed values.
        # Strip exactly its delimited store and ABI additions, then compare ALL
        # remaining observation/reward/physics code against the frozen source.
        new_kernel, captures = re.subn(
            r"    // FUTURE_AUX_BEGIN:[^\n]*\n.*?    // FUTURE_AUX_END\n", "", new_kernel, flags=re.S)
        self.assertEqual(captures, 1)
        new_kernel = new_kernel.replace("const double* acth, float* out, double* aux)",
                                        "const double* acth, float* out)")
        new_kernel = new_kernel.replace(
            "double OSLON, double OCLON,\n    double* aux_features)", "double OSLON, double OCLON)")
        new_kernel = new_kernel.replace(
            "t_sec[e], act_hist + a*20, obs + a*214,\n"
            "                  (aux_features && !(a & 1)) ? aux_features + e*21 : nullptr);",
            "t_sec[e], act_hist + a*20, obs + a*214);")
        new_kernel = new_kernel.replace("const double* states, const unsigned char* env_mask,\n    const double* hp", 
                                        "const double* states,\n    const double* hp")
        new_kernel = new_kernel.replace("    if (!env_mask[a >> 1]) return;  // autoreset: untouched lanes already hold terminal obs\n", "")
        # The only later approved kernel behavior change is the exact match
        # horizon. Reward/observation formulas must still match the old source.
        new_kernel = new_kernel.replace(
            "// Same inclusive task horizon as dogfight.envs.termination (FP64 time drift).\n"
            "#define TIME_LIMIT_TOLERANCE_SEC 1.0e-8\n", "")
        new_kernel = new_kernel.replace(
            "(t_new >= max_time - TIME_LIMIT_TOLERANCE_SEC)", "(t_new > max_time)")
        # The later authorized altitude change is restricted to reward trackers,
        # the reward suffix, and two runtime scalar arguments. Physics, damage/HP
        # integration, angular-rate reconstruction and observation remain exact.
        old_advance = cuda_function(old_kernel, "advance_kernel").split("    // final-safe reward")[0]
        new_advance = cuda_function(new_kernel, "advance_kernel").split("    // Standard:")[0]
        new_advance = new_advance.replace("prev_alt_log", "prev_safety")
        new_advance = new_advance.replace("double win_r, double loss_r,\n",
            "double win_r, double loss_r, double own_alt_r, double tgt_alt_r,\n")
        new_advance = new_advance.replace("double timeout_draw_r,\n    int reward_mode, double alt_hunt_coef)",
                                         "double timeout_draw_r)")
        self.assertEqual(old_advance, new_advance)
        for name in ("low_alt_potential", "init_reward_kernel", "advance_kernel"):
            old_kernel = old_kernel.replace(cuda_function(old_kernel, name), "")
        for name in ("altitude_log", "init_reward_kernel", "advance_kernel"):
            new_kernel = new_kernel.replace(cuda_function(new_kernel, name), "")
        old_kernel = re.sub(r"#define LOW_ALT_[^\n]+\n", "", old_kernel)
        new_kernel = re.sub(r"#define ALT_LOG_[^\n]+\n", "", new_kernel)
        self.assertEqual(old_kernel, new_kernel)

    def test_frozen_observation_model_and_unmodified_geometry_sources(self):
        manifest = json.loads((EXPERIMENT / "manifest.json").read_text(encoding="utf-8"))
        for name in ("claude_code/my_observation.py", "claude_code/model.py"):
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                             manifest["source_sha256"][str(Path(name))], name)
        # Do not rebaseline the old manifest: compare all unchanged geometry
        # functions and every non-altitude config entry to its preserved source.
        old = (ALTITUDE_BEFORE / "claude_code__my_reward.py").read_text(encoding="utf-8")
        new = (ROOT / "claude_code/my_reward.py").read_text(encoding="utf-8")
        approved = {"initialize_reward_episode", "reset_distance_tracker", "compute_reward",
                    "_ownship_climb_rate_fps", "_low_altitude_potential"}
        for node in ast.parse(old).body:
            if isinstance(node, ast.FunctionDef) and node.name not in approved:
                self.assertEqual(definition(old, node.name), definition(new, node.name), node.name)
        def config(source):
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "MY_REWARD_CONFIG" for t in n.targets))
            return ast.literal_eval(node.value)
        old_cfg, new_cfg = config(old), config(new)
        removed = {"low_altitude_warning_start_ft", "low_altitude_full_penalty_ft",
                   "low_altitude_potential_budget", "low_altitude_descent_lookahead_sec",
                   "low_altitude_max_descent_rate_fps"}
        self.assertFalse(removed & set(new_cfg))
        for key in removed:
            old_cfg.pop(key)
        old_cfg.update(ownship_alt_reward=-10., target_alt_reward=10., altitude_terminal_mode="remaining_hp",
                       reward_mode=0, alt_hunt_coef=5.)
        self.assertEqual(old_cfg, new_cfg)

    def test_initialization_or_resume_failure_records_marker_without_saving(self):
        for stage in ("init", "resume"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "healthy.pt"
                path.write_bytes(b"healthy-checkpoint-sentinel")
                argv = ["train_gpu", "--device", "cpu", "--no-wandb", "--save", str(path)]
                fake = SimpleNamespace(load=lambda _: (_ for _ in ()).throw(ValueError("protocol mismatch")))
                if stage == "resume":
                    argv += ["--resume", str(path)]
                with patch.object(sys, "argv", argv), patch.object(train_gpu, "GpuDogfightVecEnv"), \
                        patch.object(train_gpu, "PPOGPUTrainer", return_value=fake,
                                     side_effect=FloatingPointError("bad warmup") if stage == "init" else None):
                    with self.assertRaises((FloatingPointError, ValueError)):
                        train_gpu.main()
                marker = json.loads((Path(tmp) / "INTEGRITY_FAILURE.json").read_text(encoding="utf-8"))
                self.assertEqual(marker["phase"], "initialization_or_resume")
                self.assertEqual(path.read_bytes(), b"healthy-checkpoint-sentinel")

    def test_checkpoint_save_failure_records_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "healthy.pt"
            path.write_bytes(b"healthy-checkpoint-sentinel")
            argv = ["train_gpu", "--device", "cpu", "--no-wandb", "--save", str(path)]
            def fail_save(_):
                raise FloatingPointError("bad optimizer")
            fake = SimpleNamespace(train=lambda **kwargs: [], save=fail_save)
            with patch.object(sys, "argv", argv), patch.object(train_gpu, "GpuDogfightVecEnv"), \
                    patch.object(train_gpu, "PPOGPUTrainer", return_value=fake):
                with self.assertRaises(FloatingPointError):
                    train_gpu.main()
            marker = json.loads((Path(tmp) / "INTEGRITY_FAILURE.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["phase"], "checkpoint_save")
            self.assertEqual(path.read_bytes(), b"healthy-checkpoint-sentinel")


if __name__ == "__main__":
    unittest.main()
