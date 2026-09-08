"""External contest coordinate, altitude, and deployment-policy contracts."""
from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

import torch

from cuda_fdm import rl_env
from cuda_fdm.ppo_gpu import PPOGPUTrainer, TRAINING_PROTOCOL
from claude_code.my_observation import damage_rate


RUNTIME = Path(__file__).resolve().parents[2]


class OfficialContractTests(unittest.TestCase):
    def test_official_origin_is_the_only_release_runtime_origin(self):
        self.assertEqual(rl_env.ORIGIN_LAT_DEG, 37.240778)
        self.assertEqual(rl_env.ORIGIN_LON_DEG, 131.869556)
        checked = (
            RUNTIME / "cuda_fdm" / "rl_env.py",
            RUNTIME / "cuda_fdm" / "obs_reward.py",
            RUNTIME / "FighterSim.py",
            RUNTIME / "claude_code" / "validate_jsbsim_port.py",
        )
        for path in checked:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("37.91455691666666", source, str(path))
            self.assertNotIn("128.18188127777776", source, str(path))
            self.assertIn("37.240778", source, str(path))
            self.assertIn("131.869556", source, str(path))

    def test_state_down_is_direct_jsbsim_msl_not_tangent_vertical(self):
        # This aircraft is above the official 1000ft MSL hard deck, while the
        # former tangent-D projection would place it below that same boundary.
        # Horizontal position must not change the exported MSL altitude.
        altitude = torch.tensor([305.5], dtype=torch.float64)
        lat = torch.tensor([(37.240778 + 3957.2 / 111_320.0) * math.pi / 180.0],
                           dtype=torch.float64)
        lon = torch.tensor([131.869556 * math.pi / 180.0], dtype=torch.float64)
        x, y, z = rl_env._geodetic2ecef(lat, lon, altitude)
        ecef = torch.stack((x, y, z), dim=1)
        north, east, down = rl_env.ned_from_ecef_altasl(ecef, altitude)
        self.assertTrue(torch.isfinite(torch.stack((north, east, down))).all())
        torch.testing.assert_close(down, -altitude, rtol=0, atol=0)
        dx, dy, dz = x - rl_env._OX, y - rl_env._OY, z - rl_env._OZ
        legacy_down = -(
            rl_env._O_CLAT * rl_env._O_CLON * dx
            + rl_env._O_CLAT * rl_env._O_SLON * dy
            + rl_env._O_SLAT * dz
        )
        self.assertGreater(float(-down[0]), rl_env.HARD_DECK_M)
        self.assertLess(float(-legacy_down[0]), rl_env.HARD_DECK_M)

    def test_ic_seed_keeps_requested_msl_and_exact_1000ft_cutoff(self):
        _, _, altitude_ft = rl_env.ned_to_geodetic_np(
            3957.2, 0.0, -305.5)
        self.assertAlmostEqual(altitude_ft * rl_env.FT2M, 305.5, places=10)
        default = inspect.signature(rl_env.GpuDogfightVecEnv.__init__).parameters[
            "min_altitude_m"].default
        self.assertEqual(rl_env.HARD_DECK_FT, 1000.0)
        self.assertEqual(rl_env.HARD_DECK_M, 304.8)
        self.assertEqual(default, 304.8)

    def test_damage_phases_use_phase_specific_range_normalization(self):
        # Attached/release contract: phase 1 has priority, while phases 2/3
        # normalize with their own 3500ft/4000ft maximum ranges.
        self.assertAlmostEqual(damage_rate(1000.0, 0.5, 160.0), 0.8, places=12)
        self.assertAlmostEqual(
            damage_rate(3200.0, 1.5, 120.0),
            0.3 * (3500.0 - 3200.0) / (3500.0 - 500.0), places=12)
        self.assertAlmostEqual(
            damage_rate(3800.0, 2.5, 160.0),
            0.1 * (4000.0 - 3800.0) / (4000.0 - 500.0), places=12)

    def test_cuda_kernel_uses_direct_msl_down(self):
        source = (RUNTIME / "cuda_fdm" / "gen" / "obs_kernel.cu").read_text(
            encoding="utf-8")
        self.assertIn("s9[2] = -alt_asl_m;", source)
        self.assertNotIn(
            "s9[2] = -(OCLAT*OCLON*dx + OCLAT*OSLON*dy + OSLAT*dz);", source)

    def test_league_evaluates_the_packaged_submission_policy(self):
        submission = (RUNTIME / "claude_code" / "submission_client.py").read_text(
            encoding="utf-8")
        builder = (RUNTIME / "claude_code" / "build_submission.py").read_text(
            encoding="utf-8")
        evaluator = inspect.getsource(PPOGPUTrainer._run_clean_paired_evaluation)
        self.assertIn('ENTRY = ROOT / "claude_code" / "submission_client.py"', builder)
        self.assertIn("MLPActionProvider(bundle_dir=str(bundle_dir), stochastic=True)",
                      submission)
        self.assertIn("sample=True", evaluator)
        self.assertIn("active_league_v15_posterior_nash", TRAINING_PROTOCOL)

    def test_admission_source_requires_point_and_lcb(self):
        source = inspect.getsource(PPOGPUTrainer._evaluate_and_admit_exploiter)
        self.assertIn('target_eval["score"] >= self.cfg.league_admission_score', source)
        self.assertIn('and target_eval["lcb95"] >= self.cfg.league_admission_lcb', source)
        self.assertNotIn('target_eval["score"] >= self.cfg.league_admission_score\n                    or',
                         source)


if __name__ == "__main__":
    unittest.main()
