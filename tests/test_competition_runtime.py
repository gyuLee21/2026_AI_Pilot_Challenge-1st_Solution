"""No GPU or weights: verify the restored lineage and launcher isolation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / 'research/competition_36500'


class CompetitionRuntimeTest(unittest.TestCase):
    def test_preserved_source_hashes(self):
        manifest = json.loads((LINEAGE / 'source_manifest.json').read_text())
        self.assertGreater(len(manifest['files']), 40)
        for entry in manifest['files']:
            path = LINEAGE / 'runtime' / entry['path']
            with self.subTest(path=entry['path']):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])

    def test_recorded_checkpoint_actor_and_critic_are_184_not_204(self):
        record = json.loads((LINEAGE / 'checkpoint_settings.json').read_text())
        self.assertEqual(record['iteration'], 36500)
        self.assertEqual(record['model_shapes']['actor_logits.0.weight'], [384, 184])
        self.assertEqual(record['model_shapes']['critic.0.weight'], [384, 184])
        self.assertEqual(record['config']['rollout_steps'], 104)

    def test_command_uses_isolated_runtime_and_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'models'
            command = [sys.executable, str(ROOT / 'scripts/train.py'),
                       '--runtime', 'competition36500', '--scenario', 'three_nine',
                       '--run-name', 'fixture', '--artifacts-root', str(output),
                       '--config', str(ROOT / 'configs/training/competition36500.example.json'),
                       '--dry-run']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(Path(plan['cwd']), LINEAGE / 'runtime')
            self.assertIn('--active-league', plan['command'])
            self.assertNotIn('--scenario', plan['command'])
            self.assertFalse(output.exists())
            command[command.index('three_nine')] = 'headon'
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_fuel_and_fcs_constants_restored_without_changing_primary_runtime(self):
        env = (LINEAGE / 'runtime/cuda_fdm/rl_env.py').read_text(encoding='utf-8')
        self.assertIn('fuel_lbs=3000.0', env)
        self.assertIn('self.scenario_b_prob = 0.0', env)
        self.assertIn('def _initialize_physical_seeds', env)
        self.assertIn('self.ic_pool = self._initialize_physical_seeds(pool)', env)
        fdm = (LINEAGE / 'runtime/cuda_fdm/gen/fdm.cuh').read_text(encoding='utf-8')
        self.assertIn('#define FCS_DT (1.0/60.0)', fdm)
        current = (ROOT / 'cuda_fdm/rl_env.py').read_text(encoding='utf-8')
        self.assertIn('fuel_lbs=6000.0', current)


if __name__ == '__main__':
    unittest.main()
