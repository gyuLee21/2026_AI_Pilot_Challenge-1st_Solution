import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts' / 'train.py'


class ManagedTrainingTest(unittest.TestCase):
    def invoke(self, root, scenario='headon', name='demo', extra=()):
        return subprocess.run(
            [sys.executable, str(LAUNCHER), '--scenario', scenario,
             '--run-name', name, '--artifacts-root', str(root), '--dry-run', *extra],
            capture_output=True, text=True,
        )

    def test_scenarios_save_under_separate_roots_without_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'models'
            for scenario, directory in [('three_nine', '3-9'), ('headon', 'headon'), ('mixed', 'common')]:
                result = self.invoke(root, scenario, extra=('--', '--iters', '1234', '--rollout', '96'))
                self.assertEqual(result.returncode, 0, result.stderr)
                command = json.loads(result.stdout)['command']
                self.assertEqual(Path(command[command.index('--save') + 1]), root / directory / 'demo' / 'checkpoint.pt')
                self.assertEqual(command[command.index('--scenario') + 1], scenario)
                self.assertEqual(command[-4:], ['--iters', '1234', '--rollout', '96'])
            self.assertFalse(root.exists())

    def test_path_traversal_and_storage_overrides_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['../escape', '.', 'a/b', 'a\\b']:
                self.assertNotEqual(self.invoke(folder, name=name).returncode, 0)
            for override in ['--save', '--save=x.pt', '--sav', '--scenario', '--league-dir', '--resu']:
                self.assertNotEqual(self.invoke(folder, extra=('--', override, 'x')).returncode, 0)

    def test_existing_run_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'headon/demo/checkpoint.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fixture')
            self.assertNotEqual(self.invoke(root).returncode, 0)
            result = self.invoke(root, extra=('--resume',))
            self.assertEqual(result.returncode, 0, result.stderr)
            command = json.loads(result.stdout)['command']
            self.assertEqual(command[command.index('--resume') + 1], str(checkpoint))
            self.assertEqual(command[command.index('--save') + 1], str(checkpoint))

    def test_config_arguments_are_forwarded_and_missing_resume_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'models'
            config = Path(folder) / 'example.json'
            config.write_text(json.dumps({'arguments': ['--iters', '200', '--no-wandb']}))
            result = self.invoke(root, extra=('--config', str(config)))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['command'][-3:], ['--iters', '200', '--no-wandb'])
            self.assertNotEqual(self.invoke(root, extra=('--resume',)).returncode, 0)


if __name__ == '__main__':
    unittest.main()
