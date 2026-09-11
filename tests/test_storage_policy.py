from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StoragePolicyTest(unittest.TestCase):
    def test_local_payloads_and_campaigns_cannot_be_added_accidentally(self):
        paths = [
            'artifacts/models/rl/headon/demo/checkpoint.pt',
            'artifacts/models/bt/core/Rule.xml',
            'artifacts/models/mpc/controller/main.py',
            'artifacts/models/external/inbox/team_x/model/policy.py',
            'artifacts/models/catalog.local.json',
            'configs/evaluation/private.local.json',
            'evaluation/campaign.headon.json',
            'evaluation/smoke.headon.json',
        ]
        result = subprocess.run(['git', 'check-ignore', '--no-index', *paths],
                                cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(set(result.stdout.splitlines()), set(paths))

    def test_documentation_and_example_configs_are_trackable(self):
        paths = ['artifacts/README.md', 'configs/training/minimal.example.json']
        result = subprocess.run(['git', 'check-ignore', '--no-index', *paths],
                                cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
