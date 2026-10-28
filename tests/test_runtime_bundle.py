"""Runtime staging must be explicit, hash-checked and non-destructive."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_runtime', ROOT / 'scripts/prepare_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeBundleTest(unittest.TestCase):
    def fixture(self, base):
        bundle = base / 'bundle'
        payload = bundle / 'win64/demo.dll'
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b'fixture; not executable')
        manifest = {'schema_version': 1, 'files': [{
            'path': 'win64/demo.dll', 'runtime_path': 'demo.dll',
            'size': payload.stat().st_size,
            'sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
        }]}
        (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return bundle, manifest

    def test_check_does_not_create_or_load_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            bundle, _ = self.fixture(base)
            target = base / 'target'
            runtime.prepare(bundle, target, install=False)
            self.assertFalse(target.exists())

    def test_install_is_idempotent_and_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            bundle, _ = self.fixture(base)
            target = base / 'target'
            runtime.prepare(bundle, target, install=True)
            runtime.prepare(bundle, target, install=True)
            self.assertEqual((target / 'demo.dll').read_bytes(), b'fixture; not executable')

    def test_corrupt_source_fails_before_creating_target(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            bundle, _ = self.fixture(base)
            (bundle / 'win64/demo.dll').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                runtime.prepare(bundle, base / 'target', install=True)
            self.assertFalse((base / 'target').exists())

    def test_conflicting_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            bundle, _ = self.fixture(base)
            target = base / 'target'
            target.mkdir()
            (target / 'demo.dll').write_bytes(b'keep my version')
            with self.assertRaises(FileExistsError):
                runtime.prepare(bundle, target, install=True)
            self.assertEqual((target / 'demo.dll').read_bytes(), b'keep my version')

    def test_unsafe_paths_and_case_duplicate_targets_are_rejected(self):
        for bad in ('../escape.dll', '/absolute.dll', 'C:/drive.dll', 'x\\escape.dll'):
            with self.subTest(path=bad), tempfile.TemporaryDirectory() as temp:
                bundle, manifest = self.fixture(Path(temp))
                manifest['files'][0]['runtime_path'] = bad
                (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    runtime.prepare(bundle, Path(temp) / 'target', install=True)
        with tempfile.TemporaryDirectory() as temp:
            bundle, manifest = self.fixture(Path(temp))
            manifest['files'].append(dict(manifest['files'][0], runtime_path='DEMO.DLL'))
            (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                runtime.prepare(bundle, Path(temp) / 'target', install=True)

    def test_published_manifest_and_exact_binary_allowlist(self):
        bundle = ROOT / 'third_party/competition_runtime'
        result = runtime.verify(bundle)
        dlls = {entry['path'] for entry in result['files'] if entry['path'].endswith('.dll')}
        self.assertEqual(dlls, {'win64/JSBSimAIPLib.dll', 'win64/AIP_BASE.dll', 'win64/AIP_BASE_target.dll'})


if __name__ == '__main__':
    unittest.main()
