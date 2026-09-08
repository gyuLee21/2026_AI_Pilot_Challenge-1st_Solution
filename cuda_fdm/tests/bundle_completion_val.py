"""Actual CPU converter + finalization/restart/fail-closed regression."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from claude_code.model import make_actor_critic, load_bundle, WEIGHTS_FILENAME
from cuda_fdm import gpu_ckpt_to_bundle as converter
from cuda_fdm import mlp_size_search as manager
from cuda_fdm.bundle_completion import complete_final_bundle, RECEIPT


class BundleCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bundle-finalize-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.final = self.root / "final_20000"; self.final.mkdir()
        self.ckpt, self.bundle = self.final / "checkpoint.pt", self.final / "bundle"
        self.selected = "mlp512_d2"
        self.model = make_actor_critic(obs_dim=214, act_dim=4, num_bins=3,
            hidden=(8, 8), activation="tanh", gru_size=0)
        torch.save(dict(model=self.model.state_dict(), cfg=dict(hidden=(8, 8),
            activation="tanh", architecture="mlp", gru_size=0, num_bins=3, aux_pred=False),
            norm=None, iteration=20000), self.ckpt)
        self.export_count = 0
        def export(label, args):
            self.export_count += 1
            self.assertEqual(label, "final-bundle")
            cfg = SimpleNamespace(ckpt=args[args.index("--ckpt")+1],
                output_dir=args[args.index("--output-dir")+1],
                observation_module="claude_code.my_observation", reward_module="claude_code.my_reward")
            with patch.object(converter, "parse_args", return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
                converter.main()
        self.search = SimpleNamespace(folder=self.root, command=export)

    def finish(self):
        return complete_final_bundle(self.search, self.selected)

    def hashes(self):
        return {p.relative_to(self.final).as_posix(): (manager.sha(p), p.stat().st_mtime_ns)
                for p in self.final.rglob("*") if p.is_file() and p.name != "bundle_finalize.lock"}

    def test_repeat_completion_is_noop_and_parameters_are_exact(self):
        record = self.finish(); before = self.hashes()
        self.assertEqual(record, self.finish())
        self.assertEqual(before, self.hashes())
        self.assertEqual(self.export_count, 1)
        loaded, _ = load_bundle(self.bundle, device="cpu")
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, loaded.state_dict()[name], atol=0, rtol=0)

    def test_changed_checkpoint_rejected_without_reexport(self):
        self.finish()
        with self.ckpt.open("ab") as stream:
            stream.write(b"changed")
        before = self.hashes()
        with self.assertRaises(FloatingPointError):
            self.finish()
        self.assertEqual(self.export_count, 1); self.assertEqual(before, self.hashes())

    def test_changed_weight_hash_rejected_without_reexport(self):
        self.finish()
        with (self.bundle / WEIGHTS_FILENAME).open("ab") as stream:
            stream.write(b"changed")
        before = self.hashes()
        with self.assertRaisesRegex(FloatingPointError, "hash mismatch"):
            self.finish()
        self.assertEqual(self.export_count, 1); self.assertEqual(before, self.hashes())

    def test_changed_selection_rejected(self):
        self.finish(); self.selected = "mlp1024_d4"
        with self.assertRaisesRegex(FloatingPointError, "selection mismatch"):
            self.finish()
        self.assertEqual(self.export_count, 1)

    def test_extra_file_rejected_without_deletion(self):
        self.finish(); extra = self.bundle / "unexpected.txt"; extra.write_text("preserve")
        with self.assertRaisesRegex(FloatingPointError, "file set"):
            self.finish()
        self.assertEqual(extra.read_text(), "preserve")

    def test_incomplete_staging_preserved_for_review(self):
        pending = self.final / "bundle.pending"; pending.mkdir()
        partial = pending / WEIGHTS_FILENAME; partial.write_bytes(b"partial")
        with self.assertRaisesRegex(FloatingPointError, "incomplete bundle staging"):
            self.finish()
        self.assertEqual(partial.read_bytes(), b"partial"); self.assertEqual(self.export_count, 0)

    def test_existing_unverified_bundle_is_not_overwritten(self):
        self.bundle.mkdir(); partial = self.bundle / WEIGHTS_FILENAME; partial.write_bytes(b"preserve")
        with self.assertRaisesRegex(FloatingPointError, "lacks a complete export receipt"):
            self.finish()
        self.assertEqual(partial.read_bytes(), b"preserve"); self.assertEqual(self.export_count, 0)

    def test_restart_after_rename_recovers_without_reexport(self):
        real_freeze = manager.freeze_json
        def interrupt(path, data):
            if Path(path).name == "bundle_verified.json":
                raise RuntimeError("interrupted after rename")
            return real_freeze(path, data)
        with patch.object(manager, "freeze_json", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.finish()
        before = {p.name: manager.sha(p) for p in self.bundle.iterdir()}
        self.assertTrue(self.finish()["passed"])
        self.assertEqual(self.export_count, 1)
        self.assertEqual(before, {p.name: manager.sha(p) for p in self.bundle.iterdir()})

    def test_restart_after_completed_staging_promotes_without_reexport(self):
        real_rename = Path.rename
        def interrupt(path, target):
            if path.name == "bundle.pending":
                raise OSError("interrupted before rename")
            return real_rename(path, target)
        with patch.object(Path, "rename", interrupt):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.finish()
        self.assertFalse(self.bundle.exists())
        self.assertTrue((self.final / "bundle.pending" / RECEIPT).is_file())
        self.assertTrue(self.finish()["passed"]); self.assertEqual(self.export_count, 1)

    def test_checkpoint_modified_during_export_rejected(self):
        exporter = self.search.command
        def mutate(*args):
            exporter(*args)
            with self.ckpt.open("ab") as stream:
                stream.write(b"changed while exporting")
        self.search.command = mutate
        with self.assertRaisesRegex(FloatingPointError, "changed during bundle export"):
            self.finish()
        self.assertFalse((self.final / "bundle_verified.json").exists())

    def test_nonfinite_model_fails_before_completion(self):
        checkpoint = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        next(iter(checkpoint["model"].values())).fill_(float("nan"))
        torch.save(checkpoint, self.ckpt)
        with self.assertRaises(FloatingPointError):
            self.finish()
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.final / "bundle_verified.json").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
