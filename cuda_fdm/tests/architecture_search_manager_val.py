"""CPU-only dry-run coverage for the optional MLP extension and integrity stops.

Run: python -B -m unittest cuda_fdm.tests.architecture_search_manager_val -v
All checkpoints, subprocesses and evaluations are mocked; no CUDA is imported.
Only temporary test directories are written.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cuda_fdm import architecture_search as manager


def ranking(names):
    return [dict(family=name, score=.7 - i * .05, crossplay=.7, reference=.6,
                 win_rate=.5, damage_diff=.1, seed_scores=[.7])
            for i, name in enumerate(names)]


class DrySearch(manager.Search):
    def __init__(self, folder, extension=False, fail_phase=None, preflight_passed=True):
        self.folder = Path(folder)
        self.stop = self.folder / "STOP"
        self.state = self.folder / "status.json"
        self.reference = self.folder / "reference.pt"
        self.manifest = {"source_sha256": {}}
        if extension:
            self.manifest["mlp_extension"] = copy.deepcopy(manager.MLP_EXTENSION)
        self.manifest_path = self.folder / "manifest.json"
        manager.write_json(self.manifest_path, self.manifest)
        self.events = []
        self.evaluations = {}
        self.fail_phase = fail_phase
        self.preflight_passed = preflight_passed

    def command(self, label, args, log_folder=None, retries=1, allow_failure=False):
        self.events.append(("command", label))
        if label == "preflight":
            manager.write_json(self.folder / "preflight/result.json", {"passed": self.preflight_passed})
        if label.startswith("latency-"):
            if self.mlp_extension() is not None:
                assert self._expanded_confirmation_complete
            manager.write_json(Path(args[args.index("--output") + 1]), {"passed": True})
        return 0

    def train_to(self, name, seed, target, final=False):
        if final and self.mlp_extension() is not None:
            assert self._expanded_confirmation_complete
        self.events.append(("train", name, seed, target, final))

    def snapshot(self, name, seed, iteration):
        return str(self.directory(name, seed) / f"iter_{iteration:05d}.pt")

    def evaluate(self, phase, candidates, refs, seed, extra_pairs=None):
        self.events.append(("evaluate", phase))
        self.evaluations[phase] = dict(candidates=candidates, refs=refs, seed=seed)
        if phase == self.fail_phase:
            raise FloatingPointError(f"non-finite evaluation: {phase}")
        if phase == "architecture_800":
            return ranking(["mlp", "gru"])
        if phase == "size_400":
            return ranking(["mlp672", "mlp512_d2", "mlp512", "mlp384"])
        if phase == "mlp_extension_800":
            # The baseline may win overall; only the best NEW model advances.
            return ranking(["mlp672", "mlp672_d4", "mlp768"])
        return ranking(list(dict.fromkeys(family for _, family in candidates.values())))


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aip-manager-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def original_decisions(self):
        manager.write_json(self.folder / "architecture_decision.json",
                           {"selected": "mlp", "ranking": ranking(["mlp", "gru"])})
        manager.write_json(self.folder / "screen_decision.json",
                           {"top_two": ["mlp672", "mlp512_d2"], "ranking": ranking(["mlp672", "mlp512_d2"])})

    def bare_search(self, extension=False):
        search = manager.Search.__new__(manager.Search)
        search.folder = self.folder
        search.stop = self.folder / "STOP"
        search.state = self.folder / "status.json"
        search.manifest = {"source_sha256": {}}
        if extension:
            search.manifest["mlp_extension"] = copy.deepcopy(manager.MLP_EXTENSION)
        return search

    def test_extension_preserves_decisions_and_compares_equal_budgets(self):
        self.original_decisions()
        search = DrySearch(self.folder, extension=True)
        preserved = {p: p.read_bytes() for p in [self.folder / "architecture_decision.json",
                                                self.folder / "screen_decision.json", search.manifest_path]}
        search.run()
        self.assertTrue(all(p.read_bytes() == content for p, content in preserved.items()))
        self.assertNotIn("architecture_800", search.evaluations)
        self.assertNotIn("size_400", search.evaluations)
        comparison = search.evaluations["mlp_extension_800"]
        self.assertEqual(comparison["seed"], 74001)
        self.assertEqual(set(comparison["candidates"]), {"mlp672_s0", "mlp768_s0", "mlp672_d4_s0"})
        self.assertTrue(all(Path(path).name == "iter_00800.pt" for path, _ in comparison["candidates"].values()))
        self.assertEqual(set(comparison["refs"]),
                         {"reference_gru100", "reference_gru512_h256_200", "reference_mlp672_200"})
        training = [event for event in search.events if event[0] == "train"]
        self.assertEqual([e for e in training if e[3] == 800],
                         [("train", "mlp768", 0, 800, False), ("train", "mlp672_d4", 0, 800, False)])
        confirmed = {(e[1], e[2]) for e in training if e[3] == 1500}
        self.assertEqual(confirmed, {(name, seed) for name in ("mlp672", "mlp512_d2", "mlp672_d4")
                                     for seed in range(3)})
        self.assertFalse(any(e[1] == "mlp768" and e[3] > 800 for e in training))

    def test_expanded_refs_curves_and_final_order(self):
        self.original_decisions()
        search = DrySearch(self.folder, extension=True)
        search.run()
        final = search.evaluations["confirmation_expanded_1500"]
        self.assertEqual(final["seed"], 94001)
        self.assertEqual(len(final["candidates"]), 9)
        self.assertTrue({"reference_mlp768_400", "reference_mlp672_d4_400"} <= set(final["refs"]))
        for iteration in (500, 1000):
            phase = f"confirmation_expanded_curve_{iteration}"
            self.assertEqual(search.evaluations[phase]["refs"], final["refs"])
            self.assertNotIn(f"confirmation_curve_{iteration}", search.evaluations)
        confirmation_index = search.events.index(("evaluate", "confirmation_expanded_1500"))
        for index, event in enumerate(search.events):
            if (event[0] == "command" and event[1].startswith(("bundle-", "latency-"))) or \
                    (event[0] == "train" and event[-1]):
                self.assertGreater(index, confirmation_index)
        self.assertEqual(len([e for e in search.events if e[0] == "train" and e[-1]]), 1)

    def test_failed_expanded_confirmation_cannot_reach_latency_or_final(self):
        self.original_decisions()
        search = DrySearch(self.folder, extension=True, fail_phase="confirmation_expanded_1500")
        with self.assertRaises(FloatingPointError):
            search.run()
        self.assertFalse(search._expanded_confirmation_complete)
        self.assertFalse(any(e[0] == "command" and e[1].startswith(("bundle-", "latency-")) for e in search.events))
        self.assertFalse(any(e[0] == "train" and e[-1] for e in search.events))

    def test_direct_final_training_requires_expanded_confirmation(self):
        search = self.bare_search(extension=True)
        with patch.object(search, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "completed expanded confirmation"):
                search.train_to("mlp672", 0, 10000, final=True)
        command.assert_not_called()
        self.assertFalse((self.folder / "final_10000").exists())

    def test_opt_out_retains_original_schedule(self):
        search = DrySearch(self.folder)
        with patch.object(manager, "checkpoint_iteration", return_value=2):
            search.run()
        self.assertEqual(list(search.evaluations), ["architecture_800", "size_400", "confirmation_1500",
                         "architecture_curve_200", "architecture_curve_400", "confirmation_curve_500",
                         "confirmation_curve_1000"])
        training = [e for e in search.events if e[0] == "train"]
        self.assertFalse(any(e[1] in manager.MLP_EXTENSION["models"] for e in training))
        self.assertEqual({e[1] for e in training if e[3] == 1500}, {"mlp672", "mlp512_d2"})

    def test_changed_extension_config_fails_before_preflight(self):
        search = DrySearch(self.folder, extension=True)
        search.manifest["mlp_extension"]["comparison_seed"] += 1
        with self.assertRaisesRegex(ValueError, "frozen approved"):
            search.run()
        self.assertEqual(search.events, [])

    def test_hold_and_root_failure_block_run_and_command(self):
        for marker_name in ("INTEGRITY_HOLD.json", "INTEGRITY_FAILURE.json"):
            with self.subTest(marker=marker_name):
                marker = self.folder / marker_name
                manager.write_json(marker, {"reason": "audit hold"})
                search = self.bare_search()
                with patch.object(manager.subprocess, "Popen") as popen:
                    with self.assertRaises(FloatingPointError):
                        search.command("test", ["-m", "does_not_run"])
                    with self.assertRaises(FloatingPointError):
                        search.run()
                    popen.assert_not_called()
                marker.unlink()

    def test_child_failure_after_exit_is_not_retried_or_allowed(self):
        search = self.bare_search()
        logs = self.folder / "logs"
        output = self.folder / "evaluation/results.json"
        proc = SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)

        def child(*args, **kwargs):
            manager.write_json(output.parent / "INTEGRITY_FAILURE.json", {"reason": "NaN"})
            return proc

        with patch.object(manager.subprocess, "Popen", side_effect=child) as popen:
            with self.assertRaises(FloatingPointError):
                search.command("eval-test", ["--output", str(output)], logs, retries=3, allow_failure=True)
            self.assertEqual(popen.call_count, 1)

    def test_child_marker_destination_inference(self):
        search = self.bare_search()
        for flag, destination in [("--save", self.folder / "train/checkpoint.pt"),
                                  ("--output", self.folder / "preflight"),
                                  ("--output-dir", self.folder / "bundle")]:
            with self.subTest(flag=flag):
                directory = destination.parent if flag == "--save" else destination
                marker = directory / "INTEGRITY_FAILURE.json"
                manager.write_json(marker, {"reason": "prior failure"})
                with patch.object(manager.subprocess, "Popen") as popen:
                    with self.assertRaises(FloatingPointError):
                        search.command("test", [flag, str(destination)])
                    popen.assert_not_called()
                marker.unlink()

    def test_failed_preflight_exit_is_terminal_without_marker(self):
        search = self.bare_search()
        proc = SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)
        with patch.object(manager.subprocess, "Popen", return_value=proc) as popen:
            with self.assertRaisesRegex(FloatingPointError, "preflight failed"):
                search.command("preflight", ["--output", str(self.folder / "preflight")], retries=3)
            self.assertEqual(popen.call_count, 1)

    def test_failed_preflight_result_stops_before_training(self):
        search = DrySearch(self.folder, preflight_passed=False)
        with self.assertRaisesRegex(FloatingPointError, "preflight failed"):
            search.run()
        self.assertEqual(search.events, [("command", "preflight")])

    def test_training_numerical_failure_bypasses_retry(self):
        search = self.bare_search()
        with patch.object(search, "command", side_effect=FloatingPointError("bad state")) as command:
            with self.assertRaises(FloatingPointError):
                search.train_to("mlp768", 0, 800)
            self.assertEqual(command.call_count, 1)

    def test_new_training_keeps_original_hyperparameters_and_intermediate_snapshot(self):
        search = self.bare_search()
        with patch.object(search, "command") as command, \
                patch.object(manager, "checkpoint_iteration", return_value=800):
            search.train_to("mlp768", 0, 800)
        args = command.call_args.args[1]
        for flag, value in {"--hidden": "768,768,768", "--nenv": "4096", "--rollout": "64",
                            "--epochs": "4", "--minibatches": "8", "--gamma": "0.997",
                            "--substeps": "6", "--iters": "800"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        retained = args[args.index("--keep-iterations") + 1].split(",")
        self.assertTrue({"400", "800", "1500"} <= set(retained))

    def test_main_records_terminal_integrity_failure(self):
        search = self.bare_search()
        search.run = Mock(side_effect=FloatingPointError("non-finite preflight"))
        with patch.object(manager, "Search", return_value=search), \
                patch.object(manager.sys, "argv", ["architecture_search", "--root", str(self.folder)]):
            with self.assertRaises(FloatingPointError):
                manager.main()
        self.assertEqual(manager.read_json(search.state)["phase"], "failed")
        self.assertTrue(manager.read_json(search.state)["integrity_failure"])
        self.assertEqual(manager.read_json(self.folder / "INTEGRITY_FAILURE.json")["reason"],
                         "non-finite preflight")

    def test_partial_or_mismatched_evaluation_cannot_rank(self):
        search = self.bare_search()
        candidates = {"a": ("a.pt", "a"), "b": ("b.pt", "b")}
        for defect in ("missing_match", "wrong_game_count", "wrong_spec", "invalid_score"):
            with self.subTest(defect=defect):
                def child(label, args, folder, **kwargs):
                    spec = manager.read_json(folder / "spec.json")
                    matches = {}
                    for scenario in ("three_nine", "headon"):
                        key = f"a|b|{scenario}|reset0"
                        matches[key] = dict(a="a", b="b", scenario=scenario, reset_a=False,
                                            records=[dict(score=.5, damage_diff=0.) for _ in range(spec["games"])])
                    result = {"spec": copy.deepcopy(spec), "matches": matches}
                    if defect == "missing_match":
                        matches.pop("a|b|headon|reset0")
                    elif defect == "wrong_game_count":
                        matches["a|b|headon|reset0"]["records"].pop()
                    elif defect == "wrong_spec":
                        result["spec"]["seed"] += 1
                    else:
                        # JSON fixture serialization deliberately avoids invalid NaN JSON.
                        result["matches"]["a|b|headon|reset0"]["records"][0]["score"] = None
                    manager.write_json(folder / "results.json", result)

                with patch.object(search, "command", side_effect=child):
                    with self.assertRaises(FloatingPointError):
                        search.evaluate(defect, candidates, {}, 94001)
                self.assertFalse((self.folder / "evaluation" / defect / "ranking.json").exists())

    def test_complete_evaluation_is_eligible_to_rank(self):
        spec = {"pairs": [["a", "b"]], "games": 2, "seed": 94001}
        matches = {f"a|b|{scenario}|reset0": dict(a="a", b="b", scenario=scenario, reset_a=False,
                   records=[dict(score=.5, damage_diff=0.), dict(score=1., damage_diff=.1)])
                   for scenario in ("three_nine", "headon")}
        manager.verify_evaluation_complete({"spec": spec, "matches": matches}, spec)

    def test_nonfinite_evaluation_values_are_terminal(self):
        spec = {"pairs": [["a", "b"]], "games": 1, "seed": 94001}
        for field in ("score", "damage_diff"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    matches = {f"a|b|{scenario}|reset0": dict(a="a", b="b", scenario=scenario, reset_a=False,
                               records=[dict(score=.5, damage_diff=0.)]) for scenario in ("three_nine", "headon")}
                    matches["a|b|headon|reset0"]["records"][0][field] = value
                    with self.assertRaises(FloatingPointError):
                        manager.verify_evaluation_complete({"spec": spec, "matches": matches}, spec)


if __name__ == "__main__":
    unittest.main()
