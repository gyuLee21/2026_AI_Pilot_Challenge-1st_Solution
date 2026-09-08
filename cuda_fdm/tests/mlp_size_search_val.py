"""CPU-only dry orchestration tests for the new depth-first auxiliary cohort."""
import copy
import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cuda_fdm import mlp_size_search as m
from cuda_fdm.future_aux import AUX_CONTRACT
from cuda_fdm.reward_modes import REWARD_CONTRACT
from cuda_fdm.train_gpu import csv_header


def ranked(names):
    return [dict(family=name, score=.8-i*.1, crossplay=.8-i*.1, reference=.6,
                 win_rate=.5, damage_diff=.1, seed_scores=[.8-i*.1]*3, seed_mean_ci95=[.5, 1.])
            for i, name in enumerate(names)]


class DrySearch(m.MLPSearch):
    def __init__(self, folder, depth=3, width_winner=768, fail=None, no_latency=False):
        self.folder = Path(folder)
        self.state, self.stop = self.folder / "status.json", self.folder / "STOP"
        self.stage, self._confirmation_complete = "init", False
        self.events, self.evals, self.budgets = [], {}, {}
        self.depth, self.width_winner, self.fail, self.no_latency = depth, width_winner, fail, no_latency

    def check_sources(self):
        pass

    def preflight(self):
        self.events.append(("preflight",))
        if self.fail == "preflight":
            raise m.IntegrityError("preflight failure")

    def train_to(self, name, seed, target, final=False):
        if final and not self._confirmation_complete:
            raise m.IntegrityError("main was reached too early")
        key = (name, seed, final)
        if self.budgets.get(key, 0) >= target:
            return
        self.events.append(("train", name, seed, target, final))
        self.budgets[key] = target

    def snapshot(self, name, seed, iteration):
        if self.budgets.get((name, seed, False), 0) < iteration:
            raise AssertionError(f"missing same-budget snapshot: {name}/{seed}/{iteration}")
        return str(self.directory(name, seed) / f"iter_{iteration:05d}.pt")

    def evaluate(self, phase, candidates, refs, seed, extra_pairs=None):
        self.events.append(("eval", phase))
        self.evals[phase] = dict(candidates=candidates, refs=refs, seed=seed)
        if phase == self.fail:
            raise m.IntegrityError("invalid evaluation")
        if phase == "depth_800":
            return ranked([m.model_name(512, self.depth)] + [m.model_name(512, d) for d in m.DEPTHS if d != self.depth])
        if phase == "width_extended_800":
            return ranked([m.model_name(w, self.depth) for w in (384, 512, 768, 1024)])
        if phase == "width_800":
            order = [self.width_winner] + [w for w in m.WIDTHS if w != self.width_winner]
            return ranked([m.model_name(w, self.depth) for w in order])
        return ranked(list(dict.fromkeys(family for _, family in candidates.values())))

    def latency_gate(self, name, checkpoint, label):
        self.events.append(("latency", name, label))
        return not self.no_latency

    def command(self, label, args, *a, **kw):
        self.events.append(("command", label, args))
        return 0


class MLPSearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="mlp-new-search-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def bare(self):
        search = m.MLPSearch.__new__(m.MLPSearch)
        search.folder = self.folder
        search.stop, search.state = self.folder / "STOP", self.folder / "status.json"
        search.stage, search._confirmation_complete = "test", False
        search.manifest = {"source_sha256": {}}
        search.check_sources = lambda: None
        return search

    def write_metrics(self, folder, count=2, **changes):
        path = folder / "metrics.csv"
        fields = csv_header(True).split(",")
        rows = []
        for iteration in range(1, count+1):
            row = dict.fromkeys(fields, "0.1")
            row.update(iter=str(iteration), gstep=str(iteration*4096*64), eps="5", early_stop="False")
            row.update(changes)
            rows.append(row)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        return path

    def test_depth_then_width_no_crossproduct_and_no_gru(self):
        search = DrySearch(self.folder, depth=2)
        search.run()
        d = search.evals["depth_800"]
        self.assertEqual(len(d["candidates"]), 9)
        self.assertEqual({family for _, family in d["candidates"].values()}, {"mlp512_d2", "mlp512_d3", "mlp512_d4"})
        self.assertEqual(len(d["refs"]), 3)
        self.assertTrue(all(Path(p).name == "iter_00800.pt" for p, _ in d["candidates"].values()))
        w = search.evals["width_800"]
        self.assertEqual(m.WIDTHS, (512, 768, 1024))
        self.assertEqual(len(w["candidates"]), 3)
        self.assertEqual({family for _, family in w["candidates"].values()}, {f"mlp{v}_d2" for v in m.WIDTHS})
        self.assertEqual(w["refs"], d["refs"])
        depth_index = search.events.index(("eval", "depth_800"))
        width_training = [i for i, e in enumerate(search.events) if e[0] == "train" and not e[1].startswith("mlp512")]
        self.assertTrue(all(i > depth_index for i in width_training))
        self.assertFalse(any("gru" in str(e).lower() for e in search.events))

    def test_512_reused_and_final_three_seeds(self):
        search = DrySearch(self.folder)
        search.run()
        self.assertEqual(search.events.count(("train", "mlp512_d3", 0, 800, False)), 1)
        final = search.evals["confirmation_1500"]
        self.assertEqual(len(final["candidates"]), 6)
        self.assertEqual({family for _, family in final["candidates"].values()}, {"mlp768_d3", "mlp512_d3"})
        self.assertTrue(all(Path(p).name == "iter_01500.pt" for p, _ in final["candidates"].values()))
        self.assertEqual(final["seed"], 95001)
        self.assertEqual(len(final["refs"]), 6)
        for it in (500, 1000):
            self.assertEqual(search.evals[f"confirmation_curve_{it}"]["refs"], final["refs"])
        self.assertEqual([e for e in search.events if e[0] == "train" and e[-1]],
                         [("train", "mlp768_d3", 0, 20000, True)])

    def test_lower_extension_only_512_win_once_same_depth_keep_eliminated_refs(self):
        search = DrySearch(self.folder, depth=4, width_winner=512)
        search.run()
        self.assertEqual(search.events.count(("train", "mlp384_d4", 0, 800, False)), 1)
        self.assertEqual(search.events.count(("train", "mlp1024_d4", 0, 800, False)), 1)
        self.assertGreater(search.events.index(("train", "mlp384_d4", 0, 800, False)),
                           search.events.index(("eval", "width_800")))
        candidates = search.evals["width_extended_800"]["candidates"]
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(family.endswith("_d4") for _, family in candidates.values()))
        self.assertTrue(all(Path(p).name == "iter_00800.pt" for p, _ in candidates.values()))
        self.assertEqual(search.evals["width_extended_800"]["seed"], search.evals["width_800"]["seed"])
        self.assertEqual(search.evals["width_extended_800"]["refs"], search.evals["width_800"]["refs"])
        self.assertEqual(len(search.evals["confirmation_1500"]["refs"]), 7)
        self.assertTrue({"ref_mlp384_d4_400", "ref_mlp1024_d4_400"} <= set(search.evals["confirmation_1500"]["refs"]))
        self.assertFalse(any("1280" in str(e) for e in search.events))

    def test_no_lower_or_upper_extension_when_768_or_1024_wins(self):
        for winner in (768, 1024):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as tmp:
                search = DrySearch(tmp, width_winner=winner)
                search.run()
                self.assertNotIn("width_extended_800", search.evals)
                self.assertFalse(any(e[0] == "train" and e[1].startswith("mlp384") for e in search.events))
                self.assertFalse(any("1280" in str(e) for e in search.events))

    def test_latency_failure_blocks_main_not_user_requested_lower_probe(self):
        search = DrySearch(self.folder, width_winner=512, no_latency=True)
        with self.assertRaises(m.IntegrityError):
            search.run()
        self.assertIn("width_extended_800", search.evals)
        self.assertFalse(any(e[0] == "train" and e[-1] for e in search.events))

    def test_incomplete_lower_probe_evaluation_blocks_main(self):
        search = DrySearch(self.folder, width_winner=512, fail="width_extended_800")
        with self.assertRaises(m.IntegrityError):
            search.run()
        self.assertFalse(any(e[0] == "train" and e[-1] for e in search.events))

    def test_incomplete_evaluation_or_preflight_stops_before_main(self):
        for fail in ("preflight", "depth_800", "width_800", "confirmation_1500"):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                search = DrySearch(tmp, fail=fail)
                with self.assertRaises(m.IntegrityError):
                    search.run()
                self.assertFalse(any(e[0] == "train" and e[-1] for e in search.events))

    def test_args_aux_on_fixed_settings_and_wandb_main_only(self):
        search = self.bare()
        for final in (False, True):
            args, _ = search.train_args("mlp768_d4", 1, 20000 if final else 800, final)
            for flag, value in {"--architecture": "mlp", "--hidden": "768,768,768,768", "--aux-coef": "0.1",
                    "--gamma": "0.997", "--nenv": "4096", "--rollout": "64", "--epochs": "4",
                    "--minibatches": "8", "--substeps": "6", "--save-every": "100"}.items():
                self.assertEqual(args[args.index(flag)+1], value)
            self.assertIn("--aux-pred", args)
            self.assertIn("--save-runtime", args)
            self.assertNotIn("--resume", args)
            self.assertIn("--wandb" if final else "--no-wandb", args)
            if final:
                self.assertEqual(args[args.index("--wandb-entity")+1], "leeai021213-ajou-university")
                self.assertEqual(args[args.index("--milestone-period")+1], "500")
                self.assertEqual(args[args.index("--exploiter-iters")+1], "1000")
                self.assertEqual(args[args.index("--exploiter-win-target")+1], "0.75")
                self.assertIn("--exploiter-alternate-altitude-hunt", args)
                self.assertEqual(args[args.index("--exploiter-alt-hunt-coef")+1], "5.0")
                self.assertEqual(args[args.index("--keep-iterations")+1].split(",")[-1], "20000")

    def test_main_cannot_be_called_before_gates(self):
        search = self.bare()
        with self.assertRaises(m.IntegrityError):
            search.train_to("mlp512_d3", 0, 20000, final=True)
        self.assertFalse((self.folder / "final_20000").exists())

    def test_stop_hold_and_failure_prevent_spawn(self):
        for filename in ("STOP", "INTEGRITY_HOLD.json", "INTEGRITY_FAILURE.json"):
            marker = self.folder / filename
            marker.write_text("{}", encoding="utf-8")
            search = self.bare()
            with patch.object(m.subprocess, "Popen") as proc:
                with self.assertRaises((FloatingPointError, InterruptedError)):
                    search.command("must-not-run", ["-m", "cuda_fdm.train_gpu"])
                proc.assert_not_called()
            marker.unlink()

    def test_freeze_json_tuples_and_mutation(self):
        path = self.folder / "frozen.json"
        m.freeze_json(path, {"hidden": (512, 512)})
        before = path.read_bytes()
        m.freeze_json(path, {"hidden": (512, 512)})
        self.assertEqual(before, path.read_bytes())
        with self.assertRaises(m.IntegrityError):
            m.freeze_json(path, {"hidden": (512, 512, 512)})

    def test_metrics_finite_aux_grad_and_allowed_empty_episodes(self):
        for key in ("pl", "aux_actor_mse", "actor_grad_norm", "critic_grad_norm"):
            path = self.write_metrics(self.folder, **{key: "nan"})
            with self.assertRaises(m.IntegrityError):
                m.check_metrics(path)
        path = self.write_metrics(self.folder, eps="0", mean_ret="nan", mean_len="nan", win_rate="nan")
        self.assertEqual(len(m.check_metrics(path)), 2)

    def test_csv_recovery_archives_uncheckpointed_iterations(self):
        search = self.bare()
        path = self.write_metrics(self.folder, count=4)
        original = path.read_bytes()
        search.reconcile_metrics(self.folder, 2)
        self.assertEqual(len(m.check_metrics(path)), 2)
        backups = list(self.folder.glob("metrics_before_resume_*.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        with self.assertRaises(m.IntegrityError):
            search.reconcile_metrics(self.folder, 3)

    def test_candidate_limits_and_actual_pid(self):
        for name in ("gru512", "mlp512_d5", "mlp1280_d3", "mlp672_d3", "mlp832_d3"):
            with self.assertRaises(m.IntegrityError):
                m.model_spec(name)
        for width in (384, 512, 768, 1024):
            self.assertEqual(m.model_spec(f"mlp{width}_d3")["width"], width)
        self.assertTrue(m.pid_alive(os.getpid()))
        self.assertFalse(m.pid_alive(None))

    def test_constructor_never_resumes_old_cohort_and_uses_exclusive_lock(self):
        with patch.object(m, "ROOT", self.folder), patch.object(m, "sources", return_value={}):
            parent = self.folder / "runs/architecture_search"
            with self.assertRaises(ValueError):
                m.MLPSearch(parent / "experiment_v1")
            with self.assertRaises(ValueError):
                m.MLPSearch(parent / "mlp_depth_width_aux_v1")
            a = m.MLPSearch(parent / "new_a")
            try:
                with self.assertRaises(OSError):
                    m.MLPSearch(parent / "new_b")
            finally:
                a.close()
            b = m.MLPSearch(parent / "new_b")
            b.close()

    def test_constructor_checks_orphan_before_overwriting_status(self):
        with patch.object(m, "ROOT", self.folder), patch.object(m, "sources", return_value={}):
            folder = self.folder / "runs/architecture_search/new_a"
            folder.mkdir(parents=True)
            state = folder / "status.json"
            m.write_json(state, {"child_pid": 783427})
            original = state.read_bytes()
            with patch.object(m, "pid_alive", return_value=True), self.assertRaisesRegex(RuntimeError, "still alive"):
                m.MLPSearch(folder)
            self.assertEqual(original, state.read_bytes())

    def test_source_hash_no_rebaseline(self):
        search = self.bare()
        path = self.folder / "source.txt"
        path.write_text("before", encoding="utf-8")
        search.manifest = {"source_sha256": {str(path): m.sha(path)}}
        search.manifest_path = self.folder / "manifest.json"
        m.write_json(search.manifest_path, search.manifest)
        search.manifest_digest = m.sha(search.manifest_path)
        before = search.manifest_path.read_bytes()
        path.write_text("after", encoding="utf-8")
        with self.assertRaises(m.IntegrityError):
            m.MLPSearch.check_sources(search)
        self.assertEqual(search.manifest_path.read_bytes(), before)

    def test_checkpoint_aux_config_and_transition_budget(self):
        cfg = m.expected_cfg("mlp512_d3", 0, False)
        saved = dict(cfg=cfg, iteration=2, global_step=2*4096*64,
            training_protocol=m.PROTOCOL["training_protocol"], auxiliary_contract=AUX_CONTRACT, reward_contract=REWARD_CONTRACT,
            runtime={"trainer": {"opp_assign": torch.zeros(4096)}, "reward_mode": 0, "alt_hunt_coef": 5.0})
        with patch.object(torch, "load", return_value=saved):
            self.assertEqual(m.verify_checkpoint("unused", "mlp512_d3", 0), 2)
        for field, value in (("aux_coef", .2), ("gamma", .99), ("hidden", (512, 512)), ("save_runtime", False)):
            changed = copy.deepcopy(saved); changed["cfg"][field] = value
            with patch.object(torch, "load", return_value=changed), self.assertRaises(m.IntegrityError):
                m.verify_checkpoint("unused", "mlp512_d3", 0)
        changed = copy.deepcopy(saved); changed["training_protocol"] = "old"
        with patch.object(torch, "load", return_value=changed), self.assertRaises(m.IntegrityError):
            m.verify_checkpoint("unused", "mlp512_d3", 0)

    def test_main_resume_validates_scheduled_values_not_initial_values(self):
        cfg = m.expected_cfg("mlp768_d4", 0, True)
        cfg["ent_coef"] /= 3
        cfg["rollout_steps"] = 72
        saved = dict(cfg=cfg, iteration=2001, global_step=4096*(2000*64+72),
            training_protocol=m.PROTOCOL["training_protocol"], auxiliary_contract=AUX_CONTRACT, reward_contract=REWARD_CONTRACT,
            runtime={"trainer": {"opp_assign": torch.zeros(4096)}, "reward_mode": 0, "alt_hunt_coef": 5.0})
        with patch.object(torch, "load", return_value=saved):
            self.assertEqual(m.verify_checkpoint("unused", "mlp768_d4", 0, final=True), 2001)
        saved["global_step"] += 64*4096
        with patch.object(torch, "load", return_value=saved), self.assertRaises(m.IntegrityError):
            m.verify_checkpoint("unused", "mlp768_d4", 0, final=True)

    def test_eval_score_pairing_and_duration_gate(self):
        rows = [dict(score=.5, damage_diff=0., ic_pair=i % 2, role_swapped=i >= 2,
                     duration_sec=200., first_hit_sec=None, first_hit_b_sec=None) for i in range(4)]
        data = {"matches": {"pair": {"records": rows}}}
        m.validate_records(data, games=4)
        for field, value in (("score", .7), ("score", float("nan")), ("duration_sec", float("inf")),
                             ("ic_pair", 7), ("role_swapped", True), ("first_hit_sec", 201.)):
            changed = copy.deepcopy(data); changed["matches"]["pair"]["records"][0][field] = value
            with self.assertRaises(m.IntegrityError):
                m.validate_records(changed, games=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
