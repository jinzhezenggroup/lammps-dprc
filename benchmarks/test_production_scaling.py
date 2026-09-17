#!/usr/bin/env python3
"""CPU-only contract tests for production-state integer batch scaling."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("production_scaling", Path(__file__).with_name("production_scaling.py"))
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class ScalingTests(unittest.TestCase):
    def test_nested_windows_keep_all_identities(self):
        original = [S.W.Window(i, i - 31) for i in range(48)]
        order = S.nested_windows(original)
        self.assertEqual(len(set(order)), 48)
        self.assertEqual(set(order), set(original))
        self.assertEqual([w.center_tenths for w in order[:3]], [0, -31, 16])
        self.assertEqual(order, S.nested_windows(original))

    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_text("original")
            artifact = S.identity(path)
            self.assertEqual(S.checked(artifact), path)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Changed"):
                S.checked(artifact)

    def test_parity_checks_forces_charges_and_energy(self):
        row = dict(Step=0., Temp=298., TotEng=-20000., f_qmmm=-100.)
        limits = dict(energy_atol=1e-8, energy_rtol=1e-12, reduction_atol=2e-5, force_atol=1e-5, charge_atol=1e-10)
        artifact = dict(log={}, trajectory={})
        with mock.patch.object(S, "checked"), mock.patch.object(S.N, "thermo_rows", return_value=[row]), mock.patch.object(S, "first_frame", return_value=[[1., -.4, .1, .2, .3]]):
            self.assertTrue(S.parity(artifact, artifact, limits)["passed"])
        with mock.patch.object(S, "checked"), mock.patch.object(S.N, "thermo_rows", return_value=[row]), mock.patch.object(S, "first_frame", side_effect=[[[1., -.4, .1, .2, .3]], [[1., -.4, .1001, .2, .3]]]):
            self.assertFalse(S.parity(artifact, artifact, limits)["passed"])
        with mock.patch.object(S, "checked"), mock.patch.object(S.N, "thermo_rows", side_effect=[[row], [{**row, "f_qmmm": -99.999}]]), mock.patch.object(S, "first_frame", return_value=[[1., -.4, .1, .2, .3]]):
            self.assertFalse(S.parity(artifact, artifact, limits)["passed"])

    def test_summary_keeps_failures_and_missing_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = dict(method="qmmm-dpa4c", coordinates={str(b): dict(status="not-started") for b in range(1, 49)})
            summary["coordinates"]["17"] = dict(status="failed", error="parity")
            S.write_summary(root, summary)
            with (root / "summary.csv").open() as handle:
                rows = list(S.csv.DictReader(handle))
            self.assertEqual(len(rows), 48)
            self.assertEqual(rows[16]["status"], "failed")
            self.assertTrue(all(row["publication_qualified"] == "False" for row in rows))
            self.assertTrue(all(row["batch_size"] == row["xtb_batch_size"] == row["deepmd_frames_per_call"] for row in rows))

    def test_refuses_unscheduled_gpu_work(self):
        with mock.patch.dict(S.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Slurm"):
                S.run(None)

    def test_full_deepmd_batch_rejects_chunked_plans_and_caps(self):
        plan = dict(layout=dict(deepmd_batch_policy="full", deepmd_max_frames_per_call=None))
        with mock.patch.dict(S.os.environ, {}, clear=True):
            S.require_full_deepmd_batch(plan, "qmmm-dpa4c")
            with self.assertRaisesRegex(ValueError, "chunked plans"):
                S.require_full_deepmd_batch(dict(layout=dict(deepmd_max_frames_per_call=2)), "qmmm-dpa4c")
        for cap in ("", "0", "2", "48"):
            with self.subTest(cap=cap), mock.patch.dict(S.os.environ, {"DPRC_DEEPMD_MAX_FRAMES_PER_CALL": cap}):
                with self.assertRaisesRegex(ValueError, "must equal B"):
                    S.require_full_deepmd_batch(plan, "qmmm-dpa4c")
                S.require_full_deepmd_batch(plan, "qmmm")

    def test_nonfirst_preflight_failure_blocks_all_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = dict(status="running", references={}, coordinates={str(b): dict(status="not-started") for b in range(1, 49)})
            S.check_preflight(root, summary, "first", dict(passed=True))
            with self.assertRaisesRegex(S.FullBatchParityError, "second"):
                S.check_preflight(root, summary, "second", dict(passed=False))
            saved = S.read(root / "summary.json")
            self.assertEqual(saved["status"], "blocked-full-batch-parity")
            self.assertEqual(len(saved["coordinates"]), 48)
            self.assertTrue(all(r["status"] == "blocked" for r in saved["coordinates"].values()))
            self.assertNotIn("median_steps_per_s_per_gpu", saved["coordinates"]["48"])

    def test_maps_abstract_smt_ids_and_disjoint_allocations(self):
        topology = "0,0,0\n1,1,0\n2,2,0\n3,3,0\n4,0,0\n5,1,0\n6,2,0\n7,3,0\n"
        first = S.physical_allocation("Nodes=test CPU_IDs=0-3 Mem=100", topology, "test", required=2)
        second = S.physical_allocation("Nodes=test CPU_IDs=4-7 Mem=100", topology, "test", required=2)
        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [2, 3])
        self.assertFalse(set(first) & set(second))
        with self.assertRaisesRegex(ValueError, "reserved physical cores"):
            S.physical_allocation("Nodes=test CPU_IDs=0-1", topology, "test", required=2)
        with self.assertRaisesRegex(ValueError, "local Slurm node"):
            S.physical_allocation("Nodes=other CPU_IDs=0-3", topology, "test", required=2)


if __name__ == "__main__":
    unittest.main()
