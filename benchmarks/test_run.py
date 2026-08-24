#!/usr/bin/env python3
"""Self-tests for the ETP/ETH comparison-matrix runner."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dprc_benchmark_run", ROOT / "benchmarks/run.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class BenchmarkRunnerTest(unittest.TestCase):
    def test_matrix_preserves_three_modes_and_batch_48(self) -> None:
        matrix = RUNNER.load_matrix(ROOT / "benchmarks/matrix.json")
        self.assertEqual(matrix["axes"]["mode"], list(RUNNER.MODES))
        self.assertEqual(matrix["axes"]["batch_size"], [1, 2, 4, 8, 16, 32, 48])

    def test_synchronized_sample_uses_slowest_partition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-benchmark-logs-") as temporary:
            root = Path(temporary)
            first = root / "log.lammps.0"
            second = root / "log.lammps.1"
            first.write_text(
                "Loop time of 2.0 on 1 procs for 5 steps with 8938 atoms\n"
                "Loop time of 4.0 on 1 procs for 10 steps with 8938 atoms\n"
                "Loop time of 5.0 on 1 procs for 10 steps with 8938 atoms\n",
                encoding="utf-8",
            )
            second.write_text(
                "Loop time of 3.0 on 1 procs for 5 steps with 8938 atoms\n"
                "Loop time of 8.0 on 1 procs for 10 steps with 8938 atoms\n"
                "Loop time of 4.0 on 1 procs for 10 steps with 8938 atoms\n",
                encoding="utf-8",
            )
            result = RUNNER.collect_samples(
                [first, second],
                batch_size=2,
                warmup_steps=5,
                sample_steps=10,
                repetitions=2,
                timestep_fs=1.0,
            )
            self.assertEqual(result["warmup"]["synchronized_loop_seconds"], 3.0)
            self.assertEqual(
                [sample["synchronized_loop_seconds"] for sample in result["samples"]],
                [8.0, 5.0],
            )
            self.assertEqual(
                [sample["aggregate_window_steps_per_second"] for sample in result["samples"]],
                [2.5, 4.0],
            )

    def test_dpa4c_runtime_pins_deepmd_thread_pools(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-runtime-environment-") as temporary:
            root = Path(temporary)
            xtbloom = root / "libxtbloom.so"
            xtbloom.write_bytes(b"diagnostic")
            arguments = SimpleNamespace(
                xtbloom_library=xtbloom,
                library_dir=[],
                cuda_visible_devices="0",
                classical_backend="batched-dprc",
            )
            environment, selected = RUNNER.runtime_environment(
                "qmmm-dpa4c", arguments
            )
            for name in (
                "DP_INTRA_OP_PARALLELISM_THREADS",
                "DP_INTER_OP_PARALLELISM_THREADS",
            ):
                self.assertEqual(environment[name], "1")
                self.assertEqual(selected[name], "1")

            qmmm_environment, qmmm_selected = RUNNER.runtime_environment(
                "qmmm", arguments
            )
            self.assertNotIn("DP_INTRA_OP_PARALLELISM_THREADS", qmmm_selected)
            self.assertNotIn("DP_INTER_OP_PARALLELISM_THREADS", qmmm_selected)

    def test_renderer_distinguishes_all_execution_modes(self) -> None:
        manifest = RUNNER.WORKLOAD.load_manifest(
            ROOT / "workloads/etpeth/manifest.json"
        )
        windows = RUNNER.WORKLOAD.windows_from_manifest(manifest)
        with tempfile.TemporaryDirectory(prefix="dprc-benchmark-render-") as temporary:
            root = Path(temporary)
            tutorial = root / "tutorial"
            (tutorial / "lammps").mkdir(parents=True)
            (tutorial / "lammps/forcefield_qmmm_hybrid.inc").write_text(
                "pair_coeff 1 1 lj/cut 0.1 3.0\n", encoding="utf-8"
            )
            item = RUNNER.WORKLOAD.RunWindow(
                windows[0], root / "start.data", root / "out", root, 1234
            )
            common = {
                "manifest": manifest,
                "tutorial": tutorial,
                "run_windows": [item],
                "steps": 2,
                "trajectory_frequency": 0,
                "run_commands": ["timer full sync", "run 1", "run 1 pre no"],
            }
            classical = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so", mode="classical", **common
            )
            classical_reference = RUNNER.WORKLOAD.render_lammps_input(
                plugin=None,
                mode="classical",
                classical_backend="upstream-gpu",
                **common,
            )
            qmmm = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so", mode="qmmm", **common
            )
            primary_model = root / "primary.pt2"
            dpa4c = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="qmmm-dpa4c",
                deepmd_models=[primary_model],
                **common,
            )
            self.assertIn("lj/cut/dprc/batch", classical)
            self.assertIn("pppm/tip4p/dprc/batch", classical)
            self.assertIn("fix classical all dprc/classical/batch", classical)
            self.assertNotIn("fix qmmm", classical)
            self.assertIn(
                "pair_style lj/cut/tip4p/long/gpu", classical_reference
            )
            self.assertIn(
                "kspace_style pppm/tip4p 1.0e-6", classical_reference
            )
            self.assertIn("pppm/tip4p/dprc/batch", qmmm)
            self.assertIn("fix qmmm qm qmmm/xtb/dprc", qmmm)
            for device_input in (classical, qmmm, dpa4c):
                self.assertIn("atom_style full/kk", device_input)
                self.assertIn("newton on", device_input)
                self.assertIn("run_style verlet/kk", device_input)
                self.assertIn("pair_style hybrid/overlay/kk", device_input)
                self.assertIn("bond_style harmonic/kk", device_input)
                self.assertIn("angle_style harmonic/kk", device_input)
                self.assertIn("fix water_shake water shake/kk", device_input)
                self.assertIn("fix integrate all nve/kk", device_input)
                self.assertIn("fix thermostat all langevin/kk", device_input)
                self.assertIn("fix remove_com all momentum/kk", device_input)
                self.assertIn("fix restraints all colvars/kk", device_input)
            self.assertIn("atom_style full", classical_reference)
            self.assertNotIn("atom_style full/kk", classical_reference)
            self.assertLess(dpa4c.index("newton on"), dpa4c.index("read_data"))
            self.assertLess(
                dpa4c.index("read_data ${start_data}"),
                dpa4c.index("run_style verlet/kk"),
            )
            self.assertIn("center_group qm", dpa4c)
            self.assertIn("P O O C H OW HW", dpa4c)
            self.assertIn(
                f"dprc/deepmd/batch/kk {primary_model} partition_batch yes",
                dpa4c,
            )
            self.assertIn("pair_coeff * * dprc/deepmd/batch/kk", dpa4c)
            self.assertNotIn("model_deviation", dpa4c)
            self.assertEqual(dpa4c.count("plugin load"), 1)
            self.assertEqual(dpa4c.count("run 1"), 2)

            relative = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="qmmm",
                execution_directory=root / "coordinate",
                **common,
            )
            self.assertIn("../out/m3p1", relative)
            self.assertNotIn(str(root.resolve()), relative)

    def test_dpa4c_schedule_rejects_ambiguous_model_counts(self) -> None:
        manifest = RUNNER.WORKLOAD.load_manifest(
            ROOT / "workloads/etpeth/manifest.json"
        )
        windows = RUNNER.WORKLOAD.windows_from_manifest(manifest)
        with tempfile.TemporaryDirectory(prefix="dprc-schedule-render-") as temporary:
            root = Path(temporary)
            tutorial = root / "tutorial"
            (tutorial / "lammps").mkdir(parents=True)
            (tutorial / "lammps/forcefield_qmmm_hybrid.inc").write_text(
                "pair_coeff 1 1 lj/cut 0.1 3.0\n", encoding="utf-8"
            )
            item = RUNNER.WORKLOAD.RunWindow(
                windows[0], root / "start.data", root / "out", root, 1234
            )
            common = {
                "manifest": manifest,
                "tutorial": tutorial,
                "plugin": root / "dprcplugin.so",
                "run_windows": [item],
                "steps": 1,
                "trajectory_frequency": 0,
                "mode": "qmmm-dpa4c",
            }
            models = [root / f"model-{index}.pt2" for index in range(2)]
            with self.assertRaisesRegex(ValueError, "requires exactly one model"):
                RUNNER.WORKLOAD.render_lammps_input(
                    deepmd_models=models,
                    model_deviation_frequency=0,
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "model deviation to be disabled"):
                RUNNER.WORKLOAD.render_lammps_input(
                    deepmd_models=models[:1],
                    model_deviation_frequency=100,
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "must be nonnegative"):
                RUNNER.WORKLOAD.render_lammps_input(
                    deepmd_models=models[:1],
                    model_deviation_frequency=-1,
                    **common,
                )

    def test_dpa4c_evidence_identity_records_in_plugin_batch_schedule(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-schedule-identity-") as temporary:
            root = Path(temporary)
            paths = {
                name: root / name
                for name in ("lmp", "dprcplugin.so", "libxtbloom.so")
            }
            models = [root / "model.pt2"]
            for path in (*paths.values(), *models):
                path.write_bytes(path.name.encode("utf-8"))
            arguments = SimpleNamespace(
                lammps=paths["lmp"],
                plugin=paths["dprcplugin.so"],
                xtbloom_library=paths["libxtbloom.so"],
                deepmd_model=models,
                model_deviation_frequency=0,
                dpa4c_models_qualified=True,
                classical_backend="batched-dprc",
            )
            identity = RUNNER.coordinate_identity("qmmm-dpa4c", 8, arguments)
            self.assertEqual(
                identity["dprc_schedule"],
                {
                    "primary_model_index": 0,
                    "model_count": 1,
                    "model_deviation_frequency_steps": 0,
                    "model_deviation_enabled": False,
                    "execution_backend": "dprcplugin-deepmd-c-api-batch",
                    "models_qualified_as_xtb_dprc": True,
                },
            )

    def test_dpa4c_availability_follows_the_selected_schedule(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-schedule-availability-") as temporary:
            root = Path(temporary)
            required = [
                root / name
                for name in (
                    "lmp",
                    "mpiexec",
                    "dprcplugin.so",
                    "libxtbloom.so",
                )
            ]
            models = [root / f"model-{index}.pt2" for index in range(2)]
            for path in (*required, *models):
                path.write_bytes(path.name.encode("utf-8"))
            arguments = SimpleNamespace(
                lammps=required[0],
                mpiexec=required[1],
                plugin=required[2],
                xtbloom_library=required[3],
                deepmd_model=models[:1],
                model_deviation_frequency=0,
                dpa4c_models_qualified=True,
                allow_unqualified_dpa4c_models=False,
                classical_backend="batched-dprc",
            )
            self.assertEqual(
                RUNNER.availability_reasons("qmmm-dpa4c", 8, arguments), []
            )
            arguments.deepmd_model = models
            self.assertTrue(
                any(
                    "requires exactly one model" in reason
                    for reason in RUNNER.availability_reasons(
                        "qmmm-dpa4c", 8, arguments
                    )
                )
            )
            arguments.deepmd_model = models[:1]
            arguments.model_deviation_frequency = 100
            self.assertTrue(
                any(
                    "model deviation" in reason
                    for reason in RUNNER.availability_reasons(
                        "qmmm-dpa4c", 8, arguments
                    )
                )
            )
            arguments.model_deviation_frequency = 0

            arguments.dpa4c_models_qualified = False
            self.assertTrue(
                any(
                    "not qualified" in reason
                    for reason in RUNNER.availability_reasons(
                        "qmmm-dpa4c", 8, arguments
                    )
                )
            )
            arguments.allow_unqualified_dpa4c_models = True
            self.assertEqual(
                RUNNER.availability_reasons("qmmm-dpa4c", 8, arguments), []
            )

    def test_unqualified_dpa4c_override_is_explicit_and_mutually_exclusive(self) -> None:
        parser = RUNNER.build_parser()
        common = [
            "--tutorial",
            "/tmp/tutorial",
            "--output",
            "/tmp/output",
            "--lammps",
            "/tmp/lmp",
        ]
        diagnostic = parser.parse_args(
            [*common, "--allow-unqualified-dpa4c-models"]
        )
        self.assertTrue(diagnostic.allow_unqualified_dpa4c_models)
        self.assertFalse(diagnostic.dpa4c_models_qualified)

        contradictory = parser.parse_args(
            [
                *common,
                "--dpa4c-models-qualified",
                "--allow-unqualified-dpa4c-models",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            RUNNER.validate_dpa4c_policy(contradictory)

    def test_correctness_evidence_must_cover_every_gate(self) -> None:
        matrix = RUNNER.load_matrix(ROOT / "benchmarks/matrix.json")
        with tempfile.TemporaryDirectory(prefix="dprc-correctness-") as temporary:
            path = Path(temporary) / "qmmm.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "passed",
                        "mode": "qmmm",
                        "batch_sizes": [1, 2],
                        "checks": {
                            name: True for name in matrix["required_correctness"]
                        },
                    }
                ),
                encoding="utf-8",
            )
            record, reasons = RUNNER.correctness_record(
                "qmmm", 2, matrix["required_correctness"], {"qmmm": path}
            )
            self.assertEqual(record["status"], "passed")
            self.assertEqual(reasons, [])
            _, reasons = RUNNER.correctness_record(
                "qmmm", 4, matrix["required_correctness"], {"qmmm": path}
            )
            self.assertIn("does not cover this batch size", reasons[0])


if __name__ == "__main__":
    unittest.main()
