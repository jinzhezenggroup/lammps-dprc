#!/usr/bin/env python3
"""Self-tests for the ETP/ETH comparison-matrix runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dprc_benchmark_run", ROOT / "benchmarks/run.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class BenchmarkRunnerTest(unittest.TestCase):
    @staticmethod
    def write_topology_fixture(
        path: Path, manifest: dict[str, object], mode: str
    ) -> None:
        """Write a minimal data-file header matching one topology contract."""
        contract = RUNNER.WORKLOAD.topology_contract(manifest, mode)
        path.write_text(
            "fixture\n\n"
            f"{contract['atoms']} atoms\n"
            f"{contract['bonds']} bonds\n"
            f"{contract['angles']} angles\n"
            f"{contract['dihedrals']} dihedrals\n"
            "0 impropers\n\nMasses\n",
            encoding="utf-8",
        )

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
            (tutorial / "lammps/forcefield_mm_hybrid.inc").write_text(
                "pair_coeff 1 1 lj/cut 0.1 3.0\n", encoding="utf-8"
            )
            classical_start = root / "classical-start.data"
            qmmm_start = root / "qmmm-start.data"
            self.write_topology_fixture(classical_start, manifest, "classical")
            self.write_topology_fixture(qmmm_start, manifest, "qmmm")
            classical_item = RUNNER.WORKLOAD.RunWindow(
                windows[0], classical_start, root / "classical-out", root, 1234
            )
            qmmm_item = RUNNER.WORKLOAD.RunWindow(
                windows[0], qmmm_start, root / "out", root, 1234
            )
            common = {
                "manifest": manifest,
                "tutorial": tutorial,
                "steps": 2,
                "trajectory_frequency": 0,
                "run_commands": ["timer full sync", "run 1", "run 1 pre no"],
            }
            classical = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="classical",
                run_windows=[classical_item],
                lammps_execution_backend="host",
                **common,
            )
            classical_reference = RUNNER.WORKLOAD.render_lammps_input(
                plugin=None,
                mode="classical",
                classical_backend="upstream-gpu",
                run_windows=[classical_item],
                lammps_execution_backend="host",
                **common,
            )
            qmmm = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="qmmm",
                run_windows=[qmmm_item],
                lammps_execution_backend="host",
                **common,
            )
            primary_model = root / "primary.pt2"
            dpa4c = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="qmmm-dpa4c",
                deepmd_models=[primary_model],
                run_windows=[qmmm_item],
                lammps_execution_backend="host",
                **common,
            )
            kokkos_dpa4c = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="qmmm-dpa4c",
                deepmd_models=[primary_model],
                lammps_execution_backend="kokkos",
                run_windows=[qmmm_item],
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
            for host_input in (classical, qmmm, dpa4c):
                self.assertIn("atom_style full\n", host_input)
                self.assertIn("pair_style hybrid/overlay ", host_input)
                self.assertNotIn("/kk", host_input)
            self.assertIn("atom_style full/kk", kokkos_dpa4c)
            self.assertIn("newton on", kokkos_dpa4c)
            self.assertIn("run_style verlet/kk", kokkos_dpa4c)
            self.assertIn("pair_style hybrid/overlay/kk", kokkos_dpa4c)
            self.assertIn("bond_style harmonic/kk", kokkos_dpa4c)
            self.assertIn("angle_style harmonic/kk", kokkos_dpa4c)
            self.assertIn("fix water_shake water shake/kk", kokkos_dpa4c)
            self.assertIn("fix integrate all nve/kk", kokkos_dpa4c)
            self.assertIn("fix thermostat all langevin/kk", kokkos_dpa4c)
            self.assertIn("fix remove_com all momentum/kk", kokkos_dpa4c)
            self.assertIn("fix restraints all colvars/kk", kokkos_dpa4c)
            self.assertIn("atom_style full", classical_reference)
            self.assertNotIn("atom_style full/kk", classical_reference)
            self.assertLess(
                kokkos_dpa4c.index("newton on"),
                kokkos_dpa4c.index("read_data"),
            )
            self.assertLess(
                kokkos_dpa4c.index("read_data ${start_data}"),
                kokkos_dpa4c.index("run_style verlet/kk"),
            )
            self.assertIn("center_group qm", dpa4c)
            self.assertIn("P O O C H OW HW", dpa4c)
            self.assertIn(
                f"dprc/deepmd/batch {primary_model} partition_batch yes",
                dpa4c,
            )
            self.assertIn("pair_coeff * * dprc/deepmd/batch", dpa4c)
            self.assertIn(
                f"dprc/deepmd/batch/kk {primary_model} partition_batch yes",
                kokkos_dpa4c,
            )
            self.assertNotIn("model_deviation", dpa4c)
            self.assertEqual(dpa4c.count("plugin load"), 1)
            self.assertEqual(dpa4c.count("run 1"), 2)

            relative = RUNNER.WORKLOAD.render_lammps_input(
                plugin=root / "dprcplugin.so",
                mode="qmmm",
                execution_directory=root / "coordinate",
                run_windows=[qmmm_item],
                lammps_execution_backend="host",
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
            deepmd_manifest = root / "deepmd-artifact-manifest.json"
            deepmd_manifest.write_text("{}\n", encoding="utf-8")
            for path in (*paths.values(), *models):
                path.write_bytes(path.name.encode("utf-8"))
            arguments = SimpleNamespace(
                lammps=paths["lmp"],
                plugin=paths["dprcplugin.so"],
                xtbloom_library=paths["libxtbloom.so"],
                deepmd_model=models,
                deepmd_artifact_manifest=deepmd_manifest,
                expected_deepmd_revision="a" * 40,
                model_deviation_frequency=0,
                dpa4c_models_qualified=True,
                classical_backend="batched-dprc",
                lammps_execution_backend="host",
            )
            identity = RUNNER.coordinate_identity("qmmm-dpa4c", 8, arguments)
            self.assertEqual(identity["lammps_execution_backend"], "host")
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

            arguments.lammps_execution_backend = "kokkos"
            self.assertEqual(
                RUNNER.coordinate_execution_backend("qmmm", arguments),
                "kokkos",
            )
            arguments.classical_backend = "upstream-gpu"
            self.assertEqual(
                RUNNER.coordinate_execution_backend("classical", arguments),
                "host",
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
            deepmd_manifest = root / "deepmd-artifact-manifest.json"
            deepmd_manifest.write_text("{}\n", encoding="utf-8")
            deepmd_source = root / "deepmd-source"
            deepmd_source.mkdir()
            deepmd_include = root / "deepmd-include/deepmd"
            deepmd_include.mkdir(parents=True)
            (deepmd_include / "c_api.h").write_text("fixture\n", encoding="utf-8")
            for path in (*required, *models):
                path.write_bytes(path.name.encode("utf-8"))
            arguments = SimpleNamespace(
                lammps=required[0],
                mpiexec=required[1],
                plugin=required[2],
                xtbloom_library=required[3],
                deepmd_model=models[:1],
                deepmd_artifact_manifest=deepmd_manifest,
                deepmd_source=deepmd_source,
                deepmd_include_dir=deepmd_include.parent,
                expected_deepmd_revision="a" * 40,
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

    def test_deepmd_manifest_and_final_source_can_be_evidence_eligible(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-eligibility-") as temporary:
            root = Path(temporary)
            manifest = root / "deepmd-artifact-manifest.json"
            revision = "a" * 40
            library_sha256 = "b" * 64
            header_sha256 = "c" * 64
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_revision": revision,
                        "source_clean": True,
                        "source_state_sha256": "d" * 64,
                        "c_api_version": 31,
                        "source_c_api_header_sha256": header_sha256,
                        "installed_c_api_header_sha256": header_sha256,
                        "c_api_library_sha256": library_sha256,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            arguments = SimpleNamespace(
                deepmd_artifact_manifest=manifest,
                deepmd_source=root / "source",
                deepmd_include_dir=root / "include",
                expected_deepmd_revision=revision,
            )
            with mock.patch.object(
                RUNNER.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout=manifest.read_text(), stderr=""
                ),
            ):
                deepmd = RUNNER.deepmd_dependency_record(
                    arguments,
                    {
                        "soname": "libdeepmd_c.so",
                        "sha256": library_sha256,
                        "resolved_path": str(root / "libdeepmd_c.so"),
                    },
                )
            self.assertTrue(deepmd["publication_qualified"])
            reasons = RUNNER.publication_eligibility_reasons(
                mode="qmmm-dpa4c",
                source={"qualification": "final"},
                project={
                    "dirty": False,
                    "dependencies": {
                        "lammps": {
                            "required": True,
                            "publication_qualified": True,
                        },
                        "xtbloom": {
                            "required": True,
                            "publication_qualified": True,
                        },
                    },
                },
                deepmd_dependency=deepmd,
                dpa4c_models_qualified=True,
                dangerous_builds={"window": 0},
                correctness_reasons=[],
            )
            self.assertEqual(reasons, [])

            wrong_pin = dict(deepmd)
            wrong_pin["publication_qualified"] = False
            wrong_pin["qualification_reasons"] = ["source revision differs"]
            reasons = RUNNER.publication_eligibility_reasons(
                mode="qmmm-dpa4c",
                source={"qualification": "final"},
                project={
                    "dirty": False,
                    "dependencies": {
                        "xtbloom": {
                            "required": True,
                            "publication_qualified": False,
                            "qualification_reasons": ["revision differs"],
                        }
                    },
                },
                deepmd_dependency=wrong_pin,
                dpa4c_models_qualified=True,
                dangerous_builds={"window": 0},
                correctness_reasons=[],
            )
            self.assertTrue(any("xtbloom" in reason for reason in reasons))
            self.assertTrue(any("DeePMD" in reason for reason in reasons))

    def test_partitioned_availability_resolves_mpi_launcher_from_path(self) -> None:
        """A command-name launcher is valid when discovered through PATH."""
        with tempfile.TemporaryDirectory(prefix="dprc-path-mpiexec-") as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            launcher = bin_dir / "mpiexec"
            launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            launcher.chmod(0o755)
            lammps = root / "lmp"
            lammps.write_bytes(b"lammps")
            arguments = SimpleNamespace(
                lammps=lammps,
                mpiexec=Path("mpiexec"),
                plugin=None,
                xtbloom_library=None,
                deepmd_model=[],
                model_deviation_frequency=0,
                dpa4c_models_qualified=False,
                allow_unqualified_dpa4c_models=False,
                classical_backend="upstream-gpu",
            )
            with mock.patch.dict(
                os.environ,
                {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
            ):
                self.assertEqual(
                    RUNNER.availability_reasons("classical", 2, arguments), []
                )

            arguments.mpiexec = Path("missing-mpiexec")
            self.assertIn(
                "MPI launcher is unavailable",
                RUNNER.availability_reasons("classical", 2, arguments),
            )

    def test_serial_coordinate_does_not_require_mpi_launcher(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-serial-launcher-") as temporary:
            root = Path(temporary)
            lammps = root / "lmp"
            lammps.write_bytes(b"lammps")
            arguments = SimpleNamespace(
                lammps=lammps,
                mpiexec=root / "missing-mpiexec",
                plugin=None,
                xtbloom_library=None,
                deepmd_model=[],
                model_deviation_frequency=0,
                dpa4c_models_qualified=False,
                allow_unqualified_dpa4c_models=False,
                classical_backend="upstream-gpu",
            )
            self.assertEqual(
                RUNNER.availability_reasons("classical", 1, arguments), []
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
        self.assertEqual(diagnostic.lammps_execution_backend, "host")
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
            runtime_identity = {
                "lammps": {"sha256": "lammps-sha256"},
                "lammps_execution_backend": "host",
                "plugin": {"sha256": "plugin-sha256"},
                "xtbloom": {"sha256": "xtbloom-sha256"},
            }
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "passed",
                        "mode": "qmmm",
                        "batch_sizes": [1, 2],
                        "runtime_identity": runtime_identity,
                        "checks": {
                            name: True for name in matrix["required_correctness"]
                        },
                    }
                ),
                encoding="utf-8",
            )
            record, reasons = RUNNER.correctness_record(
                "qmmm",
                2,
                matrix["required_correctness"],
                {"qmmm": path},
                runtime_identity,
            )
            self.assertEqual(record["status"], "passed")
            self.assertEqual(reasons, [])
            _, reasons = RUNNER.correctness_record(
                "qmmm",
                4,
                matrix["required_correctness"],
                {"qmmm": path},
                runtime_identity,
            )
            self.assertIn("does not cover this batch size", reasons[0])

            kokkos_identity = {
                **runtime_identity,
                "lammps_execution_backend": "kokkos",
            }
            record, reasons = RUNNER.correctness_record(
                "qmmm",
                2,
                matrix["required_correctness"],
                {"qmmm": path},
                kokkos_identity,
            )
            self.assertEqual(record["status"], "unqualified")
            self.assertIn(
                "correctness evidence runtime identity does not match", reasons
            )

    def test_correctness_runtime_identity_is_relocatable_and_batch_independent(
        self,
    ) -> None:
        inputs = {
            "lammps": {"path": "/private/a/lmp", "bytes": 4, "sha256": "abc"},
            "lammps_execution_backend": "host",
            "plugin": {
                "path": "/private/a/dprcplugin.so",
                "bytes": 8,
                "sha256": "def",
            },
            "batch_size": 32,
        }
        self.assertEqual(
            RUNNER.correctness_runtime_identity(inputs),
            {
                "lammps": {"sha256": "abc"},
                "lammps_execution_backend": "host",
                "plugin": {"sha256": "def"},
            },
        )

    def test_scientific_contract_binds_manifest_renderer_and_mpi_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-science-contract-") as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            matrix_path = root / "matrix.json"
            mpiexec = root / "mpiexec"
            manifest_path.write_text('{"system":{"cutoff_angstrom":9.0}}\n')
            matrix_path.write_text('{"schema_version":2}\n')
            mpiexec.write_text("reviewed launcher\n")
            arguments = argparse.Namespace(
                manifest=manifest_path,
                matrix=matrix_path,
                mpiexec=mpiexec,
                mpi_arg=["--bind-to", "core"],
            )
            source = {
                "revision": "tutorial-revision",
                "artifacts": [
                    {
                        "path": "/private/tutorial/input.data",
                        "relative_path": "lammps/input.data",
                        "sha256": "input-sha256",
                        "classification": "runtime input",
                    }
                ],
            }
            contract = RUNNER.scientific_execution_contract(
                arguments=arguments,
                manifest={"system": {"cutoff_angstrom": 9.0}},
                matrix={"schema_version": 2},
                source=source,
                execution_backend="host",
                warmup_steps=10,
                sample_steps=20,
                repetitions=3,
            )
            identity = RUNNER.correctness_runtime_identity(
                {"scientific_execution_contract": contract, "batch_size": 32}
            )
            self.assertEqual(
                identity["scientific_execution_contract"]["scientific_parameters"],
                {"system": {"cutoff_angstrom": 9.0}},
            )
            self.assertEqual(
                identity["scientific_execution_contract"]["launch_policy"],
                {
                    "mpi_launcher": {"sha256": RUNNER.WORKLOAD.sha256(mpiexec)},
                    "mpi_arguments": ["--bind-to", "core"],
                    "worlds": "one-per-selected-umbrella-window",
                    "ranks_per_window": 1,
                    "lammps_arguments": [],
                },
            )
            self.assertNotIn(
                "path",
                identity["scientific_execution_contract"]["tutorial_source"][
                    "artifacts"
                ][0],
            )

            changed = json.loads(json.dumps(identity))
            changed["scientific_execution_contract"]["scientific_parameters"][
                "system"
            ]["cutoff_angstrom"] = 8.0
            self.assertNotEqual(identity, changed)

    def test_correctness_evidence_rejects_deepmd_library_mismatch(self) -> None:
        matrix = RUNNER.load_matrix(ROOT / "benchmarks/matrix.json")
        qualified_identity = {
            "lammps": {"sha256": "lammps-sha256"},
            "lammps_execution_backend": "host",
            "plugin": {"sha256": "plugin-sha256"},
            "xtbloom": {"sha256": "xtbloom-sha256"},
            "models": [{"sha256": "model-sha256"}],
            "loaded_deepmd_c": {
                "soname": "libdeepmd_c.so.3",
                "sha256": "qualified-library-sha256",
            },
        }
        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-evidence-") as temporary:
            path = Path(temporary) / "qmmm-dpa4c.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "passed",
                        "mode": "qmmm-dpa4c",
                        "batch_sizes": [32],
                        "runtime_identity": qualified_identity,
                        "checks": {
                            name: True for name in matrix["required_correctness"]
                        },
                    }
                ),
                encoding="utf-8",
            )
            timing_identity = {
                **qualified_identity,
                "loaded_deepmd_c": {
                    "soname": "libdeepmd_c.so.3",
                    "sha256": "different-library-sha256",
                },
            }
            record, reasons = RUNNER.correctness_record(
                "qmmm-dpa4c",
                32,
                matrix["required_correctness"],
                {"qmmm-dpa4c": path},
                timing_identity,
            )
            self.assertEqual(record["status"], "unqualified")
            self.assertIn(
                "correctness evidence runtime identity does not match", reasons
            )


if __name__ == "__main__":
    unittest.main()
