#!/usr/bin/env python3
"""Focused tests for the external ETP/ETH workload runner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "etpeth_workload", ROOT / "tools/etpeth_workload.py"
)
assert SPEC is not None and SPEC.loader is not None
WORKLOAD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKLOAD
SPEC.loader.exec_module(WORKLOAD)


class ETPETHWorkloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = WORKLOAD.load_manifest(ROOT / "workloads/etpeth/manifest.json")

    def write_topology_fixture(self, path: Path, mode: str) -> None:
        """Write the minimal LAMMPS header needed by the topology guard."""
        contract = WORKLOAD.topology_contract(self.manifest, mode)
        path.write_text(
            "fixture\n\n"
            f"{contract['atoms']} atoms\n"
            f"{contract['bonds']} bonds\n"
            f"{contract['angles']} angles\n"
            f"{contract['dihedrals']} dihedrals\n"
            "0 impropers\n\nMasses\n",
            encoding="utf-8",
        )

    def test_exact_grid_and_stable_tags(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        self.assertEqual(len(windows), 48)
        self.assertEqual((windows[0].tag, windows[0].center), ("m3p1", -3.1))
        self.assertEqual((windows[16].tag, windows[16].center), ("m1p5", -1.5))
        self.assertEqual((windows[-1].tag, windows[-1].center), ("p1p6", 1.6))

    def test_representative_nve_windows_are_fixed_before_results(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        selected = WORKLOAD.representative_nve_windows(windows)
        self.assertEqual(
            [window.center_tenths for window in selected],
            [-31, -8, 16],
        )

    def test_deepmd_c_loader_identity_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-loader-") as temporary:
            root = Path(temporary)
            plugin = root / "dprcplugin.so"
            library = root / "libdeepmd_c.so.3"
            plugin.write_bytes(b"plugin")
            library.write_bytes(b"public C API")
            output = f"libdeepmd_c.so.3 => {library} (0x1234)"
            with mock.patch.object(WORKLOAD, "command_output", return_value=output):
                identity = WORKLOAD.verify_loaded_deepmd_c(plugin, {})
            self.assertEqual(
                identity,
                {
                    "soname": "libdeepmd_c.so.3",
                    "resolved_path": str(library.resolve()),
                    "sha256": WORKLOAD.sha256(library),
                },
            )

            with mock.patch.object(WORKLOAD, "command_output", return_value=""):
                self.assertIsNone(
                    WORKLOAD.verify_loaded_deepmd_c(plugin, {}, required=False)
                )
                with self.assertRaisesRegex(ValueError, "no libdeepmd_c"):
                    WORKLOAD.verify_loaded_deepmd_c(plugin, {})

    def test_runtime_environment_contract_ignores_only_gpu_ordinal(self) -> None:
        selected = {
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/fixture/lib",
            "OMP_NUM_THREADS": "1",
        }
        expected = {
            "LD_LIBRARY_PATH": "/fixture/lib",
            "OMP_NUM_THREADS": "1",
        }
        self.assertEqual(
            WORKLOAD.runtime_environment_contract(selected), expected
        )
        selected["CUDA_VISIBLE_DEVICES"] = "3"
        self.assertEqual(
            WORKLOAD.runtime_environment_contract(selected), expected
        )
        selected["LD_LIBRARY_PATH"] = "/different/lib"
        self.assertNotEqual(
            WORKLOAD.runtime_environment_contract(selected), expected
        )
        self.assertIsNone(WORKLOAD.runtime_environment_contract({}))

    def test_retry_inputs_are_content_addressed_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-retry-input-") as temporary:
            root = Path(temporary)
            first = WORKLOAD.generated_lammps_input_path(
                root, "seed-round-11", "variable seed equal 1001\n"
            )
            second = WORKLOAD.generated_lammps_input_path(
                root, "seed-round-11", "variable seed equal 2010\n"
            )
            self.assertNotEqual(first, second)
            WORKLOAD.write_generated(first, "variable seed equal 1001\n")
            WORKLOAD.write_generated(second, "variable seed equal 2010\n")
            self.assertEqual(
                first.read_text(encoding="utf-8"),
                "variable seed equal 1001\n",
            )
            self.assertEqual(
                second.read_text(encoding="utf-8"),
                "variable seed equal 2010\n",
            )

    def test_adaptive_seed_schedule_is_manifest_bounded(self) -> None:
        self.assertEqual(
            WORKLOAD.seed_attempt_schedule(self.manifest, 20),
            (
                3000,
                3000,
                6000,
                6000,
                12000,
                12000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
                24000,
            ),
        )
        with self.assertRaisesRegex(ValueError, "exceeds the manifest"):
            WORKLOAD.seed_attempt_schedule(self.manifest, 21)
        malformed = copy.deepcopy(self.manifest)
        malformed["protocol"]["seed_walk_attempt_step_multipliers"] = [1, 0, 2]
        with self.assertRaisesRegex(ValueError, "positive integers"):
            WORKLOAD.seed_attempt_schedule(malformed, 3)
        self.assertEqual(
            WORKLOAD.restart_checkpoint_timesteps(
                timestep_offset=100, steps=80, frequency=25
            ),
            [125, 150, 175],
        )

    def test_superseded_seed_attempt_preserves_record_input_logs_and_outputs(
        self,
    ) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[16]
        with tempfile.TemporaryDirectory(prefix="dprc-seed-archive-") as temporary:
            root = Path(temporary)
            start = root / "parent.data"
            start.write_bytes(b"accepted parent\n")
            output_directory = root / "states/seeds" / window.tag
            output_directory.mkdir(parents=True)
            (output_directory / "partial.data").write_bytes(b"failed state\n")
            log_directory = root / "logs/seed-round-01"
            log_directory.mkdir(parents=True)
            (log_directory / "launcher.log").write_text("failed\n", encoding="utf-8")
            generated_input = root / "generated/inputs/attempt.in"
            generated_input.parent.mkdir(parents=True)
            generated_input.write_text("run 3000\n", encoding="utf-8")
            record_path = root / "records/seed-round-01.json"
            record_path.parent.mkdir(parents=True)
            record_path.write_text(
                json.dumps({"input": {"path": str(generated_input)}}),
                encoding="utf-8",
            )
            run_window = WORKLOAD.RunWindow(
                window,
                start,
                output_directory,
                root,
                seed=12345,
                colvars_profile="seed",
            )
            with mock.patch.object(WORKLOAD.time, "time_ns", return_value=42):
                archived = WORKLOAD.archive_stale_invocation_artifacts(
                    name="seed-round-01",
                    output=root,
                    record_path=record_path,
                    log_directory=log_directory,
                    run_windows=[run_window],
                )
            self.assertEqual(
                archived,
                root / "superseded/seed-round-01-42",
            )
            assert archived is not None
            self.assertTrue((archived / "record.json").is_file())
            self.assertEqual(
                (archived / "input.lammps").read_text(encoding="utf-8"),
                "run 3000\n",
            )
            self.assertTrue((archived / "logs/launcher.log").is_file())
            self.assertTrue(
                (archived / f"outputs/{window.tag}/partial.data").is_file()
            )
            self.assertTrue(generated_input.is_file())
            self.assertEqual(list(log_directory.iterdir()), [])
            self.assertEqual(list(output_directory.iterdir()), [])

    def test_source_verification_requires_explicit_dirty_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-source-") as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )
            artifact = repository / "runtime.dat"
            artifact.write_bytes(b"reviewed runtime bytes\n")
            subprocess.run(
                ["git", "-C", str(repository), "add", "runtime.dat"], check=True
            )
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True
            )
            revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            manifest = copy.deepcopy(self.manifest)
            manifest["source"].update(
                {
                    "revision": revision,
                    "license": "NOASSERTION",
                    "assets_complete": False,
                    "artifacts": [
                        {
                            "path": "runtime.dat",
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            "classification": "test runtime input",
                        }
                    ],
                }
            )
            with self.assertRaisesRegex(ValueError, "license is unresolved"):
                WORKLOAD.verify_source(
                    repository, manifest, allow_unqualified_source=False
                )
            (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow-unqualified-source"):
                WORKLOAD.verify_source(
                    repository, manifest, allow_unqualified_source=False
                )
            result = WORKLOAD.verify_source(
                repository, manifest, allow_unqualified_source=True
            )
            self.assertEqual(result["qualification"], "private-diagnostic")
            self.assertTrue(result["dirty"])
            with self.assertRaisesRegex(ValueError, "outside the external tutorial"):
                WORKLOAD.prepare_workspace(
                    repository / "runs",
                    repository,
                    ROOT / "workloads/etpeth/manifest.json",
                    manifest,
                    result,
                )

    def test_artifact_only_source_snapshot_is_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-snapshot-") as temporary:
            snapshot = Path(temporary)
            artifact = snapshot / "runtime.dat"
            artifact.write_bytes(b"reviewed runtime bytes\n")
            manifest = copy.deepcopy(self.manifest)
            manifest["source"].update(
                {
                    "artifacts": [
                        {
                            "path": "runtime.dat",
                            "sha256": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest(),
                            "classification": "test runtime input",
                        }
                    ]
                }
            )
            with self.assertRaisesRegex(ValueError, "no Git revision"):
                WORKLOAD.verify_source(
                    snapshot, manifest, allow_unqualified_source=False
                )
            result = WORKLOAD.verify_source(
                snapshot, manifest, allow_unqualified_source=True
            )
            self.assertEqual(
                result["identity_kind"], "artifact-verified-source-snapshot"
            )
            self.assertIsNone(result["revision"])
            self.assertEqual(
                result["expected_revision"], manifest["source"]["revision"]
            )
            self.assertEqual(result["qualification"], "private-diagnostic")

    def test_unversioned_source_tree_is_not_reported_clean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-source-tree-") as temporary:
            snapshot = Path(temporary)
            record = WORKLOAD.source_tree_record(snapshot)
            self.assertEqual(record["identity_kind"], "unversioned-source-snapshot")
            self.assertIsNone(record["revision"])
            self.assertIsNone(record["dirty"])

            (snapshot / ".source-revision").write_text(
                "a" * 40 + "\n", encoding="utf-8"
            )
            stamped = WORKLOAD.source_tree_record(snapshot)
            self.assertEqual(
                stamped["identity_kind"], "revision-stamped-source-snapshot"
            )
            self.assertEqual(stamped["revision"], "a" * 40)
            self.assertIsNone(stamped["dirty"])

    def test_rendered_batch_has_unique_world_state_and_private_style(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-render-") as temporary:
            root = Path(temporary)
            tutorial = root / "tutorial"
            (tutorial / "lammps").mkdir(parents=True)
            (tutorial / "lammps/forcefield_qmmm_hybrid.inc").write_text(
                "# qmmm fixture\n", encoding="utf-8"
            )
            (tutorial / "lammps/forcefield_mm_hybrid.inc").write_text(
                "# full-mm fixture\n", encoding="utf-8"
            )
            plugin = root / "dprcplugin.so"
            plugin.write_bytes(b"plugin")
            qmmm_starts = [root / f"qmmm-start-{index}.data" for index in (0, 1)]
            for path in qmmm_starts:
                self.write_topology_fixture(path, "qmmm")
            run_windows = [
                WORKLOAD.RunWindow(
                    windows[index],
                    start,
                    root / "run" / "states" / f"window-{index}",
                    root / "run",
                    1000 + index,
                )
                for index, start in zip((0, 1), qmmm_starts)
            ]
            text = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                timestep_offset=5000,
                trajectory_frequency=0,
                restart_checkpoint_frequency=25,
            )
            self.assertIn("variable start_data world &", text)
            self.assertIn("reset_timestep 5000", text)
            self.assertIn("timestep 1.000000", text)
            self.assertIn("neigh_modify every 1 delay 0 check no", text)
            self.assertIn("variable checkpoint_restart_root world &", text)
            self.assertIn("restart 25 ${checkpoint_restart_root}", text)
            self.assertIn("fix qmmm qm qmmm/xtb/dprc", text)
            self.assertIn("pair_style hybrid/overlay/kk lj/cut/dprc/batch", text)
            self.assertIn("tip4p/long/dprc/batch", text)
            self.assertIn("kspace_style pppm/tip4p/dprc/batch", text)
            self.assertNotIn("fix qmmm qm qmmm/xtb elements", text)
            self.assertNotIn("kspace_style pppm/tip4p/xtb", text)
            generated_forcefield = root / "run/generated/forcefield_dprc_batch.inc"
            self.assertTrue(generated_forcefield.is_file())
            self.assertEqual(
                generated_forcefield.read_text(encoding="utf-8"),
                "# qmmm fixture\n",
            )
            self.assertEqual(text.count("write_data ${final_data} nocoeff"), 1)

            classical_starts = [
                root / f"classical-start-{index}.data" for index in (0, 1)
            ]
            for path in classical_starts:
                self.write_topology_fixture(path, "classical")
            classical_windows = [
                WORKLOAD.RunWindow(
                    windows[index],
                    start,
                    root / "classical-run" / "states" / f"window-{index}",
                    root / "classical-run",
                    2000 + index,
                )
                for index, start in zip((0, 1), classical_starts)
            ]
            classical = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                classical_windows,
                steps=3,
                trajectory_frequency=0,
                mode="classical",
            )
            self.assertIn("fix classical all dprc/classical/batch", classical)
            self.assertIn(
                "pair_style hybrid/overlay/kk lj/cut/dprc/batch", classical
            )
            self.assertIn("pppm/tip4p/dprc/batch", classical)
            self.assertIn("dihedral_style harmonic/kk", classical)
            self.assertIn("edihed", classical)
            self.assertIn(
                "pair_coeff * * tip4p/long/dprc/batch", classical
            )
            self.assertNotIn(
                "pair_coeff 6*7 6*7 tip4p/long/dprc/batch", classical
            )
            self.assertNotIn("fix qmmm", classical)
            self.assertEqual(
                (
                    root
                    / "classical-run/generated/forcefield_dprc_batch.inc"
                ).read_text(encoding="utf-8"),
                "# full-mm fixture\n",
            )

            host_classical = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                classical_windows,
                steps=3,
                trajectory_frequency=0,
                mode="classical",
                lammps_execution_backend="host",
            )
            self.assertIn("atom_style full\n", host_classical)
            self.assertNotIn("run_style verlet/kk", host_classical)
            self.assertIn("dihedral_style harmonic\n", host_classical)
            self.assertIn(
                "pair_style hybrid/overlay lj/cut/dprc/batch", host_classical
            )
            self.assertIn("fix classical all dprc/classical/batch", host_classical)
            self.assertNotIn("/kk", host_classical)

            host_qmmm = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                trajectory_frequency=0,
                mode="qmmm",
                lammps_execution_backend="host",
            )
            self.assertIn("atom_style full\n", host_qmmm)
            self.assertNotIn("run_style verlet/kk", host_qmmm)
            self.assertIn("fix qmmm qm qmmm/xtb/dprc", host_qmmm)
            self.assertIn("fix integrate all nve\n", host_qmmm)
            self.assertNotIn("/kk", host_qmmm)

            dpa4c_model = root / "dpa4c.pt2"
            dpa4c_model.write_bytes(b"dpa4c-model")
            dpa4c = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                trajectory_frequency=0,
                mode="qmmm-dpa4c",
                deepmd_models=[dpa4c_model],
            )
            self.assertIn(
                f"dprc/deepmd/batch/kk {dpa4c_model.resolve()} partition_batch yes",
                dpa4c,
            )
            self.assertIn(
                "compute dprc_lj_energy all pair lj/cut/dprc/batch evdwl",
                dpa4c,
            )
            self.assertIn(
                "compute dprc_correction_energy all pair "
                "dprc/deepmd/batch/kk evdwl",
                dpa4c,
            )
            self.assertIn(
                "c_dprc_lj_energy c_dprc_correction_energy", dpa4c
            )
            self.assertEqual(dpa4c.count("plugin load"), 1)
            self.assertNotIn("partition_batch yes", classical)
            self.assertNotIn("c_dprc_correction_energy", classical)

            host_dpa4c = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                trajectory_frequency=0,
                mode="qmmm-dpa4c",
                deepmd_models=[dpa4c_model],
                lammps_execution_backend="host",
            )
            self.assertIn("atom_style full\n", host_dpa4c)
            self.assertIn("pair_style hybrid/overlay lj/cut/dprc/batch", host_dpa4c)
            self.assertIn(
                f"dprc/deepmd/batch {dpa4c_model.resolve()} partition_batch yes",
                host_dpa4c,
            )
            self.assertIn("fix water_shake water shake ", host_dpa4c)
            self.assertIn("fix integrate all nve\n", host_dpa4c)
            self.assertIn("fix restraints all colvars ", host_dpa4c)
            self.assertNotIn("/kk", host_dpa4c)

            nve = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                trajectory_frequency=0,
                thermostat_enabled=False,
            )
            self.assertNotIn("fix thermostat", nve)
            self.assertNotIn("fix remove_com", nve)
            self.assertIn("fix integrate all nve/kk", nve)
            self.assertIn("fix restraints all colvars/kk", nve)

    def test_nve_execution_record_is_explicit_without_changing_nvt_ledger(self) -> None:
        arguments = {
            "mode": "qmmm",
            "model_deviation_frequency": 0,
            "dpa4c_models_qualified": False,
            "allow_unqualified_dpa4c_models": False,
        }
        self.assertEqual(
            WORKLOAD.execution_record(**arguments),
            {"mode": "qmmm", "lammps_execution_backend": "kokkos"},
        )
        self.assertEqual(
            WORKLOAD.execution_record(**arguments, thermostat_enabled=False),
            {
                "mode": "qmmm",
                "lammps_execution_backend": "kokkos",
                "dynamics": {
                    "ensemble": "NVE",
                    "thermostat": "disabled",
                    "center_of_mass_momentum_removal": "disabled",
                    "umbrella_restraints": "enabled",
                },
            },
        )
        self.assertEqual(
            WORKLOAD.execution_record(
                **arguments, lammps_execution_backend="host"
            )["lammps_execution_backend"],
            "host",
        )

    def test_lammps_backend_arguments_omit_kokkos_for_host(self) -> None:
        self.assertEqual(WORKLOAD.lammps_backend_arguments("host"), ())
        kokkos = WORKLOAD.lammps_backend_arguments("kokkos")
        self.assertEqual(kokkos[:4], ("-k", "on", "g", "1"))
        self.assertIn("kokkos", kokkos)
        with self.assertRaisesRegex(ValueError, "execution backend"):
            WORKLOAD.lammps_backend_arguments("invalid")

    def test_production_requires_current_fixed_threshold_nve_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-nve-ledger-") as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            record = root / "records/nve-stability.json"
            model = root / "model.pt2"
            qualification = root / "qualification/nve-stability.json"
            record.parent.mkdir(parents=True)
            qualification.parent.mkdir(parents=True)
            manifest.write_bytes(b"manifest\n")
            record.write_bytes(b"record\n")
            model.write_bytes(b"model\n")
            qualification.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "scope": "three-window-qmmm-dpa4c-nve-stability",
                        "thresholds": {
                            "minimum_samples": WORKLOAD.NVE_MINIMUM_SAMPLES,
                            "maximum_absolute_drift_rate_kcal_mol_ps_atom": (
                                WORKLOAD.NVE_MAXIMUM_ABSOLUTE_DRIFT_RATE_KCAL_MOL_PS_ATOM
                            ),
                            "maximum_absolute_net_drift_kcal_mol_atom": (
                                WORKLOAD.NVE_MAXIMUM_ABSOLUTE_NET_DRIFT_KCAL_MOL_ATOM
                            ),
                            "minimum_mean_temperature_kelvin": (
                                WORKLOAD.NVE_MINIMUM_MEAN_TEMPERATURE_KELVIN
                            ),
                            "maximum_mean_temperature_kelvin": (
                                WORKLOAD.NVE_MAXIMUM_MEAN_TEMPERATURE_KELVIN
                            ),
                        },
                        "inputs": {
                            "record": {
                                "path": str(record),
                                "sha256": WORKLOAD.sha256(record),
                            },
                            "manifest": {
                                "path": str(manifest),
                                "sha256": WORKLOAD.sha256(manifest),
                            },
                            "models": [
                                {
                                    "path": str(model),
                                    "sha256": WORKLOAD.sha256(model),
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = WORKLOAD.require_nve_stability_qualification(
                root, manifest, [model]
            )
            self.assertEqual(result["status"], "passed")
            model.write_bytes(b"changed model\n")
            with self.assertRaisesRegex(ValueError, "different DPA4c model bytes"):
                WORKLOAD.require_nve_stability_qualification(
                    root, manifest, [model]
                )

    def test_topology_guard_rejects_qmmm_data_for_classical_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-topology-") as temporary:
            path = Path(temporary) / "start.data"
            self.write_topology_fixture(path, "qmmm")
            with self.assertRaisesRegex(ValueError, "Refusing to mix"):
                WORKLOAD.validate_lammps_topology(
                    path, self.manifest, "classical"
                )

    def test_topology_guard_accepts_lammps_omitted_zero_count(self) -> None:
        """A QM/MM checkpoint may omit its zero-dihedral header."""
        with tempfile.TemporaryDirectory(prefix="dprc-zero-topology-") as temporary:
            path = Path(temporary) / "start.data"
            contract = WORKLOAD.topology_contract(self.manifest, "qmmm")
            path.write_text(
                "fixture\n\n"
                f"{contract['atoms']} atoms\n"
                f"{contract['bonds']} bonds\n"
                f"{contract['angles']} angles\n"
                "0 impropers\n\nMasses\n",
                encoding="utf-8",
            )
            self.assertEqual(
                WORKLOAD.validate_lammps_topology(path, self.manifest, "qmmm"),
                {
                    "atoms": contract["atoms"],
                    "bonds": contract["bonds"],
                    "angles": contract["angles"],
                    "dihedrals": 0,
                },
            )

    def test_initial_data_path_depends_on_execution_mode(self) -> None:
        tutorial = Path("/fixture/tutorial")
        self.assertEqual(
            WORKLOAD.initial_data_for_mode(
                self.manifest, tutorial, "classical"
            ),
            tutorial / "lammps/ETP_ETH.mm.data",
        )
        self.assertEqual(
            WORKLOAD.initial_data_for_mode(self.manifest, tutorial, "qmmm"),
            tutorial / "lammps/ETP_ETH.data",
        )

    def test_unchecked_dangerous_summary_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-neighbor-summary-") as temporary:
            root = Path(temporary)
            (root / "log.lammps").write_text(
                "Neighbor list builds = 10\nDangerous builds not checked\n",
                encoding="utf-8",
            )
            self.assertEqual(
                WORKLOAD.inspect_dangerous_builds(root, 1),
                {"log.lammps": None},
            )

    def test_seed_colvars_profile_is_stronger_but_sampling_contract_is_unchanged(
        self,
    ) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[17]
        sampling = WORKLOAD.render_colvars(
            self.manifest, window, profile="sampling"
        )
        seed = WORKLOAD.render_colvars(self.manifest, window, profile="seed")
        self.assertIn("forceConstant 200.0", sampling)
        self.assertIn("forceConstant 10000.0", seed)
        self.assertIn("production uses the sampling force constant", seed)
        run = WORKLOAD.RunWindow(
            window,
            Path("start.data"),
            Path("states"),
            Path("workspace"),
            1234,
            colvars_profile="seed",
        )
        self.assertEqual(
            run.colvars_config,
            Path("workspace/generated/colvars-seed/m1p4.conf"),
        )
        with self.assertRaisesRegex(ValueError, "unsupported Colvars profile"):
            WORKLOAD.render_colvars(self.manifest, window, profile="invalid")

    def test_chunk_partition_has_no_gap_or_duplicate_steps(self) -> None:
        self.assertEqual(WORKLOAD.chunk_sizes(12001, 5000), [5000, 5000, 2001])
        chunks = WORKLOAD.chunk_sizes(100000, 5000)
        self.assertEqual(len(chunks), 20)
        self.assertEqual(sum(chunks), 100000)
        self.assertTrue(all(chunk > 0 for chunk in chunks))

    def test_lammps_command_uses_partition_logs_only_for_multiple_worlds(self) -> None:
        common = {
            "lammps": Path("/runtime/lmp"),
            "mpi_launcher": Path("/runtime/mpiexec"),
            "mpi_args": [],
            "log_directory": Path("/run/logs"),
            "input_path": Path("/run/input.in"),
        }
        one_world = WORKLOAD.build_lammps_command(
            worlds=1, ranks_per_window=2, **common
        )
        self.assertIn("-log", one_world)
        self.assertIn("-screen", one_world)
        self.assertNotIn("-plog", one_world)
        self.assertNotIn("-partition", one_world)

        two_worlds = WORKLOAD.build_lammps_command(
            worlds=2,
            ranks_per_window=1,
            lammps_args=(
                "-k",
                "on",
                "g",
                "1",
                "-pk",
                "kokkos",
                "newton",
                "on",
                "neigh",
                "half",
            ),
            **common,
        )
        self.assertIn("-plog", two_worlds)
        self.assertIn("-pscreen", two_worlds)
        self.assertIn("-partition", two_worlds)
        self.assertNotIn("-log", two_worlds)
        lammps_index = two_worlds.index(str(common["lammps"].resolve()))
        self.assertEqual(
            two_worlds[lammps_index + 1 : lammps_index + 12],
            [
                "-k",
                "on",
                "g",
                "1",
                "-pk",
                "kokkos",
                "newton",
                "on",
                "neigh",
                "half",
                "-partition",
            ],
        )

    def test_single_window_smoke_honors_step_override(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-smoke-") as temporary:
            root = Path(temporary)
            arguments = Namespace(
                output=root / "run",
                tutorial=root / "tutorial",
                lammps=root / "lmp",
                plugin=root / "dprcplugin.so",
                xtbloom_library=root / "libxtbloom.so",
                manifest=ROOT / "workloads/etpeth/manifest.json",
                ranks_per_window=1,
                mpiexec=root / "mpiexec",
                mpi_arg=[],
                library_dir=[],
                cuda_visible_devices="0",
                stage="smoke",
                smoke_window_count=2,
                smoke_steps=25,
                chunk_steps=5000,
                trial=[],
            )
            with mock.patch.object(WORKLOAD, "run_invocation") as invocation:
                WORKLOAD.run_stage(arguments, self.manifest, windows)
            self.assertEqual(invocation.call_args.kwargs["steps"], 25)
            self.assertEqual(len(invocation.call_args.kwargs["run_windows"]), 1)

    def test_rendered_seed_search_halts_on_the_colvars_gate(self) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[28]
        with tempfile.TemporaryDirectory(prefix="dprc-first-hit-render-") as temporary:
            root = Path(temporary)
            tutorial = root / "tutorial/lammps"
            tutorial.mkdir(parents=True)
            (tutorial / "forcefield_mm_hybrid.inc").write_text(
                "# full-mm fixture\n", encoding="utf-8"
            )
            start = root / "start.data"
            self.write_topology_fixture(start, "classical")
            run_window = WORKLOAD.RunWindow(
                window,
                start,
                root / "run/attempt",
                root / "run",
                12345,
                colvars_profile="seed",
            )
            text = WORKLOAD.render_lammps_input(
                self.manifest,
                root / "tutorial",
                root / "dprcplugin.so",
                [run_window],
                steps=3000,
                trajectory_frequency=0,
                mode="classical",
                stop_on_seed_acceptance=True,
            )
            self.assertIn("f_restraints[1][1]", text)
            self.assertIn("f_restraints[2][1]", text)
            self.assertIn(f"-({window.center:.17g})", text)
            self.assertIn(
                "fix seed_first_hit all halt 25 v_seed_gate_reached != 0 "
                "error soft message yes",
                text,
            )
            self.assertLess(text.index("fix seed_first_hit"), text.index("run 3000"))
            self.assertLess(text.index("run 3000"), text.index("write_data"))

    def test_anchor_stage_publishes_same_process_first_hit(self) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[16]
        with tempfile.TemporaryDirectory(prefix="dprc-anchor-first-hit-") as temporary:
            root = Path(temporary)
            output = root / "run"
            start = root / "start.data"
            start.write_bytes(b"initial anchor\n")
            final_data = output / "states/anchor" / window.tag / f"{window.tag}.data"
            final_restart = final_data.with_suffix(".restart")
            colvars = final_data.with_suffix(".colvars.traj")
            final_data.parent.mkdir(parents=True)
            final_data.write_bytes(b"accepted first-hit state\n")
            final_restart.write_bytes(b"accepted first-hit restart\n")
            colvars.write_text(
                "# step reaction_coordinate attack_angle\n"
                "0 -1.50 180.0\n"
                "25 -1.445 172.8\n",
                encoding="utf-8",
            )
            record_path = output / "records/anchor-first-hit.json"
            record_path.parent.mkdir(parents=True)
            start_identity = {
                "path": str(start.resolve()),
                "sha256": WORKLOAD.sha256(start),
            }
            output_record = {
                "center_angstrom": window.center,
                "data": {
                    "path": str(final_data.resolve()),
                    "sha256": WORKLOAD.sha256(final_data),
                },
                "restart": {
                    "path": str(final_restart.resolve()),
                    "sha256": WORKLOAD.sha256(final_restart),
                },
                "colvars": {
                    "path": str(colvars.resolve()),
                    "sha256": WORKLOAD.sha256(colvars),
                },
                "final_values": {
                    "step": 25.0,
                    "reaction_coordinate": -1.445,
                    "attack_angle": 172.8,
                },
                "reaction_coordinate_error_angstrom": 0.055,
                "attack_angle_error_degree": 7.2,
                "seed_acceptance": True,
                "first_hit": {
                    "check_frequency_steps": 25,
                    "scheduled_steps": 20000,
                    "completed_steps": 25,
                    "halted_early": True,
                    "accepted_sample_found": True,
                },
            }
            record = {
                "schema_version": 1,
                "record_kind": "native-invocation",
                "name": "anchor-first-hit",
                "status": "passed",
                "wall_seconds": 2.0,
                "steps_per_window": 20000,
                "timestep_offset": 0,
                "restart_checkpoint_frequency_steps": 0,
                "stop_on_seed_acceptance": True,
                "worlds": 1,
                "window_order": [window.tag],
                "ranks_per_window": 1,
                "aggregate_window_steps": 25,
                "aggregate_window_steps_per_second": 12.5,
                "start_inputs": {window.tag: start_identity},
                "outputs": {window.tag: output_record},
            }
            record_path.write_text(json.dumps(record), encoding="utf-8")
            common = {
                "output": output,
                "manifest": self.manifest,
                "ranks_per_window": 1,
            }
            with mock.patch.object(
                WORKLOAD, "run_invocation", return_value=record_path
            ) as invocation:
                ledger_path = WORKLOAD.run_first_hit_stage(
                    stage="anchor",
                    manifest=self.manifest,
                    window=window,
                    start_data=start,
                    stage_root=output / "states/anchor",
                    total_steps=20000,
                    maximum_chunk_steps=5000,
                    trajectory_frequency=0,
                    common=common,
                )
            invocation.assert_called_once()
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["qualification"], "native-first-hit")
            self.assertEqual(ledger["selected_step"], 25)
            self.assertEqual(
                ledger["outputs"][window.tag]["final_values"]["step"], 25.0
            )

    def test_anchor_no_hit_record_is_not_resumable(self) -> None:
        """A legacy no-hit record must be converted from passed to failed."""
        window = WORKLOAD.windows_from_manifest(self.manifest)[16]
        with tempfile.TemporaryDirectory(prefix="dprc-anchor-no-hit-") as temporary:
            root = Path(temporary)
            output = root / "run"
            start = root / "start.data"
            start.write_bytes(b"initial anchor\n")
            final_data = output / "states/anchor" / window.tag / f"{window.tag}.data"
            final_restart = final_data.with_suffix(".restart")
            colvars = final_data.with_suffix(".colvars.traj")
            final_data.parent.mkdir(parents=True)
            final_data.write_bytes(b"unaccepted endpoint\n")
            final_restart.write_bytes(b"unaccepted restart\n")
            colvars.write_text(
                "# step reaction_coordinate attack_angle\n0 -1.50 180.0\n",
                encoding="utf-8",
            )
            record_path = output / "records/anchor-first-hit.json"
            record_path.parent.mkdir(parents=True)
            identity = lambda path: {
                "path": str(path.resolve()),
                "sha256": WORKLOAD.sha256(path),
            }
            record_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "outputs": {
                            window.tag: {
                                "seed_acceptance": False,
                                "first_hit": {
                                    "accepted_sample_found": False,
                                },
                                "data": identity(final_data),
                                "restart": identity(final_restart),
                                "colvars": identity(colvars),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            common = {"output": output, "manifest": self.manifest, "ranks_per_window": 1}
            with mock.patch.object(WORKLOAD, "run_invocation", return_value=record_path):
                with self.assertRaisesRegex(ValueError, "did not encounter"):
                    WORKLOAD.run_first_hit_stage(
                        stage="anchor",
                        manifest=self.manifest,
                        window=window,
                        start_data=start,
                        stage_root=output / "states/anchor",
                        total_steps=20000,
                        maximum_chunk_steps=5000,
                        trajectory_frequency=0,
                        common=common,
                    )
            failed = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")

    def test_sparse_colvars_output_does_not_undercount_normal_runs(self) -> None:
        self.assertEqual(
            WORKLOAD.completed_steps_for_record(
                requested_steps=5,
                timestep_offset=0,
                final_colvars_step=0.0,
                stop_on_seed_acceptance=False,
            ),
            5,
        )
        self.assertEqual(
            WORKLOAD.completed_steps_for_record(
                requested_steps=3000,
                timestep_offset=0,
                final_colvars_step=25.0,
                stop_on_seed_acceptance=True,
            ),
            25,
        )

    def test_seed_stage_adapts_duration_from_the_unchanged_parent(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["umbrella"].update(
            {
                "start_tenths_angstrom": -16,
                "stop_tenths_angstrom": -14,
                "count": 3,
                "available_initial_center_tenths_angstrom": -15,
            }
        )
        windows = WORKLOAD.windows_from_manifest(manifest)
        anchor = windows[1]
        with tempfile.TemporaryDirectory(prefix="dprc-adaptive-seeds-") as temporary:
            root = Path(temporary)
            output = root / "run"
            anchor_state = WORKLOAD.state_output(output, "anchor", anchor)
            anchor_state.parent.mkdir(parents=True)
            anchor_state.write_bytes(b"accepted anchor\n")
            anchor_ledger = {
                "outputs": {
                    anchor.tag: {
                        "data": {
                            "path": str(anchor_state),
                            "sha256": WORKLOAD.sha256(anchor_state),
                        }
                    }
                }
            }
            arguments = Namespace(
                output=output,
                tutorial=root / "tutorial",
                lammps=root / "lmp",
                plugin=root / "dprcplugin.so",
                xtbloom_library=root / "libxtbloom.so",
                manifest=ROOT / "workloads/etpeth/manifest.json",
                ranks_per_window=1,
                mpiexec=root / "mpiexec",
                mpi_arg=[],
                library_dir=[],
                cuda_visible_devices="0",
                mode="classical",
                deepmd_model=[],
                model_deviation_frequency=0,
                dpa4c_models_qualified=False,
                allow_unqualified_dpa4c_models=False,
                stage="seeds",
                smoke_window_count=2,
                smoke_steps=5,
                chunk_steps=5000,
                seed_max_attempts=3,
                trial=[],
            )
            calls = []

            def simulate_first_hit_search(**kwargs):
                calls.append(kwargs)
                item = kwargs["run_windows"][0]
                name = kwargs["name"]
                record_path = output / "records" / f"{name}.json"
                record_path.parent.mkdir(parents=True, exist_ok=True)
                start_identity = {
                    "path": str(item.start_data.resolve()),
                    "sha256": WORKLOAD.sha256(item.start_data),
                }
                colvars = Path(str(item.colvars_prefix) + ".colvars.traj")
                colvars.parent.mkdir(parents=True, exist_ok=True)
                attempt_index = kwargs["seed_attempt"]["attempt_index"]
                accepted = (
                    item.window == windows[0] and attempt_index == 2
                ) or (item.window == windows[2] and attempt_index == 0)
                completed_steps = (
                    150
                    if accepted and item.window == windows[0]
                    else 100
                    if accepted
                    else kwargs["steps"]
                )
                reaction = (
                    item.window.center if accepted else item.window.center - 0.5
                )
                rows = [
                    "# step reaction_coordinate attack_angle",
                    f"0 {item.window.center - 0.5} 180.0",
                    f"{completed_steps} {reaction} 180.0",
                ]
                colvars.write_text("\n".join(rows) + "\n", encoding="utf-8")
                item.final_data.parent.mkdir(parents=True, exist_ok=True)
                item.final_data.write_bytes(
                    f"first-hit-data-{item.window.tag}-{attempt_index}\n".encode()
                )
                item.final_restart.write_bytes(
                    f"first-hit-restart-{item.window.tag}-{attempt_index}\n".encode()
                )
                final_values = {
                    "step": float(completed_steps),
                    "reaction_coordinate": reaction,
                    "attack_angle": 180.0,
                }
                payload = {
                    "outputs": {
                        item.window.tag: {
                            "data": {
                                "path": str(item.final_data),
                                "sha256": WORKLOAD.sha256(item.final_data),
                            },
                            "restart": {
                                "path": str(item.final_restart),
                                "sha256": WORKLOAD.sha256(item.final_restart),
                            },
                            "colvars": {
                                "path": str(colvars),
                                "sha256": WORKLOAD.sha256(colvars),
                            },
                            "final_values": final_values,
                            "reaction_coordinate_error_angstrom": abs(
                                reaction - item.window.center
                            ),
                            "attack_angle_error_degree": 0.0,
                            "seed_acceptance": accepted,
                            "first_hit": {
                                "check_frequency_steps": 25,
                                "scheduled_steps": kwargs["steps"],
                                "completed_steps": completed_steps,
                                "halted_early": completed_steps < kwargs["steps"],
                                "accepted_sample_found": accepted,
                            },
                        }
                    }
                }
                # Exercise the real schema-v4 round validator below.  The
                # simulator supplies the invocation-level protocol fields it
                # consumes; only the runtime/binary revalidation layer is
                # replaced because this focused test intentionally launches
                # no LAMMPS process.
                payload.update(
                    {
                        "name": name,
                        "status": "passed",
                        "steps_per_window": kwargs["steps"],
                        "timestep_offset": 0,
                        "restart_checkpoint_frequency_steps": 0,
                        "stop_on_seed_acceptance": kwargs[
                            "stop_on_seed_acceptance"
                        ],
                        "completed_steps_by_window": {
                            item.window.tag: completed_steps
                        },
                        "aggregate_window_steps": completed_steps,
                        "worlds": 1,
                        "window_order": [item.window.tag],
                        "ranks_per_window": 1,
                        "start_inputs": {item.window.tag: start_identity},
                        "seed_attempt": kwargs["seed_attempt"],
                    }
                )
                WORKLOAD.write_json_atomic(record_path, payload)
                return record_path

            def load_invocation_record(path, _common):
                return json.loads(path.read_text(encoding="utf-8"))

            with (
                mock.patch.object(
                    WORKLOAD,
                    "require_completed_stage",
                    return_value=anchor_ledger,
                ),
                mock.patch.object(
                    WORKLOAD,
                    "require_seed_records",
                    return_value={window.tag: anchor_state for window in windows},
                ),
                mock.patch.object(
                    WORKLOAD,
                    "run_invocation",
                    side_effect=simulate_first_hit_search,
                ),
                mock.patch.object(
                    WORKLOAD,
                    "validate_invocation_record_current",
                    side_effect=load_invocation_record,
                ),
            ):
                WORKLOAD.run_stage(arguments, manifest, windows)

            ledger = json.loads(
                (output / "records/seed-round-01.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(ledger["schema_version"], 4)
            self.assertEqual(ledger["record_kind"], "seed-first-hit-ledger")

            self.assertEqual(
                [call["steps"] for call in calls], [3000, 3000, 6000, 3000]
            )
            self.assertTrue(
                all(
                    item.start_data == anchor_state
                    for call in calls
                    for item in call["run_windows"]
                )
            )
            self.assertTrue(
                all(call["stop_on_seed_acceptance"] is True for call in calls)
            )
            self.assertEqual(
                [call["seed_attempt"]["attempt_index"] for call in calls],
                [0, 1, 2, 0],
            )
            self.assertEqual(
                [
                    tuple(call["seed_attempt"]["thermostat_seeds"].values())
                    for call in calls
                ],
                [tuple(item.seed for item in call["run_windows"]) for call in calls],
            )
            self.assertEqual(
                len(
                    {
                        seed
                        for call in calls
                        for seed in call["seed_attempt"]["thermostat_seeds"].values()
                    }
                ),
                4,
            )
            for window in (windows[0], windows[2]):
                published = WORKLOAD.state_output(output, "seeds", window)
                capture = ledger["captures"][window.tag]
                self.assertEqual(
                    WORKLOAD.sha256(published), capture["source_data"]["sha256"]
                )

    def test_seed_stage_resumes_an_accepted_later_attempt_without_rerun(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["umbrella"].update(
            {
                "start_tenths_angstrom": -16,
                "stop_tenths_angstrom": -14,
                "count": 3,
                "available_initial_center_tenths_angstrom": -15,
            }
        )
        windows = WORKLOAD.windows_from_manifest(manifest)
        anchor = windows[1]
        with tempfile.TemporaryDirectory(prefix="dprc-resume-seeds-") as temporary:
            root = Path(temporary)
            output = root / "run"
            anchor_state = WORKLOAD.state_output(output, "anchor", anchor)
            anchor_state.parent.mkdir(parents=True)
            anchor_state.write_bytes(b"accepted anchor\n")
            anchor_identity = {
                "path": str(anchor_state),
                "sha256": WORKLOAD.sha256(anchor_state),
            }
            anchor_ledger = {
                "outputs": {anchor.tag: {"data": anchor_identity}}
            }
            record_path = output / "records/seed-round-01.json"
            record_path.parent.mkdir(parents=True)
            record_path.write_text("{}\n", encoding="utf-8")
            resumed_record = {
                "seed_attempt": {"attempt_number": 3},
                "outputs": {
                    window.tag: {
                        "data": {
                            "path": str(WORKLOAD.state_output(output, "seeds", window)),
                            "sha256": "fixture",
                        }
                    }
                    for window in (windows[0], windows[2])
                },
            }
            arguments = Namespace(
                output=output,
                tutorial=root / "tutorial",
                lammps=root / "lmp",
                plugin=root / "dprcplugin.so",
                xtbloom_library=root / "libxtbloom.so",
                manifest=ROOT / "workloads/etpeth/manifest.json",
                ranks_per_window=1,
                mpiexec=root / "mpiexec",
                mpi_arg=[],
                library_dir=[],
                cuda_visible_devices="0",
                mode="classical",
                deepmd_model=[],
                model_deviation_frequency=0,
                dpa4c_models_qualified=False,
                allow_unqualified_dpa4c_models=False,
                stage="seeds",
                smoke_window_count=2,
                smoke_steps=5,
                chunk_steps=5000,
                seed_max_attempts=3,
                trial=[],
            )
            with (
                mock.patch.object(
                    WORKLOAD,
                    "require_completed_stage",
                    return_value=anchor_ledger,
                ),
                mock.patch.object(
                    WORKLOAD,
                    "validate_seed_round_record",
                    return_value=resumed_record,
                ) as validate_round,
                mock.patch.object(
                    WORKLOAD,
                    "require_seed_records",
                    return_value={window.tag: anchor_state for window in windows},
                ),
                mock.patch.object(WORKLOAD, "run_invocation") as invocation,
            ):
                WORKLOAD.run_stage(arguments, manifest, windows)
            validate_round.assert_called_once()
            invocation.assert_not_called()

    def test_production_parser_exposes_dpa4c_execution_policy(self) -> None:
        arguments = WORKLOAD.build_parser().parse_args(
            [
                "run",
                "--tutorial",
                "/fixture/tutorial",
                "--output",
                "/fixture/output",
                "--lammps",
                "/fixture/lmp",
                "--plugin",
                "/fixture/dprcplugin.so",
                "--xtbloom-library",
                "/fixture/libxtbloom.so",
                "--mode",
                "qmmm-dpa4c",
                "--deepmd-model",
                "/fixture/model-0.pt2",
                "--model-deviation-frequency",
                "0",
                "--allow-unqualified-dpa4c-models",
                "--lammps-execution-backend",
                "host",
                "--stage",
                "batch-smoke",
            ]
        )
        self.assertEqual(arguments.mode, "qmmm-dpa4c")
        self.assertEqual(arguments.deepmd_model, [Path("/fixture/model-0.pt2")])
        self.assertEqual(arguments.model_deviation_frequency, 0)
        self.assertTrue(arguments.allow_unqualified_dpa4c_models)
        self.assertFalse(arguments.dpa4c_models_qualified)
        self.assertEqual(arguments.lammps_execution_backend, "host")

    def test_production_parser_exposes_classical_control(self) -> None:
        arguments = WORKLOAD.build_parser().parse_args(
            [
                "run",
                "--tutorial",
                "/fixture/tutorial",
                "--output",
                "/fixture/output",
                "--lammps",
                "/fixture/lmp",
                "--plugin",
                "/fixture/dprcplugin.so",
                "--xtbloom-library",
                "/fixture/libxtbloom.so",
                "--mode",
                "classical",
                "--stage",
                "batch-smoke",
            ]
        )
        self.assertEqual(arguments.mode, "classical")

    def test_parser_exposes_fixed_length_nve_stability_stage(self) -> None:
        arguments = WORKLOAD.build_parser().parse_args(
            [
                "run",
                "--tutorial",
                "/fixture/tutorial",
                "--output",
                "/fixture/output",
                "--lammps",
                "/fixture/lmp",
                "--plugin",
                "/fixture/dprcplugin.so",
                "--xtbloom-library",
                "/fixture/libxtbloom.so",
                "--stage",
                "nve-stability",
            ]
        )
        self.assertEqual(arguments.stage, "nve-stability")
        self.assertEqual(arguments.nve_steps, 5000)

    def test_parser_exposes_bounded_lammps_process_retries(self) -> None:
        arguments = WORKLOAD.build_parser().parse_args(
            [
                "run",
                "--tutorial",
                "/fixture/tutorial",
                "--output",
                "/fixture/output",
                "--lammps",
                "/fixture/lmp",
                "--plugin",
                "/fixture/dprcplugin.so",
                "--xtbloom-library",
                "/fixture/libxtbloom.so",
                "--lammps-attempts",
                "3",
                "--stage",
                "equilibrate",
            ]
        )
        self.assertEqual(arguments.lammps_attempts, 3)

    def test_lammps_retry_is_limited_to_process_failures(self) -> None:
        accepted = Path("/fixture/accepted.json")
        with mock.patch.object(
            WORKLOAD,
            "run_invocation",
            side_effect=[WORKLOAD.LammpsExecutionError("signal 11"), accepted],
        ) as invocation:
            result = WORKLOAD.run_invocation_with_retries(
                maximum_attempts=2,
                name="fixture",
            )
        self.assertEqual(result, accepted)
        self.assertEqual(invocation.call_count, 2)
        self.assertEqual(
            invocation.call_args_list[0].kwargs["process_attempt"],
            {"attempt_number": 1, "maximum_attempts": 2},
        )
        self.assertEqual(
            invocation.call_args_list[1].kwargs["process_attempt"],
            {"attempt_number": 2, "maximum_attempts": 2},
        )

        with mock.patch.object(
            WORKLOAD,
            "run_invocation",
            side_effect=ValueError("scientific gate failed"),
        ) as invocation:
            with self.assertRaisesRegex(ValueError, "scientific gate failed"):
                WORKLOAD.run_invocation_with_retries(
                    maximum_attempts=3,
                    name="fixture",
                )
        invocation.assert_called_once()

        with self.assertRaisesRegex(ValueError, "must be positive"):
            WORKLOAD.run_invocation_with_retries(
                maximum_attempts=0,
                name="fixture",
            )

    def test_dpa4c_execution_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-policy-") as temporary:
            root = Path(temporary)
            models = [root / f"model-{index}.pt2" for index in range(2)]
            for model in models:
                model.write_bytes(model.name.encode("utf-8"))

            with self.assertRaisesRegex(ValueError, "requires either"):
                WORKLOAD.validate_execution_policy(
                    mode="qmmm-dpa4c",
                    deepmd_plugin=None,
                    deepmd_models=models[:1],
                    model_deviation_frequency=0,
                    dpa4c_models_qualified=False,
                    allow_unqualified_dpa4c_models=False,
                )
            with self.assertRaisesRegex(ValueError, "exactly one primary model"):
                WORKLOAD.validate_execution_policy(
                    mode="qmmm-dpa4c",
                    deepmd_plugin=None,
                    deepmd_models=models[:1],
                    model_deviation_frequency=100,
                    dpa4c_models_qualified=False,
                    allow_unqualified_dpa4c_models=True,
                )

            WORKLOAD.validate_execution_policy(
                mode="qmmm-dpa4c",
                deepmd_plugin=None,
                deepmd_models=models[:1],
                model_deviation_frequency=0,
                dpa4c_models_qualified=False,
                allow_unqualified_dpa4c_models=True,
            )

    def test_qualified_dpa4c_requires_canonical_environment_edge_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-canonical-") as temporary:
            model = Path(temporary) / "dpa4c.pt2"
            metadata = {
                "lower_input_kind": "dpa4c_canonical",
                "graph_edge_dtype": "float32",
                "canonical_index_dtype": "uint32",
                "dprc_graph_policy": WORKLOAD.DPA4C_DPRC_GRAPH_POLICY,
                "type_map": ["P", "O", "C", "H", "OW", "HW"],
                "dprc_environment_type_names": ["OW", "HW"],
                "dprc_environment_type_indices": [4, 5],
                "pair_exclude_types": [[4, 4], [4, 5], [5, 5]],
            }
            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("model/extra/metadata.json", json.dumps(metadata))
                archive.writestr(
                    "model/extra/model.json",
                    json.dumps(
                        {
                            "model": {
                                "descriptor": {"exclude_types": []},
                                "pair_exclude_types": [[4, 4], [4, 5], [5, 5]],
                            }
                        }
                    ),
                )
            WORKLOAD.validate_execution_policy(
                mode="qmmm-dpa4c",
                deepmd_plugin=None,
                deepmd_models=[model],
                model_deviation_frequency=0,
                dpa4c_models_qualified=True,
                allow_unqualified_dpa4c_models=False,
            )
            metadata["lower_input_kind"] = "graph"
            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("model/extra/metadata.json", json.dumps(metadata))
                archive.writestr(
                    "model/extra/model.json",
                    json.dumps(
                        {
                            "model": {
                                "descriptor": {"exclude_types": []},
                                "pair_exclude_types": [[4, 4], [4, 5], [5, 5]],
                            }
                        }
                    ),
                )
            with self.assertRaisesRegex(ValueError, "canonical metadata mismatch"):
                WORKLOAD.validate_execution_policy(
                    mode="qmmm-dpa4c",
                    deepmd_plugin=None,
                    deepmd_models=[model],
                    model_deviation_frequency=0,
                    dpa4c_models_qualified=True,
                    allow_unqualified_dpa4c_models=False,
                )

    def test_run_stage_forwards_dpa4c_runtime_and_qualification(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-forward-") as temporary:
            root = Path(temporary)
            model = root / "dpa4c.pt2"
            model.write_bytes(b"model")
            arguments = Namespace(
                output=root / "run",
                tutorial=root / "tutorial",
                lammps=root / "lmp",
                plugin=root / "dprcplugin.so",
                xtbloom_library=root / "libxtbloom.so",
                manifest=ROOT / "workloads/etpeth/manifest.json",
                ranks_per_window=1,
                mpiexec=root / "mpiexec",
                mpi_arg=[],
                library_dir=[],
                cuda_visible_devices="0",
                mode="qmmm-dpa4c",
                deepmd_model=[model],
                model_deviation_frequency=0,
                dpa4c_models_qualified=False,
                allow_unqualified_dpa4c_models=True,
                lammps_execution_backend="host",
                stage="smoke",
                smoke_window_count=2,
                smoke_steps=5,
                chunk_steps=5000,
                trial=[],
            )
            with mock.patch.object(WORKLOAD, "run_invocation") as invocation:
                WORKLOAD.run_stage(arguments, self.manifest, windows)
            forwarded = invocation.call_args.kwargs
            self.assertEqual(forwarded["mode"], "qmmm-dpa4c")
            self.assertIsNone(forwarded["deepmd_plugin"])
            self.assertEqual(forwarded["deepmd_models"], (model,))
            self.assertEqual(forwarded["model_deviation_frequency"], 0)
            self.assertFalse(forwarded["dpa4c_models_qualified"])
            self.assertTrue(forwarded["allow_unqualified_dpa4c_models"])
            self.assertEqual(forwarded["lammps_execution_backend"], "host")

    def test_chunked_stage_chains_outputs_and_preserves_ordered_series(self) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[16]
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-chunks-") as temporary:
            root = Path(temporary)
            initial = root / "initial.data"
            initial.write_bytes(b"initial\n")
            calls = []

            def fake_run_invocation(**kwargs):
                calls.append(
                    {
                        "name": kwargs["name"],
                        "steps": kwargs["steps"],
                        "offset": kwargs["timestep_offset"],
                        "start": kwargs["run_windows"][0].start_data,
                        "colvars_profile": kwargs["run_windows"][0].colvars_profile,
                    }
                )
                item = kwargs["run_windows"][0]
                item.output_directory.mkdir(parents=True, exist_ok=True)
                item.final_data.write_bytes(f"state-{len(calls)}\n".encode())
                item.final_restart.write_bytes(f"restart-{len(calls)}\n".encode())
                colvars = Path(str(item.colvars_prefix) + ".colvars.traj")
                colvars.write_text(
                    f"# step reaction_coordinate attack_angle\n"
                    f"{kwargs['timestep_offset'] + kwargs['steps']} -1.5 180.0\n",
                    encoding="utf-8",
                )
                trajectory = item.trajectory
                trajectory.write_bytes(f"trajectory-{len(calls)}\n".encode())

                def identity(path):
                    return {"path": str(path), "sha256": WORKLOAD.sha256(path)}

                record = root / "records" / f"{kwargs['name']}.json"
                WORKLOAD.write_json_atomic(
                    record,
                    {
                        "name": kwargs["name"],
                        "status": "passed",
                        "wall_seconds": float(len(calls)),
                        "steps_per_window": kwargs["steps"],
                        "timestep_offset": kwargs["timestep_offset"],
                        "worlds": 1,
                        "window_order": [item.window.tag],
                        "ranks_per_window": 1,
                        "start_inputs": {item.window.tag: identity(item.start_data)},
                        "outputs": {
                            item.window.tag: {
                                "data": identity(item.final_data),
                                "restart": identity(item.final_restart),
                                "colvars": identity(colvars),
                                "trajectory": identity(trajectory),
                            }
                        },
                    },
                )
                return record

            with mock.patch.object(
                WORKLOAD, "run_invocation", side_effect=fake_run_invocation
            ):
                ledger = WORKLOAD.run_chunked_stage(
                    stage="fixture",
                    manifest=self.manifest,
                    windows=[window],
                    start_data={window.tag: initial},
                    stage_root=root / "states/fixture",
                    total_steps=12,
                    maximum_chunk_steps=5,
                    seed_stage_code=8,
                    trial=0,
                    trajectory_frequency=1,
                    require_final_acceptance=False,
                    common={
                        "output": root,
                        "ranks_per_window": 1,
                        "manifest": self.manifest,
                    },
                    colvars_profile="seed",
                )
            self.assertEqual(
                [(call["offset"], call["steps"]) for call in calls],
                [(0, 5), (5, 5), (10, 2)],
            )
            self.assertEqual(calls[0]["start"], initial)
            self.assertTrue(
                all(call["colvars_profile"] == "seed" for call in calls)
            )
            self.assertEqual(
                calls[1]["start"],
                root / "states/fixture/checkpoints/chunk-001/m1p5/m1p5.data",
            )
            completed = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(completed["total_steps_per_window"], 12)
            self.assertEqual(len(completed["colvars_by_window"][window.tag]), 3)
            self.assertEqual(len(completed["trajectories_by_window"][window.tag]), 3)
            self.assertEqual(completed["summed_chunk_wall_seconds"], 6.0)
            self.assertEqual(completed["aggregate_window_steps_per_second"], 2.0)
            with mock.patch.object(
                WORKLOAD,
                "validate_invocation_record_current",
                side_effect=lambda path, _: json.loads(
                    path.read_text(encoding="utf-8")
                ),
            ):
                WORKLOAD.require_completed_stage(
                    ledger,
                    stage="fixture",
                    windows=[window],
                    expected_start_data={window.tag: initial},
                    expected_final_data={
                        window.tag: root / "states/fixture/m1p5/m1p5.data"
                    },
                    total_steps=12,
                    maximum_chunk_steps=5,
                    trajectory_frequency=1,
                    common={
                        "output": root,
                        "ranks_per_window": 1,
                        "manifest": self.manifest,
                    },
                    colvars_profile="seed",
                )
                with self.assertRaisesRegex(ValueError, "stage protocol"):
                    WORKLOAD.require_completed_stage(
                        ledger,
                        stage="fixture",
                        windows=[window],
                        expected_start_data={window.tag: initial},
                        expected_final_data={
                            window.tag: root / "states/fixture/m1p5/m1p5.data"
                        },
                        total_steps=12,
                        maximum_chunk_steps=4,
                        trajectory_frequency=1,
                        common={
                            "output": root,
                            "ranks_per_window": 1,
                            "manifest": self.manifest,
                        },
                        colvars_profile="seed",
                    )

                first_record_path = Path(completed["chunks"][0]["record"]["path"])
                first_record = json.loads(first_record_path.read_text(encoding="utf-8"))
                first_record["outputs"][window.tag].pop("trajectory")
                WORKLOAD.write_json_atomic(first_record_path, first_record)
                completed["chunks"][0]["record"]["sha256"] = WORKLOAD.sha256(
                    first_record_path
                )
                completed["trajectories_by_window"][window.tag].pop(0)
                WORKLOAD.write_json_atomic(ledger, completed)
                with self.assertRaisesRegex(ValueError, "outputs are incomplete"):
                    WORKLOAD.require_completed_stage(
                        ledger,
                        stage="fixture",
                        windows=[window],
                        expected_start_data={window.tag: initial},
                        expected_final_data={
                            window.tag: root / "states/fixture/m1p5/m1p5.data"
                        },
                        total_steps=12,
                        maximum_chunk_steps=5,
                        trajectory_frequency=1,
                        common={
                            "output": root,
                            "ranks_per_window": 1,
                            "manifest": self.manifest,
                        },
                        colvars_profile="seed",
                    )

    def test_parse_colvars_is_strict_and_merges_duplicate_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-colvars-") as temporary:
            root = Path(temporary)
            first = root / "first.traj"
            first.write_text(
                "# step reaction_coordinate attack_angle\n0 -1.4 179.0\n25 -1.5 178.5\n",
                encoding="utf-8",
            )
            self.assertEqual(
                WORKLOAD.parse_colvars(first, expected_final_step=25)[
                    "reaction_coordinate"
                ],
                -1.5,
            )
            second = root / "second.traj"
            second.write_text(
                "# step reaction_coordinate attack_angle\n"
                "25 -1.5000000000005 178.50000000005\n50 -1.45 179.0\n",
                encoding="utf-8",
            )
            reaction_tolerance, angle_tolerance = (
                WORKLOAD.checkpoint_boundary_tolerances(self.manifest)
            )
            merged = WORKLOAD.merge_colvars(
                [first, second], reaction_tolerance, angle_tolerance
            )
            self.assertEqual([row["step"] for row in merged], [0.0, 25.0, 50.0])
            conflicting = root / "conflicting.traj"
            conflicting.write_text(
                "# step reaction_coordinate attack_angle\n"
                "25 -1.50000001 178.5\n50 -1.45 179.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conflicting Colvars samples"):
                WORKLOAD.merge_colvars(
                    [first, conflicting], reaction_tolerance, angle_tolerance
                )
            fallback = root / "no-header.traj"
            fallback.write_text("0 -1.4 179.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "before a recognized header"):
                WORKLOAD.parse_colvars(fallback)
            with self.assertRaisesRegex(ValueError, "differs from expected"):
                WORKLOAD.parse_colvars(first, expected_final_step=50)

    def test_seed_dag_rejects_downstream_state_after_parent_rerun(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)[14:19]
        anchor = windows[2]
        step_schedule = WORKLOAD.seed_attempt_schedule(self.manifest, 3)
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-seeds-") as temporary:
            root = Path(temporary)

            def identity(path: Path) -> dict[str, str]:
                return {"path": str(path), "sha256": WORKLOAD.sha256(path)}

            anchor_state = WORKLOAD.state_output(root, "anchor", anchor)
            anchor_state.parent.mkdir(parents=True)
            anchor_state.write_bytes(b"anchor\n")
            anchor_ledger = {"outputs": {anchor.tag: {"data": identity(anchor_state)}}}
            lower = list(reversed(windows[:2]))
            upper = windows[3:]
            previous = {
                "lower": identity(anchor_state),
                "upper": identity(anchor_state),
            }
            records: list[Path] = []
            for round_index in range(2):
                start_inputs = {}
                outputs = {}
                round_windows = []
                for branch, branch_windows in (("lower", lower), ("upper", upper)):
                    window = branch_windows[round_index]
                    round_windows.append(window)
                    state = WORKLOAD.state_output(root, "seeds", window)
                    state.parent.mkdir(parents=True)
                    state.write_bytes(f"{window.tag}-round-{round_index}\n".encode())
                    start_inputs[window.tag] = previous[branch]
                    outputs[window.tag] = {
                        "data": identity(state),
                        "seed_acceptance": True,
                    }
                    previous[branch] = outputs[window.tag]["data"]
                record_path = (
                    root / "records" / f"seed-round-{round_index + 1:02d}.json"
                )
                WORKLOAD.write_json_atomic(
                    record_path,
                    {
                        "name": f"seed-round-{round_index + 1:02d}",
                        "steps_per_window": step_schedule[round_index + 1],
                        "timestep_offset": 0,
                        "worlds": 2,
                        "window_order": list(start_inputs),
                        "ranks_per_window": 1,
                        "start_inputs": start_inputs,
                        "outputs": outputs,
                        "seed_attempt": WORKLOAD.seed_attempt_metadata(
                            self.manifest,
                            round_windows,
                            attempt_index=round_index + 1,
                            step_schedule=step_schedule,
                        ),
                    },
                )
                records.append(record_path)

            def load_record(path: Path, _: object) -> dict[str, object]:
                return json.loads(path.read_text(encoding="utf-8"))

            with mock.patch.object(
                WORKLOAD,
                "validate_invocation_record_current",
                side_effect=load_record,
            ):
                accepted = WORKLOAD.require_seed_records(
                    root,
                    windows,
                    anchor,
                    anchor_ledger=anchor_ledger,
                    manifest=self.manifest,
                    step_schedule=step_schedule,
                    common={"output": root, "ranks_per_window": 1},
                )
                self.assertEqual(set(accepted), {window.tag for window in windows})

                second = json.loads(records[1].read_text(encoding="utf-8"))
                second["steps_per_window"] = 4500
                WORKLOAD.write_json_atomic(records[1], second)
                with self.assertRaisesRegex(ValueError, "seed round protocol mismatch"):
                    WORKLOAD.require_seed_records(
                        root,
                        windows,
                        anchor,
                        anchor_ledger=anchor_ledger,
                        manifest=self.manifest,
                        step_schedule=step_schedule,
                        common={"output": root, "ranks_per_window": 1},
                    )
                second["steps_per_window"] = step_schedule[2]
                second["seed_attempt"]["scheduled_steps"] = 4500
                WORKLOAD.write_json_atomic(records[1], second)
                with self.assertRaisesRegex(
                    ValueError, "seed attempt protocol mismatch"
                ):
                    WORKLOAD.require_seed_records(
                        root,
                        windows,
                        anchor,
                        anchor_ledger=anchor_ledger,
                        manifest=self.manifest,
                        step_schedule=step_schedule,
                        common={"output": root, "ranks_per_window": 1},
                    )
                second["seed_attempt"] = WORKLOAD.seed_attempt_metadata(
                    self.manifest,
                    [lower[1], upper[1]],
                    attempt_index=2,
                    step_schedule=step_schedule,
                )
                WORKLOAD.write_json_atomic(records[1], second)

                first = json.loads(records[0].read_text(encoding="utf-8"))
                rerun_window = lower[0]
                rerun_state = WORKLOAD.state_output(root, "seeds", rerun_window)
                rerun_state.write_bytes(b"rerun parent bytes\n")
                first["outputs"][rerun_window.tag]["data"] = identity(rerun_state)
                WORKLOAD.write_json_atomic(records[0], first)
                with self.assertRaisesRegex(ValueError, "seed branch parent changed"):
                    WORKLOAD.require_seed_records(
                        root,
                        windows,
                        anchor,
                        anchor_ledger=anchor_ledger,
                        manifest=self.manifest,
                        step_schedule=step_schedule,
                        common={"output": root, "ranks_per_window": 1},
                    )

    def test_downstream_checkpoint_requires_unchanged_accepted_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-checkpoint-") as temporary:
            root = Path(temporary)
            state = root / "m1p5.data"
            state.write_bytes(b"accepted state\n")
            record = root / "anchor.json"
            record.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "wall_seconds": 2.0,
                        "outputs": {
                            "m1p5": {
                                "data": {
                                    "path": str(state),
                                    "sha256": WORKLOAD.sha256(state),
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            WORKLOAD.require_recorded_output(record, "m1p5", state)
            state.write_bytes(b"changed state\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                WORKLOAD.require_recorded_output(record, "m1p5", state)

    def test_resume_is_invalidated_by_changed_parent_or_missing_output(self) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[16]
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-resume-") as temporary:
            root = Path(temporary)
            files = {}
            for name in (
                "start.data",
                "final.data",
                "final.restart",
                "window.colvars.traj",
                "input.in",
                "lmp",
                "plugin.so",
                "libxtbloom.so",
                "mpiexec",
                "runner.py",
                "manifest.json",
                "provenance.json",
                "launcher.log",
                "log.lammps",
            ):
                path = root / name
                path.write_bytes(f"{name}\n".encode())
                files[name] = path

            run_window = WORKLOAD.RunWindow(
                window,
                files["start.data"],
                root,
                root,
                1234,
            )
            run_window.colvars_config.parent.mkdir(parents=True, exist_ok=True)
            run_window.colvars_config.write_bytes(b"colvars config\n")
            # Match RunWindow's deterministic output names.
            run_window.final_data.write_bytes(files["final.data"].read_bytes())
            run_window.final_restart.write_bytes(files["final.restart"].read_bytes())
            colvars = Path(str(run_window.colvars_prefix) + ".colvars.traj")
            colvars.write_bytes(files["window.colvars.traj"].read_bytes())

            def identity(path: Path) -> dict[str, str]:
                return {"path": str(path), "sha256": WORKLOAD.sha256(path)}

            environment = {"CUDA_VISIBLE_DEVICES": "0"}
            environment_contract = WORKLOAD.runtime_environment_contract(
                environment
            )
            record = root / "record.json"
            record.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "wall_seconds": 2.0,
                        "steps_per_window": 10,
                        "timestep_offset": 0,
                        "worlds": 1,
                        "window_order": [window.tag],
                        "ranks_per_window": 1,
                        "execution": WORKLOAD.execution_record(
                            mode="qmmm",
                            model_deviation_frequency=0,
                            dpa4c_models_qualified=False,
                            allow_unqualified_dpa4c_models=False,
                        ),
                        "selected_environment": environment,
                        "environment_contract": environment_contract,
                        "input": identity(files["input.in"]),
                        "colvars_configs": {
                            window.tag: identity(run_window.colvars_config)
                        },
                        "runtime": {
                            "lammps": identity(files["lmp"]),
                            "plugin": identity(files["plugin.so"]),
                            "xtbloom": identity(files["libxtbloom.so"]),
                            "mpiexec": identity(files["mpiexec"]),
                        },
                        "project": {
                            "runner": identity(files["runner.py"]),
                            "manifest": identity(files["manifest.json"]),
                            "provenance": identity(files["provenance.json"]),
                        },
                        "loaded_xtbloom": {
                            "soname": "libxtbloom.so.0",
                            "resolved_path": str(files["libxtbloom.so"].resolve()),
                            "sha256": WORKLOAD.sha256(files["libxtbloom.so"]),
                        },
                        "launcher_log": identity(files["launcher.log"]),
                        "lammps_logs": {"log.lammps": identity(files["log.lammps"])},
                        "dangerous_builds": {"log.lammps": 0},
                        "start_inputs": {window.tag: identity(files["start.data"])},
                        "outputs": {
                            window.tag: {
                                "data": identity(run_window.final_data),
                                "restart": identity(run_window.final_restart),
                                "colvars": identity(colvars),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            arguments = {
                "input_path": files["input.in"],
                "steps": 10,
                "timestep_offset": 0,
                "trajectory_frequency": 0,
                "ranks_per_window": 1,
                "lammps": files["lmp"],
                "plugin": files["plugin.so"],
                "xtbloom_library": files["libxtbloom.so"],
                "mpiexec": files["mpiexec"],
                "runner_path": files["runner.py"],
                "loaded_xtbloom": {
                    "soname": "libxtbloom.so.0",
                    "resolved_path": str(files["libxtbloom.so"].resolve()),
                    "sha256": WORKLOAD.sha256(files["libxtbloom.so"]),
                },
                "plugin_cmake_cache": None,
                "selected_environment": environment,
                "manifest_path": files["manifest.json"],
                "provenance_path": files["provenance.json"],
                "lammps_execution_backend": "kokkos",
            }
            self.assertTrue(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            arguments["selected_environment"] = {"CUDA_VISIBLE_DEVICES": "3"}
            self.assertTrue(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            arguments["selected_environment"] = {
                "CUDA_VISIBLE_DEVICES": "3",
                "OMP_NUM_THREADS": "2",
            }
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            arguments["selected_environment"] = environment
            arguments["lammps_execution_backend"] = "host"
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            arguments["lammps_execution_backend"] = "kokkos"
            extra_log = root / "log.lammps.extra"
            extra_log.write_bytes(b"stale partition log\n")
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            extra_log.unlink()
            files["plugin.so"].write_bytes(b"changed plugin\n")
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            files["plugin.so"].write_bytes(b"plugin.so\n")
            files["manifest.json"].write_bytes(b"changed manifest\n")
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            files["manifest.json"].write_bytes(b"manifest.json\n")
            run_window.colvars_config.write_bytes(b"changed colvars config\n")
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            run_window.colvars_config.write_bytes(b"colvars config\n")
            files["start.data"].write_bytes(b"changed parent\n")
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
            files["start.data"].write_bytes(b"start.data\n")
            run_window.final_restart.unlink()
            self.assertFalse(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )

    def test_workspace_lock_rejects_concurrent_launcher(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-lock-") as temporary:
            root = Path(temporary)
            with (
                WORKLOAD.WorkspaceLock(root),
                self.assertRaisesRegex(ValueError, "already locked"),
                WORKLOAD.WorkspaceLock(root),
            ):
                self.fail("a second launcher acquired the same workspace")
            self.assertFalse((root / ".etpeth-run.lock").exists())

    def test_workspace_lock_recovers_only_proven_stale_owner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-stale-") as temporary:
            root = Path(temporary)
            lock = root / ".etpeth-run.lock"
            lock.write_text(
                json.dumps(
                    {
                        "pid": 99_999_999,
                        "host": WORKLOAD.socket.gethostname(),
                        "process_start_ticks": 1,
                        "started_utc": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            with WORKLOAD.WorkspaceLock(root, recover_stale=True):
                self.assertTrue(lock.exists())
            self.assertEqual(len(list(root.glob(".etpeth-run.lock.stale-*"))), 1)


if __name__ == "__main__":
    unittest.main()
