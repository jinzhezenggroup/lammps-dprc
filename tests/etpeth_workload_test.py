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

    def test_exact_grid_and_stable_tags(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        self.assertEqual(len(windows), 48)
        self.assertEqual((windows[0].tag, windows[0].center), ("m3p1", -3.1))
        self.assertEqual((windows[16].tag, windows[16].center), ("m1p5", -1.5))
        self.assertEqual((windows[-1].tag, windows[-1].center), ("p1p6", 1.6))

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

    def test_rendered_batch_has_unique_world_state_and_private_style(self) -> None:
        windows = WORKLOAD.windows_from_manifest(self.manifest)
        with tempfile.TemporaryDirectory(prefix="dprc-etpeth-render-") as temporary:
            root = Path(temporary)
            tutorial = root / "tutorial"
            (tutorial / "lammps").mkdir(parents=True)
            (tutorial / "lammps/forcefield_qmmm_hybrid.inc").write_text(
                "# fixture\n", encoding="utf-8"
            )
            plugin = root / "dprcplugin.so"
            plugin.write_bytes(b"plugin")
            run_windows = [
                WORKLOAD.RunWindow(
                    windows[index],
                    root / f"start-{index}.data",
                    root / "run" / "states" / f"window-{index}",
                    root / "run",
                    1000 + index,
                )
                for index in (0, 1)
            ]
            text = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                timestep_offset=5000,
                trajectory_frequency=0,
            )
            self.assertIn("variable start_data world &", text)
            self.assertIn("reset_timestep 5000", text)
            self.assertIn("fix qmmm qm qmmm/xtb/dprc", text)
            self.assertIn("pair_style hybrid/overlay/kk lj/cut/dprc/batch", text)
            self.assertIn("tip4p/long/dprc/batch", text)
            self.assertIn("kspace_style pppm/tip4p/dprc/batch", text)
            self.assertNotIn("fix qmmm qm qmmm/xtb elements", text)
            self.assertNotIn("kspace_style pppm/tip4p/xtb", text)
            generated_forcefield = root / "run/generated/forcefield_dprc_batch.inc"
            self.assertTrue(generated_forcefield.is_file())
            self.assertEqual(text.count("write_data ${final_data} nocoeff"), 1)

            classical = WORKLOAD.render_lammps_input(
                self.manifest,
                tutorial,
                plugin,
                run_windows,
                steps=3,
                trajectory_frequency=0,
                mode="classical",
            )
            self.assertIn("fix classical all dprc/classical/batch", classical)
            self.assertIn(
                "pair_style hybrid/overlay/kk lj/cut/dprc/batch", classical
            )
            self.assertIn("pppm/tip4p/dprc/batch", classical)
            self.assertNotIn("fix qmmm", classical)

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
            self.assertEqual(dpa4c.count("plugin load"), 1)
            self.assertNotIn("partition_batch yes", classical)

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
            self.assertIn("pair_style hybrid/overlay lj/cut/dprc/batch", host_qmmm)
            self.assertIn("fix integrate all nve\n", host_qmmm)
            self.assertNotIn("/kk", host_qmmm)

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
            self.assertIn(
                f"dprc/deepmd/batch {dpa4c_model.resolve()} partition_batch yes",
                host_dpa4c,
            )
            self.assertNotIn("/kk", host_dpa4c)

    def test_seed_colvars_profile_is_stronger_but_sampling_contract_is_unchanged(
        self,
    ) -> None:
        window = WORKLOAD.windows_from_manifest(self.manifest)[17]
        sampling = WORKLOAD.render_colvars(
            self.manifest, window, profile="sampling"
        )
        seed = WORKLOAD.render_colvars(self.manifest, window, profile="seed")
        self.assertIn("forceConstant 200.0", sampling)
        self.assertIn("forceConstant 1000.0", seed)
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

    def test_lammps_backend_arguments_omit_kokkos_for_host(self) -> None:
        self.assertEqual(WORKLOAD.lammps_backend_arguments("host"), ())
        kokkos = WORKLOAD.lammps_backend_arguments("kokkos")
        self.assertEqual(kokkos[:4], ("-k", "on", "g", "1"))
        self.assertIn("kokkos", kokkos)
        with self.assertRaisesRegex(ValueError, "execution backend"):
            WORKLOAD.lammps_backend_arguments("invalid")

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
                )
            self.assertEqual(
                [(call["offset"], call["steps"]) for call in calls],
                [(0, 5), (5, 5), (10, 2)],
            )
            self.assertEqual(calls[0]["start"], initial)
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
                for branch, branch_windows in (("lower", lower), ("upper", upper)):
                    window = branch_windows[round_index]
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
                        "steps_per_window": 3000,
                        "timestep_offset": 0,
                        "worlds": 2,
                        "window_order": list(start_inputs),
                        "ranks_per_window": 1,
                        "start_inputs": start_inputs,
                        "outputs": outputs,
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
                    seed_steps=3000,
                    common={"ranks_per_window": 1},
                )
                self.assertEqual(set(accepted), {window.tag for window in windows})

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
                        seed_steps=3000,
                        common={"ranks_per_window": 1},
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
                        "selected_environment": environment,
                        "execution": {
                            "mode": "qmmm",
                            "lammps_execution_backend": "kokkos",
                        },
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
            }
            self.assertTrue(
                WORKLOAD.record_is_resumable(record, [run_window], **arguments)
            )
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
