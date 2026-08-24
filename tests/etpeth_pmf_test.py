#!/usr/bin/env python3
"""Independent synthetic and provenance tests for ETP/ETH PMF analysis."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_etpeth_pmf", ROOT / "tools/analyze_etpeth_pmf.py"
)
assert SPEC is not None and SPEC.loader is not None
PMF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PMF
SPEC.loader.exec_module(PMF)


class ETPETHPMFTest(unittest.TestCase):
    @staticmethod
    def synthetic_manifest(production_trials: int = 2) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "umbrella": {
                "start_tenths_angstrom": -10,
                "stop_tenths_angstrom": 10,
                "step_tenths_angstrom": 10,
                "count": 3,
                "reaction_coordinate": {"force_constant_kcal_mol_angstrom2": 5.0},
            },
            "dynamics": {
                "temperature_kelvin": 298.0,
                "colvars_frequency_steps": 25,
                "colvars_checkpoint_boundary_tolerance": {
                    "reaction_coordinate_angstrom": 1.0e-12,
                    "attack_angle_degree": 1.0e-10,
                },
            },
            "protocol": {
                "production_trials": production_trials,
                "equilibration_steps_per_window": 2500,
                "production_steps_per_window": 2500,
                "overlap_acceptance": {
                    "minimum_adjacent_overlap_coefficient": 0.01,
                    "minimum_effective_samples_per_window": 1.0,
                },
            },
        }

    @staticmethod
    def synthetic_contract() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "estimator": {
                "name": "histogram-wham",
                "temperature_kelvin": 298.0,
                "reaction_coordinate_unit": "angstrom",
                "energy_unit": "kcal/mol",
                "bin_start_angstrom": -4.0,
                "bin_stop_angstrom": 4.0,
                "bin_width_angstrom": 0.1,
                "maximum_iterations": 10000,
                "dimensionless_tolerance": 1.0e-9,
                "pmf_zero": "global-observed-minimum",
                "trial_combination": (
                    "pooled-counts-with-trial-separated-correlation-blocks"
                ),
            },
            "sampling": {
                "burn_in_samples_per_window": 0,
                "autocorrelation": "geyer-initial-monotone-positive-sequence",
                "effective_sample_definition": (
                    "sum-over-trials(N/max(g_reaction,g_angle))"
                ),
                "chunk_boundary_policy": (
                    "deduplicate-roundtrip-equivalent-absolute-timestep"
                ),
            },
            "uncertainty": {
                "method": "nonoverlapping-circular-block-bootstrap",
                "replicates": 4,
                "confidence_level": 0.8,
                "minimum_finite_fraction_per_observed_bin": 0.0,
                "random_seed": 17239,
                "block_length_samples": (
                    "ceil(maximum-statistical-inefficiency-per-trial-window)"
                ),
            },
            "overlap": {
                "coefficient": (
                    "bhattacharyya-coefficient-of-observed-window-histograms"
                ),
                "use_common_bins": True,
            },
            "trial_consistency": {
                "alignment": "minimum-pooled-pmf-bin-with-common-trial-support",
                "maximum_pmf_region_kcal_mol": 20.0,
                "maximum_pairwise_absolute_difference_kcal_mol": 100.0,
            },
        }

    def write_trial(
        self,
        root: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        trial: int,
        reaction_by_window: dict[str, list[float]],
        *,
        source_qualification: str,
    ) -> Path:
        windows = PMF.windows_from_manifest(manifest)
        order = [window.tag for window in windows]
        frequency = int(manifest["dynamics"]["colvars_frequency_steps"])
        total_steps = int(manifest["protocol"]["production_steps_per_window"])
        expected_samples = total_steps // frequency + 1

        def artifact(name: str, contents: str = "fixture\n") -> dict[str, str]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(contents, encoding="utf-8")
            return PMF.identity(path)

        source_artifact = artifact("source/input.data")
        runner = artifact("project/tools/etpeth_workload.py", "# fixture runner\n")
        provenance_path = root / "provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "source": {
                        "qualification": source_qualification,
                        "artifacts": [source_artifact],
                    },
                    "workload_manifest": PMF.identity(manifest_path),
                    "window_order": [
                        {"tag": window.tag, "center_angstrom": window.center}
                        for window in windows
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime = {
            "lammps": artifact("runtime/lmp"),
            "plugin": artifact("runtime/dprcplugin.so"),
            "xtbloom": artifact("runtime/libxtbloom.so"),
            "mpiexec": artifact("runtime/mpiexec"),
            "plugin_cmake_cache": artifact("runtime/CMakeCache.txt"),
        }
        equilibration_ledger = root / "records/equilibrate-complete.json"
        if not equilibration_ledger.is_file():
            equilibration_stage = "equilibrate"
            equilibration_name = "equilibrate-chunk-001-of-001"
            equilibration_steps = int(
                manifest["protocol"]["equilibration_steps_per_window"]
            )
            equilibration_starts = {
                window.tag: artifact(f"seed-starts/{window.tag}.data")
                for window in windows
            }
            equilibration_outputs = {}
            equilibration_colvars = {}
            equilibration_trajectories = {}
            for window in windows:
                colvars_identity = artifact(
                    f"{equilibration_stage}/{window.tag}.colvars.traj"
                )
                trajectory_identity = artifact(
                    f"{equilibration_stage}/{window.tag}.lammpstrj"
                )
                equilibration_outputs[window.tag] = {
                    "data": artifact(f"{equilibration_stage}/{window.tag}.data"),
                    "restart": artifact(f"{equilibration_stage}/{window.tag}.restart"),
                    "colvars": colvars_identity,
                    "trajectory": trajectory_identity,
                }
                equilibration_colvars[window.tag] = [colvars_identity]
                equilibration_trajectories[window.tag] = [trajectory_identity]
            equilibration_logs = {
                f"log.lammps.{index}": artifact(
                    f"logs/{equilibration_name}/log.lammps.{index}"
                )
                for index, _window in enumerate(windows)
            }
            equilibration_record = root / "records" / f"{equilibration_name}.json"
            equilibration_record.parent.mkdir(parents=True, exist_ok=True)
            equilibration_record.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "returncode": 0,
                        "name": equilibration_name,
                        "timestep_offset": 0,
                        "steps_per_window": equilibration_steps,
                        "worlds": len(windows),
                        "ranks_per_window": 1,
                        "window_order": order,
                        "start_inputs": equilibration_starts,
                        "outputs": equilibration_outputs,
                        "runtime": runtime,
                        "loaded_xtbloom": {
                            "resolved_path": runtime["xtbloom"]["path"],
                            "sha256": runtime["xtbloom"]["sha256"],
                        },
                        "input": artifact(f"generated/{equilibration_name}.in"),
                        "launcher_log": artifact(
                            f"logs/{equilibration_name}/launcher.log"
                        ),
                        "lammps_logs": equilibration_logs,
                        "dangerous_builds": {name: 0 for name in equilibration_logs},
                        "project": {
                            "dirty": False,
                            "qualification": "clean-source",
                            "runner": runner,
                            "manifest": PMF.identity(manifest_path),
                            "provenance": PMF.identity(provenance_path),
                            "dependencies": {
                                "lammps": {"dirty": False},
                                "xtbloom": {"dirty": False},
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            equilibration_ledger.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": equilibration_stage,
                        "status": "passed",
                        "qualification": "native-chunked",
                        "window_order": order,
                        "total_steps_per_window": equilibration_steps,
                        "maximum_chunk_steps": equilibration_steps,
                        "chunk_count": 1,
                        "chunks": [
                            {
                                "name": equilibration_name,
                                "timestep_offset": 0,
                                "steps_per_window": equilibration_steps,
                                "record": PMF.identity(equilibration_record),
                            }
                        ],
                        "outputs": equilibration_outputs,
                        "series_merge_policy": PMF.series_merge_policy(manifest),
                        "colvars_by_window": equilibration_colvars,
                        "trajectories_by_window": equilibration_trajectories,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        equilibrated_outputs = PMF.load_json(equilibration_ledger)["outputs"]
        stage = f"production-trial-{trial}"
        name = f"{stage}-chunk-001-of-001"
        outputs = {}
        starts = {}
        colvars_by_window = {}
        for window in windows:
            values = reaction_by_window[window.tag]
            self.assertEqual(len(values), expected_samples)
            colvars = root / f"{stage}/{window.tag}.colvars.traj"
            colvars.parent.mkdir(parents=True, exist_ok=True)
            lines = ["# step reaction_coordinate attack_angle"]
            for sample, reaction in enumerate(values):
                angle = 180.0 + 0.1 * math.sin(sample + trial)
                lines.append(f"{sample * frequency} {reaction:.17g} {angle:.17g}")
            colvars.write_text("\n".join(lines) + "\n", encoding="utf-8")
            colvars_identity = PMF.identity(colvars)
            starts[window.tag] = equilibrated_outputs[window.tag]["data"]
            outputs[window.tag] = {
                "data": artifact(f"{stage}/{window.tag}.data"),
                "restart": artifact(f"{stage}/{window.tag}.restart"),
                "colvars": colvars_identity,
                "trajectory": artifact(f"{stage}/{window.tag}.lammpstrj"),
            }
            colvars_by_window[window.tag] = [colvars_identity]
        lammps_logs = {
            f"log.lammps.{index}": artifact(f"logs/{name}/log.lammps.{index}")
            for index, _window in enumerate(windows)
        }
        record = root / "records" / f"{name}.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "returncode": 0,
                    "name": name,
                    "timestep_offset": 0,
                    "steps_per_window": total_steps,
                    "worlds": len(windows),
                    "ranks_per_window": 1,
                    "window_order": order,
                    "start_inputs": starts,
                    "outputs": outputs,
                    "runtime": runtime,
                    "loaded_xtbloom": {
                        "resolved_path": runtime["xtbloom"]["path"],
                        "sha256": runtime["xtbloom"]["sha256"],
                    },
                    "input": artifact(f"generated/{name}.in"),
                    "launcher_log": artifact(f"logs/{name}/launcher.log"),
                    "lammps_logs": lammps_logs,
                    "dangerous_builds": {name: 0 for name in lammps_logs},
                    "project": {
                        "dirty": False,
                        "qualification": "clean-source",
                        "runner": runner,
                        "manifest": PMF.identity(manifest_path),
                        "provenance": PMF.identity(provenance_path),
                        "dependencies": {
                            "lammps": {"dirty": False},
                            "xtbloom": {"dirty": False},
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ledger = root / "records" / f"{stage}-complete.json"
        ledger.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "status": "passed",
                    "qualification": "native-chunked",
                    "window_order": order,
                    "total_steps_per_window": total_steps,
                    "maximum_chunk_steps": total_steps,
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "name": name,
                            "timestep_offset": 0,
                            "steps_per_window": total_steps,
                            "record": PMF.identity(record),
                        }
                    ],
                    "outputs": outputs,
                    "series_merge_policy": PMF.series_merge_policy(manifest),
                    "colvars_by_window": colvars_by_window,
                    "trajectories_by_window": {
                        tag: [outputs[tag]["trajectory"]] for tag in order
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return ledger

    def test_wham_recovers_known_harmonic_free_energy_differences(self) -> None:
        rng = random.Random(7819301)
        temperature = 298.0
        thermal = PMF.K_B_KCAL_PER_MOL_K * temperature
        physical_curvature = 2.0
        umbrella_curvature = 20.0
        centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
        edges = tuple(-1.6 + 0.04 * index for index in range(81))
        bins = tuple((edges[index] + edges[index + 1]) * 0.5 for index in range(80))
        counts = []
        for center in centers:
            total_curvature = physical_curvature + umbrella_curvature
            mean = umbrella_curvature * center / total_curvature
            sigma = math.sqrt(thermal / total_curvature)
            samples = [rng.gauss(mean, sigma) for _ in range(12000)]
            counts.append(PMF.histogram(samples, edges))
        result = PMF.solve_wham(
            counts,
            centers,
            bins,
            umbrella_curvature,
            temperature,
            1.0e-11,
            100000,
        )

        def nearest(value: float) -> int:
            return min(range(len(bins)), key=lambda index: abs(bins[index] - value))

        zero = result.pmf[nearest(0.0)]
        half = result.pmf[nearest(0.8)]
        assert zero is not None and half is not None
        expected = 0.5 * physical_curvature * 0.8**2
        self.assertAlmostEqual(half - zero, expected, delta=0.18)
        self.assertLessEqual(result.residual, 1.0e-11)

    def test_statistical_inefficiency_distinguishes_independent_and_correlated(
        self,
    ) -> None:
        rng = random.Random(334214467)
        independent = [rng.gauss(0.0, 1.0) for _ in range(4096)]
        correlated = []
        value = 0.0
        for _ in range(4096):
            value = 0.95 * value + rng.gauss(0.0, math.sqrt(1.0 - 0.95**2))
            correlated.append(value)
        independent_g = PMF.statistical_inefficiency(independent)
        correlated_g = PMF.statistical_inefficiency(correlated)
        self.assertLess(independent_g, 2.0)
        self.assertGreater(correlated_g, 15.0)
        self.assertGreater(correlated_g, independent_g)

    def test_chunk_merge_deduplicates_only_identical_boundary(self) -> None:
        left = PMF.Series((0, 25), (-1.0, -0.9), (180.0, 179.0))
        right = PMF.Series((25, 50), (-0.9, -0.8), (179.0, 178.0))
        merged = PMF.merge_series([left, right])
        self.assertEqual(merged.steps, (0, 25, 50))
        rounded = PMF.Series((25, 50), (-0.9 - 5.0e-13, -0.8), (179.0 + 5.0e-11, 178.0))
        rounded_merge = PMF.merge_series([left, rounded], 1.0e-12, 1.0e-10)
        self.assertEqual(rounded_merge.steps, (0, 25, 50))
        changed = PMF.Series((25, 50), (-0.91, -0.8), (179.0, 178.0))
        with self.assertRaisesRegex(ValueError, "boundary values disagree"):
            PMF.merge_series([left, changed])

    def test_colvars_timeline_uses_absolute_frequency_after_unaligned_restart(
        self,
    ) -> None:
        self.assertEqual(
            PMF.expected_colvars_steps(5001, 100, 25),
            (5025, 5050, 5075, 5100),
        )

    def test_block_bootstrap_preserves_exact_sample_count(self) -> None:
        edges = (0.0, 1.0, 2.0, 3.0)
        blocks = PMF.block_histograms([0.1, 0.2, 1.2, 1.3, 2.2], 3, edges)
        for seed in range(20):
            sampled = PMF.resample_histogram_blocks(blocks, random.Random(seed))
            self.assertEqual(sum(sampled), 5)

    def test_trial_ledger_rejects_changed_colvars_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-pmf-ledger-") as temporary:
            root = Path(temporary)
            manifest = self.synthetic_manifest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            windows = PMF.windows_from_manifest(manifest)
            values = {window.tag: [window.center] * 101 for window in windows}
            ledger = self.write_trial(
                root,
                manifest_path,
                manifest,
                0,
                values,
                source_qualification="private-diagnostic",
            )
            equilibrated_outputs = PMF.load_json(
                root / "records/equilibrate-complete.json"
            )["outputs"]
            _stage, _series, qualification = PMF.load_trial_ledger(
                ledger,
                manifest_path,
                manifest,
                windows,
                0,
                equilibrated_outputs,
            )
            self.assertEqual(qualification, "private-diagnostic")
            first = root / f"production-trial-0/{windows[0].tag}.colvars.traj"
            first.write_text(first.read_text(encoding="utf-8") + "2525 0.0 180.0\n")
            with self.assertRaisesRegex(
                ValueError, ".*production-trial-0 .* colvars bytes changed"
            ):
                PMF.load_trial_ledger(
                    ledger,
                    manifest_path,
                    manifest,
                    windows,
                    0,
                    equilibrated_outputs,
                )

    def test_trial_ledger_rejects_changed_bias_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-pmf-manifest-") as temporary:
            root = Path(temporary)
            manifest = self.synthetic_manifest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            windows = PMF.windows_from_manifest(manifest)
            values = {window.tag: [window.center] * 101 for window in windows}
            ledger = self.write_trial(
                root,
                manifest_path,
                manifest,
                0,
                values,
                source_qualification="final",
            )
            equilibrated_outputs = PMF.load_json(
                root / "records/equilibrate-complete.json"
            )["outputs"]
            _stage, _series, qualification = PMF.load_trial_ledger(
                ledger,
                manifest_path,
                manifest,
                windows,
                0,
                equilibrated_outputs,
            )
            self.assertEqual(qualification, "final")
            changed = json.loads(json.dumps(manifest))
            changed["umbrella"]["reaction_coordinate"][
                "force_constant_kcal_mol_angstrom2"
            ] = 7.5
            changed_path = root / "changed-manifest.json"
            changed_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "analysis manifest differs from the production manifest"
            ):
                PMF.load_trial_ledger(
                    ledger,
                    changed_path,
                    changed,
                    windows,
                    0,
                    equilibrated_outputs,
                )

    def test_trial_ledger_rejects_unrelated_equilibration_parent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-pmf-parent-") as temporary:
            root = Path(temporary)
            manifest = self.synthetic_manifest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            windows = PMF.windows_from_manifest(manifest)
            values = {window.tag: [window.center] * 101 for window in windows}
            ledger_path = self.write_trial(
                root,
                manifest_path,
                manifest,
                0,
                values,
                source_qualification="final",
            )
            equilibrated_outputs = PMF.load_json(
                root / "records/equilibrate-complete.json"
            )["outputs"]
            record_path = root / "records/production-trial-0-chunk-001-of-001.json"
            record = PMF.load_json(record_path)
            rogue = root / "rogue-start.data"
            rogue.write_text("unrelated checkpoint\n", encoding="utf-8")
            record["start_inputs"][windows[0].tag] = PMF.identity(rogue)
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            ledger = PMF.load_json(ledger_path)
            ledger["chunks"][0]["record"] = PMF.identity(record_path)
            ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint parent changed"):
                PMF.load_trial_ledger(
                    ledger_path,
                    manifest_path,
                    manifest,
                    windows,
                    0,
                    equilibrated_outputs,
                )

    def test_end_to_end_analysis_writes_hash_linked_pmf_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-pmf-e2e-") as temporary:
            root = Path(temporary)
            manifest = self.synthetic_manifest()
            contract = self.synthetic_contract()
            manifest_path = root / "manifest.json"
            contract_path = root / "analysis.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            windows = PMF.windows_from_manifest(manifest)
            ledger_paths = []
            for trial in range(2):
                reaction_by_window = {
                    window.tag: [
                        -3.95 + 0.1 * ((sample + trial) % 80) for sample in range(101)
                    ]
                    for window in windows
                }
                ledger_paths.append(
                    self.write_trial(
                        root,
                        manifest_path,
                        manifest,
                        trial,
                        reaction_by_window,
                        source_qualification="final",
                    )
                )

            prefix = root / "analysis/pmf"
            result = PMF.analyze(manifest_path, contract_path, ledger_paths, prefix)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["input_qualification"], "final")
            self.assertEqual(
                result["qualification"],
                "private-diagnostic"
                if result["analysis_project"]["dirty"]
                else "final",
            )
            self.assertEqual(
                result["analysis_project"]["analyzer"]["sha256"],
                PMF.sha256(ROOT / "tools/analyze_etpeth_pmf.py"),
            )
            self.assertTrue(result["acceptance"]["passed"])
            self.assertTrue(prefix.with_suffix(".json").is_file())
            self.assertTrue(prefix.with_suffix(".csv").is_file())
            self.assertEqual(
                result["pmf_csv"]["sha256"], PMF.sha256(prefix.with_suffix(".csv"))
            )
            with self.assertRaisesRegex(ValueError, "trial set/order changed"):
                PMF.analyze(
                    manifest_path,
                    contract_path,
                    list(reversed(ledger_paths)),
                    root / "analysis/reversed",
                )

    def test_trial_consistency_rejects_missing_pooled_core_basin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-pmf-core-") as temporary:
            root = Path(temporary)
            manifest = self.synthetic_manifest()
            contract = self.synthetic_contract()
            manifest_path = root / "manifest.json"
            contract_path = root / "analysis.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            windows = PMF.windows_from_manifest(manifest)
            trial_values = (
                [-1.05] * 90 + [0.05] * 11,
                [1.05] * 90 + [0.05] * 11,
            )
            ledgers = []
            for trial, values in enumerate(trial_values):
                ledgers.append(
                    self.write_trial(
                        root,
                        manifest_path,
                        manifest,
                        trial,
                        {window.tag: values for window in windows},
                        source_qualification="final",
                    )
                )
            result = PMF.analyze(
                manifest_path,
                contract_path,
                ledgers,
                root / "analysis/missing-core",
            )
            self.assertEqual(result["status"], "acceptance-failed")
            self.assertFalse(
                result["acceptance"]["trial_consistency_core_support_passed"]
            )
            self.assertFalse(result["acceptance"]["trial_consistency_passed"])
            self.assertTrue(
                result["acceptance"]["trial_consistency_missing_core_support"]
            )

    def test_contract_rejects_claimed_units_not_implemented(self) -> None:
        manifest = self.synthetic_manifest()
        contract = self.synthetic_contract()
        contract["estimator"]["energy_unit"] = "kJ/mol"
        with self.assertRaisesRegex(ValueError, "unsupported estimator.energy_unit"):
            PMF.validate_analysis_contract(manifest, contract)


if __name__ == "__main__":
    unittest.main()
