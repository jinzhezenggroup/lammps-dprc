#!/usr/bin/env python3
"""Focused tests for fail-closed LAMMPS/xTBloom label publication."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_xtbloom_label", ROOT / "tools/publish_xtbloom_label.py"
)
assert SPEC is not None and SPEC.loader is not None
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


class PublishXTBloomLabelTest(unittest.TestCase):
    def fixture(self, root: Path) -> dict[str, Path]:
        high_path = root / "high.bin"
        high = PUBLISH.IO.Label(
            frame_index=1,
            extra_point_count=1,
            virtual_site_policy=PUBLISH.IO.TIP4P_REDISTRIBUTED_POLICY,
            total_potential_energy_kcal_mol=100.0,
            qmmm_scf_energy_kcal_mol=70.0,
            coordinates_angstrom=np.asarray(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.8, 5.6, 6.0],
                 [3.2, 5.6, 6.0], [4.0, 5.125, 6.0]],
                dtype=np.float64,
            ),
            forces_kcal_mol_angstrom=np.asarray(
                [[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0], [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            cell_lengths_angstrom=np.asarray([10.0, 10.0, 10.0]),
            cell_angles_degrees=np.asarray([90.0, 90.0, 90.0]),
        )
        PUBLISH.write_label(high_path, high)
        baseline_path = root / "baseline.bin"
        baseline = high._replace(
            total_potential_energy_kcal_mol=30.0,
            qmmm_scf_energy_kcal_mol=0.0,
            forces_kcal_mol_angstrom=np.asarray(
                [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        )
        PUBLISH.write_label(baseline_path, baseline)
        qmmm = root / "qmmm.txt"
        qmmm.write_text(
            "energy=20 correction=-5 elong=-20 pair_reference_mm_only=1 "
            "lx=10 ly=10 lz=10 xy=0 xz=0 yz=0\n",
            encoding="utf-8",
        )
        mm = root / "mm.txt"
        mm.write_text("mm_elong=-15\n", encoding="utf-8")
        dump = root / "state.dump"
        dump.write_text(
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n4\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type q xu yu zu fx fy fz "
            "f_pre_xtb[1] f_pre_xtb[2] f_pre_xtb[3]\n"
            "1 1 0 1 2 3 4 5 6 1 1 1\n"
            "2 6 0 4 5 6 -4 -5 -6 -1 -1 -1\n"
            "3 7 0 4.8 5.6 6 0 0 0 0 0 0\n"
            "4 7 0 3.2 5.6 6 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
        atom_map = root / "atom_map.tsv"
        mm_dump = root / "mm.dump"
        mm_dump.write_text(
            dump.read_text().split("ITEM: ATOMS", 1)[0]
            + "ITEM: ATOMS id type q xu yu zu fx fy fz\n"
            + "1 1 0 1 2 3 1 1 1\n2 6 0 4 5 6 -1 -1 -1\n"
            + "3 7 0 4.8 5.6 6 0 0 0\n4 7 0 3.2 5.6 6 0 0 0\n",
            encoding="utf-8",
        )
        atom_map.write_text(
            "amber_id\tlammps_id\tmolecule_id\tresidue\tatom\ttype\n"
            "1\t1\t1\tQM\tP\tp5\n"
            "2\t2\t2\tWAT\tO\tOW\n"
            "3\t3\t2\tWAT\tH1\tHW\n4\t4\t2\tWAT\tH2\tHW\n",
            encoding="utf-8",
        )
        return {
            "high": high_path,
            "baseline": baseline_path,
            "qmmm": qmmm,
            "mm": mm,
            "dump": dump,
            "mm_dump": mm_dump,
            "map": atom_map,
        }

    def test_publication_reconstructs_energy_and_zero_extra_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            output = root / "low.bin"
            report = PUBLISH.publish(
                paths["high"],
                paths["baseline"],
                paths["qmmm"],
                paths["mm"],
                paths["dump"],
                paths["map"],
                output,
                root / "manifest.json",
                coordinate_tolerance_angstrom=1.0e-12,
                baseline_energy_tolerance_kcal_mol=1.0e-12,
                mm_dump_path=paths["mm_dump"],
            )
            low = PUBLISH.IO.read_label(output)
            self.assertEqual(low.qmmm_scf_energy_kcal_mol, -10.0)
            self.assertEqual(low.total_potential_energy_kcal_mol, 20.0)
            np.testing.assert_array_equal(
                low.forces_kcal_mol_angstrom[:2],
                np.asarray([[4.0, 6.0, 8.0], [-4.0, -6.0, -8.0]]),
            )
            np.testing.assert_array_equal(low.forces_kcal_mol_angstrom[4], 0.0)
            self.assertTrue(report["checks"]["sander_tip4p_whole_water_geometry_validated"])
            self.assertFalse(
                report["force_assembly"]["cross_engine_total_force_subtraction_used"]
            )
            self.assertIn("F_LAMMPS_zero_QM", report["force_assembly"]["formula"])
            self.assertNotIn("pre_xTBloom", report["force_assembly"]["formula"])

    def test_changed_periodic_cell_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            paths["qmmm"].write_text(paths["qmmm"].read_text().replace("lx=10", "lx=11"))
            with self.assertRaisesRegex(ValueError, "periodic cells"):
                PUBLISH.publish(paths["high"], paths["baseline"], paths["qmmm"], paths["mm"],
                    paths["dump"], paths["map"], root / "low.bin", root / "manifest.json",
                    mm_dump_path=paths["mm_dump"], coordinate_tolerance_angstrom=1e-12,
                    baseline_energy_tolerance_kcal_mol=1e-12)
            self.assertFalse((root / "low.bin").exists())

    def test_sander_baseline_energy_mismatch_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            baseline = PUBLISH.IO.read_label(paths["baseline"])._replace(
                total_potential_energy_kcal_mol=30.01
            )
            paths["baseline"].unlink()
            PUBLISH.write_label(paths["baseline"], baseline)
            with self.assertRaisesRegex(ValueError, "baseline total"):
                PUBLISH.publish(
                    paths["high"], paths["baseline"], paths["qmmm"], paths["mm"],
                    paths["dump"], paths["map"],
                    root / "low.bin", root / "manifest.json",
                    coordinate_tolerance_angstrom=1.0e-12,
                    baseline_energy_tolerance_kcal_mol=1.0e-12,
                    mm_dump_path=paths["mm_dump"],
                )

    def test_matching_but_split_or_misplaced_water_labels_are_rejected(self) -> None:
        """Identical bad labels must not pass merely because MM cancels."""
        for index, delta, message in ((2, 10., "direct O--H"), (4, 0.1, "extra-point position")):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root)
                for kind in ("high", "baseline"):
                    label = PUBLISH.IO.read_label(paths[kind])
                    label.coordinates_angstrom[index, 0] += delta
                    paths[kind].unlink()
                    PUBLISH.write_label(paths[kind], label)
                with self.assertRaisesRegex(ValueError, message):
                    PUBLISH.publish(paths["high"], paths["baseline"], paths["qmmm"],
                                    paths["mm"], paths["dump"], paths["map"], root / "low.bin",
                                    root / "manifest.json", mm_dump_path=paths["mm_dump"],
                                    coordinate_tolerance_angstrom=1e-12,
                                    baseline_energy_tolerance_kcal_mol=1e-12)
                self.assertFalse((root / "low.bin").exists())

    def test_one_ulp_baseline_roundoff_is_accepted_and_reported(self) -> None:
        """A representable one-ULP assembly difference is not physical drift."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            high = PUBLISH.IO.read_label(paths["high"])._replace(
                total_potential_energy_kcal_mol=433_415_698.051293,
                qmmm_scf_energy_kcal_mol=-524_534.518807143,
            )
            baseline = PUBLISH.IO.read_label(paths["baseline"])._replace(
                total_potential_energy_kcal_mol=433_940_232.5701001,
            )

            high_classical, residual, effective, roundoff = (
                PUBLISH._require_matching_baseline(
                    high,
                    baseline,
                    energy_tolerance_kcal_mol=1.0e-8,
                )
            )

            self.assertEqual(high_classical, 433_940_232.5701002)
            self.assertEqual(residual, -5.960464477539063e-08)
            self.assertEqual(roundoff, 5.960464477539063e-08)
            self.assertEqual(effective, roundoff)

    def test_more_than_one_ulp_baseline_mismatch_is_rejected(self) -> None:
        """The scale-aware gate must remain fail-closed beyond roundoff."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            high = PUBLISH.IO.read_label(paths["high"])._replace(
                total_potential_energy_kcal_mol=433_415_698.051293,
                qmmm_scf_energy_kcal_mol=-524_534.518807143,
            )
            baseline = PUBLISH.IO.read_label(paths["baseline"])._replace(
                total_potential_energy_kcal_mol=433_940_232.5701000,
            )

            with self.assertRaisesRegex(ValueError, "effective tolerance"):
                PUBLISH._require_matching_baseline(
                    high,
                    baseline,
                    energy_tolerance_kcal_mol=1.0e-8,
                )

    def test_raw_cross_engine_classical_mismatch_is_only_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            paths["qmmm"].write_text(
                "energy=21 correction=-5 elong=-20 pair_reference_mm_only=1 "
                "lx=10 ly=10 lz=10 xy=0 xz=0 yz=0\n",
                encoding="utf-8",
            )
            report = PUBLISH.publish(
                paths["high"], paths["baseline"], paths["qmmm"], paths["mm"],
                paths["dump"], paths["map"],
                root / "low.bin", root / "manifest.json",
                coordinate_tolerance_angstrom=1.0e-12,
                baseline_energy_tolerance_kcal_mol=1.0e-12,
                mm_dump_path=paths["mm_dump"],
            )
            low = PUBLISH.IO.read_label(root / "low.bin")
            self.assertEqual(low.total_potential_energy_kcal_mol, 20.0)
            self.assertEqual(
                report["raw_cross_engine_classical_energy_residual_kcal_mol"], -1.0
            )

    def test_legacy_dump_without_pre_xtb_forces_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            text = paths["dump"].read_text(encoding="utf-8")
            text = text.replace(
                " fx fy fz f_pre_xtb[1] f_pre_xtb[2] f_pre_xtb[3]", " fx fy fz"
            ).replace(" 4 5 6 1 1 1", " 4 5 6").replace(
                " -4 -5 -6 -1 -1 -1", " -4 -5 -6"
            )
            paths["dump"].write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "component dump columns"):
                PUBLISH.publish(
                    paths["high"], paths["baseline"], paths["qmmm"], paths["mm"],
                    paths["dump"], paths["map"], root / "low.bin",
                    root / "manifest.json", coordinate_tolerance_angstrom=1.0e-12,
                    baseline_energy_tolerance_kcal_mol=1.0e-12,
                    mm_dump_path=paths["mm_dump"],
                )

    def test_reciprocal_force_is_not_removed_with_prefixed_snapshot(self) -> None:
        """POST_FORCE already contains QM-dependent PPPM: never subtract it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.fixture(root)
            paths["mm_dump"].write_text(paths["mm_dump"].read_text().replace(
                "1 1 0 1 2 3 1 1 1", "1 1 0 1 2 3 0.25 0.5 0.75"
            ))
            PUBLISH.publish(
                paths["high"], paths["baseline"], paths["qmmm"], paths["mm"],
                paths["dump"], paths["map"], root / "low.bin", root / "manifest.json",
                mm_dump_path=paths["mm_dump"], coordinate_tolerance_angstrom=1e-12,
                baseline_energy_tolerance_kcal_mol=1e-12,
            )
            low = PUBLISH.IO.read_label(root / "low.bin")
            np.testing.assert_array_equal(low.forces_kcal_mol_angstrom[0],
                                          [4.75, 6.5, 8.25])
            self.assertEqual(low.qmmm_scf_energy_kcal_mol, -10.0)

    def test_mismatched_zero_qm_baseline_is_rejected_without_output(self) -> None:
        """A nearby frame, different cell, or charged baseline is not a peer."""
        for old, new, message in (
            ("1 1 0 1 2 3", "1 1 0 1.01 2 3", "coordinates differ"),
            ("1 1 0 1 2 3", "1 1 0.1 1 2 3", "nonzero QM charges"),
            ("2 6 0 4 5 6", "2 6 0.1 4 5 6", "changes MM charges"),
            ("0 10", "0 11", "timestep or cell differs"),
            ("TIMESTEP\n0", "TIMESTEP\n1", "timestep or cell differs"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.fixture(root)
                paths["mm_dump"].write_text(paths["mm_dump"].read_text().replace(old, new))
                with self.assertRaisesRegex(ValueError, message):
                    PUBLISH.publish(
                        paths["high"], paths["baseline"], paths["qmmm"], paths["mm"],
                        paths["dump"], paths["map"], root / "low.bin", root / "manifest.json",
                        mm_dump_path=paths["mm_dump"], coordinate_tolerance_angstrom=1e-12,
                        baseline_energy_tolerance_kcal_mol=1e-12,
                    )
                self.assertFalse((root / "low.bin").exists())
                self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
