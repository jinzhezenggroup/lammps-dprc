#!/usr/bin/env python3
"""Focused tests for implicit-TIP4P LAMMPS-to-DPRC frame conversion."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lammps_data_to_dprc_frames", ROOT / "tools/lammps_data_to_dprc_frames.py"
)
assert SPEC is not None and SPEC.loader is not None
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


class LammpsDataToDPRCFramesTest(unittest.TestCase):
    def test_independent_hydrogen_images_produce_whole_identical_water(self) -> None:
        """Image changes must not alter the geometry Sander reconstructs."""
        mapping = [CONVERT.AtomMapRow(i + 1, i + 1, 1, "WAT", atom, kind)
                   for i, (atom, kind) in enumerate((("O", "OW"), ("H1", "HW"), ("H2", "HW")))]
        # Unequal OH lengths also exercise Amber's normalized-bond bisector.
        xyz = np.asarray([[2., 3., 4.], [2.8, 3.6, 4.], [1.24, 3.57, 4.]])
        for cell in (np.eye(3) * 10, np.asarray([[10., -3., -3.], [0., 9., -4.], [0., 0., 8.]])):
            reference = None
            for images in (np.zeros((3, 3)), np.asarray([[0, 0, 0], [1, -2, 1], [-2, 1, 0]])):
                shifted = xyz + images @ cell.T
                data = CONVERT.LammpsData(dict(enumerate(shifted, 1)),
                                          {1: 1, 2: 1, 3: 1}, {1: 6, 2: 7, 3: 7}, cell)
                frame = CONVERT.convert_frame(data, mapping, tip4p_om_distance_angstrom=0.125)
                if reference is None:
                    reference = frame.coordinates_angstrom
                np.testing.assert_allclose(frame.coordinates_angstrom, reference, atol=1e-13)
                np.testing.assert_allclose(frame.coordinates_angstrom[:3], xyz, atol=1e-13)
                np.testing.assert_array_equal(np.asarray(list(data.coordinates_by_id.values())), shifted)
                self.assertLess(np.linalg.norm(frame.coordinates_angstrom[1] - frame.coordinates_angstrom[0]), 1.1)

    def test_geometry_guard_rejects_split_parents_and_wrong_ep(self) -> None:
        xyz = np.asarray([[0., 0., 0.], [0.8, 0.6, 0.], [-0.8, 0.6, 0.], [0., 0.125, 0.]])
        CONVERT.validate_tip4p_sites(xyz, [(0, 1, 2, 3)])
        for index, displacement, message in ((1, 10., "direct O--H"), (3, 0.02, "extra-point position")):
            wrong = xyz.copy()
            wrong[index, 0] += displacement
            with self.assertRaisesRegex(ValueError, message):
                CONVERT.validate_tip4p_sites(wrong, [(0, 1, 2, 3)])

    def test_triclinic_images_and_tip4p_slot_are_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atom_map = root / "atom_map.tsv"
            atom_map.write_text(
                "amber_id\tlammps_id\tmolecule_id\tresidue\tatom\ttype\n"
                "1\t1\t1\tQM\tA\tP\n"
                "2\t2\t2\tWAT\tO\tOW\n"
                "3\t3\t2\tWAT\tH1\tHW\n"
                "4\t4\t2\tWAT\tH2\tHW\n",
                encoding="utf-8",
            )
            data_path = root / "frame.data"
            data_path.write_text(
                "fixture\n\n4 atoms\n\n2 atom types\n\n"
                "0 10 xlo xhi\n0 9 ylo yhi\n0 8 zlo zhi\n"
                "1 2 3 xy xz yz\n\nAtoms # full\n\n"
                "1 1 1 0 1 2 3 1 0 0\n"
                "2 2 2 0 4 4 4 0 0 0\n"
                "3 2 2 0 4.8 4.6 4 0 0 0\n"
                "4 2 2 0 3.2 4.6 4 0 0 0\n\nVelocities\n",
                encoding="utf-8",
            )
            mapping = CONVERT.read_atom_map(atom_map)
            parsed = CONVERT.read_lammps_data(data_path)
            frame = CONVERT.convert_frame(
                parsed, mapping, tip4p_om_distance_angstrom=0.125
            )
            np.testing.assert_allclose(frame.coordinates_angstrom[0], [11.0, 2.0, 3.0])
            np.testing.assert_allclose(frame.coordinates_angstrom[4], [4.0, 4.125, 4.0])
            self.assertEqual(frame.coordinates_angstrom.shape, (5, 3))
            self.assertGreater(float(np.prod(frame.cell_lengths_angstrom)), 0.0)

    def test_map_molecule_mismatch_is_rejected(self) -> None:
        mapping = [CONVERT.AtomMapRow(1, 1, 2, "QM", "A", "P")]
        parsed = CONVERT.LammpsData(
            {1: np.zeros(3)}, {1: 1}, {1: 1}, np.eye(3) * 10.0
        )
        with self.assertRaisesRegex(ValueError, "molecule ID differs"):
            CONVERT.convert_frame(parsed, mapping, tip4p_om_distance_angstrom=0.125)


if __name__ == "__main__":
    unittest.main()
