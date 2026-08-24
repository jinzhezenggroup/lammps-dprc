#!/usr/bin/env python3
"""Focused tests for the ETP/ETH xTB force-oracle comparison helpers."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_lammps_xtb_oracle", ROOT / "tools/compare_lammps_xtb_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class LammpsXtbOracleCompareTest(unittest.TestCase):
    def test_parse_result_and_dump_require_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.txt"
            result.write_text(
                "energy=-1 correction=-2 pxx=1 pyy=2 pzz=3 pxy=4 pxz=5 pyz=6 "
                "lx=10 ly=11 lz=12 xy=1 xz=2 yz=3\n",
                encoding="utf-8",
            )
            self.assertEqual(COMPARE.parse_result(result)["energy"], -1.0)

            dump = root / "state.dump"
            dump.write_text(
                "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
                "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
                "ITEM: ATOMS id type q xu yu zu fx fy fz\n"
                "2 7 0.5 1 2 3 4 5 6\n"
                "1 6 -1 7 8 9 10 11 12\n",
                encoding="utf-8",
            )
            state = COMPARE.parse_dump(dump)
            np.testing.assert_array_equal(state.atom_ids, [1, 2])
            np.testing.assert_array_equal(state.charges, [-1.0, 0.5])
            np.testing.assert_array_equal(state.forces_kcal_mol_angstrom[0], [10, 11, 12])

            result.write_text(result.read_text(encoding="utf-8") + "extra=1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields differ"):
                COMPARE.parse_result(result)

    def test_periodic_residual_removes_integer_triclinic_shifts(self) -> None:
        cell = np.asarray(
            [[10.0, -3.0, -2.0], [0.0, 9.0, -4.0], [0.0, 0.0, 8.0]]
        )
        reference = np.asarray([[0.2, 0.3, 0.4], [2.0, 3.0, 4.0]])
        shifts = np.asarray([[1.0, -2.0, 3.0], [-1.0, 0.0, 2.0]])
        actual = reference + shifts @ cell.T
        residual, recovered = COMPARE.periodic_residual(actual, reference, cell)
        np.testing.assert_allclose(residual, 0.0, atol=2.0e-15)
        np.testing.assert_array_equal(recovered, shifts)

    def test_finite_difference_distinguishes_variational_force(self) -> None:
        minus = {name: 0.0 for name in COMPARE.EXPECTED_RESULT_FIELDS}
        plus = dict(minus)
        minus["energy"] = 2.001
        plus["energy"] = 1.999
        force = COMPARE.finite_difference_force(minus, plus, 5.0e-4)
        self.assertAlmostEqual(force, 2.0)
        self.assertLess(abs(force - 2.0005), 1.0e-3)
        self.assertGreater(abs(force - 10.0), 1.0)


if __name__ == "__main__":
    unittest.main()
