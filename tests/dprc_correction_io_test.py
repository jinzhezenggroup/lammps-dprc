#!/usr/bin/env python3
"""Focused tests for binary64 PBE0-minus-xTB correction records."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dprc_correction_io", ROOT / "tools/dprc_correction_io.py"
)
assert SPEC is not None and SPEC.loader is not None
CORR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORR)


class DPRcCorrectionIOTest(unittest.TestCase):
    def label(self, *, total: float, qmmm: float, force: float) -> object:
        return CORR.IO.Label(
            frame_index=1,
            extra_point_count=1,
            virtual_site_policy=CORR.IO.TIP4P_REDISTRIBUTED_POLICY,
            total_potential_energy_kcal_mol=total,
            qmmm_scf_energy_kcal_mol=qmmm,
            coordinates_angstrom=np.asarray(
                [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype="<f8"
            ),
            forces_kcal_mol_angstrom=np.asarray(
                [[force, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype="<f8"
            ),
            cell_lengths_angstrom=np.asarray([10.0, 11.0, 12.0], dtype="<f8"),
            cell_angles_degrees=np.asarray([90.0, 91.0, 92.0], dtype="<f8"),
        )

    def atom_map(self, path: Path) -> None:
        path.write_text("amber_id\n1\n", encoding="utf-8")

    def test_subtraction_roundtrip_preserves_high_minus_low_semantics(self) -> None:
        high = self.label(total=10.0, qmmm=5.0, force=2.0)
        low = self.label(total=7.0, qmmm=2.0, force=0.5)
        correction, residual = CORR.subtract_labels(high, low, {1})
        self.assertEqual(residual, 0.0)
        self.assertEqual(correction.total_energy_kcal_mol, 3.0)
        self.assertEqual(correction.qmmm_energy_kcal_mol, 3.0)
        self.assertEqual(correction.forces_kcal_mol_angstrom[0, 0], 1.5)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "correction.bin"
            atom_map = root / "atom-map.tsv"
            self.atom_map(atom_map)
            CORR.write_correction(path, correction)
            loaded = CORR.read_correction(path)
            np.testing.assert_array_equal(
                loaded.forces_kcal_mol_angstrom,
                correction.forces_kcal_mol_angstrom,
            )
            report = CORR.inspect_correction(path, atom_map)
            self.assertEqual(report["format"], "DPRCCOR1")
            self.assertTrue(
                report["force_correction"]["extra_point_forces_exactly_zero"]
            )

    def test_subtraction_rejects_geometry_or_classical_mismatch(self) -> None:
        high = self.label(total=10.0, qmmm=5.0, force=2.0)
        low = self.label(total=7.0, qmmm=2.0, force=0.5)
        changed = low._replace(coordinates_angstrom=low.coordinates_angstrom.copy())
        changed.coordinates_angstrom[0, 0] += np.finfo(np.float64).eps
        with self.assertRaisesRegex(ValueError, "coordinates differ bitwise"):
            CORR.subtract_labels(high, changed, {1})

        wrong_classical = low._replace(qmmm_scf_energy_kcal_mol=3.0)
        with self.assertRaisesRegex(ValueError, "classical energy does not cancel"):
            CORR.subtract_labels(high, wrong_classical, {1})

    def test_writer_refuses_overwrite_and_cleans_partial(self) -> None:
        correction, _ = CORR.subtract_labels(
            self.label(total=10.0, qmmm=5.0, force=2.0),
            self.label(total=7.0, qmmm=2.0, force=0.5),
            {1},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correction.bin"
            CORR.write_correction(path, correction)
            with self.assertRaises(FileExistsError):
                CORR.write_correction(path, correction)
            self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_shared_baseline_requires_exact_reconstruction(self) -> None:
        high = self.label(total=1e8 + 0.1, qmmm=0.1, force=2.0)
        low = self.label(total=1e8 + 0.2, qmmm=0.2, force=1.0)
        correction, residual = CORR.subtract_labels(
            high, low, {1}, classical_tolerance_kcal_mol=0.0,
            shared_classical_reference_kcal_mol=1e8,
        )
        self.assertEqual(correction.total_energy_kcal_mol, -0.1)
        self.assertEqual(residual, 0.0)
        with self.assertRaisesRegex(ValueError, "not constructed"):
            CORR.subtract_labels(high, low._replace(total_potential_energy_kcal_mol=np.nextafter(low.total_potential_energy_kcal_mol, np.inf)),
                                 {1}, shared_classical_reference_kcal_mol=1e8)
        with self.assertRaisesRegex(ValueError, "differs"):
            CORR.subtract_labels(high, low, {1}, shared_classical_reference_kcal_mol=1e8 + 1)


if __name__ == "__main__":
    unittest.main()
