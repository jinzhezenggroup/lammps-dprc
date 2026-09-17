#!/usr/bin/env python3
"""Focused tests for compact real-label DPA4c dataset construction."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_dpa4c_dataset", ROOT / "tools/build_dpa4c_dataset.py"
)
assert SPEC is not None and SPEC.loader is not None
DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET)


class BuildDPA4cDatasetTest(unittest.TestCase):
    def test_force_repair_alone_does_not_qualify_water_geometry(self) -> None:
        checks = {"xTBloom_component_uses_total_minus_same_engine_zero_qm_force": True}
        for value in (None, False):
            checks["sander_tip4p_whole_water_geometry_validated"] = value
            with self.assertRaisesRegex(ValueError, "relabel split-water inputs"):
                DATASET.require_whole_water_labels({"checks": checks})
        checks["sander_tip4p_whole_water_geometry_validated"] = True
        DATASET.require_whole_water_labels({"checks": checks})

    def test_legacy_prefixed_component_corpus_is_rejected(self) -> None:
        for checks in ({}, {"xTBloom_component_uses_total_minus_pre_fix_force": True},
                       {"xTBloom_component_uses_total_minus_same_engine_zero_qm_force": False}):
            with self.subTest(checks=checks), self.assertRaisesRegex(ValueError, "repair legacy labels"):
                DATASET.require_complete_electronic_force({"status": "component-assembled", "checks": checks})
        DATASET.require_complete_electronic_force({"checks": {
            "xTBloom_component_uses_total_minus_same_engine_zero_qm_force": True}})

    def test_complete_water_selection_and_units(self) -> None:
        rows = [
            DATASET.AtomMapRow(i + 1, i + 1, 1, "QM", "Q", "H")
            for i in range(16)
        ]
        rows.extend(
            [
                DATASET.AtomMapRow(17, 17, 2, "WAT", "O", "OW"),
                DATASET.AtomMapRow(18, 18, 2, "WAT", "H1", "HW"),
                DATASET.AtomMapRow(19, 19, 2, "WAT", "H2", "HW"),
                DATASET.AtomMapRow(21, 20, 3, "WAT", "O", "OW"),
                DATASET.AtomMapRow(22, 21, 3, "WAT", "H1", "HW"),
                DATASET.AtomMapRow(23, 22, 3, "WAT", "H2", "HW"),
            ]
        )
        coordinates = np.zeros((23, 3), dtype=np.float64)
        coordinates[:16, 0] = np.linspace(0.0, 1.0, 16)
        coordinates[16:19] = [[5.5, 0, 0], [6.4, 0, 0], [5.5, 0.9, 0]]
        coordinates[20:23] = [[8.0, 0, 0], [8.9, 0, 0], [8.0, 0.9, 0]]
        correction = DATASET.CORRECTION_IO.Correction(
            1,
            1,
            1,
            23.06054783061903,
            23.06054783061903,
            coordinates,
            np.ones((23, 3), dtype=np.float64) * 23.06054783061903,
            np.asarray([20.0, 20.0, 20.0]),
            np.asarray([90.0, 90.0, 90.0]),
        )
        sample = DATASET.compact_sample(correction, rows, cutoff_angstrom=6.0)
        self.assertEqual(sample["selected_atom_count"], 19)
        self.assertEqual(sample["selected_molecule_ids"], [2])
        np.testing.assert_array_equal(sample["amber_ids"][-3:], [17, 18, 19])
        self.assertEqual(sample["types"][-3:].tolist(), [4, 5, 5])

    def test_singular_cell_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "singular"):
            DATASET.cell_matrix(
                np.asarray([10.0, 10.0, 10.0]), np.asarray([90.0, 90.0, 180.0])
            )


if __name__ == "__main__":
    unittest.main()
