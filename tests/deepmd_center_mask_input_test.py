#!/usr/bin/env python3
"""CPU-only rendering checks for the DeePMD center-energy regression."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_lammps_deepmd_center_mask",
    ROOT / "tests/run_lammps_deepmd_center_mask.py",
)
assert SPEC is not None and SPEC.loader is not None
CENTER_MASK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CENTER_MASK
SPEC.loader.exec_module(CENTER_MASK)
PARTITION_SPEC = importlib.util.spec_from_file_location(
    "run_lammps_deepmd_partition_batch",
    ROOT / "tests/run_lammps_deepmd_partition_batch.py",
)
assert PARTITION_SPEC is not None and PARTITION_SPEC.loader is not None
PARTITION_BATCH = importlib.util.module_from_spec(PARTITION_SPEC)
sys.modules[PARTITION_SPEC.name] = PARTITION_BATCH
PARTITION_SPEC.loader.exec_module(PARTITION_BATCH)


class DeepMDCenterMaskInputTest(unittest.TestCase):
    def test_host_and_kokkos_inputs_share_the_center_mask_contract(self) -> None:
        paths = (
            Path("/runtime/dprcplugin.so"),
            Path("/runtime/model.pt2"),
            Path("/runtime/system.data"),
            Path("/runtime/energy.txt"),
            Path("/runtime/atoms.dump"),
        )
        host = CENTER_MASK.render_input(False, *paths)
        kokkos = CENTER_MASK.render_input(True, *paths)

        for text in (host, kokkos):
            self.assertEqual(text.count("plugin load"), 1)
            self.assertIn("plugin load /runtime/dprcplugin.so", text)
            self.assertIn("group qm id 1:4", text)
            self.assertIn("read_data /runtime/system.data", text)
            self.assertIn("center_group qm", text)
            self.assertIn("environment_cutoff 6.0", text)
            self.assertIn("C H HW O OW P", text)
            self.assertIn("compute dprc_atom all pe/atom pair", text)
            self.assertIn("partition_batch yes", text)
            self.assertNotIn("deepmdplugin.so", text)
        self.assertIn("pair_style dprc/deepmd/batch ", host)
        self.assertIn("atom_style atomic/kk", kokkos)
        self.assertIn("pair_style dprc/deepmd/batch/kk ", kokkos)
        self.assertIn("run_style verlet/kk", kokkos)

    def test_data_preserves_center_environment_and_outside_atom_ids(self) -> None:
        text = CENTER_MASK.render_data()
        self.assertIn("8 atoms", text)
        self.assertIn("1 1 15.0 15.0 15.0", text)
        self.assertIn("5 5 17.1 15.4 14.8", text)
        self.assertIn("7 5 25.0 25.0 25.0", text)

    def test_partition_regression_grows_the_graph_before_final_output(self) -> None:
        paths = (
            Path("/runtime/dprcplugin.so"),
            Path("/runtime/model.pt2"),
            Path("/runtime/system.data"),
            Path("/runtime/energy.txt"),
            Path("/runtime/atoms.dump"),
        )
        text = PARTITION_BATCH.add_capacity_growth_sequence(
            CENTER_MASK.render_input(False, *paths)
        )
        self.assertEqual(text.count("run 0"), 2)
        self.assertIn("group dprc_environment id 5:6 9:80", text)
        self.assertGreater(
            PARTITION_BATCH.FINAL_TOTAL_NODES,
            PARTITION_BATCH.INITIAL_NODE_CAPACITY,
        )
        self.assertLess(text.index("move 8.0 8.0 8.0"), text.index("run 0"))
        self.assertLess(
            text.index("move -8.0 -8.0 -8.0"),
            text.index("compute dprc_atom all pe/atom pair"),
        )

        expanded = PARTITION_BATCH.frame_data(0)
        self.assertIn("80 atoms", expanded)
        self.assertIn("80 3 ", expanded)


if __name__ == "__main__":
    unittest.main()
