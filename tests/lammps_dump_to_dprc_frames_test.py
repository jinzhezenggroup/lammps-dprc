#!/usr/bin/env python3
"""Focused tests for balanced LAMMPS-dump to DPRC frame conversion."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lammps_dump_to_dprc_frames", ROOT / "tools/lammps_dump_to_dprc_frames.py"
)
assert SPEC is not None and SPEC.loader is not None
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


def dump_frame(timestep: int, x_offset: float = 0.0) -> str:
    """Return a triclinic four-real-atom dump fixture."""
    return (
        "ITEM: TIMESTEP\n"
        f"{timestep}\n"
        "ITEM: NUMBER OF ATOMS\n"
        "4\n"
        "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        "0 13 1\n"
        "0 11 2\n"
        "0 8 3\n"
        "ITEM: ATOMS id mol type q xu yu zu\n"
        f"1 1 1 0 {1 + x_offset} 2 3\n"
        "2 2 2 0 4 4 4\n"
        "3 2 2 0 4.8 4.6 4\n"
        "4 2 2 0 3.2 4.6 4\n"
    )


class LammpsDumpToDPRCFramesTest(unittest.TestCase):
    def test_wrapped_dump_hydrogens_are_made_whole_before_amber_export(self) -> None:
        """Cover the x/y/z input path that corrupted the historical labels."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrapped.lammpstrj"
            path.write_text(dump_frame(1000).replace("xu yu zu", "x y z")
                            .replace("4.8 4.6 4", "14.8 4.6 4"), encoding="utf-8")
            raw = next(CONVERT.read_dump_frames(path))
            mapping = [CONVERT.CONVERT.AtomMapRow(1, 1, 1, "QM", "Q", "p5")]
            mapping += [CONVERT.CONVERT.AtomMapRow(i + 2, i + 2, 2, "WAT", atom, kind)
                        for i, (atom, kind) in enumerate((("O", "OW"), ("H1", "HW"), ("H2", "HW")))]
            frame = CONVERT.CONVERT.convert_frame(raw.data, mapping, tip4p_om_distance_angstrom=0.125)
            np.testing.assert_allclose(frame.coordinates_angstrom[2], [4.8, 4.6, 4.])
            np.testing.assert_allclose(frame.coordinates_angstrom[4], [4., 4.125, 4.])

    def test_dump_parser_recovers_triclinic_cell_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m0p1.lammpstrj"
            path.write_text(dump_frame(1000), encoding="utf-8")
            parsed = list(CONVERT.read_dump_frames(path))
            self.assertEqual([frame.timestep for frame in parsed], [1000])
            np.testing.assert_allclose(
                parsed[0].data.cell,
                [[10.0, 1.0, 2.0], [0.0, 8.0, 3.0], [0.0, 0.0, 8.0]],
            )
            np.testing.assert_allclose(
                parsed[0].data.coordinates_by_id[1], [1.0, 2.0, 3.0]
            )

    def test_balanced_quotas_and_stream_order(self) -> None:
        self.assertEqual(CONVERT.balanced_quotas([2, 2], 3), [2, 1])
        with self.assertRaisesRegex(ValueError, "only 4 are available"):
            CONVERT.balanced_quotas([2, 2], 5)

    def test_duplicate_atom_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.lammpstrj"
            path.write_text(
                dump_frame(0).replace("4 2 2 0 3.2", "3 2 2 0 3.2"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate atom ID 3"):
                list(CONVERT.read_dump_frames(path))


if __name__ == "__main__":
    unittest.main()
