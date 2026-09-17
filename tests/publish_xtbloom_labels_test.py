#!/usr/bin/env python3
"""Focused tests for synchronized bulk xTBloom correction publication."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_xtbloom_labels", ROOT / "tools/publish_xtbloom_labels.py"
)
assert SPEC is not None and SPEC.loader is not None
BULK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BULK)


def dump_frame(timestep: int, offset: float, *, baseline_only: bool = False) -> str:
    """Return one QM atom and one complete TIP4P water force state."""
    header = (
        "ITEM: TIMESTEP\n"
        f"{timestep}\n"
        "ITEM: NUMBER OF ATOMS\n"
        "4\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0 10\n0 10\n0 10\n"
    )
    if baseline_only:
        return header + (
            "ITEM: ATOMS id type q x y z fx fy fz\n"
            f"1 1 0 {1.0 + offset} 2 3 0.5 0.5 0.5\n"
            f"2 6 0 {4.0 + offset} 5 6 -0.5 -0.5 -0.5\n"
            f"3 7 0 {4.8 + offset} 5.6 6 0 0 0\n"
            f"4 7 0 {3.2 + offset} 5.6 6 0 0 0\n"
        )
    return header + (
        "ITEM: ATOMS id type q x y z fx fy fz "
        "f_pre_xtb[1] f_pre_xtb[2] f_pre_xtb[3]\n"
        f"1 1 0 {1.0 + offset} 2 3 {1.5 + offset} 2.5 3.5 0.5 0.5 0.5\n"
        f"2 6 0 {4.0 + offset} 5 6 {-1.5 - offset} -2.5 -3.5 -0.5 -0.5 -0.5\n"
        f"3 7 0 {4.8 + offset} 5.6 6 0 0 0 0 0 0\n"
        f"4 7 0 {3.2 + offset} 5.6 6 0 0 0 0 0 0\n"
    )


class PublishXTBloomLabelsTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path]:
        """Create two windows whose global labels span two GPU shards."""
        high_root = root / "high"
        baseline_root = root / "baseline"
        raw_root = root / "raw"
        frames = [
            {"frame_index": 1, "window": "w0", "frame_name": "w0", "timestep": 1000},
            {"frame_index": 2, "window": "w1", "frame_name": "w1", "timestep": 1000},
            {"frame_index": 3, "window": "w0", "frame_name": "w0", "timestep": 2000},
        ]
        frame_manifest = root / "frames.json"
        frame_manifest.write_text(
            json.dumps({"frame_count": 3, "frames": frames}), encoding="utf-8"
        )
        atom_map = root / "atom_map.tsv"
        atom_map.write_text(
            "amber_id\tlammps_id\tmolecule_id\tresidue\tatom\ttype\n"
            "1\t1\t1\tQM\tP\tp5\n"
            "2\t2\t2\tWAT\tO\tOW\n"
            "3\t3\t2\tWAT\tH1\tHW\n4\t4\t2\tWAT\tH2\tHW\n",
            encoding="utf-8",
        )

        for global_index, frame in enumerate(frames, start=1):
            offset = global_index / 10.0
            label = BULK.PUBLISH.IO.Label(
                frame_index=(global_index - 1) // 2 + 1,
                extra_point_count=1,
                virtual_site_policy=BULK.PUBLISH.IO.TIP4P_REDISTRIBUTED_POLICY,
                total_potential_energy_kcal_mol=100.0 + global_index,
                qmmm_scf_energy_kcal_mol=70.0 + global_index,
                coordinates_angstrom=np.asarray(
                    [
                        [1.0 + offset, 2.0, 3.0],
                        [4.0 + offset, 5.0, 6.0],
                        [4.8 + offset, 5.6, 6.0],
                        [3.2 + offset, 5.6, 6.0],
                        [4.0 + offset, 5.125, 6.0],
                    ],
                    dtype=np.float64,
                ),
                forces_kcal_mol_angstrom=np.zeros((5, 3), dtype=np.float64),
                cell_lengths_angstrom=np.asarray([10.0, 10.0, 10.0]),
                cell_angles_degrees=np.asarray([90.0, 90.0, 90.0]),
            )
            high_path, _shard, _local = BULK.high_label_path(
                high_root, global_index, 2
            )
            high_path.parent.mkdir(parents=True, exist_ok=True)
            BULK.PUBLISH.write_label(high_path, label)
            baseline = label._replace(
                total_potential_energy_kcal_mol=30.0,
                qmmm_scf_energy_kcal_mol=0.0,
                forces_kcal_mol_angstrom=np.zeros((5, 3), dtype=np.float64),
            )
            baseline_path, _baseline_shard, _baseline_local = BULK.high_label_path(
                baseline_root, global_index, 2
            )
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            BULK.PUBLISH.write_label(baseline_path, baseline)

        for window, selected in (
            ("w0", [(1000, 0.1), (2000, 0.3)]),
            ("w1", [(1000, 0.2)]),
        ):
            directory = raw_root / window
            directory.mkdir(parents=True)
            (directory / "qmmm.dump").write_text(
                "".join(dump_frame(step, offset) for step, offset in selected),
                encoding="utf-8",
            )
            (directory / "qmmm.tsv").write_text(
                " ".join(BULK.QMMM_COLUMNS)
                + "\n"
                + "".join(
                    f"{step} 20 -5 -20 1 10 10 10 0 0 0\n"
                    for step, _offset in selected
                ),
                encoding="utf-8",
            )
            (directory / "mm.dump").write_text(
                "".join(dump_frame(step, offset, baseline_only=True)
                        for step, offset in selected), encoding="utf-8",
            )
            (directory / "mm.tsv").write_text(
                " ".join(BULK.MM_COLUMNS)
                + "\n"
                + "".join(f"{step} -15\n" for step, _offset in selected),
                encoding="utf-8",
            )
        return {
            "frames": frame_manifest,
            "high": high_root,
            "baseline": baseline_root,
            "raw": raw_root,
            "map": atom_map,
        }

    def test_global_shard_mapping_and_resumable_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            low = root / "low"
            corrections = root / "corrections"
            manifest = root / "manifest.json"
            result = BULK.publish_corpus(
                paths["frames"],
                paths["high"],
                paths["baseline"],
                paths["raw"],
                paths["map"],
                low,
                corrections,
                manifest,
                high_shards=2,
                baseline_shards=2,
                coordinate_tolerance_angstrom=1.0e-12,
                baseline_energy_tolerance_kcal_mol=1.0e-12,
                correction_tolerance_kcal_mol=1.0e-12,
            )
            self.assertEqual(result["frame_count"], 3)
            self.assertEqual(
                [record["high_level_local_index"] for record in result["records"]],
                [1, 1, 2],
            )
            self.assertEqual(
                BULK.CORRECTION.read_correction(
                    corrections / "correction.000003.bin"
                ).frame_index,
                3,
            )
            manifest.unlink()
            resumed = BULK.publish_corpus(
                paths["frames"],
                paths["high"],
                paths["baseline"],
                paths["raw"],
                paths["map"],
                low,
                corrections,
                manifest,
                high_shards=2,
                baseline_shards=2,
                coordinate_tolerance_angstrom=1.0e-12,
                baseline_energy_tolerance_kcal_mol=1.0e-12,
                correction_tolerance_kcal_mol=1.0e-12,
            )
            self.assertEqual(resumed["frame_count"], 3)

    def test_mismatched_rerun_timestep_sets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            (paths["raw"] / "w1" / "mm.tsv").write_text(
                "step mm_elong\n2000 -15\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "timestep sets differ"):
                BULK.publish_corpus(
                    paths["frames"], paths["high"], paths["baseline"],
                    paths["raw"], paths["map"],
                    root / "low", root / "corrections", root / "manifest.json",
                    high_shards=2,
                    baseline_shards=2,
                    coordinate_tolerance_angstrom=1.0e-12,
                    baseline_energy_tolerance_kcal_mol=1.0e-12,
                    correction_tolerance_kcal_mol=1.0e-12,
                )

    def test_missing_or_shifted_zero_qm_dump_is_rejected(self) -> None:
        for mode in ("missing", "shifted"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.make_fixture(root)
                baseline = paths["raw"] / "w0" / "mm.dump"
                if mode == "missing":
                    baseline.unlink()
                else:
                    baseline.write_text(baseline.read_text().replace(
                        "TIMESTEP\n1000", "TIMESTEP\n1001"))
                with self.assertRaises((FileNotFoundError, ValueError)):
                    BULK.publish_corpus(
                        paths["frames"], paths["high"], paths["baseline"],
                        paths["raw"], paths["map"], root / "low", root / "corrections",
                        root / "manifest.json", high_shards=2, baseline_shards=2,
                        coordinate_tolerance_angstrom=1e-12,
                        baseline_energy_tolerance_kcal_mol=1e-12,
                        correction_tolerance_kcal_mol=1e-12,
                    )
                self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
