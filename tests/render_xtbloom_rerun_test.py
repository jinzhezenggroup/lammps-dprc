#!/usr/bin/env python3
"""Focused tests for synchronized xTBloom rerun input rendering."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_xtbloom_rerun", ROOT / "tools/render_xtbloom_rerun.py"
)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def dump(frames: int) -> str:
    """Return a minimal stream sufficient for rerun frame-count preflight."""
    return "".join(
        f"ITEM: TIMESTEP\n{index}\n" for index in range(1, frames + 1)
    )


def component_template() -> str:
    """Return the minimum template satisfying the component-force contract."""
    return (
        "clear\n"
        "read_data ${data_file}\n"
        "fix pre_xtb all store/force\n"
        "fix qmmm qm qmmm/xtb/dprc model\n"
        "dump state all custom 1 ${qmmm_dump} id type q x y z fx fy fz "
        "f_pre_xtb[1] f_pre_xtb[2] f_pre_xtb[3]\n"
        "dump mm_state all custom 1 ${mm_dump} id type q x y z fx fy fz\n"
    )


class RenderXTBloomRerunTest(unittest.TestCase):
    def test_renderer_keeps_sorted_partition_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            trajectories = root / "trajectories"
            raw = root / "raw"
            for window in ("p0p1", "m0p1"):
                (data / window).mkdir(parents=True)
                (data / window / f"{window}.data").write_text("data\n")
                (trajectories / window).mkdir(parents=True)
                (trajectories / window / f"{window}.lammpstrj").write_text(
                    dump(2), encoding="utf-8"
                )
            template = root / "template.in"
            template.write_text(component_template(), encoding="utf-8")
            plugin = root / "plugin.so"
            forcefield = root / "forcefield.inc"
            plugin.write_bytes(b"plugin")
            forcefield.write_text("coefficients\n", encoding="utf-8")
            rendered = RENDER.render(
                template, data, trajectories, raw, plugin, forcefield
            )
            self.assertLess(rendered.index("m0p1.data"), rendered.index("p0p1.data"))
            self.assertIn("variable qmmm_dump world &", rendered)
            self.assertIn("frames/partition: 2", rendered)
            self.assertIn("clear\nread_data ${data_file}", rendered)

    def test_explicit_equal_length_subset_is_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for window, frames in (("m0p1", 2), ("m0p2", 3), ("m0p3", 3)):
                data = root / "data" / window
                trajectory = root / "trajectories" / window
                data.mkdir(parents=True)
                trajectory.mkdir(parents=True)
                (data / f"{window}.data").write_text("data\n", encoding="utf-8")
                (trajectory / f"{window}.lammpstrj").write_text(
                    dump(frames), encoding="utf-8"
                )
            template = root / "template.in"
            template.write_text(component_template(), encoding="utf-8")
            rendered = RENDER.render(
                template,
                root / "data",
                root / "trajectories",
                root / "raw",
                root / "plugin.so",
                root / "forcefield.inc",
                ["m0p3", "m0p2"],
            )
            self.assertNotIn("m0p1.data", rendered)
            self.assertLess(rendered.index("m0p2.data"), rendered.index("m0p3.data"))
            self.assertIn("Synchronized partitions: 2; frames/partition: 3", rendered)

    def test_unequal_frame_counts_are_rejected_before_lammps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for window, frames in (("m0p1", 2), ("m0p2", 3)):
                data = root / "data" / window
                trajectory = root / "trajectories" / window
                data.mkdir(parents=True)
                trajectory.mkdir(parents=True)
                (data / f"{window}.data").write_text("data\n", encoding="utf-8")
                (trajectory / f"{window}.lammpstrj").write_text(
                    dump(frames), encoding="utf-8"
                )
            template = root / "template.in"
            template.write_text(component_template(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires equal frame counts"):
                RENDER.render(
                    template,
                    root / "data",
                    root / "trajectories",
                    root / "raw",
                    root / "plugin.so",
                    root / "forcefield.inc",
                )

    def test_missing_trajectory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data" / "m0p1"
            data.mkdir(parents=True)
            (data / "m0p1.data").write_text("data\n")
            template = root / "template.in"
            template.write_text(component_template(), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "trajectory is unavailable"):
                RENDER.render(
                    template,
                    root / "data",
                    root / "trajectories",
                    root / "raw",
                    root / "plugin.so",
                    root / "forcefield.inc",
                )

    def test_legacy_total_force_template_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data" / "m0p1"
            trajectory = root / "trajectories" / "m0p1"
            data.mkdir(parents=True)
            trajectory.mkdir(parents=True)
            (data / "m0p1.data").write_text("data\n", encoding="utf-8")
            (trajectory / "m0p1.lammpstrj").write_text(
                dump(1), encoding="utf-8"
            )
            template = root / "template.in"
            template.write_text(
                "fix qmmm qm qmmm/xtb/dprc model\n"
                "dump state all custom 1 ${qmmm_dump} id type q x y z fx fy fz\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "zero-QM force baseline"):
                RENDER.render(
                    template,
                    root / "data",
                    root / "trajectories",
                    root / "raw",
                    root / "plugin.so",
                    root / "forcefield.inc",
                )

    def test_prefixed_snapshot_without_zero_qm_baseline_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-QM force baseline"):
            RENDER.validate_component_template(component_template().split("dump mm_state")[0])


if __name__ == "__main__":
    unittest.main()
