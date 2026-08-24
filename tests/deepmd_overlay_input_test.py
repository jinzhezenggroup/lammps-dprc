#!/usr/bin/env python3
"""Guard the private fused KSpace pairing in optional DeePMD inputs."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_lammps_deepmd_overlay", ROOT / "tests/run_lammps_deepmd_overlay.py"
)
assert SPEC is not None and SPEC.loader is not None
OVERLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OVERLAY
SPEC.loader.exec_module(OVERLAY)


class DeepMDOverlayInputTest(unittest.TestCase):
    def test_xtb_modes_use_the_matching_private_kspace_style(self) -> None:
        paths = (
            Path("/runtime/dprcplugin.so"),
            Path("/runtime/model.pt2"),
            Path("/runtime/result.txt"),
        )
        for mode in ("xtb", "overlay"):
            with self.subTest(mode=mode):
                text = OVERLAY.render_input(mode, *paths)
                self.assertIn("fix qmmm qm qmmm/xtb/dprc", text)
                self.assertIn("kspace_style pppm/dprc", text)
                self.assertNotIn("kspace_style pppm/xtb", text)
                self.assertEqual(text.count("plugin load"), 1)
                self.assertNotIn("deepmdplugin.so", text)


if __name__ == "__main__":
    unittest.main()
