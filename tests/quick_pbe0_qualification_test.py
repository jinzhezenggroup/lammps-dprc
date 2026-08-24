#!/usr/bin/env python3
"""Focused tests for QUICK PBE0 evidence parsing and pinned patch provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualify_quick_pbe0", ROOT / "tools/qualify_quick_pbe0.py"
)
assert SPEC is not None and SPEC.loader is not None
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


class QuickPBE0QualificationTest(unittest.TestCase):
    def manifest(self) -> dict:
        return json.loads(
            (ROOT / "config/quick_pbe0_engine.json").read_text(encoding="utf-8")
        )

    def quick_trace(self) -> str:
        return """\
  KEYWORD=PBE0 BASIS=6-31G* SCF=250 DENSERMS=   0.0000000100 CHARGE=0 MULT=1 GRADIENT DIPOLE EXTCHARGES
 USING LIBXC VERSION: 4.3.4
 NAME = PBEH (PBE0) FAMILY = HYBRID GGA KIND = EXCHANGE CORRELATION
| REACH CONVERGENCE AFTER  17 CYCLES
| MAX ERROR = 0.230845E-07   RMS CHANGE = 0.277346E-08   MAX CHANGE = 0.379426E-07
 TOTAL ENERGY         =   -248.213189416
| TOTAL SCF TIME      =     2.801937000( 33.13%)
"""

    def test_quick_trace_requires_exact_method_and_convergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.out"
            trace.write_text(self.quick_trace(), encoding="utf-8")
            parsed = QUALIFY.parse_quick_trace(trace, self.manifest())
            self.assertEqual(parsed["cycles"], 17)
            self.assertAlmostEqual(parsed["energy_hartree"], -248.213189416)

            trace.write_text(
                self.quick_trace().replace("PBEH (PBE0)", "PBE"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "functional"):
                QUALIFY.parse_quick_trace(trace, self.manifest())

    def test_sander_parser_rejects_partial_gradient_publication(self) -> None:
        block = """\
 QUICK execution success; Processing QUICK results...
qm2_extern_quick_module - final energy:
    -155756.12679685
qm2_extern_quick_module - final gradient(s):
QM region:
     1.0 2.0 3.0
     4.0 5.0 6.0
MM region:
     7.0 8.0 9.0
<<<<< Left print_results (qm2_extern_util_module)
"""
        with tempfile.TemporaryDirectory() as directory:
            mdout = Path(directory) / "run.mdout"
            mdout.write_text(block, encoding="utf-8")
            calls = QUALIFY.parse_sander_mdout(mdout, 1, 2, 1)
            self.assertEqual(calls[0]["qm_link_gradient_vectors"], 2)
            with self.assertRaisesRegex(ValueError, "extent mismatch"):
                QUALIFY.parse_sander_mdout(mdout, 1, 3, 1)

    def test_telemetry_reduction_preserves_peak_gpu_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            telemetry = Path(directory) / "gpu.csv"
            with telemetry.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "utilization.gpu [%]",
                        "power.draw [W]",
                        "memory.used [MiB]",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "t0",
                        "utilization.gpu [%]": "25 %",
                        "power.draw [W]": "75.0 W",
                        "memory.used [MiB]": "1000 MiB",
                    }
                )
                writer.writerow(
                    {
                        "timestamp": "t1",
                        "utilization.gpu [%]": "100 %",
                        "power.draw [W]": "138.2 W",
                        "memory.used [MiB]": "16002 MiB",
                    }
                )
            reduced = QUALIFY.parse_telemetry(telemetry)
            self.assertEqual(reduced["samples"], 2)
            self.assertEqual(reduced["maximum_gpu_utilization_percent"], 100.0)

    def test_retained_cuda_patch_and_license_are_hash_pinned(self) -> None:
        manifest = self.manifest()
        source = manifest["source"]
        patch = ROOT / manifest["source"]["cuda_compatibility_patch"]["path"]
        patch_digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(
            patch_digest,
            manifest["source"]["cuda_compatibility_patch"]["sha256"],
        )
        license_path = ROOT / "LICENSES/GPL-3.0-only.txt"
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes()).hexdigest(),
            "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
        )
        self.assertEqual(
            source["license_files"]["ambertools"]["sha256"],
            "912af0215a173b10e44c254b1ee2ed844393298a503bea4596216afcb42ec509",
        )
        self.assertEqual(
            source["license_files"]["quick"]["sha256"],
            "1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5",
        )


if __name__ == "__main__":
    unittest.main()
