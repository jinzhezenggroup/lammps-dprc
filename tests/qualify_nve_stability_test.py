"""Focused tests for the pre-production three-window NVE gate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualify_nve_stability",
    ROOT / "tools/qualify_nve_stability.py",
)
assert SPEC is not None and SPEC.loader is not None
NVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NVE)


class QualifyNVEStabilityTest(unittest.TestCase):
    def fixtures(self, root: Path, drift_per_step: float) -> tuple[Path, Path]:
        """Write a minimal qualified record with three hash-pinned thermo logs."""
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "system": {"lammps_atoms": 100},
                    "dynamics": {"timestep_fs": 1.0},
                }
            ),
            encoding="utf-8",
        )
        logs = {}
        for world in range(3):
            path = root / f"log.lammps.{world}"
            lines = ["Step Temp PotEng TotEng"]
            for step in range(0, 5001, 100):
                total = -1000.0 + drift_per_step * step
                lines.append(f"{step} 298.0 {-1100.0 + drift_per_step * step} {total}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logs[path.name] = {"path": str(path), "sha256": NVE.sha256(path)}
        record = root / "record.json"
        record.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "steps_per_window": 5000,
                    "window_order": ["m3p1", "m0p8", "p1p6"],
                    "execution": {
                        "mode": "qmmm-dpa4c",
                        "dynamics": {
                            "ensemble": "NVE",
                            "thermostat": "disabled",
                            "center_of_mass_momentum_removal": "disabled",
                        },
                        "dprc_schedule": {
                            "models_qualified_as_xtb_dprc": True,
                        },
                    },
                    "runtime": {"models": [{"path": "/model.pt2", "sha256": "x"}]},
                    "lammps_logs": logs,
                }
            ),
            encoding="utf-8",
        )
        return record, manifest

    def test_small_conserved_drift_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-nve-gate-") as temporary:
            record, manifest = self.fixtures(Path(temporary), drift_per_step=1.0e-6)
            result = NVE.qualify(record, manifest)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["windows"]), {"m3p1", "m0p8", "p1p6"})

    def test_transient_requires_additional_full_measurement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-nve-transient-") as temporary:
            record, manifest = self.fixtures(Path(temporary), drift_per_step=0.0)
            with self.assertRaisesRegex(ValueError, "at least 5 ps"):
                NVE.qualify(record, manifest, transient_steps=500)
            data = json.loads(record.read_text())
            data["steps_per_window"] = 5500
            for identity in data["lammps_logs"].values():
                path = Path(identity["path"])
                # A single startup jump followed by a conserved 5 ps segment.
                rows = ["Step Temp PotEng TotEng"] + [
                    f"{step} 298 -1100 {-990 if step == 0 else -1000}"
                    for step in range(0, 5501, 100)
                ]
                path.write_text("\n".join(rows) + "\n")
                identity["sha256"] = NVE.sha256(path)
            record.write_text(json.dumps(data))
            self.assertEqual(NVE.qualify(record, manifest)["status"], "failed")
            result = NVE.qualify(record, manifest, transient_steps=500)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["windows"]["m3p1"]["duration_ps"], 5.0)
            self.assertEqual(result["windows"]["m3p1"]["first_step"], 500)

    def test_uncorrected_qmmm_requires_explicit_hamiltonian(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qmmm-nve-gate-") as temporary:
            record, manifest = self.fixtures(Path(temporary), drift_per_step=1.0e-6)
            payload = json.loads(record.read_text())
            payload["execution"]["mode"] = "qmmm"
            del payload["execution"]["dprc_schedule"]
            payload["runtime"]["models"] = []
            record.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Hamiltonian"):
                NVE.qualify(record, manifest)
            result = NVE.qualify(record, manifest, expected_mode="qmmm")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["scope"], "three-window-qmmm-nve-stability")
            self.assertEqual(result["inputs"]["models"], [])
            self.assertEqual(result["thresholds"]["maximum_absolute_net_drift_kcal_mol_atom"], 5e-4)
            payload["runtime"]["models"] = [{"path": "/unexpected.pt2", "sha256": "x"}]
            record.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "contains DPRc"):
                NVE.qualify(record, manifest, expected_mode="qmmm")

    def test_large_systematic_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-nve-gate-") as temporary:
            record, manifest = self.fixtures(Path(temporary), drift_per_step=1.0e-3)
            result = NVE.qualify(record, manifest)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["windows"]["m3p1"]["passed"])

    def test_nonfinite_thresholds_and_short_measurements_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-nve-limits-") as temporary:
            record, manifest = self.fixtures(Path(temporary), drift_per_step=0.0)
            with self.assertRaisesRegex(ValueError, "finite"):
                NVE.qualify(record, manifest, maximum_absolute_net_drift_kcal_mol_atom=float("inf"))
            payload = json.loads(record.read_text())
            payload["steps_per_window"] = 1000
            record.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "at least 5 ps"):
                NVE.qualify(record, manifest)

    def test_thermostatted_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-nve-gate-") as temporary:
            record, manifest = self.fixtures(Path(temporary), drift_per_step=0.0)
            payload = json.loads(record.read_text(encoding="utf-8"))
            payload["execution"]["dynamics"]["ensemble"] = "NVT"
            record.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disable the thermostat"):
                NVE.qualify(record, manifest)


if __name__ == "__main__":
    unittest.main()
