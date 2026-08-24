#!/usr/bin/env python3
"""Unit tests for the manifest-driven DPRc label qualification logic."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_dprc_labels", ROOT / "tools/audit_dprc_labels.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class DPRcLabelsTest(unittest.TestCase):
    def contract(self) -> dict:
        return json.loads(
            (ROOT / "workloads/etpeth/dprc-labels.json").read_text(encoding="utf-8")
        )

    def inspection(self) -> dict:
        return {
            "group_count": 389,
            "frame_count": 3600,
            "minimum_atoms": 226,
            "maximum_atoms": 285,
            "type_maps": [["C", "H", "HW", "O", "OW", "P"]],
            "water_completion": {"stoichiometrically_unbalanced_frames": 3406},
        }

    def write_minimal_hdf5(self, path: Path) -> None:
        """Write one valid legacy-shaped frame for inspector and CLI tests."""
        import h5py
        import numpy as np

        type_map = ["C", "H", "HW", "O", "OW", "P"]
        types = np.asarray(
            [0] * 3 + [1] * 7 + [2] * 2 + [3] * 5 + [4] + [5],
            dtype=np.int32,
        )
        with h5py.File(path, "w") as handle:
            group = handle.create_group("C3H7HW2O5OW1P1")
            group.create_dataset("nopbc", data=np.asarray(True))
            group.create_dataset("type.raw", data=types)
            group.create_dataset(
                "type_map.raw",
                data=np.asarray(type_map, dtype=h5py.string_dtype("utf-8")),
            )
            frames = group.create_group("set.000")
            values = np.arange(3 * len(types), dtype=np.float64)[None, :]
            frames.create_dataset("coord.npy", data=values)
            frames.create_dataset("energy.npy", data=np.asarray([1.25]))
            frames.create_dataset("force.npy", data=-values)

    def test_formula_parser_preserves_distinct_qm_and_water_types(self) -> None:
        self.assertEqual(
            AUDIT.parse_formula("C3H7HW141O5OW69P1"),
            {"C": 3, "H": 7, "HW": 141, "O": 5, "OW": 69, "P": 1},
        )
        with self.assertRaisesRegex(ValueError, "unexpected DPRc system formula"):
            AUDIT.parse_formula("C3H148O74P1")

    def test_contract_fixes_correction_orientation_and_xtb_target(self) -> None:
        contract = self.contract()
        AUDIT.require_contract(contract)
        contract["production_target"]["label_meaning"] = "absolute PBE0"
        with self.assertRaisesRegex(ValueError, "label_meaning"):
            AUDIT.require_contract(contract)

    def test_contract_pins_real_quick_engine_basis_and_evidence(self) -> None:
        contract = self.contract()
        high_level = contract["production_target"]["high_level"]
        self.assertIn("QUICK 25.03", high_level["engine"])
        self.assertIn("6-31GD.BAS", high_level["basis"])
        evidence = ROOT / high_level["qualification_evidence"]["path"]
        report = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"])
        self.assertEqual(len(report["calls"]), 2)

        forged = self.contract()
        forged["production_target"]["high_level"]["basis"] = "def2-TZVP"
        with self.assertRaisesRegex(ValueError, "basis"):
            AUDIT.require_contract(forged)

    def test_contract_rejects_missing_or_ambiguous_production_semantics(self) -> None:
        mutations = []

        missing_units = self.contract()
        del missing_units["production_target"]["training_units"]
        mutations.append(missing_units)

        duplicate_model_type = self.contract()
        duplicate_model_type["production_target"]["model_type_map"] = [
            "P",
            "O",
            "O",
            "C",
            "H",
            "OW",
            "HW",
        ]
        mutations.append(duplicate_model_type)

        two_atom_qm_region = self.contract()
        two_atom_qm_region["production_target"]["qm_region"]["count"] = 2
        mutations.append(two_atom_qm_region)

        missing_operator_derivatives = self.contract()
        missing_operator_derivatives["production_target"]["embedding"][
            "periodic_response"
        ]["operator_derivative_force_required"] = False
        mutations.append(missing_operator_derivatives)

        missing_point_charge_force = self.contract()
        missing_point_charge_force["production_target"]["required_source_fields"].remove(
            "xtb_raw_point_charge_forces"
        )
        mutations.append(missing_point_charge_force)

        for contract in mutations:
            with self.subTest(contract=contract), self.assertRaises(
                (TypeError, ValueError)
            ):
                AUDIT.require_contract(contract)

    def test_contract_rejects_forged_or_missing_legacy_provenance(self) -> None:
        forged_values = (
            ("license", "MIT"),
            ("producer_version", "1.0"),
        )
        for key, value in forged_values:
            contract = self.contract()
            contract["legacy_dataset"]["source"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, f"legacy_dataset.source.{key}"
            ):
                AUDIT.require_contract(contract)

        forged_label = self.contract()
        forged_label["legacy_dataset"]["label"].update(
            {
                "low_level": "GFN2-xTB",
                "selection": "complete-molecule compact environment",
            }
        )
        with self.assertRaisesRegex(ValueError, "legacy_dataset.label"):
            AUDIT.require_contract(forged_label)

        missing_source = self.contract()
        del missing_source["legacy_dataset"]["source"]
        with self.assertRaisesRegex(TypeError, "legacy_dataset.source"):
            AUDIT.require_contract(missing_source)

    def test_legacy_archive_is_not_an_xtb_production_dataset(self) -> None:
        reasons = AUDIT.qualification_reasons(self.contract(), self.inspection())
        self.assertTrue(any("MNDOD" in reason for reason in reasons))
        self.assertTrue(any("incomplete water" in reason for reason in reasons))
        self.assertTrue(any("b+Aq" in reason for reason in reasons))
        self.assertTrue(any("NOASSERTION" in reason for reason in reasons))
        self.assertTrue(any("correction-label pipeline" in reason for reason in reasons))

    def test_expected_schema_is_fail_closed(self) -> None:
        inspection = self.inspection()
        AUDIT.verify_expected(self.contract(), inspection)
        inspection["frame_count"] -= 1
        with self.assertRaisesRegex(ValueError, "frame_count"):
            AUDIT.verify_expected(self.contract(), inspection)

    def test_hdf5_inspector_names_stoichiometry_as_necessary_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "fixture.hdf5"
            self.write_minimal_hdf5(dataset)
            inspection = AUDIT.inspect_hdf5(dataset)
        water = inspection["water_completion"]
        self.assertEqual(water["stoichiometrically_balanced_frames"], 1)
        self.assertEqual(water["stoichiometrically_unbalanced_frames"], 0)
        self.assertIn("cannot prove complete waters", water["criterion_scope"])

    def test_cli_enforces_license_hash_and_production_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "fixture.hdf5"
            output = root / "audit.json"
            dataset.write_bytes(b"fixture")
            base = [
                "audit_dprc_labels.py",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ]
            expected_hash = self.contract()["legacy_dataset"]["source"][
                "dataset_sha256"
            ]
            real_sha256 = AUDIT.sha256

            def accepted_hash(path: Path) -> str:
                return expected_hash if path == dataset else real_sha256(path)

            with (
                mock.patch.object(AUDIT, "sha256", side_effect=accepted_hash),
                mock.patch.object(AUDIT, "inspect_hdf5", return_value=self.inspection()),
                mock.patch.object(sys, "argv", base),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as error,
            ):
                self.assertEqual(AUDIT.main(), 1)
            self.assertIn("license-unqualified", error.getvalue())
            self.assertFalse(output.exists())

            with (
                mock.patch.object(AUDIT, "sha256", side_effect=accepted_hash),
                mock.patch.object(AUDIT, "inspect_hdf5", return_value=self.inspection()),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        *base,
                        "--allow-unqualified-source",
                        "--require-production-qualified",
                    ],
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(AUDIT.main(), 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["production_qualified"])

            def rejected_hash(path: Path) -> str:
                return "0" * 64 if path == dataset else real_sha256(path)

            with (
                mock.patch.object(AUDIT, "sha256", side_effect=rejected_hash),
                mock.patch.object(sys, "argv", [*base, "--allow-unqualified-source"]),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as error,
            ):
                self.assertEqual(AUDIT.main(), 1)
            self.assertIn("differs from expected", error.getvalue())


if __name__ == "__main__":
    unittest.main()
