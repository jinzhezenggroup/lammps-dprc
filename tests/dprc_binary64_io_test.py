#!/usr/bin/env python3
"""Focused tests for the versioned DPRc binary64 frame and label streams."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dprc_binary64_io", ROOT / "tools/dprc_binary64_io.py"
)
assert SPEC is not None and SPEC.loader is not None
IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IO)


class DPRcBinary64IOTest(unittest.TestCase):
    def frame(self) -> object:
        return IO.Frame(
            np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float64),
            np.asarray([10.0, 11.0, 12.0], dtype=np.float64),
            np.asarray([90.0, 91.0, 92.0], dtype=np.float64),
        )

    def write_label(
        self, path: Path, force: float = -1.25, energy: float = 4.0
    ) -> None:
        coordinates = np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype="<f8")
        forces = np.asarray([[force, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype="<f8")
        with path.open("wb") as handle:
            handle.write(
                IO.LABEL_HEADER.pack(
                    IO.LABEL_MAGIC,
                    IO.SCHEMA_VERSION,
                    IO.ENDIAN_MARKER,
                    1,
                    2,
                    1,
                    IO.TIP4P_REDISTRIBUTED_POLICY,
                )
            )
            handle.write(np.asarray([energy, -3.0], dtype="<f8").tobytes())
            handle.write(np.asarray([10.0, 11.0, 12.0], dtype="<f8").tobytes())
            handle.write(np.asarray([90.0, 91.0, 92.0], dtype="<f8").tobytes())
            handle.write(coordinates.tobytes())
            handle.write(forces.tobytes())
            IO._write_trailer(handle)

    def test_frame_roundtrip_and_perturbation_preserve_binary64(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            IO.write_frames(path, [self.frame()])
            loaded = IO.read_frames(path)
            self.assertEqual(len(loaded), 1)
            self.assertFalse(path.with_name(path.name + ".partial").exists())
            np.testing.assert_array_equal(
                loaded[0].coordinates_angstrom, self.frame().coordinates_angstrom
            )
            perturbed = IO.perturb_frame(loaded[0], 2, IO.AXES["z"], 1.0e-4)
            self.assertAlmostEqual(perturbed.coordinates_angstrom[1, 2], 1.3001)
            self.assertEqual(perturbed.coordinates_angstrom.dtype, np.dtype("<f8"))

    def test_label_parser_rejects_truncation_and_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.bin"
            self.write_label(path)
            label = IO.read_label(path)
            self.assertEqual(label.coordinates_angstrom.shape, (2, 3))
            self.assertEqual(label.extra_point_count, 1)
            self.assertAlmostEqual(label.forces_kcal_mol_angstrom[0, 0], -1.25)

            truncated = Path(directory) / "truncated.bin"
            truncated.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "label stream size mismatch"):
                IO.read_label(truncated)

            trailing = Path(directory) / "trailing.bin"
            trailing.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "label stream size mismatch"):
                IO.read_label(trailing)

            wrong_size = Path(directory) / "wrong-size.bin"
            payload = bytearray(path.read_bytes())
            struct.pack_into("<q", payload, len(payload) - 8, len(payload) - 1)
            wrong_size.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "does not match parsed size"):
                IO.read_label(wrong_size)

    def test_frame_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            IO.write_frames(path, [self.frame()])
            with self.assertRaises(FileExistsError):
                IO.write_frames(path, [self.frame()])
            self.assertFalse(path.with_name(path.name + ".partial").exists())

            invalid = self.frame()._replace(
                cell_angles_degrees=np.asarray([90.0, 0.0, 92.0])
            )
            with self.assertRaisesRegex(ValueError, "invalid periodic cell"):
                IO.write_frames(Path(directory) / "invalid.bin", [invalid])

            zero_volume = self.frame()._replace(
                cell_angles_degrees=np.asarray([10.0, 10.0, 170.0])
            )
            with self.assertRaisesRegex(ValueError, "non-positive volume"):
                IO.write_frames(Path(directory) / "zero-volume.bin", [zero_volume])

    def test_atomic_publication_does_not_overwrite_a_racing_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"

            def racing_writer(handle: object) -> None:
                handle.write(b"candidate")
                path.write_bytes(b"winner")

            with self.assertRaises(FileExistsError):
                IO._publish_stream(path, racing_writer)
            self.assertEqual(path.read_bytes(), b"winner")
            self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_header_validation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            IO.write_frames(path, [self.frame()])
            payload = bytearray(path.read_bytes())
            struct.pack_into("<I", payload, 12, 0xDEADBEEF)
            broken = Path(directory) / "broken.bin"
            broken.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "endian marker"):
                IO.read_frames(broken)

    def test_finite_difference_uses_negative_energy_derivative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minus_path, plus_path, reference_path = (
                root / "minus.bin",
                root / "plus.bin",
                root / "reference.bin",
            )
            delta = 0.01
            self.write_label(minus_path, energy=4.025)
            self.write_label(plus_path, energy=3.975)
            self.write_label(reference_path, force=2.5, energy=4.0)
            result = IO.finite_difference(
                IO.read_label(minus_path),
                IO.read_label(plus_path),
                IO.read_label(reference_path),
                1,
                0,
                delta,
            )
            self.assertAlmostEqual(
                result["finite_difference_force_kcal_mol_angstrom"], 2.5
            )
            self.assertAlmostEqual(result["absolute_error_kcal_mol_angstrom"], 0.0)

    def test_label_frame_identity_checks_real_atoms_and_cell_bitwise(self) -> None:
        frame = self.frame()
        label = IO.Label(
            frame_index=1,
            extra_point_count=1,
            virtual_site_policy=IO.TIP4P_REDISTRIBUTED_POLICY,
            total_potential_energy_kcal_mol=4.0,
            qmmm_scf_energy_kcal_mol=-3.0,
            coordinates_angstrom=frame.coordinates_angstrom.copy(),
            forces_kcal_mol_angstrom=np.zeros((2, 3), dtype=np.float64),
            cell_lengths_angstrom=frame.cell_lengths_angstrom.copy(),
            cell_angles_degrees=frame.cell_angles_degrees.copy(),
        )
        IO.require_label_frame_identity(label, frame, {1}, 1)

        extra_changed = label._replace(coordinates_angstrom=label.coordinates_angstrom.copy())
        extra_changed.coordinates_angstrom[1, 0] += 1.0
        IO.require_label_frame_identity(extra_changed, frame, {1}, 1)

        real_changed = label._replace(coordinates_angstrom=label.coordinates_angstrom.copy())
        real_changed.coordinates_angstrom[0, 0] += np.finfo(np.float64).eps
        with self.assertRaisesRegex(ValueError, "real-atom coordinates"):
            IO.require_label_frame_identity(real_changed, frame, {1}, 1)

        cell_changed = label._replace(cell_angles_degrees=label.cell_angles_degrees.copy())
        cell_changed.cell_angles_degrees[0] += np.finfo(np.float64).eps * 128
        with self.assertRaisesRegex(ValueError, "cell angles"):
            IO.require_label_frame_identity(cell_changed, frame, {1}, 1)

    def test_multi_frame_finite_difference_set_has_stable_pair_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            output = root / "differences.bin"
            manifest = root / "manifest.json"
            IO.write_frames(source, [self.frame()])
            payload = IO.write_finite_difference_set(
                source,
                output,
                manifest,
                [(1, IO.AXES["x"]), (2, IO.AXES["z"])],
                [1.0e-3, 5.0e-4],
                warmup_base_frame=True,
            )
            frames = IO.read_frames(output)
            self.assertEqual(len(frames), 9)
            self.assertEqual(payload["warmup_frame_indices"], [1])
            self.assertEqual(payload["records"][0]["frame_index"], 2)
            self.assertEqual(payload["records"][0]["sign"], -1)
            self.assertEqual(payload["records"][1]["sign"], 1)
            self.assertAlmostEqual(frames[0].coordinates_angstrom[0, 0], 0.1)
            self.assertAlmostEqual(frames[1].coordinates_angstrom[0, 0], 0.099)
            self.assertAlmostEqual(frames[2].coordinates_angstrom[0, 0], 0.101)
            self.assertEqual(payload, IO.json.loads(manifest.read_text()))

    def test_finite_difference_manifest_binds_every_declared_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.bin"
            frames_path = root / "differences.bin"
            manifest_path = root / "manifest.json"
            IO.write_frames(source_path, [self.frame()])
            manifest = IO.write_finite_difference_set(
                source_path,
                frames_path,
                manifest_path,
                [(1, IO.AXES["x"])],
                [1.0e-3],
                warmup_base_frame=True,
            )
            source = IO.read_frames(source_path)[0]
            frames = IO.read_frames(frames_path)
            normalized = IO.validate_finite_difference_manifest(
                manifest, source, frames, {1}
            )
            self.assertEqual([record["frame_index"] for record in normalized], [2, 3])

            duplicate = json.loads(json.dumps(manifest))
            duplicate["records"][0]["frame_index"] = 1
            with self.assertRaisesRegex(ValueError, "reuses a frame index"):
                IO.validate_finite_difference_manifest(duplicate, source, frames, {1})

            relabeled = json.loads(json.dumps(manifest))
            relabeled["records"][0]["delta_angstrom"] = 2.0e-3
            with self.assertRaisesRegex(ValueError, "declared perturbation.*coordinates"):
                IO.validate_finite_difference_manifest(relabeled, source, frames, {1})

            virtual_site_probe = json.loads(json.dumps(manifest))
            virtual_site_probe["records"][0]["atom_id"] = 2
            with self.assertRaisesRegex(ValueError, "not a mapped real atom"):
                IO.validate_finite_difference_manifest(
                    virtual_site_probe, source, frames, {1}
                )

            incomplete = json.loads(json.dumps(manifest))
            incomplete["records"] = incomplete["records"][:-1]
            with self.assertRaisesRegex(ValueError, "do not cover the stream exactly"):
                IO.validate_finite_difference_manifest(incomplete, source, frames, {1})

    def test_retained_periodic_qualification_is_scoped_and_hash_pinned(self) -> None:
        evidence = json.loads(
            (
                ROOT
                / "workloads/etpeth/evidence/quick-pbe0-binary64-label-qualification.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(evidence["qualified"])
        self.assertFalse(evidence["production_ready"])
        self.assertEqual(evidence["reference_label"]["atom_count"], 11912)
        self.assertEqual(evidence["reference_label"]["extra_point_count"], 2974)
        self.assertTrue(evidence["reference_label"]["extra_point_forces_exactly_zero"])
        self.assertEqual(evidence["multi_frame"]["published_label_count"], 12)
        self.assertEqual(evidence["multi_frame"]["successful_quick_call_count"], 12)
        self.assertEqual(
            evidence["source"]["io_tool"]["sha256"],
            IO.sha256(ROOT / evidence["source"]["io_tool"]["path"]),
        )
        self.assertEqual(
            evidence["source"]["ambertools_label_patch"]["sha256"],
            IO.sha256(ROOT / evidence["source"]["ambertools_label_patch"]["path"]),
        )
        self.assertEqual(evidence["qualified_configuration"]["cntrl"]["ntc"], 2)
        self.assertEqual(evidence["qualified_configuration"]["cntrl"]["ntf"], 1)
        self.assertTrue(
            evidence["multi_frame"][
                "manifest_frame_indices_cover_stream_exactly"
            ]
        )
        self.assertTrue(
            evidence["multi_frame"]["declared_perturbations_match_frames_bitwise"]
        )
        self.assertTrue(
            evidence["multi_frame"][
                "real_atom_coordinates_and_cells_match_frames_bitwise"
            ]
        )
        self.assertEqual(
            set(evidence["negative_guards"]),
            {"wrong_method", "wrong_qm_region", "wrong_force_protocol"},
        )
        self.assertIn("independent PBE0", " ".join(evidence["remaining_gates"]))


if __name__ == "__main__":
    unittest.main()
