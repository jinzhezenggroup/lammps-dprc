#!/usr/bin/env python3
"""Create, inspect, and perturb auditable DPRc binary64 frame/label streams.

The paired AmberTools patch reads ``DPRCFRM1`` inputs and writes ``DPRCLBL1``
outputs.  Both formats are fixed little-endian streams with no compiler record
markers, so byte size, precision, and completeness can be checked independently
of Amber NetCDF, whose coordinate and force variables are binary32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

FRAME_MAGIC = b"DPRCFRM1"
LABEL_MAGIC = b"DPRCLBL1"
TRAILER_MAGIC = b"DPRCEND1"
SCHEMA_VERSION = 1
ENDIAN_MARKER = 0x01020304
FRAME_HEADER = struct.Struct("<8sIIqq")
LABEL_HEADER = struct.Struct("<8sIIqqqI")
TRAILER = struct.Struct("<8sq")
FLOAT64 = np.dtype("<f8")
AXES = {"x": 0, "y": 1, "z": 2}
TIP4P_REDISTRIBUTED_POLICY = 1
EXPECTED_ETPETH_ATOMS = 11912
EXPECTED_ETPETH_EXTRA_POINTS = 2974


class Frame(NamedTuple):
    """One complete Amber-site geometry in Angstrom and degrees."""

    coordinates_angstrom: np.ndarray
    cell_lengths_angstrom: np.ndarray
    cell_angles_degrees: np.ndarray


class Label(NamedTuple):
    """One complete Sander force result before any DPRc subtraction."""

    frame_index: int
    extra_point_count: int
    virtual_site_policy: int
    total_potential_energy_kcal_mol: float
    qmmm_scf_energy_kcal_mol: float
    coordinates_angstrom: np.ndarray
    forces_kcal_mol_angstrom: np.ndarray
    cell_lengths_angstrom: np.ndarray
    cell_angles_degrees: np.ndarray


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary64(
    values: Iterable[float] | np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(values, dtype=FLOAT64).reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError("DPRc binary64 payload contains a non-finite value")
    return np.ascontiguousarray(array)


def _validate_cell(lengths: np.ndarray, angles: np.ndarray) -> None:
    if np.any(lengths <= 0.0) or np.any((angles <= 0.0) | (angles >= 180.0)):
        raise ValueError("DPRc stream contains an invalid periodic cell")
    cosines = np.cos(np.deg2rad(angles))
    gram_determinant = (
        1.0
        + 2.0 * cosines[0] * cosines[1] * cosines[2]
        - float(np.dot(cosines, cosines))
    )
    if not math.isfinite(gram_determinant) or gram_determinant <= 1.0e-12:
        raise ValueError("DPRc stream periodic cell has non-positive volume")


def _require_file_size(handle: Any, expected: int, stream: str) -> None:
    actual = os.fstat(handle.fileno()).st_size
    if actual != expected:
        raise ValueError(
            f"DPRc {stream} stream size mismatch: expected {expected} bytes, "
            f"found {actual}"
        )


def _read_exact(handle: Any, byte_count: int, label: str) -> bytes:
    payload = handle.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError(
            f"truncated {label}: expected {byte_count} bytes, found {len(payload)}"
        )
    return payload


def _require_frame_header(header: bytes) -> tuple[int, int]:
    magic, schema, endian, atom_count, frame_count = FRAME_HEADER.unpack(header)
    if magic != FRAME_MAGIC:
        raise ValueError(f"unexpected DPRc stream magic: {magic!r}")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported DPRc stream schema: {schema}")
    if endian != ENDIAN_MARKER:
        raise ValueError(f"unexpected DPRc endian marker: 0x{endian:08x}")
    if atom_count < 1 or frame_count < 1:
        raise ValueError("DPRc stream header contains a non-positive count")
    return int(atom_count), int(frame_count)


def _require_label_header(header: bytes) -> tuple[int, int, int, int]:
    magic, schema, endian, frame_index, atom_count, extra_count, policy = (
        LABEL_HEADER.unpack(header)
    )
    if magic != LABEL_MAGIC:
        raise ValueError(f"unexpected DPRc stream magic: {magic!r}")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported DPRc stream schema: {schema}")
    if endian != ENDIAN_MARKER:
        raise ValueError(f"unexpected DPRc endian marker: 0x{endian:08x}")
    if frame_index < 1 or atom_count < 1:
        raise ValueError("DPRc stream header contains a non-positive count")
    if extra_count < 0 or extra_count >= atom_count:
        raise ValueError("DPRc label header contains an invalid extra-point count")
    if policy != TIP4P_REDISTRIBUTED_POLICY:
        raise ValueError(f"unsupported DPRc virtual-site force policy: {policy}")
    return int(frame_index), int(atom_count), int(extra_count), int(policy)


def _write_trailer(handle: Any) -> None:
    total_size = handle.tell() + TRAILER.size
    handle.write(TRAILER.pack(TRAILER_MAGIC, total_size))


def _require_trailer(handle: Any, stream: str) -> None:
    magic, recorded_size = TRAILER.unpack(
        _read_exact(handle, TRAILER.size, f"{stream} completion trailer")
    )
    if magic != TRAILER_MAGIC:
        raise ValueError(f"invalid {stream} completion trailer magic: {magic!r}")
    if recorded_size != handle.tell():
        raise ValueError(
            f"{stream} completion trailer size {recorded_size} "
            f"does not match parsed size {handle.tell()}"
        )
    if handle.read(1):
        raise ValueError(f"DPRc {stream} stream contains trailing bytes")


def _publish_stream(path: Path, writer: Any) -> None:
    """Write a completed stream privately and atomically publish it."""
    partial = path.with_name(path.name + ".partial")
    if path.exists():
        raise FileExistsError(path)
    created = False
    try:
        with partial.open("xb") as handle:
            created = True
            writer(handle)
            _write_trailer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes the completed inode atomically and fails with
        # EEXIST if a concurrent writer claimed the final name.  The partial
        # and final names are deliberately siblings, hence one filesystem.
        os.link(partial, path)
        partial.unlink()
        created = False
    except BaseException:
        if created:
            partial.unlink(missing_ok=True)
        raise


def read_amber_restart(path: Path) -> Frame:
    """Read coordinates and six cell values from a formatted Amber restart.

    Decimal fields are parsed directly into binary64, but a formatted restart
    cannot recover coordinate bits that its fixed-width decimal representation
    did not record. Optional restart velocities are ignored; the DPRc
    single-point path constructs no trajectory dynamics.
    """
    with path.open("r", encoding="utf-8") as handle:
        title = handle.readline()
        count_line = handle.readline().split()
        values = [float(token) for line in handle for token in line.split()]
    if not title or not count_line:
        raise ValueError("Amber restart header is incomplete")
    atom_count = int(count_line[0])
    coordinate_count = 3 * atom_count
    trailing = len(values) - coordinate_count
    if trailing not in (6, coordinate_count + 6):
        raise ValueError(
            "Amber restart must contain coordinates, optional velocities, and six cell values"
        )
    coordinates = _binary64(values[:coordinate_count], (atom_count, 3))
    cell = _binary64(values[-6:], (6,))
    if np.any(cell[:3] <= 0.0) or np.any((cell[3:] <= 0.0) | (cell[3:] >= 180.0)):
        raise ValueError("Amber restart contains an invalid periodic cell")
    return Frame(coordinates, cell[:3], cell[3:])


def write_frames(path: Path, frames: list[Frame]) -> None:
    """Write one or more same-topology input frames without silent overwrite."""
    if not frames:
        raise ValueError("at least one DPRc input frame is required")
    atom_count = frames[0].coordinates_angstrom.shape[0]
    normalized: list[Frame] = []
    for frame in frames:
        candidate = Frame(
            _binary64(frame.coordinates_angstrom, (atom_count, 3)),
            _binary64(frame.cell_lengths_angstrom, (3,)),
            _binary64(frame.cell_angles_degrees, (3,)),
        )
        _validate_cell(candidate.cell_lengths_angstrom, candidate.cell_angles_degrees)
        normalized.append(candidate)

    def write_payload(handle: Any) -> None:
        handle.write(
            FRAME_HEADER.pack(
                FRAME_MAGIC,
                SCHEMA_VERSION,
                ENDIAN_MARKER,
                atom_count,
                len(normalized),
            )
        )
        for frame in normalized:
            handle.write(frame.coordinates_angstrom.tobytes(order="C"))
            handle.write(frame.cell_lengths_angstrom.tobytes(order="C"))
            handle.write(frame.cell_angles_degrees.tobytes(order="C"))

    _publish_stream(path, write_payload)


def read_frames(path: Path) -> list[Frame]:
    """Read and completely validate a DPRCFRM1 stream."""
    with path.open("rb") as handle:
        atom_count, frame_count = _require_frame_header(
            _read_exact(handle, FRAME_HEADER.size, "frame header")
        )
        _require_file_size(
            handle,
            FRAME_HEADER.size
            + frame_count * (atom_count * 3 * FLOAT64.itemsize + 6 * FLOAT64.itemsize)
            + TRAILER.size,
            "frame",
        )
        frames: list[Frame] = []
        coordinate_bytes = atom_count * 3 * FLOAT64.itemsize
        vector_bytes = 3 * FLOAT64.itemsize
        for frame_index in range(frame_count):
            coordinates = np.frombuffer(
                _read_exact(
                    handle, coordinate_bytes, f"frame {frame_index + 1} coordinates"
                ),
                dtype=FLOAT64,
            ).copy()
            lengths = np.frombuffer(
                _read_exact(handle, vector_bytes, f"frame {frame_index + 1} lengths"),
                dtype=FLOAT64,
            ).copy()
            angles = np.frombuffer(
                _read_exact(handle, vector_bytes, f"frame {frame_index + 1} angles"),
                dtype=FLOAT64,
            ).copy()
            _validate_cell(lengths, angles)
            frames.append(
                Frame(
                    _binary64(coordinates, (atom_count, 3)),
                    _binary64(lengths, (3,)),
                    _binary64(angles, (3,)),
                )
            )
        _require_trailer(handle, "frame")
    return frames


def read_label(path: Path) -> Label:
    """Read and completely validate one DPRCLBL1 output."""
    with path.open("rb") as handle:
        frame_index, atom_count, extra_count, policy = _require_label_header(
            _read_exact(handle, LABEL_HEADER.size, "label header")
        )
        _require_file_size(
            handle,
            LABEL_HEADER.size
            + 8 * FLOAT64.itemsize
            + 2 * atom_count * 3 * FLOAT64.itemsize
            + TRAILER.size,
            "label",
        )
        scalar_bytes = 2 * FLOAT64.itemsize
        vector_bytes = 3 * FLOAT64.itemsize
        array_bytes = atom_count * 3 * FLOAT64.itemsize
        energies = np.frombuffer(
            _read_exact(handle, scalar_bytes, "label energies"), dtype=FLOAT64
        ).copy()
        lengths = np.frombuffer(
            _read_exact(handle, vector_bytes, "label cell lengths"), dtype=FLOAT64
        ).copy()
        angles = np.frombuffer(
            _read_exact(handle, vector_bytes, "label cell angles"), dtype=FLOAT64
        ).copy()
        _validate_cell(lengths, angles)
        coordinates = np.frombuffer(
            _read_exact(handle, array_bytes, "label coordinates"), dtype=FLOAT64
        ).copy()
        forces = np.frombuffer(
            _read_exact(handle, array_bytes, "label forces"), dtype=FLOAT64
        ).copy()
        _require_trailer(handle, "label")
    all_values = np.concatenate((energies, lengths, angles, coordinates, forces))
    if not np.all(np.isfinite(all_values)):
        raise ValueError("DPRc label stream contains a non-finite value")
    return Label(
        frame_index=frame_index,
        extra_point_count=extra_count,
        virtual_site_policy=policy,
        total_potential_energy_kcal_mol=float(energies[0]),
        qmmm_scf_energy_kcal_mol=float(energies[1]),
        coordinates_angstrom=coordinates.reshape(atom_count, 3),
        forces_kcal_mol_angstrom=forces.reshape(atom_count, 3),
        cell_lengths_angstrom=lengths,
        cell_angles_degrees=angles,
    )


def perturb_frame(frame: Frame, atom_id: int, axis: int, delta: float) -> Frame:
    """Return a copy with one one-based Amber atom coordinate displaced."""
    coordinates = frame.coordinates_angstrom.copy()
    if atom_id < 1 or atom_id > coordinates.shape[0]:
        raise ValueError(f"atom ID {atom_id} is outside 1:{coordinates.shape[0]}")
    if axis not in range(3) or not math.isfinite(delta) or delta == 0.0:
        raise ValueError("perturbation axis or displacement is invalid")
    coordinates[atom_id - 1, axis] += delta
    return Frame(
        coordinates,
        frame.cell_lengths_angstrom.copy(),
        frame.cell_angles_degrees.copy(),
    )


def parse_probe(value: str) -> tuple[int, int]:
    """Parse ``ATOM_ID:AXIS`` for reproducible finite-difference sets."""
    fields = value.split(":")
    if len(fields) != 2 or fields[1] not in AXES:
        raise ValueError(
            f"invalid finite-difference probe {value!r}; use ATOM_ID:x|y|z"
        )
    atom_id = int(fields[0])
    if atom_id < 1:
        raise ValueError("finite-difference atom ID must be positive")
    return atom_id, AXES[fields[1]]


def write_finite_difference_set(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    probes: list[tuple[int, int]],
    deltas: list[float],
    warmup_base_frame: bool = False,
) -> dict[str, Any]:
    """Create one multi-frame stream containing paired central differences."""
    source_frames = read_frames(input_path)
    if len(source_frames) != 1:
        raise ValueError("finite-difference generation requires a one-frame input")
    if not probes or not deltas:
        raise ValueError("finite-difference generation requires probes and deltas")
    source = source_frames[0]
    frames: list[Frame] = [source] if warmup_base_frame else []
    warmup_indices = [1] if warmup_base_frame else []
    records: list[dict[str, Any]] = []
    for atom_id, axis in probes:
        for delta in deltas:
            if not math.isfinite(delta) or delta <= 0.0:
                raise ValueError(
                    "finite-difference displacements must be positive and finite"
                )
            for sign in (-1, 1):
                frames.append(perturb_frame(source, atom_id, axis, sign * delta))
                records.append(
                    {
                        "frame_index": len(frames),
                        "atom_id": atom_id,
                        "axis": tuple(AXES)[axis],
                        "delta_angstrom": delta,
                        "sign": sign,
                    }
                )
    write_frames(output_path, frames)
    manifest = {
        "schema_version": 1,
        "source_frame": str(input_path.resolve()),
        "source_frame_sha256": sha256(input_path),
        "frame_stream": str(output_path.resolve()),
        "frame_stream_sha256": sha256(output_path),
        "frame_count": len(frames),
        "warmup_frame_indices": warmup_indices,
        "records": records,
    }
    write_json(manifest, manifest_path)
    return manifest


def evaluate_finite_difference_set(
    manifest_path: Path,
    label_prefix: Path,
    reference_path: Path,
    atom_map: Path,
) -> dict[str, Any]:
    """Qualify every paired label in a generated finite-difference set."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("records"), list
    ):
        raise ValueError("finite-difference manifest schema is invalid")
    source_path = Path(manifest["source_frame"])
    frame_path = Path(manifest["frame_stream"])
    if sha256(source_path) != manifest.get("source_frame_sha256"):
        raise ValueError("finite-difference source frame changed after generation")
    if sha256(frame_path) != manifest.get("frame_stream_sha256"):
        raise ValueError("finite-difference frame stream changed after generation")

    source_frames = read_frames(source_path)
    frames = read_frames(frame_path)
    if len(source_frames) != 1:
        raise ValueError("finite-difference source must contain exactly one frame")
    if len(frames) != int(manifest.get("frame_count", -1)):
        raise ValueError("finite-difference frame count differs from its manifest")
    real_ids = real_atom_ids(atom_map)
    records = validate_finite_difference_manifest(
        manifest, source_frames[0], frames, real_ids
    )
    reference = read_label(reference_path)
    require_label_frame_identity(reference, source_frames[0], real_ids, 1)
    grouped: dict[tuple[int, str, float], dict[int, Label]] = {}
    label_hashes: dict[int, str] = {}
    for frame_index in manifest.get("warmup_frame_indices", []):
        label_path = Path(f"{label_prefix}.{frame_index:06d}.bin")
        inspect_label(label_path, atom_map)
        label = read_label(label_path)
        require_label_frame_identity(label, frames[frame_index - 1], real_ids, frame_index)
        label_hashes[frame_index] = sha256(label_path)
    for record in records:
        frame_index = record["frame_index"]
        label_path = Path(f"{label_prefix}.{frame_index:06d}.bin")
        # inspect_label rejects payloads that merely widen binary32 results and
        # enforces exact zero force publication in every TIP4P virtual-site
        # slot. The external manifest, not this header, binds parm7/topology.
        inspect_label(label_path, atom_map)
        label = read_label(label_path)
        require_label_frame_identity(label, frames[frame_index - 1], real_ids, frame_index)
        key = (
            record["atom_id"],
            record["axis"],
            record["delta_angstrom"],
        )
        sign = record["sign"]
        if sign not in (-1, 1) or sign in grouped.setdefault(key, {}):
            raise ValueError(
                "finite-difference manifest does not define unique +/- pairs"
            )
        grouped[key][sign] = label
        label_hashes[frame_index] = sha256(label_path)

    results: list[dict[str, Any]] = []
    for (atom_id, axis, delta), pair in grouped.items():
        if set(pair) != {-1, 1}:
            raise ValueError("finite-difference manifest contains an incomplete pair")
        result = finite_difference(
            pair[-1], pair[1], reference, atom_id, AXES[axis], delta
        )
        result.update(
            {
                "atom_id": atom_id,
                "axis": axis,
                "delta_angstrom": delta,
                "minus_energy_kcal_mol": pair[-1].total_potential_energy_kcal_mol,
                "plus_energy_kcal_mol": pair[1].total_potential_energy_kcal_mol,
            }
        )
        results.append(result)
    results.sort(
        key=lambda item: (item["atom_id"], item["axis"], item["delta_angstrom"])
    )
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "reference_label": str(reference_path.resolve()),
        "reference_label_sha256": sha256(reference_path),
        "label_prefix": str(label_prefix.resolve()),
        "real_atom_count": len(real_ids),
        "manifest_frame_indices_cover_stream_exactly": True,
        "declared_perturbations_match_frames_bitwise": True,
        "real_atom_coordinates_and_cells_match_frames_bitwise": True,
        "label_sha256": {
            str(key): value for key, value in sorted(label_hashes.items())
        },
        "results": results,
    }


def validate_finite_difference_manifest(
    manifest: dict[str, Any], source: Frame, frames: list[Frame], real_ids: set[int]
) -> list[dict[str, int | float | str]]:
    """Bind every manifest record to one exact binary64 input frame.

    The evaluator uses manifest metadata to name each derivative probe.  This
    validation prevents a stale or edited manifest from relabeling an otherwise
    valid frame/label pair as a different atom, axis, sign, or displacement.
    """

    def require_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"finite-difference {name} must be an integer")
        return value

    claimed_indices: set[int] = set()
    normalized: list[dict[str, int | float | str]] = []
    warmup = manifest.get("warmup_frame_indices", [])
    if not isinstance(warmup, list):
        raise TypeError("finite-difference warmup frame indices must be a list")
    for value in warmup:
        frame_index = require_integer(value, "warmup frame index")
        claim_manifest_frame_index(frame_index, frames, claimed_indices)
        require_frame_bitwise_identity(
            frames[frame_index - 1], source, f"warmup frame {frame_index}"
        )

    for raw_record in manifest["records"]:
        if not isinstance(raw_record, dict):
            raise TypeError("finite-difference record must be an object")
        frame_index = require_integer(raw_record.get("frame_index"), "frame index")
        atom_id = require_integer(raw_record.get("atom_id"), "atom ID")
        sign = require_integer(raw_record.get("sign"), "sign")
        axis = raw_record.get("axis")
        delta_value = raw_record.get("delta_angstrom")
        if atom_id not in real_ids:
            raise ValueError("finite-difference probe atom is not a mapped real atom")
        if not isinstance(axis, str) or axis not in AXES:
            raise ValueError("finite-difference probe axis must be x, y, or z")
        if (
            isinstance(delta_value, bool)
            or not isinstance(delta_value, (int, float))
            or not math.isfinite(float(delta_value))
            or float(delta_value) <= 0.0
        ):
            raise ValueError(
                "finite-difference displacement must be positive and finite"
            )
        if sign not in (-1, 1):
            raise ValueError("finite-difference sign must be -1 or 1")
        delta = float(delta_value)
        claim_manifest_frame_index(frame_index, frames, claimed_indices)
        expected = perturb_frame(source, atom_id, AXES[axis], sign * delta)
        require_frame_bitwise_identity(
            frames[frame_index - 1],
            expected,
            f"declared perturbation frame {frame_index}",
        )
        normalized.append(
            {
                "frame_index": frame_index,
                "atom_id": atom_id,
                "axis": axis,
                "delta_angstrom": delta,
                "sign": sign,
            }
        )

    expected_indices = set(range(1, len(frames) + 1))
    if claimed_indices != expected_indices:
        raise ValueError(
            "finite-difference manifest frame indices do not cover the stream exactly"
        )
    return normalized


def claim_manifest_frame_index(
    frame_index: int, frames: list[Frame], claimed_indices: set[int]
) -> None:
    """Require one in-range frame index to be claimed exactly once."""
    if frame_index < 1 or frame_index > len(frames):
        raise ValueError("finite-difference frame index is outside the frame stream")
    if frame_index in claimed_indices:
        raise ValueError("finite-difference manifest reuses a frame index")
    claimed_indices.add(frame_index)


def require_frame_bitwise_identity(
    actual: Frame, expected: Frame, context: str
) -> None:
    """Require two complete binary64 frame records to be bitwise identical."""
    if actual.coordinates_angstrom.shape != expected.coordinates_angstrom.shape:
        raise ValueError(f"{context} coordinate extents differ")
    for actual_values, expected_values, name in (
        (actual.coordinates_angstrom, expected.coordinates_angstrom, "coordinates"),
        (actual.cell_lengths_angstrom, expected.cell_lengths_angstrom, "cell lengths"),
        (actual.cell_angles_degrees, expected.cell_angles_degrees, "cell angles"),
    ):
        if not np.array_equal(
            actual_values.view(np.uint64), expected_values.view(np.uint64)
        ):
            raise ValueError(f"{context} {name} differ")


def finite_difference(
    minus: Label, plus: Label, reference: Label, atom_id: int, axis: int, delta: float
) -> dict[str, float]:
    """Compare a central energy derivative with the published Sander force."""
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError("finite-difference displacement must be positive and finite")
    if atom_id < 1 or atom_id > reference.coordinates_angstrom.shape[0]:
        raise ValueError("finite-difference atom ID is outside the label")
    topology = {
        (
            label.coordinates_angstrom.shape[0],
            label.extra_point_count,
            label.virtual_site_policy,
        )
        for label in (minus, plus, reference)
    }
    if len(topology) != 1:
        raise ValueError("finite-difference labels do not share one topology contract")
    derivative_force = -(
        plus.total_potential_energy_kcal_mol - minus.total_potential_energy_kcal_mol
    ) / (2.0 * delta)
    analytic_force = float(reference.forces_kcal_mol_angstrom[atom_id - 1, axis])
    return {
        "analytic_force_kcal_mol_angstrom": analytic_force,
        "finite_difference_force_kcal_mol_angstrom": derivative_force,
        "absolute_error_kcal_mol_angstrom": abs(analytic_force - derivative_force),
    }


def float32_exact_summary(values: np.ndarray) -> dict[str, int | bool]:
    """Report whether binary64 values merely contain widened binary32 bytes."""
    array = np.asarray(values, dtype=np.float64)
    exact = array == array.astype(np.float32).astype(np.float64)
    count = int(np.count_nonzero(exact))
    return {
        "all_values_exactly_representable_in_binary32": bool(np.all(exact)),
        "exact_binary32_value_count": count,
        "value_count": int(array.size),
    }


def real_atom_ids(path: Path) -> set[int]:
    """Read the Amber IDs retained by the implicit-site LAMMPS topology."""
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        try:
            amber_column = header.index("amber_id")
        except ValueError as error:
            raise ValueError("atom map does not define an amber_id column") from error
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            ids.add(int(fields[amber_column]))
    if not ids:
        raise ValueError("atom map contains no Amber IDs")
    return ids


def require_label_frame_identity(
    label: Label, frame: Frame, real_ids: set[int], expected_frame_index: int
) -> None:
    """Require one label to publish the exact real-atom geometry it consumed.

    Sander reconstructs TIP4P extra-point coordinates, so only real atoms are
    compared with the input stream. Cell lengths and angles must remain
    bitwise identical for every frame.
    """
    if label.frame_index != expected_frame_index:
        raise ValueError(
            f"label frame index {label.frame_index} differs from expected "
            f"{expected_frame_index}"
        )
    atom_count = frame.coordinates_angstrom.shape[0]
    if label.coordinates_angstrom.shape != frame.coordinates_angstrom.shape:
        raise ValueError("label and frame coordinate extents differ")
    if not real_ids or min(real_ids) < 1 or max(real_ids) > atom_count:
        raise ValueError("real-atom IDs are outside the frame extent")
    indices = np.asarray(sorted(real_ids), dtype=np.int64) - 1
    label_real = label.coordinates_angstrom[indices]
    frame_real = frame.coordinates_angstrom[indices]
    if not np.array_equal(label_real.view(np.uint64), frame_real.view(np.uint64)):
        raise ValueError("label real-atom coordinates differ from the input frame")
    for label_cell, frame_cell, name in (
        (label.cell_lengths_angstrom, frame.cell_lengths_angstrom, "cell lengths"),
        (label.cell_angles_degrees, frame.cell_angles_degrees, "cell angles"),
    ):
        if not np.array_equal(label_cell.view(np.uint64), frame_cell.view(np.uint64)):
            raise ValueError(f"label {name} differ from the input frame")


def inspect_frame(path: Path) -> dict[str, Any]:
    frames = read_frames(path)
    atom_count = int(frames[0].coordinates_angstrom.shape[0])
    if atom_count != EXPECTED_ETPETH_ATOMS:
        raise ValueError(
            f"ETP/ETH DPRc frame requires {EXPECTED_ETPETH_ATOMS} sites, found {atom_count}"
        )
    coordinates = np.concatenate(
        [frame.coordinates_angstrom.ravel() for frame in frames]
    )
    precision = float32_exact_summary(coordinates)
    if precision["all_values_exactly_representable_in_binary32"]:
        raise ValueError("DPRc frame coordinates are only widened binary32 values")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema_version": SCHEMA_VERSION,
        "storage": "little-endian IEEE binary64",
        "frame_count": len(frames),
        "atom_count": atom_count,
        "coordinates": precision,
        "cell_lengths_angstrom": frames[0].cell_lengths_angstrom.tolist(),
        "cell_angles_degrees": frames[0].cell_angles_degrees.tolist(),
    }


def inspect_label(path: Path, atom_map: Path | None = None) -> dict[str, Any]:
    label = read_label(path)
    atom_count = int(label.coordinates_angstrom.shape[0])
    if atom_count != EXPECTED_ETPETH_ATOMS:
        raise ValueError(
            f"ETP/ETH DPRc label requires {EXPECTED_ETPETH_ATOMS} sites, found {atom_count}"
        )
    if label.extra_point_count != EXPECTED_ETPETH_EXTRA_POINTS:
        raise ValueError(
            "ETP/ETH DPRc label extra-point count is "
            f"{label.extra_point_count}, expected {EXPECTED_ETPETH_EXTRA_POINTS}"
        )
    coordinate_precision = float32_exact_summary(label.coordinates_angstrom)
    force_precision = float32_exact_summary(label.forces_kcal_mol_angstrom)
    if coordinate_precision["all_values_exactly_representable_in_binary32"]:
        raise ValueError("DPRc label coordinates are only widened binary32 values")
    if force_precision["all_values_exactly_representable_in_binary32"]:
        raise ValueError("DPRc label forces are only widened binary32 values")
    report: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema_version": SCHEMA_VERSION,
        "storage": "little-endian IEEE binary64",
        "frame_index": label.frame_index,
        "atom_count": atom_count,
        "extra_point_count": label.extra_point_count,
        "virtual_site_force_policy": "redistributed-to-parent-atoms-and-zeroed",
        "all_values_finite": True,
        "total_potential_energy_kcal_mol": label.total_potential_energy_kcal_mol,
        "qmmm_scf_energy_kcal_mol": label.qmmm_scf_energy_kcal_mol,
        "coordinates": coordinate_precision,
        "forces": force_precision,
        "cell_lengths_angstrom": label.cell_lengths_angstrom.tolist(),
        "cell_angles_degrees": label.cell_angles_degrees.tolist(),
    }
    if atom_map is not None:
        real_ids = real_atom_ids(atom_map)
        all_ids = set(range(1, label.coordinates_angstrom.shape[0] + 1))
        if not real_ids <= all_ids:
            raise ValueError("atom map contains IDs outside the label")
        extra_ids = sorted(all_ids - real_ids)
        if len(extra_ids) != label.extra_point_count:
            raise ValueError(
                f"atom map implies {len(extra_ids)} extra points, "
                f"but label records {label.extra_point_count}"
            )
        extra_forces = label.forces_kcal_mol_angstrom[np.asarray(extra_ids) - 1]
        if not np.all(extra_forces == 0.0):
            raise ValueError("TIP4P extra-point force slots are not exactly zero")
        report["implicit_tip4p_mapping"] = {
            "real_atom_count": len(real_ids),
            "extra_point_count": len(extra_ids),
            "extra_point_forces_exactly_zero": bool(np.all(extra_forces == 0.0)),
            "maximum_extra_point_force_kcal_mol_angstrom": float(
                np.max(np.abs(extra_forces), initial=0.0)
            ),
        }
    return report


def write_json(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    from_restart = subparsers.add_parser("frame-from-rst7")
    from_restart.add_argument("--restart", type=Path, required=True)
    from_restart.add_argument("--output", type=Path, required=True)

    perturb = subparsers.add_parser("perturb")
    perturb.add_argument("--input", type=Path, required=True)
    perturb.add_argument("--output", type=Path, required=True)
    perturb.add_argument("--atom-id", type=int, required=True)
    perturb.add_argument("--axis", choices=tuple(AXES), required=True)
    perturb.add_argument("--delta-angstrom", type=float, required=True)

    make_derivatives = subparsers.add_parser("make-finite-difference-set")
    make_derivatives.add_argument("--input", type=Path, required=True)
    make_derivatives.add_argument("--output", type=Path, required=True)
    make_derivatives.add_argument("--manifest", type=Path, required=True)
    make_derivatives.add_argument("--probe", action="append", required=True)
    make_derivatives.add_argument(
        "--delta-angstrom", action="append", type=float, required=True
    )
    make_derivatives.add_argument("--warmup-base-frame", action="store_true")

    evaluate_derivatives = subparsers.add_parser("evaluate-finite-difference-set")
    evaluate_derivatives.add_argument("--manifest", type=Path, required=True)
    evaluate_derivatives.add_argument("--label-prefix", type=Path, required=True)
    evaluate_derivatives.add_argument("--reference", type=Path, required=True)
    evaluate_derivatives.add_argument("--atom-map", type=Path, required=True)
    evaluate_derivatives.add_argument("--output", type=Path)

    frame_info = subparsers.add_parser("inspect-frame")
    frame_info.add_argument("--input", type=Path, required=True)
    frame_info.add_argument("--output", type=Path)

    label_info = subparsers.add_parser("inspect-label")
    label_info.add_argument("--input", type=Path, required=True)
    label_info.add_argument("--atom-map", type=Path)
    label_info.add_argument("--output", type=Path)

    derivative = subparsers.add_parser("finite-difference")
    derivative.add_argument("--minus", type=Path, required=True)
    derivative.add_argument("--plus", type=Path, required=True)
    derivative.add_argument("--reference", type=Path, required=True)
    derivative.add_argument("--atom-id", type=int, required=True)
    derivative.add_argument("--axis", choices=tuple(AXES), required=True)
    derivative.add_argument("--delta-angstrom", type=float, required=True)
    derivative.add_argument("--output", type=Path)

    arguments = parser.parse_args()
    try:
        if arguments.command == "frame-from-rst7":
            write_frames(arguments.output, [read_amber_restart(arguments.restart)])
            write_json(inspect_frame(arguments.output), None)
        elif arguments.command == "perturb":
            frames = read_frames(arguments.input)
            if len(frames) != 1:
                raise ValueError("perturb currently requires a one-frame stream")
            write_frames(
                arguments.output,
                [
                    perturb_frame(
                        frames[0],
                        arguments.atom_id,
                        AXES[arguments.axis],
                        arguments.delta_angstrom,
                    )
                ],
            )
            write_json(inspect_frame(arguments.output), None)
        elif arguments.command == "make-finite-difference-set":
            manifest = write_finite_difference_set(
                arguments.input,
                arguments.output,
                arguments.manifest,
                [parse_probe(value) for value in arguments.probe],
                arguments.delta_angstrom,
                arguments.warmup_base_frame,
            )
            write_json(manifest, None)
        elif arguments.command == "evaluate-finite-difference-set":
            write_json(
                evaluate_finite_difference_set(
                    arguments.manifest,
                    arguments.label_prefix,
                    arguments.reference,
                    arguments.atom_map,
                ),
                arguments.output,
            )
        elif arguments.command == "inspect-frame":
            write_json(inspect_frame(arguments.input), arguments.output)
        elif arguments.command == "inspect-label":
            write_json(
                inspect_label(arguments.input, arguments.atom_map), arguments.output
            )
        elif arguments.command == "finite-difference":
            result = finite_difference(
                read_label(arguments.minus),
                read_label(arguments.plus),
                read_label(arguments.reference),
                arguments.atom_id,
                AXES[arguments.axis],
                arguments.delta_angstrom,
            )
            result.update(
                {
                    "atom_id": arguments.atom_id,
                    "axis": arguments.axis,
                    "delta_angstrom": arguments.delta_angstrom,
                }
            )
            write_json(result, arguments.output)
        else:
            raise AssertionError(f"unhandled command {arguments.command}")
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"DPRc binary64 I/O failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
