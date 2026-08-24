#!/usr/bin/env python3
"""Create and inspect binary64 PBE0-minus-xTB DPRc correction records.

The source ``DPRCLBL1`` files remain immutable engine outputs.  This tool
requires their complete geometry, cell, frame identity, TIP4P publication
policy, and classical-energy contribution to match before subtracting

``correction = high-level QM/MM - low-level QM/MM``.

The resulting ``DPRCCOR1`` stream stores the matching geometry, two correction
energies, and the complete full-system correction force in little-endian
binary64.  It is published atomically without overwriting an existing record.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IO_SPEC = importlib.util.spec_from_file_location(
    "dprc_binary64_io_for_correction", ROOT / "tools/dprc_binary64_io.py"
)
if IO_SPEC is None or IO_SPEC.loader is None:
    raise RuntimeError("could not load tools/dprc_binary64_io.py")
IO = importlib.util.module_from_spec(IO_SPEC)
IO_SPEC.loader.exec_module(IO)

CORRECTION_MAGIC = b"DPRCCOR1"
CORRECTION_HEADER = IO.LABEL_HEADER
DEFAULT_CLASSICAL_CANCELLATION_TOLERANCE_KCAL_MOL = 1.0e-8


class Correction(NamedTuple):
    """One complete high-minus-low correction on an identical QM/MM frame."""

    frame_index: int
    extra_point_count: int
    virtual_site_policy: int
    total_energy_kcal_mol: float
    qmmm_energy_kcal_mol: float
    coordinates_angstrom: np.ndarray
    forces_kcal_mol_angstrom: np.ndarray
    cell_lengths_angstrom: np.ndarray
    cell_angles_degrees: np.ndarray


def _bitwise_equal(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare binary64 payload bytes, not rounded numerical values."""
    left = np.ascontiguousarray(first, dtype=IO.FLOAT64)
    right = np.ascontiguousarray(second, dtype=IO.FLOAT64)
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def _require_matching_labels(high: Any, low: Any) -> None:
    """Reject subtraction unless both engines evaluated one identical frame."""
    if high.frame_index != low.frame_index:
        raise ValueError("high- and low-level label frame indices differ")
    if high.extra_point_count != low.extra_point_count:
        raise ValueError("high- and low-level extra-point counts differ")
    if high.virtual_site_policy != low.virtual_site_policy:
        raise ValueError("high- and low-level virtual-site policies differ")
    for high_values, low_values, name in (
        (high.coordinates_angstrom, low.coordinates_angstrom, "coordinates"),
        (high.cell_lengths_angstrom, low.cell_lengths_angstrom, "cell lengths"),
        (high.cell_angles_degrees, low.cell_angles_degrees, "cell angles"),
    ):
        if not _bitwise_equal(high_values, low_values):
            raise ValueError(f"high- and low-level label {name} differ bitwise")


def _require_tip4p_slots_zero(label: Any, real_ids: set[int], name: str) -> None:
    """Prove each source already redistributed implicit-site forces."""
    atom_count = int(label.coordinates_angstrom.shape[0])
    all_ids = set(range(1, atom_count + 1))
    if not real_ids <= all_ids:
        raise ValueError("atom map contains IDs outside the source labels")
    extra_ids = sorted(all_ids - real_ids)
    if len(extra_ids) != label.extra_point_count:
        raise ValueError(
            f"atom map implies {len(extra_ids)} extra points but {name} "
            f"records {label.extra_point_count}"
        )
    extra_forces = label.forces_kcal_mol_angstrom[np.asarray(extra_ids) - 1]
    if not np.all(extra_forces == 0.0):
        raise ValueError(f"{name} label has nonzero TIP4P extra-point forces")


def subtract_labels(
    high: Any,
    low: Any,
    real_ids: set[int],
    *,
    classical_tolerance_kcal_mol: float = (
        DEFAULT_CLASSICAL_CANCELLATION_TOLERANCE_KCAL_MOL
    ),
) -> tuple[Correction, float]:
    """Form a correction after exact identity and classical-cancellation gates."""
    if (
        not math.isfinite(classical_tolerance_kcal_mol)
        or classical_tolerance_kcal_mol < 0.0
    ):
        raise ValueError("classical cancellation tolerance must be finite and nonnegative")
    _require_matching_labels(high, low)
    _require_tip4p_slots_zero(high, real_ids, "high-level")
    _require_tip4p_slots_zero(low, real_ids, "low-level")

    total = high.total_potential_energy_kcal_mol - low.total_potential_energy_kcal_mol
    qmmm = high.qmmm_scf_energy_kcal_mol - low.qmmm_scf_energy_kcal_mol
    residual = total - qmmm
    if abs(residual) > classical_tolerance_kcal_mol:
        raise ValueError(
            "classical energy does not cancel between labels: "
            f"residual={residual:.17g} kcal/mol exceeds "
            f"{classical_tolerance_kcal_mol:.17g}"
        )
    forces = np.ascontiguousarray(
        high.forces_kcal_mol_angstrom - low.forces_kcal_mol_angstrom,
        dtype=IO.FLOAT64,
    )
    if not np.all(np.isfinite(forces)):
        raise ValueError("DPRc correction forces contain a non-finite value")
    return (
        Correction(
            frame_index=high.frame_index,
            extra_point_count=high.extra_point_count,
            virtual_site_policy=high.virtual_site_policy,
            total_energy_kcal_mol=float(total),
            qmmm_energy_kcal_mol=float(qmmm),
            coordinates_angstrom=np.ascontiguousarray(
                high.coordinates_angstrom, dtype=IO.FLOAT64
            ),
            forces_kcal_mol_angstrom=forces,
            cell_lengths_angstrom=np.ascontiguousarray(
                high.cell_lengths_angstrom, dtype=IO.FLOAT64
            ),
            cell_angles_degrees=np.ascontiguousarray(
                high.cell_angles_degrees, dtype=IO.FLOAT64
            ),
        ),
        float(residual),
    )


def write_correction(path: Path, correction: Correction) -> None:
    """Atomically publish one versioned binary64 correction stream."""
    atom_count = int(correction.coordinates_angstrom.shape[0])
    coordinates = IO._binary64(correction.coordinates_angstrom, (atom_count, 3))
    forces = IO._binary64(correction.forces_kcal_mol_angstrom, (atom_count, 3))
    lengths = IO._binary64(correction.cell_lengths_angstrom, (3,))
    angles = IO._binary64(correction.cell_angles_degrees, (3,))
    IO._validate_cell(lengths, angles)
    if correction.frame_index < 1:
        raise ValueError("DPRc correction frame index must be positive")
    if correction.extra_point_count < 0 or correction.extra_point_count >= atom_count:
        raise ValueError("DPRc correction extra-point count is invalid")
    if correction.virtual_site_policy != IO.TIP4P_REDISTRIBUTED_POLICY:
        raise ValueError("DPRc correction virtual-site policy is unsupported")
    energies = IO._binary64(
        [correction.total_energy_kcal_mol, correction.qmmm_energy_kcal_mol],
        (2,),
    )

    def write_payload(handle: Any) -> None:
        handle.write(
            CORRECTION_HEADER.pack(
                CORRECTION_MAGIC,
                IO.SCHEMA_VERSION,
                IO.ENDIAN_MARKER,
                correction.frame_index,
                atom_count,
                correction.extra_point_count,
                correction.virtual_site_policy,
            )
        )
        for values in (energies, lengths, angles, coordinates, forces):
            handle.write(values.tobytes(order="C"))

    IO._publish_stream(path, write_payload)


def read_correction(path: Path) -> Correction:
    """Read and completely validate one ``DPRCCOR1`` stream."""
    with path.open("rb") as handle:
        magic, schema, endian, frame_index, atom_count, extra_count, policy = (
            CORRECTION_HEADER.unpack(
                IO._read_exact(handle, CORRECTION_HEADER.size, "correction header")
            )
        )
        if magic != CORRECTION_MAGIC:
            raise ValueError(f"unexpected correction magic: {magic!r}")
        if schema != IO.SCHEMA_VERSION:
            raise ValueError(f"unsupported correction schema: {schema}")
        if endian != IO.ENDIAN_MARKER:
            raise ValueError(f"unexpected correction endian marker: 0x{endian:08x}")
        if frame_index < 1 or atom_count < 1:
            raise ValueError("correction header contains a non-positive count")
        if extra_count < 0 or extra_count >= atom_count:
            raise ValueError("correction header contains an invalid extra-point count")
        if policy != IO.TIP4P_REDISTRIBUTED_POLICY:
            raise ValueError("correction header contains an unsupported virtual-site policy")
        IO._require_file_size(
            handle,
            CORRECTION_HEADER.size
            + 8 * IO.FLOAT64.itemsize
            + 2 * atom_count * 3 * IO.FLOAT64.itemsize
            + IO.TRAILER.size,
            "correction",
        )

        def values(count: int, name: str) -> np.ndarray:
            return np.frombuffer(
                IO._read_exact(handle, count * IO.FLOAT64.itemsize, name),
                dtype=IO.FLOAT64,
            ).copy()

        energies = values(2, "correction energies")
        lengths = values(3, "correction cell lengths")
        angles = values(3, "correction cell angles")
        coordinates = values(atom_count * 3, "correction coordinates")
        forces = values(atom_count * 3, "correction forces")
        IO._require_trailer(handle, "correction")
    IO._validate_cell(lengths, angles)
    if not np.all(
        np.isfinite(np.concatenate((energies, lengths, angles, coordinates, forces)))
    ):
        raise ValueError("DPRc correction stream contains a non-finite value")
    return Correction(
        int(frame_index),
        int(extra_count),
        int(policy),
        float(energies[0]),
        float(energies[1]),
        coordinates.reshape(int(atom_count), 3),
        forces.reshape(int(atom_count), 3),
        lengths,
        angles,
    )


def inspect_correction(path: Path, atom_map: Path) -> dict[str, Any]:
    """Return compact scientific and provenance evidence for one correction."""
    correction = read_correction(path)
    real_ids = IO.real_atom_ids(atom_map)
    atom_count = int(correction.coordinates_angstrom.shape[0])
    all_ids = set(range(1, atom_count + 1))
    if not real_ids <= all_ids:
        raise ValueError("atom map contains IDs outside the correction")
    extra_ids = sorted(all_ids - real_ids)
    if len(extra_ids) != correction.extra_point_count:
        raise ValueError("atom map and correction extra-point counts differ")
    extra_forces = correction.forces_kcal_mol_angstrom[np.asarray(extra_ids) - 1]
    if not np.all(extra_forces == 0.0):
        raise ValueError("correction has nonzero TIP4P extra-point force slots")
    real_forces = correction.forces_kcal_mol_angstrom[
        np.asarray(sorted(real_ids), dtype=np.int64) - 1
    ]
    return {
        "schema_version": IO.SCHEMA_VERSION,
        "format": "DPRCCOR1",
        "storage": "little-endian IEEE binary64",
        "path": str(path.resolve()),
        "sha256": IO.sha256(path),
        "frame_index": correction.frame_index,
        "atom_count": atom_count,
        "real_atom_count": len(real_ids),
        "extra_point_count": correction.extra_point_count,
        "virtual_site_force_policy": "redistributed-to-parent-atoms-and-zeroed",
        "total_energy_correction_kcal_mol": correction.total_energy_kcal_mol,
        "qmmm_energy_correction_kcal_mol": correction.qmmm_energy_kcal_mol,
        "classical_energy_cancellation_residual_kcal_mol": (
            correction.total_energy_kcal_mol - correction.qmmm_energy_kcal_mol
        ),
        "force_correction": {
            "unit": "kcal/mol/angstrom",
            "maximum_absolute_real_atom_component": float(np.max(np.abs(real_forces))),
            "rms_real_atom_component": float(np.sqrt(np.mean(real_forces**2))),
            "net_real_atom_vector": np.sum(real_forces, axis=0).tolist(),
            "extra_point_forces_exactly_zero": bool(np.all(extra_forces == 0.0)),
        },
        "cell_lengths_angstrom": correction.cell_lengths_angstrom.tolist(),
        "cell_angles_degrees": correction.cell_angles_degrees.tolist(),
    }


def write_json_new(payload: dict[str, Any], path: Path) -> None:
    """Atomically publish JSON evidence without replacing existing bytes."""
    text = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    created = False
    try:
        with partial.open("xb") as handle:
            created = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial, path)
        partial.unlink()
        created = False
    except BaseException:
        if created:
            partial.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subtract = subparsers.add_parser("subtract-labels")
    subtract.add_argument("--high", type=Path, required=True)
    subtract.add_argument("--low", type=Path, required=True)
    subtract.add_argument("--atom-map", type=Path, required=True)
    subtract.add_argument("--output", type=Path, required=True)
    subtract.add_argument("--manifest", type=Path, required=True)
    subtract.add_argument(
        "--classical-cancellation-tolerance-kcal-mol",
        type=float,
        default=DEFAULT_CLASSICAL_CANCELLATION_TOLERANCE_KCAL_MOL,
    )

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--atom-map", type=Path, required=True)
    inspect.add_argument("--output", type=Path)

    arguments = parser.parse_args()
    try:
        if arguments.command == "subtract-labels":
            real_ids = IO.real_atom_ids(arguments.atom_map)
            correction, residual = subtract_labels(
                IO.read_label(arguments.high),
                IO.read_label(arguments.low),
                real_ids,
                classical_tolerance_kcal_mol=(
                    arguments.classical_cancellation_tolerance_kcal_mol
                ),
            )
            write_correction(arguments.output, correction)
            report = inspect_correction(arguments.output, arguments.atom_map)
            report.update(
                {
                    "operation": "high-level QM/MM minus low-level QM/MM",
                    "high_level_label": {
                        "path": str(arguments.high.resolve()),
                        "sha256": IO.sha256(arguments.high),
                    },
                    "low_level_label": {
                        "path": str(arguments.low.resolve()),
                        "sha256": IO.sha256(arguments.low),
                    },
                    "atom_map": {
                        "path": str(arguments.atom_map.resolve()),
                        "sha256": IO.sha256(arguments.atom_map),
                    },
                    "classical_cancellation_tolerance_kcal_mol": (
                        arguments.classical_cancellation_tolerance_kcal_mol
                    ),
                    "classical_cancellation_residual_before_serialization_kcal_mol": residual,
                }
            )
            write_json_new(report, arguments.manifest)
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        elif arguments.command == "inspect":
            report = inspect_correction(arguments.input, arguments.atom_map)
            if arguments.output is None:
                sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            else:
                write_json_new(report, arguments.output)
        else:
            raise AssertionError(f"unhandled command {arguments.command}")
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"DPRc correction I/O failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
