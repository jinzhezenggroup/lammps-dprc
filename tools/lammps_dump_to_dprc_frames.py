#!/usr/bin/env python3
"""Convert balanced LAMMPS dump samples into a ``DPRCFRM1`` stream.

The production ETP/ETH sampler writes one trajectory per umbrella window with
stable atom and molecule IDs.  This converter selects frames round-robin across
those trajectories, reconstructs the implicit TIP4P M sites through the same
reviewed atom map used for LAMMPS data files, and streams the complete Amber
site geometries to disk without retaining the full corpus in memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERT = _load(
    "lammps_data_to_dprc_frames_for_dump",
    ROOT / "tools/lammps_data_to_dprc_frames.py",
)
IO = CONVERT.IO


class DumpFrame(NamedTuple):
    """One complete real-atom state parsed from a custom LAMMPS dump."""

    timestep: int
    data: Any


def _required_line(handle: TextIO, context: str) -> str:
    """Read one line and report a truncated dump with useful context."""
    line = handle.readline()
    if not line:
        raise ValueError(f"LAMMPS dump ended while reading {context}")
    return line.rstrip("\n")


def _cell_from_dump(header: str, rows: Sequence[str]) -> np.ndarray:
    """Recover the restricted-triclinic cell from LAMMPS dump bounds."""
    tokens = header.split()[3:]
    triclinic = tokens[:3] == ["xy", "xz", "yz"]
    values = [[float(value) for value in row.split()] for row in rows]
    if triclinic:
        if any(len(row) != 3 for row in values):
            raise ValueError("triclinic LAMMPS dump bounds require three values")
        xlo_bound, xhi_bound, xy = values[0]
        ylo_bound, yhi_bound, xz = values[1]
        zlo, zhi, yz = values[2]
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
    else:
        if any(len(row) != 2 for row in values):
            raise ValueError("orthogonal LAMMPS dump bounds require two values")
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = values
        xy = xz = yz = 0.0
    cell = np.asarray(
        [[xhi - xlo, xy, xz], [0.0, yhi - ylo, yz], [0.0, 0.0, zhi - zlo]],
        dtype=np.float64,
    )
    volume = float(np.linalg.det(cell))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("LAMMPS dump cell has non-positive volume")
    return cell


def _column_indices(header: str) -> tuple[dict[str, int], tuple[str, str, str]]:
    """Validate atom identity columns and choose one coordinate convention."""
    columns = header.split()[2:]
    indices = {name: index for index, name in enumerate(columns)}
    missing = [name for name in ("id", "mol", "type") if name not in indices]
    if missing:
        raise ValueError(
            "LAMMPS dump atom header is missing columns: " + ", ".join(missing)
        )
    for coordinates in (("xu", "yu", "zu"), ("x", "y", "z")):
        if all(name in indices for name in coordinates):
            return indices, coordinates
    raise ValueError("LAMMPS dump requires xu/yu/zu or x/y/z coordinates")


def read_dump_frames(path: Path) -> Iterator[DumpFrame]:
    """Yield every validated frame from one custom LAMMPS trajectory."""
    with path.open("r", encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                return
            if marker.rstrip("\n") != "ITEM: TIMESTEP":
                raise ValueError(f"unexpected LAMMPS dump record in {path}: {marker!r}")
            try:
                timestep = int(_required_line(handle, "timestep"))
            except ValueError as error:
                raise ValueError(f"LAMMPS dump has an invalid timestep: {path}") from error
            if _required_line(handle, "atom-count header") != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"LAMMPS dump atom-count header is invalid: {path}")
            try:
                atom_count = int(_required_line(handle, "atom count"))
            except ValueError as error:
                raise ValueError(f"LAMMPS dump has an invalid atom count: {path}") from error
            if atom_count < 1:
                raise ValueError(f"LAMMPS dump has a non-positive atom count: {path}")
            box_header = _required_line(handle, "box header")
            if not box_header.startswith("ITEM: BOX BOUNDS "):
                raise ValueError(f"LAMMPS dump box header is invalid: {path}")
            cell = _cell_from_dump(
                box_header,
                [_required_line(handle, "box bounds") for _ in range(3)],
            )
            atom_header = _required_line(handle, "atom header")
            if not atom_header.startswith("ITEM: ATOMS "):
                raise ValueError(f"LAMMPS dump atom header is invalid: {path}")
            indices, coordinate_names = _column_indices(atom_header)
            coordinates: dict[int, np.ndarray] = {}
            molecules: dict[int, int] = {}
            types: dict[int, int] = {}
            for _ in range(atom_count):
                fields = _required_line(handle, "atom row").split()
                try:
                    atom_id = int(fields[indices["id"]])
                    molecule_id = int(fields[indices["mol"]])
                    atom_type = int(fields[indices["type"]])
                    position = np.asarray(
                        [float(fields[indices[name]]) for name in coordinate_names],
                        dtype=np.float64,
                    )
                except (IndexError, ValueError) as error:
                    raise ValueError(
                        f"LAMMPS dump contains an invalid atom row: {path}"
                    ) from error
                if atom_id in coordinates:
                    raise ValueError(
                        f"LAMMPS dump contains duplicate atom ID {atom_id}: {path}"
                    )
                if not np.all(np.isfinite(position)):
                    raise ValueError(f"LAMMPS dump contains a non-finite coordinate: {path}")
                coordinates[atom_id] = position
                molecules[atom_id] = molecule_id
                types[atom_id] = atom_type
            yield DumpFrame(
                timestep,
                CONVERT.LammpsData(coordinates, molecules, types, cell),
            )


def count_dump_frames(path: Path) -> int:
    """Count exact timestep markers without parsing the large atom payload twice."""
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line == b"ITEM: TIMESTEP\n" or line == b"ITEM: TIMESTEP\r\n":
                count += 1
    if count < 1:
        raise ValueError(f"LAMMPS dump contains no frames: {path}")
    return count


def balanced_quotas(available: Sequence[int], requested: int) -> list[int]:
    """Distribute a requested corpus round-robin without exceeding any source."""
    if requested < 1:
        raise ValueError("requested frame count must be positive")
    if any(count < 0 for count in available):
        raise ValueError("available frame counts must be nonnegative")
    if requested > sum(available):
        raise ValueError(
            f"requested {requested} frames but only {sum(available)} are available"
        )
    quotas = [0] * len(available)
    remaining = requested
    while remaining:
        progressed = False
        for index, count in enumerate(available):
            if quotas[index] >= count:
                continue
            quotas[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("balanced frame allocation made no progress")
    return quotas


def _write_stream(
    path: Path,
    *,
    atom_count: int,
    frame_count: int,
    frames: Iterator[Any],
) -> None:
    """Write an exact-size frame stream while holding only one frame in memory."""
    def payload(handle: Any) -> None:
        handle.write(
            IO.FRAME_HEADER.pack(
                IO.FRAME_MAGIC,
                IO.SCHEMA_VERSION,
                IO.ENDIAN_MARKER,
                atom_count,
                frame_count,
            )
        )
        written = 0
        for frame in frames:
            coordinates = IO._binary64(frame.coordinates_angstrom, (atom_count, 3))
            lengths = IO._binary64(frame.cell_lengths_angstrom, (3,))
            angles = IO._binary64(frame.cell_angles_degrees, (3,))
            IO._validate_cell(lengths, angles)
            handle.write(coordinates.tobytes(order="C"))
            handle.write(lengths.tobytes(order="C"))
            handle.write(angles.tobytes(order="C"))
            written += 1
        if written != frame_count:
            raise ValueError(
                f"converted {written} frames but the stream declares {frame_count}"
            )

    IO._publish_stream(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, nargs="+", required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tip4p-om-distance-angstrom", type=float, default=0.125)
    arguments = parser.parse_args()

    dumps = [path.resolve() for path in arguments.dump]
    if len(set(dumps)) != len(dumps):
        raise ValueError("LAMMPS dump paths must be unique")
    names = [path.stem for path in dumps]
    if len(set(names)) != len(names):
        raise ValueError("LAMMPS dump stems must identify unique windows")
    available = [count_dump_frames(path) for path in dumps]
    quotas = balanced_quotas(available, arguments.frame_count)
    atom_map = CONVERT.read_atom_map(arguments.atom_map)
    amber_site_count = len(atom_map) + len(
        {row.molecule_id for row in atom_map if row.residue == "WAT"}
    )
    iterators = [read_dump_frames(path) for path in dumps]
    frame_records: list[dict[str, Any]] = []

    def converted_frames() -> Iterator[Any]:
        frame_index = 0
        try:
            for sample_index in range(max(quotas)):
                for source_index, (iterator, quota) in enumerate(
                    zip(iterators, quotas, strict=True)
                ):
                    if sample_index >= quota:
                        continue
                    try:
                        raw = next(iterator)
                    except StopIteration as error:
                        raise ValueError(
                            f"LAMMPS dump ended before its selected quota: "
                            f"{dumps[source_index]}"
                        ) from error
                    frame = CONVERT.convert_frame(
                        raw.data,
                        atom_map,
                        tip4p_om_distance_angstrom=(
                            arguments.tip4p_om_distance_angstrom
                        ),
                    )
                    frame_index += 1
                    frame_records.append(
                        {
                            "frame_index": frame_index,
                            "source_index": source_index,
                            "window": names[source_index],
                            "frame_name": names[source_index],
                            "timestep": raw.timestep,
                            "cell_lengths_angstrom": (
                                frame.cell_lengths_angstrom.tolist()
                            ),
                            "cell_angles_degrees": (
                                frame.cell_angles_degrees.tolist()
                            ),
                        }
                    )
                    yield frame
        finally:
            for iterator in iterators:
                iterator.close()

    _write_stream(
        arguments.output,
        atom_count=amber_site_count,
        frame_count=arguments.frame_count,
        frames=converted_frames(),
    )
    sources = []
    for index, path in enumerate(dumps):
        record = CONVERT.artifact(path)
        record.update(
            {
                "source_index": index,
                "window": names[index],
                "available_frames": available[index],
                "selected_frames": quotas[index],
            }
        )
        sources.append(record)
    manifest = {
        "schema_version": 1,
        "status": "converted-unlabeled",
        "format": "DPRCFRM1",
        "storage": "little-endian IEEE binary64",
        "source_kind": "balanced implicit-TIP4P LAMMPS custom dumps",
        "selection": "round-robin across windows, then increasing timestep",
        "atom_map": CONVERT.artifact(arguments.atom_map),
        "tip4p_om_distance_angstrom": arguments.tip4p_om_distance_angstrom,
        "water_image_policy": "whole O/H/H molecules anchored at oxygen; Amber normalized-bond EP bisector",
        "frame_count": arguments.frame_count,
        "real_atom_count": len(atom_map),
        "amber_site_count": amber_site_count,
        "sources": sources,
        "frames": frame_records,
        "output": CONVERT.artifact(arguments.output),
    }
    CONVERT.write_json_new(arguments.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
