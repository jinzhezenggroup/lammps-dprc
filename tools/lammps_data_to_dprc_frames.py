#!/usr/bin/env python3
"""Convert implicit-TIP4P LAMMPS data files into ``DPRCFRM1`` frames.

LAMMPS stores only the real O/H/H atoms used by its implicit TIP4P pair style,
whereas the Amber topology consumed by the binary64 PBE0 labeler also contains
one explicit extra-point slot per water.  This converter uses the reviewed
Amber-to-LAMMPS atom map, reconstructs each M site at the configured O--M
distance, and makes each water whole before Sander rebuilds its local frame.
Only hydrogen lattice images change; oxygen anchors, solute coordinates, and
the full triclinic cell are preserved in binary64. Source files are immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IO_SPEC = importlib.util.spec_from_file_location(
    "dprc_binary64_io_for_lammps_data", ROOT / "tools/dprc_binary64_io.py"
)
if IO_SPEC is None or IO_SPEC.loader is None:
    raise RuntimeError("could not load tools/dprc_binary64_io.py")
IO = importlib.util.module_from_spec(IO_SPEC)
IO_SPEC.loader.exec_module(IO)

BOX_LINE = re.compile(
    r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+(xlo xhi|ylo yhi|zlo zhi)\s*$"
)
TILT_LINE = re.compile(
    r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+xy xz yz\s*$"
)


class AtomMapRow(NamedTuple):
    """One real-atom identity shared by Amber and LAMMPS."""

    amber_id: int
    lammps_id: int
    molecule_id: int
    residue: str
    atom: str
    atom_type: str


class LammpsData(NamedTuple):
    """Coordinates and restricted-triclinic cell parsed from one data file."""

    coordinates_by_id: dict[int, np.ndarray]
    molecule_by_id: dict[int, int]
    type_by_id: dict[int, int]
    cell: np.ndarray


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an immutable source artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_atom_map(path: Path) -> list[AtomMapRow]:
    """Read and validate the complete real-atom identity map."""
    rows: list[AtomMapRow] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        required = ("amber_id", "lammps_id", "molecule_id", "residue", "atom", "type")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"atom map is missing columns: {', '.join(missing)}")
        columns = {name: header.index(name) for name in required}
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                raise ValueError(f"atom map row {line_number} has the wrong field count")
            rows.append(
                AtomMapRow(
                    int(fields[columns["amber_id"]]),
                    int(fields[columns["lammps_id"]]),
                    int(fields[columns["molecule_id"]]),
                    fields[columns["residue"]],
                    fields[columns["atom"]],
                    fields[columns["type"]],
                )
            )
    if not rows:
        raise ValueError("atom map contains no rows")
    for values, name in (
        ([row.amber_id for row in rows], "Amber atom ID"),
        ([row.lammps_id for row in rows], "LAMMPS atom ID"),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"atom map contains duplicate {name}s")
    lammps_ids = sorted(row.lammps_id for row in rows)
    if lammps_ids != list(range(1, len(rows) + 1)):
        raise ValueError("atom map LAMMPS IDs are not the contiguous range 1:N")
    return rows


def _cell_lengths_angles(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a column-vector cell to Amber lengths and alpha/beta/gamma."""
    vectors = [cell[:, index] for index in range(3)]
    lengths = np.asarray([np.linalg.norm(vector) for vector in vectors], dtype=np.float64)

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    angles = np.asarray(
        [angle(vectors[1], vectors[2]), angle(vectors[0], vectors[2]), angle(vectors[0], vectors[1])],
        dtype=np.float64,
    )
    IO._validate_cell(lengths, angles)
    return lengths, angles


def read_lammps_data(path: Path) -> LammpsData:
    """Parse one ``atom_style full`` data file without trusting row order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    bounds: dict[str, tuple[float, float]] = {}
    tilt = (0.0, 0.0, 0.0)
    atom_count: int | None = None
    atoms_header: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if atom_count is None:
            fields = stripped.split()
            if len(fields) == 2 and fields[1] == "atoms":
                atom_count = int(fields[0])
        match = BOX_LINE.match(line)
        if match:
            bounds[match.group(3)] = (float(match.group(1)), float(match.group(2)))
        match = TILT_LINE.match(line)
        if match:
            tilt = tuple(float(match.group(i)) for i in range(1, 4))
        if stripped.startswith("Atoms"):
            atoms_header = index
    if atom_count is None or atoms_header is None:
        raise ValueError(f"LAMMPS data header or Atoms section is missing: {path}")
    if set(bounds) != {"xlo xhi", "ylo yhi", "zlo zhi"}:
        raise ValueError(f"LAMMPS data cell bounds are incomplete: {path}")
    lx = bounds["xlo xhi"][1] - bounds["xlo xhi"][0]
    ly = bounds["ylo yhi"][1] - bounds["ylo yhi"][0]
    lz = bounds["zlo zhi"][1] - bounds["zlo zhi"][0]
    xy, xz, yz = tilt
    cell = np.asarray([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]], dtype=np.float64)
    if not math.isfinite(float(np.linalg.det(cell))) or np.linalg.det(cell) <= 0.0:
        raise ValueError(f"LAMMPS data cell has non-positive volume: {path}")

    coordinates: dict[int, np.ndarray] = {}
    molecules: dict[int, int] = {}
    types: dict[int, int] = {}
    started = False
    for raw_line in lines[atoms_header + 1 :]:
        payload = raw_line.partition("#")[0].strip()
        if not payload:
            continue
        fields = payload.split()
        try:
            atom_id, molecule_id, atom_type = map(int, fields[:3])
            charge = float(fields[3])
            wrapped = np.asarray([float(value) for value in fields[4:7]], dtype=np.float64)
        except (ValueError, IndexError):
            if started:
                break
            continue
        del charge
        started = True
        if atom_id in coordinates:
            raise ValueError(f"LAMMPS data contains duplicate atom ID {atom_id}: {path}")
        image = np.zeros(3, dtype=np.int64)
        if len(fields) >= 10:
            try:
                image = np.asarray([int(value) for value in fields[7:10]], dtype=np.int64)
            except ValueError as error:
                raise ValueError(f"LAMMPS data atom {atom_id} has invalid image flags") from error
        coordinates[atom_id] = wrapped + cell @ image
        molecules[atom_id] = molecule_id
        types[atom_id] = atom_type
    if len(coordinates) != atom_count:
        raise ValueError(
            f"LAMMPS data declares {atom_count} atoms but parsed {len(coordinates)}: {path}"
        )
    return LammpsData(coordinates, molecules, types, cell)


def _minimum_image(displacement: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return the nearest-image displacement in one triclinic cell."""
    fractional = np.linalg.solve(cell, displacement)
    return cell @ (fractional - np.rint(fractional))


def validate_tip4p_sites(
    coordinates: np.ndarray,
    water_sites: list[tuple[int, int, int, int]],
    *,
    om_distance_angstrom: float = 0.125,
) -> dict[str, float | int]:
    """Reject split waters or incorrect Amber O/H/H/EP local frames.

    Site indices are zero-based. This check deliberately uses *direct* bonds,
    not minimum images: Sander's bonded and extra-point routines consume
    these direct vectors. Bounds of 0.7--1.3 A are a broad structural sanity
    gate for rigid TIP4P waters, not a replacement for SHAKE qualification.
    The 1e-8 A EP gate tests the normalized-bond bisector used by Sander.
    """
    if not water_sites:
        return {"water_count": 0, "maximum_ep_residual_angstrom": 0.0}
    sites = np.asarray(water_sites, dtype=np.int64)
    xyz = np.asarray(coordinates, dtype=np.float64)
    if (sites.shape != (len(water_sites), 4) or np.any(sites < 0)
            or np.any(sites >= len(xyz)) or not np.all(np.isfinite(xyz))):
        raise ValueError("TIP4P site indices or coordinates are invalid")
    oxygen = xyz[sites[:, 0]]
    bonds = xyz[sites[:, 1:3]] - oxygen[:, None, :]
    lengths = np.linalg.norm(bonds, axis=2)
    if np.any((lengths < 0.7) | (lengths > 1.3)):
        raise ValueError("TIP4P direct O--H distance outside 0.7--1.3 A; make waters whole before labeling")
    bisector = np.sum(bonds / lengths[:, :, None], axis=1)
    norm = np.linalg.norm(bisector, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("TIP4P normalized-bond bisector is degenerate")
    expected = oxygen + om_distance_angstrom * bisector / norm[:, None]
    residual = float(np.max(np.linalg.norm(xyz[sites[:, 3]] - expected, axis=1)))
    if residual > 1e-8:
        raise ValueError(f"TIP4P extra-point position differs from the whole-water bisector by {residual:.9g} A")
    return {"water_count": len(sites), "minimum_oh_distance_angstrom": float(lengths.min()),
            "maximum_oh_distance_angstrom": float(lengths.max()),
            "maximum_ep_residual_angstrom": residual}


def make_tip4p_waters_whole(
    coordinates: np.ndarray,
    water_sites: list[tuple[int, int, int, int]],
    cell: np.ndarray,
    *,
    om_distance_angstrom: float = 0.125,
) -> np.ndarray:
    """Copy coordinates, changing only H lattice images and Amber EP sites.

    The vectorized operation is shared by initial conversion and immutable
    binary-stream recovery. Coordinates already in the oxygen's image are
    retained exactly; no bond lengths or physical configurations are relaxed.
    """
    xyz = np.asarray(coordinates, dtype=np.float64).copy()
    if not water_sites:
        return xyz
    sites = np.asarray(water_sites, dtype=np.int64)
    oxygen = xyz[sites[:, 0]]
    bonds = xyz[sites[:, 1:3]] - oxygen[:, None, :]
    fractional = np.linalg.solve(cell, bonds.reshape(-1, 3).T).T.reshape(bonds.shape)
    shifts = np.rint(fractional) @ cell.T
    xyz[sites[:, 1:3]] -= shifts
    bonds = xyz[sites[:, 1:3]] - oxygen[:, None, :]
    lengths = np.linalg.norm(bonds, axis=2)
    if np.any((lengths < 0.7) | (lengths > 1.3)):
        raise ValueError("water has O--H distance outside 0.7--1.3 A after periodic imaging")
    bisector = np.sum(bonds / lengths[:, :, None], axis=1)
    norm = np.linalg.norm(bisector, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("water has a degenerate TIP4P bisector")
    xyz[sites[:, 3]] = oxygen + om_distance_angstrom * bisector / norm[:, None]
    validate_tip4p_sites(xyz, water_sites, om_distance_angstrom=om_distance_angstrom)
    return xyz


def convert_frame(
    data: LammpsData,
    atom_map: list[AtomMapRow],
    *,
    tip4p_om_distance_angstrom: float,
) -> IO.Frame:
    """Construct the full Amber-site frame represented by one LAMMPS state."""
    if not math.isfinite(tip4p_om_distance_angstrom) or tip4p_om_distance_angstrom <= 0.0:
        raise ValueError("TIP4P O--M distance must be positive and finite")
    if len(data.coordinates_by_id) != len(atom_map):
        raise ValueError("LAMMPS data and atom map real-atom counts differ")
    for row in atom_map:
        if row.lammps_id not in data.coordinates_by_id:
            raise ValueError(f"LAMMPS data is missing mapped atom {row.lammps_id}")
        if data.molecule_by_id[row.lammps_id] != row.molecule_id:
            raise ValueError(
                f"mapped molecule ID differs for LAMMPS atom {row.lammps_id}"
            )

    water_rows: dict[int, list[AtomMapRow]] = {}
    for row in atom_map:
        if row.residue == "WAT":
            water_rows.setdefault(row.molecule_id, []).append(row)
    full_atom_count = len(atom_map) + len(water_rows)
    real_amber_ids = {row.amber_id for row in atom_map}
    expected_ids = set(range(1, full_atom_count + 1))
    missing_ids = expected_ids - real_amber_ids
    if len(missing_ids) != len(water_rows) or not real_amber_ids <= expected_ids:
        raise ValueError("atom map does not define one implicit extra point per water")

    coordinates = np.zeros((full_atom_count, 3), dtype=np.float64)
    for row in atom_map:
        coordinates[row.amber_id - 1] = data.coordinates_by_id[row.lammps_id]

    water_sites = []
    for molecule_id, rows in water_rows.items():
        rows.sort(key=lambda row: row.amber_id)
        if len(rows) != 3 or [row.atom for row in rows] != ["O", "H1", "H2"]:
            raise ValueError(f"water molecule {molecule_id} is not mapped as O/H1/H2")
        extra_id = rows[-1].amber_id + 1
        if extra_id not in missing_ids:
            raise ValueError(f"water molecule {molecule_id} has no following extra-point slot")
        water_sites.append(tuple(row.amber_id - 1 for row in rows) + (extra_id - 1,))

    # Sander reconstructs EP and redistributes its force using direct bonds,
    # so the parent hydrogens must be made whole *before* that reconstruction.
    coordinates = make_tip4p_waters_whole(coordinates, water_sites, data.cell,
                                        om_distance_angstrom=tip4p_om_distance_angstrom)

    lengths, angles = _cell_lengths_angles(data.cell)
    return IO.Frame(coordinates, lengths, angles)


def artifact(path: Path) -> dict[str, Any]:
    """Describe one source file in a provenance ledger."""
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Publish a JSON ledger atomically without overwriting evidence."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    created = False
    try:
        with partial.open("xb") as handle:
            created = True
            handle.write(encoded)
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
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tip4p-om-distance-angstrom", type=float, default=0.125)
    arguments = parser.parse_args()

    atom_map = read_atom_map(arguments.atom_map)
    frames: list[Any] = []
    sources: list[dict[str, Any]] = []
    for frame_index, path in enumerate(arguments.data, start=1):
        parsed = read_lammps_data(path)
        frame = convert_frame(
            parsed,
            atom_map,
            tip4p_om_distance_angstrom=arguments.tip4p_om_distance_angstrom,
        )
        frames.append(frame)
        sources.append(
            {
                "frame_index": frame_index,
                "frame_name": path.stem,
                "source": artifact(path),
                "cell_lengths_angstrom": frame.cell_lengths_angstrom.tolist(),
                "cell_angles_degrees": frame.cell_angles_degrees.tolist(),
            }
        )
    IO.write_frames(arguments.output, frames)
    manifest = {
        "schema_version": 1,
        "status": "converted-unlabeled",
        "format": "DPRCFRM1",
        "storage": "little-endian IEEE binary64",
        "source_kind": "implicit-TIP4P LAMMPS atom_style full data",
        "atom_map": artifact(arguments.atom_map),
        "tip4p_om_distance_angstrom": arguments.tip4p_om_distance_angstrom,
        "water_image_policy": "whole O/H/H molecules anchored at oxygen; Amber normalized-bond EP bisector",
        "frame_count": len(frames),
        "real_atom_count": len(atom_map),
        "amber_site_count": int(frames[0].coordinates_angstrom.shape[0]),
        "output": artifact(arguments.output),
        "frames": sources,
    }
    write_json_new(arguments.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
