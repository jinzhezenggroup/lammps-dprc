#!/usr/bin/env python3
"""Publish one component-assembled xTBloom ``DPRCLBL1`` label.

The matching Sander zero-QM/MM baseline supplies the classical, PME,
constraint, and TIP4P force assembly. LAMMPS dumps both the full QM/MM force
and a separate same-engine zero-QM-charge force, so the only cross-engine
quantity added to the Sander baseline is

``F_xTBloom_component = F_total_after_xTBloom - F_same_engine_zero_QM``.

The pre-fix snapshot is diagnostic only: it already contains QM-dependent
PPPM forces and cannot serve as the classical baseline.

This avoids subtracting Sander and LAMMPS total forces, whose different TIP4P
virtual-site and constraint-force conventions contaminate a DPRc target.

For the ETP/ETH label input, the Coulomb pair sub-style is mapped only to MM
types.  The xTBloom fix therefore publishes

``correction = E_xTB + E_MM,KSpace - E_full,KSpace``.

The companion MM-only reciprocal run still supplies ``E_MM,KSpace`` so the
tool can recover the xTB QM/MM energy without private LAMMPS plugin symbols.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    """Load one repository tool without relying on installation state."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IO = load_module("dprc_binary64_io_for_xtbloom", ROOT / "tools/dprc_binary64_io.py")
COMPARE = load_module(
    "compare_lammps_xtb_oracle_for_label",
    ROOT / "tools/compare_lammps_xtb_oracle.py",
)
GEOMETRY = load_module(
    "lammps_data_geometry_for_label", ROOT / "tools/lammps_data_to_dprc_frames.py"
)

QMMM_FIELDS = {
    "energy",
    "correction",
    "elong",
    "pair_reference_mm_only",
    "lx",
    "ly",
    "lz",
    "xy",
    "xz",
    "yz",
}
EXPECTED_LAMMPS_TYPES = {"p5": 1, "o": 2, "os": 3, "c3": 4, "h1": 5, "OW": 6, "HW": 7}
EXPECTED_COMPONENT_DUMP_COLUMNS = (
    "id",
    "type",
    "q",
    "xu",
    "yu",
    "zu",
    "fx",
    "fy",
    "fz",
    "f_pre_xtb[1]",
    "f_pre_xtb[2]",
    "f_pre_xtb[3]",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    """Describe one immutable input or output artifact."""
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def parse_scalars(path: Path, expected: set[str]) -> dict[str, float]:
    """Parse a complete whitespace-separated ``name=value`` record."""
    values: dict[str, float] = {}
    for token in path.read_text(encoding="utf-8").split():
        key, separator, raw = token.partition("=")
        if not separator or key in values:
            raise ValueError(f"invalid or duplicate result token {token!r}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite result {key}={value}")
        values[key] = value
    if set(values) != expected:
        raise ValueError(
            f"result fields differ: expected {sorted(expected)}, found {sorted(values)}"
        )
    return values


def read_atom_map(path: Path) -> list[dict[str, Any]]:
    """Read atom identity and expected LAMMPS type in LAMMPS-ID order."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        required = ("amber_id", "lammps_id", "type")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"atom map is missing columns: {', '.join(missing)}")
        columns = {name: header.index(name) for name in required}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            atom_type = fields[columns["type"]]
            if atom_type not in EXPECTED_LAMMPS_TYPES:
                raise ValueError(f"atom map has unsupported type {atom_type!r}")
            rows.append(
                {
                    "amber_id": int(fields[columns["amber_id"]]),
                    "lammps_id": int(fields[columns["lammps_id"]]),
                    "lammps_type": EXPECTED_LAMMPS_TYPES[atom_type],
                }
            )
    rows.sort(key=lambda row: row["lammps_id"])
    if [row["lammps_id"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("atom map LAMMPS IDs are not the contiguous range 1:N")
    if len({row["amber_id"] for row in rows}) != len(rows):
        raise ValueError("atom map contains duplicate Amber IDs")
    return rows


def parse_component_dump(path: Path, *, baseline_only: bool = False) -> tuple[Any, np.ndarray]:
    """Read one complete dump containing total and stored pre-xTB forces."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        atoms_header = next(
            index for index, line in enumerate(lines) if line.startswith("ITEM: ATOMS ")
        )
    except StopIteration as error:
        raise ValueError("LAMMPS component dump has no atom table") from error
    columns = tuple(lines[atoms_header].split()[2:])
    expected = EXPECTED_COMPONENT_DUMP_COLUMNS[:9] if baseline_only else EXPECTED_COMPONENT_DUMP_COLUMNS
    if columns != expected:
        raise ValueError(
            "LAMMPS component dump columns must be "
            f"{EXPECTED_COMPONENT_DUMP_COLUMNS}, found {columns}"
        )
    try:
        count_header = lines.index("ITEM: NUMBER OF ATOMS")
        atom_count = int(lines[count_header + 1])
    except (IndexError, ValueError) as error:
        raise ValueError("LAMMPS component dump atom count is invalid") from error
    atom_lines = lines[atoms_header + 1 :]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"LAMMPS component dump declares {atom_count} atoms but contains "
            f"{len(atom_lines)} rows"
        )
    table = np.asarray(
        [[float(value) for value in line.split()] for line in atom_lines],
        dtype=np.float64,
    )
    if table.shape != (atom_count, len(columns)) or not np.all(np.isfinite(table)):
        raise ValueError("LAMMPS component dump table is malformed or non-finite")
    order = np.argsort(table[:, 0])
    table = table[order]
    atom_ids = table[:, 0].astype(np.int64)
    atom_types = table[:, 1].astype(np.int64)
    expected_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    if not np.array_equal(atom_ids, expected_ids):
        raise ValueError("LAMMPS component dump atom IDs are not contiguous")
    if not np.array_equal(table[:, 0], atom_ids.astype(np.float64)) or not np.array_equal(
        table[:, 1], atom_types.astype(np.float64)
    ):
        raise ValueError("LAMMPS component dump atom IDs or types are not integral")
    state = COMPARE.LammpsState(
        atom_ids,
        atom_types,
        table[:, 2].copy(),
        table[:, 3:6].copy(),
        table[:, 6:9].copy(),
    )
    return state, table[:, 9:12].copy()


def _bitwise_equal(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare complete binary64 arrays by their IEEE bit patterns."""
    left = np.ascontiguousarray(first, dtype=IO.FLOAT64)
    right = np.ascontiguousarray(second, dtype=IO.FLOAT64)
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def _require_matching_baseline(
    high: Any,
    baseline: Any,
    *,
    energy_tolerance_kcal_mol: float,
) -> tuple[float, float, float, float]:
    """Validate the zero-QM/MM baseline before any force is assembled."""
    if not math.isfinite(energy_tolerance_kcal_mol) or energy_tolerance_kcal_mol < 0.0:
        raise ValueError("baseline energy tolerance must be finite and nonnegative")
    for name in ("frame_index", "extra_point_count", "virtual_site_policy"):
        if getattr(high, name) != getattr(baseline, name):
            raise ValueError(f"high-level and baseline labels differ in {name}")
    for name, description in (
        ("coordinates_angstrom", "coordinates"),
        ("cell_lengths_angstrom", "cell lengths"),
        ("cell_angles_degrees", "cell angles"),
    ):
        if not _bitwise_equal(getattr(high, name), getattr(baseline, name)):
            raise ValueError(
                f"high-level and baseline label {description} differ bitwise"
            )
    if np.asarray([baseline.qmmm_scf_energy_kcal_mol], dtype=IO.FLOAT64).view(
        np.uint64
    )[0] != np.asarray([0.0], dtype=IO.FLOAT64).view(np.uint64)[0]:
        raise ValueError("Sander zero-QM/MM baseline SCF energy is not exactly +0.0")
    high_classical = (
        high.total_potential_energy_kcal_mol - high.qmmm_scf_energy_kcal_mol
    )
    residual = baseline.total_potential_energy_kcal_mol - high_classical
    energy_values = (
        high.total_potential_energy_kcal_mol,
        high.qmmm_scf_energy_kcal_mol,
        high_classical,
        baseline.total_potential_energy_kcal_mol,
    )
    if not all(math.isfinite(value) for value in energy_values):
        raise ValueError("Sander high-level or baseline energy is not finite")

    # The two Sander runs assemble the same classical term through different
    # floating-point expression trees.  At O(1e8) kcal/mol, one binary64 ULP
    # is larger than the fixed 1e-8 absolute gate.  Accept at most one ULP at
    # the largest participating scale; larger discrepancies remain failures.
    roundoff_allowance = max(math.ulp(value) for value in energy_values)
    effective_tolerance = max(energy_tolerance_kcal_mol, roundoff_allowance)
    if abs(residual) > effective_tolerance:
        raise ValueError(
            "Sander baseline total does not match high total minus high SCF: "
            f"residual={residual:.17g} kcal/mol exceeds effective tolerance "
            f"{effective_tolerance:.17g} (absolute gate "
            f"{energy_tolerance_kcal_mol:.17g}, binary64 allowance "
            f"{roundoff_allowance:.17g})"
        )
    return (
        float(high_classical),
        float(residual),
        float(effective_tolerance),
        float(roundoff_allowance),
    )


def write_label(path: Path, label: Any) -> None:
    """Publish a fully validated ``DPRCLBL1`` stream without overwriting."""
    atom_count = int(label.coordinates_angstrom.shape[0])
    coordinates = IO._binary64(label.coordinates_angstrom, (atom_count, 3))
    forces = IO._binary64(label.forces_kcal_mol_angstrom, (atom_count, 3))
    lengths = IO._binary64(label.cell_lengths_angstrom, (3,))
    angles = IO._binary64(label.cell_angles_degrees, (3,))
    IO._validate_cell(lengths, angles)
    energies = IO._binary64(
        [label.total_potential_energy_kcal_mol, label.qmmm_scf_energy_kcal_mol],
        (2,),
    )

    def writer(handle: Any) -> None:
        handle.write(
            IO.LABEL_HEADER.pack(
                IO.LABEL_MAGIC,
                IO.SCHEMA_VERSION,
                IO.ENDIAN_MARKER,
                label.frame_index,
                atom_count,
                label.extra_point_count,
                label.virtual_site_policy,
            )
        )
        for values in (energies, lengths, angles, coordinates, forces):
            handle.write(values.tobytes(order="C"))

    IO._publish_stream(path, writer)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Publish provenance atomically without replacing an existing ledger."""
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


def prepare_label(
    high: Any,
    baseline: Any,
    qmmm: dict[str, float],
    mm: dict[str, float],
    state: Any,
    pre_xtb_forces_kcal_mol_angstrom: np.ndarray,
    atom_map: list[dict[str, Any]],
    *,
    coordinate_tolerance_angstrom: float,
    baseline_energy_tolerance_kcal_mol: float,
    mm_state: Any,
) -> tuple[Any, dict[str, Any], list[int]]:
    """Construct one low-level label from matching, validated components.

    This is the shared scientific core for single-frame and batched
    publication.  No payload is returned until the high/baseline identity,
    zero-SCF baseline, cell, atom map, coordinates, and both force arrays have
    all passed their gates.
    """
    if not math.isfinite(coordinate_tolerance_angstrom) or coordinate_tolerance_angstrom < 0:
        raise ValueError("coordinate tolerance must be finite and nonnegative")
    (
        high_classical,
        baseline_energy_residual,
        baseline_energy_effective_tolerance,
        baseline_energy_roundoff_allowance,
    ) = _require_matching_baseline(
        high,
        baseline,
        energy_tolerance_kcal_mol=baseline_energy_tolerance_kcal_mol,
    )
    if qmmm["pair_reference_mm_only"] != 1.0:
        raise ValueError("xTBloom result does not prove the MM-only pair mapping")
    if state.atom_ids.size != len(atom_map):
        raise ValueError("LAMMPS dump and atom map real-atom counts differ")
    pre_xtb_forces = np.asarray(
        pre_xtb_forces_kcal_mol_angstrom, dtype=np.float64
    )
    if pre_xtb_forces.shape != state.forces_kcal_mol_angstrom.shape:
        raise ValueError("stored pre-xTB and total LAMMPS force extents differ")
    if not np.all(np.isfinite(pre_xtb_forces)):
        raise ValueError("stored pre-xTB LAMMPS forces contain a non-finite value")
    expected_types = np.asarray(
        [row["lammps_type"] for row in atom_map], dtype=np.int64
    )
    if not np.array_equal(state.atom_types, expected_types):
        raise ValueError("LAMMPS dump atom types differ from the reviewed atom map")

    amber_indices = np.asarray(
        [row["amber_id"] for row in atom_map], dtype=np.int64
    ) - 1
    if (
        np.min(amber_indices, initial=0) < 0
        or np.max(amber_indices, initial=-1) >= high.coordinates_angstrom.shape[0]
    ):
        raise ValueError("atom map contains an Amber ID outside the peer label")
    real_ids = set(int(index) + 1 for index in amber_indices)
    extra_ids = sorted(
        set(range(1, high.coordinates_angstrom.shape[0] + 1)) - real_ids
    )
    if len(extra_ids) != high.extra_point_count:
        raise ValueError("atom map and peer-label extra-point counts differ")
    # Equal high/baseline coordinates and exact classical cancellation do not
    # prove a valid geometry: both engines can share the same split-water bug.
    types_by_amber = {row["amber_id"]: row["lammps_type"] for row in atom_map}
    extra_id_set = set(extra_ids)
    waters = []
    for oxygen_id, atom_type in types_by_amber.items():
        if atom_type != EXPECTED_LAMMPS_TYPES["OW"]:
            continue
        if (types_by_amber.get(oxygen_id + 1) != EXPECTED_LAMMPS_TYPES["HW"]
                or types_by_amber.get(oxygen_id + 2) != EXPECTED_LAMMPS_TYPES["HW"]
                or oxygen_id + 3 not in extra_id_set):
            raise ValueError("TIP4P atom map must contain complete O/H1/H2/EP waters")
        waters.append((oxygen_id - 1, oxygen_id, oxygen_id + 1, oxygen_id + 2))
    if len(waters) != high.extra_point_count:
        raise ValueError("TIP4P water and extra-point counts differ")
    water_geometry = GEOMETRY.validate_tip4p_sites(high.coordinates_angstrom, waters)
    extra_indices = np.asarray(extra_ids, dtype=np.int64) - 1
    if not np.all(high.forces_kcal_mol_angstrom[extra_indices] == 0.0):
        raise ValueError("high-level label has nonzero TIP4P extra-point forces")
    if not np.all(baseline.forces_kcal_mol_angstrom[extra_indices] == 0.0):
        raise ValueError("Sander baseline has nonzero TIP4P extra-point forces")
    direct_cell = COMPARE.cell_matrix(qmmm)
    # Coordinate equivalence alone cannot detect labels from a different
    # periodic Hamiltonian if no atom happens to cross the box boundary.
    lengths, angles = GEOMETRY._cell_lengths_angles(direct_cell)
    if not np.allclose(lengths, high.cell_lengths_angstrom, rtol=0., atol=coordinate_tolerance_angstrom) or not np.allclose(
        angles, high.cell_angles_degrees, rtol=0., atol=1e-10
    ):
        raise ValueError("LAMMPS and peer-label periodic cells differ")
    residual, lattice_shifts = COMPARE.periodic_residual(
        state.coordinates_angstrom,
        high.coordinates_angstrom[amber_indices],
        direct_cell,
    )
    coordinate_error = float(np.max(np.abs(residual), initial=0.0))
    if coordinate_error > coordinate_tolerance_angstrom:
        raise ValueError(
            f"LAMMPS and peer-label coordinates differ by {coordinate_error:.17g} Angstrom"
        )

    qmmm_energy = qmmm["correction"] - mm["mm_elong"] + qmmm["elong"]
    raw_lammps_classical = qmmm["energy"] - qmmm_energy
    raw_cross_engine_classical_residual = high_classical - raw_lammps_classical
    normalized_total_energy = high_classical + qmmm_energy

    # A full-charge PPPM solve precedes POST_FORCE. Its QM-dependent forces
    # must survive subtraction, so use a separate same-engine zero-QM solve.
    for field in ("atom_ids", "atom_types"):
        if not np.array_equal(getattr(state, field), getattr(mm_state, field)):
            raise ValueError(f"zero-QM force baseline {field} differs")
    if (not np.all(np.isfinite(mm_state.forces_kcal_mol_angstrom))
            or mm_state.forces_kcal_mol_angstrom.shape != state.forces_kcal_mol_angstrom.shape):
        raise ValueError("zero-QM force baseline is malformed or non-finite")
    if not np.allclose(state.coordinates_angstrom, mm_state.coordinates_angstrom,
                       rtol=0.0, atol=coordinate_tolerance_angstrom):
        raise ValueError("zero-QM force baseline coordinates differ")
    qm_mask = state.atom_types <= 5
    if not np.all(mm_state.charges[qm_mask] == 0.0):
        raise ValueError("zero-QM force baseline retains nonzero QM charges")
    if not np.array_equal(state.charges[~qm_mask], mm_state.charges[~qm_mask]):
        raise ValueError("zero-QM force baseline changes MM charges")
    xtb_component = np.ascontiguousarray(
        state.forces_kcal_mol_angstrom - mm_state.forces_kcal_mol_angstrom,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(xtb_component)):
        raise ValueError("xTBloom component force contains a non-finite value")
    if not np.any(xtb_component != 0.0):
        raise ValueError("xTBloom component force is identically zero")
    forces = np.ascontiguousarray(
        baseline.forces_kcal_mol_angstrom, dtype=np.float64
    ).copy()
    forces[amber_indices] += xtb_component
    if not np.all(forces[extra_indices] == 0.0):
        raise ValueError("component assembly changed TIP4P extra-point force slots")
    label = IO.Label(
        frame_index=high.frame_index,
        extra_point_count=high.extra_point_count,
        virtual_site_policy=high.virtual_site_policy,
        total_potential_energy_kcal_mol=normalized_total_energy,
        qmmm_scf_energy_kcal_mol=qmmm_energy,
        coordinates_angstrom=high.coordinates_angstrom.copy(),
        forces_kcal_mol_angstrom=forces,
        cell_lengths_angstrom=high.cell_lengths_angstrom.copy(),
        cell_angles_degrees=high.cell_angles_degrees.copy(),
    )
    report = {
        "schema_version": 1,
        "status": "published-component-assembled",
        "scope": "ETP/ETH Sander-baseline plus LAMMPS/xTBloom component label",
        "frame_index": high.frame_index,
        "units": {"energy": "kcal/mol", "force": "kcal/mol/angstrom"},
        "energy_reconstruction": {
            "formula": "E_xTB = fix_correction - E_MM_KSpace + E_full_KSpace",
            "fix_correction_kcal_mol": qmmm["correction"],
            "mm_kspace_energy_kcal_mol": mm["mm_elong"],
            "full_kspace_energy_kcal_mol": qmmm["elong"],
            "xtb_qmmm_energy_kcal_mol": qmmm_energy,
            "raw_lammps_total_potential_energy_kcal_mol": qmmm["energy"],
            "normalized_total_potential_energy_kcal_mol": normalized_total_energy,
            "raw_lammps_classical_energy_kcal_mol": raw_lammps_classical,
            "shared_sander_classical_energy_kcal_mol": high_classical,
        },
        "checks": {
            "high_and_sander_baseline_identity_matches_bitwise": True,
            "sander_baseline_scf_energy_is_exactly_zero": True,
            "sander_baseline_total_matches_high_total_minus_high_scf": True,
            "pair_reference_is_mm_only": True,
            "coordinates_periodically_equivalent": True,
            "total_and_pre_xtb_force_arrays_are_complete": True,
            "same_engine_zero_qm_force_baseline_is_validated": True,
            "tip4p_extra_point_force_slots_exactly_zero": True,
            "sander_tip4p_whole_water_geometry_validated": True,
        },
        "sander_water_geometry": water_geometry,
        "sander_baseline_energy_residual_kcal_mol": baseline_energy_residual,
        "sander_baseline_energy_absolute_tolerance_kcal_mol": (
            baseline_energy_tolerance_kcal_mol
        ),
        "sander_baseline_energy_binary64_roundoff_allowance_kcal_mol": (
            baseline_energy_roundoff_allowance
        ),
        "sander_baseline_energy_effective_tolerance_kcal_mol": (
            baseline_energy_effective_tolerance
        ),
        "coordinate_maximum_absolute_residual_angstrom": coordinate_error,
        "maximum_absolute_lattice_shift_index": float(
            np.max(np.abs(lattice_shifts), initial=0.0)
        ),
        "raw_cross_engine_classical_energy_residual_kcal_mol": (
            raw_cross_engine_classical_residual
        ),
        "force_assembly": {
            "formula": "F_low = F_Sander_zero_QMMM + (F_LAMMPS_total - F_LAMMPS_zero_QM)",
            "baseline_source": "matching Sander zero-QM/MM label",
            "electronic_component_source": "LAMMPS total minus matching zero-QM-charge classical force",
            "pre_fix_force_is_diagnostic_only": True,
            "cross_engine_total_force_subtraction_used": False,
            "maximum_absolute_xtb_component_kcal_mol_angstrom": float(
                np.max(np.abs(xtb_component), initial=0.0)
            ),
            "rms_xtb_component_kcal_mol_angstrom": float(
                np.sqrt(np.mean(xtb_component**2))
            ),
            "qualification": "production-eligible-component-assembly",
        },
    }
    return label, report, extra_ids


def require_published_label(
    path: Path, expected: Any, extra_ids: list[int]
) -> Any:
    """Read a published label and require exact expected binary64 content."""
    published = IO.read_label(path)
    scalar_fields = (
        "frame_index",
        "extra_point_count",
        "virtual_site_policy",
        "total_potential_energy_kcal_mol",
        "qmmm_scf_energy_kcal_mol",
    )
    for name in scalar_fields:
        if getattr(published, name) != getattr(expected, name):
            raise RuntimeError(f"published xTBloom label changed {name}")
    for name in (
        "coordinates_angstrom",
        "forces_kcal_mol_angstrom",
        "cell_lengths_angstrom",
        "cell_angles_degrees",
    ):
        actual = np.ascontiguousarray(getattr(published, name), dtype=IO.FLOAT64)
        reference = np.ascontiguousarray(getattr(expected, name), dtype=IO.FLOAT64)
        if not np.array_equal(actual.view(np.uint64), reference.view(np.uint64)):
            raise RuntimeError(f"published xTBloom label changed {name}")
    if not np.all(
        published.forces_kcal_mol_angstrom[np.asarray(extra_ids) - 1] == 0.0
    ):
        raise RuntimeError("published xTBloom extra-point force slots are nonzero")
    return published


def publish(
    high_label_path: Path,
    baseline_label_path: Path,
    qmmm_result_path: Path,
    mm_result_path: Path,
    dump_path: Path,
    atom_map_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    coordinate_tolerance_angstrom: float,
    baseline_energy_tolerance_kcal_mol: float,
    mm_dump_path: Path,
) -> dict[str, Any]:
    """Validate one run-zero result and publish its component-assembled peer."""
    high = IO.read_label(high_label_path)
    baseline = IO.read_label(baseline_label_path)
    qmmm = parse_scalars(qmmm_result_path, QMMM_FIELDS)
    mm = parse_scalars(mm_result_path, {"mm_elong"})
    state, pre_xtb_forces = parse_component_dump(dump_path)
    mm_state, _ = parse_component_dump(mm_dump_path, baseline_only=True)
    # Single-frame inputs must have the same timestep and exact box records.
    headers = [p.read_text().split("ITEM: ATOMS", 1)[0] for p in (dump_path, mm_dump_path)]
    if headers[0] != headers[1]:
        raise ValueError("zero-QM baseline timestep or cell differs")
    atom_map = read_atom_map(atom_map_path)
    label, report, extra_ids = prepare_label(
        high,
        baseline,
        qmmm,
        mm,
        state,
        pre_xtb_forces,
        atom_map,
        mm_state=mm_state,
        coordinate_tolerance_angstrom=coordinate_tolerance_angstrom,
        baseline_energy_tolerance_kcal_mol=baseline_energy_tolerance_kcal_mol,
    )
    write_label(output_path, label)
    require_published_label(output_path, label, extra_ids)
    report["artifacts"] = {
        "high_level_peer": artifact(high_label_path),
        "sander_zero_qmmm_baseline": artifact(baseline_label_path),
        "qmmm_result": artifact(qmmm_result_path),
        "mm_result": artifact(mm_result_path),
        "lammps_dump": artifact(dump_path),
        "lammps_zero_qm_dump": artifact(mm_dump_path),
        "atom_map": artifact(atom_map_path),
        "output": artifact(output_path),
    }
    write_json_new(manifest_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-label", type=Path, required=True)
    parser.add_argument("--baseline-label", type=Path, required=True)
    parser.add_argument("--qmmm-result", type=Path, required=True)
    parser.add_argument("--mm-result", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--mm-dump", type=Path, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate-tolerance-angstrom", type=float, default=5.0e-10)
    parser.add_argument(
        "--baseline-energy-tolerance-kcal-mol", type=float, default=1.0e-8
    )
    arguments = parser.parse_args()
    report = publish(
        arguments.high_label,
        arguments.baseline_label,
        arguments.qmmm_result,
        arguments.mm_result,
        arguments.dump,
        arguments.atom_map,
        arguments.output,
        arguments.manifest,
        mm_dump_path=arguments.mm_dump,
        coordinate_tolerance_angstrom=arguments.coordinate_tolerance_angstrom,
        baseline_energy_tolerance_kcal_mol=(
            arguments.baseline_energy_tolerance_kcal_mol
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
