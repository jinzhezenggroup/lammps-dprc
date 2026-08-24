#!/usr/bin/env python3
"""Qualify one production LAMMPS/xTBloom ETP/ETH force evaluation.

The independent libxTB and AmberTools/xTB paths remain valuable energy,
charge, and point-force diagnostics.  Their custom periodic callbacks do not,
however, provide a usable force oracle when the fixed ``b + A*q`` operator
changes the self-consistent Mulliken charges: the analytic QM force must first
pass a total-energy finite difference.  This tool therefore gates the
production xTBloom force against its own public variational energy, while
retaining the cross-engine discrepancies explicitly instead of hiding them.
"""

from __future__ import annotations

import argparse
import hashlib
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
    "dprc_binary64_io_for_lammps_oracle", ROOT / "tools/dprc_binary64_io.py"
)
if IO_SPEC is None or IO_SPEC.loader is None:
    raise RuntimeError("could not load tools/dprc_binary64_io.py")
IO = importlib.util.module_from_spec(IO_SPEC)
IO_SPEC.loader.exec_module(IO)

EXPECTED_REAL_ATOMS = 8938
EXPECTED_QM_ATOMS = 16
EXPECTED_RESULT_FIELDS = {
    "energy",
    "correction",
    "pxx",
    "pyy",
    "pzz",
    "pxy",
    "pxz",
    "pyz",
    "lx",
    "ly",
    "lz",
    "xy",
    "xz",
    "yz",
}
EXPECTED_DUMP_COLUMNS = ("id", "type", "q", "xu", "yu", "zu", "fx", "fy", "fz")


class LammpsState(NamedTuple):
    """One sorted, full-system LAMMPS state in real units."""

    atom_ids: np.ndarray
    atom_types: np.ndarray
    charges: np.ndarray
    coordinates_angstrom: np.ndarray
    forces_kcal_mol_angstrom: np.ndarray


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for evidence identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    """Describe one immutable input to the comparison."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def parse_result(path: Path) -> dict[str, float]:
    """Parse the exact scalar record emitted by the run-zero input."""
    values: dict[str, float] = {}
    for token in path.read_text(encoding="utf-8").split():
        key, separator, raw_value = token.partition("=")
        if not separator or key in values:
            raise ValueError(f"invalid or duplicate LAMMPS result token {token!r}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite LAMMPS result {key}={value}")
        values[key] = value
    if set(values) != EXPECTED_RESULT_FIELDS:
        raise ValueError(
            "LAMMPS result fields differ: "
            f"expected {sorted(EXPECTED_RESULT_FIELDS)}, found {sorted(values)}"
        )
    return values


def parse_dump(path: Path) -> LammpsState:
    """Read one complete custom dump and require contiguous sorted atom IDs."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        atoms_header = next(
            index for index, line in enumerate(lines) if line.startswith("ITEM: ATOMS ")
        )
    except StopIteration as error:
        raise ValueError("LAMMPS dump has no atom table") from error
    columns = tuple(lines[atoms_header].split()[2:])
    if columns != EXPECTED_DUMP_COLUMNS:
        raise ValueError(
            f"LAMMPS dump columns must be {EXPECTED_DUMP_COLUMNS}, found {columns}"
        )
    try:
        count_header = lines.index("ITEM: NUMBER OF ATOMS")
        atom_count = int(lines[count_header + 1])
    except (IndexError, ValueError) as error:
        raise ValueError("LAMMPS dump atom count is invalid") from error
    atom_lines = lines[atoms_header + 1 :]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"LAMMPS dump declares {atom_count} atoms but contains {len(atom_lines)} rows"
        )
    table = np.asarray(
        [[float(value) for value in line.split()] for line in atom_lines],
        dtype=np.float64,
    )
    if table.shape != (atom_count, len(EXPECTED_DUMP_COLUMNS)) or not np.all(
        np.isfinite(table)
    ):
        raise ValueError("LAMMPS dump table is malformed or non-finite")
    order = np.argsort(table[:, 0])
    table = table[order]
    atom_ids = table[:, 0].astype(np.int64)
    expected_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    if not np.array_equal(atom_ids, expected_ids):
        raise ValueError("LAMMPS dump atom IDs are not the contiguous range 1:N")
    atom_types = table[:, 1].astype(np.int64)
    if not np.array_equal(table[:, 0], atom_ids.astype(np.float64)) or not np.array_equal(
        table[:, 1], atom_types.astype(np.float64)
    ):
        raise ValueError("LAMMPS dump atom IDs or types are not integral")
    return LammpsState(
        atom_ids,
        atom_types,
        table[:, 2].copy(),
        table[:, 3:6].copy(),
        table[:, 6:9].copy(),
    )


def read_atom_map(path: Path) -> list[tuple[int, int]]:
    """Read unique Amber-to-LAMMPS real-atom IDs in LAMMPS order."""
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        try:
            amber_column = header.index("amber_id")
            lammps_column = header.index("lammps_id")
        except ValueError as error:
            raise ValueError("atom map requires amber_id and lammps_id columns") from error
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            rows.append((int(fields[amber_column]), int(fields[lammps_column])))
    if not rows:
        raise ValueError("atom map contains no rows")
    if len({amber for amber, _ in rows}) != len(rows) or len(
        {lammps for _, lammps in rows}
    ) != len(rows):
        raise ValueError("atom map contains duplicate Amber or LAMMPS IDs")
    rows.sort(key=lambda row: row[1])
    if [lammps for _, lammps in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("atom map LAMMPS IDs are not the contiguous range 1:N")
    return rows


def cell_matrix(result: dict[str, float]) -> np.ndarray:
    """Return the restricted-triclinic direct lattice with vectors as columns."""
    matrix = np.asarray(
        [
            [result["lx"], result["xy"], result["xz"]],
            [0.0, result["ly"], result["yz"]],
            [0.0, 0.0, result["lz"]],
        ],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("LAMMPS result defines a non-positive triclinic cell")
    return matrix


def periodic_residual(
    actual: np.ndarray, reference: np.ndarray, direct_cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the nearest integer lattice shift from paired coordinates."""
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError("periodic coordinate arrays have incompatible extents")
    displacement = actual - reference
    fractional = np.linalg.solve(direct_cell, displacement.T).T
    integer_shifts = np.rint(fractional)
    residual = displacement - integer_shifts @ direct_cell.T
    return residual, integer_shifts


def error_summary(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Summarize component errors without dropping vector conservation data."""
    if actual.shape != reference.shape:
        raise ValueError("comparison arrays have incompatible extents")
    difference = np.asarray(actual - reference, dtype=np.float64)
    return {
        "maximum_absolute_component": float(np.max(np.abs(difference), initial=0.0)),
        "rms_component": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "net_difference_vector": (
            np.sum(difference, axis=0).tolist() if difference.ndim == 2 else [float(np.sum(difference))]
        ),
    }


def finite_difference_force(
    minus: dict[str, float], plus: dict[str, float], step_angstrom: float
) -> float:
    """Return the negative central derivative of total potential energy."""
    if not math.isfinite(step_angstrom) or step_angstrom <= 0.0:
        raise ValueError("finite-difference step must be positive and finite")
    return -(plus["energy"] - minus["energy"]) / (2.0 * step_angstrom)


def build_report(
    xtbloom_result: dict[str, float],
    xtbloom_state: LammpsState,
    libxtb_result: dict[str, float],
    libxtb_state: LammpsState,
    sander_label: Any,
    atom_map: list[tuple[int, int]],
    xtbloom_fd_minus: dict[str, float],
    xtbloom_fd_plus: dict[str, float],
    libxtb_fd_minus: dict[str, float],
    libxtb_fd_plus: dict[str, float],
    *,
    step_angstrom: float,
    energy_tolerance_kcal_mol: float,
    charge_tolerance_e: float,
    mm_force_tolerance_kcal_mol_angstrom: float,
    coordinate_tolerance_angstrom: float,
    finite_difference_tolerance_kcal_mol_angstrom: float,
    net_force_tolerance_kcal_mol_angstrom: float,
) -> dict[str, Any]:
    """Build the correctness ledger and retain non-gating oracle diagnostics."""
    if xtbloom_state.atom_ids.size != EXPECTED_REAL_ATOMS:
        raise ValueError(
            f"ETP/ETH comparison requires {EXPECTED_REAL_ATOMS} LAMMPS atoms"
        )
    if xtbloom_state.atom_ids.size != libxtb_state.atom_ids.size or not np.array_equal(
        xtbloom_state.atom_ids, libxtb_state.atom_ids
    ):
        raise ValueError("xTBloom and libxTB dumps have different atom identities")
    if len(atom_map) != EXPECTED_REAL_ATOMS:
        raise ValueError(f"ETP/ETH atom map requires {EXPECTED_REAL_ATOMS} rows")
    if sander_label.coordinates_angstrom.shape[0] != IO.EXPECTED_ETPETH_ATOMS:
        raise ValueError("Sander label has the wrong ETP/ETH site count")

    amber_indices = np.asarray([amber for amber, _ in atom_map], dtype=np.int64) - 1
    sander_coordinates = sander_label.coordinates_angstrom[amber_indices]
    sander_forces = sander_label.forces_kcal_mol_angstrom[amber_indices]
    direct_cell = cell_matrix(xtbloom_result)
    sander_coordinate_residual, lattice_shifts = periodic_residual(
        xtbloom_state.coordinates_angstrom, sander_coordinates, direct_cell
    )
    cross_coordinate_error = float(
        np.max(
            np.abs(
                xtbloom_state.coordinates_angstrom
                - libxtb_state.coordinates_angstrom
            ),
            initial=0.0,
        )
    )
    periodic_coordinate_error = float(
        np.max(np.abs(sander_coordinate_residual), initial=0.0)
    )

    qm_slice = slice(0, EXPECTED_QM_ATOMS)
    mm_slice = slice(EXPECTED_QM_ATOMS, EXPECTED_REAL_ATOMS)
    charge_error = float(
        np.max(
            np.abs(
                xtbloom_state.charges[qm_slice] - libxtb_state.charges[qm_slice]
            ),
            initial=0.0,
        )
    )
    cross_force = error_summary(
        xtbloom_state.forces_kcal_mol_angstrom,
        libxtb_state.forces_kcal_mol_angstrom,
    )
    cross_qm_force = error_summary(
        xtbloom_state.forces_kcal_mol_angstrom[qm_slice],
        libxtb_state.forces_kcal_mol_angstrom[qm_slice],
    )
    cross_mm_force = error_summary(
        xtbloom_state.forces_kcal_mol_angstrom[mm_slice],
        libxtb_state.forces_kcal_mol_angstrom[mm_slice],
    )
    xtbloom_fd_force = finite_difference_force(
        xtbloom_fd_minus, xtbloom_fd_plus, step_angstrom
    )
    libxtb_fd_force = finite_difference_force(
        libxtb_fd_minus, libxtb_fd_plus, step_angstrom
    )
    xtbloom_analytic_force = float(xtbloom_state.forces_kcal_mol_angstrom[0, 0])
    libxtb_analytic_force = float(libxtb_state.forces_kcal_mol_angstrom[0, 0])
    xtbloom_fd_error = abs(xtbloom_fd_force - xtbloom_analytic_force)
    libxtb_fd_error = abs(libxtb_fd_force - libxtb_analytic_force)
    xtbloom_net_force = np.sum(
        xtbloom_state.forces_kcal_mol_angstrom, axis=0
    )

    checks = {
        "xTBloom_and_libxTB_coordinates_identical": (
            cross_coordinate_error <= coordinate_tolerance_angstrom
        ),
        "LAMMPS_and_Sander_coordinates_periodically_equivalent": (
            periodic_coordinate_error <= coordinate_tolerance_angstrom
        ),
        "xTBloom_energy_matches_libxTB": (
            abs(xtbloom_result["energy"] - libxtb_result["energy"])
            <= energy_tolerance_kcal_mol
        ),
        "xTBloom_QM_charges_match_libxTB": charge_error <= charge_tolerance_e,
        "xTBloom_MM_forces_match_libxTB": (
            cross_mm_force["maximum_absolute_component"]
            <= mm_force_tolerance_kcal_mol_angstrom
        ),
        "xTBloom_QM_force_matches_total_energy_finite_difference": (
            xtbloom_fd_error <= finite_difference_tolerance_kcal_mol_angstrom
        ),
        "xTBloom_total_force_is_conserved": (
            float(np.max(np.abs(xtbloom_net_force), initial=0.0))
            <= net_force_tolerance_kcal_mol_angstrom
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "scope": "one-frame ETP/ETH production LAMMPS xTBloom QM/MM qualification",
        "units": {
            "energy": "kcal/mol",
            "force": "kcal/mol/angstrom",
            "charge": "elementary_charge",
            "coordinate": "angstrom",
        },
        "tolerances": {
            "energy_kcal_mol": energy_tolerance_kcal_mol,
            "charge_e": charge_tolerance_e,
            "MM_force_kcal_mol_angstrom": mm_force_tolerance_kcal_mol_angstrom,
            "coordinate_angstrom": coordinate_tolerance_angstrom,
            "finite_difference_force_kcal_mol_angstrom": (
                finite_difference_tolerance_kcal_mol_angstrom
            ),
            "net_force_kcal_mol_angstrom": net_force_tolerance_kcal_mol_angstrom,
        },
        "checks": checks,
        "coordinates": {
            "xTBloom_vs_libxTB_maximum_absolute_angstrom": cross_coordinate_error,
            "LAMMPS_vs_Sander_periodic_maximum_absolute_angstrom": (
                periodic_coordinate_error
            ),
            "maximum_absolute_lattice_shift_index": float(
                np.max(np.abs(lattice_shifts), initial=0.0)
            ),
        },
        "energies": {
            "xTBloom_total_kcal_mol": xtbloom_result["energy"],
            "libxTB_total_kcal_mol": libxtb_result["energy"],
            "Sander_xTB_total_kcal_mol": sander_label.total_potential_energy_kcal_mol,
            "xTBloom_minus_libxTB_kcal_mol": (
                xtbloom_result["energy"] - libxtb_result["energy"]
            ),
            "xTBloom_minus_Sander_xTB_kcal_mol": (
                xtbloom_result["energy"]
                - sander_label.total_potential_energy_kcal_mol
            ),
        },
        "QM_charges": {
            "maximum_absolute_xTBloom_minus_libxTB_e": charge_error,
            "xTBloom": xtbloom_state.charges[qm_slice].tolist(),
            "libxTB": libxtb_state.charges[qm_slice].tolist(),
        },
        "forces": {
            "xTBloom_minus_libxTB_all_real_atoms": cross_force,
            "xTBloom_minus_libxTB_QM_atoms": cross_qm_force,
            "xTBloom_minus_libxTB_MM_atoms": cross_mm_force,
            "xTBloom_minus_Sander_xTB_real_atoms": error_summary(
                xtbloom_state.forces_kcal_mol_angstrom, sander_forces
            ),
            "libxTB_minus_Sander_xTB_real_atoms": error_summary(
                libxtb_state.forces_kcal_mol_angstrom, sander_forces
            ),
            "xTBloom_net_vector": xtbloom_net_force.tolist(),
        },
        "finite_difference": {
            "coordinate": "LAMMPS atom 1 x",
            "step_angstrom": step_angstrom,
            "xTBloom": {
                "analytic_force_kcal_mol_angstrom": xtbloom_analytic_force,
                "finite_difference_force_kcal_mol_angstrom": xtbloom_fd_force,
                "absolute_error_kcal_mol_angstrom": xtbloom_fd_error,
            },
            "libxTB": {
                "analytic_force_kcal_mol_angstrom": libxtb_analytic_force,
                "finite_difference_force_kcal_mol_angstrom": libxtb_fd_force,
                "absolute_error_kcal_mol_angstrom": libxtb_fd_error,
            },
        },
        "oracle_interpretation": {
            "libxTB_and_Sander_energy_charge_MM_force_scope": "cross-engine diagnostic",
            "libxTB_and_Sander_QM_force_scope": "not a production force oracle",
            "reason": (
                "the custom fixed periodic-response libxTB force fails the same "
                "total-energy central difference that the public xTBloom force passes; "
                "the discrepancy is retained explicitly and is not converted into a "
                "weaker force tolerance"
            ),
        },
    }


def write_json_new(payload: dict[str, Any], path: Path) -> None:
    """Atomically publish evidence without replacing an existing record."""
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
    parser.add_argument("--xtbloom-result", type=Path, required=True)
    parser.add_argument("--xtbloom-dump", type=Path, required=True)
    parser.add_argument("--libxtb-result", type=Path, required=True)
    parser.add_argument("--libxtb-dump", type=Path, required=True)
    parser.add_argument("--sander-label", type=Path, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--xtbloom-fd-minus-result", type=Path, required=True)
    parser.add_argument("--xtbloom-fd-plus-result", type=Path, required=True)
    parser.add_argument("--libxtb-fd-minus-result", type=Path, required=True)
    parser.add_argument("--libxtb-fd-plus-result", type=Path, required=True)
    parser.add_argument("--fd-step-angstrom", type=float, default=5.0e-4)
    parser.add_argument("--energy-tolerance-kcal-mol", type=float, default=1.0e-3)
    parser.add_argument("--charge-tolerance-e", type=float, default=5.0e-7)
    parser.add_argument(
        "--mm-force-tolerance-kcal-mol-angstrom", type=float, default=1.0e-4
    )
    parser.add_argument("--coordinate-tolerance-angstrom", type=float, default=5.0e-12)
    parser.add_argument(
        "--finite-difference-tolerance-kcal-mol-angstrom",
        type=float,
        default=3.0e-3,
    )
    parser.add_argument(
        "--net-force-tolerance-kcal-mol-angstrom", type=float, default=1.0e-6
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    input_paths = {
        "xtbloom_result": arguments.xtbloom_result,
        "xtbloom_dump": arguments.xtbloom_dump,
        "libxtb_result": arguments.libxtb_result,
        "libxtb_dump": arguments.libxtb_dump,
        "sander_label": arguments.sander_label,
        "atom_map": arguments.atom_map,
        "xtbloom_fd_minus_result": arguments.xtbloom_fd_minus_result,
        "xtbloom_fd_plus_result": arguments.xtbloom_fd_plus_result,
        "libxtb_fd_minus_result": arguments.libxtb_fd_minus_result,
        "libxtb_fd_plus_result": arguments.libxtb_fd_plus_result,
    }
    try:
        report = build_report(
            parse_result(arguments.xtbloom_result),
            parse_dump(arguments.xtbloom_dump),
            parse_result(arguments.libxtb_result),
            parse_dump(arguments.libxtb_dump),
            IO.read_label(arguments.sander_label),
            read_atom_map(arguments.atom_map),
            parse_result(arguments.xtbloom_fd_minus_result),
            parse_result(arguments.xtbloom_fd_plus_result),
            parse_result(arguments.libxtb_fd_minus_result),
            parse_result(arguments.libxtb_fd_plus_result),
            step_angstrom=arguments.fd_step_angstrom,
            energy_tolerance_kcal_mol=arguments.energy_tolerance_kcal_mol,
            charge_tolerance_e=arguments.charge_tolerance_e,
            mm_force_tolerance_kcal_mol_angstrom=(
                arguments.mm_force_tolerance_kcal_mol_angstrom
            ),
            coordinate_tolerance_angstrom=arguments.coordinate_tolerance_angstrom,
            finite_difference_tolerance_kcal_mol_angstrom=(
                arguments.finite_difference_tolerance_kcal_mol_angstrom
            ),
            net_force_tolerance_kcal_mol_angstrom=(
                arguments.net_force_tolerance_kcal_mol_angstrom
            ),
        )
        report["inputs"] = {
            name: artifact(path) for name, path in input_paths.items()
        }
        write_json_new(report, arguments.output)
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0 if report["status"] == "passed" else 1
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"LAMMPS/xTB oracle comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
