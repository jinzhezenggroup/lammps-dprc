#!/usr/bin/env python3
"""Publish component-assembled xTBloom labels and PBE0-minus-xTB corrections.

The input is a synchronized, restraint-free LAMMPS ``rerun`` of every
production umbrella trajectory.  Per-window force dumps and scalar tables are
joined to the global frame manifest, while high-level labels are recovered
from their round-robin GPU shards.  Outputs use global frame indices and are
safe to resume: an existing binary is accepted only when its complete payload
matches the independently reconstructed expected record bit-for-bit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TextIO

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


PUBLISH = load_module(
    "publish_xtbloom_label_for_bulk", ROOT / "tools/publish_xtbloom_label.py"
)
CORRECTION = load_module(
    "dprc_correction_io_for_bulk", ROOT / "tools/dprc_correction_io.py"
)

QMMM_COLUMNS = (
    "step",
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
)
MM_COLUMNS = ("step", "mm_elong")
EXPECTED_DUMP_PREFIX = ("id", "type", "q")
EXPECTED_DUMP_SUFFIX = (
    "fx",
    "fy",
    "fz",
    "f_pre_xtb[1]",
    "f_pre_xtb[2]",
    "f_pre_xtb[3]",
)


class RerunFrame(NamedTuple):
    """One complete force state from a native LAMMPS rerun dump."""

    timestep: int
    cell: dict[str, float]
    state: Any
    pre_xtb_forces_kcal_mol_angstrom: np.ndarray


def _required_line(handle: TextIO, context: str) -> str:
    """Read one required line and diagnose a truncated native dump."""
    line = handle.readline()
    if not line:
        raise ValueError(f"LAMMPS rerun dump ended while reading {context}")
    return line.rstrip("\n")


def _parse_box(header: str, rows: Sequence[str]) -> dict[str, float]:
    """Recover restricted-triclinic lengths and tilts from dump bounds."""
    tokens = header.split()[3:]
    triclinic = tokens[:3] == ["xy", "xz", "yz"]
    values = [[float(value) for value in row.split()] for row in rows]
    if triclinic:
        if any(len(row) != 3 for row in values):
            raise ValueError("triclinic LAMMPS rerun bounds require three values")
        xlo_bound, xhi_bound, xy = values[0]
        ylo_bound, yhi_bound, xz = values[1]
        zlo, zhi, yz = values[2]
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
    else:
        if any(len(row) != 2 for row in values):
            raise ValueError("orthogonal LAMMPS rerun bounds require two values")
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = values
        xy = xz = yz = 0.0
    result = {
        "lx": xhi - xlo,
        "ly": yhi - ylo,
        "lz": zhi - zlo,
        "xy": xy,
        "xz": xz,
        "yz": yz,
    }
    direct = PUBLISH.COMPARE.cell_matrix(result)
    if not math.isfinite(float(np.linalg.det(direct))):
        raise ValueError("LAMMPS rerun dump cell is non-finite")
    return result


def read_rerun_dump(path: Path, *, baseline_only: bool = False) -> Iterator[RerunFrame]:
    """Stream validated force states without retaining a full trajectory."""
    with path.open("r", encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                return
            if marker.rstrip("\n") != "ITEM: TIMESTEP":
                raise ValueError(f"unexpected LAMMPS rerun record in {path}: {marker!r}")
            timestep = int(_required_line(handle, "timestep"))
            if _required_line(handle, "atom-count header") != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"LAMMPS rerun atom-count header is invalid: {path}")
            atom_count = int(_required_line(handle, "atom count"))
            if atom_count < 1:
                raise ValueError(f"LAMMPS rerun atom count is non-positive: {path}")
            box_header = _required_line(handle, "box header")
            if not box_header.startswith("ITEM: BOX BOUNDS "):
                raise ValueError(f"LAMMPS rerun box header is invalid: {path}")
            cell = _parse_box(
                box_header,
                [_required_line(handle, "box bounds") for _ in range(3)],
            )
            atom_header = _required_line(handle, "atom header")
            if not atom_header.startswith("ITEM: ATOMS "):
                raise ValueError(f"LAMMPS rerun atom header is invalid: {path}")
            columns = tuple(atom_header.split()[2:])
            suffix = ("fx", "fy", "fz") if baseline_only else EXPECTED_DUMP_SUFFIX
            width = 9 if baseline_only else 12
            if (
                columns[:3] != EXPECTED_DUMP_PREFIX
                or columns[-len(suffix) :] != suffix
                or columns[3:6] not in (("x", "y", "z"), ("xu", "yu", "zu"))
                or len(columns) != width
            ):
                raise ValueError(
                    "LAMMPS rerun dump columns must be id type q, one supported "
                    "coordinate triplet, total force, and stored pre-xTB "
                    f"force; found {columns}"
                )
            table = np.empty((atom_count, width), dtype=np.float64)
            for index in range(atom_count):
                fields = _required_line(handle, "atom row").split()
                if len(fields) != width:
                    raise ValueError(f"LAMMPS rerun atom row is malformed: {path}")
                table[index] = [float(value) for value in fields]
            if not np.all(np.isfinite(table)):
                raise ValueError(f"LAMMPS rerun atom table is non-finite: {path}")
            order = np.argsort(table[:, 0])
            table = table[order]
            atom_ids = table[:, 0].astype(np.int64)
            atom_types = table[:, 1].astype(np.int64)
            if not np.array_equal(atom_ids, np.arange(1, atom_count + 1)):
                raise ValueError(f"LAMMPS rerun atom IDs are not contiguous: {path}")
            if not np.array_equal(table[:, 0], atom_ids.astype(np.float64)):
                raise ValueError(f"LAMMPS rerun atom IDs are not integral: {path}")
            if not np.array_equal(table[:, 1], atom_types.astype(np.float64)):
                raise ValueError(f"LAMMPS rerun atom types are not integral: {path}")
            yield RerunFrame(
                timestep,
                cell,
                PUBLISH.COMPARE.LammpsState(
                    atom_ids,
                    atom_types,
                    table[:, 2].copy(),
                    table[:, 3:6].copy(),
                    table[:, 6:9].copy(),
                ),
                table[:, 9:12].copy(),
            )


def parse_table(path: Path, expected_columns: tuple[str, ...]) -> dict[int, dict[str, float]]:
    """Read one exact finite scalar table keyed by unique absolute timestep."""
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = _required_line(handle, "scalar-table header").lstrip("# ").split()
        if tuple(header) != expected_columns:
            raise ValueError(
                f"scalar table {path} columns differ: expected {expected_columns}, "
                f"found {tuple(header)}"
            )
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != len(expected_columns):
                raise ValueError(f"scalar table {path}:{line_number} has the wrong width")
            try:
                timestep = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(
                    f"scalar table {path}:{line_number} contains an invalid value"
                ) from error
            if timestep in rows:
                raise ValueError(f"scalar table {path} repeats timestep {timestep}")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"scalar table {path}:{line_number} is non-finite")
            rows[timestep] = dict(zip(expected_columns[1:], values, strict=True))
    if not rows:
        raise ValueError(f"scalar table contains no data rows: {path}")
    return rows


def _bitwise_array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare complete binary64 arrays by stored bit pattern."""
    left = np.ascontiguousarray(first, dtype=PUBLISH.IO.FLOAT64)
    right = np.ascontiguousarray(second, dtype=PUBLISH.IO.FLOAT64)
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def require_correction(path: Path, expected: Any) -> Any:
    """Require an existing correction to match the expected global record."""
    actual = CORRECTION.read_correction(path)
    for name in (
        "frame_index",
        "extra_point_count",
        "virtual_site_policy",
        "total_energy_kcal_mol",
        "qmmm_energy_kcal_mol",
    ):
        if getattr(actual, name) != getattr(expected, name):
            raise RuntimeError(f"published DPRc correction changed {name}: {path}")
    for name in (
        "coordinates_angstrom",
        "forces_kcal_mol_angstrom",
        "cell_lengths_angstrom",
        "cell_angles_degrees",
    ):
        if not _bitwise_array_equal(getattr(actual, name), getattr(expected, name)):
            raise RuntimeError(f"published DPRc correction changed {name}: {path}")
    return actual


def high_label_path(root: Path, global_index: int, shard_count: int) -> tuple[Path, int, int]:
    """Map one global frame to its round-robin shard and local label index."""
    zero_based = global_index - 1
    shard = zero_based % shard_count
    local = zero_based // shard_count + 1
    return root / f"shard-{shard:03d}" / f"label.{local:06d}.bin", shard, local


def validate_frame_manifest(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Read the exact global frame ledger and group records by window."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame manifest contains no frame records")
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for expected_index, record in enumerate(frames, start=1):
        if int(record.get("frame_index", -1)) != expected_index:
            raise ValueError("frame manifest indices are not the contiguous range 1:N")
        window = str(record.get("window", ""))
        if not window or str(record.get("frame_name", window)) != window:
            raise ValueError(f"frame manifest window identity is invalid at {expected_index}")
        timestep = int(record.get("timestep", -1))
        if timestep < 0:
            raise ValueError(f"frame manifest timestep is invalid at {expected_index}")
        by_window[window].append(record)
    for window, records in by_window.items():
        timesteps = [int(record["timestep"]) for record in records]
        if len(set(timesteps)) != len(timesteps):
            raise ValueError(f"frame manifest repeats a timestep for window {window}")
    if int(payload.get("frame_count", len(frames))) != len(frames):
        raise ValueError("frame manifest frame_count differs from its records")
    return payload, dict(by_window)


def _cell_matches(first: dict[str, float], second: dict[str, float]) -> bool:
    """Return whether scalar and reconstructed dump cells agree numerically."""
    return all(
        abs(first[name] - second[name]) <= 1.0e-10
        for name in ("lx", "ly", "lz", "xy", "xz", "yz")
    )


def publish_corpus(
    frame_manifest_path: Path,
    high_root: Path,
    baseline_root: Path,
    raw_root: Path,
    atom_map_path: Path,
    low_output: Path,
    correction_output: Path,
    manifest_path: Path,
    *,
    high_shards: int,
    baseline_shards: int,
    coordinate_tolerance_angstrom: float,
    baseline_energy_tolerance_kcal_mol: float,
    correction_tolerance_kcal_mol: float,
) -> dict[str, Any]:
    """Join all engines, publish resumable binaries, and return one ledger."""
    if high_shards < 1:
        raise ValueError("high-level shard count must be positive")
    if baseline_shards < 1:
        raise ValueError("baseline shard count must be positive")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace corpus manifest: {manifest_path}")
    frame_manifest, by_window = validate_frame_manifest(frame_manifest_path)
    atom_map = PUBLISH.read_atom_map(atom_map_path)
    real_ids = {int(row["amber_id"]) for row in atom_map}
    low_output.mkdir(parents=True, exist_ok=True)
    correction_output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    raw_sources: list[dict[str, Any]] = []
    maximum_coordinate_error = 0.0
    maximum_baseline_energy_residual = 0.0
    maximum_baseline_energy_roundoff_allowance = 0.0
    baseline_roundoff_limited_frame_count = 0
    maximum_xtb_component = 0.0

    for window in sorted(by_window):
        window_root = raw_root / window
        dump_path = window_root / "qmmm.dump"
        qmmm_path = window_root / "qmmm.tsv"
        mm_path = window_root / "mm.tsv"
        mm_dump_path = window_root / "mm.dump"
        qmmm_rows = parse_table(qmmm_path, QMMM_COLUMNS)
        mm_rows = parse_table(mm_path, MM_COLUMNS)
        states: dict[int, RerunFrame] = {}
        mm_states: dict[int, RerunFrame] = {}
        for state in read_rerun_dump(mm_dump_path, baseline_only=True):
            if state.timestep in mm_states:
                raise ValueError("zero-QM baseline repeats a timestep")
            mm_states[state.timestep] = state
        for state in read_rerun_dump(dump_path):
            if state.timestep in states:
                raise ValueError(
                    f"LAMMPS rerun dump repeats timestep {state.timestep}: {dump_path}"
                )
            states[state.timestep] = state
        if set(states) != set(qmmm_rows) or set(states) != set(mm_rows) or set(states) != set(mm_states):
            raise ValueError(f"rerun force and scalar timestep sets differ for {window}")

        selected_timesteps = {int(record["timestep"]) for record in by_window[window]}
        missing = selected_timesteps - set(states)
        if missing:
            raise ValueError(f"rerun output for {window} misses timesteps {sorted(missing)}")
        raw_sources.append(
            {
                "window": window,
                "rerun_frame_count": len(states),
                "selected_frame_count": len(selected_timesteps),
                "qmmm_dump": PUBLISH.artifact(dump_path),
                "zero_qm_dump": PUBLISH.artifact(mm_dump_path),
                "qmmm_scalars": PUBLISH.artifact(qmmm_path),
                "mm_scalars": PUBLISH.artifact(mm_path),
            }
        )

        for frame_record in by_window[window]:
            global_index = int(frame_record["frame_index"])
            timestep = int(frame_record["timestep"])
            rerun = states[timestep]
            mm_rerun = mm_states[timestep]
            if not _cell_matches(rerun.cell, mm_rerun.cell):
                raise ValueError("zero-QM baseline cell differs from QM/MM")
            qmmm = qmmm_rows[timestep]
            if not _cell_matches(qmmm, rerun.cell):
                raise ValueError(
                    f"rerun scalar and force-dump cells differ for {window} step {timestep}"
                )
            high_path, shard, local_index = high_label_path(
                high_root, global_index, high_shards
            )
            high_local = PUBLISH.IO.read_label(high_path)
            if high_local.frame_index != local_index:
                raise ValueError(
                    f"high-level shard label index differs for global frame {global_index}"
                )
            high = high_local._replace(frame_index=global_index)
            baseline_path, baseline_shard, baseline_local_index = high_label_path(
                baseline_root, global_index, baseline_shards
            )
            baseline_local = PUBLISH.IO.read_label(baseline_path)
            if baseline_local.frame_index != baseline_local_index:
                raise ValueError(
                    "Sander baseline shard label index differs for global frame "
                    f"{global_index}"
                )
            baseline = baseline_local._replace(frame_index=global_index)
            low, low_report, extra_ids = PUBLISH.prepare_label(
                high,
                baseline,
                qmmm,
                mm_rows[timestep],
                rerun.state,
                rerun.pre_xtb_forces_kcal_mol_angstrom,
                atom_map,
                mm_state=mm_rerun.state,
                coordinate_tolerance_angstrom=coordinate_tolerance_angstrom,
                baseline_energy_tolerance_kcal_mol=(
                    baseline_energy_tolerance_kcal_mol
                ),
            )
            low_path = low_output / f"label.{global_index:06d}.bin"
            if low_path.exists():
                published_low = PUBLISH.require_published_label(low_path, low, extra_ids)
            else:
                PUBLISH.write_label(low_path, low)
                published_low = PUBLISH.require_published_label(low_path, low, extra_ids)

            correction, cancellation_residual = CORRECTION.subtract_labels(
                high,
                published_low,
                real_ids,
                classical_tolerance_kcal_mol=correction_tolerance_kcal_mol,
                shared_classical_reference_kcal_mol=(
                    high.total_potential_energy_kcal_mol
                    - high.qmmm_scf_energy_kcal_mol
                ),
            )
            correction_path = correction_output / f"correction.{global_index:06d}.bin"
            if correction_path.exists():
                require_correction(correction_path, correction)
            else:
                CORRECTION.write_correction(correction_path, correction)
                require_correction(correction_path, correction)

            coordinate_error = float(
                low_report["coordinate_maximum_absolute_residual_angstrom"]
            )
            baseline_energy_residual = float(
                low_report["sander_baseline_energy_residual_kcal_mol"]
            )
            baseline_energy_roundoff_allowance = float(
                low_report[
                    "sander_baseline_energy_binary64_roundoff_allowance_kcal_mol"
                ]
            )
            xtb_component_maximum = float(
                low_report["force_assembly"][
                    "maximum_absolute_xtb_component_kcal_mol_angstrom"
                ]
            )
            maximum_coordinate_error = max(maximum_coordinate_error, coordinate_error)
            maximum_baseline_energy_residual = max(
                maximum_baseline_energy_residual, abs(baseline_energy_residual)
            )
            maximum_baseline_energy_roundoff_allowance = max(
                maximum_baseline_energy_roundoff_allowance,
                baseline_energy_roundoff_allowance,
            )
            if abs(baseline_energy_residual) > baseline_energy_tolerance_kcal_mol:
                baseline_roundoff_limited_frame_count += 1
            maximum_xtb_component = max(maximum_xtb_component, xtb_component_maximum)
            records.append(
                {
                    "frame_index": global_index,
                    "window": window,
                    "timestep": timestep,
                    "high_level_shard": shard,
                    "high_level_local_index": local_index,
                    "high_level_label": PUBLISH.artifact(high_path),
                    "sander_baseline_shard": baseline_shard,
                    "sander_baseline_local_index": baseline_local_index,
                    "sander_zero_qmmm_baseline": PUBLISH.artifact(baseline_path),
                    "low_level_label": PUBLISH.artifact(low_path),
                    "correction": PUBLISH.artifact(correction_path),
                    "coordinate_maximum_absolute_residual_angstrom": coordinate_error,
                    "sander_baseline_energy_residual_kcal_mol": (
                        baseline_energy_residual
                    ),
                    "sander_baseline_energy_binary64_roundoff_allowance_kcal_mol": (
                        baseline_energy_roundoff_allowance
                    ),
                    "correction_classical_cancellation_residual_kcal_mol": (
                        cancellation_residual
                    ),
                    "maximum_absolute_xtb_component_kcal_mol_angstrom": (
                        xtb_component_maximum
                    ),
                    "force_assembly_qualification": "production-eligible-component-assembly",
                    "sander_water_geometry": low_report["sander_water_geometry"],
                }
            )

    records.sort(key=lambda record: int(record["frame_index"]))
    if [record["frame_index"] for record in records] != list(
        range(1, len(frame_manifest["frames"]) + 1)
    ):
        raise RuntimeError("published corpus does not cover every global frame exactly once")
    manifest = {
        "schema_version": 1,
        "status": "component-assembled",
        "target": "periodic PBE0 QM/MM minus xTBloom GFN2-xTB QM/MM",
        "frame_count": len(records),
        "window_count": len(by_window),
        "high_level_shard_count": high_shards,
        "sander_baseline_shard_count": baseline_shards,
        "mapping": {
            "high_level_shard": "(global_frame_index - 1) modulo shard_count",
            "high_level_local_index": (
                "floor((global_frame_index - 1) / shard_count) + 1"
            ),
            "sander_baseline_shard": (
                "(global_frame_index - 1) modulo baseline_shard_count"
            ),
        },
        "checks": {
            "every_selected_frame_has_synchronized_rerun_data": True,
            "high_and_sander_baseline_identity_matches_bitwise": True,
            "sander_baseline_scf_energy_is_exactly_zero": True,
            "sander_baseline_energy_uses_absolute_or_one_ulp_gate": True,
            "xTBloom_component_uses_total_minus_same_engine_zero_qm_force": True,
            "cross_engine_total_force_subtraction_is_absent": True,
            "high_and_component_assembled_low_coordinates_match_bitwise": True,
            "tip4p_extra_point_force_slots_are_zero": True,
            "sander_tip4p_whole_water_geometry_validated": True,
            "correction_classical_energy_cancels": True,
        },
        "maximum_coordinate_residual_angstrom": maximum_coordinate_error,
        "maximum_sander_baseline_energy_residual_kcal_mol": (
            maximum_baseline_energy_residual
        ),
        "maximum_sander_baseline_energy_binary64_roundoff_allowance_kcal_mol": (
            maximum_baseline_energy_roundoff_allowance
        ),
        "sander_baseline_roundoff_limited_frame_count": (
            baseline_roundoff_limited_frame_count
        ),
        "maximum_absolute_xtb_component_kcal_mol_angstrom": maximum_xtb_component,
        "artifacts": {
            "frame_manifest": PUBLISH.artifact(frame_manifest_path),
            "atom_map": PUBLISH.artifact(atom_map_path),
            "raw_sources": raw_sources,
        },
        "records": records,
        "limitations": [
            "corpus assembly alone does not qualify a trained model or a free-energy result",
            "trajectory stability, ensemble deviation, and free-energy closure remain separate gates",
        ],
    }
    PUBLISH.write_json_new(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--high-root", type=Path, required=True)
    parser.add_argument("--high-shards", type=int, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-shards", type=int, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--low-output", type=Path, required=True)
    parser.add_argument("--correction-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate-tolerance-angstrom", type=float, default=5.0e-10)
    parser.add_argument(
        "--baseline-energy-tolerance-kcal-mol", type=float, default=1.0e-8
    )
    parser.add_argument("--correction-tolerance-kcal-mol", type=float, default=1.0e-8)
    arguments = parser.parse_args()
    manifest = publish_corpus(
        arguments.frame_manifest,
        arguments.high_root,
        arguments.baseline_root,
        arguments.raw_root,
        arguments.atom_map,
        arguments.low_output,
        arguments.correction_output,
        arguments.manifest,
        high_shards=arguments.high_shards,
        baseline_shards=arguments.baseline_shards,
        coordinate_tolerance_angstrom=arguments.coordinate_tolerance_angstrom,
        baseline_energy_tolerance_kcal_mol=(
            arguments.baseline_energy_tolerance_kcal_mol
        ),
        correction_tolerance_kcal_mol=arguments.correction_tolerance_kcal_mol,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
