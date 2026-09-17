#!/usr/bin/env python3
"""Build compact DeePMD systems from full periodic ``DPRCCOR1`` records.

Selection occurs only after high-minus-low subtraction.  Every sample retains
the 16 QM atoms plus complete water molecules with at least one real atom
within the configured cutoff of a QM atom under the triclinic minimum-image
convention.  The resulting systems are nonperiodic compact payloads; the
manifest preserves their periodic source identity and compact-to-Amber map.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TYPE_MAP = ["P", "O", "C", "H", "OW", "HW"]
SOURCE_TO_MODEL_TYPE = {
    "p5": "P",
    "o": "O",
    "os": "O",
    "c3": "C",
    "h1": "H",
    "OW": "OW",
    "HW": "HW",
}
KCAL_PER_MOL_PER_EV = 23.06054783061903


def load_module(name: str, path: Path) -> Any:
    """Load one repository tool without installation side effects."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORRECTION_IO = load_module(
    "dprc_correction_io_for_dataset", ROOT / "tools/dprc_correction_io.py"
)


class AtomMapRow(NamedTuple):
    """One real atom and its compact-model species identity."""

    amber_id: int
    lammps_id: int
    molecule_id: int
    residue: str
    atom: str
    model_type: str


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    """Describe one immutable dataset input or output."""
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def read_atom_map(path: Path) -> list[AtomMapRow]:
    """Read exact real-atom, molecule, and species identities."""
    rows: list[AtomMapRow] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        required = ("amber_id", "lammps_id", "molecule_id", "residue", "atom", "type")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"atom map is missing columns: {', '.join(missing)}")
        columns = {name: header.index(name) for name in required}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            source_type = fields[columns["type"]]
            if source_type not in SOURCE_TO_MODEL_TYPE:
                raise ValueError(f"unsupported ETP/ETH atom type {source_type!r}")
            rows.append(
                AtomMapRow(
                    int(fields[columns["amber_id"]]),
                    int(fields[columns["lammps_id"]]),
                    int(fields[columns["molecule_id"]]),
                    fields[columns["residue"]],
                    fields[columns["atom"]],
                    SOURCE_TO_MODEL_TYPE[source_type],
                )
            )
    rows.sort(key=lambda row: row.lammps_id)
    if [row.lammps_id for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("atom map LAMMPS IDs are not the contiguous range 1:N")
    if len({row.amber_id for row in rows}) != len(rows):
        raise ValueError("atom map contains duplicate Amber IDs")
    return rows


def cell_matrix(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Construct a right-handed cell with lattice vectors as columns."""
    a, b, c = (float(value) for value in lengths)
    alpha, beta, gamma = np.deg2rad(angles)
    sin_gamma = math.sin(float(gamma))
    if abs(sin_gamma) <= 1.0e-12:
        raise ValueError("periodic cell gamma angle is singular")
    vector_a = np.asarray([a, 0.0, 0.0])
    vector_b = np.asarray([b * math.cos(float(gamma)), b * sin_gamma, 0.0])
    cx = c * math.cos(float(beta))
    cy = c * (
        math.cos(float(alpha)) - math.cos(float(beta)) * math.cos(float(gamma))
    ) / sin_gamma
    cz_squared = c * c - cx * cx - cy * cy
    if cz_squared <= 1.0e-10:
        raise ValueError("periodic cell has non-positive volume")
    vector_c = np.asarray([cx, cy, math.sqrt(cz_squared)])
    return np.column_stack((vector_a, vector_b, vector_c))


def minimum_image(displacement: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return nearest-image displacement vectors for a triclinic cell."""
    fractional = np.linalg.solve(cell, np.asarray(displacement).T).T
    return (fractional - np.rint(fractional)) @ cell.T


def compact_sample(
    correction: Any,
    atom_map: list[AtomMapRow],
    *,
    cutoff_angstrom: float,
    qm_atom_count: int = 16,
) -> dict[str, Any]:
    """Select and unwrap one complete-molecule compact DPRc sample."""
    if not math.isfinite(cutoff_angstrom) or cutoff_angstrom <= 0.0:
        raise ValueError("compact cutoff must be positive and finite")
    if len(atom_map) < qm_atom_count:
        raise ValueError("atom map contains fewer atoms than the QM region")
    amber_indices = np.asarray([row.amber_id for row in atom_map], dtype=np.int64) - 1
    coordinates = correction.coordinates_angstrom[amber_indices]
    forces = correction.forces_kcal_mol_angstrom[amber_indices]
    cell = cell_matrix(correction.cell_lengths_angstrom, correction.cell_angles_degrees)

    compact_coordinates = np.zeros_like(coordinates)
    # Preserve each QM molecule internally, then place its anchor in the same
    # image as the first QM atom.  Both ETP/ETH QM fragments are smaller than
    # half the box, so nearest-image anchoring is unambiguous.
    qm_rows = atom_map[:qm_atom_count]
    qm_molecules: dict[int, list[int]] = {}
    for index, row in enumerate(qm_rows):
        qm_molecules.setdefault(row.molecule_id, []).append(index)
    global_anchor = qm_molecules[min(qm_molecules)][0]
    compact_coordinates[global_anchor] = coordinates[global_anchor]
    for molecule_id in sorted(qm_molecules):
        indices = qm_molecules[molecule_id]
        anchor = indices[0]
        if anchor != global_anchor:
            compact_coordinates[anchor] = compact_coordinates[global_anchor] + minimum_image(
                coordinates[anchor] - coordinates[global_anchor], cell
            )
        for index in indices:
            compact_coordinates[index] = compact_coordinates[anchor] + minimum_image(
                coordinates[index] - coordinates[anchor], cell
            )

    mm_molecules: dict[int, list[int]] = {}
    for index, row in enumerate(atom_map[qm_atom_count:], start=qm_atom_count):
        mm_molecules.setdefault(row.molecule_id, []).append(index)
    selected_indices = list(range(qm_atom_count))
    selected_molecules: list[int] = []
    minimum_distances: dict[int, float] = {}
    qm_coordinates = compact_coordinates[:qm_atom_count]
    for molecule_id in sorted(mm_molecules):
        indices = mm_molecules[molecule_id]
        distances: list[tuple[float, int, int]] = []
        for index in indices:
            displacements = minimum_image(
                coordinates[index][None, :] - coordinates[:qm_atom_count], cell
            )
            norms = np.linalg.norm(displacements, axis=1)
            qm_index = int(np.argmin(norms))
            distances.append((float(norms[qm_index]), index, qm_index))
        nearest_distance, nearest_index, nearest_qm = min(distances)
        minimum_distances[molecule_id] = nearest_distance
        if nearest_distance > cutoff_angstrom:
            continue
        selected_molecules.append(molecule_id)
        selected_indices.extend(indices)
        anchor = indices[0]
        compact_coordinates[anchor] = qm_coordinates[nearest_qm] + minimum_image(
            coordinates[anchor] - coordinates[nearest_qm], cell
        )
        for index in indices[1:]:
            compact_coordinates[index] = compact_coordinates[anchor] + minimum_image(
                coordinates[index] - coordinates[anchor], cell
            )

    selected = np.asarray(selected_indices, dtype=np.int64)
    excluded = np.asarray(
        sorted(set(range(len(atom_map))) - set(selected_indices)), dtype=np.int64
    )
    return {
        "coordinates_angstrom": np.ascontiguousarray(compact_coordinates[selected]),
        "forces_kcal_mol_angstrom": np.ascontiguousarray(forces[selected]),
        "energy_kcal_mol": correction.total_energy_kcal_mol,
        "types": np.asarray(
            [TYPE_MAP.index(atom_map[index].model_type) for index in selected],
            dtype=np.int32,
        ),
        "amber_ids": np.asarray(
            [atom_map[index].amber_id for index in selected], dtype=np.int64
        ),
        "selected_molecule_ids": selected_molecules,
        "selected_atom_count": int(selected.size),
        "excluded_atom_count": int(excluded.size),
        "excluded_force_maximum_absolute_kcal_mol_angstrom": float(
            np.max(np.abs(forces[excluded]), initial=0.0)
        ),
        "excluded_force_rms_kcal_mol_angstrom": (
            float(np.sqrt(np.mean(forces[excluded] ** 2))) if excluded.size else 0.0
        ),
        "minimum_selected_mm_distance_angstrom": (
            min(minimum_distances[molecule] for molecule in selected_molecules)
            if selected_molecules
            else None
        ),
        "maximum_selected_mm_distance_angstrom": (
            max(minimum_distances[molecule] for molecule in selected_molecules)
            if selected_molecules
            else None
        ),
    }


def write_text_new(path: Path, text: str) -> None:
    """Write one small dataset metadata file without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish the dataset ledger without replacement."""
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


def require_complete_electronic_force(correction_manifest: dict[str, Any]) -> None:
    """Reject legacy component corpora that discarded QM-dependent PPPM forces.

    The historical ``component-assembled`` status alone is insufficient: those
    labels used total minus pre-fix force. Only a publisher with an explicitly
    validated same-engine zero-QM baseline can satisfy the corrected contract.
    """
    checks = correction_manifest.get("checks", {})
    if checks.get("xTBloom_component_uses_total_minus_same_engine_zero_qm_force") is not True:
        raise ValueError("correction corpus lacks the complete zero-QM-baseline electronic force; repair legacy labels before training")


def require_whole_water_labels(correction_manifest: dict[str, Any]) -> None:
    """Do not train on the earlier force-repaired but image-corrupted corpus."""
    if correction_manifest.get("checks", {}).get("sander_tip4p_whole_water_geometry_validated") is not True:
        raise ValueError("correction corpus lacks whole-water Sander geometry validation; relabel split-water inputs before training")


def build_dataset(
    corrections: list[Path],
    correction_manifest_path: Path,
    frame_manifest_path: Path,
    atom_map_path: Path,
    output: Path,
    manifest_path: Path,
    *,
    cutoff_angstrom: float,
    validation_windows: set[str],
    test_windows: set[str],
) -> dict[str, Any]:
    """Build one-frame ragged systems with a deterministic window-block split."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset directory: {output}")
    overlap = validation_windows & test_windows
    if overlap:
        raise ValueError(f"validation and test windows overlap: {sorted(overlap)}")
    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    frames = frame_manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != len(corrections):
        raise ValueError("frame manifest and correction counts differ")
    correction_manifest = json.loads(
        correction_manifest_path.read_text(encoding="utf-8")
    )
    if correction_manifest.get("status") != "component-assembled":
        raise ValueError(
            "correction manifest is not a qualified component-assembled corpus"
        )
    require_complete_electronic_force(correction_manifest)
    require_whole_water_labels(correction_manifest)
    if int(correction_manifest.get("frame_count", -1)) != len(corrections):
        raise ValueError("correction manifest frame count differs from inputs")
    correction_records = correction_manifest.get("records")
    if not isinstance(correction_records, list) or len(correction_records) != len(
        corrections
    ):
        raise ValueError("correction manifest records differ from inputs")
    for expected_index, (path, record) in enumerate(
        zip(corrections, correction_records, strict=True), start=1
    ):
        if int(record.get("frame_index", -1)) != expected_index:
            raise ValueError("correction manifest frame indices are not contiguous")
        expected_artifact = record.get("correction")
        if not isinstance(expected_artifact, dict):
            raise ValueError("correction manifest omits a correction artifact")
        actual = artifact(path)
        if (
            expected_artifact.get("sha256") != actual["sha256"]
            or int(expected_artifact.get("bytes", -1)) != actual["bytes"]
        ):
            raise ValueError(
                f"correction input differs from component manifest at frame {expected_index}"
            )
    atom_map = read_atom_map(atom_map_path)
    output.mkdir(parents=True)
    split_systems: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    records: list[dict[str, Any]] = []
    for expected_index, (path, frame_record) in enumerate(zip(corrections, frames), start=1):
        correction = CORRECTION_IO.read_correction(path)
        if correction.frame_index != expected_index:
            raise ValueError(
                f"correction {path} has frame index {correction.frame_index}, expected {expected_index}"
            )
        window = str(frame_record["frame_name"])
        split = (
            "test"
            if window in test_windows
            else "validation"
            if window in validation_windows
            else "train"
        )
        sample = compact_sample(
            correction, atom_map, cutoff_angstrom=cutoff_angstrom
        )
        system_name = f"{expected_index:06d}-{window}"
        system = output / split / system_name
        set_directory = system / "set.000"
        set_directory.mkdir(parents=True)
        write_text_new(system / "type.raw", "\n".join(str(int(v)) for v in sample["types"]) + "\n")
        write_text_new(system / "type_map.raw", "\n".join(TYPE_MAP) + "\n")
        write_text_new(system / "nopbc", "")
        write_text_new(
            system / "compact_to_amber_id.raw",
            "\n".join(str(int(v)) for v in sample["amber_ids"]) + "\n",
        )
        np.save(
            set_directory / "coord.npy",
            sample["coordinates_angstrom"].reshape(1, -1).astype(np.float64),
        )
        np.save(
            set_directory / "force.npy",
            (sample["forces_kcal_mol_angstrom"] / KCAL_PER_MOL_PER_EV)
            .reshape(1, -1)
            .astype(np.float64),
        )
        np.save(
            set_directory / "energy.npy",
            np.asarray([[sample["energy_kcal_mol"] / KCAL_PER_MOL_PER_EV]], dtype=np.float64),
        )
        split_systems[split].append(str(system.resolve()))
        records.append(
            {
                "frame_index": expected_index,
                "window": window,
                "split": split,
                "correction": artifact(path),
                "system": str(system.resolve()),
                "selected_atom_count": sample["selected_atom_count"],
                "selected_water_molecule_count": len(sample["selected_molecule_ids"]),
                "excluded_atom_count": sample["excluded_atom_count"],
                "excluded_force_maximum_absolute_kcal_mol_angstrom": sample[
                    "excluded_force_maximum_absolute_kcal_mol_angstrom"
                ],
                "excluded_force_rms_kcal_mol_angstrom": sample[
                    "excluded_force_rms_kcal_mol_angstrom"
                ],
                "maximum_selected_mm_distance_angstrom": sample[
                    "maximum_selected_mm_distance_angstrom"
                ],
            }
        )
    unknown = (validation_windows | test_windows) - {record["window"] for record in records}
    if unknown:
        raise ValueError(f"split names windows absent from the frame manifest: {sorted(unknown)}")
    manifest = {
        "schema_version": 1,
        "status": "component-assembled-pending-model-validation",
        "target": "periodic PBE0 QM/MM minus production xTBloom GFN2-xTB QM/MM",
        "type_map": TYPE_MAP,
        "qm_atom_ids": "1:16",
        "compact_cutoff_angstrom": cutoff_angstrom,
        "selection": "QM atoms plus complete water molecules with any real atom within cutoff",
        "periodicity_in_training_payload": "discarded after triclinic minimum-image unwrapping",
        "units": {"coordinate": "angstrom", "energy": "eV", "force": "eV/angstrom"},
        "split_policy": "complete umbrella-window blocks",
        "split_counts": {name: len(paths) for name, paths in split_systems.items()},
        "systems": split_systems,
        "frame_manifest": artifact(frame_manifest_path),
        "correction_manifest": artifact(correction_manifest_path),
        "atom_map": artifact(atom_map_path),
        "records": records,
        "limitations": [
            "sampling qualification is inherited from the immutable frame manifest and is not re-established by dataset compaction",
            "component assembly removes cross-engine total-force subtraction but does not itself qualify a model",
            "model, trajectory, and free-energy closure remain required before release qualification",
        ],
    }
    write_json_new(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, nargs="+", required=True)
    parser.add_argument("--correction-manifest", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cutoff-angstrom", type=float, default=6.0)
    parser.add_argument("--validation-window", action="append", default=[])
    parser.add_argument("--test-window", action="append", default=[])
    arguments = parser.parse_args()
    manifest = build_dataset(
        arguments.correction,
        arguments.correction_manifest,
        arguments.frame_manifest,
        arguments.atom_map,
        arguments.output,
        arguments.manifest,
        cutoff_angstrom=arguments.cutoff_angstrom,
        validation_windows=set(arguments.validation_window),
        test_windows=set(arguments.test_window),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
