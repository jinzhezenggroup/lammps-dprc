#!/usr/bin/env python3
"""Audit an external DPRc HDF5 payload against the ETP/ETH label contract.

The legacy tutorial archive is useful for architecture experiments, but its
MNDOD-to-PBE0 labels are not the production xTB-to-PBE0 correction target.
This tool separates byte/data integrity from scientific qualification and
publishes both conclusions in a deterministic JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "workloads/etpeth/dprc-labels.json"
FORMULA = re.compile(
    r"C(?P<C>\d+)H(?P<H>\d+)HW(?P<HW>\d+)O(?P<O>\d+)OW(?P<OW>\d+)P(?P<P>\d+)"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest without loading a large dataset at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_formula(name: str) -> dict[str, int]:
    """Parse the canonical dpdata MultiSystems formula used by the archive."""
    match = FORMULA.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected DPRc system formula: {name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def _require_mapping(parent: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    """Return one required object while preserving a precise error path."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}.{key} must be an object")
    return value


def _require_exact(
    parent: dict[str, Any], key: str, expected: Any, path: str
) -> None:
    """Require an immutable contract value instead of accepting arbitrary JSON."""
    if parent.get(key) != expected:
        raise ValueError(f"{path}.{key} must equal {expected!r}")


def _require_string_list(
    parent: dict[str, Any],
    key: str,
    expected: tuple[str, ...],
    path: str,
    *,
    unique: bool = False,
) -> list[str]:
    """Validate an ordered string list and optionally its type-space uniqueness."""
    value = parent.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{path}.{key} must be a string array")
    if tuple(value) != expected:
        raise ValueError(f"{path}.{key} must equal {list(expected)!r}")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"{path}.{key} must not contain duplicate model types")
    return value


def git_identity(repository: Path) -> dict[str, Any]:
    """Record the analysis source identity without requiring a Git checkout."""
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def require_contract(payload: dict[str, Any]) -> None:
    """Validate every immutable field that determines label qualification."""
    if payload.get("schema_version") != 2:
        raise ValueError("DPRc label contract schema_version must be 2")
    _require_exact(payload, "id", "etpeth-dprc-label-contract", "contract")
    legacy = payload.get("legacy_dataset")
    production = payload.get("production_target")
    if not isinstance(legacy, dict) or not isinstance(production, dict):
        raise TypeError("DPRc label contract is missing legacy or production sections")
    source = _require_mapping(legacy, "source", "legacy_dataset")
    for key, expected in (
        ("name", "dprc-tutorial"),
        ("upstream", "https://github.com/njzjz/dprc-tutorial"),
        ("revision", "f8716f28b03ef09734b74ae7f2ca67ab45c3d40f"),
        ("license", "NOASSERTION"),
        ("archive_path", "init_data/init_data.hdf5.tar.bz2"),
        (
            "archive_sha256",
            "ae3bd54434e72a4158a551d585eb8323b9c724f03ddb3ae5d57aa35c92553b5b",
        ),
        (
            "dataset_sha256",
            "a64bd5b20030720d496ddecab31a0e2b0c5b0ff300012ceb46bcd90587c72a04",
        ),
        (
            "documented_command",
            (
                "dpamber corr --cutoff 6. --qm_region :1-2 --parm7_file "
                "ETP_ETH.parm7 --nc mndod.nc --hl pbe0 --ll mndod --out "
                "init_data.hdf5"
            ),
        ),
        ("producer_version", None),
        (
            "producer_version_note",
            (
                "The tutorial does not pin the dpamber, dpdata, AmberTools, or "
                "electronic-structure revisions that produced the archive."
            ),
        ),
    ):
        _require_exact(source, key, expected, "legacy_dataset.source")
    label = _require_mapping(legacy, "label", "legacy_dataset")
    expected_legacy_label = {
        "meaning": "high-level QM/MM minus low-level QM/MM",
        "high_level": "PBE0",
        "low_level": "MNDOD",
        "coordinate_unit": "angstrom",
        "energy_unit": "eV",
        "force_unit": "eV/angstrom",
        "cutoff_angstrom": 6.0,
        "qm_region": ":1-2",
        "selection": "per-atom Amber distance mask",
        "periodicity_in_training_payload": "discarded after wrapping",
    }
    if label != expected_legacy_label:
        raise ValueError("legacy_dataset.label semantics are incomplete or changed")
    expected = _require_mapping(legacy, "expected", "legacy_dataset")
    _require_string_list(
        expected,
        "type_map",
        ("C", "H", "HW", "O", "OW", "P"),
        "legacy_dataset.expected",
        unique=True,
    )
    qm_composition = _require_mapping(
        expected, "qm_composition", "legacy_dataset.expected"
    )
    if qm_composition != {"C": 3, "H": 7, "O": 5, "P": 1}:
        raise ValueError("legacy_dataset.expected.qm_composition is incorrect")
    for key, value in (
        ("group_count", 389),
        ("frame_count", 3600),
        ("minimum_atoms", 226),
        ("maximum_atoms", 285),
    ):
        _require_exact(expected, key, value, "legacy_dataset.expected")
    _require_exact(
        production,
        "contract_state",
        "high-level-periodic-labeler-qualified-low-level-pending",
        "production_target",
    )
    _require_exact(
        production,
        "label_meaning",
        "high-level QM/MM minus xTBloom GFN2-xTB QM/MM",
        "production_target",
    )

    high_level = _require_mapping(production, "high_level", "production_target")
    _require_exact(high_level, "method", "PBE0", "production_target.high_level")
    _require_exact(
        high_level,
        "engine",
        "QUICK 25.03 through AmberTools 26 update.1 Sander EXTERN",
        "production_target.high_level",
    )
    _require_exact(
        high_level,
        "basis",
        "6-31G* (QUICK 6-31GD.BAS)",
        "production_target.high_level",
    )
    expected_references = {
        "engine_manifest": {
            "path": "config/quick_pbe0_engine.json",
            "sha256": "1459f84d99f683af632cad70a412d9d34e2047ab518f1a1855ebf347f902e0d0",
        },
        "qualification_evidence": {
            "path": "workloads/etpeth/evidence/quick-pbe0-qualification.json",
            "sha256": "87f129cb44221af3e6d706810999c25d6b68fe025393e7914582dd3342841ac3",
        },
    }
    for key, expected_reference in expected_references.items():
        reference = _require_mapping(high_level, key, "production_target.high_level")
        if reference != expected_reference:
            raise ValueError(f"production_target.high_level.{key} is not pinned")
        referenced_path = ROOT / reference["path"]
        if not referenced_path.is_file() or sha256(referenced_path) != reference["sha256"]:
            raise ValueError(f"production_target.high_level.{key} bytes do not match")
    _require_exact(
        high_level,
        "qualification_scope",
        "engine-qualified-periodic-labeler-pending",
        "production_target.high_level",
    )
    _require_exact(
        high_level,
        "qualification_scope_note",
        "engine-only nonperiodic fixture state; periodic qualification is recorded separately",
        "production_target.high_level",
    )
    _require_exact(
        high_level,
        "periodic_embedding_status",
        "target-topology binary64 periodic label publication qualified; independent PBE0 oracle unavailable",
        "production_target.high_level",
    )
    evidence = json.loads(
        (ROOT / expected_references["qualification_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if evidence.get("qualified") is not True or evidence.get(
        "qualification_scope"
    ) != high_level["qualification_scope"]:
        raise ValueError("QUICK PBE0 qualification evidence is incomplete")
    periodic_reference = _require_mapping(
        high_level,
        "periodic_label_qualification",
        "production_target.high_level",
    )
    expected_periodic_reference = {
        "path": "workloads/etpeth/evidence/quick-pbe0-binary64-label-qualification.json",
        "sha256": "68971a0d2a1386cc871bce6ed6410dd0513f5b2b2a2c1cff3121fcba6f79e4ef",
    }
    if periodic_reference != expected_periodic_reference:
        raise ValueError(
            "production_target.high_level.periodic_label_qualification is not pinned"
        )
    periodic_path = ROOT / periodic_reference["path"]
    if not periodic_path.is_file() or sha256(periodic_path) != periodic_reference[
        "sha256"
    ]:
        raise ValueError("periodic label qualification evidence bytes do not match")
    periodic_evidence = json.loads(periodic_path.read_text(encoding="utf-8"))
    if periodic_evidence.get("qualified") is not True or periodic_evidence.get(
        "production_ready"
    ) is not False:
        raise ValueError("periodic label qualification scope is overstated")
    _require_string_list(
        high_level,
        "required_provenance",
        (
            "engine_name",
            "engine_revision",
            "executable_sha256",
            "basis_definition",
            "basis_sha256",
            "functional_definition",
            "scf_convergence",
            "qmmm_embedding_definition",
            "failed_frame_ledger",
        ),
        "production_target.high_level",
        unique=True,
    )

    low_level = _require_mapping(production, "low_level", "production_target")
    for key, expected in (
        ("engine", "xTBloom"),
        ("model", "GFN2-xTB"),
        ("energy_quantity", "electronic Helmholtz free energy"),
        ("electronic_temperature_kelvin", 300.0),
        ("public_abi_precision", "IEEE binary64"),
    ):
        _require_exact(low_level, key, expected, "production_target.low_level")
    _require_string_list(
        low_level,
        "required_provenance",
        (
            "xtbloom_revision",
            "xtbloom_library_sha256",
            "parameter_manifest_sha256",
            "backend",
            "compute_options",
            "per_frame_scc_status",
            "per_frame_scc_iterations",
            "result_flags",
        ),
        "production_target.low_level",
        unique=True,
    )

    units = _require_mapping(production, "training_units", "production_target")
    if units != {
        "coordinates": "angstrom",
        "cell": "angstrom",
        "energy": "eV",
        "forces": "eV/angstrom",
    }:
        raise ValueError("production_target.training_units are incomplete or changed")
    native_units = _require_mapping(
        production, "xtbloom_native_units", "production_target"
    )
    if native_units != {
        "positions": "bohr",
        "energy": "hartree",
        "forces": "hartree/bohr",
        "periodic_atomic_potential_shifts": "hartree",
        "periodic_charge_response_matrix": "hartree",
        "point_charge_gamma": "hartree (atomic-unit hardness)",
    }:
        raise ValueError(
            "production_target.xtbloom_native_units are incomplete or changed"
        )

    model_types = _require_string_list(
        production,
        "model_type_map",
        ("P", "O", "C", "H", "OW", "HW"),
        "production_target",
        unique=True,
    )
    lammps_types = _require_mapping(
        production, "lammps_type_to_model_species", "production_target"
    )
    _require_exact(
        lammps_types,
        "index_base",
        1,
        "production_target.lammps_type_to_model_species",
    )
    species = _require_string_list(
        lammps_types,
        "species",
        ("P", "O", "O", "C", "H", "OW", "HW"),
        "production_target.lammps_type_to_model_species",
    )
    if any(item not in model_types for item in species):
        raise ValueError("LAMMPS type mapping refers to an unknown model type")

    qm_region = _require_mapping(production, "qm_region", "production_target")
    for key, expected in (
        ("id_space", "LAMMPS atom ID"),
        ("selection", "inclusive_range"),
        ("first", 1),
        ("last", 16),
        ("count", 16),
        ("charge", -2),
        ("uhf", 0),
    ):
        _require_exact(qm_region, key, expected, "production_target.qm_region")
    composition = _require_mapping(
        qm_region, "composition", "production_target.qm_region"
    )
    if composition != {"P": 1, "O": 5, "C": 3, "H": 7}:
        raise ValueError("production_target.qm_region.composition is incorrect")
    if qm_region["last"] - qm_region["first"] + 1 != qm_region["count"]:
        raise ValueError("production_target.qm_region range/count disagree")
    if sum(composition.values()) != qm_region["count"]:
        raise ValueError("production_target.qm_region composition/count disagree")

    embedding = _require_mapping(production, "embedding", "production_target")
    for key, expected in (
        ("environment_cutoff_angstrom", 6.0),
        ("include_molecule", True),
        ("point_charge_forces_required", True),
    ):
        _require_exact(embedding, key, expected, "production_target.embedding")
    response = _require_mapping(
        embedding, "periodic_response", "production_target.embedding"
    )
    for key, expected in (
        ("potential", "b + A*q"),
        ("energy", "q^T*b + 0.5*q^T*A*q"),
        ("xtbloom_force_semantics", "b and A held fixed"),
        (
            "required_result_flag",
            "XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES",
        ),
        ("operator_derivative_force_required", True),
    ):
        _require_exact(
            response,
            key,
            expected,
            "production_target.embedding.periodic_response",
        )
    tip4p = _require_mapping(embedding, "tip4p", "production_target.embedding")
    expected_tip4p = {
        "model": "TIP4P-Ew",
        "oxygen_lammps_type": 6,
        "hydrogen_lammps_type": 7,
        "oh_bond_lammps_type": 1,
        "hh_bond_lammps_type": 2,
        "angle_lammps_type": 1,
        "qdist_angstrom": 0.125,
        "virtual_site_definition_required": True,
        "m_site_force_redistribution_required": True,
    }
    if tip4p != expected_tip4p:
        raise ValueError("production_target.embedding.tip4p is incomplete or changed")

    _require_string_list(
        production,
        "required_source_fields",
        (
            "full_system_coordinates",
            "triclinic_cell",
            "stable_atom_ids",
            "molecule_ids",
            "element_or_lammps_type",
            "mm_point_charges",
            "mm_point_charge_gammas",
            "mm_gamma_policy_manifest_sha256",
            "periodic_atomic_potential_shifts",
            "periodic_charge_response_matrix",
            "periodic_operator_derivative_forces",
            "tip4p_virtual_site_definition",
            "tip4p_m_site_force_redistribution",
            "high_level_energy",
            "high_level_full_qmmm_forces",
            "xtb_helmholtz_free_energy",
            "xtb_raw_qm_forces",
            "xtb_raw_point_charge_forces",
            "xtb_qm_charges",
            "xtb_result_flags",
            "xtb_final_full_qmmm_forces",
            "correction_energy",
            "correction_full_system_forces",
            "selection_membership",
            "compact_to_full_atom_map",
        ),
        "production_target",
        unique=True,
    )
    _require_exact(
        production,
        "precision",
        "binary64 labels; FP32 model inference requires independent qualification",
        "production_target",
    )


def _all_finite(dataset: Any) -> bool:
    """Check a numeric HDF5 dataset in bounded frame-sized slices."""
    import numpy as np

    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]))
    chunk = max(1, min(64, dataset.shape[0]))
    for start in range(0, dataset.shape[0], chunk):
        if not bool(np.isfinite(dataset[start : start + chunk]).all()):
            return False
    return True


def inspect_hdf5(dataset_path: Path) -> dict[str, Any]:
    """Inspect integrity, composition, fields, units-compatible dtypes, and ranges."""
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("HDF5 audit requires h5py and numpy") from error

    group_count = 0
    frame_count = 0
    frames_per_group: list[int] = []
    atom_counts: list[int] = []
    type_maps: set[tuple[str, ...]] = set()
    energy_min = math.inf
    energy_max = -math.inf
    maximum_absolute_force = 0.0
    water_delta_groups: Counter[int] = Counter()
    water_delta_frames: Counter[int] = Counter()
    field_names: set[str] = set()
    coordinate_float32_exact = True
    energy_float32_exact = True
    force_float32_exact = True

    with h5py.File(dataset_path, "r") as handle:
        for name in sorted(handle.keys()):
            group = handle[name]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"{name} is not an HDF5 group")
            formula = parse_formula(name)
            group_count += 1
            nopbc = group.get("nopbc")
            if not isinstance(nopbc, h5py.Dataset) or not bool(nopbc[()]):
                raise ValueError(f"{name} is not explicitly nonperiodic")
            field_names.add("nopbc")

            type_raw = group.get("type.raw")
            type_map_raw = group.get("type_map.raw")
            if not isinstance(type_raw, h5py.Dataset) or not isinstance(
                type_map_raw, h5py.Dataset
            ):
                raise TypeError(f"{name} is missing type.raw or type_map.raw")
            field_names.update(("type.raw", "type_map.raw"))
            type_map = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in type_map_raw[()]
            )
            type_maps.add(type_map)
            types = np.asarray(type_raw[()])
            if types.ndim != 1 or types.dtype.kind not in "iu":
                raise ValueError(f"{name}/type.raw has an invalid shape or dtype")
            if (
                len(types) == 0
                or int(types.min()) < 0
                or int(types.max()) >= len(type_map)
            ):
                raise ValueError(f"{name}/type.raw contains an invalid type index")
            observed = Counter(type_map[int(index)] for index in types)
            if observed != Counter(formula):
                raise ValueError(
                    f"{name} formula does not match type.raw: {dict(observed)}"
                )
            natoms = len(types)
            atom_counts.append(natoms)

            group_keys = cast(Iterable[str], group)
            set_names = sorted(key for key in group_keys if key.startswith("set."))
            if not set_names:
                raise ValueError(f"{name} has no set.* frame group")
            group_frames = 0
            for set_name in set_names:
                frame_group = group[set_name]
                if not isinstance(frame_group, h5py.Group):
                    raise TypeError(f"{name}/{set_name} is not an HDF5 group")
                required = ("coord.npy", "energy.npy", "force.npy")
                if any(key not in frame_group for key in required):
                    raise ValueError(
                        f"{name}/{set_name} is missing a required label array"
                    )
                field_names.update(f"{set_name}/{key}" for key in frame_group)
                coord = frame_group.get("coord.npy")
                energy = frame_group.get("energy.npy")
                force = frame_group.get("force.npy")
                if not isinstance(coord, h5py.Dataset):
                    raise TypeError(f"{name}/{set_name}/coord.npy is not a dataset")
                if not isinstance(energy, h5py.Dataset):
                    raise TypeError(f"{name}/{set_name}/energy.npy is not a dataset")
                if not isinstance(force, h5py.Dataset):
                    raise TypeError(f"{name}/{set_name}/force.npy is not a dataset")
                nframes = int(energy.shape[0]) if energy.ndim == 1 else -1
                if (
                    nframes < 1
                    or coord.shape != (nframes, 3 * natoms)
                    or force.shape != coord.shape
                    or coord.dtype != np.dtype("float64")
                    or energy.dtype != np.dtype("float64")
                    or force.dtype != np.dtype("float64")
                ):
                    raise ValueError(
                        f"{name}/{set_name} has an invalid label shape or dtype"
                    )
                if not (
                    _all_finite(coord) and _all_finite(energy) and _all_finite(force)
                ):
                    raise ValueError(f"{name}/{set_name} contains non-finite values")
                energies = np.asarray(energy[()])
                coordinates = np.asarray(coord[()])
                forces = np.asarray(force[()])
                coordinate_float32_exact = coordinate_float32_exact and bool(
                    np.array_equal(
                        coordinates, coordinates.astype(np.float32).astype(np.float64)
                    )
                )
                energy_float32_exact = energy_float32_exact and bool(
                    np.array_equal(
                        energies, energies.astype(np.float32).astype(np.float64)
                    )
                )
                force_float32_exact = force_float32_exact and bool(
                    np.array_equal(forces, forces.astype(np.float32).astype(np.float64))
                )
                energy_min = min(energy_min, float(energies.min()))
                energy_max = max(energy_max, float(energies.max()))
                maximum_absolute_force = max(
                    maximum_absolute_force, float(np.abs(forces).max())
                )
                frame_count += nframes
                group_frames += nframes

            water_delta = formula["HW"] - 2 * formula["OW"]
            water_delta_groups[water_delta] += 1
            water_delta_frames[water_delta] += group_frames
            frames_per_group.append(group_frames)

    return {
        "h5py_version": h5py.__version__,
        "numpy_version": np.__version__,
        "group_count": group_count,
        "frame_count": frame_count,
        "frames_per_group": {
            "minimum": min(frames_per_group),
            "median": statistics.median(frames_per_group),
            "maximum": max(frames_per_group),
            "singleton_groups": sum(value == 1 for value in frames_per_group),
        },
        "minimum_atoms": min(atom_counts),
        "maximum_atoms": max(atom_counts),
        "distinct_atom_counts": len(set(atom_counts)),
        "type_maps": [list(value) for value in sorted(type_maps)],
        "energy_range_eV": [energy_min, energy_max],
        "maximum_absolute_force_eV_per_angstrom": maximum_absolute_force,
        "exact_binary32_roundtrip": {
            "coordinates": coordinate_float32_exact,
            "energies": energy_float32_exact,
            "forces": force_float32_exact,
            "note": (
                "Source-value roundtrip does not qualify FP32 model inference or "
                "free-energy accuracy."
            ),
        },
        "water_completion": {
            "criterion": "HW count equals two times OW count",
            "criterion_scope": (
                "necessary stoichiometric balance only; atom identities and molecule "
                "membership are absent, so it cannot prove complete waters"
            ),
            "stoichiometrically_balanced_groups": water_delta_groups[0],
            "stoichiometrically_balanced_frames": water_delta_frames[0],
            "stoichiometrically_unbalanced_groups": (
                group_count - water_delta_groups[0]
            ),
            "stoichiometrically_unbalanced_frames": (
                frame_count - water_delta_frames[0]
            ),
            "hw_minus_2ow_group_counts": {
                str(key): value for key, value in sorted(water_delta_groups.items())
            },
        },
        "field_names": sorted(field_names),
    }


def qualification_reasons(
    contract: dict[str, Any], inspection: dict[str, Any]
) -> list[str]:
    """Return every independent reason the legacy payload is not production data."""
    legacy = contract["legacy_dataset"]
    source = legacy["source"]
    label = legacy["label"]
    production = contract["production_target"]
    reasons: list[str] = []
    if production["contract_state"] != "ready-for-production-label-generation":
        reasons.append(
            "production correction-label pipeline remains incomplete: the matching "
            "xTBloom periodic operator and correction corpus are pending"
        )
    if source["license"] == "NOASSERTION":
        reasons.append("source redistribution license is NOASSERTION")
    if source.get("producer_version") is None:
        reasons.append("the producing dpamber/dpdata/AmberTools revisions are unknown")
    if label["low_level"] != production["low_level"]["model"]:
        reasons.append(
            "low-level Hamiltonian is "
            f"{label['low_level']}, not {production['low_level']['model']}"
        )
    if label["selection"] != "complete-molecule compact environment":
        reasons.append("legacy compact selection is per atom, not include_molecule")
    if inspection["water_completion"]["stoichiometrically_unbalanced_frames"]:
        reasons.append("legacy compact frames contain incomplete water molecules")
    absent = (
        "triclinic cell, stable atom IDs, molecule IDs, MM charges/gamma, and periodic "
        "b+Aq operator provenance are absent from the training payload"
    )
    reasons.append(absent)
    return reasons


def verify_expected(contract: dict[str, Any], inspection: dict[str, Any]) -> None:
    """Fail when the immutable legacy archive no longer matches its recorded schema."""
    expected = contract["legacy_dataset"]["expected"]
    for key in ("group_count", "frame_count", "minimum_atoms", "maximum_atoms"):
        if inspection[key] != expected[key]:
            raise ValueError(
                f"legacy dataset {key} is {inspection[key]}, expected {expected[key]}"
            )
    if inspection["type_maps"] != [expected["type_map"]]:
        raise ValueError(
            f"legacy type map is {inspection['type_maps']}, expected {expected['type_map']}"
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish the complete audit without exposing a partial report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unqualified-source", action="store_true")
    parser.add_argument("--require-production-qualified", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        contract = json.loads(arguments.contract.read_text(encoding="utf-8"))
        require_contract(contract)
        expected_hash = contract["legacy_dataset"]["source"]["dataset_sha256"]
        actual_hash = sha256(arguments.dataset)
        if actual_hash != expected_hash:
            raise ValueError(
                f"dataset SHA-256 {actual_hash} differs from expected {expected_hash}"
            )
        if (
            contract["legacy_dataset"]["source"]["license"] == "NOASSERTION"
            and not arguments.allow_unqualified_source
        ):
            raise ValueError(
                "legacy source is license-unqualified; pass --allow-unqualified-source "
                "only for private diagnostic audit"
            )
        inspection = inspect_hdf5(arguments.dataset)
        verify_expected(contract, inspection)
        reasons = qualification_reasons(contract, inspection)
        report = {
            "schema_version": 1,
            "status": "passed",
            "qualification": "private-diagnostic",
            "production_qualified": not reasons,
            "production_disqualification_reasons": reasons,
            "dataset": {
                "path": str(arguments.dataset.resolve()),
                "sha256": actual_hash,
            },
            "contract": {
                "path": str(arguments.contract.resolve()),
                "sha256": sha256(arguments.contract),
            },
            "analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
                "repository": git_identity(ROOT),
            },
            "inspection": inspection,
            "production_target": contract["production_target"],
        }
        write_json_atomic(arguments.output, report)
        print(
            f"audited {inspection['frame_count']} frames in {inspection['group_count']} "
            f"groups; production_qualified={report['production_qualified']}"
        )
        if arguments.require_production_qualified and reasons:
            return 2
        return 0
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        print(f"DPRc label audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
