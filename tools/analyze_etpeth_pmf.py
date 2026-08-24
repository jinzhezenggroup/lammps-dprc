#!/usr/bin/env python3
"""Validate ETP/ETH production ledgers and reconstruct a 1-D umbrella PMF.

The implementation intentionally uses only the Python standard library.  It
consumes the hash-qualified Colvars series recorded by the workload runner,
solves histogram WHAM in binary64, estimates time-correlation ESS with Geyer's
initial monotone positive sequence, and obtains uncertainty by resampling
correlation-length blocks within each independent production trial.
"""

from __future__ import annotations

import argparse
import bisect
import cmath
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "workloads/etpeth/manifest.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "workloads/etpeth/analysis.json"
K_B_KCAL_PER_MOL_K = 0.00198720425864083


@dataclass(frozen=True)
class Window:
    """One ordered umbrella center from the workload manifest."""

    tag: str
    center: float


@dataclass(frozen=True)
class Series:
    """One trial/window reaction-coordinate time series."""

    steps: tuple[int, ...]
    reaction_coordinate: tuple[float, ...]
    attack_angle: tuple[float, ...]


@dataclass(frozen=True)
class WhamResult:
    """Converged binned probability and relative PMF."""

    probabilities: tuple[float, ...]
    pmf: tuple[float | None, ...]
    dimensionless_offsets: tuple[float, ...]
    iterations: int
    residual: float


@dataclass(frozen=True)
class HistogramBlocks:
    """Circular block histograms plus an exact-size resampling protocol."""

    full: tuple[tuple[int, ...], ...]
    prefixes: tuple[tuple[int, ...], ...]
    full_draws: int
    remainder: int
    sample_count: int


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one recorded artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_matches(identity: object, path: Path) -> bool:
    """Require both the absolute path and bytes recorded by a ledger."""
    return bool(
        isinstance(identity, dict)
        and path.is_file()
        and Path(str(identity.get("path", ""))).resolve() == path.resolve()
        and identity.get("sha256") == sha256(path)
    )


def identity(path: Path) -> dict[str, str]:
    """Describe an immutable analysis input or output."""
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def git_output(repository: Path, *arguments: str) -> str:
    """Return one Git value or fail closed when provenance is unavailable."""
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(
            f"could not record analysis Git provenance ({' '.join(arguments)}): "
            f"{diagnostic}"
        )
    return completed.stdout.strip()


def analysis_project_record() -> dict[str, Any]:
    """Record the exact analyzer bytes and repository state used for a PMF."""
    analyzer = Path(__file__).resolve()
    dirty_output = git_output(PROJECT_ROOT, "status", "--porcelain=v1")
    return {
        "path": str(PROJECT_ROOT),
        "revision": git_output(PROJECT_ROOT, "rev-parse", "HEAD"),
        "dirty": bool(dirty_output),
        "dirty_entries": dirty_output.splitlines() if dirty_output else [],
        "analyzer": identity(analyzer),
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with a contextual error on malformed input."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def stable_tag(center_tenths: int) -> str:
    """Use the same collision-free center tag as the workload runner."""
    sign = "m" if center_tenths < 0 else "p"
    magnitude = abs(center_tenths)
    return f"{sign}{magnitude // 10}p{magnitude % 10}"


def windows_from_manifest(manifest: dict[str, Any]) -> list[Window]:
    """Reconstruct and validate the exact ordered umbrella grid."""
    umbrella = manifest["umbrella"]
    start = int(umbrella["start_tenths_angstrom"])
    stop = int(umbrella["stop_tenths_angstrom"])
    step = int(umbrella["step_tenths_angstrom"])
    if step <= 0 or stop < start or (stop - start) % step:
        raise ValueError("manifest has an invalid umbrella grid")
    centers = list(range(start, stop + 1, step))
    if len(centers) != int(umbrella["count"]):
        raise ValueError("manifest umbrella count differs from its grid")
    return [Window(stable_tag(center), center / 10.0) for center in centers]


def checkpoint_boundary_tolerances(
    manifest: dict[str, Any],
) -> tuple[float, float]:
    """Return the predeclared write-data/Colvars round-trip tolerances."""
    values = manifest["dynamics"]["colvars_checkpoint_boundary_tolerance"]
    reaction = float(values["reaction_coordinate_angstrom"])
    angle = float(values["attack_angle_degree"])
    if (
        not math.isfinite(reaction)
        or not math.isfinite(angle)
        or reaction < 0.0
        or angle < 0.0
    ):
        raise ValueError("Colvars checkpoint-boundary tolerances are invalid")
    return reaction, angle


def series_merge_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    """Describe the exact absolute-timestep boundary de-duplication rule."""
    reaction, angle = checkpoint_boundary_tolerances(manifest)
    return {
        "key": "absolute_timestep",
        "drop_duplicate_boundary_sample": True,
        "reaction_coordinate_absolute_tolerance_angstrom": reaction,
        "attack_angle_absolute_tolerance_degree": angle,
    }


def validate_analysis_contract(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> None:
    """Reject contracts that name semantics this implementation does not perform."""
    estimator = contract.get("estimator", {})
    sampling = contract.get("sampling", {})
    uncertainty = contract.get("uncertainty", {})
    overlap = contract.get("overlap", {})
    trial_consistency = contract.get("trial_consistency", {})
    expected = {
        "estimator.name": (estimator.get("name"), "histogram-wham"),
        "estimator.reaction_coordinate_unit": (
            estimator.get("reaction_coordinate_unit"),
            "angstrom",
        ),
        "estimator.energy_unit": (estimator.get("energy_unit"), "kcal/mol"),
        "estimator.pmf_zero": (
            estimator.get("pmf_zero"),
            "global-observed-minimum",
        ),
        "estimator.trial_combination": (
            estimator.get("trial_combination"),
            "pooled-counts-with-trial-separated-correlation-blocks",
        ),
        "sampling.autocorrelation": (
            sampling.get("autocorrelation"),
            "geyer-initial-monotone-positive-sequence",
        ),
        "sampling.effective_sample_definition": (
            sampling.get("effective_sample_definition"),
            "sum-over-trials(N/max(g_reaction,g_angle))",
        ),
        "sampling.chunk_boundary_policy": (
            sampling.get("chunk_boundary_policy"),
            "deduplicate-roundtrip-equivalent-absolute-timestep",
        ),
        "uncertainty.method": (
            uncertainty.get("method"),
            "nonoverlapping-circular-block-bootstrap",
        ),
        "uncertainty.block_length_samples": (
            uncertainty.get("block_length_samples"),
            "ceil(maximum-statistical-inefficiency-per-trial-window)",
        ),
        "overlap.coefficient": (
            overlap.get("coefficient"),
            "bhattacharyya-coefficient-of-observed-window-histograms",
        ),
        "overlap.use_common_bins": (overlap.get("use_common_bins"), True),
        "trial_consistency.alignment": (
            trial_consistency.get("alignment"),
            "minimum-pooled-pmf-bin-with-common-trial-support",
        ),
    }
    for label, (actual, supported) in expected.items():
        if actual != supported:
            raise ValueError(
                f"analysis contract requests unsupported {label}: {actual!r}; "
                f"supported value is {supported!r}"
            )
    temperature = float(estimator.get("temperature_kelvin", math.nan))
    manifest_temperature = float(manifest["dynamics"]["temperature_kelvin"])
    if not math.isclose(
        temperature, manifest_temperature, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("analysis temperature differs from the production thermostat")
    if int(sampling.get("burn_in_samples_per_window", -1)) < 0:
        raise ValueError("analysis burn-in must be non-negative")
    if int(estimator.get("maximum_iterations", 0)) <= 0:
        raise ValueError("WHAM maximum_iterations must be positive")
    tolerance = float(estimator.get("dimensionless_tolerance", math.nan))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("WHAM tolerance must be finite and positive")
    replicates = int(uncertainty.get("replicates", 0))
    confidence = float(uncertainty.get("confidence_level", math.nan))
    finite_fraction = float(
        uncertainty.get("minimum_finite_fraction_per_observed_bin", math.nan)
    )
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap replicates/confidence are invalid")
    if not 0.0 <= finite_fraction <= 1.0:
        raise ValueError("bootstrap finite-coverage threshold is invalid")
    for name in (
        "maximum_pmf_region_kcal_mol",
        "maximum_pairwise_absolute_difference_kcal_mol",
    ):
        value = float(trial_consistency.get(name, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"trial consistency threshold is invalid: {name}")


def parse_colvars(path: Path) -> Series:
    """Parse the two declared Colvars columns and reject timeline ambiguity."""
    columns: list[str] | None = None
    rows: list[tuple[int, float, float]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            candidate = stripped[1:].split()
            if {
                "step",
                "reaction_coordinate",
                "attack_angle",
            }.issubset(candidate):
                columns = candidate
            continue
        if columns is None:
            raise ValueError(
                f"Colvars data precedes its header in {path}:{line_number}"
            )
        fields = stripped.split()
        if len(fields) != len(columns):
            raise ValueError(f"Colvars column count changed in {path}:{line_number}")
        values = dict(zip(columns, fields, strict=True))
        try:
            step = int(values["step"])
            reaction = float(values["reaction_coordinate"])
            angle = float(values["attack_angle"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid Colvars row in {path}:{line_number}") from error
        if step < 0 or not math.isfinite(reaction) or not math.isfinite(angle):
            raise ValueError(
                f"non-finite or negative Colvars row in {path}:{line_number}"
            )
        if rows and step <= rows[-1][0]:
            raise ValueError(f"Colvars timesteps are not strictly increasing in {path}")
        rows.append((step, reaction, angle))
    if not rows:
        raise ValueError(f"Colvars file has no samples: {path}")
    return Series(
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
    )


def merge_series(
    parts: Sequence[Series],
    reaction_tolerance: float = 0.0,
    angle_tolerance: float = 0.0,
) -> Series:
    """Merge chunks, admitting only predeclared decimal round-trip noise."""
    if not parts:
        raise ValueError("cannot merge an empty Colvars ledger")
    merged: list[tuple[int, float, float]] = []
    for part in parts:
        for row in zip(
            part.steps,
            part.reaction_coordinate,
            part.attack_angle,
            strict=True,
        ):
            if merged and row[0] == merged[-1][0]:
                previous = merged[-1]
                if not (
                    math.isclose(
                        row[1], previous[1], rel_tol=0.0, abs_tol=reaction_tolerance
                    )
                    and math.isclose(
                        row[2], previous[2], rel_tol=0.0, abs_tol=angle_tolerance
                    )
                ):
                    raise ValueError(
                        f"chunk boundary values disagree at timestep {row[0]}"
                    )
                continue
            if merged and row[0] < merged[-1][0]:
                raise ValueError("Colvars chunk ledger is not chronologically ordered")
            merged.append(row)
    return Series(
        tuple(row[0] for row in merged),
        tuple(row[1] for row in merged),
        tuple(row[2] for row in merged),
    )


def expected_colvars_steps(offset: int, steps: int, frequency: int) -> tuple[int, ...]:
    """Return outputs selected by Colvars' absolute-timestep cadence.

    A resumed chunk whose offset is not divisible by ``frequency`` does not
    emit an artificial sample at its first step.  The first eligible output is
    the next absolute multiple, matching Colvars' runtime scheduling.
    """
    if offset < 0 or steps <= 0 or frequency <= 0:
        raise ValueError("Colvars timeline parameters are invalid")
    first = ((offset + frequency - 1) // frequency) * frequency
    stop = offset + steps
    return tuple(range(first, stop + 1, frequency))


def require_recorded_artifact(identity_value: object, label: str) -> Path:
    """Resolve one ledger artifact only after its path and digest still match."""
    if not isinstance(identity_value, dict) or "path" not in identity_value:
        raise ValueError(f"recorded {label} identity is malformed")
    path = Path(str(identity_value["path"]))
    if not artifact_matches(identity_value, path):
        raise ValueError(f"recorded {label} bytes changed: {path}")
    return path


def validate_invocation_provenance(
    record: dict[str, Any],
    manifest_path: Path,
    expected_order: Sequence[str],
) -> str:
    """Validate immutable runtime/project/source inputs for one chunk record."""
    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("production runtime identity is missing")
    for name in ("lammps", "plugin", "xtbloom", "mpiexec"):
        require_recorded_artifact(runtime.get(name), f"runtime {name}")
    if "plugin_cmake_cache" in runtime:
        require_recorded_artifact(runtime["plugin_cmake_cache"], "plugin CMake cache")
    loaded = record.get("loaded_xtbloom")
    xtbloom = runtime["xtbloom"]
    if (
        not isinstance(loaded, dict)
        or Path(str(loaded.get("resolved_path", ""))).resolve()
        != Path(str(xtbloom["path"])).resolve()
        or loaded.get("sha256") != xtbloom.get("sha256")
    ):
        raise ValueError("loaded xTBloom identity differs from the runtime record")

    require_recorded_artifact(record.get("input"), "generated LAMMPS input")
    require_recorded_artifact(record.get("launcher_log"), "launcher log")
    logs = record.get("lammps_logs")
    if not isinstance(logs, dict) or len(logs) != len(expected_order):
        raise ValueError("production partition log set is incomplete")
    for name, log_identity in logs.items():
        path = require_recorded_artifact(log_identity, f"LAMMPS log {name}")
        if path.name != name:
            raise ValueError("LAMMPS log name differs from its recorded path")
    dangerous = record.get("dangerous_builds")
    if (
        not isinstance(dangerous, dict)
        or set(dangerous) != set(logs)
        or any(value != 0 for value in dangerous.values())
    ):
        raise ValueError("production record has missing or nonzero dangerous builds")

    project = record.get("project")
    if not isinstance(project, dict):
        raise TypeError("production project identity is missing")
    runner_path = require_recorded_artifact(project.get("runner"), "workload runner")
    if runner_path.name != "etpeth_workload.py":
        raise ValueError("production record names an unexpected workload runner")
    recorded_manifest_path = require_recorded_artifact(
        project.get("manifest"), "workload manifest"
    )
    if sha256(recorded_manifest_path) != sha256(manifest_path):
        raise ValueError("analysis manifest differs from the production manifest")
    provenance_path = require_recorded_artifact(
        project.get("provenance"), "source provenance"
    )
    provenance = load_json(provenance_path)
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise TypeError("source provenance is incomplete")
    source_qualification = source.get("qualification")
    if source_qualification not in {"final", "private-diagnostic"}:
        raise ValueError("source provenance qualification is unsupported")
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("source provenance has no artifact ledger")
    for artifact in artifacts:
        require_recorded_artifact(artifact, "tutorial source artifact")
    workload_manifest = provenance.get("workload_manifest")
    if not isinstance(workload_manifest, dict) or workload_manifest.get(
        "sha256"
    ) != sha256(manifest_path):
        raise ValueError("source provenance names a different workload manifest")
    provenance_windows = provenance.get("window_order")
    if not isinstance(provenance_windows, list) or [
        entry.get("tag") for entry in provenance_windows if isinstance(entry, dict)
    ] != list(expected_order):
        raise ValueError("source provenance window order differs from production")

    project_qualification = project.get("qualification")
    if project_qualification not in {"clean-source", "private-diagnostic"}:
        raise ValueError("project qualification is unsupported")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("production dependency provenance is missing")
    dependency_clean = all(
        isinstance(value, dict) and value.get("dirty") is False
        for value in dependencies.values()
    )
    return (
        "final"
        if source_qualification == "final"
        and project_qualification == "clean-source"
        and project.get("dirty") is False
        and dependency_clean
        else "private-diagnostic"
    )


def validate_stage_checkpoint_dag(
    ledger_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    windows: Sequence[Window],
    *,
    expected_stage: str,
    expected_total_steps: int,
    parent_outputs: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Validate one chunked stage and every immutable checkpoint edge.

    ``parent_outputs`` is omitted only for the equilibration stage because its
    seed-walk ancestry is outside the PMF estimator's input contract.  Every
    production trial supplies the accepted equilibration outputs here, so an
    internally self-consistent trial cannot silently start from an unrelated
    or unequilibrated state.
    """
    ledger = load_json(ledger_path)
    expected_order = [window.tag for window in windows]
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "passed"
        or ledger.get("qualification") != "native-chunked"
        or ledger.get("stage") != expected_stage
        or ledger.get("window_order") != expected_order
        or ledger.get("total_steps_per_window") != expected_total_steps
        or ledger.get("series_merge_policy") != series_merge_policy(manifest)
    ):
        raise ValueError(
            f"stage ledger does not match the analysis protocol: {ledger_path}"
        )

    maximum_chunk_steps = int(ledger.get("maximum_chunk_steps", 0))
    if maximum_chunk_steps <= 0:
        raise ValueError(f"stage chunk length is invalid: {ledger_path}")
    full_chunks, remainder = divmod(expected_total_steps, maximum_chunk_steps)
    expected_sizes = [maximum_chunk_steps] * full_chunks
    if remainder:
        expected_sizes.append(remainder)
    chunks = ledger.get("chunks")
    if (
        not isinstance(chunks, list)
        or len(chunks) != ledger.get("chunk_count")
        or len(chunks) != len(expected_sizes)
    ):
        raise ValueError(f"stage chunk ledger is incomplete: {ledger_path}")

    chunk_records: list[dict[str, Any]] = []
    qualifications: list[str] = []
    previous_outputs: dict[str, Any] | None = parent_outputs
    offset = 0
    ranks_per_window: int | None = None
    colvars_by_window: dict[str, list[dict[str, str]]] = {
        tag: [] for tag in expected_order
    }
    trajectories_by_window: dict[str, list[dict[str, str]]] = {
        tag: [] for tag in expected_order
    }
    for chunk_index, (chunk, expected_steps) in enumerate(
        zip(chunks, expected_sizes, strict=True)
    ):
        if not isinstance(chunk, dict):
            raise TypeError(f"stage chunk entry is malformed: {ledger_path}")
        record = chunk.get("record")
        if not isinstance(record, dict) or "path" not in record:
            raise ValueError(f"stage chunk identity is malformed: {ledger_path}")
        record_path = Path(str(record["path"]))
        if not artifact_matches(record, record_path):
            raise ValueError(f"stage chunk record changed: {record_path}")
        payload = load_json(record_path)
        expected_name = (
            f"{expected_stage}-chunk-{chunk_index + 1:03d}-of-{len(chunks):03d}"
        )
        if (
            payload.get("status") != "passed"
            or payload.get("returncode") != 0
            or chunk.get("name") != expected_name
            or payload.get("name") != expected_name
            or chunk.get("timestep_offset") != offset
            or payload.get("timestep_offset") != offset
            or chunk.get("steps_per_window") != expected_steps
            or payload.get("steps_per_window") != expected_steps
            or payload.get("worlds") != len(expected_order)
            or payload.get("window_order") != expected_order
            or set(payload.get("start_inputs", {})) != set(expected_order)
            or set(payload.get("outputs", {})) != set(expected_order)
        ):
            raise ValueError(f"stage chunk record is not accepted: {record_path}")
        current_ranks = int(payload.get("ranks_per_window", 0))
        if current_ranks <= 0:
            raise ValueError("stage ranks-per-window is invalid")
        if ranks_per_window is None:
            ranks_per_window = current_ranks
        elif current_ranks != ranks_per_window:
            raise ValueError("stage ranks-per-window changed between chunks")
        qualifications.append(
            validate_invocation_provenance(payload, manifest_path, expected_order)
        )

        for tag in expected_order:
            start = payload["start_inputs"][tag]
            require_recorded_artifact(start, f"{expected_stage} start state {tag}")
            if previous_outputs is not None and start != previous_outputs[tag]["data"]:
                raise ValueError(
                    f"{expected_stage} checkpoint parent changed for {tag}"
                )
            output = payload["outputs"][tag]
            if not isinstance(output, dict):
                raise TypeError(f"{expected_stage} output is malformed for {tag}")
            for kind in ("data", "restart", "colvars", "trajectory"):
                require_recorded_artifact(
                    output.get(kind), f"{expected_stage} {tag} {kind}"
                )
            colvars_by_window[tag].append(output["colvars"])
            trajectories_by_window[tag].append(output["trajectory"])
        chunk_records.append(payload)
        previous_outputs = payload["outputs"]
        offset += expected_steps

    if offset != expected_total_steps or previous_outputs is None:
        raise ValueError(
            f"stage timeline does not cover the manifest length: {ledger_path}"
        )
    if ledger.get("outputs") != previous_outputs:
        raise ValueError(
            f"stage final outputs differ from the final chunk: {ledger_path}"
        )
    if ledger.get("colvars_by_window") != colvars_by_window:
        raise ValueError(f"stage Colvars ledger differs from its chunks: {ledger_path}")
    if ledger.get("trajectories_by_window") != trajectories_by_window:
        raise ValueError(
            f"stage trajectory ledger differs from its chunks: {ledger_path}"
        )
    qualification = (
        "final"
        if all(value == "final" for value in qualifications)
        else "private-diagnostic"
    )
    return ledger, chunk_records, qualification


def load_trial_ledger(
    ledger_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    windows: Sequence[Window],
    burn_in_samples: int,
    equilibrated_outputs: dict[str, Any],
) -> tuple[str, dict[str, Series], str]:
    """Validate one production ledger and load its hash-pinned Colvars DAG."""
    stage_value = load_json(ledger_path).get("stage")
    if not isinstance(stage_value, str) or not stage_value.startswith(
        "production-trial-"
    ):
        raise ValueError(f"ledger is not an accepted production trial: {ledger_path}")
    stage = stage_value
    expected_order = [window.tag for window in windows]
    total_steps = int(manifest["protocol"]["production_steps_per_window"])
    ledger, chunk_records, overall_qualification = validate_stage_checkpoint_dag(
        ledger_path,
        manifest_path,
        manifest,
        windows,
        expected_stage=stage,
        expected_total_steps=total_steps,
        parent_outputs=equilibrated_outputs,
    )
    chunks = ledger["chunks"]

    recorded = ledger.get("colvars_by_window")
    if not isinstance(recorded, dict) or set(recorded) != set(expected_order):
        raise ValueError(f"production Colvars window set changed in {ledger_path}")
    result: dict[str, Series] = {}
    reaction_tolerance, angle_tolerance = checkpoint_boundary_tolerances(manifest)
    for window in windows:
        artifacts = recorded.get(window.tag)
        if not isinstance(artifacts, list) or len(artifacts) != len(chunks):
            raise ValueError(f"Colvars chunk count changed for {window.tag}")
        parts: list[Series] = []
        for chunk_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict) or "path" not in artifact:
                raise ValueError(f"invalid Colvars identity for {window.tag}")
            chunk_output = chunk_records[chunk_index]["outputs"].get(window.tag, {})
            if artifact != chunk_output.get("colvars"):
                raise ValueError(
                    f"Colvars ledger differs from chunk record for {window.tag}"
                )
            path = Path(artifact["path"])
            if not artifact_matches(artifact, path):
                raise ValueError(f"Colvars bytes changed for {window.tag}: {path}")
            part = parse_colvars(path)
            chunk = chunks[chunk_index]
            chunk_offset = int(chunk["timestep_offset"])
            chunk_steps = int(chunk["steps_per_window"])
            frequency = int(manifest["dynamics"]["colvars_frequency_steps"])
            expected_steps = expected_colvars_steps(
                chunk_offset, chunk_steps, frequency
            )
            if part.steps != expected_steps:
                raise ValueError(
                    f"Colvars timeline is incomplete for {window.tag} chunk {chunk_index + 1}"
                )
            parts.append(part)
        series = merge_series(parts, reaction_tolerance, angle_tolerance)
        if burn_in_samples < 0 or burn_in_samples >= len(series.steps):
            raise ValueError(f"burn-in removes every sample for {window.tag}")
        result[window.tag] = Series(
            series.steps[burn_in_samples:],
            series.reaction_coordinate[burn_in_samples:],
            series.attack_angle[burn_in_samples:],
        )
    return stage, result, overall_qualification


def bin_edges(contract: dict[str, Any]) -> tuple[float, ...]:
    """Construct a predeclared uniform bin grid without cumulative drift."""
    estimator = contract["estimator"]
    start = float(estimator["bin_start_angstrom"])
    stop = float(estimator["bin_stop_angstrom"])
    width = float(estimator["bin_width_angstrom"])
    if not all(math.isfinite(value) for value in (start, stop, width)):
        raise ValueError("analysis bin specification is non-finite")
    if width <= 0.0 or stop <= start:
        raise ValueError("analysis bin specification is invalid")
    count_float = (stop - start) / width
    count = round(count_float)
    if count <= 0 or not math.isclose(count_float, count, abs_tol=1.0e-10):
        raise ValueError("analysis range must be an integer number of bins")
    return tuple(start + index * width for index in range(count + 1))


def histogram(values: Iterable[float], edges: Sequence[float]) -> list[int]:
    """Count values on a fixed grid and fail instead of clipping tails."""
    counts = [0] * (len(edges) - 1)
    for value in values:
        if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
            raise ValueError(
                f"reaction coordinate {value!r} is outside the predeclared analysis range"
            )
        index = bisect.bisect_right(edges, value) - 1
        if index == len(counts):
            index -= 1
        counts[index] += 1
    return counts


def logsumexp(values: Sequence[float]) -> float:
    """Evaluate log(sum(exp(values))) without avoidable overflow."""
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def solve_dense(
    matrix: Sequence[Sequence[float]], right_hand_side: Sequence[float]
) -> list[float]:
    """Solve one small dense system with partial pivoting for WHAM DIIS."""
    size = len(matrix)
    if (
        size == 0
        or len(right_hand_side) != size
        or any(len(row) != size for row in matrix)
    ):
        raise ValueError("dense solve dimensions differ")
    augmented = [
        list(row) + [right_hand_side[index]] for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1.0e-24:
            raise ArithmeticError("dense system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = 1.0 / augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] *= inverse
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    return [augmented[row][size] for row in range(size)]


def diis_extrapolate(
    mapped_history: Sequence[Sequence[float]],
    residual_history: Sequence[Sequence[float]],
) -> list[float]:
    """Accelerate the slow multi-window WHAM fixed point with Pulay DIIS."""
    count = len(mapped_history)
    if count != len(residual_history) or count < 2:
        raise ValueError("DIIS requires matching histories with at least two entries")
    scale = max(
        math.fsum(value * value for value in residual) for residual in residual_history
    )
    regularization = max(1.0e-24, 1.0e-14 * scale)
    matrix = [[0.0] * (count + 1) for _ in range(count + 1)]
    for left in range(count):
        for right in range(count):
            matrix[left][right] = math.fsum(
                first * second
                for first, second in zip(
                    residual_history[left], residual_history[right], strict=True
                )
            )
        matrix[left][left] += regularization
        matrix[left][count] = -1.0
        matrix[count][left] = -1.0
    coefficients = solve_dense(matrix, [0.0] * count + [-1.0])[:count]
    candidate = [
        math.fsum(
            coefficients[history] * mapped_history[history][index]
            for history in range(count)
        )
        for index in range(len(mapped_history[0]))
    ]
    origin = candidate[0]
    candidate = [value - origin for value in candidate]
    if not all(math.isfinite(value) for value in candidate):
        raise ArithmeticError("DIIS produced a non-finite WHAM iterate")
    return candidate


def solve_wham(
    counts: Sequence[Sequence[int]],
    centers: Sequence[float],
    bin_centers: Sequence[float],
    force_constant: float,
    temperature: float,
    tolerance: float,
    maximum_iterations: int,
) -> WhamResult:
    """Solve the standard binned WHAM fixed point in dimensionless form."""
    if len(counts) != len(centers) or not counts:
        raise ValueError("WHAM needs one nonempty histogram per umbrella center")
    nbins = len(bin_centers)
    if any(len(row) != nbins for row in counts):
        raise ValueError("WHAM histogram widths differ")
    samples = [sum(row) for row in counts]
    if any(total <= 0 for total in samples):
        raise ValueError("WHAM received an empty umbrella window")
    if force_constant <= 0.0 or temperature <= 0.0:
        raise ValueError("WHAM force constant and temperature must be positive")
    beta = 1.0 / (K_B_KCAL_PER_MOL_K * temperature)
    beta_bias = [
        [0.5 * beta * force_constant * (x - center) ** 2 for x in bin_centers]
        for center in centers
    ]
    total_by_bin = [sum(row[index] for row in counts) for index in range(nbins)]
    offsets = [0.0] * len(centers)
    probabilities = [0.0] * nbins
    mapped_history: list[list[float]] = []
    residual_history: list[list[float]] = []
    residual = math.inf
    for iteration in range(1, maximum_iterations + 1):
        for index, total in enumerate(total_by_bin):
            if total == 0:
                probabilities[index] = 0.0
                continue
            denominator = logsumexp(
                [
                    math.log(samples[window])
                    + offsets[window]
                    - beta_bias[window][index]
                    for window in range(len(centers))
                ]
            )
            probabilities[index] = math.exp(math.log(total) - denominator)
        normalization = sum(probabilities)
        if not math.isfinite(normalization) or normalization <= 0.0:
            raise RuntimeError("WHAM probability normalization failed")
        probabilities = [value / normalization for value in probabilities]
        updated = []
        for window in range(len(centers)):
            terms = [
                math.log(probabilities[index]) - beta_bias[window][index]
                for index in range(nbins)
                if probabilities[index] > 0.0
            ]
            updated.append(-logsumexp(terms))
        origin = updated[0]
        updated = [value - origin for value in updated]
        residual = max(
            abs(updated[index] - offsets[index]) for index in range(len(offsets))
        )
        if residual <= tolerance:
            offsets = updated
            break
        mapped_history.append(updated)
        residual_history.append(
            [updated[index] - offsets[index] for index in range(len(offsets))]
        )
        if len(mapped_history) > 8:
            mapped_history.pop(0)
            residual_history.pop(0)
        if len(mapped_history) >= 2:
            try:
                offsets = diis_extrapolate(mapped_history, residual_history)
            except ArithmeticError:
                # Nearly dependent residuals are normal near convergence. The
                # unaccelerated fixed point remains a safe fallback.
                offsets = updated
        else:
            offsets = updated
    else:
        raise RuntimeError(
            f"WHAM did not converge in {maximum_iterations} iterations; residual={residual}"
        )

    thermal = K_B_KCAL_PER_MOL_K * temperature
    raw_pmf: list[float | None] = [
        -thermal * math.log(value) if value > 0.0 else None for value in probabilities
    ]
    minimum = min(value for value in raw_pmf if value is not None)
    pmf = tuple(value - minimum if value is not None else None for value in raw_pmf)
    return WhamResult(tuple(probabilities), pmf, tuple(offsets), iteration, residual)


def fft_in_place(values: list[complex], inverse: bool = False) -> None:
    """Radix-2 FFT used only for unbiased autocorrelation estimation."""
    size = len(values)
    if size <= 0 or size & (size - 1):
        raise ValueError("FFT length must be a positive power of two")
    target = 0
    for index in range(1, size):
        bit = size >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target ^= bit
        if index < target:
            values[index], values[target] = values[target], values[index]
    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= size:
        root = cmath.exp(sign * 2j * math.pi / length)
        half = length // 2
        for start in range(0, size, length):
            factor = 1.0 + 0.0j
            for offset in range(half):
                even = values[start + offset]
                odd = values[start + offset + half] * factor
                values[start + offset] = even + odd
                values[start + offset + half] = even - odd
                factor *= root
        length *= 2
    if inverse:
        scale = 1.0 / size
        for index in range(size):
            values[index] *= scale


def statistical_inefficiency(values: Sequence[float]) -> float:
    """Estimate g with Geyer's initial monotone positive autocorrelation sum."""
    size = len(values)
    if size < 2:
        return 1.0
    mean = math.fsum(values) / size
    padded = 1
    while padded < 2 * size:
        padded <<= 1
    work = [complex(value - mean, 0.0) for value in values]
    work.extend([0.0j] * (padded - size))
    fft_in_place(work)
    for index, value in enumerate(work):
        work[index] = complex(value.real * value.real + value.imag * value.imag, 0.0)
    fft_in_place(work, inverse=True)
    variance = work[0].real / size
    if variance <= 0.0 or not math.isfinite(variance):
        return 1.0
    pair_sum = 0.0
    previous = math.inf
    lag = 1
    while lag < size:
        rho_first = (work[lag].real / (size - lag)) / variance
        rho_second = 0.0
        if lag + 1 < size:
            rho_second = (work[lag + 1].real / (size - lag - 1)) / variance
        pair = min(previous, rho_first + rho_second)
        if not math.isfinite(pair) or pair <= 0.0:
            break
        pair_sum += pair
        previous = pair
        lag += 2
    return max(1.0, min(float(size), 1.0 + 2.0 * pair_sum))


def adjacent_overlaps(
    counts: Sequence[Sequence[int]], windows: Sequence[Window]
) -> list[dict[str, Any]]:
    """Compute empirical neighboring Bhattacharyya coefficients."""
    normalized = [[value / sum(row) for value in row] for row in counts]
    result = []
    for index in range(len(windows) - 1):
        coefficient = math.fsum(
            math.sqrt(left * right)
            for left, right in zip(
                normalized[index], normalized[index + 1], strict=True
            )
        )
        result.append(
            {
                "left": windows[index].tag,
                "right": windows[index + 1].tag,
                "coefficient": coefficient,
            }
        )
    return result


def block_histograms(
    values: Sequence[float], block_length: int, edges: Sequence[float]
) -> HistogramBlocks:
    """Represent circular correlation-length blocks for exact-size resampling.

    Full blocks all have the same length. A separate prefix histogram is kept
    for every start so the bootstrap can draw floor(N/L) full blocks and one
    length-(N mod L) prefix, preserving the original N exactly.
    """
    if block_length <= 0:
        raise ValueError("bootstrap block length must be positive")
    if not values:
        raise ValueError("bootstrap needs at least one sample")
    block_length = min(block_length, len(values))
    full_blocks = []
    prefix_blocks = []
    full_draws, remainder = divmod(len(values), block_length)
    for start in range(0, len(values), block_length):
        stop = start + block_length
        if stop <= len(values):
            block = list(values[start:stop])
        else:
            block = list(values[start:]) + list(values[: stop - len(values)])
        full_blocks.append(tuple(histogram(block, edges)))
        prefix_blocks.append(tuple(histogram(block[:remainder], edges)))
    return HistogramBlocks(
        tuple(full_blocks),
        tuple(prefix_blocks),
        full_draws,
        remainder,
        len(values),
    )


def resample_histogram_blocks(blocks: HistogramBlocks, rng: random.Random) -> list[int]:
    """Draw one exact-N histogram from precomputed circular blocks."""
    result = [0] * len(blocks.full[0])
    for _ in range(blocks.full_draws):
        selected = blocks.full[rng.randrange(len(blocks.full))]
        for index, value in enumerate(selected):
            result[index] += value
    if blocks.remainder:
        selected = blocks.prefixes[rng.randrange(len(blocks.prefixes))]
        for index, value in enumerate(selected):
            result[index] += value
    if sum(result) != blocks.sample_count:
        raise RuntimeError("block bootstrap changed the window sample count")
    return result


def bootstrap_pmf(
    trial_series: Sequence[dict[str, Series]],
    windows: Sequence[Window],
    edges: Sequence[float],
    bin_centers: Sequence[float],
    force_constant: float,
    temperature: float,
    tolerance: float,
    maximum_iterations: int,
    replicates: int,
    seed: int,
    inefficiencies: dict[str, list[float]],
) -> list[list[float | None]]:
    """Resample each trial/window independently in correlation-length blocks."""
    if replicates <= 0:
        return []
    block_sets: list[list[HistogramBlocks]] = []
    for trial_index, trial in enumerate(trial_series):
        trial_blocks = []
        for window in windows:
            length = max(1, math.ceil(inefficiencies[window.tag][trial_index]))
            trial_blocks.append(
                block_histograms(trial[window.tag].reaction_coordinate, length, edges)
            )
        block_sets.append(trial_blocks)
    rng = random.Random(seed)
    samples: list[list[float | None]] = []
    for _ in range(replicates):
        counts = [[0] * (len(edges) - 1) for _window in windows]
        for trial_blocks in block_sets:
            for window_index, blocks in enumerate(trial_blocks):
                selected = resample_histogram_blocks(blocks, rng)
                target = counts[window_index]
                for bin_index, value in enumerate(selected):
                    target[bin_index] += value
        result = solve_wham(
            counts,
            [window.center for window in windows],
            bin_centers,
            force_constant,
            temperature,
            tolerance,
            maximum_iterations,
        )
        samples.append(list(result.pmf))
    return samples


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated sample percentile."""
    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def confidence_intervals(
    bootstrap: Sequence[Sequence[float | None]], confidence: float, nbins: int
) -> tuple[list[float | None], list[float | None], list[int]]:
    """Summarize pointwise bootstrap intervals without inventing empty-bin values."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    alpha = 0.5 * (1.0 - confidence)
    lower: list[float | None] = []
    upper: list[float | None] = []
    finite_counts: list[int] = []
    for index in range(nbins):
        values = sorted(
            value
            for sample in bootstrap
            if (value := sample[index]) is not None and math.isfinite(value)
        )
        finite_counts.append(len(values))
        if not values:
            lower.append(None)
            upper.append(None)
        else:
            lower.append(percentile(values, alpha))
            upper.append(percentile(values, 1.0 - alpha))
    return lower, upper, finite_counts


def require_pmf_value(values: Sequence[float | None], index: int) -> float:
    """Return one observed PMF value with explicit optional-value narrowing."""
    value = values[index]
    if value is None:
        raise ValueError(f"PMF bin {index} is unobserved")
    return value


def write_pmf_csv_atomic(
    path: Path,
    rows: Iterable[Sequence[object]],
) -> None:
    """Publish PMF rows atomically before the JSON record references them."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "reaction_coordinate_angstrom",
                "probability_mass",
                "pmf_kcal_mol",
                "ci_lower_kcal_mol",
                "ci_upper_kcal_mol",
                "finite_bootstrap_replicates",
            ]
        )
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def analyze(
    manifest_path: Path,
    contract_path: Path,
    ledger_paths: Sequence[Path],
    output_prefix: Path,
) -> dict[str, Any]:
    """Run the complete predeclared PMF analysis and publish JSON plus CSV."""
    manifest = load_json(manifest_path)
    contract = load_json(contract_path)
    if manifest.get("schema_version") != 1 or contract.get("schema_version") != 1:
        raise ValueError("unsupported workload or analysis schema")
    validate_analysis_contract(manifest, contract)
    windows = windows_from_manifest(manifest)
    sampling = contract["sampling"]
    burn_in = int(sampling["burn_in_samples_per_window"])
    resolved_ledgers = [path.resolve() for path in ledger_paths]
    run_roots = {path.parent.parent for path in resolved_ledgers}
    if len(run_roots) != 1:
        raise ValueError("all production ledgers must belong to one run root")
    run_root = next(iter(run_roots))
    equilibration_path = run_root / "records/equilibrate-complete.json"
    equilibration, _equilibration_chunks, equilibration_qualification = (
        validate_stage_checkpoint_dag(
            equilibration_path,
            manifest_path,
            manifest,
            windows,
            expected_stage="equilibrate",
            expected_total_steps=int(
                manifest["protocol"]["equilibration_steps_per_window"]
            ),
            parent_outputs=None,
        )
    )
    trials = []
    stages = []
    qualifications = [equilibration_qualification]
    expected_trials = int(manifest["protocol"]["production_trials"])
    if len(resolved_ledgers) != expected_trials:
        raise ValueError(
            f"analysis requires {expected_trials} independent production trials, "
            f"found {len(resolved_ledgers)}"
        )
    for trial_index, path in enumerate(resolved_ledgers):
        stage, series, qualification = load_trial_ledger(
            path,
            manifest_path,
            manifest,
            windows,
            burn_in,
            equilibration["outputs"],
        )
        expected_stage = f"production-trial-{trial_index}"
        if stage != expected_stage:
            raise ValueError(
                f"production trial set/order changed: expected {expected_stage}, found {stage}"
            )
        stages.append(stage)
        trials.append(series)
        qualifications.append(qualification)

    edges = bin_edges(contract)
    centers = tuple(
        (edges[index] + edges[index + 1]) * 0.5 for index in range(len(edges) - 1)
    )
    pooled_counts = [[0] * len(centers) for _window in windows]
    inefficiencies: dict[str, list[float]] = {window.tag: [] for window in windows}
    window_statistics: dict[str, Any] = {}
    for window_index, window in enumerate(windows):
        trial_stats = []
        total_ess = 0.0
        total_samples = 0
        pooled_reaction: list[float] = []
        pooled_angle: list[float] = []
        for trial_index, trial in enumerate(trials):
            series = trial[window.tag]
            reaction_g = statistical_inefficiency(series.reaction_coordinate)
            angle_g = statistical_inefficiency(series.attack_angle)
            maximum_g = max(reaction_g, angle_g)
            inefficiencies[window.tag].append(maximum_g)
            ess = len(series.reaction_coordinate) / maximum_g
            total_ess += ess
            total_samples += len(series.reaction_coordinate)
            pooled_reaction.extend(series.reaction_coordinate)
            pooled_angle.extend(series.attack_angle)
            counts = histogram(series.reaction_coordinate, edges)
            for bin_index, value in enumerate(counts):
                pooled_counts[window_index][bin_index] += value
            trial_stats.append(
                {
                    "trial": stages[trial_index],
                    "samples": len(series.reaction_coordinate),
                    "reaction_coordinate_statistical_inefficiency": reaction_g,
                    "attack_angle_statistical_inefficiency": angle_g,
                    "bootstrap_statistical_inefficiency": maximum_g,
                    "effective_samples": ess,
                    "first_step": series.steps[0],
                    "last_step": series.steps[-1],
                }
            )
        window_statistics[window.tag] = {
            "center_angstrom": window.center,
            "samples": total_samples,
            "effective_samples": total_ess,
            "reaction_coordinate_mean_angstrom": statistics.fmean(pooled_reaction),
            "reaction_coordinate_stddev_angstrom": statistics.stdev(pooled_reaction),
            "attack_angle_mean_degree": statistics.fmean(pooled_angle),
            "attack_angle_stddev_degree": statistics.stdev(pooled_angle),
            "trials": trial_stats,
        }

    estimator = contract["estimator"]
    temperature = float(estimator["temperature_kelvin"])
    force_constant = float(
        manifest["umbrella"]["reaction_coordinate"]["force_constant_kcal_mol_angstrom2"]
    )
    tolerance = float(estimator["dimensionless_tolerance"])
    maximum_iterations = int(estimator["maximum_iterations"])
    wham = solve_wham(
        pooled_counts,
        [window.center for window in windows],
        centers,
        force_constant,
        temperature,
        tolerance,
        maximum_iterations,
    )
    trial_wham = []
    for trial in trials:
        trial_counts = [
            histogram(trial[window.tag].reaction_coordinate, edges)
            for window in windows
        ]
        trial_wham.append(
            solve_wham(
                trial_counts,
                [window.center for window in windows],
                centers,
                force_constant,
                temperature,
                tolerance,
                maximum_iterations,
            )
        )
    overlaps = adjacent_overlaps(pooled_counts, windows)

    uncertainty = contract["uncertainty"]
    replicate_count = int(uncertainty["replicates"])
    if replicate_count <= 0:
        raise ValueError("analysis contract requires at least one bootstrap replicate")
    bootstrap = bootstrap_pmf(
        trials,
        windows,
        edges,
        centers,
        force_constant,
        temperature,
        tolerance,
        maximum_iterations,
        replicate_count,
        int(uncertainty["random_seed"]),
        inefficiencies,
    )
    ci_lower, ci_upper, ci_finite = confidence_intervals(
        bootstrap, float(uncertainty["confidence_level"]), len(centers)
    )

    acceptance_contract = manifest["protocol"]["overlap_acceptance"]
    minimum_overlap = min(item["coefficient"] for item in overlaps)
    minimum_ess = min(item["effective_samples"] for item in window_statistics.values())
    common_trial_bins = [
        index
        for index, value in enumerate(wham.pmf)
        if value is not None
        and all(result.pmf[index] is not None for result in trial_wham)
    ]
    if not common_trial_bins:
        raise ValueError("production trials have no common PMF support")
    reference_bin = min(
        common_trial_bins,
        key=lambda index: require_pmf_value(wham.pmf, index),
    )
    aligned_trials: list[list[float | None]] = []
    for result in trial_wham:
        reference = result.pmf[reference_bin]
        if reference is None:
            raise RuntimeError(
                "trial-consistency reference support changed unexpectedly"
            )
        aligned_trials.append(
            [value - reference if value is not None else None for value in result.pmf]
        )
    consistency_contract = contract["trial_consistency"]
    core_limit = float(consistency_contract["maximum_pmf_region_kcal_mol"])
    core_bins = [
        index
        for index, value in enumerate(wham.pmf)
        if value is not None and value <= core_limit
    ]
    if not core_bins:
        raise ValueError("pooled production PMF has no declared core region")
    missing_core_support = [
        {
            "trial": stages[trial_index],
            "bin_indices": [index for index in core_bins if trial[index] is None],
            "coordinates_angstrom": [
                centers[index] for index in core_bins if trial[index] is None
            ],
        }
        for trial_index, trial in enumerate(aligned_trials)
        if any(trial[index] is None for index in core_bins)
    ]
    core_support_passed = not missing_core_support
    maximum_trial_difference = (
        max(
            abs(require_pmf_value(left, index) - require_pmf_value(right, index))
            for trial_index, left in enumerate(aligned_trials)
            for right in aligned_trials[trial_index + 1 :]
            for index in core_bins
        )
        if core_support_passed
        else None
    )
    acceptance = {
        "minimum_adjacent_overlap_coefficient": minimum_overlap,
        "required_minimum_adjacent_overlap_coefficient": float(
            acceptance_contract["minimum_adjacent_overlap_coefficient"]
        ),
        "minimum_effective_samples_per_window": minimum_ess,
        "required_minimum_effective_samples_per_window": float(
            acceptance_contract["minimum_effective_samples_per_window"]
        ),
        "minimum_finite_bootstrap_fraction_per_observed_bin": min(
            ci_finite[index] / len(bootstrap)
            for index, probability in enumerate(wham.probabilities)
            if probability > 0.0
        ),
        "required_minimum_finite_bootstrap_fraction_per_observed_bin": float(
            uncertainty["minimum_finite_fraction_per_observed_bin"]
        ),
        "maximum_pairwise_trial_pmf_difference_kcal_mol": maximum_trial_difference,
        "allowed_maximum_pairwise_trial_pmf_difference_kcal_mol": float(
            consistency_contract["maximum_pairwise_absolute_difference_kcal_mol"]
        ),
        "trial_consistency_core_bins": len(core_bins),
        "trial_consistency_core_limit_kcal_mol": core_limit,
        "trial_consistency_core_support_passed": core_support_passed,
        "trial_consistency_missing_core_support": missing_core_support,
    }
    acceptance["overlap_passed"] = (
        minimum_overlap >= acceptance["required_minimum_adjacent_overlap_coefficient"]
    )
    acceptance["ess_passed"] = (
        minimum_ess >= acceptance["required_minimum_effective_samples_per_window"]
    )
    acceptance["uncertainty_passed"] = (
        acceptance["minimum_finite_bootstrap_fraction_per_observed_bin"]
        >= acceptance["required_minimum_finite_bootstrap_fraction_per_observed_bin"]
    )
    acceptance["trial_consistency_passed"] = (
        maximum_trial_difference is not None
        and maximum_trial_difference
        <= acceptance["allowed_maximum_pairwise_trial_pmf_difference_kcal_mol"]
    )
    acceptance["passed"] = bool(
        acceptance["overlap_passed"]
        and acceptance["ess_passed"]
        and acceptance["uncertainty_passed"]
        and acceptance["trial_consistency_passed"]
    )

    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    write_pmf_csv_atomic(
        csv_path,
        zip(
            centers,
            wham.probabilities,
            wham.pmf,
            ci_lower,
            ci_upper,
            ci_finite,
            strict=True,
        ),
    )

    analysis_project = analysis_project_record()
    input_qualification = (
        "final"
        if all(value == "final" for value in qualifications)
        else "private-diagnostic"
    )
    result = {
        "schema_version": 1,
        "status": "passed" if acceptance["passed"] else "acceptance-failed",
        "qualification": (
            "final"
            if input_qualification == "final" and not analysis_project["dirty"]
            else "private-diagnostic"
        ),
        "input_qualification": input_qualification,
        "analysis_project": analysis_project,
        "manifest": identity(manifest_path),
        "analysis_contract": identity(contract_path),
        "equilibration_ledger": identity(equilibration_path),
        "production_ledgers": [identity(path) for path in resolved_ledgers],
        "trial_order": stages,
        "estimator": contract["estimator"],
        "sampling": contract["sampling"],
        "uncertainty": contract["uncertainty"],
        "trial_consistency": {
            "contract": contract["trial_consistency"],
            "pooled_reference_bin": reference_bin,
            "pooled_reference_coordinate_angstrom": centers[reference_bin],
            "trial_wham": [
                {
                    "trial": stages[index],
                    "iterations": result.iterations,
                    "dimensionless_residual": result.residual,
                }
                for index, result in enumerate(trial_wham)
            ],
        },
        "window_statistics": window_statistics,
        "adjacent_overlaps": overlaps,
        "acceptance": acceptance,
        "wham": {
            "iterations": wham.iterations,
            "dimensionless_residual": wham.residual,
            "dimensionless_offsets": list(wham.dimensionless_offsets),
            "observed_bins": sum(value > 0.0 for value in wham.probabilities),
            "total_bins": len(wham.probabilities),
        },
        "pmf_csv": None,
        "interpretation": (
            "Reaction-coordinate PMF under the common 180-degree attack-angle restraint; "
            "the angle restraint is identical in every umbrella window and is not removed."
        ),
    }
    # Record the CSV only after it is closed and its exact bytes are stable.
    result["pmf_csv"] = identity(csv_path)
    write_json_atomic(json_path, result)
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a complete analysis record."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface without hidden estimator choices."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    result.add_argument(
        "--run",
        type=Path,
        required=True,
        help="ETP/ETH run root containing records/production-trial-N-complete.json",
    )
    result.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path without extension; .json and .csv are written atomically",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze exactly the manifest-declared number of production trials."""
    arguments = parser().parse_args(argv)
    manifest = load_json(arguments.manifest)
    trial_count = int(manifest["protocol"]["production_trials"])
    ledgers = [
        arguments.run.resolve() / "records" / f"production-trial-{trial}-complete.json"
        for trial in range(trial_count)
    ]
    result = analyze(
        arguments.manifest.resolve(),
        arguments.contract.resolve(),
        ledgers,
        arguments.output_prefix,
    )
    acceptance = result["acceptance"]
    print(
        f"PMF {result['status']}: min overlap={acceptance['minimum_adjacent_overlap_coefficient']:.6g}, "
        f"min ESS={acceptance['minimum_effective_samples_per_window']:.6g}"
    )
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
