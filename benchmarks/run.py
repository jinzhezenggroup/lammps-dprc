#!/usr/bin/env python3
"""Run the complete ETP/ETH three-mode batch performance matrix.

The runner keeps process setup separate from steady-state LAMMPS loop timing.
Each available coordinate is warmed once, then sampled repeatedly in the same
LAMMPS process with ``run ... pre no``.  For partitioned runs, the physical-GPU
throughput denominator is the maximum synchronized loop time reported by any
window, never the sum or a favorable individual partition.

Missing runtimes and models remain explicit ``unavailable`` rows.  A timing is
scientifically eligible only when source cleanliness and an external
correctness ledger cover the exact mode, batch size, and runtime identity.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "benchmarks/matrix.json"
DEFAULT_MANIFEST = ROOT / "workloads/etpeth/manifest.json"
MODES = ("classical", "qmmm", "qmmm-dpa4c")
CLASSICAL_BACKENDS = ("batched-dprc", "upstream-gpu")
LOOP_TIME = re.compile(
    r"Loop time of\s+([0-9.eE+-]+)\s+on\s+\d+\s+procs?\s+for\s+(\d+)\s+steps"
)
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def load_workload_module() -> Any:
    """Load the repository runner without requiring ``tools`` to be a package."""
    path = ROOT / "tools/etpeth_workload.py"
    spec = importlib.util.spec_from_file_location("etpeth_workload_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load workload runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKLOAD = load_workload_module()


def load_matrix(path: Path) -> dict[str, Any]:
    """Read and validate the comparison axes that must never be silently reduced."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("benchmark matrix schema_version must be 2")
    axes = payload.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("benchmark matrix axes are missing")
    if tuple(axes.get("mode", ())) != MODES:
        raise ValueError(f"benchmark modes must be {MODES}")
    if axes.get("batch_size") != [1, 2, 4, 8, 16, 32, 48]:
        raise ValueError("benchmark batch sizes must be 1,2,4,8,16,32,48")
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("benchmark measurement policy is missing")
    for name in ("warmup_steps", "sample_steps", "repetitions"):
        if int(measurement.get(name, 0)) < 1:
            raise ValueError(f"benchmark measurement {name} must be positive")
    if measurement.get("model_deviation_frequency_default") != 0:
        raise ValueError(
            "benchmark production model-deviation frequency default must be zero"
        )
    required = payload.get("required_correctness")
    if not isinstance(required, list) or not required:
        raise ValueError("benchmark correctness requirements are missing")
    return payload


def artifact(path: Path) -> dict[str, Any]:
    """Record an immutable runtime input identity."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": WORKLOAD.sha256(resolved),
    }


def deepmd_dependency_record(
    arguments: argparse.Namespace, loaded_deepmd_c: dict[str, str]
) -> dict[str, Any]:
    """Qualify the declared DeePMD cohort against source and loaded bytes."""
    assert arguments.deepmd_artifact_manifest is not None
    assert arguments.deepmd_source is not None
    assert arguments.deepmd_include_dir is not None
    manifest_path = arguments.deepmd_artifact_manifest.resolve()
    reasons: list[str] = []
    verifier = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/deepmd_artifact_manifest.py"),
            "verify",
            "--source",
            str(arguments.deepmd_source.resolve()),
            "--include-dir",
            str(arguments.deepmd_include_dir.resolve()),
            "--library",
            str(Path(loaded_deepmd_c["resolved_path"]).resolve()),
            "--manifest",
            str(manifest_path),
            "--expected-revision",
            arguments.expected_deepmd_revision,
            "--expected-library-sha256",
            loaded_deepmd_c["sha256"],
        ],
        check=False,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        payload = json.loads(verifier.stdout)
    except (OSError, json.JSONDecodeError) as error:
        payload = {}
        reasons.append(f"artifact-manifest verifier returned invalid output: {error}")
    if verifier.returncode != 0:
        diagnostic = verifier.stdout.strip().splitlines()
        reasons.append(
            "artifact-manifest verification failed"
            + (f": {diagnostic[-1]}" if diagnostic else "")
        )

    required_digests = (
        "source_state_sha256",
        "source_c_api_header_sha256",
        "installed_c_api_header_sha256",
        "c_api_library_sha256",
    )
    if payload.get("schema_version") != 1:
        reasons.append("artifact manifest schema_version is not 1")
    if payload.get("source_revision") != arguments.expected_deepmd_revision:
        reasons.append("source revision differs from the reviewed DeePMD pin")
    if payload.get("source_clean") is not True:
        reasons.append("declared DeePMD source checkout is not clean")
    if not isinstance(payload.get("c_api_version"), int) or payload.get(
        "c_api_version", 0
    ) < 31:
        reasons.append("declared DeePMD C API version is older than 31")
    for name in required_digests:
        value = payload.get(name)
        if not isinstance(value, str) or SHA256_HEX.fullmatch(value) is None:
            reasons.append(f"artifact manifest has invalid {name}")
    if (
        payload.get("source_c_api_header_sha256")
        != payload.get("installed_c_api_header_sha256")
    ):
        reasons.append("installed C API header differs from declared source header")
    if payload.get("c_api_library_sha256") != loaded_deepmd_c.get("sha256"):
        reasons.append("loaded DeePMD C API library differs from the declared cohort")

    return {
        "manifest": artifact(manifest_path),
        "expected_source_revision": arguments.expected_deepmd_revision,
        "declared": payload,
        "loaded_library": {
            "soname": loaded_deepmd_c.get("soname"),
            "sha256": loaded_deepmd_c.get("sha256"),
        },
        "publication_qualified": not reasons,
        "qualification_reasons": reasons,
    }


def publication_eligibility_reasons(
    *,
    mode: str,
    source: dict[str, Any],
    project: dict[str, Any],
    deepmd_dependency: dict[str, Any] | None,
    dpa4c_models_qualified: bool,
    dangerous_builds: dict[str, int],
    correctness_reasons: Sequence[str],
) -> list[str]:
    """Return every reason a successful timing cannot support publication."""
    reasons = list(correctness_reasons)
    if source.get("qualification") != "final":
        reasons.append(
            f"tutorial source qualification is {source.get('qualification')}"
        )
    if project.get("dirty"):
        reasons.append("LAMMPS-DPRc source tree is dirty")
    for name, record in project.get("dependencies", {}).items():
        if record.get("required") and not record.get("publication_qualified"):
            detail = ", ".join(record.get("qualification_reasons", []))
            reasons.append(
                f"required dependency {name} is not publication-qualified"
                + (f": {detail}" if detail else "")
            )
    if mode == "qmmm-dpa4c":
        if deepmd_dependency is None:
            reasons.append("DeePMD artifact qualification is unavailable")
        elif not deepmd_dependency.get("publication_qualified"):
            detail = ", ".join(
                deepmd_dependency.get("qualification_reasons", [])
            )
            reasons.append(
                "DeePMD artifact cohort is not publication-qualified"
                + (f": {detail}" if detail else "")
            )
        if not dpa4c_models_qualified:
            reasons.append(
                "DPA4c artifacts were admitted only for an unqualified performance "
                "diagnostic and are not xTB-based DPRc evidence"
            )
    if any(value != 0 for value in dangerous_builds.values()):
        reasons.append("dangerous neighbor builds were reported")
    return reasons


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile for a small raw sample."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def parse_loop_records(path: Path) -> list[tuple[float, int]]:
    """Read every LAMMPS loop timing and its exact step count from one log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    records = [(float(seconds), int(steps)) for seconds, steps in LOOP_TIME.findall(text)]
    if any(not math.isfinite(seconds) or seconds <= 0.0 for seconds, _ in records):
        raise ValueError(f"LAMMPS log contains an invalid loop time: {path}")
    return records


def collect_samples(
    logs: Sequence[Path],
    *,
    batch_size: int,
    warmup_steps: int,
    sample_steps: int,
    repetitions: int,
    timestep_fs: float,
) -> dict[str, Any]:
    """Build synchronized raw samples from all partition timing summaries."""
    if len(logs) != batch_size:
        raise ValueError(
            f"expected {batch_size} LAMMPS logs, found {len(logs)}"
        )
    expected_steps = [warmup_steps] + [sample_steps] * repetitions
    by_log: dict[str, list[float]] = {}
    for path in logs:
        records = parse_loop_records(path)
        actual_steps = [steps for _, steps in records]
        if actual_steps != expected_steps:
            raise ValueError(
                f"LAMMPS timing segments in {path} are {actual_steps}, "
                f"expected {expected_steps}"
            )
        by_log[path.name] = [seconds for seconds, _ in records]

    warmup_per_world = {name: values[0] for name, values in by_log.items()}
    samples: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        per_world = {
            name: values[repetition + 1] for name, values in by_log.items()
        }
        synchronized_seconds = max(per_world.values())
        throughput = batch_size * sample_steps / synchronized_seconds
        samples.append(
            {
                "repetition": repetition,
                "per_world_loop_seconds": per_world,
                "synchronized_loop_seconds": synchronized_seconds,
                "aggregate_window_steps_per_second": throughput,
                "aggregate_nanoseconds_per_day": (
                    throughput * timestep_fs * 1.0e-6 * 86400.0
                ),
            }
        )

    throughputs = [
        float(sample["aggregate_window_steps_per_second"]) for sample in samples
    ]
    seconds = [float(sample["synchronized_loop_seconds"]) for sample in samples]
    return {
        "warmup": {
            "steps_per_window": warmup_steps,
            "per_world_loop_seconds": warmup_per_world,
            "synchronized_loop_seconds": max(warmup_per_world.values()),
        },
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "aggregate_window_steps_per_second": {
                "minimum": min(throughputs),
                "median": statistics.median(throughputs),
                "p95": percentile(throughputs, 0.95),
                "maximum": max(throughputs),
            },
            "synchronized_loop_seconds": {
                "minimum": min(seconds),
                "median": statistics.median(seconds),
                "p95": percentile(seconds, 0.95),
                "maximum": max(seconds),
            },
        },
    }


def parse_mode_artifacts(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``MODE=PATH`` arguments without accepting duplicates."""
    result: dict[str, Path] = {}
    for value in values:
        mode, separator, raw_path = value.partition("=")
        if not separator or mode not in MODES or not raw_path:
            raise ValueError(f"expected MODE=PATH with MODE in {MODES}: {value}")
        if mode in result:
            raise ValueError(f"duplicate artifact for mode {mode}")
        result[mode] = Path(raw_path)
    return result


def correctness_record(
    mode: str,
    batch_size: int,
    required_checks: Sequence[str],
    evidence_by_mode: dict[str, Path],
    runtime_identity: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Require correctness evidence for this exact runtime execution path."""
    path = evidence_by_mode.get(mode)
    if path is None:
        return {"status": "unqualified"}, ["correctness evidence was not supplied"]
    if not path.is_file():
        return {"status": "unqualified", "path": str(path)}, [
            "correctness evidence file is unavailable"
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks", {})
    reasons: list[str] = []
    if payload.get("schema_version") != 2:
        reasons.append("correctness evidence schema is not 2")
    if payload.get("status") != "passed" or payload.get("mode") != mode:
        reasons.append("correctness evidence status or mode does not match")
    if batch_size not in payload.get("batch_sizes", []):
        reasons.append("correctness evidence does not cover this batch size")
    if payload.get("runtime_identity") != runtime_identity:
        reasons.append("correctness evidence runtime identity does not match")
    missing = [name for name in required_checks if checks.get(name) is not True]
    if missing:
        reasons.append("correctness checks are incomplete: " + ", ".join(missing))
    return {
        "status": "passed" if not reasons else "unqualified",
        "evidence": artifact(path),
        "checks": checks,
    }, reasons


def availability_reasons(
    mode: str, batch_size: int, arguments: argparse.Namespace
) -> list[str]:
    """Explain why a coordinate cannot execute instead of dropping it."""
    reasons: list[str] = []
    if not arguments.lammps.is_file():
        reasons.append("LAMMPS executable is unavailable")
    if batch_size > 1:
        try:
            WORKLOAD.resolve_executable(arguments.mpiexec)
        except ValueError:
            reasons.append("MPI launcher is unavailable")
    uses_dprc_plugin = mode != "classical" or arguments.classical_backend == "batched-dprc"
    if uses_dprc_plugin:
        if arguments.plugin is None or not arguments.plugin.is_file():
            reasons.append("LAMMPS-DPRc plugin is unavailable")
        if arguments.xtbloom_library is None or not arguments.xtbloom_library.is_file():
            reasons.append("xTBloom library is unavailable")
    if mode == "qmmm-dpa4c":
        if (
            arguments.deepmd_artifact_manifest is None
            or not arguments.deepmd_artifact_manifest.is_file()
        ):
            reasons.append("DeePMD artifact manifest is unavailable")
        if arguments.deepmd_source is None or not arguments.deepmd_source.is_dir():
            reasons.append("DeePMD source checkout is unavailable")
        if (
            arguments.deepmd_include_dir is None
            or not (arguments.deepmd_include_dir / "deepmd/c_api.h").is_file()
        ):
            reasons.append("DeePMD C API include directory is unavailable")
        if not arguments.expected_deepmd_revision:
            reasons.append("reviewed DeePMD source revision is unspecified")
        if arguments.model_deviation_frequency != 0:
            reasons.append(
                "the in-plugin DeePMD C API path requires model deviation to be disabled"
            )
        if len(arguments.deepmd_model) != 1:
            reasons.append(
                "DPA4c schedule requires exactly one model "
                f"artifact(s), found {len(arguments.deepmd_model)}"
            )
        elif any(not path.is_file() for path in arguments.deepmd_model):
            reasons.append("one or more DPA4c model artifacts are unavailable")
        if (
            not arguments.dpa4c_models_qualified
            and not arguments.allow_unqualified_dpa4c_models
        ):
            reasons.append("DPA4c artifacts are not qualified as xTB-based DPRc models")
    return reasons


def runtime_environment(
    mode: str, arguments: argparse.Namespace
) -> tuple[dict[str, str], dict[str, str]]:
    """Construct the explicit one-GPU, one-thread-per-rank runtime boundary."""
    uses_dprc_plugin = mode != "classical" or arguments.classical_backend == "batched-dprc"
    if uses_dprc_plugin:
        assert arguments.xtbloom_library is not None
        environment, selected = WORKLOAD.build_runtime_environment(
            arguments.xtbloom_library,
            arguments.library_dir,
            arguments.cuda_visible_devices,
        )
    else:
        environment = os.environ.copy()
        library_paths = [str(path.resolve()) for path in arguments.library_dir]
        previous = environment.get("LD_LIBRARY_PATH", "")
        if previous:
            library_paths.append(previous)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
        environment["CUDA_VISIBLE_DEVICES"] = arguments.cuda_visible_devices
        environment["OMP_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment.setdefault("HYDRA_LAUNCHER", "fork")
        selected = {
            name: environment.get(name, "")
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "HYDRA_LAUNCHER",
            )
        }
    if mode == "qmmm-dpa4c":
        # Rank zero owns the shared DeePMD model, but the backend can still
        # create CPU operator pools. Pin both pools to avoid oversubscription.
        environment["DP_INTRA_OP_PARALLELISM_THREADS"] = "1"
        environment["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
        selected["DP_INTRA_OP_PARALLELISM_THREADS"] = "1"
        selected["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
    return environment, selected


def selected_windows(
    windows: Sequence[Any], manifest: dict[str, Any], batch_size: int
) -> list[Any]:
    """Choose a stable contiguous subset centered on the tutorial anchor."""
    anchor_center = int(
        manifest["umbrella"]["available_initial_center_tenths_angstrom"]
    )
    anchor_index = next(
        index
        for index, window in enumerate(windows)
        if window.center_tenths == anchor_center
    )
    first = max(0, min(anchor_index - batch_size // 2, len(windows) - batch_size))
    return list(windows[first : first + batch_size])


def coordinate_execution_backend(
    mode: str, arguments: argparse.Namespace
) -> str:
    """Return the LAMMPS backend that actually executes one coordinate."""
    uses_batched_classical = (
        mode != "classical" or arguments.classical_backend == "batched-dprc"
    )
    return (
        arguments.lammps_execution_backend if uses_batched_classical else "host"
    )


def coordinate_identity(
    mode: str, batch_size: int, arguments: argparse.Namespace
) -> dict[str, Any]:
    """Record every binary or model whose bytes determine one row."""
    uses_batched_classical = (
        mode != "classical" or arguments.classical_backend == "batched-dprc"
    )
    execution_backend = coordinate_execution_backend(mode, arguments)
    inputs: dict[str, Any] = {
        "lammps": artifact(arguments.lammps),
        "lammps_execution_backend": execution_backend,
    }
    uses_dprc_plugin = uses_batched_classical
    if uses_dprc_plugin:
        assert arguments.plugin is not None and arguments.xtbloom_library is not None
        inputs["plugin"] = artifact(arguments.plugin)
        inputs["xtbloom"] = artifact(arguments.xtbloom_library)
    if mode == "qmmm-dpa4c":
        assert arguments.deepmd_artifact_manifest is not None
        inputs["deepmd_artifact_manifest"] = artifact(
            arguments.deepmd_artifact_manifest
        )
        inputs["expected_deepmd_revision"] = arguments.expected_deepmd_revision
        inputs["models"] = [artifact(path) for path in arguments.deepmd_model]
        inputs["dprc_schedule"] = {
            "primary_model_index": 0,
            "model_count": len(arguments.deepmd_model),
            "model_deviation_frequency_steps": arguments.model_deviation_frequency,
            "model_deviation_enabled": False,
            "execution_backend": "dprcplugin-deepmd-c-api-batch",
            "models_qualified_as_xtb_dprc": arguments.dpa4c_models_qualified,
        }
    inputs["batch_size"] = batch_size
    if mode == "classical":
        inputs["classical_backend"] = arguments.classical_backend
    return inputs


def scientific_execution_contract(
    *,
    arguments: argparse.Namespace,
    manifest: dict[str, Any],
    matrix: dict[str, Any],
    source: dict[str, Any],
    execution_backend: str,
    warmup_steps: int,
    sample_steps: int,
    repetitions: int,
) -> dict[str, Any]:
    """Bind correctness to the scientific inputs and launch protocol.

    The contract is intentionally independent of the individual batch size so
    one ledger can qualify several explicitly listed batch sizes. The batch
    membership is deterministic from the manifest and benchmark-runner bytes.
    Local staging paths are excluded; content hashes and numerical parameters
    are retained.
    """
    tutorial_artifacts = sorted(
        (
            {
                "relative_path": str(item["relative_path"]),
                "sha256": str(item["sha256"]),
            }
            for item in source["artifacts"]
        ),
        key=lambda item: item["relative_path"],
    )
    return {
        "workload_manifest": artifact(arguments.manifest),
        # Keep the parsed payload as well as its byte hash so the QM region,
        # PME mesh, cutoffs, timestep, thermostat, and umbrella protocol are
        # directly auditable from a correctness ledger.
        "scientific_parameters": manifest,
        "tutorial_source": {
            "revision": source["revision"],
            "artifacts": tutorial_artifacts,
        },
        "input_protocol": {
            "workload_renderer": artifact(ROOT / "tools/etpeth_workload.py"),
            "benchmark_runner": artifact(Path(__file__)),
            "benchmark_matrix": artifact(arguments.matrix),
            "window_selection": "contiguous-centered-on-manifest-anchor",
            "warmup_steps_per_window": warmup_steps,
            "sample_steps_per_window": sample_steps,
            "repetitions": repetitions,
            "trajectory_frequency_steps": 0,
        },
        "launch_policy": {
            "mpi_launcher": artifact(arguments.mpiexec),
            "mpi_arguments": list(arguments.mpi_arg),
            "worlds": "one-per-selected-umbrella-window",
            "ranks_per_window": 1,
            "lammps_arguments": list(
                WORKLOAD.lammps_backend_arguments(execution_backend)
            ),
        },
    }


def correctness_runtime_identity(inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove relocatable paths and batch size from a coordinate identity.

    A correctness ledger can cover several batch sizes, but it must still bind
    the exact executable, plugin, libraries, models, scientific inputs,
    backend, and launch policy used for timing. Artifact byte counts and local
    paths are not part of that identity because identical reviewed files may be
    staged elsewhere.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            if "sha256" in value and {"path", "bytes", "sha256"}.issubset(value):
                return {"sha256": value["sha256"]}
            return {
                key: normalize(item)
                for key, item in value.items()
                if key != "batch_size"
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(inputs)


def run_coordinate(
    *,
    mode: str,
    batch_size: int,
    arguments: argparse.Namespace,
    matrix: dict[str, Any],
    manifest: dict[str, Any],
    windows: Sequence[Any],
    output: Path,
    project: dict[str, Any],
    source: dict[str, Any],
    correctness_by_mode: dict[str, Path],
) -> dict[str, Any]:
    """Execute one available coordinate and return a complete evidence row."""
    measurement = matrix["measurement"]
    warmup_steps = int(arguments.warmup_steps or measurement["warmup_steps"])
    sample_steps = int(arguments.sample_steps or measurement["sample_steps"])
    repetitions = int(arguments.repetitions or measurement["repetitions"])
    execution_backend = coordinate_execution_backend(mode, arguments)
    coordinate = (
        output
        / "coordinates"
        / execution_backend
        / mode
        / f"batch-{batch_size}"
    )
    log_directory = coordinate / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    input_path = coordinate / "input.lammps"
    launcher_log = coordinate / "launcher.log"
    initial_data = arguments.tutorial.resolve() / "lammps/ETP_ETH.data"
    run_windows = [
        WORKLOAD.RunWindow(
            window,
            initial_data,
            coordinate / "outputs" / window.tag,
            output,
            WORKLOAD.seed_for(manifest, window, 7),
        )
        for window in selected_windows(windows, manifest, batch_size)
    ]
    run_commands = ["timer full sync", f"run {warmup_steps}"]
    run_commands.extend(
        f"run {sample_steps} pre no" for _ in range(repetitions)
    )
    input_path.write_text(
        WORKLOAD.render_lammps_input(
            manifest,
            arguments.tutorial.resolve(),
            arguments.plugin,
            run_windows,
            steps=warmup_steps + repetitions * sample_steps,
            trajectory_frequency=0,
            mode=mode,
            classical_backend=arguments.classical_backend,
            lammps_execution_backend=execution_backend,
            # DeePMD model inputs belong only to the DPRc coordinate. Passing
            # them to other modes would violate the renderer's fail-closed
            # boundary.
            deepmd_models=(
                arguments.deepmd_model if mode == "qmmm-dpa4c" else ()
            ),
            model_deviation_frequency=(
                arguments.model_deviation_frequency
                if mode == "qmmm-dpa4c"
                else 0
            ),
            run_commands=run_commands,
            execution_directory=coordinate,
        ),
        encoding="utf-8",
    )

    # ``--mpiexec`` accepts either an explicit executable path or a command
    # name discovered through PATH.  A serial coordinate does not invoke MPI,
    # so it remains runnable when no launcher is installed.
    mpi_launcher = (
        WORKLOAD.resolve_executable(arguments.mpiexec)
        if batch_size > 1
        else arguments.mpiexec
    )
    command = WORKLOAD.build_lammps_command(
        lammps=arguments.lammps,
        mpi_launcher=mpi_launcher,
        mpi_args=arguments.mpi_arg,
        worlds=batch_size,
        ranks_per_window=1,
        log_directory=log_directory,
        input_path=input_path,
        lammps_args=WORKLOAD.lammps_backend_arguments(execution_backend),
    )
    environment, selected_environment = runtime_environment(mode, arguments)
    loaded_xtbloom = None
    loaded_deepmd_c = None
    uses_dprc_plugin = mode != "classical" or arguments.classical_backend == "batched-dprc"
    if uses_dprc_plugin:
        assert arguments.plugin is not None and arguments.xtbloom_library is not None
        loaded_xtbloom = WORKLOAD.verify_loaded_xtbloom(
            arguments.plugin, arguments.xtbloom_library, environment
        )
        loaded_deepmd_c = WORKLOAD.verify_loaded_deepmd_c(
            arguments.plugin,
            environment,
            required=mode == "qmmm-dpa4c",
        )

    print(f"run: {mode} batch={batch_size}: {shlex.join(command)}", flush=True)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    with launcher_log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            cwd=coordinate,
        )
    process_wall_seconds = time.monotonic() - started
    inputs = coordinate_identity(mode, batch_size, arguments)
    inputs["scientific_execution_contract"] = scientific_execution_contract(
        arguments=arguments,
        manifest=manifest,
        matrix=matrix,
        source=source,
        execution_backend=execution_backend,
        warmup_steps=warmup_steps,
        sample_steps=sample_steps,
        repetitions=repetitions,
    )
    if mode == "qmmm-dpa4c":
        # Unlike xTBloom, the CLI does not name an expected DeePMD library.
        # Bind correctness to the exact runtime selected by the dynamic loader
        # so a different libdeepmd_c cannot reuse an older qualification.
        assert loaded_deepmd_c is not None
        inputs["loaded_deepmd_c"] = {
            "soname": loaded_deepmd_c["soname"],
            "sha256": loaded_deepmd_c["sha256"],
        }
        deepmd_dependency = deepmd_dependency_record(arguments, loaded_deepmd_c)
        inputs["deepmd_dependency"] = deepmd_dependency
    else:
        deepmd_dependency = None
    base: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "batch_size": batch_size,
        "lammps_execution_backend": execution_backend,
        "classical_backend": (
            arguments.classical_backend if mode == "classical" else None
        ),
        "status": "failed",
        "started_utc": started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "process_wall_seconds_including_setup": process_wall_seconds,
        "command": command,
        "selected_environment": selected_environment,
        "inputs": inputs,
        "generated_input": artifact(input_path),
        "launcher_log": artifact(launcher_log),
        "loaded_xtbloom": loaded_xtbloom,
        "loaded_deepmd_c": loaded_deepmd_c,
        "deepmd_dependency": deepmd_dependency,
        "project": project,
        "source_qualification": source["qualification"],
        "window_order": [item.window.tag for item in run_windows],
    }
    if process.returncode != 0:
        base["returncode"] = process.returncode
        diagnostics = launcher_log.read_text(
            encoding="utf-8", errors="replace"
        )
        if not diagnostics:
            failed_logs = sorted(log_directory.glob("log.lammps*"))
            diagnostics = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")[-8000:]
                for path in failed_logs
            )
        base["error"] = diagnostics[-16000:]
        return base

    logs = (
        [log_directory / "log.lammps"]
        if batch_size == 1
        else sorted(log_directory.glob("log.lammps.*"))
    )
    timing = collect_samples(
        logs,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        sample_steps=sample_steps,
        repetitions=repetitions,
        timestep_fs=float(manifest["dynamics"]["timestep_fs"]),
    )
    dangerous = WORKLOAD.inspect_dangerous_builds(log_directory, batch_size)
    missing_outputs: list[str] = []
    outputs: dict[str, Any] = {}
    for item in run_windows:
        if not item.final_data.is_file() or not item.final_restart.is_file():
            missing_outputs.append(item.window.tag)
            continue
        outputs[item.window.tag] = {
            "data": artifact(item.final_data),
            "restart": artifact(item.final_restart),
        }
    if missing_outputs:
        base["error"] = "missing final outputs: " + ", ".join(missing_outputs)
        return base

    correctness, correctness_reasons = correctness_record(
        mode,
        batch_size,
        matrix["required_correctness"],
        correctness_by_mode,
        correctness_runtime_identity(inputs),
    )
    eligibility_reasons = publication_eligibility_reasons(
        mode=mode,
        source=source,
        project=project,
        deepmd_dependency=deepmd_dependency,
        dpa4c_models_qualified=arguments.dpa4c_models_qualified,
        dangerous_builds=dangerous,
        correctness_reasons=correctness_reasons,
    )

    base.update(
        {
            "status": "passed",
            "returncode": 0,
            "timing": timing,
            "dangerous_builds": dangerous,
            "logs": [artifact(path) for path in logs],
            "outputs": outputs,
            "correctness": correctness,
            "evidence_eligible": not eligibility_reasons,
            "eligibility_reasons": eligibility_reasons,
        }
    )
    return base


def environment_record(arguments: argparse.Namespace) -> dict[str, Any]:
    """Capture host/tool identities once for the complete physical-GPU matrix."""
    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": WORKLOAD.socket.gethostname(),
        "python": sys.version,
        "cpu": WORKLOAD.command_output(["lscpu"]),
        "nvidia_smi": WORKLOAD.command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap,pstate,clocks.sm,clocks.mem,power.limit",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": WORKLOAD.command_output(["nvcc", "--version"]),
        "mpiexec": WORKLOAD.command_output([str(arguments.mpiexec), "--version"]),
        "process_affinity_cpus": sorted(os.sched_getaffinity(0)),
    }


def write_samples_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write one compact raw-sample table while retaining unavailable rows in JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "mode",
                "batch_size",
                "repetition",
                "synchronized_loop_seconds",
                "aggregate_window_steps_per_second",
                "aggregate_nanoseconds_per_day",
                "evidence_eligible",
            ),
        )
        writer.writeheader()
        for row in rows:
            for sample in row.get("timing", {}).get("samples", []):
                writer.writerow(
                    {
                        "mode": row["mode"],
                        "batch_size": row["batch_size"],
                        "repetition": sample["repetition"],
                        "synchronized_loop_seconds": sample[
                            "synchronized_loop_seconds"
                        ],
                        "aggregate_window_steps_per_second": sample[
                            "aggregate_window_steps_per_second"
                        ],
                        "aggregate_nanoseconds_per_day": sample[
                            "aggregate_nanoseconds_per_day"
                        ],
                        "evidence_eligible": row.get("evidence_eligible", False),
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible matrix CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unqualified-source", action="store_true")
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--xtbloom-library", type=Path)
    parser.add_argument("--deepmd-model", type=Path, action="append", default=[])
    parser.add_argument(
        "--deepmd-artifact-manifest",
        type=Path,
        help=(
            "declared source/header/library cohort for the loaded DeePMD C API"
        ),
    )
    parser.add_argument(
        "--deepmd-source",
        type=Path,
        help="DeePMD-kit checkout used to verify the declared artifact cohort",
    )
    parser.add_argument(
        "--deepmd-include-dir",
        type=Path,
        help="include directory containing the loaded DeePMD C API header",
    )
    parser.add_argument(
        "--expected-deepmd-revision",
        help="reviewed DeePMD C API source revision required for evidence",
    )
    parser.add_argument(
        "--model-deviation-frequency",
        type=int,
        default=0,
        metavar="STEPS",
        help=(
            "must remain zero; the in-plugin DeePMD C API path currently "
            "supports one primary DPA4c model without model deviation"
        ),
    )
    parser.add_argument(
        "--dpa4c-models-qualified",
        action="store_true",
        help="assert that every supplied model passed the DPRc scientific gates",
    )
    parser.add_argument(
        "--allow-unqualified-dpa4c-models",
        action="store_true",
        help=(
            "permit random, pretrained, or otherwise unqualified DPA4c artifacts "
            "for diagnostic timing only; every resulting row is evidence-ineligible"
        ),
    )
    parser.add_argument(
        "--correctness-evidence",
        action="append",
        default=[],
        metavar="MODE=PATH",
    )
    parser.add_argument(
        "--mpiexec",
        type=Path,
        default=Path(WORKLOAD.shutil.which("mpiexec") or "mpiexec"),
    )
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--lammps-execution-backend",
        choices=("host", "kokkos"),
        default="host",
        help=(
            "run ordinary batched LAMMPS work on the host (default) or "
            "explicitly through Kokkos; broker-owned GPU work is unchanged"
        ),
    )
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument(
        "--classical-backend",
        choices=CLASSICAL_BACKENDS,
        default="batched-dprc",
        help=(
            "use the shared batched DPRc CUDA classical path (default) or the "
            "upstream GPU-pair plus CPU-PPPM reference"
        ),
    )
    parser.add_argument("--batch-size", type=int, action="append")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="write the full availability matrix without launching LAMMPS",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="return nonzero when any requested coordinate is unavailable",
    )
    return parser


def validate_dpa4c_policy(arguments: argparse.Namespace) -> None:
    """Reject contradictory model scheduling and scientific qualification claims."""
    if arguments.model_deviation_frequency < 0:
        raise ValueError("--model-deviation-frequency must be nonnegative")
    if arguments.model_deviation_frequency != 0:
        raise ValueError(
            "--model-deviation-frequency must be zero for the in-plugin "
            "DeePMD C API path"
        )
    if (
        arguments.dpa4c_models_qualified
        and arguments.allow_unqualified_dpa4c_models
    ):
        raise ValueError(
            "--dpa4c-models-qualified and "
            "--allow-unqualified-dpa4c-models are mutually exclusive"
        )


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        matrix = load_matrix(arguments.matrix)
        manifest = WORKLOAD.load_manifest(arguments.manifest)
        if arguments.warmup_steps is not None and arguments.warmup_steps < 1:
            raise ValueError("--warmup-steps must be positive")
        if arguments.sample_steps is not None and arguments.sample_steps < 1:
            raise ValueError("--sample-steps must be positive")
        if arguments.repetitions is not None and arguments.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        validate_dpa4c_policy(arguments)
        selected_modes = set(arguments.mode or matrix["axes"]["mode"])
        selected_batches = set(arguments.batch_size or matrix["axes"]["batch_size"])
        unknown_batches = selected_batches.difference(matrix["axes"]["batch_size"])
        if unknown_batches:
            raise ValueError(f"batch sizes are outside the matrix: {unknown_batches}")
        for path in arguments.library_dir:
            if not path.is_dir():
                raise ValueError(f"runtime library directory is unavailable: {path}")
        correctness_by_mode = parse_mode_artifacts(arguments.correctness_evidence)

        source = WORKLOAD.verify_source(
            arguments.tutorial,
            manifest,
            allow_unqualified_source=arguments.allow_unqualified_source,
        )
        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        with WORKLOAD.WorkspaceLock(
            output, recover_stale=arguments.recover_stale_lock
        ):
            windows = WORKLOAD.prepare_workspace(
                output,
                arguments.tutorial,
                arguments.manifest,
                manifest,
                source,
            )
            project = WORKLOAD.project_record(arguments.manifest, output)
            # The workload renderer and the benchmark orchestrator are both
            # executable parts of the measurement protocol.  Hash them
            # separately so a result cannot be attributed only to the former.
            project["benchmark_runner"] = artifact(Path(__file__))
            environment = environment_record(arguments)
            WORKLOAD.write_json_atomic(output / "environment.json", environment)

            rows: list[dict[str, Any]] = []
            unavailable_requested = False
            for mode in matrix["axes"]["mode"]:
                for batch_size in matrix["axes"]["batch_size"]:
                    if mode not in selected_modes or batch_size not in selected_batches:
                        rows.append(
                            {
                                "schema_version": 1,
                                "mode": mode,
                                "batch_size": batch_size,
                                "lammps_execution_backend": (
                                    coordinate_execution_backend(mode, arguments)
                                ),
                                "status": "unavailable",
                                "reason": "coordinate was not selected for this invocation",
                            }
                        )
                        continue
                    reasons = availability_reasons(mode, batch_size, arguments)
                    if arguments.inventory_only:
                        reasons.append("inventory-only invocation did not execute LAMMPS")
                    if reasons:
                        unavailable_requested = True
                        row = {
                            "schema_version": 1,
                            "mode": mode,
                            "batch_size": batch_size,
                            "lammps_execution_backend": (
                                coordinate_execution_backend(mode, arguments)
                            ),
                            "status": "unavailable",
                            "reasons": reasons,
                        }
                    else:
                        row = run_coordinate(
                            mode=mode,
                            batch_size=batch_size,
                            arguments=arguments,
                            matrix=matrix,
                            manifest=manifest,
                            windows=windows,
                            output=output,
                            project=project,
                            source=source,
                            correctness_by_mode=correctness_by_mode,
                        )
                    rows.append(row)
                    WORKLOAD.write_json_atomic(
                        output / "summary.json",
                        {
                            "schema_version": 1,
                            "claim": matrix["claim"],
                            "matrix": artifact(arguments.matrix),
                            "manifest": artifact(arguments.manifest),
                            "requested_lammps_execution_backend": (
                                arguments.lammps_execution_backend
                            ),
                            "source": source,
                            "project": project,
                            "environment": artifact(output / "environment.json"),
                            "rows": rows,
                        },
                    )

            summary_path = output / "summary.json"
            write_samples_csv(output / "samples.csv", rows)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["samples_csv"] = artifact(output / "samples.csv")
            summary["coordinate_counts"] = {
                status: sum(row.get("status") == status for row in rows)
                for status in ("passed", "failed", "unavailable")
            }
            WORKLOAD.write_json_atomic(summary_path, summary)
            print(json.dumps(summary["coordinate_counts"], sort_keys=True))
            if arguments.fail_on_unavailable and unavailable_requested:
                return 2
            if any(row.get("status") == "failed" for row in rows):
                return 1
        return 0
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"ETP/ETH benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
