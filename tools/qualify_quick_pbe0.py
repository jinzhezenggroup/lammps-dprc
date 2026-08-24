#!/usr/bin/env python3
"""Qualify pinned QUICK PBE0 CUDA QM/MM evidence without trusting exit code alone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config/quick_pbe0_engine.json"
HARTREE_TO_KCAL_PER_MOL = 627.50946943


def sha256(path: Path) -> str:
    """Return a streaming digest for binaries and the large source archive."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> str:
    """Reject missing or substituted evidence before interpreting its contents."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 {actual} differs from expected {expected}")
    return actual


def _single(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{label} must appear exactly once, found {len(matches)}")
    value = matches[0]
    return value if isinstance(value, str) else value[0]


def parse_quick_trace(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract method identity and convergence from one preserved QUICK call."""
    text = path.read_text(encoding="utf-8", errors="strict")
    if "Error Termination" in text:
        raise ValueError(f"QUICK trace terminated with an error: {path}")

    method = manifest["method"]
    basis = manifest["basis"]
    scf = manifest["scf"]
    keyword = _single(r"^\s*KEYWORD=(.+)$", text, "QUICK job keyword")
    required_tokens = (
        method["quick_keyword"],
        f"BASIS={basis['name']}",
        f"SCF={scf['maximum_cycles']}",
        "DENSERMS=   0.0000000100",
        "CHARGE=0",
        "MULT=1",
        "GRADIENT",
        "EXTCHARGES",
    )
    if any(token not in keyword for token in required_tokens):
        raise ValueError(f"QUICK keyword does not match the pinned job: {keyword}")

    libxc_version = _single(
        r"^\s*USING LIBXC VERSION:\s*(\S+)\s*$", text, "LibXC version"
    )
    functional = _single(
        r"^\s*NAME = (.+?)\s*$", text, "LibXC functional definition"
    )
    expected_functional = (
        f"{method['libxc_name']} FAMILY = {method['libxc_family']} "
        f"KIND = {method['libxc_kind']}"
    )
    if libxc_version != method["libxc_version"]:
        raise ValueError(f"LibXC version {libxc_version} is not pinned")
    if functional != expected_functional:
        raise ValueError(f"LibXC functional is not the pinned PBE0 definition: {functional}")

    cycles = int(
        _single(
            r"^\| REACH CONVERGENCE AFTER\s+(\d+) CYCLES\s*$",
            text,
            "SCF convergence",
        )
    )
    if cycles > int(scf["maximum_cycles"]):
        raise ValueError("SCF convergence exceeds the configured cycle limit")
    error_match = re.search(
        r"^\| MAX ERROR =\s*(\S+)\s+RMS CHANGE =\s*(\S+)\s+MAX CHANGE =\s*(\S+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if error_match is None:
        raise ValueError("SCF convergence metrics are missing")
    total_energy = float(
        _single(
            r"^ TOTAL ENERGY\s*=\s*([-+0-9.Ee]+)\s*$",
            text,
            "QUICK total energy",
        )
    )
    scf_seconds = float(
        _single(
            r"^\| TOTAL SCF TIME\s*=\s*([-+0-9.Ee]+)",
            text,
            "QUICK total SCF time",
        )
    )
    numeric = [total_energy, scf_seconds, *(float(value) for value in error_match.groups())]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("QUICK trace contains non-finite convergence data")
    return {
        "path": f"{path.parent.name}/{path.name}",
        "sha256": sha256(path),
        "cycles": cycles,
        "maximum_error": float(error_match.group(1)),
        "rms_change": float(error_match.group(2)),
        "maximum_change": float(error_match.group(3)),
        "energy_hartree": total_energy,
        "scf_seconds": scf_seconds,
    }


def _gradient_vectors(block: str, label: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for line in block.strip().splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{label} gradient line is incomplete: {line!r}")
        vector = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{label} gradient contains a non-finite value")
        vectors.append(vector)
    return vectors


def parse_sander_mdout(
    path: Path, expected_calls: int, expected_qm: int, expected_mm: int
) -> list[dict[str, Any]]:
    """Require complete finite energy, QM/link, and MM point-charge slices."""
    text = path.read_text(encoding="utf-8", errors="strict")
    if "SANDER BOMB" in text:
        raise ValueError("Sander reported a fatal QM/MM error")
    if text.count("QUICK execution success; Processing QUICK results...") != expected_calls:
        raise ValueError("Sander did not report the expected number of QUICK successes")

    starts = [
        match.start()
        for match in re.finditer(
            r"^qm2_extern_quick_module - final energy:\s*$", text, re.MULTILINE
        )
    ]
    if len(starts) != expected_calls:
        raise ValueError(f"expected {expected_calls} Sander result blocks, found {len(starts)}")

    calls: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = text.find("<<<<< Left print_results", start)
        if end < 0:
            raise ValueError("Sander result block is missing its commit terminator")
        block = text[start:end]
        energy_match = re.search(
            r"final energy:\s*\n\s*([-+0-9.Ee]+)\s*\n", block
        )
        gradient_match = re.search(
            r"QM region:\s*\n(?P<qm>.*?)^MM region:\s*\n(?P<mm>.*)",
            block,
            flags=re.MULTILINE | re.DOTALL,
        )
        if energy_match is None or gradient_match is None:
            raise ValueError("Sander result block lacks energy or gradient sections")
        energy = float(energy_match.group(1))
        qm_vectors = _gradient_vectors(gradient_match.group("qm"), "QM/link")
        mm_vectors = _gradient_vectors(gradient_match.group("mm"), "MM point-charge")
        if len(qm_vectors) != expected_qm or len(mm_vectors) != expected_mm:
            raise ValueError(
                "Sander gradient extent mismatch: "
                f"QM/link={len(qm_vectors)} MM={len(mm_vectors)}"
            )
        if not math.isfinite(energy):
            raise ValueError("Sander published a non-finite energy")
        calls.append(
            {
                "call_index": index + 1,
                "energy_kcal_per_mol": energy,
                "qm_link_gradient_vectors": len(qm_vectors),
                "mm_point_charge_gradient_vectors": len(mm_vectors),
            }
        )
    return calls


def parse_telemetry(path: Path) -> dict[str, Any]:
    """Reduce raw nvidia-smi samples to compact, reviewable evidence."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        # nvidia-smi emits a space after each comma; normalize those field names
        # rather than binding the evidence parser to incidental CSV padding.
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if not rows:
        raise ValueError("GPU telemetry has no samples")

    def number(value: str) -> float:
        match = re.search(r"[-+0-9.]+", value)
        if match is None:
            raise ValueError(f"telemetry value is not numeric: {value!r}")
        return float(match.group(0))

    utilization = [number(row["utilization.gpu [%]"]) for row in rows]
    power = [number(row["power.draw [W]"]) for row in rows]
    memory = [number(row["memory.used [MiB]"]) for row in rows]
    return {
        "samples": len(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "maximum_gpu_utilization_percent": max(utilization),
        "maximum_power_watts": max(power),
        "maximum_memory_mib": max(memory),
    }


def parse_timing(path: Path) -> dict[str, Any]:
    """Require successful process termination and retain resource bounds."""
    text = path.read_text(encoding="utf-8", errors="strict")
    exit_status = int(_single(r"^\s*Exit status:\s*(\d+)\s*$", text, "exit status"))
    if exit_status != 0:
        raise ValueError(f"Sander process exited with status {exit_status}")
    return {
        "exit_status": exit_status,
        "wall_clock": _single(
            r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)\s*$",
            text,
            "wall-clock time",
        ),
        "maximum_resident_kbytes": int(
            _single(
                r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$",
                text,
                "maximum resident set size",
            )
        ),
    }


def qualify(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify bytes first, then numerical publication and runtime provenance."""
    manifest_path = arguments.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("QUICK engine manifest schema_version must be 1")
    run_directory = arguments.run_directory.resolve()

    for relative, expected in manifest["qualification"]["raw_artifacts"].items():
        require_digest(run_directory / relative, expected, f"raw artifact {relative}")
    artifacts = manifest["artifacts"]
    runtime_paths = {
        "sander.quick.cuda": arguments.sander_executable.resolve(),
        "quick.cuda": arguments.quick_executable.resolve(),
        "libquick_cuda.so": arguments.quick_library.resolve(),
    }
    for name, path in runtime_paths.items():
        require_digest(path, artifacts[name], name)

    source = manifest["source"]
    require_digest(
        arguments.source_archive.resolve(),
        source["archive"]["sha256"],
        "AmberTools source archive",
    )
    require_digest(
        arguments.update_patch.resolve(),
        source["official_update"]["sha256"],
        "AmberTools official update",
    )
    require_digest(
        arguments.ambertools_license.resolve(),
        source["license_files"]["ambertools"]["sha256"],
        "AmberTools license file",
    )
    require_digest(
        arguments.quick_license.resolve(),
        source["license_files"]["quick"]["sha256"],
        "QUICK license file",
    )
    require_digest(
        arguments.cuda_config.resolve(),
        source["cuda_compatibility_patch"]["patched_cuda_config_sha256"],
        "patched AmberTools CUDA configuration",
    )
    require_digest(
        ROOT / source["cuda_compatibility_patch"]["path"],
        source["cuda_compatibility_patch"]["sha256"],
        "retained CUDA compatibility patch",
    )
    require_digest(
        arguments.cmake_cache.resolve(),
        manifest["build"]["cmake_cache_sha256"],
        "AmberTools CMake cache",
    )
    basis_directory = arguments.basis_directory.resolve()
    require_digest(
        basis_directory / manifest["basis"]["quick_definition"],
        manifest["basis"]["definition_sha256"],
        "QUICK basis definition",
    )
    require_digest(
        basis_directory / "basis_link",
        manifest["basis"]["basis_link_sha256"],
        "QUICK basis index",
    )

    environment = (run_directory / "environment.txt").read_text(encoding="utf-8")
    required_environment = (
        "LD_PRELOAD=<unset>",
        "NVIDIA GeForce RTX 5090",
        "580.95.05",
        "release 12.9, V12.9.86",
        "libstdc++.so.6",
        "(RUNPATH)",
    )
    if any(value not in environment for value in required_environment):
        raise ValueError("runtime environment does not satisfy the pinned loader/GPU contract")
    if "(RPATH)" in environment:
        raise ValueError("old-style DT_RPATH remains in the qualified runtime")

    expected_calls = int(manifest["qualification"]["expected_force_calls"])
    traces = [
        parse_quick_trace(run_directory / relative, manifest)
        for relative in sorted(
            name
            for name in manifest["qualification"]["raw_artifacts"]
            if name.startswith("quick-traces/") and name.endswith(".out")
        )
    ]
    if len(traces) != expected_calls:
        raise ValueError("preserved QUICK trace count does not match expected force calls")
    calls = parse_sander_mdout(
        run_directory / "aladip.pbe0-6-31g-star.cuda.mdout",
        expected_calls,
        int(manifest["qualification"]["expected_qm_gradient_vectors_per_call"]),
        int(manifest["qualification"]["expected_mm_gradient_vectors_per_call"]),
    )
    for call, trace in zip(calls, traces, strict=True):
        expected_kcal = trace["energy_hartree"] * HARTREE_TO_KCAL_PER_MOL
        if abs(call["energy_kcal_per_mol"] - expected_kcal) > 5.0e-5:
            raise ValueError("Sander and preserved QUICK energies disagree")

    return {
        "schema_version": 1,
        "qualified": True,
        "qualification_scope": manifest["qualification_state"],
        "engine_manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(manifest_path),
        },
        "source": {
            "archive_sha256": source["archive"]["sha256"],
            "official_update_sha256": source["official_update"]["sha256"],
            "ambertools_license_sha256": source["license_files"]["ambertools"][
                "sha256"
            ],
            "quick_license_sha256": source["license_files"]["quick"]["sha256"],
            "cuda_compatibility_patch_sha256": source["cuda_compatibility_patch"][
                "sha256"
            ],
        },
        "runtime_artifacts": artifacts,
        "method": manifest["method"],
        "basis": manifest["basis"],
        "scf": manifest["scf"],
        "calls": [
            {**call, "quick": trace} for call, trace in zip(calls, traces, strict=True)
        ],
        "failed_frame_ledger": [],
        "timing": parse_timing(run_directory / "time.txt"),
        "gpu_telemetry": parse_telemetry(run_directory / "gpu-telemetry.csv"),
        "limitations": manifest["qualification"]["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--sander-executable", type=Path, required=True)
    parser.add_argument("--quick-executable", type=Path, required=True)
    parser.add_argument("--quick-library", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--update-patch", type=Path, required=True)
    parser.add_argument("--ambertools-license", type=Path, required=True)
    parser.add_argument("--quick-license", type=Path, required=True)
    parser.add_argument("--cuda-config", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path, required=True)
    parser.add_argument("--basis-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = qualify(arguments)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"qualified QUICK PBE0 evidence: {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
