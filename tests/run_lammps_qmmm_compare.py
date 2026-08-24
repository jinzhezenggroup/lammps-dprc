#!/usr/bin/env python3
"""Compare qmmm/xtb/dprc with the pinned LAMMPS libxTB reference fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def build_tolerances(natoms: int) -> dict[str, float]:
    """Return field-wise absolute tolerances in LAMMPS real units."""
    tolerances = {
        "energy": 1.0e-5,  # kcal/mol
        "correction": 1.0e-5,  # kcal/mol
        "q1": 1.0e-7,  # electron
        "q2": 1.0e-7,  # electron
    }
    for atom_id in range(1, natoms + 1):
        for component in "xyz":
            tolerances[f"f{component}{atom_id}"] = 1.0e-4  # kcal/mol/Angstrom
    for component in ("xx", "yy", "zz", "xy", "xz", "yz"):
        tolerances[f"p{component}"] = 1.0e-4  # atm
    return tolerances


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result(path: Path, tolerances: dict[str, float]) -> dict[str, float]:
    tokens = path.read_text(encoding="utf-8").split()
    values: dict[str, float] = {}
    for token in tokens:
        key, separator, raw_value = token.partition("=")
        if not separator:
            raise ValueError(f"invalid result token {token!r}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite result {key}={value}")
        values[key] = value
    if values.keys() != tolerances.keys():
        raise ValueError(f"unexpected result fields {sorted(values)}")
    return values


def run_style(
    executable: Path,
    plugin: Path,
    input_file: Path,
    qmmm_style: str,
    kspace_style: str,
    lj_style: str,
    coulomb_style: str,
    result: Path,
    tolerances: dict[str, float],
    launcher: list[str],
    lammps_arguments: list[str],
    hybrid_style: str,
    atom_style: str,
    data_file: Path,
    run_style_name: str,
    bond_style: str,
    angle_style: str,
) -> tuple[dict[str, float], str]:
    """Run one reference or DPRc style and retain its diagnostic screen."""
    screen = result.with_suffix(result.suffix + ".screen")
    process = subprocess.run(
        [
            *launcher,
            str(executable),
            *lammps_arguments,
            "-log",
            "none",
            "-screen",
            str(screen),
            "-var",
            "dprc_plugin",
            str(plugin),
            "-var",
            "qmmm_style",
            qmmm_style,
            "-var",
            "kspace_style",
            kspace_style,
            "-var",
            "lj_style",
            lj_style,
            "-var",
            "coulomb_style",
            coulomb_style,
            "-var",
            "hybrid_style",
            hybrid_style,
            "-var",
            "atom_style",
            atom_style,
            "-var",
            "data_file",
            str(data_file),
            "-var",
            "run_style_name",
            run_style_name,
            "-var",
            "bond_style",
            bond_style,
            "-var",
            "angle_style",
            angle_style,
            "-var",
            "result_file",
            str(result),
            "-in",
            str(input_file),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    diagnostic = process.stdout
    if screen.is_file():
        diagnostic += screen.read_text(encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"LAMMPS styles {qmmm_style} + {kspace_style} failed with exit "
            f"{process.returncode}:\n{diagnostic}"
        )
    return parse_result(result, tolerances), diagnostic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument(
        "--reference-lammps",
        type=Path,
        help=(
            "optional QMMM-XTB-enabled executable for the pinned libxTB "
            "oracle; defaults to --lammps"
        ),
    )
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--data-file",
        type=Path,
        help="optional LAMMPS data fixture referenced by the selected input",
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--natoms", type=int, default=4)
    parser.add_argument("--reference-kspace", default="pppm/xtb")
    parser.add_argument("--actual-kspace", default="pppm/dprc")
    parser.add_argument("--reference-lj", default="lj/cut")
    parser.add_argument("--actual-lj", default="lj/cut")
    parser.add_argument("--reference-coulomb", default="coul/long")
    parser.add_argument("--actual-coulomb", default="coul/long")
    parser.add_argument("--reference-hybrid-style", default="hybrid/overlay")
    parser.add_argument("--actual-hybrid-style", default="hybrid/overlay")
    parser.add_argument(
        "--actual-kokkos",
        action="store_true",
        help=(
            "initialize one Kokkos GPU and use the one-owner half-neighbor "
            "configuration for the actual DPRc invocation"
        ),
    )
    parser.add_argument(
        "--expect-actual-mm-only-routing",
        action="store_true",
        help=(
            "require the fused qmmm/xtb base to recognize the batched TIP4P "
            "mapping and skip redundant MM/full pair reference evaluations"
        ),
    )
    parser.add_argument("--mpi-exec", type=Path)
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-procs", type=int, default=1)
    parser.add_argument("--expect-actual-batch-calls", type=int)
    parser.add_argument("--expect-actual-plan-rebuilds", type=int)
    parser.add_argument("--expect-actual-fresh-calls", type=int)
    parser.add_argument("--expect-actual-warm-calls", type=int)
    parser.add_argument("--expect-actual-capacity-growths", type=int)
    parser.add_argument("--require-actual-point-padding", action="store_true")
    arguments = parser.parse_args()
    tolerances = build_tolerances(arguments.natoms)
    if arguments.mpi_procs < 1:
        parser.error("--mpi-procs must be positive")
    if arguments.mpi_procs != 1 and arguments.mpi_exec is None:
        parser.error("--mpi-exec is required when --mpi-procs is not one")
    launcher = (
        [
            str(arguments.mpi_exec),
            arguments.mpi_numproc_flag,
            str(arguments.mpi_procs),
        ]
        if arguments.mpi_exec is not None
        else []
    )
    actual_lammps_arguments = (
        ["-k", "on", "g", "1", "-pk", "kokkos", "newton", "on", "neigh", "half"]
        if arguments.actual_kokkos
        else []
    )
    reference_lammps = arguments.reference_lammps or arguments.lammps
    data_file = arguments.data_file or arguments.input
    if not data_file.is_file():
        parser.error(f"--data-file does not exist: {data_file}")

    try:
        with tempfile.TemporaryDirectory(prefix="dprc-qmmm-") as temporary:
            temporary_path = Path(temporary)
            reference, _ = run_style(
                reference_lammps,
                arguments.plugin,
                arguments.input,
                "qmmm/xtb",
                arguments.reference_kspace,
                arguments.reference_lj,
                arguments.reference_coulomb,
                temporary_path / "reference.txt",
                tolerances,
                launcher,
                [],
                arguments.reference_hybrid_style,
                "full",
                data_file,
                "verlet",
                "harmonic",
                "harmonic",
            )
            actual, actual_output = run_style(
                arguments.lammps,
                arguments.plugin,
                arguments.input,
                "qmmm/xtb/dprc",
                arguments.actual_kspace,
                arguments.actual_lj,
                arguments.actual_coulomb,
                temporary_path / "dprc.txt",
                tolerances,
                launcher,
                actual_lammps_arguments,
                arguments.actual_hybrid_style,
                "full/kk" if arguments.actual_kokkos else "full",
                data_file,
                "verlet/kk" if arguments.actual_kokkos else "verlet",
                "harmonic/kk" if arguments.actual_kokkos else "harmonic",
                "harmonic/kk" if arguments.actual_kokkos else "harmonic",
            )

        mm_only_routing_message = (
            "Fix qmmm/xtb detected an MM-only "
            "tip4p/long/dprc/batch type-pair mapping; skipping the MM/full "
            "pair reference evaluations"
        )
        if (
            arguments.expect_actual_mm_only_routing
            and mm_only_routing_message not in actual_output
        ):
            raise RuntimeError(
                "actual LAMMPS run did not confirm MM-only PairHybrid routing"
            )

        profile_match = re.search(
            r"dprc xTB broker profile: batch_calls=(\d+) plan_rebuilds=(\d+) "
            r"fresh=(\d+) warm=(\d+) system_calls=(\d+) "
            r"mean_scc_iterations=([0-9.]+) max_scc_iterations=(\d+) "
            r"capacity_growths=(\d+) mean_actual_points=([0-9.]+) "
            r"mean_plan_points=([0-9.]+) max_actual_points=(\d+) "
            r"max_plan_points=(\d+)",
            actual_output,
        )
        profile = None
        expected_profile = {
            "batch_calls": arguments.expect_actual_batch_calls,
            "plan_rebuilds": arguments.expect_actual_plan_rebuilds,
            "fresh": arguments.expect_actual_fresh_calls,
            "warm": arguments.expect_actual_warm_calls,
            "capacity_growths": arguments.expect_actual_capacity_growths,
        }
        if (
            any(value is not None for value in expected_profile.values())
            or arguments.require_actual_point_padding
        ):
            if profile_match is None:
                raise RuntimeError(
                    "actual LAMMPS run did not publish the requested xTB broker profile"
                )
            profile = {
                "batch_calls": int(profile_match.group(1)),
                "plan_rebuilds": int(profile_match.group(2)),
                "fresh": int(profile_match.group(3)),
                "warm": int(profile_match.group(4)),
                "system_calls": int(profile_match.group(5)),
                "mean_scc_iterations": float(profile_match.group(6)),
                "max_scc_iterations": int(profile_match.group(7)),
                "capacity_growths": int(profile_match.group(8)),
                "mean_actual_points": float(profile_match.group(9)),
                "mean_plan_points": float(profile_match.group(10)),
                "max_actual_points": int(profile_match.group(11)),
                "max_plan_points": int(profile_match.group(12)),
            }
            mismatched_profile = {
                key: {"actual": profile[key], "expected": expected}
                for key, expected in expected_profile.items()
                if expected is not None and profile[key] != expected
            }
            if mismatched_profile:
                raise RuntimeError(
                    f"actual xTB broker profile mismatch: {mismatched_profile}"
                )
            if (
                arguments.require_actual_point_padding
                and profile["max_plan_points"] <= profile["max_actual_points"]
            ):
                raise RuntimeError(
                    "actual xTB broker profile did not exercise zero-charge padding: "
                    f"{profile}"
                )

        errors = {key: abs(actual[key] - reference[key]) for key in tolerances}
        failures = {
            key: error
            for key, error in errors.items()
            if error > tolerances[key]
        }
        evidence = {
            "schema_version": 1,
            "oracle": "pinned LAMMPS qmmm/xtb linked to libxTB",
            "reference_lammps_executable": str(reference_lammps.resolve()),
            "reference_lammps_sha256": sha256(reference_lammps),
            "actual_lammps_executable": str(arguments.lammps.resolve()),
            "actual_lammps_sha256": sha256(arguments.lammps),
            "plugin": str(arguments.plugin.resolve()),
            "plugin_sha256": sha256(arguments.plugin),
            "input": str(arguments.input.resolve()),
            "data_file": str(data_file.resolve()) if arguments.data_file else None,
            "mpi_launcher": launcher,
            "mpi_ranks": arguments.mpi_procs,
            "reference_kspace": arguments.reference_kspace,
            "actual_kspace": arguments.actual_kspace,
            "reference_lj": arguments.reference_lj,
            "actual_lj": arguments.actual_lj,
            "reference_coulomb": arguments.reference_coulomb,
            "actual_coulomb": arguments.actual_coulomb,
            "reference_hybrid_style": arguments.reference_hybrid_style,
            "actual_hybrid_style": arguments.actual_hybrid_style,
            "reference_atom_style": "full",
            "actual_atom_style": "full/kk" if arguments.actual_kokkos else "full",
            "reference_run_style": "verlet",
            "actual_run_style": "verlet/kk" if arguments.actual_kokkos else "verlet",
            "reference_bond_style": "harmonic",
            "actual_bond_style": (
                "harmonic/kk" if arguments.actual_kokkos else "harmonic"
            ),
            "reference_angle_style": "harmonic",
            "actual_angle_style": (
                "harmonic/kk" if arguments.actual_kokkos else "harmonic"
            ),
            "actual_lammps_arguments": actual_lammps_arguments,
            "actual_mm_only_routing_confirmed": (
                mm_only_routing_message in actual_output
            ),
            "actual_profile": profile,
            "reference": reference,
            "qmmm_xtb_dprc": actual,
            "absolute_errors": errors,
            "absolute_tolerances": tolerances,
            "units": {
                "energy": "kcal/mol",
                "correction": "kcal/mol",
                "q1": "electron",
                "q2": "electron",
                **{
                    f"f{component}{atom_id}": "kcal/mol/Angstrom"
                    for atom_id in range(1, arguments.natoms + 1)
                    for component in "xyz"
                },
                **{
                    f"p{component}": "atm"
                    for component in ("xx", "yy", "zz", "xy", "xz", "yz")
                },
            },
        }
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "QM/MM libxTB comparison absolute errors: "
            + " ".join(f"{key}={errors[key]:.6e}" for key in tolerances)
        )
        if failures:
            print(f"QM/MM comparison exceeded tolerances: {failures}", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"QM/MM comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
