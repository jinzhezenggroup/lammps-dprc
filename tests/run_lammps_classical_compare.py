#!/usr/bin/env python3
"""Compare the pure-classical batched CUDA styles with pinned LAMMPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the immutable identity recorded for every executable artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tolerances(natoms: int) -> dict[str, float]:
    """Declare LAMMPS-real-unit gates before examining the comparison."""
    result = {
        "energy": 2.0e-5,
        "evdwl": 2.0e-5,
        "ecoul": 2.0e-5,
        "elong": 2.0e-5,
    }
    for atom_id in range(1, natoms + 1):
        for component in "xyz":
            result[f"f{component}{atom_id}"] = 1.0e-4
    for component in ("xx", "yy", "zz", "xy", "xz", "yz"):
        result[f"p{component}"] = 1.0e-4
    return result


def parse_result(path: Path, gates: dict[str, float]) -> dict[str, float]:
    """Reject missing, extra, and non-finite publications transactionally."""
    values: dict[str, float] = {}
    for token in path.read_text(encoding="utf-8").split():
        name, separator, raw = token.partition("=")
        if not separator:
            raise ValueError(f"invalid result token {token!r}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite result {name}={value}")
        values[name] = value
    if values.keys() != gates.keys():
        raise ValueError(f"unexpected result fields {sorted(values)}")
    return values


def run_style(
    executable: Path,
    plugin: Path,
    input_file: Path,
    *,
    kspace: str,
    lj: str,
    coulomb: str,
    result: Path,
    gates: dict[str, float],
    batched: bool,
) -> dict[str, float]:
    """Run one complete classical Hamiltonian without resource subtraction."""
    log = result.with_suffix(".log")
    screen = result.with_suffix(".screen")
    process = subprocess.run(
        [
            str(executable),
            "-log",
            str(log),
            "-screen",
            str(screen),
            "-var",
            "dprc_plugin",
            str(plugin),
            "-var",
            "kspace_style",
            kspace,
            "-var",
            "lj_style",
            lj,
            "-var",
            "coulomb_style",
            coulomb,
            "-var",
            "result_file",
            str(result),
            "-var",
            "use_batched_classical",
            "1" if batched else "0",
            "-in",
            str(input_file),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0:
        diagnostic = process.stdout
        for path in (screen, log):
            if path.is_file():
                diagnostic += f"\n--- {path.name} ---\n{path.read_text(encoding='utf-8')}"
        raise RuntimeError(
            f"LAMMPS classical styles {lj} + {coulomb} + {kspace} failed "
            f"with exit {process.returncode}:\n{diagnostic}"
        )
    return parse_result(result, gates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    gates = tolerances(6)

    try:
        with tempfile.TemporaryDirectory(prefix="dprc-classical-") as temporary:
            directory = Path(temporary)
            reference = run_style(
                arguments.lammps,
                arguments.plugin,
                arguments.input,
                kspace="pppm/tip4p",
                lj="lj/cut",
                coulomb="tip4p/long",
                result=directory / "reference.txt",
                gates=gates,
                batched=False,
            )
            actual = run_style(
                arguments.lammps,
                arguments.plugin,
                arguments.input,
                kspace="pppm/tip4p/dprc/batch",
                lj="lj/cut/dprc/batch",
                coulomb="tip4p/long/dprc/batch",
                result=directory / "actual.txt",
                gates=gates,
                batched=True,
            )

        errors = {name: abs(actual[name] - reference[name]) for name in gates}
        failures = {
            name: error for name, error in errors.items() if error > gates[name]
        }
        evidence = {
            "schema_version": 1,
            "oracle": "pinned upstream LAMMPS lj/cut + tip4p/long + pppm/tip4p",
            "lammps_executable": str(arguments.lammps.resolve()),
            "lammps_sha256": sha256(arguments.lammps),
            "plugin": str(arguments.plugin.resolve()),
            "plugin_sha256": sha256(arguments.plugin),
            "input": str(arguments.input.resolve()),
            "reference": reference,
            "batched_cuda": actual,
            "absolute_errors": errors,
            "absolute_tolerances": gates,
            "units": {
                **{name: "kcal/mol" for name in ("energy", "evdwl", "ecoul", "elong")},
                **{
                    f"f{component}{atom_id}": "kcal/mol/Angstrom"
                    for atom_id in range(1, 7)
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
            "Classical LAMMPS comparison absolute errors: "
            + " ".join(f"{name}={errors[name]:.6e}" for name in gates)
        )
        if failures:
            print(
                f"Classical comparison exceeded tolerances: {failures}",
                file=sys.stderr,
            )
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Classical comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
