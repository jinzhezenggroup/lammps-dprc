#!/usr/bin/env python3
"""Check the full qmmm/xtb/dprc force against central energy differences."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


STEP_SIZES_ANGSTROM = (5.0e-4, 1.0e-3)
FORCE_TOLERANCE = 1.5e-3  # kcal/mol/Angstrom


def run_position(
    executable: Path, plugin: Path, input_file: Path, x1: float, result: Path
) -> dict[str, float]:
    process = subprocess.run(
        [
            str(executable),
            "-log",
            "none",
            "-screen",
            "none",
            "-var",
            "dprc_plugin",
            str(plugin),
            "-var",
            "x1",
            f"{x1:.17g}",
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
    if process.returncode != 0:
        raise RuntimeError(
            f"LAMMPS finite-difference point {x1} failed with exit "
            f"{process.returncode}:\n{process.stdout}"
        )
    values: dict[str, float] = {}
    for token in result.read_text(encoding="utf-8").split():
        key, separator, raw_value = token.partition("=")
        if not separator:
            raise ValueError(f"invalid result token {token!r}")
        values[key] = float(raw_value)
    if values.keys() != {"energy", "fx1"} or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise ValueError(f"invalid finite-difference result {values}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()

    center_x = 9.60
    try:
        with tempfile.TemporaryDirectory(prefix="dprc-qmmm-fd-") as temporary:
            temporary_path = Path(temporary)
            center = run_position(
                arguments.lammps,
                arguments.plugin,
                arguments.input,
                center_x,
                temporary_path / "center.txt",
            )
            rows = []
            for index, step in enumerate(STEP_SIZES_ANGSTROM):
                minus = run_position(
                    arguments.lammps,
                    arguments.plugin,
                    arguments.input,
                    center_x - step,
                    temporary_path / f"minus-{index}.txt",
                )
                plus = run_position(
                    arguments.lammps,
                    arguments.plugin,
                    arguments.input,
                    center_x + step,
                    temporary_path / f"plus-{index}.txt",
                )
                finite_difference_force = -(
                    plus["energy"] - minus["energy"]
                ) / (2.0 * step)
                rows.append(
                    {
                        "step_angstrom": step,
                        "minus_energy_kcal_per_mol": minus["energy"],
                        "plus_energy_kcal_per_mol": plus["energy"],
                        "finite_difference_force_kcal_per_mol_angstrom": finite_difference_force,
                        "analytic_force_kcal_per_mol_angstrom": center["fx1"],
                        "absolute_error_kcal_per_mol_angstrom": abs(
                            finite_difference_force - center["fx1"]
                        ),
                    }
                )

        evidence = {
            "schema_version": 1,
            "coordinate": "QM atom 1 x",
            "center_angstrom": center_x,
            "force_tolerance_kcal_per_mol_angstrom": FORCE_TOLERANCE,
            "rows": rows,
        }
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "QM/MM finite-difference force errors: "
            + " ".join(
                f"h={row['step_angstrom']:.1e}:"
                f"{row['absolute_error_kcal_per_mol_angstrom']:.6e}"
                for row in rows
            )
        )
        if any(
            row["absolute_error_kcal_per_mol_angstrom"] > FORCE_TOLERANCE
            for row in rows
        ):
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"QM/MM finite-difference test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
