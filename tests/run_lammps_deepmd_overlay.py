#!/usr/bin/env python3
"""Prove that compact DeePMD adds exactly one numerical DPRc contribution.

The test runs the same coordinates three ways: batched xTB QM/MM alone,
compact DeePMD alone, and their LAMMPS ``hybrid/overlay`` composition.  The
overlay must equal the numerical sum of the two independent calculations.
This catches style-name collisions and any net energy/force double counting.
The source-level fact that the QM/MM captures call only ``pair_long`` rather
than the complete hybrid pair object is reviewed separately; additivity alone
does not count backend invocations that might cancel numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ENERGY_TOLERANCE = 1.0e-5  # kcal/mol
FORCE_TOLERANCE = 1.0e-4  # kcal/mol/Angstrom
ATOM_COUNT = 5


def sha256(path: Path) -> str:
    """Return a stable identity for every executable runtime input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_lammps_token(path: Path) -> str:
    """Reject whitespace because LAMMPS variable quoting is not used here."""
    resolved = str(path.resolve())
    if any(character.isspace() for character in resolved):
        raise ValueError(f"LAMMPS runtime path contains whitespace: {resolved}")
    return resolved


def system_commands() -> list[str]:
    """Create one fixed QM center, selected MM atoms, and one exclusion."""
    return [
        "units real",
        "atom_style charge",
        "atom_modify map array",
        "boundary p p p",
        "region cell block 0 20 0 20 0 20 units box",
        "create_box 2 cell",
        "create_atoms 1 single 10.00 10.0 10.0 units box",
        "create_atoms 2 single 10.80 10.0 10.0 units box",
        "create_atoms 2 single 9.70 10.75 10.0 units box",
        "create_atoms 1 single 12.10 10.0 10.0 units box",
        "create_atoms 2 single 17.00 10.0 10.0 units box",
        "set atom 1 charge 0.0",
        "set atom 2 charge 0.0",
        # Zero MM charges isolate the pair-style composition claim from a
        # deliberately artificial close point charge destabilizing SCC.
        "set atom 3 charge 0.0",
        "set atom 4 charge 0.0",
        "set atom 5 charge 1.0",
        "mass 1 15.999",
        "mass 2 1.008",
        "group qm id 1 2 3",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ]


def render_input(
    mode: str,
    dprc_plugin: Path,
    deepmd_plugin: Path,
    model: Path,
    result: Path,
) -> str:
    """Render one of the three independent additivity calculations."""
    dprc = require_lammps_token(dprc_plugin)
    deepmd = require_lammps_token(deepmd_plugin)
    model_path = require_lammps_token(model)
    result_path = require_lammps_token(result)

    commands: list[str] = []
    if mode in {"deepmd", "overlay"}:
        commands.append(f"plugin load {deepmd}")
    if mode in {"xtb", "overlay"}:
        commands.append(f"plugin load {dprc}")
    commands.extend(system_commands())

    compact = (
        f"deepmd {model_path} center_group qm environment_cutoff 1.5 "
        "include_molecule no"
    )
    if mode == "deepmd":
        commands.extend([f"pair_style {compact}", "pair_coeff * *"])
    elif mode == "xtb":
        commands.extend(["pair_style coul/long 8.0", "pair_coeff * *"])
    elif mode == "overlay":
        commands.extend(
            [
                f"pair_style hybrid/overlay coul/long 8.0 {compact}",
                "pair_coeff * * coul/long",
                "pair_coeff * * deepmd",
            ]
        )
    else:
        raise ValueError(f"unknown calculation mode: {mode}")

    if mode in {"xtb", "overlay"}:
        commands.extend(
            [
                "kspace_style pppm/dprc 1.0e-5",
                "fix qmmm qm qmmm/xtb/dprc elements O H cutoff 8.0 method gfn2 "
                "charge 0 uhf 0 accuracy 1.0e-3 maxiter 250 etemp 300.0 "
                "mmhardness 0.0",
                "fix_modify qmmm energy yes",
            ]
        )

    force_variables: list[str] = []
    force_fields: list[str] = []
    for atom_id in range(1, ATOM_COUNT + 1):
        for dimension in "xyz":
            name = f"f{dimension}{atom_id}"
            force_variables.append(f"variable {name} equal f{dimension}[{atom_id}]")
            force_fields.append(f"{name}=$(v_{name}:%.16e)")
    commands.extend(force_variables)
    commands.extend(
        [
            "thermo 1",
            "thermo_style custom step pe",
            "run 0",
            f'print "energy=$(pe:%.16e) {" ".join(force_fields)}" '
            f"file {result_path} screen no",
        ]
    )
    return "\n".join(commands) + "\n"


def parse_result(path: Path) -> dict[str, float]:
    """Parse the compact key=value record emitted by a LAMMPS calculation."""
    values: dict[str, float] = {}
    for token in path.read_text(encoding="utf-8").split():
        key, separator, raw_value = token.partition("=")
        if not separator:
            raise ValueError(f"invalid result token {token!r}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite result {key}={value}")
        values[key] = value
    expected = {"energy"}
    expected.update(
        f"f{dimension}{atom_id}"
        for atom_id in range(1, ATOM_COUNT + 1)
        for dimension in "xyz"
    )
    if values.keys() != expected:
        raise ValueError(f"unexpected result fields {sorted(values)}")
    return values


def run_mode(
    executable: Path,
    mode: str,
    dprc_plugin: Path,
    deepmd_plugin: Path,
    model: Path,
    directory: Path,
) -> tuple[dict[str, float], str]:
    """Run one mode and preserve stdout for actionable failure diagnostics."""
    input_file = directory / f"in.{mode}"
    result_file = directory / f"{mode}.txt"
    input_file.write_text(
        render_input(mode, dprc_plugin, deepmd_plugin, model, result_file),
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            str(executable),
            "-log",
            "none",
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
            f"LAMMPS {mode} calculation failed with exit {process.returncode}:\n"
            f"{process.stdout}"
        )
    return parse_result(result_file), process.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--dprc-plugin", type=Path, required=True)
    parser.add_argument("--deepmd-plugin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deepmd-revision", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        for path in (
            arguments.lammps,
            arguments.dprc_plugin,
            arguments.deepmd_plugin,
            arguments.model,
        ):
            if not path.is_file():
                raise ValueError(f"required runtime input is not a file: {path}")

        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-overlay-") as temporary:
            directory = Path(temporary)
            xtb, _ = run_mode(
                arguments.lammps,
                "xtb",
                arguments.dprc_plugin,
                arguments.deepmd_plugin,
                arguments.model,
                directory,
            )
            deepmd, _ = run_mode(
                arguments.lammps,
                "deepmd",
                arguments.dprc_plugin,
                arguments.deepmd_plugin,
                arguments.model,
                directory,
            )
            overlay, _ = run_mode(
                arguments.lammps,
                "overlay",
                arguments.dprc_plugin,
                arguments.deepmd_plugin,
                arguments.model,
                directory,
            )

        residuals = {
            key: abs(overlay[key] - xtb[key] - deepmd[key]) for key in overlay
        }
        failures = {
            key: value
            for key, value in residuals.items()
            if value
            > (ENERGY_TOLERANCE if key == "energy" else FORCE_TOLERANCE)
        }
        excluded_force = max(
            abs(deepmd[f"f{dimension}5"]) for dimension in "xyz"
        )
        if excluded_force > FORCE_TOLERANCE:
            failures["excluded_atom_force"] = excluded_force

        evidence = {
            "schema_version": 1,
            "claim": "hybrid/overlay adds one net compact DeePMD contribution beside qmmm/xtb/dprc",
            "deepmd_revision": arguments.deepmd_revision,
            "inputs": {
                "lammps": {
                    "path": str(arguments.lammps.resolve()),
                    "sha256": sha256(arguments.lammps),
                },
                "dprc_plugin": {
                    "path": str(arguments.dprc_plugin.resolve()),
                    "sha256": sha256(arguments.dprc_plugin),
                },
                "deepmd_plugin": {
                    "path": str(arguments.deepmd_plugin.resolve()),
                    "sha256": sha256(arguments.deepmd_plugin),
                },
                "model": {
                    "path": str(arguments.model.resolve()),
                    "sha256": sha256(arguments.model),
                },
            },
            "units": {
                "energy": "kcal/mol",
                "force": "kcal/mol/Angstrom",
            },
            "results": {"xtb": xtb, "deepmd": deepmd, "overlay": overlay},
            "additivity_absolute_residuals": residuals,
            "absolute_tolerances": {
                "energy": ENERGY_TOLERANCE,
                "force": FORCE_TOLERANCE,
            },
            "excluded_atom_deepmd_force_max_abs": excluded_force,
        }
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "DeepMD overlay additivity residuals: "
            f"energy={residuals['energy']:.6e} "
            f"force_max={max(value for key, value in residuals.items() if key != 'energy'):.6e} "
            f"excluded_force={excluded_force:.6e}"
        )
        if failures:
            print(f"DeepMD overlay check failed: {failures}", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DeepMD overlay check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
