#!/usr/bin/env python3
"""Prove in-plugin DeePMD C API center-only energy semantics.

The compact subsystem deliberately retains nearby MM atoms as graph nodes.
Those atoms must receive the reaction force obtained by differentiating the
QM-centered correction, but their independent atomic-energy heads must remain
zero. This regression compares the host and ``/kk`` names provided by the same
LAMMPS-DPRc plugin and checks both properties through LAMMPS's public per-atom
energy and force outputs.
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


ATOM_COUNT = 8
CENTER_ATOMS = frozenset({1, 2, 3, 4})
ENVIRONMENT_ATOMS = frozenset({5, 6})
OUTSIDE_ATOMS = frozenset({7, 8})
ENERGY_TOLERANCE_EV = 2.0e-5
ATOMIC_ENERGY_TOLERANCE_EV = 2.0e-5
FORCE_TOLERANCE_EV_PER_ANGSTROM = 2.0e-4
MASK_ZERO_TOLERANCE_EV = 1.0e-12
OUTSIDE_FORCE_TOLERANCE_EV_PER_ANGSTROM = 1.0e-12
REACTION_FORCE_MIN_EV_PER_ANGSTROM = 1.0e-10


def sha256(path: Path) -> str:
    """Return a stable identity for one runtime artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_lammps_token(path: Path) -> str:
    """Reject paths that cannot be inserted as one unquoted LAMMPS token."""
    resolved = str(path.resolve())
    if any(character.isspace() for character in resolved):
        raise ValueError(f"LAMMPS runtime path contains whitespace: {resolved}")
    return resolved


def render_input(
    kokkos: bool,
    dprc_plugin: Path,
    model: Path,
    data_file: Path,
    energy_result: Path,
    atom_result: Path,
) -> str:
    """Render one host or Kokkos calculation over the same compact graph."""
    plugin = require_lammps_token(dprc_plugin)
    model_path = require_lammps_token(model)
    data_path = require_lammps_token(data_file)
    energy_path = require_lammps_token(energy_result)
    atom_path = require_lammps_token(atom_result)
    atom_style = "atomic/kk" if kokkos else "atomic"
    pair_style = (
        "dprc/deepmd/batch/kk" if kokkos else "dprc/deepmd/batch"
    )

    commands = [
        f"plugin load {plugin}",
        "units metal",
        "dimension 3",
        "boundary p p p",
        f"atom_style {atom_style}",
        "atom_modify map array",
    ]
    if kokkos:
        # The compact Kokkos graph and its force scatter require the same
        # one-owner half-list policy used by production batched workloads.
        commands.append("newton on")
    commands.append(f"read_data {data_path}")
    if kokkos:
        commands.append("run_style verlet/kk")
    commands.extend(
        [
            "group qm id 1:4",
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            (
                f"pair_style {pair_style} {model_path} partition_batch yes "
                "center_group qm environment_cutoff 6.0 include_molecule no"
            ),
            "pair_coeff * * C H HW O OW P",
            "compute dprc_atom all pe/atom pair",
            "thermo 1",
            "thermo_style custom step pe",
            "run 0",
            f'print "energy=$(pe:%.17g)" file {energy_path} screen no',
            (
                f"write_dump all custom {atom_path} id type c_dprc_atom "
                'fx fy fz modify sort id format line "%d %d %.17g %.17g %.17g %.17g"'
            ),
        ]
    )
    return "\n".join(commands) + "\n"


def render_data() -> str:
    """Return a fixed atomic data file shared by both adapters.

    ``read_data`` is also how the production umbrella workload creates its
    Kokkos atom container.  Using it here avoids testing the unrelated device
    map transitions performed by repeated ``create_atoms`` commands.
    """
    return """LAMMPS data file for the DeePMD DPRc center-mask regression

8 atoms
6 atom types

0.0 30.0 xlo xhi
0.0 30.0 ylo yhi
0.0 30.0 zlo zhi

Masses

1 12.011
2 1.008
3 1.008
4 15.999
5 15.999
6 30.974

Atoms # atomic

1 1 15.0 15.0 15.0
2 2 15.9 15.1 15.0
3 4 14.5 15.8 15.2
4 6 15.2 14.2 15.5
5 5 17.1 15.4 14.8
6 3 13.2 14.7 15.9
7 5 25.0 25.0 25.0
8 3 5.0 5.0 5.0
"""


def parse_energy(path: Path) -> float:
    """Parse the finite global potential energy emitted by LAMMPS."""
    key, separator, raw_value = path.read_text(encoding="utf-8").strip().partition(
        "="
    )
    if key != "energy" or not separator:
        raise ValueError(f"invalid energy result: {path.read_text()!r}")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite total energy: {value}")
    return value


def parse_atoms(
    path: Path, atom_count: int = ATOM_COUNT
) -> dict[int, dict[str, float | int]]:
    """Parse one sorted ``write_dump`` frame with atomic energy and force."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith("ITEM: ATOMS ")
        )
    except StopIteration as error:
        raise ValueError("LAMMPS atom dump lacks an ITEM: ATOMS header") from error
    fields = lines[header_index].split()[2:]
    expected_fields = ["id", "type", "c_dprc_atom", "fx", "fy", "fz"]
    if fields != expected_fields:
        raise ValueError(f"unexpected atom dump fields: {fields}")

    atoms: dict[int, dict[str, float | int]] = {}
    for line in lines[header_index + 1 : header_index + 1 + atom_count]:
        tokens = line.split()
        if len(tokens) != len(expected_fields):
            raise ValueError(f"invalid atom dump row: {line!r}")
        atom_id = int(tokens[0])
        values = [float(value) for value in tokens[2:]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite atom dump row: {line!r}")
        atoms[atom_id] = {
            "type": int(tokens[1]),
            "atomic_energy": values[0],
            "fx": values[1],
            "fy": values[2],
            "fz": values[3],
        }
    if atoms.keys() != set(range(1, atom_count + 1)):
        raise ValueError(f"unexpected atom IDs: {sorted(atoms)}")
    return atoms


def run_mode(
    executable: Path,
    kokkos: bool,
    dprc_plugin: Path,
    model: Path,
    directory: Path,
) -> tuple[float, dict[int, dict[str, float | int]], str]:
    """Execute one adapter and return its public LAMMPS outputs."""
    name = "kokkos" if kokkos else "host"
    input_file = directory / f"in.{name}"
    data_file = directory / "center-mask.data"
    energy_file = directory / f"{name}.energy"
    atom_file = directory / f"{name}.atoms"
    data_file.write_text(render_data(), encoding="utf-8")
    input_file.write_text(
        render_input(
            kokkos,
            dprc_plugin,
            model,
            data_file,
            energy_file,
            atom_file,
        ),
        encoding="utf-8",
    )
    command = [str(executable), "-log", "none"]
    if kokkos:
        command.extend(
            [
                "-k",
                "on",
                "g",
                "1",
                "-pk",
                "kokkos",
                "newton",
                "on",
                "neigh",
                "half",
            ]
        )
    command.extend(["-in", str(input_file)])
    process = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"LAMMPS {name} calculation failed with exit {process.returncode}:\n"
            f"{process.stdout}"
        )
    return parse_energy(energy_file), parse_atoms(atom_file), process.stdout


def force_norm(atom: dict[str, float | int]) -> float:
    """Return the Euclidean force norm of one parsed atom."""
    return math.sqrt(sum(float(atom[key]) ** 2 for key in ("fx", "fy", "fz")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--dprc-plugin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deepmd-revision", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        for path in (arguments.lammps, arguments.dprc_plugin, arguments.model):
            if not path.is_file():
                raise ValueError(f"required runtime input is not a file: {path}")

        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-center-mask-") as temp:
            directory = Path(temp)
            host_energy, host_atoms, _ = run_mode(
                arguments.lammps,
                False,
                arguments.dprc_plugin,
                arguments.model,
                directory,
            )
            kokkos_energy, kokkos_atoms, _ = run_mode(
                arguments.lammps,
                True,
                arguments.dprc_plugin,
                arguments.model,
                directory,
            )

        failures: dict[str, float] = {}
        total_energy_difference = abs(host_energy - kokkos_energy)
        if total_energy_difference > ENERGY_TOLERANCE_EV:
            failures["host_kokkos_total_energy"] = total_energy_difference

        atomic_energy_difference = max(
            abs(
                float(host_atoms[atom_id]["atomic_energy"])
                - float(kokkos_atoms[atom_id]["atomic_energy"])
            )
            for atom_id in host_atoms
        )
        if atomic_energy_difference > ATOMIC_ENERGY_TOLERANCE_EV:
            failures["host_kokkos_atomic_energy"] = atomic_energy_difference

        force_difference = max(
            abs(float(host_atoms[atom_id][key]) - float(kokkos_atoms[atom_id][key]))
            for atom_id in host_atoms
            for key in ("fx", "fy", "fz")
        )
        if force_difference > FORCE_TOLERANCE_EV_PER_ANGSTROM:
            failures["host_kokkos_force"] = force_difference

        adapter_results: dict[str, dict[str, object]] = {}
        for name, energy, atoms in (
            ("host", host_energy, host_atoms),
            ("kokkos", kokkos_energy, kokkos_atoms),
        ):
            atomic_sum = sum(float(atom["atomic_energy"]) for atom in atoms.values())
            sum_residual = abs(energy - atomic_sum)
            if sum_residual > ENERGY_TOLERANCE_EV:
                failures[f"{name}_total_vs_atomic_sum"] = sum_residual

            environment_energy_max = max(
                abs(float(atoms[atom_id]["atomic_energy"]))
                for atom_id in ENVIRONMENT_ATOMS
            )
            if environment_energy_max > MASK_ZERO_TOLERANCE_EV:
                failures[f"{name}_environment_atomic_energy"] = (
                    environment_energy_max
                )

            environment_force_min = min(
                force_norm(atoms[atom_id]) for atom_id in ENVIRONMENT_ATOMS
            )
            if environment_force_min <= REACTION_FORCE_MIN_EV_PER_ANGSTROM:
                failures[f"{name}_environment_reaction_force"] = (
                    environment_force_min
                )

            outside_energy_max = max(
                abs(float(atoms[atom_id]["atomic_energy"]))
                for atom_id in OUTSIDE_ATOMS
            )
            if outside_energy_max > MASK_ZERO_TOLERANCE_EV:
                failures[f"{name}_outside_atomic_energy"] = outside_energy_max
            outside_force_max = max(
                force_norm(atoms[atom_id]) for atom_id in OUTSIDE_ATOMS
            )
            if outside_force_max > OUTSIDE_FORCE_TOLERANCE_EV_PER_ANGSTROM:
                failures[f"{name}_outside_force"] = outside_force_max

            center_energy_max = max(
                abs(float(atoms[atom_id]["atomic_energy"]))
                for atom_id in CENTER_ATOMS
            )
            if center_energy_max <= MASK_ZERO_TOLERANCE_EV:
                failures[f"{name}_center_energy_vacuous"] = center_energy_max

            adapter_results[name] = {
                "total_energy_ev": energy,
                "atomic_energy_sum_ev": atomic_sum,
                "total_vs_atomic_sum_residual_ev": sum_residual,
                "center_atomic_energy_max_abs_ev": center_energy_max,
                "environment_atomic_energy_max_abs_ev": environment_energy_max,
                "environment_force_min_ev_per_angstrom": environment_force_min,
                "outside_atomic_energy_max_abs_ev": outside_energy_max,
                "outside_force_max_ev_per_angstrom": outside_force_max,
                "atoms": atoms,
            }

        evidence = {
            "schema_version": 1,
            "claim": (
                "the host and /kk names in dprcplugin publish energy only for "
                "DPRc center atoms while retaining MM environment reaction forces"
            ),
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
                "model": {
                    "path": str(arguments.model.resolve()),
                    "sha256": sha256(arguments.model),
                },
            },
            "atom_sets": {
                "centers": sorted(CENTER_ATOMS),
                "selected_environment": sorted(ENVIRONMENT_ATOMS),
                "outside_environment": sorted(OUTSIDE_ATOMS),
            },
            "units": {"energy": "eV", "force": "eV/Angstrom"},
            "absolute_tolerances": {
                "total_energy_ev": ENERGY_TOLERANCE_EV,
                "atomic_energy_ev": ATOMIC_ENERGY_TOLERANCE_EV,
                "force_ev_per_angstrom": FORCE_TOLERANCE_EV_PER_ANGSTROM,
                "masked_energy_ev": MASK_ZERO_TOLERANCE_EV,
                "outside_force_ev_per_angstrom": (
                    OUTSIDE_FORCE_TOLERANCE_EV_PER_ANGSTROM
                ),
                "reaction_force_min_ev_per_angstrom": (
                    REACTION_FORCE_MIN_EV_PER_ANGSTROM
                ),
            },
            "host_kokkos": {
                "total_energy_difference_ev": total_energy_difference,
                "atomic_energy_difference_max_ev": atomic_energy_difference,
                "force_difference_max_ev_per_angstrom": force_difference,
            },
            "results": adapter_results,
            "failures": failures,
        }
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "DeePMD center mask: "
            f"energy_diff={total_energy_difference:.6e} eV "
            f"atomic_diff={atomic_energy_difference:.6e} eV "
            f"force_diff={force_difference:.6e} eV/A "
            f"environment_energy={max(float(adapter_results[name]['environment_atomic_energy_max_abs_ev']) for name in adapter_results):.6e} eV "
            f"reaction_force_min={min(float(adapter_results[name]['environment_force_min_ev_per_angstrom']) for name in adapter_results):.6e} eV/A"
        )
        if failures:
            print(f"DeePMD center-mask check failed: {failures}", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DeePMD center-mask check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
