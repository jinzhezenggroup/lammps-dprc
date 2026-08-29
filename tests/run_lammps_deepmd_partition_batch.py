#!/usr/bin/env python3
"""Compare two-frame DeePMD partition batching with independent B1 calls.

The test uses the same compact DPA4c center-mask system as the host/Kokkos
regression, but perturbs the second frame.  Each independent calculation and
the two-partition calculation use ``dprc/deepmd/batch partition_batch yes``;
therefore
the comparison isolates block-diagonal assembly, frame ordering, and output
scatter in the universe broker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_lammps_deepmd_center_mask",
    ROOT / "tests/run_lammps_deepmd_center_mask.py",
)
assert SPEC is not None and SPEC.loader is not None
CENTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CENTER
SPEC.loader.exec_module(CENTER)

ENERGY_TOLERANCE_EV = 2.0e-5
ATOMIC_ENERGY_TOLERANCE_EV = 2.0e-5
FORCE_TOLERANCE_EV_PER_ANGSTROM = 2.0e-4


def sha256(path: Path) -> str:
    """Return the SHA-256 identity of one runtime artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_data(index: int) -> str:
    """Return one nondegenerate center/environment configuration."""
    text = CENTER.render_data()
    if index == 1:
        text = text.replace(
            "5 5 17.1 15.4 14.8\n6 3 13.2 14.7 15.9\n",
            "5 5 17.3 15.2 14.9\n6 3 13.0 14.9 15.7\n",
        )
    return text


def run_command(command: list[str], directory: Path) -> str:
    """Run LAMMPS and return its combined diagnostic stream."""
    process = subprocess.run(
        command,
        cwd=directory,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "HYDRA_LAUNCHER": "fork"},
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"LAMMPS command failed with exit {process.returncode}:\n"
            f"{process.stdout}"
        )
    return process.stdout


def compare_frame(
    serial_energy: float,
    serial_atoms: dict[int, dict[str, float | int]],
    batch_energy: float,
    batch_atoms: dict[int, dict[str, float | int]],
) -> dict[str, float]:
    """Return maximum absolute serial/batch differences for one frame."""
    return {
        "energy_ev": abs(serial_energy - batch_energy),
        "atomic_energy_ev": max(
            abs(
                float(serial_atoms[atom_id]["atomic_energy"])
                - float(batch_atoms[atom_id]["atomic_energy"])
            )
            for atom_id in serial_atoms
        ),
        "force_ev_per_angstrom": max(
            abs(
                float(serial_atoms[atom_id][component])
                - float(batch_atoms[atom_id][component])
            )
            for atom_id in serial_atoms
            for component in ("fx", "fy", "fz")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--dprc-plugin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deepmd-revision", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        for path in (
            arguments.lammps,
            arguments.mpiexec,
            arguments.dprc_plugin,
            arguments.model,
        ):
            if not path.is_file():
                raise ValueError(f"required runtime input is not a file: {path}")

        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-batch-") as temp:
            directory = Path(temp)
            data_files = [directory / f"frame-{frame}.data" for frame in range(2)]
            for frame, path in enumerate(data_files):
                path.write_text(frame_data(frame), encoding="utf-8")

            serial_results = []
            for frame in range(2):
                frame_dir = directory / f"serial-{frame}"
                frame_dir.mkdir()
                energy_file = frame_dir / "energy.txt"
                atom_file = frame_dir / "atoms.dump"
                input_file = frame_dir / "input.lammps"
                input_file.write_text(
                    CENTER.render_input(
                        False,
                        arguments.dprc_plugin,
                        arguments.model,
                        data_files[frame],
                        energy_file,
                        atom_file,
                    ),
                    encoding="utf-8",
                )
                run_command(
                    [
                        str(arguments.lammps.resolve()),
                        "-log",
                        "none",
                        "-screen",
                        "none",
                        "-in",
                        str(input_file.resolve()),
                    ],
                    frame_dir,
                )
                serial_results.append(
                    (CENTER.parse_energy(energy_file), CENTER.parse_atoms(atom_file))
                )

            batch_dir = directory / "batch"
            batch_dir.mkdir()
            batch_energy = [batch_dir / f"energy-{frame}.txt" for frame in range(2)]
            batch_atoms = [batch_dir / f"atoms-{frame}.dump" for frame in range(2)]
            template = CENTER.render_input(
                False,
                arguments.dprc_plugin,
                arguments.model,
                data_files[0],
                batch_energy[0],
                batch_atoms[0],
            )
            template = template.replace(
                str(data_files[0].resolve()), "${start_data}"
            ).replace(str(batch_energy[0].resolve()), "${energy_result}").replace(
                str(batch_atoms[0].resolve()), "${atom_result}"
            )
            variables = [
                "variable start_data world &",
                f"  {data_files[0].resolve()} &",
                f"  {data_files[1].resolve()}",
                "variable energy_result world &",
                f"  {batch_energy[0].resolve()} &",
                f"  {batch_energy[1].resolve()}",
                "variable atom_result world &",
                f"  {batch_atoms[0].resolve()} &",
                f"  {batch_atoms[1].resolve()}",
            ]
            batch_input = batch_dir / "input.lammps"
            batch_input.write_text(
                "\n".join(variables) + "\n" + template,
                encoding="utf-8",
            )
            run_command(
                [
                    str(arguments.mpiexec.resolve()),
                    "-n",
                    "2",
                    str(arguments.lammps.resolve()),
                    "-partition",
                    "2x1",
                    "-plog",
                    "none",
                    "-pscreen",
                    "none",
                    "-in",
                    str(batch_input.resolve()),
                ],
                batch_dir,
            )
            batch_results = [
                (CENTER.parse_energy(batch_energy[frame]), CENTER.parse_atoms(batch_atoms[frame]))
                for frame in range(2)
            ]

        differences = [
            compare_frame(*serial_results[frame], *batch_results[frame])
            for frame in range(2)
        ]
        maxima = {
            key: max(frame[key] for frame in differences)
            for key in differences[0]
        }
        failures = {
            key: value
            for key, value, tolerance in (
                ("energy_ev", maxima["energy_ev"], ENERGY_TOLERANCE_EV),
                (
                    "atomic_energy_ev",
                    maxima["atomic_energy_ev"],
                    ATOMIC_ENERGY_TOLERANCE_EV,
                ),
                (
                    "force_ev_per_angstrom",
                    maxima["force_ev_per_angstrom"],
                    FORCE_TOLERANCE_EV_PER_ANGSTROM,
                ),
            )
            if value > tolerance
        }
        evidence = {
            "schema_version": 1,
            "claim": "two-frame partition batch equals independent B1 calls",
            "deepmd_revision": arguments.deepmd_revision,
            "inputs": {
                "lammps_sha256": sha256(arguments.lammps),
                "dprc_plugin_sha256": sha256(arguments.dprc_plugin),
                "model_sha256": sha256(arguments.model),
            },
            "per_frame_differences": differences,
            "maximum_differences": maxima,
            "failures": failures,
        }
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "DeepMD partition batch: "
            f"energy={maxima['energy_ev']:.6e} eV "
            f"atomic={maxima['atomic_energy_ev']:.6e} eV "
            f"force={maxima['force_ev_per_angstrom']:.6e} eV/A"
        )
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DeepMD partition-batch check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
