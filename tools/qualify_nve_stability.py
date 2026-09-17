#!/usr/bin/env python3
"""Qualify three-window NVE energy conservation before production sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one evidence artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_path(identity: object, label: str) -> Path:
    """Resolve a hash-pinned artifact or reject changed evidence bytes."""
    if not isinstance(identity, dict) or not {"path", "sha256"} <= identity.keys():
        raise ValueError(f"{label} identity is malformed")
    path = Path(str(identity["path"]))
    if not path.is_file() or sha256(path) != identity["sha256"]:
        raise ValueError(f"{label} bytes changed: {path}")
    return path


def world_log(record: dict[str, Any], world_index: int) -> Path:
    """Resolve the partition log associated with one stable batch slot."""
    logs = record.get("lammps_logs")
    if not isinstance(logs, dict) or not logs:
        raise ValueError("LAMMPS log ledger is missing")
    suffix = f".{world_index}"
    matching = [identity for name, identity in logs.items() if name.endswith(suffix)]
    if len(matching) == 1:
        return recorded_path(matching[0], f"world {world_index} LAMMPS log")
    if len(logs) == 1 and world_index == 0:
        return recorded_path(next(iter(logs.values())), "single-world LAMMPS log")
    raise ValueError(f"could not identify LAMMPS log for world {world_index}")


def thermo_rows(path: Path) -> list[dict[str, float]]:
    """Read every finite custom-thermo row containing step, temperature, and energy."""
    rows: list[dict[str, float]] = []
    header: list[str] | None = None
    required = {"Step", "Temp", "TotEng"}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw_line.split()
        if fields and fields[0] == "Step":
            header = fields
            continue
        if header is None or len(fields) != len(header):
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite thermo value in {path}")
        row = dict(zip(header, values, strict=True))
        if required <= row.keys():
            rows.append(row)
    if not rows:
        raise ValueError(f"no complete thermo rows found in {path}")
    return rows


def linear_slope(x_values: list[float], y_values: list[float]) -> float:
    """Return the ordinary-least-squares slope without external dependencies."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("linear drift requires at least two paired observations")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("thermo rows do not span a nonzero time interval")
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator


def qualify(
    record_path: Path,
    manifest_path: Path,
    *,
    minimum_samples: int = 10,
    maximum_absolute_drift_rate_kcal_mol_ps_atom: float = 1.0e-4,
    maximum_absolute_net_drift_kcal_mol_atom: float = 5.0e-4,
    minimum_mean_temperature_kelvin: float = 200.0,
    maximum_mean_temperature_kelvin: float = 400.0,
    transient_steps: int = 0,
    expected_mode: str = "qmmm-dpa4c",
) -> dict[str, Any]:
    """Qualify the explicitly selected Hamiltonian with the same NVE gates.

    Uncorrected QM/MM must be selected explicitly; it cannot masquerade as
    a qualified DPRc trajectory, and must not carry correction models.
    """
    if expected_mode not in {"qmmm", "qmmm-dpa4c"}:
        raise ValueError("unsupported NVE Hamiltonian")
    if minimum_samples < 2:
        raise ValueError("minimum sample count must be at least two")
    if transient_steps < 0:
        raise ValueError("transient steps must be nonnegative")
    if not all(math.isfinite(value) for value in (
        maximum_absolute_drift_rate_kcal_mol_ps_atom,
        maximum_absolute_net_drift_kcal_mol_atom,
        minimum_mean_temperature_kelvin, maximum_mean_temperature_kelvin,
    )):
        raise ValueError("NVE acceptance thresholds must be finite")
    if (
        maximum_absolute_drift_rate_kcal_mol_ps_atom < 0.0
        or maximum_absolute_net_drift_kcal_mol_atom < 0.0
    ):
        raise ValueError("energy-drift tolerances must be nonnegative")
    if minimum_mean_temperature_kelvin >= maximum_mean_temperature_kelvin:
        raise ValueError("temperature acceptance interval is empty")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("status") != "passed":
        raise ValueError("NVE invocation record did not pass")
    execution = record.get("execution")
    if not isinstance(execution, dict) or execution.get("mode") != expected_mode:
        raise ValueError(f"NVE record does not match the requested {expected_mode} Hamiltonian")
    dynamics = execution.get("dynamics")
    if not isinstance(dynamics, dict) or dynamics.get("ensemble") != "NVE":
        raise ValueError("NVE record did not explicitly disable the thermostat")
    if dynamics.get("thermostat") != "disabled" or dynamics.get(
        "center_of_mass_momentum_removal"
    ) != "disabled":
        raise ValueError("NVE record contains a non-Hamiltonian dynamics operation")
    schedule = execution.get("dprc_schedule")
    if expected_mode == "qmmm-dpa4c" and (
        not isinstance(schedule, dict)
        or schedule.get("models_qualified_as_xtb_dprc") is not True
    ):
        raise ValueError("NVE record did not use the qualified primary DPRc model")
    if expected_mode == "qmmm" and (schedule is not None or record.get("runtime", {}).get("models")):
        raise ValueError("uncorrected QM/MM NVE record contains DPRc models or a DPRc schedule")

    order = record.get("window_order")
    if not isinstance(order, list) or len(order) != 3:
        raise ValueError("NVE gate requires exactly three representative windows")
    atom_count = int(manifest["system"]["lammps_atoms"])
    timestep_fs = float(manifest["dynamics"]["timestep_fs"])
    if atom_count <= 0 or timestep_fs <= 0.0:
        raise ValueError("manifest atom count and timestep must be positive")
    measurement_steps = int(record["steps_per_window"]) - transient_steps
    # A startup-transient test is an additional full-length qualification,
    # not a retrospective shortening of the original failed 5 ps test.
    if measurement_steps * timestep_fs < 5000.0:
        raise ValueError("NVE measurement must span at least 5 ps")

    windows: dict[str, Any] = {}
    for world_index, tag in enumerate(order):
        rows = thermo_rows(world_log(record, world_index))
        if len(rows) < minimum_samples:
            raise ValueError(
                f"{tag} has only {len(rows)} thermo samples; need {minimum_samples}"
            )
        steps = [row["Step"] for row in rows]
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(f"{tag} thermo steps are not strictly increasing")
        elapsed_ps = [(step - steps[0]) * timestep_fs / 1000.0 for step in steps]
        energies = [row["TotEng"] for row in rows]
        temperatures = [row["Temp"] for row in rows]
        duration_ps = elapsed_ps[-1]
        expected_duration_ps = float(record["steps_per_window"]) * timestep_fs / 1000.0
        if not math.isclose(duration_ps, expected_duration_ps, abs_tol=1.0e-12):
            raise ValueError(
                f"{tag} covers {duration_ps} ps, expected {expected_duration_ps} ps"
            )
        if transient_steps:
            measurement_start = steps[0] + transient_steps
            rows = [row for row in rows if row["Step"] >= measurement_start]
            if not rows or rows[0]["Step"] != measurement_start:
                raise ValueError(f"{tag} lacks the declared measurement-start sample")
            if len(rows) < minimum_samples:
                raise ValueError(f"{tag} has too few post-transient samples")
            steps = [row["Step"] for row in rows]
            elapsed_ps = [(step - steps[0]) * timestep_fs / 1000.0 for step in steps]
            energies = [row["TotEng"] for row in rows]
            temperatures = [row["Temp"] for row in rows]
            duration_ps = elapsed_ps[-1]
        slope = linear_slope(elapsed_ps, energies)
        net_drift = energies[-1] - energies[0]
        slope_per_atom = slope / atom_count
        net_drift_per_atom = net_drift / atom_count
        mean_temperature = sum(temperatures) / len(temperatures)
        checks = {
            "absolute_drift_rate_per_atom": (
                abs(slope_per_atom)
                <= maximum_absolute_drift_rate_kcal_mol_ps_atom
            ),
            "absolute_net_drift_per_atom": (
                abs(net_drift_per_atom)
                <= maximum_absolute_net_drift_kcal_mol_atom
            ),
            "mean_temperature": (
                minimum_mean_temperature_kelvin
                <= mean_temperature
                <= maximum_mean_temperature_kelvin
            ),
        }
        windows[str(tag)] = {
            "world_index": world_index,
            "samples": len(rows),
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
            "duration_ps": duration_ps,
            "initial_total_energy_kcal_mol": energies[0],
            "final_total_energy_kcal_mol": energies[-1],
            "net_drift_kcal_mol": net_drift,
            "net_drift_kcal_mol_atom": net_drift_per_atom,
            "linear_drift_rate_kcal_mol_ps": slope,
            "linear_drift_rate_kcal_mol_ps_atom": slope_per_atom,
            "mean_temperature_kelvin": mean_temperature,
            "minimum_temperature_kelvin": min(temperatures),
            "maximum_temperature_kelvin": max(temperatures),
            "checks": checks,
            "passed": all(checks.values()),
        }

    passed = all(bool(result["passed"]) for result in windows.values())
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": f"three-window-{expected_mode}-nve-stability",
        "analysis_protocol": {
            "transient_steps": transient_steps,
            "measurement_steps": measurement_steps,
            "measurement_duration_ps": measurement_steps * timestep_fs / 1000.0,
            "policy": "declared-startup-transient-then-continuous-NVE",
        },
        "thresholds": {
            "minimum_samples": minimum_samples,
            "maximum_absolute_drift_rate_kcal_mol_ps_atom": (
                maximum_absolute_drift_rate_kcal_mol_ps_atom
            ),
            "maximum_absolute_net_drift_kcal_mol_atom": (
                maximum_absolute_net_drift_kcal_mol_atom
            ),
            "minimum_mean_temperature_kelvin": minimum_mean_temperature_kelvin,
            "maximum_mean_temperature_kelvin": maximum_mean_temperature_kelvin,
        },
        "inputs": {
            "record": {
                "path": str(record_path.resolve()),
                "sha256": sha256(record_path),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256(manifest_path),
            },
            "models": record.get("runtime", {}).get("models", []),
        },
        "atom_count": atom_count,
        "timestep_fs": timestep_fs,
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("qmmm", "qmmm-dpa4c"), default="qmmm-dpa4c")
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument(
        "--transient-steps", type=int, default=0,
        help="predeclared initial steps excluded from a continuous run; retain at least 5 ps",
    )
    parser.add_argument(
        "--maximum-absolute-drift-rate-kcal-mol-ps-atom",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--maximum-absolute-net-drift-kcal-mol-atom",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument("--minimum-mean-temperature-kelvin", type=float, default=200.0)
    parser.add_argument("--maximum-mean-temperature-kelvin", type=float, default=400.0)
    arguments = parser.parse_args()
    result = qualify(
        arguments.record,
        arguments.manifest,
        minimum_samples=arguments.minimum_samples,
        maximum_absolute_drift_rate_kcal_mol_ps_atom=(
            arguments.maximum_absolute_drift_rate_kcal_mol_ps_atom
        ),
        maximum_absolute_net_drift_kcal_mol_atom=(
            arguments.maximum_absolute_net_drift_kcal_mol_atom
        ),
        minimum_mean_temperature_kelvin=arguments.minimum_mean_temperature_kelvin,
        maximum_mean_temperature_kelvin=arguments.maximum_mean_temperature_kelvin,
        transient_steps=arguments.transient_steps,
        expected_mode=arguments.mode,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
