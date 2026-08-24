#!/usr/bin/env python3
"""Prepare and run the hash-pinned ETP/ETH umbrella workload.

The scientific inputs stay in an external ``dprc-tutorial`` checkout.  This
runner verifies their exact identities, generates per-window Colvars inputs in
an external run directory, and records enough provenance to resume an
expensive seed/equilibration/production campaign without treating a dirty or
unlicensed source checkout as publication-qualified evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "workloads/etpeth/manifest.json"
DANGEROUS_BUILDS = re.compile(r"Dangerous builds\s*=\s*(\d+)")


@dataclass(frozen=True)
class Window:
    """One immutable stable batch slot in the ascending umbrella grid."""

    index: int
    center_tenths: int

    @property
    def center(self) -> float:
        return self.center_tenths / 10.0

    @property
    def tag(self) -> str:
        sign = "m" if self.center_tenths < 0 else "p"
        magnitude = abs(self.center_tenths)
        return f"{sign}{magnitude // 10}p{magnitude % 10}"


@dataclass(frozen=True)
class RunWindow:
    """Per-invocation file and thermostat state for one stable window."""

    window: Window
    start_data: Path
    output_directory: Path
    workspace: Path
    seed: int
    colvars_profile: str = "sampling"

    @property
    def colvars_config(self) -> Path:
        directories = {
            "sampling": "colvars",
            "seed": "colvars-seed",
        }
        try:
            directory = directories[self.colvars_profile]
        except KeyError as error:
            raise ValueError(
                f"unsupported Colvars profile {self.colvars_profile!r}"
            ) from error
        return self.workspace / "generated" / directory / f"{self.window.tag}.conf"

    @property
    def colvars_prefix(self) -> Path:
        return self.output_directory / self.window.tag

    @property
    def final_data(self) -> Path:
        return self.output_directory / f"{self.window.tag}.data"

    @property
    def final_restart(self) -> Path:
        return self.output_directory / f"{self.window.tag}.restart"

    @property
    def trajectory(self) -> Path:
        return self.output_directory / f"{self.window.tag}.lammpstrj"

    @property
    def model_deviation(self) -> Path:
        """Return the optional four-model deviation output for this window."""
        return self.output_directory / f"{self.window.tag}.model-deviation.out"


class WorkspaceLock:
    """Prevent two launchers from writing one resumable workload DAG."""

    def __init__(self, output: Path, *, recover_stale: bool = False):
        self.path = output / ".etpeth-run.lock"
        self.descriptor: int | None = None
        self.recover_stale = recover_stale

    @staticmethod
    def process_start_ticks(pid: int) -> int | None:
        """Read Linux process identity, distinguishing PID reuse."""
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            return int(fields[21])
        except (OSError, ValueError, IndexError):
            return None

    def recover(self) -> None:
        """Archive a proven-stale same-host lock; never recover by age alone."""
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            owner_host = str(owner["host"])
            owner_pid = int(owner["pid"])
            owner_start = int(owner["process_start_ticks"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot prove that malformed lock {self.path} is stale"
            ) from error
        if owner_host != socket.gethostname():
            raise ValueError(
                f"cannot recover lock owned by different host {owner_host}"
            )
        current_start = self.process_start_ticks(owner_pid)
        if current_start == owner_start:
            raise ValueError(
                f"workload lock owner PID {owner_pid} is still active on {owner_host}"
            )
        archived = self.path.with_name(
            f"{self.path.name}.stale-{owner_pid}-{int(time.time())}"
        )
        self.path.replace(archived)

    def __enter__(self) -> Self:
        if self.path.exists() and self.recover_stale:
            self.recover()
        try:
            self.descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
        except FileExistsError as error:
            owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
            raise ValueError(
                f"workload is already locked by {owner or 'an unknown launcher'}; "
                "remove the lock only after proving that process is gone"
            ) from error
        try:
            process_start = self.process_start_ticks(os.getpid())
            if process_start is None:
                raise OSError("could not identify the launcher process")
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "process_start_ticks": process_start,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            os.write(
                self.descriptor,
                (json.dumps(payload, sort_keys=True) + "\n").encode(),
            )
            os.fsync(self.descriptor)
        except Exception:
            os.close(self.descriptor)
            self.descriptor = None
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, *_: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        self.path.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a potentially large artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the workload contract and reject an unsupported schema."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported workload schema in {path}")
    return manifest


def git_output(repository: Path, *arguments: str) -> str:
    """Run one read-only Git query with an actionable diagnostic."""
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed for {repository}: {process.stdout.strip()}"
        )
    return process.stdout.strip()


def windows_from_manifest(manifest: dict[str, Any]) -> list[Window]:
    """Build the exact integer-tenths grid without binary-float drift."""
    umbrella = manifest["umbrella"]
    start = int(umbrella["start_tenths_angstrom"])
    stop = int(umbrella["stop_tenths_angstrom"])
    step = int(umbrella["step_tenths_angstrom"])
    centers = list(range(start, stop + (1 if step > 0 else -1), step))
    if len(centers) != int(umbrella["count"]) or centers[-1] != stop:
        raise ValueError("umbrella grid does not match its declared count and stop")
    return [
        Window(index=index, center_tenths=center)
        for index, center in enumerate(centers)
    ]


def checkpoint_boundary_tolerances(
    manifest: dict[str, Any],
) -> tuple[float, float]:
    """Return the declared Colvars differences allowed after ``write_data``."""
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
    """Record the exact boundary equality rule in every chunk ledger."""
    reaction, angle = checkpoint_boundary_tolerances(manifest)
    return {
        "key": "absolute_timestep",
        "drop_duplicate_boundary_sample": True,
        "reaction_coordinate_absolute_tolerance_angstrom": reaction,
        "attack_angle_absolute_tolerance_degree": angle,
    }


def verify_source(
    tutorial: Path,
    manifest: dict[str, Any],
    *,
    allow_unqualified_source: bool,
) -> dict[str, Any]:
    """Verify revision, dirty state, and every external artifact digest."""
    tutorial = tutorial.resolve()
    if not tutorial.is_dir():
        raise ValueError(f"tutorial checkout is not a directory: {tutorial}")

    source = manifest["source"]
    revision = git_output(tutorial, "rev-parse", "HEAD")
    dirty_output = git_output(tutorial, "status", "--porcelain=v1")
    dirty_entries = dirty_output.splitlines() if dirty_output else []
    if revision != source["revision"]:
        raise ValueError(
            f"tutorial revision {revision} differs from reviewed {source['revision']}"
        )
    qualification_reasons = [
        reason
        for condition, reason in (
            (bool(dirty_entries), "source checkout is dirty"),
            (source["license"] == "NOASSERTION", "source license is unresolved"),
            (
                not source["assets_complete"],
                "upstream supplies only one initial window",
            ),
        )
        if condition
    ]
    if qualification_reasons and not allow_unqualified_source:
        raise ValueError(
            "tutorial source is not publication-qualified ("
            + "; ".join(qualification_reasons)
            + "); pass --allow-unqualified-source only for private diagnostic data generation"
        )

    artifacts: list[dict[str, Any]] = []
    for expected in source["artifacts"]:
        path = tutorial / expected["path"]
        if not path.is_file():
            raise ValueError(f"required reviewed tutorial artifact is missing: {path}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(
                f"tutorial artifact {expected['path']} has SHA-256 {actual}; "
                f"expected {expected['sha256']}"
            )
        artifacts.append(
            {
                "path": str(path.resolve()),
                "relative_path": expected["path"],
                "sha256": actual,
                "classification": expected["classification"],
            }
        )

    qualified = not qualification_reasons
    return {
        "path": str(tutorial),
        "upstream": source["upstream"],
        "revision": revision,
        "dirty": bool(dirty_entries),
        "dirty_entries": dirty_entries,
        "license": source["license"],
        "artifacts": artifacts,
        "qualification": "final" if qualified else "private-diagnostic",
        "qualification_reasons": qualification_reasons,
    }


def ensure_lammps_token(path: Path, *, relative_to: Path | None = None) -> str:
    """Return a path safe for unquoted LAMMPS variable expansion.

    Scientific workflow inputs default to absolute paths because their launch
    directory is intentionally not part of the resume contract.  Performance
    coordinates may instead request paths relative to one explicit execution
    directory.  This keeps multi-partition ``world`` variable payloads short;
    the upstream GPU package has shown a reproducible long-path crash at eight
    one-rank partitions even though the same coordinate is numerically valid.
    """
    resolved = path.resolve()
    rendered = (
        os.path.relpath(resolved, relative_to.resolve())
        if relative_to is not None
        else str(resolved)
    )
    if any(character in rendered for character in " \t\r\n#$&\"'\\"):
        raise ValueError(
            f"LAMMPS runtime path contains unsupported characters: {rendered}"
        )
    return rendered


def render_colvars(
    manifest: dict[str, Any], window: Window, *, profile: str = "sampling"
) -> str:
    """Render a window restraint for sampling or transient seed generation."""
    umbrella = manifest["umbrella"]
    reaction = umbrella["reaction_coordinate"]
    angle = umbrella["attack_angle"]
    frequency = int(manifest["dynamics"]["colvars_frequency_steps"])
    a1, a2, a3 = angle["atom_ids"]
    if profile == "sampling":
        reaction_force_constant = float(
            reaction["force_constant_kcal_mol_angstrom2"]
        )
        profile_note = "Sampling restraint used for anchor, equilibration, and production."
    elif profile == "seed":
        reaction_force_constant = float(
            manifest["protocol"][
                "seed_walk_force_constant_kcal_mol_angstrom2"
            ]
        )
        profile_note = (
            "Transient seed-generation restraint; production uses the sampling "
            "force constant from the umbrella contract."
        )
    else:
        raise ValueError(f"unsupported Colvars profile {profile!r}")
    if not math.isfinite(reaction_force_constant) or reaction_force_constant <= 0.0:
        raise ValueError(f"invalid {profile} reaction-coordinate force constant")
    return f"""# Generated by tools/etpeth_workload.py from the pinned workload manifest.
# {profile_note}
colvarsTrajFrequency {frequency}

colvar {{
  name reaction_coordinate
  width 1.0
  distance {{
    group1 {{ atomNumbers 5 }}
    group2 {{ atomNumbers 1 }}
  }}
  distance {{
    componentCoeff -1.0
    group1 {{ atomNumbers 1 }}
    group2 {{ atomNumbers 12 }}
  }}
}}

harmonic {{
  name reaction_coordinate_restraint
  colvars reaction_coordinate
  centers {window.center:.1f}
  forceConstant {reaction_force_constant:.1f}
}}

colvar {{
  name attack_angle
  width 1.0
  angle {{
    group1 {{ atomNumbers {a1} }}
    group2 {{ atomNumbers {a2} }}
    group3 {{ atomNumbers {a3} }}
  }}
}}

harmonic {{
  name attack_angle_restraint
  colvars attack_angle
  centers {float(angle["center_degree"]):.1f}
  forceConstant {float(angle["force_constant_kcal_mol_degree2"]):.2f}
}}
"""


def write_generated(path: Path, content: str) -> None:
    """Create deterministic generated input and reject silent local rewrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise ValueError(
                f"generated input differs from existing {path}; use a new output directory"
            )
        return
    path.write_text(content, encoding="utf-8")


def prepare_workspace(
    output: Path,
    tutorial: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    source_record: dict[str, Any],
) -> list[Window]:
    """Create only generated controls and provenance outside the source tree."""
    output = output.resolve()
    if output == PROJECT_ROOT or PROJECT_ROOT in output.parents:
        raise ValueError("workload output must be outside the lammps-dprc source tree")
    tutorial = tutorial.resolve()
    if output == tutorial or tutorial in output.parents:
        raise ValueError(
            "workload output must be outside the external tutorial checkout"
        )
    output.mkdir(parents=True, exist_ok=True)
    windows = windows_from_manifest(manifest)
    for window in windows:
        write_generated(
            output / "generated/colvars" / f"{window.tag}.conf",
            render_colvars(manifest, window, profile="sampling"),
        )
        write_generated(
            output / "generated/colvars-seed" / f"{window.tag}.conf",
            render_colvars(manifest, window, profile="seed"),
        )

    manifest_record = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
            "id": manifest["id"],
        },
        "source": source_record,
        "tutorial_path": str(tutorial.resolve()),
        "window_order": [
            {"slot": window.index, "tag": window.tag, "center_angstrom": window.center}
            for window in windows
        ],
    }
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        # Timestamps are not part of the stable source identity.
        previous.pop("created_utc", None)
        comparable = dict(manifest_record)
        comparable.pop("created_utc", None)
        if previous != comparable:
            raise ValueError(
                f"source provenance differs from existing {provenance_path}; use a new output directory"
            )
    else:
        write_json_atomic(provenance_path, manifest_record)
    return windows


def render_world_variable(name: str, values: Iterable[str]) -> list[str]:
    """Wrap long world-variable lists below LAMMPS input line limits."""
    tokens = list(values)
    if not tokens:
        raise ValueError(f"world variable {name} has no values")
    lines = [f"variable {name} world &"]
    for index, token in enumerate(tokens):
        suffix = " &" if index + 1 < len(tokens) else ""
        lines.append(f"  {token}{suffix}")
    return lines


def render_lammps_input(
    manifest: dict[str, Any],
    tutorial: Path,
    plugin: Path | None,
    run_windows: Sequence[RunWindow],
    *,
    steps: int,
    timestep_offset: int = 0,
    trajectory_frequency: int,
    mode: str = "qmmm",
    classical_backend: str = "batched-dprc",
    deepmd_plugin: Path | None = None,
    deepmd_models: Sequence[Path] = (),
    model_deviation_frequency: int = 0,
    run_commands: Sequence[str] | None = None,
    execution_directory: Path | None = None,
) -> str:
    """Render one synchronized LAMMPS invocation for one or more windows.

    ``mode`` keeps the benchmark comparison on one reviewed input generator:
    ``classical`` is the complete Amber/TIP4P force field without xTB,
    ``qmmm`` is the batched xTBloom path, and ``qmmm-dpa4c`` overlays one
    compact DeePMD primary model.  A positive ``model_deviation_frequency``
    enables three additional ensemble models only on deviation timesteps.
    These are execution shapes; callers remain responsible for proving that
    every supplied artifact is a scientific DPRc model rather than an ordinary
    absolute potential.

    ``run_commands`` is used by the performance harness to execute warmup and
    repeated steady-state segments in one LAMMPS process.  Ordinary scientific
    stages leave it unset and retain the single ``run <steps>`` behavior.
    When ``execution_directory`` is supplied, every runtime artifact is
    rendered relative to that directory and LAMMPS must be launched there.
    """
    if steps < 0:
        raise ValueError("LAMMPS step count must be nonnegative")
    if timestep_offset < 0:
        raise ValueError("LAMMPS timestep offset must be nonnegative")
    if not run_windows:
        raise ValueError("at least one run window is required")
    if mode not in {"classical", "qmmm", "qmmm-dpa4c"}:
        raise ValueError(f"unsupported ETP/ETH execution mode: {mode}")
    if classical_backend not in {"batched-dprc", "upstream-gpu"}:
        raise ValueError(f"unsupported classical backend: {classical_backend}")
    uses_dprc_plugin = mode != "classical" or classical_backend == "batched-dprc"
    if uses_dprc_plugin and plugin is None:
        raise ValueError(f"{mode} requires the LAMMPS-DPRc plugin")
    if model_deviation_frequency < 0:
        raise ValueError("model-deviation frequency must be nonnegative")
    if mode == "qmmm-dpa4c":
        if deepmd_plugin is None:
            raise ValueError("qmmm-dpa4c requires one DeePMD plugin")
        expected_models = 1 if model_deviation_frequency == 0 else 4
        if len(deepmd_models) != expected_models:
            schedule = (
                "disabled model deviation"
                if model_deviation_frequency == 0
                else "enabled model deviation"
            )
            raise ValueError(
                f"qmmm-dpa4c with {schedule} requires exactly "
                f"{expected_models} model(s)"
            )
    elif (
        deepmd_plugin is not None
        or deepmd_models
        or model_deviation_frequency != 0
    ):
        raise ValueError("DeePMD runtime inputs are valid only for qmmm-dpa4c")
    if run_commands is not None and not run_commands:
        raise ValueError("benchmark run command list must not be empty")
    def path_token(path: Path) -> str:
        return ensure_lammps_token(path, relative_to=execution_directory)
    dynamics = manifest["dynamics"]
    system = manifest["system"]
    xtb = manifest["xtb"]
    source_forcefield = tutorial / "lammps/forcefield_qmmm_hybrid.inc"
    workspaces = {item.workspace.resolve() for item in run_windows}
    if len(workspaces) != 1:
        raise ValueError("all synchronized windows must share one generated workspace")
    workspace = next(iter(workspaces))
    if mode == "classical" and classical_backend == "upstream-gpu":
        forcefield_path = workspace / "generated/forcefield_classical_gpu.inc"
        # The non-hybrid combined TIP4P style owns both LJ and real-space
        # Coulomb.  Its pair_coeff syntax therefore omits a sub-style token.
        forcefield_text = source_forcefield.read_text(encoding="utf-8").replace(
            " lj/cut ", " "
        )
    else:
        forcefield_path = workspace / "generated/forcefield_dprc_batch.inc"
        # The reviewed tutorial coefficients name the native lj/cut sub-style.
        # Preserve every numeric byte and rewrite only that private style token
        # for the publication proxy registered by this plugin.
        forcefield_text = source_forcefield.read_text(encoding="utf-8").replace(
            " lj/cut ", " lj/cut/dprc/batch "
        )
    write_generated(forcefield_path, forcefield_text)
    forcefield = path_token(forcefield_path)

    for item in run_windows:
        item.output_directory.mkdir(parents=True, exist_ok=True)

    variables: list[str] = []
    variable_values = {
        "window_tag": [item.window.tag for item in run_windows],
        "start_data": [path_token(item.start_data) for item in run_windows],
        "colvars_config": [
            path_token(item.colvars_config) for item in run_windows
        ],
        "colvars_output": [
            path_token(item.colvars_prefix) for item in run_windows
        ],
        "final_data": [path_token(item.final_data) for item in run_windows],
        "final_restart": [
            path_token(item.final_restart) for item in run_windows
        ],
        "trajectory": [path_token(item.trajectory) for item in run_windows],
        "thermostat_seed": [str(item.seed) for item in run_windows],
    }
    if mode == "qmmm-dpa4c" and model_deviation_frequency > 0:
        variable_values["model_deviation"] = [
            path_token(item.model_deviation)
            for item in run_windows
        ]
    for name, values in variable_values.items():
        variables.extend(render_world_variable(name, values))

    mesh = " ".join(str(value) for value in system["pppm_mesh"])
    kmax = " ".join(str(value) for value in xtb["kmax"])
    elements = " ".join(system["elements_by_lammps_type"])
    # Every batched classical path can use the same Kokkos atom, neighbor,
    # bonded, integration, and Colvars chain already required by deepmd/kk.
    # Keeping the non-batched upstream GPU reference on its native host chain
    # avoids changing its baseline semantics.  Positive-stride model deviation
    # remains on DeePMD's generic adapter because deepmd/kk rejects ensembles;
    # that coordinate therefore stays entirely on the compatible host chain.
    deepmd_kokkos = mode == "qmmm-dpa4c" and model_deviation_frequency == 0
    uses_batched_classical = (
        mode != "classical" or classical_backend == "batched-dprc"
    )
    kokkos_device = uses_batched_classical and not (
        mode == "qmmm-dpa4c" and model_deviation_frequency > 0
    )
    deepmd_pair_style = "deepmd/kk" if deepmd_kokkos else "deepmd"
    commands = [
        "# Generated by tools/etpeth_workload.py; do not hand-edit.",
        "clear",
    ]
    if mode == "qmmm-dpa4c":
        assert deepmd_plugin is not None
        commands.append(f"plugin load {path_token(deepmd_plugin)}")
    if uses_dprc_plugin:
        assert plugin is not None
        commands.append(f"plugin load {path_token(plugin)}")
    commands.extend([
        "units real",
        "dimension 3",
        "boundary p p p",
        "atom_style full/kk" if kokkos_device else "atom_style full",
        "atom_modify map array",
    ])
    if kokkos_device:
        # Kokkos GPU initialization may select Newton-off defaults.  The
        # shared batched pair/PPPM broker relies on one-owner force and charge
        # accumulation, so pin both pair and bonded Newton communication before
        # read_data creates the simulation box.
        commands.append("newton on")
    # LAMMPS requires accelerator packages to be initialized before read_data
    # defines the simulation box.  Explicit /gpu styles do not relax that
    # ordering requirement.
    if mode == "classical" and classical_backend == "upstream-gpu":
        commands.append("package gpu 1")
    commands.extend([*variables, "read_data ${start_data}"])
    if kokkos_device:
        # Unlike atom_style, the integrator style requires an existing
        # simulation box and therefore must follow read_data.
        commands.append("run_style verlet/kk")
    commands.extend([
        # Every partition must expose the same timestep to the synchronized
        # broker even when its seed came from a different branch depth.
        f"reset_timestep {timestep_offset}",
        f"group qm id {system['qm_atom_ids'][0]}:{system['qm_atom_ids'][1]}",
        f"group water type {system['water_types'][0]} {system['water_types'][1]}",
        "bond_style harmonic/kk" if kokkos_device else "bond_style harmonic",
        "angle_style harmonic/kk" if kokkos_device else "angle_style harmonic",
    ])

    if mode == "classical" and classical_backend == "upstream-gpu":
        commands.extend(
            [
                # The LAMMPS GPU package accelerates the combined real-space
                # TIP4P pair style.  It has no pppm/tip4p/gpu style, so the
                # reciprocal baseline below remains the upstream CPU solver.
                (
                    "pair_style lj/cut/tip4p/long/gpu 6 7 1 1 0.125 "
                    f"{system['cutoff_angstrom']:.1f}"
                ),
                f"include {forcefield}",
                "pair_modify tail yes",
                "special_bonds amber",
                "kspace_style pppm/tip4p 1.0e-6",
            ]
        )
    else:
        hybrid_style = "hybrid/overlay/kk" if kokkos_device else "hybrid/overlay"
        pair_style = (
            f"pair_style {hybrid_style} lj/cut/dprc/batch "
            f"{system['cutoff_angstrom']:.1f} tip4p/long/dprc/batch "
            f"6 7 1 1 0.125 {system['cutoff_angstrom']:.1f}"
        )
        if mode == "qmmm-dpa4c":
            models = " ".join(path_token(path) for path in deepmd_models)
            dprc = manifest["dprc"]
            pair_style += f" {deepmd_pair_style} {models}"
            if deepmd_kokkos:
                # One universe owner loads DPA4c and evaluates the compact
                # canonical graphs from every synchronized umbrella window in
                # one block-diagonal forward.  Other partitions retain their
                # Kokkos graph builders but never create a PyTorch model.
                pair_style += " partition_batch yes"
            if model_deviation_frequency > 0:
                pair_style += " out_file ${model_deviation}"
            pair_style += (
                f" out_freq {model_deviation_frequency} atomic center_group qm "
                f"environment_cutoff {float(dprc['environment_cutoff_angstrom']):.1f} "
                "include_molecule yes"
            )
        commands.extend(
            [
                pair_style,
                f"include {forcefield}",
                "pair_coeff 6*7 6*7 tip4p/long/dprc/batch",
            ]
        )
        if mode == "qmmm-dpa4c":
            type_map = " ".join(manifest["dprc"]["deepmd_type_map"])
            commands.append(f"pair_coeff * * {deepmd_pair_style} {type_map}")
        commands.extend(
            [
                "pair_modify pair lj/cut/dprc/batch tail yes",
                "special_bonds amber",
                # The private fix and KSpace styles are a matched fused pair.
                # Using native pppm/tip4p/xtb would bypass prepared-field reuse.
                "kspace_style pppm/tip4p/dprc/batch 1.0e-6",
            ]
        )
        if mode == "classical":
            commands.append("fix classical all dprc/classical/batch")

    commands.extend(
        [
            (
                f"kspace_modify mesh {mesh} order {system['pppm_order']} "
                f"gewald {system['pppm_gewald']}"
            ),
        ]
    )
    if mode != "classical":
        commands.extend(
            [
                (
                    f"fix qmmm qm qmmm/xtb/dprc elements {elements} "
                    f"cutoff {system['cutoff_angstrom']:.1f} "
                    f"charge {system['qm_charge']} uhf {system['qm_uhf']} "
                    f"method {xtb['method']} accuracy {xtb['accuracy']} "
                    f"maxiter {xtb['max_iterations']} "
                    f"etemp {xtb['electronic_temperature_kelvin']} "
                    f"mmhardness {xtb['legacy_mm_hardness']} kmax {kmax} "
                    f"ksqmax {xtb['ksqmax']}"
                ),
                "fix_modify qmmm energy yes",
            ]
        )

    thermo_fields = (
        "step temp pe etotal evdwl etail ecoul ebond eangle elong "
        + ("f_qmmm " if mode != "classical" else "")
        + "f_restraints"
    )
    commands.extend([
        (
            "fix water_shake water shake/kk 1.0e-6 200 0 b 1 a 1"
            if kokkos_device
            else "fix water_shake water shake 1.0e-6 200 0 b 1 a 1"
        ),
        "fix integrate all nve/kk" if kokkos_device else "fix integrate all nve",
        (
            f"fix thermostat all {'langevin/kk' if kokkos_device else 'langevin'} "
            f"{dynamics['temperature_kelvin']} "
            f"{dynamics['temperature_kelvin']} {dynamics['langevin_damping_fs']} "
            "${thermostat_seed}"
        ),
        (
            "fix remove_com all momentum/kk 1000 linear 1 1 1"
            if kokkos_device
            else "fix remove_com all momentum 1000 linear 1 1 1"
        ),
        (
            f"fix restraints all {'colvars/kk' if kokkos_device else 'colvars'} "
            "${colvars_config} output ${colvars_output}"
        ),
        "fix_modify restraints energy yes",
        f"timestep {float(dynamics['timestep_fs']) / 1000.0:.6f}",
        f"neighbor {dynamics['neighbor_skin_angstrom']} bin",
        f"neigh_modify every {dynamics['neighbor_every']} delay 0 check yes",
        f"thermo {dynamics['thermo_frequency_steps']}",
        f"thermo_style custom {thermo_fields}",
        "thermo_modify format float %.12g lost error flush yes",
    ])
    if trajectory_frequency > 0:
        commands.extend(
            [
                (
                    f"dump trajectory all custom {trajectory_frequency} ${{trajectory}} "
                    "id mol type q x y z fx fy fz"
                ),
                "dump_modify trajectory sort id pbc yes",
            ]
        )
    commands.extend(run_commands if run_commands is not None else [f"run {steps}"])
    commands.extend(
        ["write_data ${final_data} nocoeff", "write_restart ${final_restart}"]
    )
    return "\n".join(commands) + "\n"


def parse_colvars_rows(path: Path) -> list[dict[str, float]]:
    """Read strict, finite Colvars rows keyed by absolute timestep."""
    if not path.is_file():
        raise ValueError(f"Colvars trajectory was not produced: {path}")
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = line.lstrip("# ").split()
            if "reaction_coordinate" in candidate and "attack_angle" in candidate:
                header = candidate
            continue
        if header is None:
            raise ValueError(
                f"Colvars data appeared before a recognized header in {path}"
            )
        values = line.split()
        if len(header) != len(values):
            raise ValueError(f"Colvars header/data column mismatch in {path}")
        fields = dict(zip(header, values))
        if "step" not in fields:
            raise ValueError(f"Colvars header has no absolute step column in {path}")
        step = float(fields["step"])
        reaction = float(fields["reaction_coordinate"])
        angle = float(fields["attack_angle"])
        if not all(math.isfinite(value) for value in (step, reaction, angle)):
            raise ValueError(f"non-finite Colvars values in {path}")
        if not step.is_integer():
            raise ValueError(f"non-integral Colvars step {step} in {path}")
        rows.append(
            {
                "step": step,
                "reaction_coordinate": reaction,
                "attack_angle": angle,
            }
        )
    if not rows:
        raise ValueError(f"Colvars trajectory has no samples: {path}")
    steps = [row["step"] for row in rows]
    if any(current <= previous for previous, current in pairwise(steps)):
        raise ValueError(f"Colvars steps are not strictly increasing in {path}")
    return rows


def parse_colvars(
    path: Path, expected_final_step: int | None = None
) -> dict[str, float]:
    """Read the final row and optionally require the exact completed timestep."""
    final = parse_colvars_rows(path)[-1]
    if expected_final_step is not None and final["step"] != expected_final_step:
        raise ValueError(
            f"Colvars final step {final['step']} in {path} differs from expected "
            f"{expected_final_step}"
        )
    return final


def merge_colvars(
    paths: Sequence[Path],
    reaction_tolerance: float = 0.0,
    angle_tolerance: float = 0.0,
) -> list[dict[str, float]]:
    """Merge chunks while admitting only declared decimal round-trip noise."""
    merged: list[dict[str, float]] = []
    by_step: dict[float, dict[str, float]] = {}
    for path in paths:
        for row in parse_colvars_rows(path):
            previous = by_step.get(row["step"])
            if previous is not None:
                if not (
                    math.isclose(
                        previous["reaction_coordinate"],
                        row["reaction_coordinate"],
                        rel_tol=0.0,
                        abs_tol=reaction_tolerance,
                    )
                    and math.isclose(
                        previous["attack_angle"],
                        row["attack_angle"],
                        rel_tol=0.0,
                        abs_tol=angle_tolerance,
                    )
                ):
                    raise ValueError(
                        f"conflicting Colvars samples at step {row['step']} across chunks"
                    )
                continue
            if merged and row["step"] < merged[-1]["step"]:
                raise ValueError(
                    "Colvars chunk order moves backward in absolute timestep"
                )
            by_step[row["step"]] = row
            merged.append(row)
    return merged


def seed_for(
    manifest: dict[str, Any], window: Window, stage_code: int, trial: int = 0
) -> int:
    """Derive deterministic, positive, distinct Langevin seeds."""
    base = int(manifest["dynamics"]["base_seed"])
    seed = base + stage_code * 100003 + trial * 1009 + window.index * 101
    return 1 + seed % 900_000_000


def command_output(
    command: Sequence[str], environment: dict[str, str] | None = None
) -> str | None:
    """Capture optional environment identity without making it a run blocker."""
    try:
        process = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def resolve_executable(path: Path) -> Path:
    """Resolve an explicit path or a command name through ``PATH``."""
    if path.is_file():
        return path.resolve()
    discovered = shutil.which(str(path))
    if discovered is None:
        raise ValueError(f"executable is unavailable: {path}")
    return Path(discovered).resolve()


def validate_execution_policy(
    *,
    mode: str,
    deepmd_plugin: Path | None,
    deepmd_models: Sequence[Path],
    model_deviation_frequency: int,
    dpa4c_models_qualified: bool,
    allow_unqualified_dpa4c_models: bool,
) -> None:
    """Fail closed on ambiguous or scientifically mislabeled DPRc inputs."""
    if mode not in {"qmmm", "qmmm-dpa4c"}:
        raise ValueError(f"unsupported production execution mode: {mode}")
    if model_deviation_frequency < 0:
        raise ValueError("--model-deviation-frequency must be nonnegative")
    if dpa4c_models_qualified and allow_unqualified_dpa4c_models:
        raise ValueError(
            "--dpa4c-models-qualified and "
            "--allow-unqualified-dpa4c-models are mutually exclusive"
        )
    if mode == "qmmm":
        if (
            deepmd_plugin is not None
            or deepmd_models
            or model_deviation_frequency != 0
            or dpa4c_models_qualified
            or allow_unqualified_dpa4c_models
        ):
            raise ValueError(
                "DeePMD runtime and model-qualification options require "
                "--mode qmmm-dpa4c"
            )
        return

    if deepmd_plugin is None or not deepmd_plugin.is_file():
        raise ValueError("qmmm-dpa4c requires an existing --deepmd-plugin")
    expected_models = 1 if model_deviation_frequency == 0 else 4
    if len(deepmd_models) != expected_models:
        raise ValueError(
            "qmmm-dpa4c requires exactly "
            f"{expected_models} --deepmd-model artifact(s) for the selected schedule"
        )
    missing = [path for path in deepmd_models if not path.is_file()]
    if missing:
        raise ValueError(f"DPA4c model artifact is unavailable: {missing[0]}")
    if not dpa4c_models_qualified and not allow_unqualified_dpa4c_models:
        raise ValueError(
            "qmmm-dpa4c requires either --dpa4c-models-qualified or the explicit "
            "diagnostic opt-in --allow-unqualified-dpa4c-models"
        )


def execution_record(
    *,
    mode: str,
    model_deviation_frequency: int,
    dpa4c_models_qualified: bool,
    allow_unqualified_dpa4c_models: bool,
) -> dict[str, Any]:
    """Describe the force composition and model-qualification boundary."""
    record: dict[str, Any] = {"mode": mode}
    if mode == "qmmm-dpa4c":
        record["dprc_schedule"] = {
            "primary_model_index": 0,
            "model_deviation_frequency_steps": model_deviation_frequency,
            "model_deviation_enabled": model_deviation_frequency > 0,
            "execution_backend": (
                "deepmd-kk-device"
                if model_deviation_frequency == 0
                else "deepmd-generic-sparse-deviation"
            ),
            "models_qualified_as_xtb_dprc": dpa4c_models_qualified,
            "diagnostic_unqualified_models_allowed": (
                allow_unqualified_dpa4c_models
            ),
        }
    return record


def build_runtime_environment(
    xtbloom_library: Path,
    library_dirs: Sequence[Path],
    cuda_visible_devices: str,
    *,
    uses_deepmd: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Construct and expose the exact loader/thread boundary used by LAMMPS."""
    environment = os.environ.copy()
    loader_dirs = [
        xtbloom_library.resolve().parent,
        *[path.resolve() for path in library_dirs],
    ]
    old_loader_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in loader_dirs]
        + ([old_loader_path] if old_loader_path else [])
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HYDRA_LAUNCHER": "fork",
        }
    )
    selected = {
        key: environment[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "HYDRA_LAUNCHER",
        )
    }
    selected["LD_PRELOAD"] = environment.get("LD_PRELOAD", "")
    if uses_deepmd:
        # Pin both DeePMD/PyTorch operator pools so a multi-window launch does
        # not create nested CPU oversubscription behind the GPU-local broker.
        environment["DP_INTRA_OP_PARALLELISM_THREADS"] = "1"
        environment["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
        selected["DP_INTRA_OP_PARALLELISM_THREADS"] = "1"
        selected["DP_INTER_OP_PARALLELISM_THREADS"] = "1"
    return environment, selected


def runtime_record(
    lammps: Path,
    plugin: Path,
    xtbloom_library: Path,
    mpiexec: Path,
    environment: dict[str, str],
    *,
    deepmd_plugin: Path | None = None,
    deepmd_models: Sequence[Path] = (),
) -> dict[str, Any]:
    """Record exact runtime bytes and selected hardware/toolchain facts."""
    for path in (lammps, plugin, xtbloom_library, *deepmd_models):
        if not path.is_file():
            raise ValueError(f"runtime artifact is not a file: {path}")
    if deepmd_plugin is not None and not deepmd_plugin.is_file():
        raise ValueError(f"runtime artifact is not a file: {deepmd_plugin}")
    cmake_cache = plugin.resolve().parent / "CMakeCache.txt"
    record: dict[str, Any] = {
        "lammps": {"path": str(lammps.resolve()), "sha256": sha256(lammps)},
        "plugin": {"path": str(plugin.resolve()), "sha256": sha256(plugin)},
        "xtbloom": {
            "path": str(xtbloom_library.resolve()),
            "sha256": sha256(xtbloom_library),
        },
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=name,driver_version,memory.total,compute_cap,"
                    "clocks.current.graphics,clocks.current.memory,power.limit"
                ),
                "--format=csv,noheader",
            ]
        ),
        "mpiexec": {
            "path": str(mpiexec),
            "sha256": sha256(mpiexec),
            "version": command_output([str(mpiexec), "--version"]),
        },
        "cpu": command_output(["lscpu"]),
        "process_affinity_cpus": sorted(os.sched_getaffinity(0)),
        "dynamic_dependencies": {
            "lammps": command_output(
                ["ldd", str(lammps.resolve())], environment=environment
            ),
            "plugin": command_output(
                ["ldd", str(plugin.resolve())], environment=environment
            ),
            "xtbloom": command_output(
                ["ldd", str(xtbloom_library.resolve())], environment=environment
            ),
        },
        "python": sys.version,
    }
    if cmake_cache.is_file():
        record["plugin_cmake_cache"] = {
            "path": str(cmake_cache),
            "sha256": sha256(cmake_cache),
        }
    if deepmd_plugin is not None:
        record["deepmd_plugin"] = {
            "path": str(deepmd_plugin.resolve()),
            "sha256": sha256(deepmd_plugin),
        }
        record["models"] = [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in deepmd_models
        ]
        record["dynamic_dependencies"]["deepmd_plugin"] = command_output(
            ["ldd", str(deepmd_plugin.resolve())], environment=environment
        )
    return record


def verify_loaded_xtbloom(
    plugin: Path, xtbloom_library: Path, environment: dict[str, str]
) -> dict[str, str]:
    """Prove the plugin loader resolves the exact requested xTBloom bytes."""
    output = command_output(["ldd", str(plugin.resolve())], environment=environment)
    if output is None:
        raise ValueError(f"could not inspect dynamic dependencies of {plugin}")
    resolved: Path | None = None
    for line in output.splitlines():
        tokens = line.strip().split()
        if tokens and tokens[0] == "libxtbloom.so.0" and len(tokens) >= 3:
            if tokens[1] != "=>" or tokens[2] == "not":
                break
            resolved = Path(tokens[2]).resolve()
            break
    expected = xtbloom_library.resolve()
    if resolved is None:
        raise ValueError(
            "dprcplugin does not resolve libxtbloom.so.0 in the run environment"
        )
    if resolved != expected:
        raise ValueError(
            f"dprcplugin resolves {resolved}, not requested xTBloom {expected}"
        )
    return {
        "soname": "libxtbloom.so.0",
        "resolved_path": str(resolved),
        "sha256": sha256(resolved),
    }


def project_record(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Identify the dirty development runner that produced diagnostic data."""
    revision = git_output(PROJECT_ROOT, "rev-parse", "HEAD")
    dirty_output = git_output(PROJECT_ROOT, "status", "--porcelain=v1")
    provenance = output / "provenance.json"
    dependencies = {}
    for name, repository in (
        ("lammps", PROJECT_ROOT.parent / "lammps"),
        ("xtbloom", PROJECT_ROOT.parent / "xtbloom"),
    ):
        if repository.is_dir():
            dependency_dirty = git_output(repository, "status", "--porcelain=v1")
            dependencies[name] = {
                "path": str(repository.resolve()),
                "revision": git_output(repository, "rev-parse", "HEAD"),
                "dirty": bool(dependency_dirty),
                "dirty_entries": dependency_dirty.splitlines()
                if dependency_dirty
                else [],
            }
    return {
        "path": str(PROJECT_ROOT),
        "revision": revision,
        "dirty": bool(dirty_output),
        "dirty_entries": dirty_output.splitlines() if dirty_output else [],
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        },
        "provenance": {
            "path": str(provenance.resolve()),
            "sha256": sha256(provenance),
        },
        "dependencies": dependencies,
        "qualification": "private-diagnostic" if dirty_output else "clean-source",
    }


def artifact_matches(record: dict[str, Any] | None, path: Path) -> bool:
    """Check both the absolute path and bytes of one recorded artifact."""
    return bool(
        record
        and path.is_file()
        and Path(record.get("path", "")).resolve() == path.resolve()
        and record.get("sha256") == sha256(path)
    )


def record_is_resumable(
    path: Path,
    run_windows: Sequence[RunWindow],
    *,
    input_path: Path,
    steps: int,
    timestep_offset: int,
    trajectory_frequency: int,
    ranks_per_window: int,
    lammps: Path,
    plugin: Path,
    xtbloom_library: Path,
    mpiexec: Path,
    runner_path: Path,
    loaded_xtbloom: dict[str, str],
    plugin_cmake_cache: Path | None,
    selected_environment: dict[str, str],
    manifest_path: Path,
    provenance_path: Path,
    mode: str = "qmmm",
    deepmd_plugin: Path | None = None,
    deepmd_models: Sequence[Path] = (),
    model_deviation_frequency: int = 0,
    dpa4c_models_qualified: bool = False,
    allow_unqualified_dpa4c_models: bool = False,
) -> bool:
    """Accept a checkpoint only when its complete dependency chain matches."""
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") != "passed":
        return False
    wall_seconds = record.get("wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0.0
    ):
        return False
    if (
        record.get("steps_per_window") != steps
        or record.get("timestep_offset") != timestep_offset
        or record.get("worlds") != len(run_windows)
        or record.get("ranks_per_window") != ranks_per_window
        or record.get("selected_environment") != selected_environment
        or not artifact_matches(record.get("input"), input_path)
        or record.get("execution", {"mode": "qmmm"})
        != execution_record(
            mode=mode,
            model_deviation_frequency=model_deviation_frequency,
            dpa4c_models_qualified=dpa4c_models_qualified,
            allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
        )
    ):
        return False
    runtime = record.get("runtime", {})
    if not all(
        artifact_matches(runtime.get(name), artifact)
        for name, artifact in (
            ("lammps", lammps),
            ("plugin", plugin),
            ("xtbloom", xtbloom_library),
            ("mpiexec", mpiexec),
        )
    ):
        return False
    if mode == "qmmm-dpa4c":
        if deepmd_plugin is None or not artifact_matches(
            runtime.get("deepmd_plugin"), deepmd_plugin
        ):
            return False
        recorded_models = runtime.get("models")
        if (
            not isinstance(recorded_models, list)
            or len(recorded_models) != len(deepmd_models)
            or not all(
                artifact_matches(identity, model)
                for identity, model in zip(
                    recorded_models, deepmd_models, strict=True
                )
            )
        ):
            return False
    elif runtime.get("deepmd_plugin") is not None or runtime.get("models") is not None:
        return False
    if record.get("loaded_xtbloom") != loaded_xtbloom:
        return False
    launcher_log = record.get("launcher_log")
    if (
        not isinstance(launcher_log, dict)
        or "path" not in launcher_log
        or not artifact_matches(launcher_log, Path(launcher_log["path"]))
    ):
        return False
    lammps_logs = record.get("lammps_logs")
    if (
        not isinstance(lammps_logs, dict)
        or len(lammps_logs) != len(run_windows)
        or not lammps_logs
    ):
        return False
    dangerous_builds = record.get("dangerous_builds")
    if not isinstance(dangerous_builds, dict) or set(dangerous_builds) != set(
        lammps_logs
    ):
        return False
    if not all(
        isinstance(identity, dict)
        and "path" in identity
        and artifact_matches(identity, Path(identity["path"]))
        for identity in lammps_logs.values()
    ):
        return False
    log_paths = [Path(identity["path"]) for identity in lammps_logs.values()]
    if (
        len({path.parent.resolve() for path in log_paths}) != 1
        or {path.name for path in log_paths} != set(lammps_logs)
        or set(log_paths[0].parent.glob("log.lammps*")) != set(log_paths)
    ):
        return False
    if plugin_cmake_cache is not None and not artifact_matches(
        runtime.get("plugin_cmake_cache"), plugin_cmake_cache
    ):
        return False
    project = record.get("project", {})
    if not artifact_matches(project.get("runner"), runner_path):
        return False
    if not artifact_matches(project.get("manifest"), manifest_path):
        return False
    if not artifact_matches(project.get("provenance"), provenance_path):
        return False

    start_inputs = record.get("start_inputs", {})
    colvars_configs = record.get("colvars_configs", {})
    expected = record.get("outputs", {})
    if not all(
        isinstance(section, dict)
        for section in (start_inputs, colvars_configs, expected)
    ):
        return False
    for item in run_windows:
        if not artifact_matches(start_inputs.get(item.window.tag), item.start_data):
            return False
        if not artifact_matches(
            colvars_configs.get(item.window.tag), item.colvars_config
        ):
            return False
        outputs = expected.get(item.window.tag, {})
        colvars_path = Path(str(item.colvars_prefix) + ".colvars.traj")
        required_outputs = [
            artifact_matches(outputs.get("data"), item.final_data),
            artifact_matches(outputs.get("restart"), item.final_restart),
            artifact_matches(outputs.get("colvars"), colvars_path),
        ]
        if trajectory_frequency > 0:
            required_outputs.append(
                artifact_matches(outputs.get("trajectory"), item.trajectory)
            )
        if mode == "qmmm-dpa4c" and model_deviation_frequency > 0:
            required_outputs.append(
                artifact_matches(outputs.get("model_deviation"), item.model_deviation)
            )
        if not all(required_outputs):
            return False
    expected_order = [item.window.tag for item in run_windows]
    return bool(
        record.get("window_order") == expected_order
        and set(record.get("start_inputs", {})) == set(expected_order)
        and set(record.get("colvars_configs", {})) == set(expected_order)
        and set(record.get("outputs", {})) == set(expected_order)
    )


def require_recorded_output(record_path: Path, tag: str, expected_path: Path) -> None:
    """Reject a downstream stage unless its exact prerequisite was accepted."""
    if not record_path.is_file():
        raise ValueError(f"required accepted checkpoint is missing: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read checkpoint {record_path}: {error}") from error
    if record.get("status") != "passed":
        raise ValueError(f"required checkpoint did not pass: {record_path}")
    data = record.get("outputs", {}).get(tag, {}).get("data")
    if not data:
        raise ValueError(f"checkpoint {record_path} has no accepted output for {tag}")
    if Path(data["path"]).resolve() != expected_path.resolve():
        raise ValueError(f"checkpoint {record_path} records a different path for {tag}")
    if not expected_path.is_file() or sha256(expected_path) != data.get("sha256"):
        raise ValueError(
            f"accepted prerequisite bytes changed for {tag}: {expected_path}"
        )


def validate_invocation_record_current(
    record_path: Path, common: dict[str, Any]
) -> dict[str, Any]:
    """Validate one accepted invocation against current bytes and loader state."""
    if not record_path.is_file():
        raise ValueError(f"accepted invocation record is missing: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read invocation record {record_path}: {error}"
        ) from error
    if record.get("status") != "passed":
        raise ValueError(f"invocation did not pass: {record_path}")
    wall_seconds = record.get("wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0.0
    ):
        raise ValueError(f"invocation has invalid wall_seconds: {record_path}")

    if record.get("record_kind", "native-invocation") != "native-invocation":
        raise ValueError(f"unsupported invocation record kind in {record_path}")

    mode = common.get("mode", "qmmm")
    deepmd_plugin = common.get("deepmd_plugin")
    deepmd_models = tuple(common.get("deepmd_models", ()))
    model_deviation_frequency = int(common.get("model_deviation_frequency", 0))
    dpa4c_models_qualified = bool(common.get("dpa4c_models_qualified", False))
    allow_unqualified_dpa4c_models = bool(
        common.get("allow_unqualified_dpa4c_models", False)
    )
    expected_execution = execution_record(
        mode=mode,
        model_deviation_frequency=model_deviation_frequency,
        dpa4c_models_qualified=dpa4c_models_qualified,
        allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
    )
    if record.get("execution", {"mode": "qmmm"}) != expected_execution:
        raise ValueError(f"execution policy changed since {record_path}")

    environment, selected = build_runtime_environment(
        common["xtbloom_library"],
        common["library_dirs"],
        common["cuda_visible_devices"],
        uses_deepmd=mode == "qmmm-dpa4c",
    )
    mpi_launcher = resolve_executable(common["mpiexec"])
    runtime = record.get("runtime", {})
    for name, current in (
        ("lammps", common["lammps"]),
        ("plugin", common["plugin"]),
        ("xtbloom", common["xtbloom_library"]),
        ("mpiexec", mpi_launcher),
    ):
        if not artifact_matches(runtime.get(name), current):
            raise ValueError(f"{name} identity changed since {record_path}")
    if mode == "qmmm-dpa4c":
        if deepmd_plugin is None or not artifact_matches(
            runtime.get("deepmd_plugin"), deepmd_plugin
        ):
            raise ValueError(f"DeePMD plugin identity changed since {record_path}")
        recorded_models = runtime.get("models")
        if (
            not isinstance(recorded_models, list)
            or len(recorded_models) != len(deepmd_models)
            or not all(
                artifact_matches(identity, model)
                for identity, model in zip(
                    recorded_models, deepmd_models, strict=True
                )
            )
        ):
            raise ValueError(f"DPA4c model identity changed since {record_path}")
    elif runtime.get("deepmd_plugin") is not None or runtime.get("models") is not None:
        raise ValueError(f"unexpected DeePMD runtime in {record_path}")
    cmake_cache = common["plugin"].resolve().parent / "CMakeCache.txt"
    if cmake_cache.is_file() and not artifact_matches(
        runtime.get("plugin_cmake_cache"), cmake_cache
    ):
        raise ValueError(f"plugin CMake cache changed since {record_path}")
    if not cmake_cache.is_file() and runtime.get("plugin_cmake_cache") is not None:
        raise ValueError(f"plugin CMake cache disappeared since {record_path}")

    project = record.get("project", {})
    for name, current in (
        ("runner", Path(__file__).resolve()),
        ("manifest", common["manifest_path"]),
        ("provenance", common["output"] / "provenance.json"),
    ):
        if not artifact_matches(project.get(name), current):
            raise ValueError(f"{name} identity changed since {record_path}")
    if record.get("selected_environment") != selected:
        raise ValueError(f"runtime environment changed since {record_path}")
    loaded = verify_loaded_xtbloom(
        common["plugin"], common["xtbloom_library"], environment
    )
    if record.get("loaded_xtbloom") != loaded:
        raise ValueError(f"xTBloom loader resolution changed since {record_path}")

    input_identity = record.get("input")
    if not isinstance(input_identity, dict) or "path" not in input_identity:
        raise ValueError(f"invocation has no generated input identity: {record_path}")
    if not artifact_matches(input_identity, Path(input_identity["path"])):
        raise ValueError(f"generated input changed since {record_path}")
    launcher_log = record.get("launcher_log")
    if not isinstance(launcher_log, dict) or "path" not in launcher_log:
        raise ValueError(f"invocation has no launcher log identity: {record_path}")
    if not artifact_matches(launcher_log, Path(launcher_log["path"])):
        raise ValueError(f"launcher log changed since {record_path}")
    lammps_logs = record.get("lammps_logs")
    if not isinstance(lammps_logs, dict) or not lammps_logs:
        raise ValueError(f"invocation has no LAMMPS log identities: {record_path}")
    if len(lammps_logs) != record.get("worlds"):
        raise ValueError(f"invocation partition log count changed in {record_path}")
    dangerous_builds = record.get("dangerous_builds")
    if not isinstance(dangerous_builds, dict) or set(dangerous_builds) != set(
        lammps_logs
    ):
        raise ValueError(f"dangerous-build log set changed in {record_path}")
    for name, identity in lammps_logs.items():
        if not isinstance(identity, dict) or "path" not in identity:
            raise ValueError(f"invalid LAMMPS log identity {name} in {record_path}")
        path = Path(identity["path"])
        if path.name != name or not artifact_matches(identity, path):
            raise ValueError(f"LAMMPS log {name} changed since {record_path}")
    log_paths = [Path(identity["path"]) for identity in lammps_logs.values()]
    if len({path.parent.resolve() for path in log_paths}) != 1 or set(
        log_paths[0].parent.glob("log.lammps*")
    ) != set(log_paths):
        raise ValueError(f"LAMMPS partition log set changed since {record_path}")
    for section in ("start_inputs", "colvars_configs", "outputs"):
        contents = record.get(section)
        if not isinstance(contents, dict) or not contents:
            raise ValueError(f"invocation has no {section}: {record_path}")
        for tag, artifacts in contents.items():
            if section in {"start_inputs", "colvars_configs"}:
                artifacts = {"start": artifacts}
            if not isinstance(artifacts, dict):
                raise TypeError(f"invalid {section} entry {tag} in {record_path}")
            for kind, identity in artifacts.items():
                if kind in {
                    "center_angstrom",
                    "final_values",
                    "reaction_coordinate_error_angstrom",
                    "attack_angle_error_degree",
                    "seed_acceptance",
                }:
                    continue
                if not isinstance(identity, dict) or "path" not in identity:
                    raise TypeError(
                        f"invalid {section} artifact {tag}/{kind} in {record_path}"
                    )
                if not artifact_matches(identity, Path(identity["path"])):
                    raise ValueError(
                        f"{section} artifact {tag}/{kind} changed since {record_path}"
                    )
    window_order = record.get("window_order")
    if (
        not isinstance(window_order, list)
        or len(window_order) != record.get("worlds")
        or set(window_order) != set(record["start_inputs"])
        or set(window_order) != set(record["colvars_configs"])
        or set(window_order) != set(record["outputs"])
    ):
        raise ValueError(f"invocation window order is invalid in {record_path}")
    return record


def require_completed_stage(
    ledger_path: Path,
    *,
    stage: str,
    windows: Sequence[Window],
    expected_start_data: dict[str, Path],
    expected_final_data: dict[str, Path],
    total_steps: int,
    maximum_chunk_steps: int,
    trajectory_frequency: int,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete chunk DAG before consuming its final states."""
    if not ledger_path.is_file():
        raise ValueError(f"completed stage ledger is missing: {ledger_path}")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read stage ledger {ledger_path}: {error}"
        ) from error
    manifest = common.get("manifest")
    if not isinstance(manifest, dict):
        raise TypeError("stage validation requires the workload manifest")
    expected_order = [window.tag for window in windows]
    qualification = ledger.get("qualification", "native-chunked")
    if qualification != "native-chunked":
        raise ValueError(f"unsupported stage qualification in {ledger_path}")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "passed"
        or ledger.get("stage") != stage
        or ledger.get("total_steps_per_window") != total_steps
        or ledger.get("maximum_chunk_steps") != maximum_chunk_steps
        or ledger.get("window_order") != expected_order
        or ledger.get("series_merge_policy") != series_merge_policy(manifest)
    ):
        raise ValueError(f"stage protocol in {ledger_path} does not match the request")

    chunks = ledger.get("chunks", [])
    if not chunks or ledger.get("chunk_count") != len(chunks):
        raise ValueError(f"stage ledger has an invalid chunk list: {ledger_path}")
    expected_sizes = chunk_sizes(total_steps, maximum_chunk_steps)
    expected_chunk_count = len(expected_sizes)
    if len(chunks) != expected_chunk_count:
        raise ValueError("stage chunk count differs from the requested protocol")
    previous_outputs: dict[str, Any] | None = None
    offset = 0
    summed_wall_seconds = 0.0
    colvars_by_window: dict[str, list[dict[str, str]]] = {
        window.tag: [] for window in windows
    }
    trajectories_by_window: dict[str, list[dict[str, str]]] = {
        window.tag: [] for window in windows
    }
    for chunk_index, (chunk, expected_steps) in enumerate(
        zip(chunks, expected_sizes, strict=True)
    ):
        if chunk.get("timestep_offset") != offset:
            raise ValueError(f"chunk timeline has a gap or overlap in {ledger_path}")
        steps = int(chunk.get("steps_per_window", 0))
        expected_name = (
            f"{stage}-chunk-{chunk_index + 1:03d}-of-{expected_chunk_count:03d}"
        )
        expected_record_path = common["output"] / "records" / f"{expected_name}.json"
        record_identity = chunk.get("record")
        if not isinstance(record_identity, dict):
            raise TypeError("stage ledger has an invalid record identity")
        if (
            steps != expected_steps
            or chunk.get("name") != expected_name
            or Path(record_identity.get("path", "")).resolve()
            != expected_record_path.resolve()
        ):
            raise ValueError(f"invalid chunk length in {ledger_path}")
        record_path = Path(record_identity.get("path", ""))
        if not artifact_matches(record_identity, record_path):
            raise ValueError(f"chunk record changed since {ledger_path}: {record_path}")
        record = validate_invocation_record_current(record_path, common)
        if (
            chunk.get("name") != record.get("name")
            or record.get("timestep_offset") != offset
            or record.get("steps_per_window") != steps
            or record.get("worlds") != len(windows)
            or record.get("ranks_per_window") != common["ranks_per_window"]
        ):
            raise ValueError(f"chunk record timeline differs from {ledger_path}")
        if (
            record.get("window_order") != expected_order
            or set(record["start_inputs"]) != set(expected_order)
            or set(record["outputs"]) != set(expected_order)
        ):
            raise ValueError(f"chunk window set differs from {ledger_path}")
        for window in windows:
            start = record["start_inputs"][window.tag]
            if previous_outputs is None:
                if not artifact_matches(start, expected_start_data[window.tag]):
                    raise ValueError(f"stage start changed for {window.tag}")
            elif start != previous_outputs[window.tag]["data"]:
                raise ValueError(f"chunk dependency chain broke for {window.tag}")
            output_record = record["outputs"][window.tag]
            required_outputs = ["data", "restart", "colvars"]
            if trajectory_frequency > 0:
                required_outputs.append("trajectory")
            if not isinstance(output_record, dict) or not all(
                kind in output_record for kind in required_outputs
            ):
                raise ValueError(f"chunk outputs are incomplete for {window.tag}")
            colvars_by_window[window.tag].append(output_record["colvars"])
            if "trajectory" in output_record:
                trajectories_by_window[window.tag].append(output_record["trajectory"])
        previous_outputs = record["outputs"]
        summed_wall_seconds += float(record["wall_seconds"])
        offset += steps
    if offset != total_steps or previous_outputs is None:
        raise ValueError(f"stage ledger covers {offset}, expected {total_steps} steps")
    if ledger.get("outputs") != previous_outputs:
        raise ValueError(
            f"stage final outputs differ from the final chunk in {ledger_path}"
        )
    if ledger.get("colvars_by_window") != colvars_by_window:
        raise ValueError("stage Colvars ledger differs from its chunk records")
    if ledger.get("trajectories_by_window") != trajectories_by_window:
        raise ValueError("stage trajectory ledger differs from its chunk records")
    try:
        recorded_wall_seconds = float(ledger["summed_chunk_wall_seconds"])
        recorded_throughput = float(ledger["aggregate_window_steps_per_second"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("stage performance aggregate is malformed") from error
    if not math.isclose(
        recorded_wall_seconds,
        summed_wall_seconds,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise ValueError("stage wall-time aggregate differs from its chunk records")
    aggregate_steps = len(windows) * total_steps
    if ledger.get("aggregate_window_steps") != aggregate_steps:
        raise ValueError("stage aggregate step count is inconsistent")
    expected_throughput = aggregate_steps / summed_wall_seconds
    if not math.isclose(
        recorded_throughput,
        expected_throughput,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("stage aggregate throughput is inconsistent")
    reaction_tolerance, angle_tolerance = checkpoint_boundary_tolerances(manifest)
    for window in windows:
        if not artifact_matches(
            previous_outputs[window.tag]["data"], expected_final_data[window.tag]
        ):
            raise ValueError(f"stage final state changed for {window.tag}")
        series = [Path(identity["path"]) for identity in colvars_by_window[window.tag]]
        merged = merge_colvars(series, reaction_tolerance, angle_tolerance)
        if merged[-1]["step"] != total_steps:
            raise ValueError(f"stage Colvars series is incomplete for {window.tag}")
    return ledger


def require_seed_records(
    output: Path,
    windows: Sequence[Window],
    anchor: Window,
    *,
    anchor_ledger: dict[str, Any],
    seed_steps: int,
    common: dict[str, Any],
) -> dict[str, Path]:
    """Validate both ordered seed branches and every parent/output hash edge."""
    anchor_index = windows.index(anchor)
    lower = list(reversed(windows[:anchor_index]))
    upper = list(windows[anchor_index + 1 :])
    rounds = max(len(lower), len(upper))
    expected_record_paths = [
        output / "records" / f"seed-round-{index + 1:02d}.json"
        for index in range(rounds)
    ]
    actual_record_paths = sorted((output / "records").glob("seed-round-*.json"))
    if actual_record_paths != expected_record_paths:
        raise ValueError(
            "seed record set differs from the exact two-branch protocol: "
            f"expected {expected_record_paths}, found {actual_record_paths}"
        )

    anchor_identity = anchor_ledger["outputs"][anchor.tag]["data"]
    previous: dict[str, dict[str, str]] = {
        "lower": anchor_identity,
        "upper": anchor_identity,
    }
    accepted: dict[str, Path] = {anchor.tag: state_output(output, "anchor", anchor)}
    for round_index, record_path in enumerate(expected_record_paths):
        expected_windows: list[tuple[str, Window]] = []
        if round_index < len(lower):
            expected_windows.append(("lower", lower[round_index]))
        if round_index < len(upper):
            expected_windows.append(("upper", upper[round_index]))
        expected_tags = [window.tag for _, window in expected_windows]
        record = validate_invocation_record_current(record_path, common)
        if (
            record.get("name") != f"seed-round-{round_index + 1:02d}"
            or record.get("steps_per_window") != seed_steps
            or record.get("timestep_offset") != 0
            or record.get("worlds") != len(expected_windows)
            or record.get("ranks_per_window") != common["ranks_per_window"]
            or record.get("window_order") != expected_tags
            or set(record.get("start_inputs", {})) != set(expected_tags)
            or set(record.get("outputs", {})) != set(expected_tags)
        ):
            raise ValueError(f"seed round protocol mismatch in {record_path}")
        for branch, window in expected_windows:
            if record["start_inputs"][window.tag] != previous[branch]:
                raise ValueError(
                    f"seed branch parent changed before {window.tag} in {record_path}"
                )
            expected_output = state_output(output, "seeds", window)
            output_record = record["outputs"][window.tag]
            if not artifact_matches(output_record.get("data"), expected_output):
                raise ValueError(f"seed output changed for {window.tag}")
            if output_record.get("seed_acceptance") is not True:
                raise ValueError(f"seed output was not accepted for {window.tag}")
            previous[branch] = output_record["data"]
            accepted[window.tag] = expected_output
    if set(accepted) != {window.tag for window in windows}:
        raise ValueError("seed branch DAG did not produce every requested window")
    return accepted


def inspect_dangerous_builds(
    log_directory: Path, expected_worlds: int
) -> dict[str, int]:
    """Require one completed dangerous-build summary per partition log."""
    results: dict[str, int] = {}
    for path in sorted(log_directory.glob("log.lammps*")):
        matches = DANGEROUS_BUILDS.findall(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if matches:
            results[path.name] = int(matches[-1])
    if not results:
        raise ValueError(f"no Dangerous builds summary found under {log_directory}")
    if len(results) != expected_worlds:
        raise ValueError(
            f"found {len(results)} completed partition logs under {log_directory}; "
            f"expected {expected_worlds}"
        )
    return results


def archive_stale_invocation_artifacts(
    *,
    name: str,
    output: Path,
    record_path: Path,
    log_directory: Path,
    run_windows: Sequence[RunWindow],
) -> Path | None:
    """Move an invalid or interrupted attempt aside before reusing its paths."""
    populated_outputs = [
        item
        for item in run_windows
        if item.output_directory.is_dir() and any(item.output_directory.iterdir())
    ]
    populated_logs = log_directory.is_dir() and any(log_directory.iterdir())
    if not record_path.exists() and not populated_logs and not populated_outputs:
        return None

    archived = output / "superseded" / f"{name}-{time.time_ns()}"
    archived.mkdir(parents=True)
    if record_path.exists():
        record_path.replace(archived / "record.json")
    if populated_logs:
        log_directory.replace(archived / "logs")
        log_directory.mkdir(parents=True)
    for item in populated_outputs:
        destination = archived / "outputs" / item.window.tag
        destination.parent.mkdir(parents=True, exist_ok=True)
        item.output_directory.replace(destination)
        item.output_directory.mkdir(parents=True)
    print(f"archive: stale {name} artifacts moved to {archived}")
    return archived


def build_lammps_command(
    *,
    lammps: Path,
    mpi_launcher: Path,
    mpi_args: Sequence[str],
    worlds: int,
    ranks_per_window: int,
    log_directory: Path,
    input_path: Path,
    lammps_args: Sequence[str] = (),
) -> list[str]:
    """Build valid LAMMPS logging flags for serial, MPI, and partition runs."""
    total_ranks = worlds * ranks_per_window
    if total_ranks == 1:
        return [
            str(lammps.resolve()),
            *lammps_args,
            "-log",
            str(log_directory / "log.lammps"),
            "-screen",
            "none",
            "-in",
            str(input_path),
        ]

    command = [
        str(mpi_launcher),
        *mpi_args,
        "-n",
        str(total_ranks),
        str(lammps.resolve()),
        *lammps_args,
    ]
    if worlds > 1:
        command.extend(
            [
                "-partition",
                f"{worlds}x{ranks_per_window}",
                "-plog",
                str(log_directory / "log.lammps"),
                "-pscreen",
                "none",
            ]
        )
    else:
        command.extend(
            [
                "-log",
                str(log_directory / "log.lammps"),
                "-screen",
                "none",
            ]
        )
    command.extend(["-in", str(input_path)])
    return command


def run_invocation(
    *,
    name: str,
    manifest: dict[str, Any],
    tutorial: Path,
    output: Path,
    lammps: Path,
    plugin: Path,
    xtbloom_library: Path,
    manifest_path: Path,
    run_windows: Sequence[RunWindow],
    steps: int,
    timestep_offset: int = 0,
    trajectory_frequency: int,
    ranks_per_window: int,
    mpiexec: Path,
    mpi_args: Sequence[str],
    library_dirs: Sequence[Path],
    cuda_visible_devices: str,
    require_seed_acceptance: bool,
    mode: str = "qmmm",
    deepmd_plugin: Path | None = None,
    deepmd_models: Sequence[Path] = (),
    model_deviation_frequency: int = 0,
    dpa4c_models_qualified: bool = False,
    allow_unqualified_dpa4c_models: bool = False,
) -> Path:
    """Run one synchronized batch and atomically publish its evidence record."""
    validate_execution_policy(
        mode=mode,
        deepmd_plugin=deepmd_plugin,
        deepmd_models=deepmd_models,
        model_deviation_frequency=model_deviation_frequency,
        dpa4c_models_qualified=dpa4c_models_qualified,
        allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
    )
    current_execution = execution_record(
        mode=mode,
        model_deviation_frequency=model_deviation_frequency,
        dpa4c_models_qualified=dpa4c_models_qualified,
        allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
    )
    record_path = output / "records" / f"{name}.json"
    input_path = output / "generated/inputs" / f"{name}.in"
    write_generated(
        input_path,
        render_lammps_input(
            manifest,
            tutorial,
            plugin,
            run_windows,
            steps=steps,
            timestep_offset=timestep_offset,
            trajectory_frequency=trajectory_frequency,
            mode=mode,
            deepmd_plugin=deepmd_plugin,
            deepmd_models=deepmd_models,
            model_deviation_frequency=model_deviation_frequency,
        ),
    )
    log_directory = output / "logs" / name
    log_directory.mkdir(parents=True, exist_ok=True)
    launcher_log = log_directory / "launcher.log"

    worlds = len(run_windows)
    mpi_launcher = resolve_executable(mpiexec)
    kokkos_device = not (
        mode == "qmmm-dpa4c" and model_deviation_frequency > 0
    )
    command = build_lammps_command(
        lammps=lammps,
        mpi_launcher=mpi_launcher,
        mpi_args=mpi_args,
        worlds=worlds,
        ranks_per_window=ranks_per_window,
        log_directory=log_directory,
        input_path=input_path,
        # The generated production input names the complete Kokkos style
        # chain. Initialize it before read_data and pin the half-list/Newton
        # contract required by the shared classical and DPA4c brokers.
        lammps_args=(
            (
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
            )
            if kokkos_device
            else ()
        ),
    )

    environment, selected_environment = build_runtime_environment(
        xtbloom_library,
        library_dirs,
        cuda_visible_devices,
        uses_deepmd=mode == "qmmm-dpa4c",
    )
    loaded_xtbloom = verify_loaded_xtbloom(plugin, xtbloom_library, environment)
    runtime_before = runtime_record(
        lammps,
        plugin,
        xtbloom_library,
        mpi_launcher,
        environment,
        deepmd_plugin=deepmd_plugin,
        deepmd_models=deepmd_models,
    )
    project_before = project_record(manifest_path, output)
    input_before = {"path": str(input_path.resolve()), "sha256": sha256(input_path)}
    start_inputs_before = {
        item.window.tag: {
            "path": str(item.start_data.resolve()),
            "sha256": sha256(item.start_data),
        }
        for item in run_windows
    }
    colvars_configs_before = {
        item.window.tag: {
            "path": str(item.colvars_config.resolve()),
            "sha256": sha256(item.colvars_config),
        }
        for item in run_windows
    }
    if record_is_resumable(
        record_path,
        run_windows,
        input_path=input_path,
        steps=steps,
        timestep_offset=timestep_offset,
        trajectory_frequency=trajectory_frequency,
        ranks_per_window=ranks_per_window,
        lammps=lammps,
        plugin=plugin,
        xtbloom_library=xtbloom_library,
        mpiexec=mpi_launcher,
        runner_path=Path(__file__).resolve(),
        loaded_xtbloom=loaded_xtbloom,
        plugin_cmake_cache=(
            plugin.resolve().parent / "CMakeCache.txt"
            if (plugin.resolve().parent / "CMakeCache.txt").is_file()
            else None
        ),
        selected_environment=selected_environment,
        manifest_path=manifest_path,
        provenance_path=output / "provenance.json",
        mode=mode,
        deepmd_plugin=deepmd_plugin,
        deepmd_models=deepmd_models,
        model_deviation_frequency=model_deviation_frequency,
        dpa4c_models_qualified=dpa4c_models_qualified,
        allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
    ):
        print(f"resume: {name} already passed with an unchanged dependency chain")
        return record_path

    archive_stale_invocation_artifacts(
        name=name,
        output=output,
        record_path=record_path,
        log_directory=log_directory,
        run_windows=run_windows,
    )

    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    print(f"run: {name}: {shlex.join(command)}")
    with launcher_log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    wall_seconds = time.monotonic() - started

    record: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "status": "failed",
        "started_utc": started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "steps_per_window": steps,
        "timestep_offset": timestep_offset,
        "worlds": worlds,
        "window_order": [item.window.tag for item in run_windows],
        "ranks_per_window": ranks_per_window,
        "aggregate_window_steps": worlds * steps,
        "aggregate_window_steps_per_second": (worlds * steps / wall_seconds)
        if wall_seconds
        else None,
        "command": command,
        "execution": current_execution,
        "selected_environment": selected_environment,
        "runtime": runtime_before,
        "loaded_xtbloom": loaded_xtbloom,
        "project": project_before,
        "input": input_before,
        "launcher_log": {"path": str(launcher_log), "sha256": sha256(launcher_log)},
        "returncode": process.returncode,
        "start_inputs": start_inputs_before,
        "colvars_configs": colvars_configs_before,
        "outputs": {},
    }

    try:
        if process.returncode != 0:
            tail = launcher_log.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(f"LAMMPS returned {process.returncode}:\n{tail}")
        changed_inputs = [
            label
            for label, identity, artifact in (
                ("LAMMPS executable", runtime_before["lammps"], lammps),
                ("plugin", runtime_before["plugin"], plugin),
                ("xTBloom library", runtime_before["xtbloom"], xtbloom_library),
                ("MPI launcher", runtime_before["mpiexec"], mpi_launcher),
                ("workload runner", project_before["runner"], Path(__file__).resolve()),
                ("generated input", input_before, input_path),
                ("workload manifest", project_before["manifest"], manifest_path),
                (
                    "source provenance",
                    project_before["provenance"],
                    output / "provenance.json",
                ),
            )
            if not artifact_matches(identity, artifact)
        ]
        if mode == "qmmm-dpa4c":
            assert deepmd_plugin is not None
            changed_inputs.extend(
                ["DeePMD plugin"]
                if not artifact_matches(
                    runtime_before.get("deepmd_plugin"), deepmd_plugin
                )
                else []
            )
            recorded_models = runtime_before.get("models", [])
            changed_inputs.extend(
                f"DPA4c model {index}"
                for index, (identity, model) in enumerate(
                    zip(recorded_models, deepmd_models, strict=True)
                )
                if not artifact_matches(identity, model)
            )
        cmake_cache = plugin.resolve().parent / "CMakeCache.txt"
        if "plugin_cmake_cache" in runtime_before and not artifact_matches(
            runtime_before["plugin_cmake_cache"], cmake_cache
        ):
            changed_inputs.append("plugin CMake cache")
        changed_inputs.extend(
            f"start input {item.window.tag}"
            for item in run_windows
            if not artifact_matches(
                start_inputs_before[item.window.tag], item.start_data
            )
        )
        changed_inputs.extend(
            f"Colvars config {item.window.tag}"
            for item in run_windows
            if not artifact_matches(
                colvars_configs_before[item.window.tag], item.colvars_config
            )
        )
        if changed_inputs:
            raise ValueError(
                "runtime inputs changed while LAMMPS was executing: "
                + ", ".join(changed_inputs)
            )
        loaded_after = verify_loaded_xtbloom(plugin, xtbloom_library, environment)
        if loaded_after != loaded_xtbloom:
            raise ValueError(
                "xTBloom loader resolution changed while LAMMPS was running"
            )
        dangerous = inspect_dangerous_builds(log_directory, worlds)
        if manifest["protocol"]["seed_acceptance"][
            "require_zero_dangerous_builds"
        ] and any(value != 0 for value in dangerous.values()):
            raise ValueError(f"dangerous neighbor builds were reported: {dangerous}")
        record["dangerous_builds"] = dangerous
        record["lammps_logs"] = {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(log_directory.glob("log.lammps*"))
            if path.is_file()
        }

        acceptance = manifest["protocol"]["seed_acceptance"]
        angle_center = float(manifest["umbrella"]["attack_angle"]["center_degree"])
        colvars_frequency = int(manifest["dynamics"]["colvars_frequency_steps"])
        completed_timestep = timestep_offset + steps
        expected_colvars_step = (
            completed_timestep if completed_timestep % colvars_frequency == 0 else None
        )
        rejected_seeds: list[str] = []
        for item in run_windows:
            colvars_path = Path(str(item.colvars_prefix) + ".colvars.traj")
            values = parse_colvars(
                colvars_path, expected_final_step=expected_colvars_step
            )
            reaction_error = abs(values["reaction_coordinate"] - item.window.center)
            angle_error = abs(values["attack_angle"] - angle_center)
            accepted = (
                reaction_error
                <= acceptance["max_abs_reaction_coordinate_error_angstrom"]
                and angle_error <= acceptance["max_abs_attack_angle_error_degree"]
            )
            if not item.final_data.is_file() or not item.final_restart.is_file():
                raise ValueError(
                    f"LAMMPS did not publish final state for {item.window.tag}"
                )
            record["outputs"][item.window.tag] = {
                "center_angstrom": item.window.center,
                "data": {
                    "path": str(item.final_data),
                    "sha256": sha256(item.final_data),
                },
                "restart": {
                    "path": str(item.final_restart),
                    "sha256": sha256(item.final_restart),
                },
                "colvars": {"path": str(colvars_path), "sha256": sha256(colvars_path)},
                "final_values": values,
                "reaction_coordinate_error_angstrom": reaction_error,
                "attack_angle_error_degree": angle_error,
                "seed_acceptance": accepted,
            }
            if trajectory_frequency > 0:
                if not item.trajectory.is_file():
                    raise ValueError(
                        f"LAMMPS did not publish trajectory for {item.window.tag}"
                    )
                record["outputs"][item.window.tag]["trajectory"] = {
                    "path": str(item.trajectory),
                    "sha256": sha256(item.trajectory),
                }
            if mode == "qmmm-dpa4c" and model_deviation_frequency > 0:
                if not item.model_deviation.is_file():
                    raise ValueError(
                        "LAMMPS did not publish model deviation for "
                        f"{item.window.tag}"
                    )
                record["outputs"][item.window.tag]["model_deviation"] = {
                    "path": str(item.model_deviation),
                    "sha256": sha256(item.model_deviation),
                }
            if require_seed_acceptance and not accepted:
                rejected_seeds.append(
                    f"{item.window.tag}: reaction error {reaction_error:.6f} "
                    f"Angstrom, angle error {angle_error:.6f} degree"
                )
        if rejected_seeds:
            raise ValueError("seed acceptance failed: " + "; ".join(rejected_seeds))
        record["status"] = "passed"
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        record["error"] = str(error)
        write_json_atomic(record_path, record)
        raise

    write_json_atomic(record_path, record)
    print(
        f"pass: {name}: {record['aggregate_window_steps_per_second']:.3f} "
        "aggregate window-steps/s"
    )
    return record_path


def state_output(output: Path, stage: str, window: Window) -> Path:
    """Return the deterministic state path produced by a completed stage."""
    return output / "states" / stage / window.tag / f"{window.tag}.data"


def chunk_sizes(total_steps: int, maximum_chunk_steps: int) -> list[int]:
    """Partition a stage exactly, retaining a final nonzero remainder."""
    if total_steps <= 0 or maximum_chunk_steps <= 0:
        raise ValueError("stage and chunk step counts must be positive")
    full_chunks, remainder = divmod(total_steps, maximum_chunk_steps)
    chunks = [maximum_chunk_steps] * full_chunks
    if remainder:
        chunks.append(remainder)
    return chunks


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete JSON checkpoint with fsync and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stage_record_path(output: Path, stage: str) -> Path:
    """Return the only accepted native or explicitly adopted stage ledger."""
    return output / "records" / f"{stage}-complete.json"


def run_chunked_stage(
    *,
    stage: str,
    manifest: dict[str, Any],
    windows: Sequence[Window],
    start_data: dict[str, Path],
    stage_root: Path,
    total_steps: int,
    maximum_chunk_steps: int,
    seed_stage_code: int,
    trial: int,
    trajectory_frequency: int,
    require_final_acceptance: bool,
    common: dict[str, Any],
) -> Path:
    """Run a long synchronized stage as an exact, resumable chunk chain."""
    sizes = chunk_sizes(total_steps, maximum_chunk_steps)
    ledger_path = common["output"] / "records" / f"{stage}-complete.json"
    expected_final_data = {
        window.tag: stage_root / window.tag / f"{window.tag}.data" for window in windows
    }
    if ledger_path.is_file():
        try:
            require_completed_stage(
                ledger_path,
                stage=stage,
                windows=windows,
                expected_start_data=start_data,
                expected_final_data=expected_final_data,
                total_steps=total_steps,
                maximum_chunk_steps=maximum_chunk_steps,
                trajectory_frequency=trajectory_frequency,
                common=common,
            )
        except (TypeError, ValueError) as error:
            print(f"resume rejected: {stage}: {error}")
        else:
            print(f"resume: {stage} complete ledger and dependency DAG are unchanged")
            return ledger_path

    current = dict(start_data)
    chunk_records: list[dict[str, Any]] = []
    summed_chunk_wall_seconds = 0.0
    colvars_by_window: dict[str, list[dict[str, str]]] = {
        window.tag: [] for window in windows
    }
    trajectories_by_window: dict[str, list[dict[str, str]]] = {
        window.tag: [] for window in windows
    }
    offset = 0

    for chunk_index, steps in enumerate(sizes):
        final_chunk = chunk_index + 1 == len(sizes)
        chunk_name = f"{stage}-chunk-{chunk_index + 1:03d}-of-{len(sizes):03d}"
        chunk_root = (
            stage_root
            if final_chunk
            else stage_root / "checkpoints" / f"chunk-{chunk_index + 1:03d}"
        )
        batch = [
            RunWindow(
                window,
                current[window.tag],
                chunk_root / window.tag,
                common["output"],
                seed_for(
                    manifest,
                    window,
                    seed_stage_code,
                    trial * 1000 + chunk_index,
                ),
            )
            for window in windows
        ]
        record_path = run_invocation(
            name=chunk_name,
            run_windows=batch,
            steps=steps,
            timestep_offset=offset,
            trajectory_frequency=trajectory_frequency,
            require_seed_acceptance=require_final_acceptance and final_chunk,
            **common,
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        summed_chunk_wall_seconds += float(record["wall_seconds"])
        chunk_records.append(
            {
                "name": chunk_name,
                "timestep_offset": offset,
                "steps_per_window": steps,
                "record": {"path": str(record_path), "sha256": sha256(record_path)},
            }
        )
        for item in batch:
            output_record = record["outputs"][item.window.tag]
            colvars_by_window[item.window.tag].append(output_record["colvars"])
            if "trajectory" in output_record:
                trajectories_by_window[item.window.tag].append(
                    output_record["trajectory"]
                )
            current[item.window.tag] = item.final_data
        offset += steps

    if offset != total_steps:
        raise RuntimeError(
            f"chunk chain covered {offset}, expected {total_steps} steps"
        )
    final_record = json.loads(
        Path(chunk_records[-1]["record"]["path"]).read_text(encoding="utf-8")
    )
    ledger = {
        "schema_version": 1,
        "stage": stage,
        "status": "passed",
        "qualification": "native-chunked",
        "window_order": [window.tag for window in windows],
        "total_steps_per_window": total_steps,
        "maximum_chunk_steps": maximum_chunk_steps,
        "chunk_count": len(sizes),
        "chunks": chunk_records,
        "summed_chunk_wall_seconds": summed_chunk_wall_seconds,
        "aggregate_window_steps": len(windows) * total_steps,
        "aggregate_window_steps_per_second": (
            len(windows) * total_steps / summed_chunk_wall_seconds
            if summed_chunk_wall_seconds
            else None
        ),
        "outputs": final_record["outputs"],
        "colvars_by_window": colvars_by_window,
        "trajectories_by_window": trajectories_by_window,
        "langevin_seed_policy": "distinct deterministic seed per chunk",
        "series_merge_policy": series_merge_policy(manifest),
        "note": (
            "Chunk boundaries restart the Langevin random stream with a distinct seed; "
            "the resulting thermostat process is valid but not bitwise identical to one "
            "monolithic LAMMPS run."
        ),
    }
    write_json_atomic(ledger_path, ledger)
    return ledger_path


def run_stage(
    arguments: argparse.Namespace, manifest: dict[str, Any], windows: list[Window]
) -> None:
    """Execute one resumable scientific stage or an ordered stage prefix."""
    output = arguments.output.resolve()
    tutorial = arguments.tutorial.resolve()
    initial_center = int(
        manifest["umbrella"]["available_initial_center_tenths_angstrom"]
    )
    by_center = {window.center_tenths: window for window in windows}
    anchor_window = by_center[initial_center]
    initial_data = tutorial / "lammps/ETP_ETH.data"
    protocol = manifest["protocol"]
    trajectory_frequency = int(manifest["dynamics"]["trajectory_frequency_steps"])
    mode = getattr(arguments, "mode", "qmmm")
    deepmd_plugin = getattr(arguments, "deepmd_plugin", None)
    deepmd_models = tuple(getattr(arguments, "deepmd_model", ()))
    model_deviation_frequency = int(
        getattr(arguments, "model_deviation_frequency", 0)
    )
    dpa4c_models_qualified = bool(
        getattr(arguments, "dpa4c_models_qualified", False)
    )
    allow_unqualified_dpa4c_models = bool(
        getattr(arguments, "allow_unqualified_dpa4c_models", False)
    )
    validate_execution_policy(
        mode=mode,
        deepmd_plugin=deepmd_plugin,
        deepmd_models=deepmd_models,
        model_deviation_frequency=model_deviation_frequency,
        dpa4c_models_qualified=dpa4c_models_qualified,
        allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
    )

    common = {
        "manifest": manifest,
        "tutorial": tutorial,
        "output": output,
        "lammps": arguments.lammps,
        "plugin": arguments.plugin,
        "xtbloom_library": arguments.xtbloom_library,
        "manifest_path": arguments.manifest,
        "ranks_per_window": arguments.ranks_per_window,
        "mpiexec": arguments.mpiexec,
        "mpi_args": arguments.mpi_arg,
        "library_dirs": arguments.library_dir,
        "cuda_visible_devices": arguments.cuda_visible_devices,
        "mode": mode,
        "deepmd_plugin": deepmd_plugin,
        "deepmd_models": deepmd_models,
        "model_deviation_frequency": model_deviation_frequency,
        "dpa4c_models_qualified": dpa4c_models_qualified,
        "allow_unqualified_dpa4c_models": allow_unqualified_dpa4c_models,
    }

    def accepted_anchor() -> dict[str, Any]:
        return require_completed_stage(
            stage_record_path(output, "anchor"),
            stage="anchor",
            windows=[anchor_window],
            expected_start_data={anchor_window.tag: initial_data},
            expected_final_data={
                anchor_window.tag: state_output(output, "anchor", anchor_window)
            },
            total_steps=int(protocol["anchor_relaxation_steps"]),
            maximum_chunk_steps=arguments.chunk_steps,
            trajectory_frequency=trajectory_frequency,
            common=common,
        )

    def accepted_seed_starts(anchor_ledger: dict[str, Any]) -> dict[str, Path]:
        return require_seed_records(
            output,
            windows,
            anchor_window,
            anchor_ledger=anchor_ledger,
            seed_steps=int(protocol["seed_walk_steps_per_center"]),
            common=common,
        )

    requested = arguments.stage
    stages = ["smoke", "batch-smoke", "anchor", "seeds", "equilibrate", "production"]
    if requested.startswith("through-"):
        end = requested.removeprefix("through-")
        selected = stages[: stages.index(end) + 1]
    else:
        selected = [requested]

    if "smoke" in selected:
        smoke = RunWindow(
            anchor_window,
            initial_data,
            output / "diagnostic/smoke" / anchor_window.tag,
            output,
            seed_for(manifest, anchor_window, 1),
        )
        run_invocation(
            name="smoke",
            run_windows=[smoke],
            # Use the same override for one-window and batched diagnostics so
            # their startup-amortized throughput rows can be compared at an
            # identical step count.
            steps=arguments.smoke_steps or int(protocol["smoke_steps"]),
            trajectory_frequency=0,
            require_seed_acceptance=False,
            **common,
        )

    if "batch-smoke" in selected:
        count = arguments.smoke_window_count
        if count < 2 or count > len(windows):
            raise ValueError("--smoke-window-count must be between 2 and 48")
        first = max(0, min(anchor_window.index - count // 2, len(windows) - count))
        batch = [
            RunWindow(
                window,
                initial_data,
                output / f"diagnostic/batch-smoke-{count}" / window.tag,
                output,
                seed_for(manifest, window, 6),
            )
            for window in windows[first : first + count]
        ]
        run_invocation(
            name=f"batch-smoke-{count}",
            run_windows=batch,
            steps=arguments.smoke_steps or int(protocol["smoke_steps"]),
            trajectory_frequency=0,
            require_seed_acceptance=False,
            **common,
        )

    if "anchor" in selected:
        run_chunked_stage(
            stage="anchor",
            manifest=manifest,
            windows=[anchor_window],
            start_data={anchor_window.tag: initial_data},
            stage_root=output / "states/anchor",
            total_steps=int(protocol["anchor_relaxation_steps"]),
            maximum_chunk_steps=arguments.chunk_steps,
            seed_stage_code=2,
            trial=0,
            trajectory_frequency=trajectory_frequency,
            require_final_acceptance=True,
            common=common,
        )
        accepted_anchor()

    if "seeds" in selected:
        anchor_ledger = accepted_anchor()
        anchor_data = state_output(output, "anchor", anchor_window)
        lower = list(
            range(
                initial_center - 1,
                int(manifest["umbrella"]["start_tenths_angstrom"]) - 1,
                -1,
            )
        )
        upper = list(
            range(
                initial_center + 1,
                int(manifest["umbrella"]["stop_tenths_angstrom"]) + 1,
            )
        )
        previous = {"lower": anchor_data, "upper": anchor_data}
        rounds = max(len(lower), len(upper))
        for round_index in range(rounds):
            batch: list[RunWindow] = []
            for branch, centers in (("lower", lower), ("upper", upper)):
                if round_index >= len(centers):
                    continue
                window = by_center[centers[round_index]]
                item = RunWindow(
                    window,
                    previous[branch],
                    output / "states/seeds" / window.tag,
                    output,
                    seed_for(manifest, window, 3),
                    colvars_profile="seed",
                )
                batch.append(item)
                previous[branch] = item.final_data
            run_invocation(
                name=f"seed-round-{round_index + 1:02d}",
                run_windows=batch,
                steps=int(protocol["seed_walk_steps_per_center"]),
                trajectory_frequency=0,
                require_seed_acceptance=True,
                **common,
            )
        accepted_seed_starts(anchor_ledger)

    if "equilibrate" in selected:
        starts = accepted_seed_starts(accepted_anchor())
        equilibration_ledger = run_chunked_stage(
            stage="equilibrate",
            manifest=manifest,
            windows=windows,
            start_data=starts,
            stage_root=output / "states/equilibrated",
            total_steps=int(protocol["equilibration_steps_per_window"]),
            maximum_chunk_steps=arguments.chunk_steps,
            seed_stage_code=4,
            trial=0,
            trajectory_frequency=trajectory_frequency,
            require_final_acceptance=False,
            common=common,
        )
        require_completed_stage(
            equilibration_ledger,
            stage="equilibrate",
            windows=windows,
            expected_start_data=starts,
            expected_final_data={
                window.tag: state_output(output, "equilibrated", window)
                for window in windows
            },
            total_steps=int(protocol["equilibration_steps_per_window"]),
            maximum_chunk_steps=arguments.chunk_steps,
            trajectory_frequency=trajectory_frequency,
            common=common,
        )

    if "production" in selected:
        seed_starts = accepted_seed_starts(accepted_anchor())
        equilibrated = {
            window.tag: state_output(output, "equilibrated", window)
            for window in windows
        }
        require_completed_stage(
            stage_record_path(output, "equilibrate"),
            stage="equilibrate",
            windows=windows,
            expected_start_data=seed_starts,
            expected_final_data=equilibrated,
            total_steps=int(protocol["equilibration_steps_per_window"]),
            maximum_chunk_steps=arguments.chunk_steps,
            trajectory_frequency=trajectory_frequency,
            common=common,
        )
        trials = (
            arguments.trial
            if arguments.trial
            else list(range(int(protocol["production_trials"])))
        )
        for trial in trials:
            if trial < 0 or trial >= int(protocol["production_trials"]):
                raise ValueError(
                    f"production trial is outside the manifest range: {trial}"
                )
            production_ledger = run_chunked_stage(
                stage=f"production-trial-{trial}",
                manifest=manifest,
                windows=windows,
                start_data=equilibrated,
                stage_root=output / f"production/trial-{trial}",
                total_steps=int(protocol["production_steps_per_window"]),
                maximum_chunk_steps=arguments.chunk_steps,
                seed_stage_code=5,
                trial=trial,
                trajectory_frequency=trajectory_frequency,
                require_final_acceptance=False,
                common=common,
            )
            require_completed_stage(
                production_ledger,
                stage=f"production-trial-{trial}",
                windows=windows,
                expected_start_data=equilibrated,
                expected_final_data={
                    window.tag: output
                    / f"production/trial-{trial}"
                    / window.tag
                    / f"{window.tag}.data"
                    for window in windows
                },
                total_steps=int(protocol["production_steps_per_window"]),
                maximum_chunk_steps=arguments.chunk_steps,
                trajectory_frequency=trajectory_frequency,
                common=common,
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the command line shared by humans, tests, and batch scripts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("verify", "prepare"):
        command = subparsers.add_parser(name)
        command.add_argument("--tutorial", type=Path, required=True)
        command.add_argument("--allow-unqualified-source", action="store_true")
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--recover-stale-lock", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--tutorial", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--allow-unqualified-source", action="store_true")
    run.add_argument("--recover-stale-lock", action="store_true")
    run.add_argument("--lammps", type=Path, required=True)
    run.add_argument("--plugin", type=Path, required=True)
    run.add_argument("--xtbloom-library", type=Path, required=True)
    run.add_argument(
        "--mode",
        choices=("qmmm", "qmmm-dpa4c"),
        default="qmmm",
        help=(
            "run batched xTB QM/MM alone or overlay a compact DPA4c DPRc "
            "model; the default preserves the historical qmmm workflow"
        ),
    )
    run.add_argument("--deepmd-plugin", type=Path)
    run.add_argument("--deepmd-model", type=Path, action="append", default=[])
    run.add_argument(
        "--model-deviation-frequency",
        type=int,
        default=0,
        metavar="STEPS",
        help=(
            "evaluate one primary DPA4c model every step when zero; a positive "
            "stride requires four models and evaluates the other three only "
            "on deviation steps"
        ),
    )
    run.add_argument(
        "--dpa4c-models-qualified",
        action="store_true",
        help=(
            "assert that every supplied model passed the xTB-based DPRc "
            "scientific qualification gates"
        ),
    )
    run.add_argument(
        "--allow-unqualified-dpa4c-models",
        action="store_true",
        help=(
            "allow non-production models only for explicit private diagnostic "
            "runs; the choice is recorded in every checkpoint"
        ),
    )
    run.add_argument(
        "--mpiexec", type=Path, default=Path(shutil.which("mpiexec") or "mpiexec")
    )
    run.add_argument("--mpi-arg", action="append", default=[])
    run.add_argument("--library-dir", type=Path, action="append", default=[])
    run.add_argument("--cuda-visible-devices", default="0")
    run.add_argument("--ranks-per-window", type=int, default=1)
    run.add_argument(
        "--chunk-steps",
        type=int,
        default=5000,
        help=(
            "maximum steps per resumable anchor/equilibration/production chunk; "
            "tune after measured batch throughput"
        ),
    )
    run.add_argument(
        "--stage",
        choices=(
            "smoke",
            "batch-smoke",
            "anchor",
            "seeds",
            "equilibrate",
            "production",
            "through-anchor",
            "through-seeds",
            "through-equilibrate",
            "through-production",
        ),
        required=True,
    )
    run.add_argument("--trial", type=int, action="append")
    run.add_argument("--smoke-window-count", type=int, default=2)
    run.add_argument(
        "--smoke-steps",
        type=int,
        help="diagnostic smoke length; defaults to the manifest smoke length",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        manifest = load_manifest(arguments.manifest)
        source = verify_source(
            arguments.tutorial,
            manifest,
            allow_unqualified_source=arguments.allow_unqualified_source,
        )
        if arguments.command == "verify":
            print(json.dumps(source, indent=2, sort_keys=True))
            return 0

        if arguments.command == "run":
            validate_execution_policy(
                mode=arguments.mode,
                deepmd_plugin=arguments.deepmd_plugin,
                deepmd_models=arguments.deepmd_model,
                model_deviation_frequency=arguments.model_deviation_frequency,
                dpa4c_models_qualified=arguments.dpa4c_models_qualified,
                allow_unqualified_dpa4c_models=(
                    arguments.allow_unqualified_dpa4c_models
                ),
            )

        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        with WorkspaceLock(output, recover_stale=arguments.recover_stale_lock):
            windows = prepare_workspace(
                output,
                arguments.tutorial,
                arguments.manifest,
                manifest,
                source,
            )
            print(
                f"prepared {len(windows)} windows in {output} "
                f"({source['qualification']})"
            )
            if arguments.command == "run":
                if arguments.ranks_per_window < 1:
                    raise ValueError("--ranks-per-window must be positive")
                if arguments.chunk_steps < 1:
                    raise ValueError("--chunk-steps must be positive")
                resolve_executable(arguments.mpiexec)
                for path in arguments.library_dir:
                    if not path.is_dir():
                        raise ValueError(
                            f"runtime library directory is unavailable: {path}"
                        )
                run_stage(arguments, manifest, windows)
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"ETP/ETH workload failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
