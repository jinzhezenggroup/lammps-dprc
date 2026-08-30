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
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "workloads/etpeth/manifest.json"
DANGEROUS_BUILDS = re.compile(r"Dangerous builds\s*=\s*(\d+)")
DANGEROUS_BUILDS_NOT_CHECKED = "Dangerous builds not checked"
LAMMPS_TOPOLOGY_COUNT = re.compile(
    r"^\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)\s*$"
)
NVE_MINIMUM_SAMPLES = 10
NVE_MAXIMUM_ABSOLUTE_DRIFT_RATE_KCAL_MOL_PS_ATOM = 1.0e-4
NVE_MAXIMUM_ABSOLUTE_NET_DRIFT_KCAL_MOL_ATOM = 5.0e-4
NVE_MINIMUM_MEAN_TEMPERATURE_KELVIN = 200.0
NVE_MAXIMUM_MEAN_TEMPERATURE_KELVIN = 400.0
DPA4C_DPRC_GRAPH_POLICY = "exclude-all-environment-environment-edges"
DPA4C_DPRC_ENVIRONMENT_TYPES = ("OW", "HW")


class LammpsExecutionError(RuntimeError):
    """Report a failed external LAMMPS process without hiding validation errors."""


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
    def checkpoint_restart_root(self) -> Path:
        """Return the root used for periodic hitting-time restart files."""
        return self.output_directory / f"{self.window.tag}.checkpoint.restart"

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


def source_tree_record(repository: Path) -> dict[str, Any]:
    """Record a Git checkout or explicitly identify an unversioned snapshot.

    Runtime binary identities are recorded separately with SHA-256.  This
    source-level record must therefore remain honest when an execution node
    carries only an exported build tree: absence of Git metadata is reported
    as unknown revision and dirty state, never silently treated as clean.
    """
    repository = repository.resolve()
    if (repository / ".git").exists():
        dirty_output = git_output(repository, "status", "--porcelain=v1")
        return {
            "path": str(repository),
            "identity_kind": "git-checkout",
            "revision": git_output(repository, "rev-parse", "HEAD"),
            "dirty": bool(dirty_output),
            "dirty_entries": dirty_output.splitlines() if dirty_output else [],
        }

    revision_file = repository / ".source-revision"
    revision = None
    identity_kind = "unversioned-source-snapshot"
    if revision_file.is_file():
        candidate = revision_file.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{40}", candidate):
            revision = candidate
            identity_kind = "revision-stamped-source-snapshot"
    return {
        "path": str(repository),
        "identity_kind": identity_kind,
        "revision": revision,
        "dirty": None,
        "dirty_entries": [],
    }


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


def representative_nve_windows(windows: Sequence[Window]) -> list[Window]:
    """Select endpoint and central windows for the pre-production NVE gate.

    The three slots sample both ends of the umbrella coordinate and its middle
    without using any result-dependent choice.  Stable ascending order is
    retained so the same trajectory always occupies the same batch slot.
    """
    if len(windows) < 3:
        raise ValueError("NVE stability diagnostics require at least three windows")
    return [windows[0], windows[(len(windows) - 1) // 2], windows[-1]]


def topology_contract(manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return the fail-closed topology identity for one execution mode.

    The QM/MM topology deliberately removes internal classical terms from the
    16-atom QM region.  Reusing it for a nominal all-MM calculation silently
    deletes the solute bonded Hamiltonian, so classical and QM/MM inputs must
    remain separate assets throughout the complete checkpoint chain.
    """
    if mode not in {"classical", "qmmm", "qmmm-dpa4c"}:
        raise ValueError(f"unsupported ETP/ETH execution mode: {mode}")
    name = "classical" if mode == "classical" else "qmmm"
    topologies = manifest.get("system", {}).get("topologies")
    if not isinstance(topologies, dict):
        raise TypeError("workload manifest has no topology-contract map")
    contract = topologies.get(name)
    if not isinstance(contract, dict):
        raise TypeError(f"workload manifest has no {name} topology contract")
    required = ("data", "forcefield", "atoms", "bonds", "angles", "dihedrals")
    missing = [field for field in required if field not in contract]
    if missing:
        raise ValueError(
            f"{name} topology contract is missing: {', '.join(missing)}"
        )
    return contract


def initial_data_for_mode(
    manifest: dict[str, Any], tutorial: Path, mode: str
) -> Path:
    """Resolve the reviewed initial topology without cross-mode fallback."""
    contract = topology_contract(manifest, mode)
    return tutorial / str(contract["data"])


def validate_lammps_topology(
    path: Path, manifest: dict[str, Any], mode: str
) -> dict[str, int]:
    """Reject a checkpoint whose bonded topology belongs to another mode."""
    if not path.is_file():
        raise ValueError(f"LAMMPS start data is missing: {path}")
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = LAMMPS_TOPOLOGY_COUNT.fullmatch(line.rstrip("\n"))
            if match:
                counts[match.group(2)] = int(match.group(1))
            if line_number >= 64 or line.strip() == "Masses":
                break
    contract = topology_contract(manifest, mode)
    expected = {
        name: int(contract[name])
        for name in ("atoms", "bonds", "angles", "dihedrals")
    }
    # LAMMPS ``write_data`` omits a topology-count header when that count is
    # zero.  Treat an absent zero-valued section as the canonical count, while
    # retaining ``None`` for every missing nonzero section so a truncated or
    # cross-mode checkpoint still fails closed.
    observed = {
        name: counts.get(name, 0 if expected[name] == 0 else None)
        for name in expected
    }
    if observed != expected:
        raise ValueError(
            f"{mode} start data {path} has topology counts {observed}; "
            f"expected {expected}. Refusing to mix full-MM and QM/MM states."
        )
    return expected


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
    """Verify source identity and every external runtime artifact digest.

    A Git checkout proves the reviewed revision and dirty state directly.  A
    stripped source snapshot cannot make that claim, but it may still be used
    for explicitly private diagnostics after every runtime artifact matches
    the manifest.  Such snapshots remain unqualified and record no observed
    revision instead of copying the expected revision into the evidence.
    """
    tutorial = tutorial.resolve()
    if not tutorial.is_dir():
        raise ValueError(f"tutorial checkout is not a directory: {tutorial}")

    source = manifest["source"]
    has_git_metadata = (tutorial / ".git").exists()
    if has_git_metadata:
        revision: str | None = git_output(tutorial, "rev-parse", "HEAD")
        dirty_output = git_output(tutorial, "status", "--porcelain=v1")
        dirty_entries = dirty_output.splitlines() if dirty_output else []
        if revision != source["revision"]:
            raise ValueError(
                f"tutorial revision {revision} differs from reviewed "
                f"{source['revision']}"
            )
        identity_kind = "git-checkout"
    else:
        revision = None
        dirty_entries = []
        identity_kind = "artifact-verified-source-snapshot"
    qualification_reasons = [
        reason
        for condition, reason in (
            (
                not has_git_metadata,
                "source snapshot has no Git revision or dirty-state metadata",
            ),
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
        "identity_kind": identity_kind,
        "revision": revision,
        "expected_revision": source["revision"],
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


def generated_lammps_input_path(output: Path, name: str, content: str) -> Path:
    """Return a content-addressed path for one generated LAMMPS input.

    A rejected stochastic seed attempt reuses the logical invocation name but
    must use a different deterministic Langevin seed on retry.  Keeping the
    content digest in the filename preserves every attempted input and lets a
    retry coexist with the failed record until that record is archived.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return output / "generated/inputs" / f"{name}-{digest}.in"


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
    lammps_execution_backend: str = "kokkos",
    restart_checkpoint_frequency: int = 0,
    stop_on_seed_acceptance: bool = False,
    thermostat_enabled: bool = True,
    run_commands: Sequence[str] | None = None,
    execution_directory: Path | None = None,
) -> str:
    """Render one synchronized LAMMPS invocation for one or more windows.

    ``mode`` keeps the benchmark comparison on one reviewed input generator:
    ``classical`` is the complete Amber/TIP4P force field without xTB,
    ``qmmm`` is the batched xTBloom path, and ``qmmm-dpa4c`` overlays one
    compact DeePMD primary model through ``dprcplugin``.  The project-owned
    batch style currently accepts exactly one qualified model; ensemble model
    deviation remains a separate offline qualification workflow.

    ``run_commands`` is used by the performance harness to execute warmup and
    repeated steady-state segments in one LAMMPS process.  Ordinary scientific
    stages leave it unset and retain the single ``run <steps>`` behavior.
    When ``execution_directory`` is supplied, every runtime artifact is
    rendered relative to that directory and LAMMPS must be launched there.

    ``thermostat_enabled=False`` is reserved for explicit NVE stability
    diagnostics. It removes both Langevin forcing and periodic center-of-mass
    momentum removal while retaining the conservative umbrella potential.
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
    if lammps_execution_backend not in {"host", "kokkos"}:
        raise ValueError(
            "unsupported LAMMPS execution backend: "
            f"{lammps_execution_backend}"
        )
    uses_dprc_plugin = mode != "classical" or classical_backend == "batched-dprc"
    if uses_dprc_plugin and plugin is None:
        raise ValueError(f"{mode} requires the LAMMPS-DPRc plugin")
    if model_deviation_frequency < 0:
        raise ValueError("model-deviation frequency must be nonnegative")
    if restart_checkpoint_frequency < 0:
        raise ValueError("restart-checkpoint frequency must be nonnegative")
    if stop_on_seed_acceptance:
        if len(run_windows) != 1:
            raise ValueError("first-hit seed stopping requires exactly one window")
        if run_windows[0].colvars_profile != "seed":
            raise ValueError("first-hit seed stopping requires the seed restraint")
        if steps <= 0:
            raise ValueError("first-hit seed stopping requires a positive run length")
    if mode == "qmmm-dpa4c":
        if deepmd_plugin is not None:
            raise ValueError(
                "qmmm-dpa4c uses the DeePMD C API inside dprcplugin; do not "
                "supply a separate DeePMD LAMMPS plugin"
            )
        if model_deviation_frequency != 0:
            raise ValueError(
                "dprc/deepmd/batch currently requires model deviation to be "
                "disabled"
            )
        if len(deepmd_models) != 1:
            raise ValueError("qmmm-dpa4c requires exactly one model")
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
    contract = topology_contract(manifest, mode)
    source_forcefield = tutorial / str(contract["forcefield"])
    if not source_forcefield.is_file():
        raise ValueError(
            f"reviewed {mode} force-field include is missing: {source_forcefield}"
        )
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
        validate_lammps_topology(item.start_data, manifest, mode)
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
    if restart_checkpoint_frequency > 0:
        variable_values["checkpoint_restart_root"] = [
            path_token(item.checkpoint_restart_root) for item in run_windows
        ]
    for name, values in variable_values.items():
        variables.extend(render_world_variable(name, values))

    mesh = " ".join(str(value) for value in system["pppm_mesh"])
    kmax = " ".join(str(value) for value in xtb["kmax"])
    elements = " ".join(system["elements_by_lammps_type"])
    # Broker-owned CUDA work does not require every LAMMPS partition to create
    # a Kokkos CUDA context. The explicit host backend keeps atom, neighbor,
    # bonded, integration, and Colvars work on the CPU while the classical,
    # xTBloom, and DeePMD brokers retain their single GPU-owner contexts.
    # Keeping the non-batched upstream GPU reference on its native host chain
    # also avoids changing its baseline semantics.
    uses_batched_classical = (
        mode != "classical" or classical_backend == "batched-dprc"
    )
    kokkos_device = (
        uses_batched_classical and lammps_execution_backend == "kokkos"
    )
    deepmd_pair_style = (
        "dprc/deepmd/batch/kk" if kokkos_device else "dprc/deepmd/batch"
    )
    commands = [
        "# Generated by tools/etpeth_workload.py; do not hand-edit.",
        "clear",
    ]
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
    if mode == "classical":
        commands.append(
            "dihedral_style harmonic/kk"
            if kokkos_device
            else "dihedral_style harmonic"
        )

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
            # One universe owner loads DPA4c through the public DeePMD C API
            # and evaluates every synchronized umbrella window in one
            # block-diagonal forward. No DeePMD LAMMPS plugin is loaded.
            pair_style += (
                " partition_batch yes center_group qm "
                f"environment_cutoff {float(dprc['environment_cutoff_angstrom']):.1f} "
                "include_molecule yes"
            )
        commands.extend(
            [
                pair_style,
                f"include {forcefield}",
                (
                    "pair_coeff * * tip4p/long/dprc/batch"
                    if mode == "classical"
                    else "pair_coeff 6*7 6*7 tip4p/long/dprc/batch"
                ),
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

    if mode == "qmmm-dpa4c":
        # LAMMPS reports the sum of hybrid-overlay van der Waals-style
        # energies as ``evdwl``.  Keep the classical Lennard-Jones and DPA4c
        # correction publications independently observable so single-window
        # versus batched qualification can identify the responsible backend
        # instead of inferring it from their potentially cancelling sum.
        commands.extend(
            [
                "compute dprc_lj_energy all pair lj/cut/dprc/batch evdwl",
                (
                    "compute dprc_correction_energy all pair "
                    f"{deepmd_pair_style} evdwl"
                ),
            ]
        )

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
        "step temp pe etotal evdwl etail ecoul ebond eangle "
        + ("edihed " if mode == "classical" else "")
        + "elong "
        + ("f_qmmm " if mode != "classical" else "")
        + (
            "c_dprc_lj_energy c_dprc_correction_energy "
            if mode == "qmmm-dpa4c"
            else ""
        )
        + "f_restraints"
    )
    commands.extend([
        (
            "fix water_shake water shake/kk 1.0e-6 200 0 b 1 a 1"
            if kokkos_device
            else "fix water_shake water shake 1.0e-6 200 0 b 1 a 1"
        ),
        "fix integrate all nve/kk" if kokkos_device else "fix integrate all nve",
    ])
    if thermostat_enabled:
        commands.extend([
            (
                f"fix thermostat all "
                f"{'langevin/kk' if kokkos_device else 'langevin'} "
                f"{dynamics['temperature_kelvin']} "
                f"{dynamics['temperature_kelvin']} "
                f"{dynamics['langevin_damping_fs']} "
                "${thermostat_seed}"
            ),
            (
                "fix remove_com all momentum/kk 1000 linear 1 1 1"
                if kokkos_device
                else "fix remove_com all momentum 1000 linear 1 1 1"
            ),
        ])
    commands.extend([
        (
            f"fix restraints all {'colvars/kk' if kokkos_device else 'colvars'} "
            "${colvars_config} output ${colvars_output}"
        ),
        "fix_modify restraints energy yes",
    ])
    if stop_on_seed_acceptance:
        item = run_windows[0]
        acceptance = manifest["protocol"]["seed_acceptance"]
        angle_center = float(manifest["umbrella"]["attack_angle"]["center_degree"])
        reaction_tolerance = float(
            acceptance["max_abs_reaction_coordinate_error_angstrom"]
        )
        angle_tolerance = float(
            acceptance["max_abs_attack_angle_error_degree"]
        )
        check_frequency = int(dynamics["colvars_frequency_steps"])
        if check_frequency <= 0:
            raise ValueError("Colvars frequency must be positive for first-hit stopping")
        # Fix colvars exposes the current scalar CVs as rows of its global
        # array.  Evaluating the reviewed gate from that array lets LAMMPS
        # stop the stochastic trajectory in the same process and publish the
        # exact accepted state; no chaotic GPU trajectory replay is involved.
        commands.extend(
            [
                (
                    'variable seed_gate_reached equal "'
                    f"(abs(f_restraints[1][1]-({item.window.center:.17g})) <= "
                    f"{reaction_tolerance:.17g}) && "
                    f"(abs(f_restraints[2][1]-({angle_center:.17g})) <= "
                    f'{angle_tolerance:.17g})"'
                ),
                (
                    f"fix seed_first_hit all halt {check_frequency} "
                    "v_seed_gate_reached != 0 error soft message yes"
                ),
            ]
        )
    commands.extend([
        # Under ``units real`` LAMMPS already expresses time in femtoseconds.
        # The manifest value is explicitly named ``timestep_fs``; converting
        # it to picoseconds here would silently shorten every scientific stage
        # by a factor of 1000.
        f"timestep {float(dynamics['timestep_fs']):.6f}",
        f"neighbor {dynamics['neighbor_skin_angstrom']} bin",
        (
            f"neigh_modify every {dynamics['neighbor_every']} delay 0 check "
            f"{'yes' if dynamics.get('neighbor_check', True) else 'no'}"
        ),
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
    if restart_checkpoint_frequency > 0:
        # With the production every-step/check-no neighbor policy, these
        # writes do not add a new rebuild point and therefore do not perturb
        # the pilot trajectory.  A single restart root makes LAMMPS append the
        # absolute timestep and preserve every Colvars-frequency checkpoint.
        commands.append(
            f"restart {restart_checkpoint_frequency} ${{checkpoint_restart_root}}"
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


def completed_steps_for_record(
    *,
    requested_steps: int,
    timestep_offset: int,
    final_colvars_step: float,
    stop_on_seed_acceptance: bool,
) -> int:
    """Return actual MD steps without mistaking sparse CV output for progress.

    Ordinary LAMMPS runs complete their requested length even when that length
    is not a multiple of the Colvars output frequency.  Only a first-hit seed
    run can end early, in which case the halt and Colvars checks share the same
    frequency and the final sampled timestep is the exact stopping point.
    """
    if not stop_on_seed_acceptance:
        return requested_steps
    completed = final_colvars_step - timestep_offset
    if not float(completed).is_integer():
        raise ValueError("first-hit completed step is not integral")
    return int(completed)


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


def seed_attempt_schedule(
    manifest: dict[str, Any], maximum_attempts: int
) -> tuple[int, ...]:
    """Return the exact adaptive seed-walk length for every allowed attempt.

    A difficult umbrella center may require time to cross a local barrier even
    under the stronger transient seed restraint.  Retrying only the Langevin
    seed at one short duration does not address that failure mode, so the
    manifest declares a reviewed, finite multiplier schedule.  The CLI may
    select a prefix but cannot invent attempts beyond that declared protocol.
    """
    if maximum_attempts < 1:
        raise ValueError("--seed-max-attempts must be positive")
    protocol = manifest["protocol"]
    base_steps = int(protocol["seed_walk_steps_per_center"])
    raw_multipliers = protocol.get("seed_walk_attempt_step_multipliers")
    if not isinstance(raw_multipliers, list) or not raw_multipliers:
        raise ValueError(
            "seed_walk_attempt_step_multipliers must be a nonempty list"
        )
    if any(
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier < 1
        for multiplier in raw_multipliers
    ):
        raise ValueError(
            "seed_walk_attempt_step_multipliers must contain positive integers"
        )
    if maximum_attempts > len(raw_multipliers):
        raise ValueError(
            "--seed-max-attempts exceeds the manifest's reviewed adaptive "
            "seed schedule"
        )
    return tuple(
        base_steps * multiplier
        for multiplier in raw_multipliers[:maximum_attempts]
    )


def seed_attempt_metadata(
    manifest: dict[str, Any],
    windows: Sequence[Window],
    *,
    attempt_index: int,
    step_schedule: Sequence[int],
) -> dict[str, Any]:
    """Describe one seed attempt completely enough for strict resumption."""
    if attempt_index < 0 or attempt_index >= len(step_schedule):
        raise ValueError("seed attempt index is outside the reviewed schedule")
    base_steps = int(manifest["protocol"]["seed_walk_steps_per_center"])
    scheduled_steps = int(step_schedule[attempt_index])
    if scheduled_steps < base_steps or scheduled_steps % base_steps != 0:
        raise ValueError("adaptive seed steps are not an integer base-step multiple")
    return {
        "schema_version": 1,
        "attempt_index": attempt_index,
        "attempt_number": attempt_index + 1,
        "maximum_attempts": len(step_schedule),
        "base_steps_per_center": base_steps,
        "step_multiplier": scheduled_steps // base_steps,
        "scheduled_steps": scheduled_steps,
        "step_schedule": [int(steps) for steps in step_schedule],
        "restart_policy": "unchanged-accepted-parent",
        "thermostat_seed_policy": "distinct-deterministic-per-attempt-and-window",
        "thermostat_seeds": {
            window.tag: seed_for(manifest, window, 3, trial=attempt_index)
            for window in windows
        },
    }


def seed_row_is_accepted(
    manifest: dict[str, Any], window: Window, row: dict[str, float]
) -> bool:
    """Return whether one sampled seed configuration satisfies the fixed gate."""
    acceptance = manifest["protocol"]["seed_acceptance"]
    angle_center = float(manifest["umbrella"]["attack_angle"]["center_degree"])
    return (
        abs(float(row["reaction_coordinate"]) - window.center)
        <= float(acceptance["max_abs_reaction_coordinate_error_angstrom"])
        and abs(float(row["attack_angle"]) - angle_center)
        <= float(acceptance["max_abs_attack_angle_error_degree"])
    )


def first_seed_hitting_row(
    manifest: dict[str, Any], window: Window, colvars_path: Path
) -> dict[str, float] | None:
    """Select the earliest positive-time Colvars sample that passes the seed gate."""
    return next(
        (
            row
            for row in parse_colvars_rows(colvars_path)
            if row["step"] > 0.0 and seed_row_is_accepted(manifest, window, row)
        ),
        None,
    )


def seed_capture_metadata(
    manifest: dict[str, Any],
    window: Window,
    *,
    attempt_index: int,
    step_schedule: Sequence[int],
    selected_row: dict[str, float],
) -> dict[str, Any]:
    """Describe publication of the first accepted pilot restart checkpoint."""
    pilot = seed_attempt_metadata(
        manifest,
        [window],
        attempt_index=attempt_index,
        step_schedule=step_schedule,
    )
    selected_step = int(selected_row["step"])
    if selected_step <= 0 or selected_step > pilot["scheduled_steps"]:
        raise ValueError("seed capture step is outside its pilot trajectory")
    return {
        "schema_version": 1,
        "strategy": "periodic-restart-hitting-time-capture",
        "selection_policy": "earliest-positive-qualified-colvars-sample",
        "pilot_attempt": pilot,
        "selected_step": selected_step,
        "selected_values": {
            "step": float(selected_row["step"]),
            "reaction_coordinate": float(selected_row["reaction_coordinate"]),
            "attack_angle": float(selected_row["attack_angle"]),
        },
        "thermostat_seed": pilot["thermostat_seeds"][window.tag],
        "restart_policy": "exact-pilot-binary-restart",
    }


def seed_first_hit_metadata(
    manifest: dict[str, Any],
    window: Window,
    *,
    attempt_index: int,
    step_schedule: Sequence[int],
    selected_row: dict[str, float],
) -> dict[str, Any]:
    """Describe an accepted state stopped and written by its pilot process."""
    attempt = seed_attempt_metadata(
        manifest,
        [window],
        attempt_index=attempt_index,
        step_schedule=step_schedule,
    )
    selected_step = int(selected_row["step"])
    if selected_step <= 0 or selected_step > attempt["scheduled_steps"]:
        raise ValueError("seed first-hit step is outside its pilot trajectory")
    return {
        "schema_version": 1,
        "strategy": "in-process-first-hitting-time-capture",
        "selection_policy": "earliest-positive-qualified-colvars-sample",
        "pilot_attempt": attempt,
        "selected_step": selected_step,
        "selected_values": {
            "step": float(selected_row["step"]),
            "reaction_coordinate": float(selected_row["reaction_coordinate"]),
            "attack_angle": float(selected_row["attack_angle"]),
        },
        "thermostat_seed": attempt["thermostat_seeds"][window.tag],
        "halt_check_frequency_steps": int(
            manifest["dynamics"]["colvars_frequency_steps"]
        ),
        "state_publication_policy": "atomic-byte-copy-of-same-process-final-state",
    }


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


def validate_canonical_dprc_artifact(path: Path) -> dict[str, Any]:
    """Require the exact graph-build contract consumed by the LAMMPS plugin.

    The compact DPA4c kernel has no runtime pair mask.  A scientifically
    qualified artifact must therefore declare that its redundant descriptor
    exclusion was removed and that all MM environment--environment edges are
    omitted by the caller before inference.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith("/extra/metadata.json")
            ]
            model_names = [
                name
                for name in archive.namelist()
                if name.endswith("/extra/model.json")
            ]
            if len(metadata_names) != 1 or len(model_names) != 1:
                raise ValueError(
                    "must contain exactly one metadata.json and model.json"
                )
            metadata = json.loads(archive.read(metadata_names[0]))
            model_json = json.loads(archive.read(model_names[0]))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise ValueError(f"invalid DPA4c canonical artifact {path}: {error}") from error

    required = {
        "lower_input_kind": "dpa4c_canonical",
        "graph_edge_dtype": "float32",
        "canonical_index_dtype": "uint32",
        "dprc_graph_policy": DPA4C_DPRC_GRAPH_POLICY,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"DPA4c canonical metadata mismatch: {mismatches}")
    type_map = metadata.get("type_map")
    if not isinstance(type_map, list) or not all(
        isinstance(name, str) and name for name in type_map
    ):
        raise ValueError("DPA4c canonical metadata has an invalid type_map")
    try:
        environment_indices = [
            type_map.index(name) for name in DPA4C_DPRC_ENVIRONMENT_TYPES
        ]
    except ValueError as error:
        raise ValueError("DPA4c canonical type_map requires OW and HW") from error
    expected_pairs = sorted(
        [min(first, second), max(first, second)]
        for position, first in enumerate(environment_indices)
        for second in environment_indices[position:]
    )
    if metadata.get("dprc_environment_type_names") != list(
        DPA4C_DPRC_ENVIRONMENT_TYPES
    ):
        raise ValueError("DPA4c artifact declares different environment type names")
    if metadata.get("dprc_environment_type_indices") != environment_indices:
        raise ValueError("DPA4c artifact environment indices disagree with type_map")
    if metadata.get("pair_exclude_types") != expected_pairs:
        raise ValueError(
            "DPA4c artifact pair exclusions do not exactly cover MM--MM pairs"
        )
    serialized = model_json.get("model")
    descriptor = serialized.get("descriptor") if isinstance(serialized, dict) else None
    if not isinstance(descriptor, dict) or descriptor.get("exclude_types") != []:
        raise ValueError(
            "DPA4c canonical deployment must remove redundant descriptor exclusions"
        )
    if serialized.get("pair_exclude_types") != expected_pairs:
        raise ValueError("DPA4c model.json lost the MM--MM exclusion contract")
    return metadata


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
    if mode not in {"classical", "qmmm", "qmmm-dpa4c"}:
        raise ValueError(f"unsupported production execution mode: {mode}")
    if model_deviation_frequency < 0:
        raise ValueError("--model-deviation-frequency must be nonnegative")
    if dpa4c_models_qualified and allow_unqualified_dpa4c_models:
        raise ValueError(
            "--dpa4c-models-qualified and "
            "--allow-unqualified-dpa4c-models are mutually exclusive"
        )
    if mode in {"classical", "qmmm"}:
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

    if deepmd_plugin is not None:
        raise ValueError(
            "qmmm-dpa4c links the DeePMD C API into dprcplugin and does not "
            "accept --deepmd-plugin"
        )
    if model_deviation_frequency != 0:
        raise ValueError(
            "dprc/deepmd/batch currently supports exactly one primary model "
            "with model deviation disabled"
        )
    if len(deepmd_models) != 1:
        raise ValueError(
            "qmmm-dpa4c requires exactly one --deepmd-model artifact"
        )
    missing = [path for path in deepmd_models if not path.is_file()]
    if missing:
        raise ValueError(f"DPA4c model artifact is unavailable: {missing[0]}")
    if not dpa4c_models_qualified and not allow_unqualified_dpa4c_models:
        raise ValueError(
            "qmmm-dpa4c requires either --dpa4c-models-qualified or the explicit "
            "diagnostic opt-in --allow-unqualified-dpa4c-models"
        )
    if dpa4c_models_qualified:
        validate_canonical_dprc_artifact(deepmd_models[0])


def execution_record(
    *,
    mode: str,
    model_deviation_frequency: int,
    dpa4c_models_qualified: bool,
    allow_unqualified_dpa4c_models: bool,
    lammps_execution_backend: str = "kokkos",
    thermostat_enabled: bool = True,
) -> dict[str, Any]:
    """Describe the force composition and model-qualification boundary."""
    if lammps_execution_backend not in {"host", "kokkos"}:
        raise ValueError(
            "unsupported LAMMPS execution backend: "
            f"{lammps_execution_backend}"
        )
    record: dict[str, Any] = {
        "mode": mode,
        "lammps_execution_backend": lammps_execution_backend,
    }
    if not thermostat_enabled:
        record["dynamics"] = {
            "ensemble": "NVE",
            "thermostat": "disabled",
            "center_of_mass_momentum_removal": "disabled",
            "umbrella_restraints": "enabled",
        }
    if mode == "qmmm-dpa4c":
        record["dprc_schedule"] = {
            "primary_model_index": 0,
            "model_deviation_frequency_steps": model_deviation_frequency,
            "model_deviation_enabled": model_deviation_frequency > 0,
            "execution_backend": "dprcplugin-deepmd-c-api-batch",
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


def runtime_environment_contract(
    selected_environment: object,
) -> dict[str, str] | None:
    """Return the cross-invocation environment identity.

    Slurm may bind two dependent jobs to different physical GPU ordinals on
    the same qualified node.  ``CUDA_VISIBLE_DEVICES`` must therefore remain
    recorded for each invocation, but its value is a scheduler-local launch
    detail rather than a scientific dependency of an accepted checkpoint.
    Loader paths and thread/launcher controls remain part of the strict
    contract because changing any of them can change the executed runtime.
    """
    if not isinstance(selected_environment, dict):
        return None
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in selected_environment.items()
    ):
        return None
    cuda_visible_devices = selected_environment.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        return None
    return {
        key: value
        for key, value in selected_environment.items()
        if key != "CUDA_VISIBLE_DEVICES"
    }


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
        raise ValueError(
            "runtime_record does not accept a separate DeePMD LAMMPS plugin"
        )
    if deepmd_models:
        record["models"] = [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in deepmd_models
        ]
    loaded_deepmd_c = verify_loaded_deepmd_c(
        plugin, environment, required=bool(deepmd_models)
    )
    if loaded_deepmd_c is not None:
        record["deepmd_c_api"] = {
            "path": loaded_deepmd_c["resolved_path"],
            "sha256": loaded_deepmd_c["sha256"],
            "soname": loaded_deepmd_c["soname"],
        }
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


def verify_loaded_deepmd_c(
    plugin: Path,
    environment: dict[str, str],
    *,
    required: bool = True,
) -> dict[str, str] | None:
    """Record the exact public DeePMD C API library resolved by the plugin."""
    output = command_output(["ldd", str(plugin.resolve())], environment=environment)
    if output is None:
        if required:
            raise ValueError(f"could not inspect dynamic dependencies of {plugin}")
        return None
    matches: list[tuple[str, Path]] = []
    for line in output.splitlines():
        tokens = line.strip().split()
        if not tokens or not tokens[0].startswith("libdeepmd_c.so"):
            continue
        if len(tokens) < 3 or tokens[1] != "=>" or tokens[2] == "not":
            raise ValueError("dprcplugin does not resolve its libdeepmd_c dependency")
        matches.append((tokens[0], Path(tokens[2]).resolve()))
    if not matches:
        if required:
            raise ValueError("dprcplugin has no libdeepmd_c dependency")
        return None
    if len(matches) != 1:
        raise ValueError("dprcplugin resolves multiple libdeepmd_c dependencies")
    soname, resolved = matches[0]
    if not resolved.is_file():
        raise ValueError(f"resolved libdeepmd_c is not a file: {resolved}")
    return {
        "soname": soname,
        "resolved_path": str(resolved),
        "sha256": sha256(resolved),
    }


def project_record(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Identify the dirty development runner that produced diagnostic data."""
    project = source_tree_record(PROJECT_ROOT)
    provenance = output / "provenance.json"
    dependencies = {}
    for name, repository in (
        ("lammps", PROJECT_ROOT.parent / "lammps"),
        ("xtbloom", PROJECT_ROOT.parent / "xtbloom"),
    ):
        if repository.is_dir():
            dependencies[name] = source_tree_record(repository)
    source_evidence_qualified = project["dirty"] is False and all(
        dependency["dirty"] is False for dependency in dependencies.values()
    )
    return {
        **project,
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
        "qualification": (
            "clean-source" if source_evidence_qualified else "private-diagnostic"
        ),
    }


def artifact_matches(record: dict[str, Any] | None, path: Path) -> bool:
    """Check both the absolute path and bytes of one recorded artifact."""
    return bool(
        record
        and path.is_file()
        and Path(record.get("path", "")).resolve() == path.resolve()
        and record.get("sha256") == sha256(path)
    )


def restart_checkpoint_timesteps(
    *, timestep_offset: int, steps: int, frequency: int
) -> list[int]:
    """Return every periodic-restart timestep expected during one run.

    LAMMPS does not write a periodic restart on the first timestep of a run.
    The first expected checkpoint is therefore the first positive multiple of
    ``frequency`` after ``timestep_offset``.
    """
    if frequency <= 0:
        return []
    completed = timestep_offset + steps
    first = (timestep_offset // frequency + 1) * frequency
    return list(range(first, completed + 1, frequency))


def restart_checkpoint_paths(
    item: RunWindow,
    *,
    timestep_offset: int,
    steps: int,
    frequency: int,
) -> list[tuple[int, Path]]:
    """Return timesteps and paths for one window's periodic restarts."""
    return [
        (timestep, Path(f"{item.checkpoint_restart_root}.{timestep}"))
        for timestep in restart_checkpoint_timesteps(
            timestep_offset=timestep_offset,
            steps=steps,
            frequency=frequency,
        )
    ]


def record_is_resumable(
    path: Path,
    run_windows: Sequence[RunWindow],
    *,
    input_path: Path,
    steps: int,
    timestep_offset: int,
    trajectory_frequency: int,
    restart_checkpoint_frequency: int = 0,
    stop_on_seed_acceptance: bool = False,
    ranks_per_window: int,
    lammps: Path,
    plugin: Path,
    xtbloom_library: Path,
    mpiexec: Path,
    runner_path: Path,
    loaded_xtbloom: dict[str, str],
    deepmd_c_api: dict[str, str] | None = None,
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
    lammps_execution_backend: str = "kokkos",
    thermostat_enabled: bool = True,
    seed_attempt: dict[str, Any] | None = None,
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
        or runtime_environment_contract(record.get("selected_environment"))
        != runtime_environment_contract(selected_environment)
        or record.get("environment_contract")
        != runtime_environment_contract(record.get("selected_environment"))
        or not artifact_matches(record.get("input"), input_path)
        or record.get("execution", {"mode": "qmmm"})
        != execution_record(
            mode=mode,
            model_deviation_frequency=model_deviation_frequency,
            dpa4c_models_qualified=dpa4c_models_qualified,
            allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
            lammps_execution_backend=lammps_execution_backend,
            thermostat_enabled=thermostat_enabled,
        )
        or record.get("seed_attempt") != seed_attempt
        or record.get("restart_checkpoint_frequency_steps", 0)
        != restart_checkpoint_frequency
        or record.get("stop_on_seed_acceptance", False)
        != stop_on_seed_acceptance
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
        if deepmd_plugin is not None or runtime.get("deepmd_plugin") is not None:
            return False
        if runtime.get("deepmd_c_api") != deepmd_c_api:
            return False
        deepmd_c_api = runtime.get("deepmd_c_api")
        if (
            not isinstance(deepmd_c_api, dict)
            or "path" not in deepmd_c_api
            or not artifact_matches(deepmd_c_api, Path(deepmd_c_api["path"]))
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
        checkpoints = outputs.get("restart_checkpoints")
        if restart_checkpoint_frequency > 0:
            expected_checkpoints = restart_checkpoint_paths(
                item,
                timestep_offset=timestep_offset,
                steps=steps,
                frequency=restart_checkpoint_frequency,
            )
            if not isinstance(checkpoints, list) or len(checkpoints) != len(
                expected_checkpoints
            ):
                return False
            for identity, (timestep, checkpoint_path) in zip(
                checkpoints, expected_checkpoints, strict=True
            ):
                if (
                    not isinstance(identity, dict)
                    or identity.get("timestep") != timestep
                    or not artifact_matches(identity, checkpoint_path)
                ):
                    return False
        elif checkpoints is not None:
            return False
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
    record_path: Path,
    common: dict[str, Any],
    *,
    allow_expected_first_hit_no_hit: bool = False,
) -> dict[str, Any]:
    """Validate one invocation against current bytes and loader state.

    Normal callers accept only ``passed`` records.  The adaptive first-hit
    seed search is the one deliberate exception: a clean LAMMPS run may reach
    its reviewed pilot endpoint without finding the gate.  Such a record is
    accepted only when the caller explicitly opts in and the record carries
    the complete, fail-closed no-hit classification below.
    """
    if not record_path.is_file():
        raise ValueError(f"accepted invocation record is missing: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read invocation record {record_path}: {error}"
        ) from error
    expected_first_hit_no_hit = (
        allow_expected_first_hit_no_hit
        and record.get("status") == "failed"
        and record.get("failure_classification") == "expected-first-hit-no-hit"
    )
    if record.get("status") != "passed" and not expected_first_hit_no_hit:
        raise ValueError(f"invocation did not pass: {record_path}")
    if expected_first_hit_no_hit:
        # A no-hit pilot is an expected scientific result, not a crashed or
        # partially completed process.  Keep this classification narrow so a
        # failed invocation cannot enter the seed ledger merely by changing a
        # status string.  The later first-hit validator independently checks
        # that the Colvars file contains no accepted row.
        returncode = record.get("returncode")
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode != 0
        ):
            raise ValueError(
                f"expected first-hit no-hit record did not exit cleanly: {record_path}"
            )
        if record.get("stop_on_seed_acceptance") is not True:
            raise ValueError(
                f"expected first-hit no-hit record was not a first-hit run: {record_path}"
            )
        error = record.get("error")
        if not isinstance(error, str) or not error.startswith(
            "first-hit acceptance failed:"
        ):
            raise ValueError(
                f"expected first-hit no-hit record has invalid failure detail: {record_path}"
            )
        steps = record.get("steps_per_window")
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
            or record.get("timestep_offset") != 0
        ):
            raise ValueError(
                f"expected first-hit no-hit record has invalid timeline: {record_path}"
            )
        outputs = record.get("outputs")
        completed_steps = record.get("completed_steps_by_window")
        if (
            not isinstance(outputs, dict)
            or not outputs
            or not isinstance(completed_steps, dict)
            or set(completed_steps) != set(outputs)
            or record.get("aggregate_window_steps")
            != steps * len(outputs)
        ):
            raise ValueError(
                f"expected first-hit no-hit record has incomplete progress: {record_path}"
            )
        for tag, output in outputs.items():
            first_hit = output.get("first_hit") if isinstance(output, dict) else None
            if (
                not isinstance(output, dict)
                or output.get("seed_acceptance") is not False
                or not isinstance(first_hit, dict)
                or first_hit.get("accepted_sample_found") is not False
                or isinstance(first_hit.get("scheduled_steps"), bool)
                or first_hit.get("scheduled_steps") != steps
                or isinstance(first_hit.get("completed_steps"), bool)
                or first_hit.get("completed_steps") != steps
                or first_hit.get("halted_early") is not False
                or isinstance(completed_steps.get(tag), bool)
                or completed_steps.get(tag) != steps
            ):
                raise ValueError(
                    f"expected first-hit no-hit record has inconsistent output: "
                    f"{record_path} ({tag})"
                )
    restart_checkpoint_frequency = record.get(
        "restart_checkpoint_frequency_steps", 0
    )
    if (
        isinstance(restart_checkpoint_frequency, bool)
        or not isinstance(restart_checkpoint_frequency, int)
        or restart_checkpoint_frequency < 0
    ):
        raise ValueError(
            f"invocation has invalid restart checkpoint frequency: {record_path}"
        )
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
    lammps_execution_backend = str(
        common.get("lammps_execution_backend", "kokkos")
    )
    expected_execution = execution_record(
        mode=mode,
        model_deviation_frequency=model_deviation_frequency,
        dpa4c_models_qualified=dpa4c_models_qualified,
        allow_unqualified_dpa4c_models=allow_unqualified_dpa4c_models,
        lammps_execution_backend=lammps_execution_backend,
        thermostat_enabled=bool(common.get("thermostat_enabled", True)),
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
    loaded_deepmd_c = verify_loaded_deepmd_c(
        common["plugin"], environment, required=mode == "qmmm-dpa4c"
    )
    current_deepmd_c_api = (
        {
            "path": loaded_deepmd_c["resolved_path"],
            "sha256": loaded_deepmd_c["sha256"],
            "soname": loaded_deepmd_c["soname"],
        }
        if loaded_deepmd_c is not None
        else None
    )
    if runtime.get("deepmd_c_api") != current_deepmd_c_api:
        raise ValueError(f"DeePMD C API loader resolution changed since {record_path}")
    if mode == "qmmm-dpa4c":
        if deepmd_plugin is not None or runtime.get("deepmd_plugin") is not None:
            raise ValueError(
                f"unexpected separate DeePMD plugin in {record_path}"
            )
        deepmd_c_api = runtime.get("deepmd_c_api")
        if (
            not isinstance(deepmd_c_api, dict)
            or "path" not in deepmd_c_api
            or not artifact_matches(deepmd_c_api, Path(deepmd_c_api["path"]))
        ):
            raise ValueError(f"DeePMD C API identity changed since {record_path}")
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
    recorded_contract = runtime_environment_contract(
        record.get("selected_environment")
    )
    if (
        recorded_contract is None
        or record.get("environment_contract") != recorded_contract
        or recorded_contract != runtime_environment_contract(selected)
    ):
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
                    "first_hit",
                }:
                    continue
                if kind == "restart_checkpoints":
                    expected_steps = restart_checkpoint_timesteps(
                        timestep_offset=int(record["timestep_offset"]),
                        steps=int(record["steps_per_window"]),
                        frequency=restart_checkpoint_frequency,
                    )
                    if not isinstance(identity, list) or len(identity) != len(
                        expected_steps
                    ):
                        raise ValueError(
                            f"restart checkpoints changed for {tag} in {record_path}"
                        )
                    for checkpoint, expected_step in zip(
                        identity, expected_steps, strict=True
                    ):
                        if (
                            not isinstance(checkpoint, dict)
                            or checkpoint.get("timestep") != expected_step
                            or "path" not in checkpoint
                            or not artifact_matches(
                                checkpoint, Path(checkpoint["path"])
                            )
                        ):
                            raise ValueError(
                                f"restart checkpoint changed for {tag}/"
                                f"{expected_step} in {record_path}"
                            )
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
    colvars_profile: str = "sampling",
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
    if qualification == "native-first-hit":
        return require_first_hit_stage(
            ledger_path,
            ledger=ledger,
            stage=stage,
            windows=windows,
            expected_start_data=expected_start_data,
            expected_final_data=expected_final_data,
            total_steps=total_steps,
            maximum_chunk_steps=maximum_chunk_steps,
            trajectory_frequency=trajectory_frequency,
            common=common,
            colvars_profile=colvars_profile,
        )
    if qualification != "native-chunked":
        raise ValueError(f"unsupported stage qualification in {ledger_path}")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "passed"
        or ledger.get("stage") != stage
        or ledger.get("total_steps_per_window") != total_steps
        or ledger.get("maximum_chunk_steps") != maximum_chunk_steps
        or ledger.get("colvars_profile", "sampling") != colvars_profile
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


def require_first_hit_stage(
    ledger_path: Path,
    *,
    ledger: dict[str, Any],
    stage: str,
    windows: Sequence[Window],
    expected_start_data: dict[str, Path],
    expected_final_data: dict[str, Path],
    total_steps: int,
    maximum_chunk_steps: int,
    trajectory_frequency: int,
    common: dict[str, Any],
    colvars_profile: str,
) -> dict[str, Any]:
    """Validate a stage whose state is captured at its first accepted sample.

    The anchor is an initialization gate rather than a production trajectory.
    Its final state must therefore be the same-process first Colvars sample
    satisfying the fixed seed gate; validating only the requested endpoint
    would allow a later thermal fluctuation to replace an already-qualified
    configuration.
    """
    expected_order = [window.tag for window in windows]
    check_frequency = int(common["manifest"]["dynamics"]["colvars_frequency_steps"])
    if (
        ledger.get("schema_version") != 1
        or ledger.get("status") != "passed"
        or ledger.get("qualification") != "native-first-hit"
        or ledger.get("stage") != stage
        or ledger.get("total_steps_per_window") != total_steps
        or ledger.get("maximum_chunk_steps") != maximum_chunk_steps
        or ledger.get("colvars_profile") != colvars_profile
        or ledger.get("window_order") != expected_order
        or ledger.get("record_kind") != "anchor-first-hit-ledger"
        or set(ledger.get("outputs", {})) != set(expected_order)
    ):
        raise ValueError(f"first-hit stage protocol in {ledger_path} does not match the request")
    if len(windows) != 1:
        raise ValueError("native first-hit stages currently require one window")

    record_identity = ledger.get("record")
    if not isinstance(record_identity, dict) or "path" not in record_identity:
        raise ValueError(f"first-hit stage record identity is missing: {ledger_path}")
    record_path = Path(record_identity["path"])
    if not artifact_matches(record_identity, record_path):
        raise ValueError(f"first-hit stage invocation changed: {record_path}")
    record = validate_invocation_record_current(record_path, common)
    tag = windows[0].tag
    if (
        record.get("name") != f"{stage}-first-hit"
        or record.get("status") != "passed"
        or record.get("steps_per_window") != total_steps
        or record.get("timestep_offset") != 0
        or record.get("worlds") != 1
        or record.get("window_order") != expected_order
        or record.get("ranks_per_window") != common["ranks_per_window"]
        or record.get("stop_on_seed_acceptance") is not True
        or set(record.get("start_inputs", {})) != {tag}
        or set(record.get("outputs", {})) != {tag}
    ):
        raise ValueError(f"first-hit stage invocation protocol mismatch in {record_path}")
    if not artifact_matches(record["start_inputs"][tag], expected_start_data[tag]):
        raise ValueError(f"first-hit stage start changed for {tag}")

    output_record = record["outputs"][tag]
    if not isinstance(output_record, dict):
        raise ValueError(f"first-hit stage output is missing for {tag}")
    expected_data = expected_final_data[tag]
    if not artifact_matches(output_record.get("data"), expected_data):
        raise ValueError(f"first-hit stage final state changed for {tag}")
    if output_record.get("seed_acceptance") is not True:
        raise ValueError(f"first-hit stage output was not accepted for {tag}")
    first_hit = output_record.get("first_hit")
    if not isinstance(first_hit, dict):
        raise ValueError(f"first-hit evidence is missing for {tag}")
    completed_steps = first_hit.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps <= 0
        or completed_steps > total_steps
        or completed_steps % check_frequency != 0
        or first_hit.get("check_frequency_steps") != check_frequency
        or first_hit.get("scheduled_steps") != total_steps
        or first_hit.get("accepted_sample_found") is not True
        or first_hit.get("halted_early") is not (completed_steps < total_steps)
    ):
        raise ValueError(f"first-hit stop evidence changed for {tag}")
    if (
        record.get("completed_steps_by_window", {}).get(tag) != completed_steps
        or record.get("aggregate_window_steps") != completed_steps
    ):
        raise ValueError(f"first-hit aggregate step count changed for {tag}")

    colvars_identity = output_record.get("colvars")
    if not isinstance(colvars_identity, dict) or "path" not in colvars_identity:
        raise ValueError(f"first-hit Colvars identity is missing for {tag}")
    colvars_path = Path(colvars_identity["path"])
    selected = first_seed_hitting_row(common["manifest"], windows[0], colvars_path)
    if selected is None or selected != output_record.get("final_values"):
        raise ValueError(f"first-hit Colvars selection changed for {tag}")
    if selected["step"] != float(completed_steps):
        raise ValueError(f"first-hit Colvars step disagrees with the halted state for {tag}")

    if trajectory_frequency > 0:
        if "trajectory" not in output_record:
            raise ValueError(f"first-hit trajectory is missing for {tag}")
    elif "trajectory" in output_record:
        raise ValueError(f"unexpected first-hit trajectory for {tag}")

    try:
        record_wall = float(record["wall_seconds"])
        ledger_wall = float(ledger["wall_seconds"])
        ledger_throughput = float(ledger["aggregate_window_steps_per_second"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("first-hit stage performance aggregate is malformed") from error
    if not math.isclose(record_wall, ledger_wall, rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError("first-hit stage wall time differs from its invocation record")
    expected_throughput = completed_steps / record_wall if record_wall else None
    if expected_throughput is None or not math.isclose(
        ledger_throughput, expected_throughput, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError("first-hit stage throughput is inconsistent")
    if ledger.get("outputs") != {tag: output_record}:
        raise ValueError(f"first-hit stage outputs differ from {record_path}")
    return ledger


def validate_hitting_time_seed_round_record(
    record_path: Path,
    record: dict[str, Any],
    *,
    manifest: dict[str, Any],
    round_index: int,
    expected_windows: Sequence[tuple[str, Window]],
    previous: dict[str, dict[str, str]],
    step_schedule: Sequence[int],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Validate periodic-restart hitting-time evidence for one seed round."""
    expected_name = f"seed-round-{round_index + 1:02d}"
    expected_tags = [window.tag for _, window in expected_windows]
    if (
        record.get("schema_version") != 3
        or record.get("record_kind") != "seed-hitting-time-ledger"
        or record.get("status") != "passed"
        or record.get("name") != expected_name
        or record.get("strategy")
        != "per-window-periodic-restart-hitting-time-capture"
        or record.get("selection_policy")
        != "earliest-positive-qualified-colvars-sample"
        or record.get("restart_policy")
        != "exact-pilot-binary-restart-at-colvars-frequency"
        or record.get("checkpoint_frequency_steps")
        != int(manifest["dynamics"]["colvars_frequency_steps"])
        or record.get("window_order") != expected_tags
        or record.get("worlds") != len(expected_tags)
        or record.get("ranks_per_window") != common["ranks_per_window"]
        or record.get("step_schedule") != list(step_schedule)
        or set(record.get("start_inputs", {})) != set(expected_tags)
        or set(record.get("outputs", {})) != set(expected_tags)
        or set(record.get("captures", {})) != set(expected_tags)
    ):
        raise ValueError(f"seed hitting-time protocol mismatch in {record_path}")

    for branch, window in expected_windows:
        tag = window.tag
        if record["start_inputs"][tag] != previous[branch]:
            raise ValueError(
                f"seed branch parent changed before {tag} in {record_path}"
            )
        capture = record["captures"][tag]
        attempt_index = capture.get("attempt_index")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
            or attempt_index >= len(step_schedule)
            or capture.get("attempt_number") != attempt_index + 1
        ):
            raise ValueError(f"seed capture attempt is invalid for {tag}")

        pilot_identity = capture.get("pilot_record")
        capture_identity = capture.get("capture_record")
        if not isinstance(pilot_identity, dict) or "path" not in pilot_identity:
            raise ValueError(f"seed pilot identity is missing for {tag}")
        if not isinstance(capture_identity, dict) or "path" not in capture_identity:
            raise ValueError(f"seed capture identity is missing for {tag}")
        pilot_path = Path(pilot_identity["path"])
        captured_path = Path(capture_identity["path"])
        if not artifact_matches(pilot_identity, pilot_path):
            raise ValueError(f"seed pilot record changed for {tag}")
        if not artifact_matches(capture_identity, captured_path):
            raise ValueError(f"seed capture record changed for {tag}")

        pilot = validate_invocation_record_current(pilot_path, common)
        expected_pilot_attempt = seed_attempt_metadata(
            manifest,
            [window],
            attempt_index=attempt_index,
            step_schedule=step_schedule,
        )
        expected_pilot_name = (
            f"seed-pilot-round-{round_index + 1:02d}-{tag}-"
            f"attempt-{attempt_index + 1:02d}"
        )
        if (
            pilot.get("name") != expected_pilot_name
            or pilot.get("steps_per_window") != step_schedule[attempt_index]
            or pilot.get("timestep_offset") != 0
            or pilot.get("worlds") != 1
            or pilot.get("window_order") != [tag]
            or pilot.get("ranks_per_window") != common["ranks_per_window"]
            or pilot.get("seed_attempt") != expected_pilot_attempt
            or pilot.get("start_inputs", {}).get(tag) != previous[branch]
            or pilot.get("restart_checkpoint_frequency_steps")
            != int(manifest["dynamics"]["colvars_frequency_steps"])
        ):
            raise ValueError(f"seed pilot protocol mismatch for {tag}")
        pilot_colvars = pilot.get("outputs", {}).get(tag, {}).get("colvars")
        if not isinstance(pilot_colvars, dict) or "path" not in pilot_colvars:
            raise ValueError(f"seed pilot Colvars identity is missing for {tag}")
        selected_row = first_seed_hitting_row(
            manifest, window, Path(pilot_colvars["path"])
        )
        if selected_row is None:
            raise ValueError(f"seed pilot has no accepted configuration for {tag}")
        expected_capture_attempt = seed_capture_metadata(
            manifest,
            window,
            attempt_index=attempt_index,
            step_schedule=step_schedule,
            selected_row=selected_row,
        )
        if (
            capture.get("selected_step")
            != expected_capture_attempt["selected_step"]
            or capture.get("selected_values")
            != expected_capture_attempt["selected_values"]
            or capture.get("selection_policy")
            != expected_capture_attempt["selection_policy"]
        ):
            raise ValueError(f"seed hitting-time selection changed for {tag}")

        checkpoint_identity = require_restart_checkpoint(
            pilot, tag, expected_capture_attempt["selected_step"]
        )
        if capture.get("checkpoint_restart") != checkpoint_identity:
            raise ValueError(f"seed checkpoint restart changed for {tag}")
        conversion_identity = capture.get("checkpoint_conversion_record")
        if not isinstance(conversion_identity, dict) or "path" not in conversion_identity:
            raise ValueError(f"seed checkpoint conversion is missing for {tag}")
        conversion_path = Path(conversion_identity["path"])
        if not artifact_matches(conversion_identity, conversion_path):
            raise ValueError(f"seed checkpoint conversion record changed for {tag}")
        conversion = validate_restart_conversion_record_current(
            conversion_path,
            restart_identity=checkpoint_identity,
            common=common,
        )
        converted_start = conversion["output"]

        captured = validate_invocation_record_current(captured_path, common)
        expected_capture_name = f"seed-capture-round-{round_index + 1:02d}-{tag}"
        if (
            captured.get("name") != expected_capture_name
            or captured.get("steps_per_window") != 0
            or captured.get("timestep_offset") != 0
            or captured.get("worlds") != 1
            or captured.get("window_order") != [tag]
            or captured.get("ranks_per_window") != common["ranks_per_window"]
            or captured.get("seed_attempt") != expected_capture_attempt
            or captured.get("start_inputs", {}).get(tag) != converted_start
        ):
            raise ValueError(f"seed capture protocol mismatch for {tag}")
        output_record = captured.get("outputs", {}).get(tag)
        expected_output = state_output(common["output"], "seeds", window)
        if (
            not isinstance(output_record, dict)
            or output_record.get("seed_acceptance") is not True
            or not artifact_matches(output_record.get("data"), expected_output)
            or record["outputs"][tag] != output_record
        ):
            raise ValueError(f"seed capture output changed for {tag}")
    return record


def validate_first_hit_seed_round_record(
    record_path: Path,
    record: dict[str, Any],
    *,
    manifest: dict[str, Any],
    round_index: int,
    expected_windows: Sequence[tuple[str, Window]],
    previous: dict[str, dict[str, str]],
    step_schedule: Sequence[int],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Validate same-process first-hit evidence and canonical byte copies."""
    expected_name = f"seed-round-{round_index + 1:02d}"
    expected_tags = [window.tag for _, window in expected_windows]
    check_frequency = int(manifest["dynamics"]["colvars_frequency_steps"])
    if (
        record.get("schema_version") != 4
        or record.get("record_kind") != "seed-first-hit-ledger"
        or record.get("name") != expected_name
        or record.get("status") != "passed"
        or record.get("strategy")
        != "per-window-in-process-first-hitting-time-capture"
        or record.get("selection_policy")
        != "earliest-positive-qualified-colvars-sample"
        or record.get("state_publication_policy")
        != "atomic-byte-copy-of-same-process-final-state"
        or record.get("halt_check_frequency_steps") != check_frequency
        or record.get("window_order") != expected_tags
        or record.get("worlds") != len(expected_tags)
        or record.get("ranks_per_window") != common["ranks_per_window"]
        or record.get("step_schedule") != list(step_schedule)
    ):
        raise ValueError(f"seed first-hit protocol mismatch in {record_path}")
    for section in ("start_inputs", "outputs", "captures", "attempts_by_window"):
        if set(record.get(section, {})) != set(expected_tags):
            raise ValueError(f"seed first-hit {section} changed in {record_path}")

    for branch, window in expected_windows:
        tag = window.tag
        if record["start_inputs"][tag] != previous[branch]:
            raise ValueError(
                f"seed branch parent changed before {tag} in {record_path}"
            )
        attempts = record["attempts_by_window"][tag]
        capture = record["captures"][tag]
        selected_attempt = capture.get("attempt_index")
        if (
            isinstance(selected_attempt, bool)
            or not isinstance(selected_attempt, int)
            or selected_attempt < 0
            or selected_attempt >= len(step_schedule)
            or len(attempts) != selected_attempt + 1
        ):
            raise ValueError(f"seed first-hit attempt chain is invalid for {tag}")

        accepted_record: dict[str, Any] | None = None
        accepted_identity: dict[str, str] | None = None
        for attempt_index, attempt_entry in enumerate(attempts):
            if (
                not isinstance(attempt_entry, dict)
                or attempt_entry.get("attempt_index") != attempt_index
                or attempt_entry.get("attempt_number") != attempt_index + 1
            ):
                raise ValueError(f"seed first-hit attempt order changed for {tag}")
            identity = attempt_entry.get("record")
            if not isinstance(identity, dict) or "path" not in identity:
                raise ValueError(f"seed first-hit record identity is missing for {tag}")
            invocation_path = Path(identity["path"])
            if not artifact_matches(identity, invocation_path):
                raise ValueError(f"seed first-hit record changed for {tag}")
            invocation = validate_invocation_record_current(
                invocation_path,
                common,
                allow_expected_first_hit_no_hit=True,
            )
            expected_attempt = seed_attempt_metadata(
                manifest,
                [window],
                attempt_index=attempt_index,
                step_schedule=step_schedule,
            )
            expected_invocation_name = (
                f"seed-first-hit-round-{round_index + 1:02d}-{tag}-"
                f"attempt-{attempt_index + 1:02d}"
            )
            if (
                invocation.get("name") != expected_invocation_name
                or invocation.get("steps_per_window") != step_schedule[attempt_index]
                or invocation.get("timestep_offset") != 0
                or invocation.get("worlds") != 1
                or invocation.get("window_order") != [tag]
                or invocation.get("ranks_per_window") != common["ranks_per_window"]
                or invocation.get("seed_attempt") != expected_attempt
                or invocation.get("stop_on_seed_acceptance") is not True
                or invocation.get("restart_checkpoint_frequency_steps", 0) != 0
                or invocation.get("start_inputs", {}).get(tag) != previous[branch]
            ):
                raise ValueError(f"seed first-hit invocation mismatch for {tag}")
            invocation_output = invocation.get("outputs", {}).get(tag)
            if not isinstance(invocation_output, dict):
                raise ValueError(f"seed first-hit output is missing for {tag}")
            accepted = invocation_output.get("seed_acceptance") is True
            if attempt_entry.get("seed_acceptance") is not accepted:
                raise ValueError(f"seed first-hit acceptance changed for {tag}")
            first_hit = invocation_output.get("first_hit")
            completed_steps = (
                first_hit.get("completed_steps")
                if isinstance(first_hit, dict)
                else None
            )
            scheduled_steps = (
                first_hit.get("scheduled_steps")
                if isinstance(first_hit, dict)
                else None
            )
            if (
                not isinstance(first_hit, dict)
                or isinstance(completed_steps, bool)
                or not isinstance(completed_steps, int)
                or isinstance(scheduled_steps, bool)
                or not isinstance(scheduled_steps, int)
                or first_hit.get("check_frequency_steps") != check_frequency
                or scheduled_steps != step_schedule[attempt_index]
                or attempt_entry.get("completed_steps")
                != completed_steps
                or first_hit.get("accepted_sample_found") is not accepted
                or first_hit.get("halted_early")
                is not (completed_steps < scheduled_steps)
                or invocation.get("completed_steps_by_window", {}).get(tag)
                != completed_steps
                or invocation.get("aggregate_window_steps") != completed_steps
            ):
                raise ValueError(f"seed first-hit stop evidence changed for {tag}")
            colvars_identity = invocation_output.get("colvars")
            if not isinstance(colvars_identity, dict) or "path" not in colvars_identity:
                raise ValueError(f"seed first-hit Colvars identity is missing for {tag}")
            selected_row = first_seed_hitting_row(
                manifest, window, Path(colvars_identity["path"])
            )
            if accepted:
                if selected_row != invocation_output.get("final_values"):
                    raise ValueError(f"seed first-hit final sample changed for {tag}")
            elif selected_row is not None:
                raise ValueError(f"rejected seed search contains an accepted hit for {tag}")
            if attempt_index < selected_attempt and accepted:
                raise ValueError(f"seed first-hit selection skipped an earlier hit for {tag}")
            if attempt_index == selected_attempt:
                if not accepted:
                    raise ValueError(f"selected seed first-hit attempt failed for {tag}")
                accepted_record = invocation
                accepted_identity = identity

        assert accepted_record is not None and accepted_identity is not None
        accepted_output = accepted_record["outputs"][tag]
        expected_capture = seed_first_hit_metadata(
            manifest,
            window,
            attempt_index=selected_attempt,
            step_schedule=step_schedule,
            selected_row=accepted_output["final_values"],
        )
        if (
            capture.get("attempt_number") != selected_attempt + 1
            or capture.get("first_hit_record") != accepted_identity
            or capture.get("selected_step") != expected_capture["selected_step"]
            or capture.get("selected_values") != expected_capture["selected_values"]
            or capture.get("selection_policy")
            != expected_capture["selection_policy"]
            or capture.get("halt_check_frequency_steps")
            != expected_capture["halt_check_frequency_steps"]
            or capture.get("state_publication_policy")
            != expected_capture["state_publication_policy"]
            or capture.get("source_data") != accepted_output["data"]
            or capture.get("source_restart") != accepted_output["restart"]
        ):
            raise ValueError(f"seed first-hit capture changed for {tag}")

        expected_data = state_output(common["output"], "seeds", window)
        expected_restart = expected_data.with_suffix(".restart")
        published_data = capture.get("published_data")
        published_restart = capture.get("published_restart")
        if (
            not artifact_matches(published_data, expected_data)
            or not artifact_matches(published_restart, expected_restart)
            or published_data.get("sha256") != accepted_output["data"].get("sha256")
            or published_restart.get("sha256")
            != accepted_output["restart"].get("sha256")
        ):
            raise ValueError(f"seed first-hit publication changed for {tag}")
        expected_output = dict(accepted_output)
        expected_output["data"] = published_data
        expected_output["restart"] = published_restart
        if record["outputs"][tag] != expected_output:
            raise ValueError(f"seed first-hit ledger output changed for {tag}")
    return record


def validate_seed_round_record(
    record_path: Path,
    *,
    manifest: dict[str, Any],
    round_index: int,
    expected_windows: Sequence[tuple[str, Window]],
    previous: dict[str, dict[str, str]],
    step_schedule: Sequence[int],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Validate one accepted seed round against its adaptive retry protocol."""
    expected_name = f"seed-round-{round_index + 1:02d}"
    expected_tags = [window.tag for _, window in expected_windows]
    try:
        candidate = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read seed record {record_path}: {error}") from error
    if candidate.get("record_kind") == "seed-hitting-time-ledger":
        return validate_hitting_time_seed_round_record(
            record_path,
            candidate,
            manifest=manifest,
            round_index=round_index,
            expected_windows=expected_windows,
            previous=previous,
            step_schedule=step_schedule,
            common=common,
        )
    if candidate.get("record_kind") == "seed-first-hit-ledger":
        return validate_first_hit_seed_round_record(
            record_path,
            candidate,
            manifest=manifest,
            round_index=round_index,
            expected_windows=expected_windows,
            previous=previous,
            step_schedule=step_schedule,
            common=common,
        )
    record = validate_invocation_record_current(record_path, common)
    metadata = record.get("seed_attempt")
    if not isinstance(metadata, dict):
        raise ValueError(f"seed attempt metadata is missing in {record_path}")
    attempt_index = metadata.get("attempt_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise ValueError(f"seed attempt index is invalid in {record_path}")
    if attempt_index < 0 or attempt_index >= len(step_schedule):
        raise ValueError(
            f"seed attempt is outside the reviewed schedule in {record_path}"
        )
    expected_metadata = seed_attempt_metadata(
        manifest,
        [window for _, window in expected_windows],
        attempt_index=attempt_index,
        step_schedule=step_schedule,
    )
    if metadata != expected_metadata:
        raise ValueError(f"seed attempt protocol mismatch in {record_path}")
    if (
        record.get("name") != expected_name
        or record.get("steps_per_window") != step_schedule[attempt_index]
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
        expected_output = state_output(common["output"], "seeds", window)
        output_record = record["outputs"][window.tag]
        if not artifact_matches(output_record.get("data"), expected_output):
            raise ValueError(f"seed output changed for {window.tag}")
        if output_record.get("seed_acceptance") is not True:
            raise ValueError(f"seed output was not accepted for {window.tag}")
    return record


def require_seed_records(
    output: Path,
    windows: Sequence[Window],
    anchor: Window,
    *,
    anchor_ledger: dict[str, Any],
    manifest: dict[str, Any],
    step_schedule: Sequence[int],
    common: dict[str, Any],
) -> dict[str, Path]:
    """Validate both adaptive seed branches and every parent/output hash edge."""
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
        record = validate_seed_round_record(
            record_path,
            manifest=manifest,
            round_index=round_index,
            expected_windows=expected_windows,
            previous=previous,
            step_schedule=step_schedule,
            common=common,
        )
        for branch, window in expected_windows:
            expected_output = state_output(output, "seeds", window)
            output_record = record["outputs"][window.tag]
            previous[branch] = output_record["data"]
            accepted[window.tag] = expected_output
    if set(accepted) != {window.tag for window in windows}:
        raise ValueError("seed branch DAG did not produce every requested window")
    return accepted


def inspect_dangerous_builds(
    log_directory: Path, expected_worlds: int
) -> dict[str, int | None]:
    """Require one completed neighbor-rebuild summary per partition log.

    ``None`` records LAMMPS' exact ``Dangerous builds not checked`` summary.
    It is accepted only by the caller's stronger every-step rebuild policy;
    this parser never silently translates an unchecked run into zero events.
    """
    results: dict[str, int | None] = {}
    for path in sorted(log_directory.glob("log.lammps*")):
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = DANGEROUS_BUILDS.findall(text)
        if matches:
            results[path.name] = int(matches[-1])
        elif DANGEROUS_BUILDS_NOT_CHECKED in text:
            results[path.name] = None
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
        # Preserve a convenient copy beside the superseded record. Generated
        # inputs are content-addressed and therefore remain immutable, while
        # this colocated copy keeps each stochastic attempt self-contained.
        try:
            stale_record = json.loads(record_path.read_text(encoding="utf-8"))
            stale_input = Path(stale_record.get("input", {}).get("path", ""))
        except (OSError, TypeError, json.JSONDecodeError):
            stale_input = Path()
        if stale_input.is_file():
            shutil.copy2(stale_input, archived / "input.lammps")
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


def lammps_backend_arguments(backend: str) -> tuple[str, ...]:
    """Return launcher arguments for the selected LAMMPS execution backend.

    The host path intentionally omits ``-k on``. This prevents non-owner MPI
    ranks from initializing otherwise unused Kokkos CUDA contexts while the
    GPU-local brokers continue to own their CUDA work through public APIs.
    """
    if backend == "host":
        return ()
    if backend == "kokkos":
        return (
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
    raise ValueError(f"unsupported LAMMPS execution backend: {backend}")


def require_restart_checkpoint(
    pilot_record: dict[str, Any], tag: str, selected_step: int
) -> dict[str, Any]:
    """Return and revalidate the pilot restart at one accepted Colvars step."""
    checkpoints = pilot_record.get("outputs", {}).get(tag, {}).get(
        "restart_checkpoints"
    )
    if not isinstance(checkpoints, list):
        raise ValueError(f"seed pilot has no restart checkpoints for {tag}")
    matches = [
        checkpoint
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict)
        and checkpoint.get("timestep") == selected_step
    ]
    if len(matches) != 1:
        raise ValueError(
            f"seed pilot has no unique restart at step {selected_step} for {tag}"
        )
    identity = matches[0]
    path = Path(str(identity.get("path", "")))
    if not artifact_matches(identity, path):
        raise ValueError(
            f"seed pilot restart changed at step {selected_step} for {tag}"
        )
    return identity


def archive_stale_restart_conversion(
    *,
    name: str,
    output: Path,
    record_path: Path,
    log_directory: Path,
    converted_data: Path,
) -> Path | None:
    """Preserve a failed checkpoint conversion before retrying its paths."""
    populated_logs = log_directory.is_dir() and any(log_directory.iterdir())
    if not record_path.exists() and not populated_logs and not converted_data.exists():
        return None
    archived = output / "superseded" / f"{name}-{time.time_ns()}"
    archived.mkdir(parents=True)
    if record_path.exists():
        record_path.replace(archived / "record.json")
    if populated_logs:
        log_directory.replace(archived / "logs")
        log_directory.mkdir(parents=True)
    if converted_data.exists():
        converted_data.replace(archived / "converted.data")
    print(f"archive: stale {name} artifacts moved to {archived}")
    return archived


def convert_restart_checkpoint_to_data(
    *,
    name: str,
    restart_identity: dict[str, Any],
    converted_data: Path,
    common: dict[str, Any],
) -> tuple[Path, Path]:
    """Convert one exact binary restart into a velocity-preserving data file.

    Seed propagation consumes data files, but the only exact state captured
    during a stochastic pilot is a LAMMPS binary restart.  This zero-dynamics
    conversion loads the same plugin/runtime boundary and writes ``nocoeff``
    data so downstream stages keep their reviewed force-field includes.
    """
    output: Path = common["output"]
    restart = Path(str(restart_identity["path"])).resolve()
    if not artifact_matches(restart_identity, restart):
        raise ValueError(f"restart checkpoint changed before conversion: {restart}")
    converted_data = converted_data.resolve()
    record_path = output / "records" / f"{name}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    log_directory = output / "logs" / name
    log_directory.mkdir(parents=True, exist_ok=True)
    launcher_log = log_directory / "launcher.log"

    plugin: Path = common["plugin"]
    lammps: Path = common["lammps"]
    xtbloom_library: Path = common["xtbloom_library"]
    manifest_path: Path = common["manifest_path"]
    mode = str(common.get("mode", "qmmm"))
    input_text = "\n".join(
        [
            "# Generated checkpoint conversion; do not hand-edit.",
            "clear",
            f"plugin load {ensure_lammps_token(plugin.resolve())}",
            f"read_restart {ensure_lammps_token(restart)}",
            f"write_data {ensure_lammps_token(converted_data)} nocoeff",
            "",
        ]
    )
    input_path = generated_lammps_input_path(output, name, input_text)
    write_generated(input_path, input_text)

    environment, selected_environment = build_runtime_environment(
        xtbloom_library,
        common["library_dirs"],
        common["cuda_visible_devices"],
        uses_deepmd=mode == "qmmm-dpa4c",
    )
    mpi_launcher = resolve_executable(common["mpiexec"])
    runtime_before = runtime_record(
        lammps,
        plugin,
        xtbloom_library,
        mpi_launcher,
        environment,
        deepmd_plugin=common.get("deepmd_plugin"),
        deepmd_models=common.get("deepmd_models", ()),
    )
    project_before = project_record(manifest_path, output)
    input_identity = {"path": str(input_path.resolve()), "sha256": sha256(input_path)}

    if record_path.is_file():
        try:
            previous = validate_restart_conversion_record_current(
                record_path,
                restart_identity=restart_identity,
                common=common,
            )
        except ValueError:
            pass
        else:
            if Path(previous["output"]["path"]).resolve() == converted_data:
                print(f"resume: {name} already converted an unchanged restart")
                return converted_data, record_path

    archive_stale_restart_conversion(
        name=name,
        output=output,
        record_path=record_path,
        log_directory=log_directory,
        converted_data=converted_data,
    )
    converted_data.parent.mkdir(parents=True, exist_ok=True)
    command = build_lammps_command(
        lammps=lammps,
        mpi_launcher=mpi_launcher,
        mpi_args=common["mpi_args"],
        worlds=1,
        ranks_per_window=common["ranks_per_window"],
        log_directory=log_directory,
        input_path=input_path,
        lammps_args=(
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
        ),
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
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_kind": "restart-checkpoint-to-data",
        "name": name,
        "status": "failed",
        "started_utc": started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "command": command,
        "returncode": process.returncode,
        "restart": restart_identity,
        "input": input_identity,
        "runtime": runtime_before,
        "project": project_before,
        "selected_environment": selected_environment,
        "environment_contract": runtime_environment_contract(selected_environment),
        "launcher_log": {
            "path": str(launcher_log.resolve()),
            "sha256": sha256(launcher_log),
        },
    }
    try:
        if process.returncode != 0:
            tail = launcher_log.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(
                f"LAMMPS restart conversion returned {process.returncode}:\n{tail}"
            )
        if not converted_data.is_file():
            raise ValueError("LAMMPS restart conversion did not write a data file")
        for label, identity, current in (
            ("restart", restart_identity, restart),
            ("generated input", input_identity, input_path),
            ("LAMMPS executable", runtime_before["lammps"], lammps),
            ("plugin", runtime_before["plugin"], plugin),
            ("xTBloom library", runtime_before["xtbloom"], xtbloom_library),
            ("runner", project_before["runner"], Path(__file__).resolve()),
            ("manifest", project_before["manifest"], manifest_path),
            ("provenance", project_before["provenance"], output / "provenance.json"),
        ):
            if not artifact_matches(identity, current):
                raise ValueError(f"{label} changed during restart conversion")
        record["output"] = {
            "path": str(converted_data),
            "sha256": sha256(converted_data),
        }
        record["status"] = "passed"
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        record["error"] = str(error)
        write_json_atomic(record_path, record)
        raise
    write_json_atomic(record_path, record)
    print(f"pass: {name}: converted restart checkpoint without dynamics")
    return converted_data, record_path


def validate_restart_conversion_record_current(
    record_path: Path,
    *,
    restart_identity: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate one checkpoint-to-data conversion and its dependency bytes."""
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read restart conversion record {record_path}: {error}"
        ) from error
    if (
        record.get("status") != "passed"
        or record.get("record_kind") != "restart-checkpoint-to-data"
        or record.get("restart") != restart_identity
    ):
        raise ValueError(f"restart conversion protocol changed in {record_path}")
    restart = Path(str(restart_identity.get("path", "")))
    if not artifact_matches(restart_identity, restart):
        raise ValueError(f"restart conversion input changed in {record_path}")
    for section in ("input", "output", "launcher_log"):
        identity = record.get(section)
        if (
            not isinstance(identity, dict)
            or "path" not in identity
            or not artifact_matches(identity, Path(identity["path"]))
        ):
            raise ValueError(f"restart conversion {section} changed in {record_path}")
    project = record.get("project", {})
    for name, current in (
        ("runner", Path(__file__).resolve()),
        ("manifest", common["manifest_path"]),
        ("provenance", common["output"] / "provenance.json"),
    ):
        if not artifact_matches(project.get(name), current):
            raise ValueError(
                f"restart conversion {name} identity changed in {record_path}"
            )
    runtime = record.get("runtime", {})
    mpi_launcher = resolve_executable(common["mpiexec"])
    for name, current in (
        ("lammps", common["lammps"]),
        ("plugin", common["plugin"]),
        ("xtbloom", common["xtbloom_library"]),
        ("mpiexec", mpi_launcher),
    ):
        if not artifact_matches(runtime.get(name), current):
            raise ValueError(
                f"restart conversion runtime {name} changed in {record_path}"
            )
    if common.get("mode", "qmmm") == "qmmm-dpa4c":
        deepmd_c_api = runtime.get("deepmd_c_api")
        if (
            not isinstance(deepmd_c_api, dict)
            or "path" not in deepmd_c_api
            or not artifact_matches(deepmd_c_api, Path(deepmd_c_api["path"]))
        ):
            raise ValueError(
                f"restart conversion DeePMD C API changed in {record_path}"
            )
        recorded_models = runtime.get("models")
        expected_models = tuple(common.get("deepmd_models", ()))
        if (
            not isinstance(recorded_models, list)
            or len(recorded_models) != len(expected_models)
            or not all(
                artifact_matches(identity, model)
                for identity, model in zip(
                    recorded_models, expected_models, strict=True
                )
            )
        ):
            raise ValueError(
                f"restart conversion DPA4c model identity changed in {record_path}"
            )
    if record.get("environment_contract") != runtime_environment_contract(
        record.get("selected_environment")
    ):
        raise ValueError(
            f"restart conversion environment contract changed in {record_path}"
        )
    return record


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
    restart_checkpoint_frequency: int = 0,
    stop_on_seed_acceptance: bool = False,
    dpa4c_models_qualified: bool = False,
    allow_unqualified_dpa4c_models: bool = False,
    lammps_execution_backend: str = "kokkos",
    thermostat_enabled: bool = True,
    seed_attempt: dict[str, Any] | None = None,
    process_attempt: dict[str, int] | None = None,
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
        lammps_execution_backend=lammps_execution_backend,
        thermostat_enabled=thermostat_enabled,
    )
    record_path = output / "records" / f"{name}.json"
    input_text = render_lammps_input(
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
        lammps_execution_backend=lammps_execution_backend,
        restart_checkpoint_frequency=restart_checkpoint_frequency,
        stop_on_seed_acceptance=stop_on_seed_acceptance,
        thermostat_enabled=thermostat_enabled,
    )
    input_path = generated_lammps_input_path(output, name, input_text)
    write_generated(
        input_path,
        input_text,
    )
    log_directory = output / "logs" / name
    log_directory.mkdir(parents=True, exist_ok=True)
    launcher_log = log_directory / "launcher.log"

    worlds = len(run_windows)
    mpi_launcher = resolve_executable(mpiexec)
    command = build_lammps_command(
        lammps=lammps,
        mpi_launcher=mpi_launcher,
        mpi_args=mpi_args,
        worlds=worlds,
        ranks_per_window=ranks_per_window,
        log_directory=log_directory,
        input_path=input_path,
        lammps_args=lammps_backend_arguments(lammps_execution_backend),
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
        restart_checkpoint_frequency=restart_checkpoint_frequency,
        stop_on_seed_acceptance=stop_on_seed_acceptance,
        ranks_per_window=ranks_per_window,
        lammps=lammps,
        plugin=plugin,
        xtbloom_library=xtbloom_library,
        mpiexec=mpi_launcher,
        runner_path=Path(__file__).resolve(),
        loaded_xtbloom=loaded_xtbloom,
        deepmd_c_api=runtime_before.get("deepmd_c_api"),
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
        lammps_execution_backend=lammps_execution_backend,
        thermostat_enabled=thermostat_enabled,
        seed_attempt=seed_attempt,
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
        "restart_checkpoint_frequency_steps": restart_checkpoint_frequency,
        "stop_on_seed_acceptance": stop_on_seed_acceptance,
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
        "environment_contract": runtime_environment_contract(
            selected_environment
        ),
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
    if seed_attempt is not None:
        # Keep the stochastic retry protocol next to the runtime evidence so
        # a later stage can prove which reviewed duration and seed were used.
        record["seed_attempt"] = seed_attempt
    if process_attempt is not None:
        # Process recovery is operational rather than scientific, but it must
        # remain visible so failed wall time is never mistaken for benchmark
        # evidence and a successful record does not conceal earlier crashes.
        record["process_attempt"] = process_attempt

    try:
        if process.returncode != 0:
            tail = launcher_log.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise LammpsExecutionError(
                f"LAMMPS returned {process.returncode}:\n{tail}"
            )
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
            deepmd_c_api = runtime_before["deepmd_c_api"]
            if not artifact_matches(deepmd_c_api, Path(deepmd_c_api["path"])):
                changed_inputs.append("DeePMD C API library")
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
        loaded_deepmd_after = verify_loaded_deepmd_c(
            plugin, environment, required=mode == "qmmm-dpa4c"
        )
        deepmd_c_api_after = (
            {
                "path": loaded_deepmd_after["resolved_path"],
                "sha256": loaded_deepmd_after["sha256"],
                "soname": loaded_deepmd_after["soname"],
            }
            if loaded_deepmd_after is not None
            else None
        )
        if deepmd_c_api_after != runtime_before.get("deepmd_c_api"):
            raise ValueError(
                "DeePMD C API loader resolution changed while LAMMPS was running"
            )
        dangerous = inspect_dangerous_builds(log_directory, worlds)
        require_zero_dangerous = manifest["protocol"]["seed_acceptance"][
            "require_zero_dangerous_builds"
        ]
        neighbor_every = int(manifest["dynamics"]["neighbor_every"])
        neighbor_check = bool(
            manifest["dynamics"].get("neighbor_check", True)
        )
        if require_zero_dangerous:
            if neighbor_check:
                if any(value is None for value in dangerous.values()):
                    raise ValueError(
                        "dangerous neighbor builds were not checked despite "
                        "the checked-rebuild contract"
                    )
                if any(value != 0 for value in dangerous.values()):
                    raise ValueError(
                        f"dangerous neighbor builds were reported: {dangerous}"
                    )
            else:
                # ``check no`` with every=1 and delay=0 rebuilds the neighbor
                # list before every force evaluation.  This is stronger than
                # accepting a nonzero warning counter: displacement-based
                # deferral is disabled, so an interaction cannot be missed
                # while waiting for a later rebuild opportunity.
                if neighbor_every != 1:
                    raise ValueError(
                        "unchecked neighbor displacement requires every-step "
                        "rebuilding"
                    )
                if any(value is not None for value in dangerous.values()):
                    raise ValueError(
                        "LAMMPS unexpectedly checked dangerous builds under "
                        "the every-step rebuild contract"
                    )
        record["dangerous_builds"] = dangerous
        record["neighbor_rebuild_policy"] = {
            "every_steps": neighbor_every,
            "delay_steps": 0,
            "displacement_check": neighbor_check,
            "rebuild_every_step": not neighbor_check and neighbor_every == 1,
        }
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
        rejected_first_hits: list[str] = []
        completed_steps_by_window: dict[str, int] = {}
        for item in run_windows:
            colvars_path = Path(str(item.colvars_prefix) + ".colvars.traj")
            rows = parse_colvars_rows(colvars_path)
            values = rows[-1]
            if stop_on_seed_acceptance:
                if values["step"] <= timestep_offset:
                    raise ValueError(
                        "first-hit seed run produced no positive-time Colvars sample"
                    )
                if values["step"] > completed_timestep:
                    raise ValueError(
                        "first-hit seed run exceeded its reviewed step schedule"
                    )
                if values["step"] % colvars_frequency != 0:
                    raise ValueError(
                        "first-hit seed run stopped outside the Colvars sampling grid"
                    )
                first_hit = next(
                    (
                        row
                        for row in rows
                        if row["step"] > timestep_offset
                        and seed_row_is_accepted(manifest, item.window, row)
                    ),
                    None,
                )
                if first_hit is not None and values != first_hit:
                    raise ValueError(
                        "first-hit seed run continued beyond its earliest accepted sample"
                    )
            else:
                first_hit = None
                if (
                    expected_colvars_step is not None
                    and values["step"] != expected_colvars_step
                ):
                    raise ValueError(
                        f"Colvars final step {values['step']} in {colvars_path} "
                        f"differs from expected {expected_colvars_step}"
                    )
            completed_steps_by_window[item.window.tag] = completed_steps_for_record(
                requested_steps=steps,
                timestep_offset=timestep_offset,
                final_colvars_step=values["step"],
                stop_on_seed_acceptance=stop_on_seed_acceptance,
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
            if stop_on_seed_acceptance:
                record["outputs"][item.window.tag]["first_hit"] = {
                    "check_frequency_steps": colvars_frequency,
                    "scheduled_steps": steps,
                    "completed_steps": completed_steps_by_window[item.window.tag],
                    "halted_early": values["step"] < completed_timestep,
                    "accepted_sample_found": first_hit is not None,
                }
                # A first-hit invocation is only a successful checkpoint when
                # the in-process halt actually observed the acceptance gate.
                # Keep this failure inside ``run_invocation`` so the evidence
                # record remains ``failed`` and cannot be mistaken for a
                # resumable successful run on the next invocation.
                if first_hit is None:
                    rejected_first_hits.append(
                        f"{item.window.tag}: no accepted Colvars sample within "
                        f"{steps} scheduled steps"
                    )
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
            if restart_checkpoint_frequency > 0:
                checkpoints = restart_checkpoint_paths(
                    item,
                    timestep_offset=timestep_offset,
                    steps=steps,
                    frequency=restart_checkpoint_frequency,
                )
                missing = [path for _, path in checkpoints if not path.is_file()]
                if missing:
                    raise ValueError(
                        "LAMMPS did not publish restart checkpoint for "
                        f"{item.window.tag}: {missing[0]}"
                    )
                record["outputs"][item.window.tag]["restart_checkpoints"] = [
                    {
                        "timestep": checkpoint_step,
                        "path": str(checkpoint_path.resolve()),
                        "sha256": sha256(checkpoint_path),
                    }
                    for checkpoint_step, checkpoint_path in checkpoints
                ]
            if require_seed_acceptance and not accepted:
                rejected_seeds.append(
                    f"{item.window.tag}: reaction error {reaction_error:.6f} "
                    f"Angstrom, angle error {angle_error:.6f} degree"
                )
        record["completed_steps_by_window"] = completed_steps_by_window
        record["aggregate_window_steps"] = sum(completed_steps_by_window.values())
        record["aggregate_window_steps_per_second"] = (
            record["aggregate_window_steps"] / wall_seconds
            if wall_seconds
            else None
        )
        if rejected_seeds:
            raise ValueError("seed acceptance failed: " + "; ".join(rejected_seeds))
        if rejected_first_hits:
            # A short pilot that did not hit the gate is an expected scientific
            # outcome for the adaptive seed schedule.  Publish it as a failed
            # record (never ``passed``/resumable), but return the record so the
            # caller can advance to the next reviewed pilot duration.
            record["status"] = "failed"
            record["failure_classification"] = "expected-first-hit-no-hit"
            record["error"] = (
                "first-hit acceptance failed: " + "; ".join(rejected_first_hits)
            )
            write_json_atomic(record_path, record)
            return record_path
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


def run_invocation_with_retries(
    *, maximum_attempts: int = 1, **arguments: Any
) -> Path:
    """Retry only process-level LAMMPS failures from an unchanged start state.

    ``run_invocation`` writes a failed evidence record before raising.  Its
    next call archives that record and every partial output before launching
    the same hash-pinned request again, so a retry cannot publish a mixture of
    failed and accepted window slices.  Scientific validation failures are not
    caught here and therefore remain immediate hard failures.
    """
    if maximum_attempts <= 0:
        raise ValueError("maximum LAMMPS attempts must be positive")
    name = str(arguments.get("name", "LAMMPS invocation"))
    for attempt in range(1, maximum_attempts + 1):
        current_arguments = dict(arguments)
        current_arguments["process_attempt"] = {
            "attempt_number": attempt,
            "maximum_attempts": maximum_attempts,
        }
        try:
            return run_invocation(**current_arguments)
        except LammpsExecutionError:
            if attempt == maximum_attempts:
                raise
            print(
                f"retry: {name}: LAMMPS process attempt "
                f"{attempt}/{maximum_attempts} failed; restarting from the "
                "last accepted input state"
            )
    raise AssertionError("unreachable LAMMPS retry loop")


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


def publish_artifact_copy(source: Path, destination: Path) -> dict[str, str]:
    """Atomically publish an exact byte copy of an immutable run artifact.

    Seed searches write into attempt-specific diagnostic directories so every
    rejected trajectory remains auditable.  Once a first-hit run succeeds,
    this helper publishes its same-process final state at the canonical branch
    path without parsing, rounding, replaying, or otherwise transforming it.
    """
    if not source.is_file():
        raise ValueError(f"source artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as handle:
            shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256(source) != sha256(destination):
        raise ValueError(f"published artifact differs from source: {destination}")
    return {"path": str(destination.resolve()), "sha256": sha256(destination)}


def stage_record_path(output: Path, stage: str) -> Path:
    """Return the only accepted native or explicitly adopted stage ledger."""
    return output / "records" / f"{stage}-complete.json"


def require_nve_stability_qualification(
    output: Path,
    manifest_path: Path,
    deepmd_models: Sequence[Path],
) -> dict[str, Any]:
    """Require the fixed-threshold, hash-pinned NVE gate before DPRc production."""
    path = output / "qualification/nve-stability.json"
    if not path.is_file():
        raise ValueError(f"required NVE stability qualification is missing: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read NVE qualification {path}: {error}") from error
    if result.get("status") != "passed":
        raise ValueError(f"NVE stability qualification did not pass: {path}")
    if result.get("scope") != "three-window-qmmm-dpa4c-nve-stability":
        raise ValueError(f"unexpected NVE qualification scope in {path}")

    inputs = result.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"NVE qualification inputs are missing from {path}")
    record_path = output / "records/nve-stability.json"
    if not artifact_matches(inputs.get("record"), record_path):
        raise ValueError("NVE qualification does not cover the current invocation record")
    if not artifact_matches(inputs.get("manifest"), manifest_path):
        raise ValueError("NVE qualification does not cover the current workload manifest")
    expected_models = [
        {"path": str(model.resolve()), "sha256": sha256(model)}
        for model in deepmd_models
    ]
    if inputs.get("models") != expected_models:
        raise ValueError("NVE qualification covers different DPA4c model bytes")

    expected_thresholds = {
        "minimum_samples": NVE_MINIMUM_SAMPLES,
        "maximum_absolute_drift_rate_kcal_mol_ps_atom": (
            NVE_MAXIMUM_ABSOLUTE_DRIFT_RATE_KCAL_MOL_PS_ATOM
        ),
        "maximum_absolute_net_drift_kcal_mol_atom": (
            NVE_MAXIMUM_ABSOLUTE_NET_DRIFT_KCAL_MOL_ATOM
        ),
        "minimum_mean_temperature_kelvin": NVE_MINIMUM_MEAN_TEMPERATURE_KELVIN,
        "maximum_mean_temperature_kelvin": NVE_MAXIMUM_MEAN_TEMPERATURE_KELVIN,
    }
    if result.get("thresholds") != expected_thresholds:
        raise ValueError("NVE qualification thresholds differ from the reviewed gate")
    return result


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
    colvars_profile: str = "sampling",
    lammps_attempts: int = 1,
) -> Path:
    """Run a long synchronized stage as an exact, resumable chunk chain.

    ``colvars_profile`` is normally the production-strength umbrella.  The
    anchor uses the already-reviewed stronger seed profile so its accepted
    final structure is not determined by a rare endpoint fluctuation in either
    the classical or QM/MM force path; equilibration and production always
    retain the sampling profile.
    """
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
                colvars_profile=colvars_profile,
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
                colvars_profile=colvars_profile,
            )
            for window in windows
        ]
        record_path = run_invocation_with_retries(
            maximum_attempts=lammps_attempts,
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
        "colvars_profile": colvars_profile,
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


def run_first_hit_stage(
    *,
    stage: str,
    manifest: dict[str, Any],
    window: Window,
    start_data: Path,
    stage_root: Path,
    total_steps: int,
    maximum_chunk_steps: int,
    trajectory_frequency: int,
    common: dict[str, Any],
    colvars_profile: str = "seed",
    lammps_attempts: int = 1,
) -> Path:
    """Run and publish a one-window stage at its earliest accepted sample.

    This path is intentionally a single in-process LAMMPS invocation.  A
    chunked endpoint restart can move away from a previously valid anchor,
    while ``fix halt`` evaluates the fixed gate on the same trajectory and
    writes the exact accepted state before the process exits.
    """
    if colvars_profile != "seed":
        raise ValueError("first-hit stages require the seed Colvars profile")
    output: Path = common["output"]
    ledger_path = output / "records" / f"{stage}-complete.json"
    expected_final_data = {window.tag: stage_root / window.tag / f"{window.tag}.data"}
    if ledger_path.is_file():
        try:
            require_completed_stage(
                ledger_path,
                stage=stage,
                windows=[window],
                expected_start_data={window.tag: start_data},
                expected_final_data=expected_final_data,
                total_steps=total_steps,
                maximum_chunk_steps=maximum_chunk_steps,
                trajectory_frequency=trajectory_frequency,
                common=common,
                colvars_profile=colvars_profile,
            )
        except (TypeError, ValueError) as error:
            print(f"resume rejected: {stage}: {error}")
        else:
            print(f"resume: {stage} first-hit ledger and dependency DAG are unchanged")
            return ledger_path

    run_window = RunWindow(
        window,
        start_data,
        stage_root / window.tag,
        output,
        seed_for(manifest, window, 2),
        colvars_profile=colvars_profile,
    )
    record_path = run_invocation_with_retries(
        maximum_attempts=lammps_attempts,
        name=f"{stage}-first-hit",
        run_windows=[run_window],
        steps=total_steps,
        timestep_offset=0,
        trajectory_frequency=trajectory_frequency,
        require_seed_acceptance=False,
        stop_on_seed_acceptance=True,
        **common,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    output_record = record["outputs"][window.tag]
    first_hit = output_record.get("first_hit")
    if (
        output_record.get("seed_acceptance") is not True
        or not isinstance(first_hit, dict)
        or first_hit.get("accepted_sample_found") is not True
    ):
        error = (
            f"{stage} did not encounter an accepted configuration within "
            f"{total_steps} steps"
        )
        # Keep this defensive transition for records produced by older runner
        # versions (or a test double): a no-hit invocation must never remain
        # marked ``passed`` because ``record_is_resumable`` keys exclusively
        # on that status field.
        if record.get("status") == "passed":
            record["status"] = "failed"
            record["error"] = error
            write_json_atomic(record_path, record)
        raise ValueError(error)

    ledger = {
        "schema_version": 1,
        "record_kind": "anchor-first-hit-ledger",
        "stage": stage,
        "status": "passed",
        "qualification": "native-first-hit",
        "window_order": [window.tag],
        "total_steps_per_window": total_steps,
        "maximum_chunk_steps": maximum_chunk_steps,
        "colvars_profile": colvars_profile,
        "record": {"path": str(record_path.resolve()), "sha256": sha256(record_path)},
        "outputs": {window.tag: output_record},
        "selected_step": first_hit["completed_steps"],
        "halt_check_frequency_steps": first_hit["check_frequency_steps"],
        "wall_seconds": record["wall_seconds"],
        "aggregate_window_steps": record["aggregate_window_steps"],
        "aggregate_window_steps_per_second": record[
            "aggregate_window_steps_per_second"
        ],
        "selection_policy": "earliest-positive-qualified-colvars-sample",
        "state_publication_policy": "same-process-final-state-at-first-hit",
    }
    write_json_atomic(ledger_path, ledger)
    return ledger_path


def run_hitting_time_seed_round(
    *,
    round_index: int,
    expected_windows: Sequence[tuple[str, Window]],
    round_start: dict[str, Path],
    step_schedule: Sequence[int],
    common: dict[str, Any],
    lammps_attempts: int = 1,
) -> Path:
    """Stop each pilot at its first qualified sample and publish that state.

    GPU/Kokkos trajectories are not guaranteed to replay bit-for-bit in a new
    process, even with the same nominal Langevin seed.  The generated LAMMPS
    input therefore evaluates the fixed gate directly from the ``fix colvars``
    global array and uses ``fix halt`` to end the original process at the
    earliest sampled hit.  Its ordinary final data and restart are the exact
    accepted state and are copied byte-for-byte to the canonical branch path.
    """
    output: Path = common["output"]
    manifest: dict[str, Any] = common["manifest"]
    round_number = round_index + 1
    round_name = f"seed-round-{round_number:02d}"
    record_path = output / "records" / f"{round_name}.json"

    canonical_windows = [
        RunWindow(
            window,
            round_start[branch],
            output / "states/seeds" / window.tag,
            output,
            seed_for(manifest, window, 3),
            colvars_profile="seed",
        )
        for branch, window in expected_windows
    ]
    archive_stale_invocation_artifacts(
        name=round_name,
        output=output,
        record_path=record_path,
        log_directory=output / "logs" / round_name,
        run_windows=canonical_windows,
    )

    captures: dict[str, dict[str, Any]] = {}
    attempts_by_window: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, Any] = {}
    start_inputs: dict[str, dict[str, str]] = {}
    for branch, window in expected_windows:
        parent = round_start[branch]
        parent_identity = {"path": str(parent.resolve()), "sha256": sha256(parent)}
        start_inputs[window.tag] = parent_identity
        attempts_by_window[window.tag] = []
        captured = False
        for attempt_index, pilot_steps in enumerate(step_schedule):
            pilot_attempt = seed_attempt_metadata(
                manifest,
                [window],
                attempt_index=attempt_index,
                step_schedule=step_schedule,
            )
            pilot_name = (
                f"seed-first-hit-round-{round_number:02d}-{window.tag}-"
                f"attempt-{attempt_index + 1:02d}"
            )
            pilot_window = RunWindow(
                window,
                parent,
                output
                / "diagnostic/seed-first-hit-searches"
                / f"round-{round_number:02d}"
                / f"attempt-{attempt_index + 1:02d}"
                / window.tag,
                output,
                pilot_attempt["thermostat_seeds"][window.tag],
                colvars_profile="seed",
            )
            pilot_path = run_invocation_with_retries(
                maximum_attempts=lammps_attempts,
                name=pilot_name,
                run_windows=[pilot_window],
                steps=pilot_steps,
                trajectory_frequency=0,
                require_seed_acceptance=False,
                seed_attempt=pilot_attempt,
                stop_on_seed_acceptance=True,
                **common,
            )
            pilot_record = json.loads(pilot_path.read_text(encoding="utf-8"))
            output_record = pilot_record["outputs"][window.tag]
            attempts_by_window[window.tag].append(
                {
                    "attempt_index": attempt_index,
                    "attempt_number": attempt_index + 1,
                    "record": {
                        "path": str(pilot_path.resolve()),
                        "sha256": sha256(pilot_path),
                    },
                    "seed_acceptance": output_record["seed_acceptance"],
                    "completed_steps": output_record["first_hit"][
                        "completed_steps"
                    ],
                }
            )
            colvars_path = Path(output_record["colvars"]["path"])
            selected_row = first_seed_hitting_row(manifest, window, colvars_path)
            if selected_row is None:
                print(
                    "retry: "
                    f"{round_name} {window.tag} first-hit search "
                    f"{attempt_index + 1}/{len(step_schedule)} had no accepted sample"
                )
                continue
            if (
                output_record.get("seed_acceptance") is not True
                or output_record.get("final_values") != selected_row
                or output_record.get("first_hit", {}).get(
                    "accepted_sample_found"
                )
                is not True
            ):
                raise ValueError(
                    f"{pilot_name} did not stop on its recorded first accepted sample"
                )

            capture_attempt = seed_first_hit_metadata(
                manifest,
                window,
                attempt_index=attempt_index,
                step_schedule=step_schedule,
                selected_row=selected_row,
            )
            canonical_directory = output / "states/seeds" / window.tag
            published_data = publish_artifact_copy(
                Path(output_record["data"]["path"]),
                canonical_directory / f"{window.tag}.data",
            )
            published_restart = publish_artifact_copy(
                Path(output_record["restart"]["path"]),
                canonical_directory / f"{window.tag}.restart",
            )
            published_output = dict(output_record)
            published_output["data"] = published_data
            published_output["restart"] = published_restart
            outputs[window.tag] = published_output
            captures[window.tag] = {
                "attempt_index": attempt_index,
                "attempt_number": attempt_index + 1,
                "first_hit_record": {
                    "path": str(pilot_path.resolve()),
                    "sha256": sha256(pilot_path),
                },
                "selected_step": capture_attempt["selected_step"],
                "selected_values": capture_attempt["selected_values"],
                "selection_policy": capture_attempt["selection_policy"],
                "halt_check_frequency_steps": capture_attempt[
                    "halt_check_frequency_steps"
                ],
                "state_publication_policy": capture_attempt[
                    "state_publication_policy"
                ],
                "source_data": output_record["data"],
                "source_restart": output_record["restart"],
                "published_data": published_data,
                "published_restart": published_restart,
            }
            captured = True
            break
        if not captured:
            raise ValueError(
                f"seed acceptance failed: {window.tag} had no accepted configuration "
                f"across {len(step_schedule)} reviewed first-hit searches"
            )

    ledger = {
        "schema_version": 4,
        "record_kind": "seed-first-hit-ledger",
        "name": round_name,
        "status": "passed",
        "strategy": "per-window-in-process-first-hitting-time-capture",
        "selection_policy": "earliest-positive-qualified-colvars-sample",
        "halt_check_frequency_steps": int(
            manifest["dynamics"]["colvars_frequency_steps"]
        ),
        "state_publication_policy": "atomic-byte-copy-of-same-process-final-state",
        "window_order": [window.tag for _, window in expected_windows],
        "worlds": len(expected_windows),
        "ranks_per_window": common["ranks_per_window"],
        "step_schedule": list(step_schedule),
        "start_inputs": start_inputs,
        "outputs": outputs,
        "captures": captures,
        "attempts_by_window": attempts_by_window,
    }
    write_json_atomic(record_path, ledger)
    return record_path


def run_stage(
    arguments: argparse.Namespace, manifest: dict[str, Any], windows: list[Window]
) -> None:
    """Execute one resumable scientific stage or an ordered stage prefix."""
    seed_max_attempts = int(getattr(arguments, "seed_max_attempts", 3))
    lammps_attempts = int(getattr(arguments, "lammps_attempts", 1))
    if lammps_attempts <= 0:
        raise ValueError("--lammps-attempts must be positive")
    seed_steps_by_attempt = seed_attempt_schedule(manifest, seed_max_attempts)
    output = arguments.output.resolve()
    tutorial = arguments.tutorial.resolve()
    initial_center = int(
        manifest["umbrella"]["available_initial_center_tenths_angstrom"]
    )
    by_center = {window.center_tenths: window for window in windows}
    anchor_window = by_center[initial_center]
    protocol = manifest["protocol"]
    trajectory_frequency = int(manifest["dynamics"]["trajectory_frequency_steps"])
    mode = getattr(arguments, "mode", "qmmm")
    initial_data = initial_data_for_mode(manifest, tutorial, mode)
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
    lammps_execution_backend = str(
        getattr(arguments, "lammps_execution_backend", "kokkos")
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
        "lammps_execution_backend": lammps_execution_backend,
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
            # The anchor is an initialization gate, not a production sample.
            # Use the strong, reviewed seed restraint for every force path so
            # thermal endpoint noise cannot turn an initially valid center
            # into a rejected parent for the two-branch seed walk.  The
            # equilibration and production stages below continue to use the
            # declared sampling restraint.
            colvars_profile="seed",
        )

    def accepted_seed_starts(anchor_ledger: dict[str, Any]) -> dict[str, Path]:
        return require_seed_records(
            output,
            windows,
            anchor_window,
            anchor_ledger=anchor_ledger,
            manifest=manifest,
            step_schedule=seed_steps_by_attempt,
            common=common,
        )

    requested = arguments.stage
    stages = [
        "smoke",
        "batch-smoke",
        "anchor",
        "seeds",
        "equilibrate",
    ]
    if mode == "qmmm-dpa4c":
        stages.append("nve-stability")
    stages.append("production")
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
        run_invocation_with_retries(
            maximum_attempts=lammps_attempts,
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
        run_invocation_with_retries(
            maximum_attempts=lammps_attempts,
            name=f"batch-smoke-{count}",
            run_windows=batch,
            steps=arguments.smoke_steps or int(protocol["smoke_steps"]),
            trajectory_frequency=0,
            require_seed_acceptance=False,
            **common,
        )

    if "anchor" in selected:
        run_first_hit_stage(
            stage="anchor",
            manifest=manifest,
            window=anchor_window,
            start_data=initial_data,
            stage_root=output / "states/anchor",
            total_steps=int(protocol["anchor_relaxation_steps"]),
            maximum_chunk_steps=arguments.chunk_steps,
            trajectory_frequency=trajectory_frequency,
            common=common,
            colvars_profile="seed",
            lammps_attempts=lammps_attempts,
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
            round_start = previous.copy()
            expected_windows = [
                (branch, by_center[centers[round_index]])
                for branch, centers in (("lower", lower), ("upper", upper))
                if round_index < len(centers)
            ]
            record_path = output / "records" / f"seed-round-{round_index + 1:02d}.json"
            previous_identities = {
                branch: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for branch, path in round_start.items()
            }
            if record_path.is_file():
                try:
                    record = validate_seed_round_record(
                        record_path,
                        manifest=manifest,
                        round_index=round_index,
                        expected_windows=expected_windows,
                        previous=previous_identities,
                        step_schedule=seed_steps_by_attempt,
                        common=common,
                    )
                except (TypeError, ValueError) as error:
                    print(
                        f"resume rejected: seed-round-{round_index + 1:02d}: {error}"
                    )
                else:
                    for branch, window in expected_windows:
                        previous[branch] = Path(
                            record["outputs"][window.tag]["data"]["path"]
                        )
                    if record.get("record_kind") == "seed-first-hit-ledger":
                        print(
                            f"resume: seed-round-{round_index + 1:02d} passed "
                            "with in-process first-hit captures"
                        )
                    elif record.get("record_kind") == "seed-hitting-time-ledger":
                        print(
                            f"resume: seed-round-{round_index + 1:02d} passed "
                            "with exact periodic-restart hitting-time captures"
                        )
                    else:
                        print(
                            f"resume: seed-round-{round_index + 1:02d} passed "
                            f"attempt {record['seed_attempt']['attempt_number']} with "
                            "an unchanged dependency chain"
                        )
                    continue
            record_path = run_hitting_time_seed_round(
                round_index=round_index,
                expected_windows=expected_windows,
                round_start=round_start,
                step_schedule=seed_steps_by_attempt,
                common=common,
                lammps_attempts=lammps_attempts,
            )
            record = validate_seed_round_record(
                record_path,
                manifest=manifest,
                round_index=round_index,
                expected_windows=expected_windows,
                previous=previous_identities,
                step_schedule=seed_steps_by_attempt,
                common=common,
            )
            for branch, window in expected_windows:
                previous[branch] = Path(record["outputs"][window.tag]["data"]["path"])
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
            lammps_attempts=lammps_attempts,
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

    if "nve-stability" in selected:
        if mode != "qmmm-dpa4c":
            raise ValueError("NVE stability gating is specific to QM/MM+DPA4c")
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
        nve_windows = representative_nve_windows(windows)
        nve_batch = [
            RunWindow(
                window,
                equilibrated[window.tag],
                output / "diagnostic/nve-stability" / window.tag,
                output,
                seed_for(manifest, window, 7),
            )
            for window in nve_windows
        ]
        nve_common = dict(common)
        nve_common["thermostat_enabled"] = False
        run_invocation_with_retries(
            maximum_attempts=lammps_attempts,
            name="nve-stability",
            run_windows=nve_batch,
            steps=arguments.nve_steps,
            trajectory_frequency=0,
            require_seed_acceptance=False,
            **nve_common,
        )

    if "production" in selected:
        if mode == "qmmm-dpa4c":
            require_nve_stability_qualification(
                output,
                arguments.manifest,
                deepmd_models,
            )
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
                lammps_attempts=lammps_attempts,
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
        choices=("classical", "qmmm", "qmmm-dpa4c"),
        default="qmmm",
        help=(
            "run the complete classical force field, batched xTB QM/MM, or "
            "QM/MM overlaid with one compact DPA4c DPRc model; the default "
            "preserves the historical qmmm workflow"
        ),
    )
    run.add_argument("--deepmd-model", type=Path, action="append", default=[])
    run.add_argument(
        "--model-deviation-frequency",
        type=int,
        default=0,
        metavar="STEPS",
        help=(
            "must be zero for the in-plugin C API batch path; model deviation "
            "is performed during offline model qualification"
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
    run.add_argument(
        "--lammps-execution-backend",
        choices=("host", "kokkos"),
        default="kokkos",
        help=(
            "run ordinary LAMMPS work on the host or through Kokkos; the host "
            "backend avoids one unused Kokkos CUDA context per umbrella window "
            "while broker-owned xTBloom and DeePMD work remains on the GPU"
        ),
    )
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
        "--lammps-attempts",
        type=int,
        default=1,
        help=(
            "maximum attempts for a process-level LAMMPS failure; validation "
            "failures are never retried and the default preserves fail-fast "
            "behavior"
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
            "nve-stability",
            "production",
            "through-anchor",
            "through-seeds",
            "through-equilibrate",
            "through-production",
        ),
        required=True,
    )
    run.add_argument("--trial", type=int, action="append")
    run.add_argument(
        "--seed-max-attempts",
        type=int,
        default=3,
        help=(
            "maximum deterministic Langevin attempts for a seed round whose "
            "endpoint misses the unchanged acceptance thresholds; each "
            "attempt uses the corresponding manifest-declared adaptive length"
        ),
    )
    run.add_argument("--smoke-window-count", type=int, default=2)
    run.add_argument(
        "--smoke-steps",
        type=int,
        help="diagnostic smoke length; defaults to the manifest smoke length",
    )
    run.add_argument(
        "--nve-steps",
        type=int,
        default=5000,
        help=(
            "thermostat-free stability diagnostic length per representative "
            "window; 5000 steps is 5 ps at the reviewed 1 fs timestep"
        ),
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
                deepmd_plugin=None,
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
                if arguments.nve_steps < 1:
                    raise ValueError("--nve-steps must be positive")
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
