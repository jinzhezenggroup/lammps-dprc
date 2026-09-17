#!/usr/bin/env python3
"""Render a synchronized multi-partition xTBloom trajectory-rerun input."""

from __future__ import annotations

import argparse
from pathlib import Path


TIMESTEP_MARKER = b"ITEM: TIMESTEP\n"
STORE_FORCE_COMMAND = "fix pre_xtb all store/force"
QMMM_FIX_COMMAND = "fix qmmm qm qmmm/xtb/dprc"
STORED_FORCE_COLUMNS = (
    "f_pre_xtb[1]",
    "f_pre_xtb[2]",
    "f_pre_xtb[3]",
)


def _safe_path(path: Path) -> str:
    """Return an absolute LAMMPS token and reject unsupported whitespace."""
    value = str(path.resolve())
    if any(character.isspace() for character in value):
        raise ValueError(f"LAMMPS rerun paths cannot contain whitespace: {value}")
    return value


def discover_windows(data_root: Path) -> list[str]:
    """Discover exact ``window/window.data`` topology inputs."""
    windows: list[str] = []
    for path in sorted(data_root.glob("*/*.data")):
        if path.stem != path.parent.name:
            raise ValueError(f"data file does not match its window directory: {path}")
        windows.append(path.stem)
    if not windows:
        raise ValueError(f"no window data files found under {data_root}")
    if len(set(windows)) != len(windows):
        raise ValueError("window data identities are not unique")
    return windows


def resolve_windows(data_root: Path, requested: list[str] | None) -> list[str]:
    """Return a stable requested subset and reject unknown identities."""
    available = discover_windows(data_root)
    if requested is None:
        return available
    if not requested:
        raise ValueError("the requested rerun window set is empty")
    if len(set(requested)) != len(requested):
        raise ValueError("requested rerun windows are not unique")
    requested_set = set(requested)
    unknown = sorted(requested_set.difference(available))
    if unknown:
        raise ValueError(f"requested rerun window is unavailable: {unknown[0]}")
    # Follow the canonical topology order even if callers list windows in a
    # different order. This keeps partition identity deterministic.
    return [window for window in available if window in requested_set]


def count_dump_frames(path: Path) -> int:
    """Count LAMMPS dump frames without loading a large trajectory at once."""
    count = 0
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            payload = overlap + chunk
            count += payload.count(TIMESTEP_MARKER)
            overlap = payload[-(len(TIMESTEP_MARKER) - 1) :]
    if count < 1:
        raise ValueError(f"production trajectory contains no frames: {path}")
    return count


def require_synchronized_frame_counts(
    windows: list[str], trajectories: list[Path]
) -> int:
    """Reject a collective rerun whose partitions would exit unevenly.

    Every xTBloom batch call is collective over the selected LAMMPS partition
    roots. If one trajectory ends before another, the remaining partitions
    would enter a collective that their peers can no longer reach.
    """
    counts = {
        window: count_dump_frames(path)
        for window, path in zip(windows, trajectories, strict=True)
    }
    unique = sorted(set(counts.values()))
    if len(unique) != 1:
        summary = ", ".join(
            f"{window}={count}" for window, count in counts.items()
        )
        raise ValueError(
            "synchronized xTBloom rerun requires equal frame counts; " + summary
        )
    return unique[0]


def world_variable(name: str, values: list[Path]) -> str:
    """Render one partition-stable LAMMPS world variable."""
    if not values:
        raise ValueError(f"world variable {name} has no values")
    lines = [f"variable {name} world &"]
    for index, path in enumerate(values):
        suffix = " &" if index + 1 < len(values) else ""
        lines.append(f"  {_safe_path(path)}{suffix}")
    return "\n".join(lines)


def validate_component_template(payload: str) -> None:
    """Require diagnostic pre-fix and actual zero-QM baseline force outputs.

    The electronic component requires total minus same-engine zero-QM force.
    The pre-fix snapshot is retained only for diagnosis: it already contains
    QM-dependent PPPM forces and cannot serve as the classical baseline.
    """
    if 'dump mm_state all custom 1 ${mm_dump}' not in payload:
        raise ValueError("rerun template must emit the same-engine zero-QM force baseline")
    if STORE_FORCE_COMMAND not in payload:
        raise ValueError(
            "xTBloom rerun template must create 'fix pre_xtb all store/force'"
        )
    if QMMM_FIX_COMMAND not in payload:
        raise ValueError("xTBloom rerun template does not create the QM/MM fix")
    if payload.index(STORE_FORCE_COMMAND) > payload.index(QMMM_FIX_COMMAND):
        raise ValueError("the pre-xTB store/force fix must precede the QM/MM fix")
    missing = [column for column in STORED_FORCE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(
            "xTBloom rerun dump is missing stored pre-fix force columns: "
            + ", ".join(missing)
        )


def render(
    template_path: Path,
    data_root: Path,
    trajectory_root: Path,
    raw_root: Path,
    plugin_path: Path,
    forcefield_path: Path,
    window_names: list[str] | None = None,
) -> str:
    """Return one complete input whose partitions retain stable identities."""
    windows = resolve_windows(data_root, window_names)
    data_files = [data_root / window / f"{window}.data" for window in windows]
    trajectories = [
        trajectory_root / window / f"{window}.lammpstrj" for window in windows
    ]
    missing = [path for path in trajectories if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"production trajectory is unavailable: {missing[0]}")
    frame_count = require_synchronized_frame_counts(windows, trajectories)
    qmmm_dumps = [raw_root / window / "qmmm.dump" for window in windows]
    qmmm_tables = [raw_root / window / "qmmm.tsv" for window in windows]
    mm_tables = [raw_root / window / "mm.tsv" for window in windows]
    mm_dumps = [raw_root / window / "mm.dump" for window in windows]
    template = template_path.read_text(encoding="utf-8")
    validate_component_template(template)
    prefix = [
        "# Generated by tools/render_xtbloom_rerun.py; do not hand-edit.",
        f"# Synchronized partitions: {len(windows)}; frames/partition: {frame_count}.",
        f"variable dprc_plugin string {_safe_path(plugin_path)}",
        f"variable forcefield_file string {_safe_path(forcefield_path)}",
        world_variable("data_file", data_files),
        world_variable("trajectory", trajectories),
        world_variable("qmmm_dump", qmmm_dumps),
        world_variable("qmmm_table", qmmm_tables),
        world_variable("mm_table", mm_tables),
        world_variable("mm_dump", mm_dumps),
        "",
    ]
    return "\n".join(prefix) + template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--forcefield", type=Path, required=True)
    parser.add_argument(
        "--window",
        action="append",
        help="render only this window; repeat to form one equal-length batch",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = render(
        arguments.template,
        arguments.data_root,
        arguments.trajectory_root,
        arguments.raw_root,
        arguments.plugin,
        arguments.forcefield,
        arguments.window,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    windows = resolve_windows(arguments.data_root, arguments.window)
    print(f"rendered {len(windows)} synchronized rerun partitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
