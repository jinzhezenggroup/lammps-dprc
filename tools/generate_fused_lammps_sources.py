#!/usr/bin/env python3
"""Generate hash-pinned fused QMMM-XTB sources in a build directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_PREFIX = PurePosixPath("src/QMMM-XTB")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source or patch file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> None:
    """Reject missing or drifted external bytes before using them."""
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 {actual} differs from reviewed {expected}: {path}"
        )


def require_object_keys(
    value: Any, required: set[str], label: str
) -> dict[str, Any]:
    """Require one JSON object with exactly the reviewed schema keys."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        unknown = sorted(actual - required)
        raise RuntimeError(
            f"{label} keys differ from schema: missing={missing} unknown={unknown}"
        )
    return value


def require_string(value: Any, label: str) -> str:
    """Require a non-empty JSON string."""
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    """Require a canonical lowercase SHA-256 digest."""
    digest = require_string(value, label)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def require_revision(value: Any, label: str) -> str:
    """Require a canonical full Git object name."""
    revision = require_string(value, label)
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise RuntimeError(f"{label} must be 40 lowercase hexadecimal digits")
    return revision


def require_relative_posix_path(value: Any, label: str) -> PurePosixPath:
    """Reject absolute, non-normalized, native, and traversing manifest paths."""
    raw = require_string(value, label)
    if "\\" in raw:
        raise RuntimeError(f"{label} must use POSIX separators")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise RuntimeError(f"{label} must be a normalized relative POSIX path")
    if relative.as_posix() != raw:
        raise RuntimeError(f"{label} must be a normalized relative POSIX path")
    return relative


def contained_path(root: Path, relative: PurePosixPath, label: str) -> Path:
    """Resolve one validated relative path and keep symlinks inside its root."""
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its allowed root: {relative}") from error
    return candidate


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate the complete generation contract."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest = require_object_keys(
        manifest,
        {"schema_version", "upstream", "revision", "license", "patch", "files"},
        "fused LAMMPS source manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise RuntimeError("unsupported fused LAMMPS source manifest schema")
    upstream = require_string(manifest["upstream"], "manifest upstream")
    if not upstream.startswith("https://"):
        raise RuntimeError("manifest upstream must be an HTTPS URL")
    require_revision(manifest["revision"], "manifest revision")
    require_string(manifest["license"], "manifest license")

    patch = require_object_keys(
        manifest["patch"], {"path", "sha256", "derived_from"}, "manifest patch"
    )
    patch_path = require_relative_posix_path(patch["path"], "manifest patch path")
    if patch_path.parts[0] != "patches" or patch_path.suffix != ".patch":
        raise RuntimeError("manifest patch path must name a .patch below patches/")
    require_sha256(patch["sha256"], "manifest patch SHA-256")
    derived = require_object_keys(
        patch["derived_from"],
        {"path", "sha256", "baseline_revision"},
        "manifest patch provenance",
    )
    require_relative_posix_path(derived["path"], "manifest provenance path")
    require_sha256(derived["sha256"], "manifest provenance SHA-256")
    require_revision(derived["baseline_revision"], "manifest baseline revision")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("fused LAMMPS source manifest has no files")
    paths: set[str] = set()
    basenames: set[str] = set()
    for index, raw_entry in enumerate(files):
        entry = require_object_keys(
            raw_entry,
            {"path", "input_sha256", "output_sha256"},
            f"manifest file entry {index}",
        )
        relative = require_relative_posix_path(
            entry["path"], f"manifest file entry {index} path"
        )
        if relative.parent != SOURCE_PREFIX:
            raise RuntimeError(
                f"manifest file entry {index} must be directly below src/QMMM-XTB/"
            )
        if relative.as_posix() in paths:
            raise RuntimeError(f"duplicate manifest source path: {relative}")
        if relative.name in basenames:
            raise RuntimeError(f"duplicate generated basename: {relative.name}")
        paths.add(relative.as_posix())
        basenames.add(relative.name)
        require_sha256(entry["input_sha256"], f"manifest file entry {index} input SHA-256")
        require_sha256(entry["output_sha256"], f"manifest file entry {index} output SHA-256")
    return manifest


def verify_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Verify the flattened build-tree output against reviewed digests."""
    output_dir = output_dir.resolve()
    for entry in manifest["files"]:
        relative = require_relative_posix_path(entry["path"], "manifest source path")
        require_digest(
            contained_path(output_dir, PurePosixPath(relative.name), "generated output"),
            entry["output_sha256"],
            "generated fused LAMMPS source",
        )


def verify_inputs(
    repository_root: Path, lammps_root: Path, manifest: dict[str, Any]
) -> Path:
    """Verify every retained patch and upstream source byte before generation."""
    patch_entry = manifest["patch"]
    patch_relative = require_relative_posix_path(
        patch_entry["path"], "manifest patch path"
    )
    patch_path = contained_path(repository_root, patch_relative, "fused LAMMPS patch")
    require_digest(patch_path, patch_entry["sha256"], "fused LAMMPS patch")
    for entry in manifest["files"]:
        relative = require_relative_posix_path(entry["path"], "manifest source path")
        require_digest(
            contained_path(lammps_root, relative, "pinned LAMMPS source input"),
            entry["input_sha256"],
            "pinned LAMMPS source input",
        )
    return patch_path


def generate(
    repository_root: Path,
    lammps_root: Path,
    manifest_path: Path,
    output_dir: Path,
) -> None:
    """Stage verified inputs and atomically replace each generated output."""
    repository_root = repository_root.resolve()
    lammps_root = lammps_root.resolve()
    output_dir = output_dir.resolve()
    manifest = load_manifest(manifest_path)
    patch_path = verify_inputs(repository_root, lammps_root, manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="dprc-fused-lammps-", dir=output_dir.parent
    ) as temporary:
        staging_root = Path(temporary)
        for entry in manifest["files"]:
            relative = require_relative_posix_path(entry["path"], "manifest source path")
            destination = contained_path(staging_root, relative, "staged LAMMPS source")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = contained_path(lammps_root, relative, "pinned LAMMPS source input")
            shutil.copyfile(source, destination)

        process = subprocess.run(
            [
                "git",
                "-C",
                str(staging_root),
                "apply",
                "--whitespace=error-all",
                str(patch_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(f"could not apply fused LAMMPS patch:\n{process.stdout}")

        generated_dir = staging_root / "src" / "QMMM-XTB"
        verify_outputs(generated_dir, manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for entry in manifest["files"]:
            relative = require_relative_posix_path(entry["path"], "manifest source path")
            filename = relative.name
            temporary_output = output_dir / f".{filename}.tmp"
            shutil.copyfile(generated_dir / filename, temporary_output)
            temporary_output.replace(output_dir / filename)

    verify_outputs(output_dir, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--lammps-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an already generated output directory without modifying it",
    )
    arguments = parser.parse_args()

    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.check:
            verify_inputs(
                arguments.repository_root.resolve(),
                arguments.lammps_root.resolve(),
                manifest,
            )
            verify_outputs(arguments.output_dir.resolve(), manifest)
        else:
            generate(
                arguments.repository_root,
                arguments.lammps_root,
                arguments.manifest,
                arguments.output_dir,
            )
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"fused LAMMPS source generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
