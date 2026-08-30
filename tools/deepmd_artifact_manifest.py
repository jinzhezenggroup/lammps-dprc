#!/usr/bin/env python3
"""Create and verify a declared DeePMD C API artifact manifest.

The manifest binds an exact source checkout, matching public header, and
shared library by content.  It deliberately does not claim that the shared
library was built from the declared checkout; proving that relationship
requires trusted build-system attestation outside this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


class ManifestError(ValueError):
    """Raised when the declared DeePMD artifact cohort is inconsistent."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(source: Path, arguments: Sequence[str]) -> bytes:
    """Run one Git query against the declared DeePMD source checkout."""
    process = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.returncode != 0:
        raise ManifestError(process.stdout.decode(errors="replace").strip())
    return process.stdout


def snapshot_revision(source: Path) -> str:
    """Read the full revision marker used by stripped source exports."""
    marker = source / ".source-revision"
    try:
        revision = marker.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ManifestError(
            "DeePMD source export is missing .source-revision"
        ) from error
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ManifestError(
            "DeePMD .source-revision must contain one full lowercase Git revision"
        )
    return revision


def source_snapshot_state(source: Path) -> tuple[str, bool, str]:
    """Return a revision and deterministic digest for a source export."""
    revision = snapshot_revision(source)
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(b"file\0")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return revision, True, digest.hexdigest()


def source_state(source: Path) -> tuple[str, bool, str]:
    """Return revision, cleanliness, and a fingerprint of the complete state.

    A normal checkout uses Git's revision and worktree digest.  A stripped
    source export is accepted only when it carries a full `.source-revision`
    marker and is fingerprinted over every regular file.
    """
    if not (source / ".git").exists():
        return source_snapshot_state(source)
    revision = git_output(source, ("rev-parse", "HEAD")).decode().strip()
    status = git_output(
        source, ("status", "--porcelain=v1", "--untracked-files=all", "-z")
    )
    digest = hashlib.sha256()
    for label, payload in (
        (b"status\0", status),
        (b"worktree-diff\0", git_output(source, ("diff", "--binary", "HEAD"))),
        (
            b"index-diff\0",
            git_output(source, ("diff", "--cached", "--binary", "HEAD")),
        ),
    ):
        digest.update(label)
        digest.update(payload)

    untracked = git_output(
        source, ("ls-files", "--others", "--exclude-standard", "-z")
    )
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative = raw_path.decode("utf-8", errors="strict")
        path = source / relative
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if path.is_file():
            digest.update(bytes.fromhex(sha256(path)))
        else:
            digest.update(b"non-regular")
    return revision, not status, digest.hexdigest()


def c_api_version(header: Path) -> int:
    """Read the public C API version from an installed or source header."""
    match = re.search(
        r"^#define\s+DP_C_API_VERSION\s+(\d+)",
        header.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ManifestError(f"DP_C_API_VERSION is absent from {header}")
    return int(match.group(1))


def create_record(
    source: Path,
    include_dir: Path,
    library: Path,
    *,
    allow_dirty_source: bool,
) -> dict[str, Any]:
    """Describe one exact source/header/library cohort without local paths."""
    source = source.resolve()
    installed_header = (include_dir / "deepmd/c_api.h").resolve()
    source_header = (source / "source/api_c/include/c_api.h").resolve()
    for path in (source_header, installed_header, library):
        if not path.is_file():
            raise ManifestError(f"required DeePMD artifact is unavailable: {path}")

    revision, clean, state_sha256 = source_state(source)
    if not clean and not allow_dirty_source:
        raise ManifestError("DeePMD source checkout is dirty")
    source_header_sha256 = sha256(source_header)
    installed_header_sha256 = sha256(installed_header)
    if source_header_sha256 != installed_header_sha256:
        raise ManifestError(
            "installed deepmd/c_api.h does not match the declared source checkout"
        )
    version = c_api_version(installed_header)
    if version < 30:
        raise ManifestError("LAMMPS-DPRc requires DeePMD C API version 30 or newer")

    return {
        "schema_version": 1,
        "source_revision": revision,
        "source_clean": clean,
        "source_state_sha256": state_sha256,
        "c_api_version": version,
        "source_c_api_header_sha256": source_header_sha256,
        "installed_c_api_header_sha256": installed_header_sha256,
        "c_api_library_sha256": sha256(library.resolve()),
    }


def verify_record(
    manifest: Path,
    source: Path,
    include_dir: Path,
    library: Path,
    *,
    expected_revision: str,
    expected_library_sha256: str,
    allow_dirty_source: bool,
) -> dict[str, Any]:
    """Require a declared manifest to match current source and artifacts."""
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"invalid DeePMD artifact manifest: {error}") from error
    current = create_record(
        source,
        include_dir,
        library,
        allow_dirty_source=allow_dirty_source,
    )
    if recorded != current:
        raise ManifestError(
            "DeePMD artifact manifest does not match the declared source/header/library cohort"
        )
    if current["source_revision"] != expected_revision:
        raise ManifestError(
            "DeePMD manifest source revision differs from the configured pin"
        )
    if current["c_api_library_sha256"] != expected_library_sha256:
        raise ManifestError(
            "DeePMD manifest library SHA-256 differs from the configured artifact hash"
        )
    return current


def build_parser() -> argparse.ArgumentParser:
    """Build the writer/verifier command line used by users and CMake."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("write", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--include-dir", type=Path, required=True)
        command.add_argument("--library", type=Path, required=True)
        command.add_argument("--allow-dirty-source", action="store_true")
        if name == "write":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--expected-revision", required=True)
            command.add_argument("--expected-library-sha256", required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "write":
            record = create_record(
                arguments.source,
                arguments.include_dir,
                arguments.library,
                allow_dirty_source=arguments.allow_dirty_source,
            )
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = arguments.output.with_name(arguments.output.name + ".tmp")
            temporary.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, arguments.output)
        else:
            record = verify_record(
                arguments.manifest,
                arguments.source,
                arguments.include_dir,
                arguments.library,
                expected_revision=arguments.expected_revision,
                expected_library_sha256=arguments.expected_library_sha256,
                allow_dirty_source=arguments.allow_dirty_source,
            )
        print(json.dumps(record, sort_keys=True))
        return 0
    except ManifestError as error:
        print(f"DeePMD artifact-manifest check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
