#!/usr/bin/env python3
"""Check local sibling repositories against the reviewed dependency manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_output(repository: Path, *arguments: str) -> tuple[int, str]:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process.returncode, process.stdout.strip()


def sha256(path: Path) -> str:
    """Return the digest of one reviewed dependency artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dependency(root: Path, dependency: dict[str, Any]) -> dict[str, Any]:
    repository = (root / dependency["path"]).resolve()
    result: dict[str, Any] = {
        "name": dependency["name"],
        "path": str(repository),
        "required": bool(dependency["required"]),
        "expected_revision": dependency["revision"],
        "available": repository.is_dir(),
        "revision_matches": False,
        "clean": False,
        "artifacts_match": False,
    }
    if not result["available"]:
        return result

    status, revision = git_output(repository, "rev-parse", "HEAD")
    if status != 0:
        result["git_error"] = revision
        return result
    result["actual_revision"] = revision
    result["revision_matches"] = revision == dependency["revision"]

    status, dirty = git_output(repository, "status", "--porcelain")
    if status != 0:
        result["git_error"] = dirty
        return result
    result["clean"] = not dirty
    if dirty:
        result["dirty_entries"] = dirty.splitlines()

    artifacts = []
    all_artifacts_match = True
    for expected in dependency.get("artifacts", []):
        artifact_path = repository / expected["path"]
        artifact = {
            "path": expected["path"],
            "expected_sha256": expected["sha256"],
            "available": artifact_path.is_file(),
            "matches": False,
        }
        if artifact["available"]:
            artifact["actual_sha256"] = sha256(artifact_path)
            artifact["matches"] = (
                artifact["actual_sha256"] == artifact["expected_sha256"]
            )
        all_artifacts_match = all_artifacts_match and artifact["matches"]
        artifacts.append(artifact)
    result["artifacts"] = artifacts
    result["artifacts_match"] = all_artifacts_match
    return result


def dependency_state(result: dict[str, Any]) -> str:
    """Return the most actionable human-readable dependency failure."""
    if not result["available"]:
        return "unavailable"
    if not result["revision_matches"]:
        return "revision-mismatch"
    if not result["artifacts_match"]:
        return "artifact-mismatch"
    if not result["clean"]:
        return "dirty"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="check only dependencies required by the initial scaffold",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="report dirty checkouts without returning failure",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (repository_root / "config/dependencies.json").read_text(encoding="utf-8")
    )
    dependencies = manifest["dependencies"]
    if arguments.required_only:
        dependencies = [item for item in dependencies if item["required"]]

    results = [inspect_dependency(repository_root, item) for item in dependencies]
    failed = False
    for result in results:
        if not result["available"] or not result["revision_matches"]:
            failed = failed or result["required"] or not arguments.required_only
        if not result["clean"] and not arguments.allow_dirty:
            failed = failed or result["required"] or not arguments.required_only
        if not result["artifacts_match"]:
            failed = failed or result["required"] or not arguments.required_only

    if arguments.json:
        print(
            json.dumps(
                {
                    "schema_version": manifest["schema_version"],
                    "dependencies": results,
                },
                indent=2,
            )
        )
    else:
        for result in results:
            state = dependency_state(result)
            print(f"{result['name']}: {state} ({result['path']})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
