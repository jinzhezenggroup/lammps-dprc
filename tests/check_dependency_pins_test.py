#!/usr/bin/env python3
"""Focused tests for dependency artifact provenance checks."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_dependency_pins", ROOT / "tools/check_dependency_pins.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class DependencyArtifactTest(unittest.TestCase):
    def test_match_mismatch_and_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-dependency-pin-") as temporary:
            root = Path(temporary)
            repository = root / "dependency"
            repository.mkdir()
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "test@example.invalid",
                ],
                check=True,
            )

            artifact = repository / "reviewed.dat"
            reviewed_bytes = b"reviewed dependency bytes\n"
            artifact.write_bytes(reviewed_bytes)
            subprocess.run(
                ["git", "-C", str(repository), "add", "reviewed.dat"], check=True
            )
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-qm", "fixture"],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            dependency = {
                "name": "fixture",
                "path": "dependency",
                "revision": revision,
                "required": True,
                "artifacts": [
                    {
                        "path": "reviewed.dat",
                        "sha256": hashlib.sha256(reviewed_bytes).hexdigest(),
                    }
                ],
            }

            matched = CHECKER.inspect_dependency(root, dependency)
            self.assertTrue(matched["artifacts_match"])
            self.assertEqual(CHECKER.dependency_state(matched), "ok")

            artifact.write_bytes(b"changed bytes\n")
            mismatched = CHECKER.inspect_dependency(root, dependency)
            self.assertFalse(mismatched["artifacts_match"])
            self.assertEqual(
                CHECKER.dependency_state(mismatched), "artifact-mismatch"
            )

            artifact.unlink()
            missing = CHECKER.inspect_dependency(root, dependency)
            self.assertFalse(missing["artifacts"][0]["available"])
            self.assertEqual(CHECKER.dependency_state(missing), "artifact-mismatch")


if __name__ == "__main__":
    unittest.main()
