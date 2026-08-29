#!/usr/bin/env python3
"""Tests for the declared DeePMD source/header/library artifact boundary."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deepmd_artifact_manifest", ROOT / "tools/deepmd_artifact_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MANIFEST
SPEC.loader.exec_module(MANIFEST)


class DeepmdArtifactManifestTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, str]:
        source = root / "deepmd-kit"
        source_header = source / "source/api_c/include/c_api.h"
        include_dir = root / "install/include"
        installed_header = include_dir / "deepmd/c_api.h"
        library = root / "install/lib/libdeepmd_c.so"
        source_header.parent.mkdir(parents=True)
        installed_header.parent.mkdir(parents=True)
        library.parent.mkdir(parents=True)
        header = "#define DP_C_API_VERSION 31\n"
        source_header.write_text(header, encoding="utf-8")
        installed_header.write_text(header, encoding="utf-8")
        library.write_bytes(b"reviewed deepmd c api fixture\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.name", "Fixture"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "fixture@example.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(source), "commit", "-qm", "fixture"], check=True
        )
        revision = (
            subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            .stdout.strip()
        )
        return source, include_dir, library, revision

    def test_clean_record_binds_source_header_and_library(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-manifest-") as temporary:
            root = Path(temporary)
            source, include_dir, library, revision = self.fixture(root)
            record = MANIFEST.create_record(
                source, include_dir, library, allow_dirty_source=False
            )
            path = root / "artifact-manifest.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            verified = MANIFEST.verify_record(
                path,
                source,
                include_dir,
                library,
                expected_revision=revision,
                expected_library_sha256=MANIFEST.sha256(library),
                allow_dirty_source=False,
            )
            self.assertTrue(verified["source_clean"])
            self.assertEqual(verified["c_api_version"], 31)

    def test_header_mismatch_and_dirty_source_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-deepmd-manifest-") as temporary:
            root = Path(temporary)
            source, include_dir, library, _ = self.fixture(root)
            (include_dir / "deepmd/c_api.h").write_text(
                "#define DP_C_API_VERSION 32\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                MANIFEST.ManifestError, "does not match"
            ):
                MANIFEST.create_record(
                    source, include_dir, library, allow_dirty_source=False
                )

            source_header = source / "source/api_c/include/c_api.h"
            source_header.write_text(
                "#define DP_C_API_VERSION 32\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(MANIFEST.ManifestError, "dirty"):
                MANIFEST.create_record(
                    source, include_dir, library, allow_dirty_source=False
                )
            diagnostic = MANIFEST.create_record(
                source, include_dir, library, allow_dirty_source=True
            )
            self.assertFalse(diagnostic["source_clean"])


if __name__ == "__main__":
    unittest.main()
