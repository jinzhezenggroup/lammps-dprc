#!/usr/bin/env python3
"""Focused fail-closed tests for fused LAMMPS source generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_fused_lammps_sources",
    ROOT / "tools/generate_fused_lammps_sources.py",
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class FusedSourceGeneratorTest(unittest.TestCase):
    @staticmethod
    def valid_manifest() -> dict[str, object]:
        """Return the smallest complete manifest accepted by schema version 1."""
        return {
            "schema_version": 1,
            "upstream": "https://example.invalid/lammps",
            "revision": "1" * 40,
            "license": "GPL-2.0-only",
            "patch": {
                "path": "patches/fused.patch",
                "sha256": "2" * 64,
                "derived_from": {
                    "path": "evidence/original.patch",
                    "sha256": "3" * 64,
                    "baseline_revision": "4" * 40,
                },
            },
            "files": [
                {
                    "path": "src/QMMM-XTB/fix.cpp",
                    "input_sha256": "5" * 64,
                    "output_sha256": "6" * 64,
                }
            ],
        }

    def test_digest_match_mismatch_and_missing_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-fused-generator-") as temporary:
            artifact = Path(temporary) / "artifact.cpp"
            payload = b"reviewed source bytes\n"
            artifact.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            GENERATOR.require_digest(artifact, expected, "test artifact")
            with self.assertRaisesRegex(RuntimeError, "differs from reviewed"):
                GENERATOR.require_digest(artifact, "0" * 64, "test artifact")
            artifact.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing test artifact"):
                GENERATOR.require_digest(artifact, expected, "test artifact")

    def test_manifest_rejects_unknown_schema_and_empty_inventory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-fused-manifest-") as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest = self.valid_manifest()
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                GENERATOR.load_manifest(manifest_path)

            manifest = self.valid_manifest()
            manifest["files"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "has no files"):
                GENERATOR.load_manifest(manifest_path)

    def test_manifest_rejects_unknown_keys_hashes_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-fused-schema-") as temporary:
            manifest_path = Path(temporary) / "manifest.json"

            cases: list[tuple[str, dict[str, object], str]] = []
            manifest = self.valid_manifest()
            manifest["unexpected"] = True
            cases.append(("unknown", manifest, "keys differ from schema"))

            manifest = self.valid_manifest()
            manifest["patch"]["sha256"] = "A" * 64  # type: ignore[index]
            cases.append(("uppercase hash", manifest, "lowercase hexadecimal"))

            for unsafe in (
                "../src/QMMM-XTB/fix.cpp",
                "/src/QMMM-XTB/fix.cpp",
                "src\\QMMM-XTB\\fix.cpp",
                "src/OTHER/fix.cpp",
            ):
                manifest = self.valid_manifest()
                manifest["files"][0]["path"] = unsafe  # type: ignore[index]
                cases.append((unsafe, manifest, "path|POSIX|src/QMMM-XTB"))

            manifest = self.valid_manifest()
            manifest["files"].append(dict(manifest["files"][0]))  # type: ignore[union-attr,index]
            cases.append(("duplicate", manifest, "duplicate manifest source path"))

            for label, manifest, message in cases:
                with self.subTest(label=label):
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        GENERATOR.load_manifest(manifest_path)

    def test_symlink_escape_fails_before_staging_writes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dprc-fused-containment-") as temporary:
            root = Path(temporary)
            repository = root / "repository"
            lammps = root / "lammps"
            outside = root / "outside.cpp"
            output = root / "uncreated" / "generated"
            patch = repository / "patches" / "fused.patch"
            source = lammps / "src" / "QMMM-XTB" / "fix.cpp"
            manifest_path = repository / "manifest.json"

            patch.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            patch.write_bytes(b"empty reviewed patch\n")
            outside.write_bytes(b"outside source\n")
            source.symlink_to(outside)

            manifest = self.valid_manifest()
            manifest["patch"]["sha256"] = hashlib.sha256(  # type: ignore[index]
                patch.read_bytes()
            ).hexdigest()
            manifest["files"][0]["input_sha256"] = hashlib.sha256(  # type: ignore[index]
                outside.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "escapes its allowed root"):
                GENERATOR.generate(repository, lammps, manifest_path, output)
            self.assertEqual(outside.read_bytes(), b"outside source\n")
            self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
