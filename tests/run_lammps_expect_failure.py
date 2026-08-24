#!/usr/bin/env python3
"""Require a LAMMPS plugin scenario to fail with one precise diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--qmmm-style", required=True)
    parser.add_argument("--kspace-style", required=True)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="dprc-failure-") as temporary:
        result_file = Path(temporary) / "unexpected-result.txt"
        process = subprocess.run(
            [
                str(arguments.lammps),
                "-log",
                "none",
                "-var",
                "dprc_plugin",
                str(arguments.plugin),
                "-var",
                "qmmm_style",
                arguments.qmmm_style,
                "-var",
                "kspace_style",
                arguments.kspace_style,
                "-var",
                "result_file",
                str(result_file),
                "-in",
                str(arguments.input),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    evidence = {
        "schema_version": 1,
        "lammps": str(arguments.lammps.resolve()),
        "lammps_sha256": sha256(arguments.lammps),
        "plugin": str(arguments.plugin.resolve()),
        "plugin_sha256": sha256(arguments.plugin),
        "input": str(arguments.input.resolve()),
        "qmmm_style": arguments.qmmm_style,
        "kspace_style": arguments.kspace_style,
        "expected_diagnostic": arguments.expect,
        "returncode": process.returncode,
        "diagnostic_observed": arguments.expect in process.stdout,
    }
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if process.returncode == 0:
        print("LAMMPS unexpectedly accepted the fail-closed scenario", file=sys.stderr)
        return 1
    if arguments.expect not in process.stdout:
        print(
            f"LAMMPS failed without expected diagnostic {arguments.expect!r}:\n"
            f"{process.stdout}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
