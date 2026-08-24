#!/usr/bin/env python3
"""Verify license texts and provenance for retained derived source patches."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

GPL2_SHA256 = "be38e38d9482c2beae35408e413045b4363e0e067b0d9f31549047948637a7ed"
GPL3_SHA256 = "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
AMBERTOOLS_LABEL_PATCH_SHA256 = (
    "31eb58e3b11a2ddd8f42928c3aa8405d9b3ff7d5e85fd5aa9a37fc32eb323496"
)
AMBERTOOLS_XTB_LABEL_PATCH_SHA256 = (
    "c80d5aa39c8b9cd6fb081e17867de38ff14e36c07aeecfafd113d4edf005e58d"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_license(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 {actual} differs from reviewed {expected_sha256}"
        )


def require_license_set(root: Path, label_prefix: str) -> None:
    """Require both verbatim texts covering the retained derived patches."""
    require_license(
        root / "GPL-2.0-only.txt",
        GPL2_SHA256,
        f"{label_prefix} GPL-2.0-only license",
    )
    require_license(
        root / "GPL-3.0-only.txt",
        GPL3_SHA256,
        f"{label_prefix} GPL-3.0-only license",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--install-prefix", type=Path)
    arguments = parser.parse_args()
    if (arguments.source_root is None) == (arguments.install_prefix is None):
        parser.error("select exactly one of --source-root or --install-prefix")

    try:
        if arguments.source_root is not None:
            root = arguments.source_root.resolve()
            require_license_set(root / "LICENSES", "source")
            require_license(
                root / "patches/ambertools26-dprc-binary64-label.patch",
                AMBERTOOLS_LABEL_PATCH_SHA256,
                "retained AmberTools binary64 label patch",
            )
            require_license(
                root / "patches/ambertools26-dprc-xtb-label.patch",
                AMBERTOOLS_XTB_LABEL_PATCH_SHA256,
                "retained AmberTools xTB label extension",
            )
            notice = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
            required_notice_text = (
                "LICENSES/GPL-2.0-only.txt",
                "LICENSES/GPL-3.0-only.txt",
                "patches/ambertools26-dprc-binary64-label.patch",
                "patches/ambertools26-dprc-xtb-label.patch",
                "retained derived patch",
                "generated LAMMPS object code compiled into the plugin",
                "AmberTools, QUICK, and xTB executables/libraries",
                "private runtime artifacts",
                "NVIDIA cuFFT",
                "libcufft.so.11",
            )
            for text in required_notice_text:
                if text not in notice:
                    raise RuntimeError(f"third-party notice does not state {text!r}")
        else:
            prefix = arguments.install_prefix.resolve()
            require_license_set(
                prefix / "share" / "doc" / "lammps-dprc" / "LICENSES",
                "installed",
            )
            for filename in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
                path = prefix / "share" / "doc" / "lammps-dprc" / filename
                if not path.is_file():
                    raise RuntimeError(
                        f"missing installed documentation payload: {path}"
                    )
            for filename in ("user-guide.md", "lammps-build-and-run.md"):
                path = (
                    prefix
                    / "share"
                    / "doc"
                    / "lammps-dprc"
                    / "docs"
                    / filename
                )
                if not path.is_file():
                    raise RuntimeError(
                        f"missing installed user documentation payload: {path}"
                    )
        return 0
    except (OSError, RuntimeError) as error:
        print(f"license payload check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
