#!/usr/bin/env python3
"""Generate a small, explicitly unqualified DPA4c CUDA diagnostic model.

This artifact exists only to exercise the compact ``dprc/deepmd/batch/kk`` execution
path before an xTB-trained DPRc ensemble is available.  Randomized parameters
make energies and forces non-vacuous, but they have no scientific meaning and
must never qualify a DPRc correctness or performance row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

TYPE_MAP = ["C", "H", "HW", "O", "OW", "P"]
MM_ATOM_EXCLUDE_TYPES = [TYPE_MAP.index("HW"), TYPE_MAP.index("OW")]


def model_config(seed: int) -> dict:
    """Return the fixed compact DPA4c diagnostic architecture."""
    return {
        "type_map": TYPE_MAP,
        # DPRc is an atom-centered correction: MM atoms provide the compact
        # environment and receive reaction forces, but their fitting heads may
        # not contribute independent MM/MM correction energy.  The compressed
        # canonical kernel applies this model mask to the energy/gradient seed
        # without deleting the corresponding graph nodes.
        "atom_exclude_types": MM_ATOM_EXCLUDE_TYPES,
        "descriptor": {
            "type": "dpa4c",
            "rcut": 6.0,
            "channels": 8,
            "lmax": 2,
            "n_radial": 8,
            "precision": "float32",
            "seed": seed,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [32, 32],
            "activation_function": "silu",
            "precision": "float32",
            "resnet_dt": False,
            "seed": seed + 1,
        },
    }


def sha256(path: Path) -> str:
    """Hash one generated artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path) -> dict:
    """Read the deployment metadata embedded by the pt2 exporter."""
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist() if name.endswith("/metadata.json")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one embedded metadata.json, found {len(candidates)}"
            )
        return json.loads(archive.read(candidates[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--min-nbor-dist", type=float, default=0.5)
    parser.add_argument(
        "--runtime-overlay",
        type=Path,
        help=(
            "installed DeePMD prefix whose lib/ directory supplies "
            "run_config.ini and the native operators for this source tree"
        ),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists() and not arguments.force:
        raise FileExistsError(f"refusing to overwrite existing model: {output}")
    if output.suffix != ".pt2":
        raise ValueError("diagnostic DPA4c output must use the .pt2 suffix")
    if arguments.min_nbor_dist <= 0.0:
        raise ValueError("--min-nbor-dist must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("DPA4c compact-canonical export requires CUDA")
    torch.manual_seed(arguments.seed)
    torch.set_float32_matmul_precision("high")

    if arguments.runtime_overlay is not None:
        import deepmd

        overlay = arguments.runtime_overlay.resolve()
        runtime_package = overlay / "lib"
        if not (runtime_package / "run_config.ini").is_file():
            raise FileNotFoundError(
                f"runtime overlay lacks lib/run_config.ini: {overlay}"
            )
        # The source checkout owns ``deepmd`` itself, while a native build
        # owns its generated ``deepmd.lib`` payload.  Extend only this package
        # search path so Python code and native operators remain tied to the
        # explicitly selected build instead of an unrelated installed wheel.
        deepmd.__path__.append(str(overlay))

    from deepmd.pt_expt.model.get_model import get_model
    from deepmd.pt_expt.utils.serialization import deserialize_to_file

    model = get_model(model_config(arguments.seed)).to("cpu")
    # Several fitting projections are intentionally zero initialized.  A
    # deterministic perturbation prevents a zero-force artifact from making
    # host/device parity or throughput checks vacuous.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.copy_(torch.randn_like(parameter) * 0.02)
    model.eval()
    model.get_descriptor().enable_compression(min_nbor_dist=arguments.min_nbor_dist)

    output.parent.mkdir(parents=True, exist_ok=True)
    deserialize_to_file(
        str(output),
        {"model": model.serialize()},
        lower_kind="dpa4c_canonical",
    )

    deployment = metadata(output)
    expected = {
        "lower_input_kind": "dpa4c_canonical",
        "graph_edge_dtype": "float32",
        "canonical_index_dtype": "uint32",
        "type_map": TYPE_MAP,
    }
    mismatches = {
        key: {"expected": value, "actual": deployment.get(key)}
        for key, value in expected.items()
        if deployment.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"generated DPA4c metadata mismatch: {mismatches}")

    print(
        json.dumps(
            {
                "qualification": "unqualified-diagnostic-only",
                "path": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": sha256(output),
                "seed": arguments.seed,
                "min_nbor_dist_angstrom": arguments.min_nbor_dist,
                "atom_exclude_types": MM_ATOM_EXCLUDE_TYPES,
                "metadata": deployment,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
