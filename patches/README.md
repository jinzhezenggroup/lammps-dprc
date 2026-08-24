# Retained upstream source patches

## Private fused LAMMPS source patch

`lammps-qmmm-xtb-fused.patch` derives from the five patched fused-fix/PPPM
files in the recovered six-file LAMMPS QMMM-XTB patch. It additionally carries the
reviewed per-initialization direct-image Ewald cache invalidation required
when `change_box` alters the cell between successive runs. The independent
`qmmm_xtb_ewald.cpp` micro-optimization is deliberately excluded. Instead the
unmodified pinned Ewald source and header are compiled into the plugin under
the private `DPRCXtbEwald` name, so a host without the QMMM-XTB package works
and a host with it cannot interpose the original `QMMMXTBEwald` class.
Hybrid type-map inspection uses LAMMPS's common `PairHybrid` base rather than
the host-only overlay subclass, so `hybrid/overlay` and
`hybrid/overlay/kk` both recognize the batched TIP4P proxy as MM-only and skip
the redundant reference pair captures. The private adapter diagnostic is
broadcast within each LAMMPS partition so CUDA allocation, internal runtime,
validation, and true SCC failures remain distinguishable in launcher logs.

The original recovered patch, its baseline revision, and both SHA-256 values
are recorded in `config/fused_lammps_sources.json`. At configure time,
`tools/generate_fused_lammps_sources.py` verifies every pinned LAMMPS input,
applies this patch only in the build tree, and verifies every generated
output before compilation. No sibling LAMMPS worktree is modified.

The resulting GPL-2.0-only source is compiled into the private plugin under
project-specific class and adapter names. See `THIRD_PARTY_NOTICES.md` for
the unresolved combined-binary redistribution boundary.

## AmberTools 26 CUDA 12.9 compatibility patch

`ambertools26-cuda-12.9.patch` applies to AmberTools 26.0.0 after official
`update.1`. It extends the existing CUDA 12.7 configuration branch from
`CUDA_VERSION < 12.9` to `< 13.0`; CUDA 12.9 therefore uses the upstream
SM70--SM120 flag set without introducing a new architecture policy.

The source archive, official update, patch input/output, and retained patch
SHA-256 values are pinned in `config/quick_pbe0_engine.json`. The patch is
distributed as reviewed GPL-3.0-only derived build-system source, accompanied
by `LICENSES/GPL-3.0-only.txt`. AmberTools/QUICK sources and binaries are not
vendored or installed by this repository.
