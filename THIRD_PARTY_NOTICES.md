# Third-party notices and provenance

The repository vendors no complete third-party source tree, model, dataset,
binary, or build download. It retains one GPL-2.0-only derived patch for pinned
LAMMPS QMMM-XTB files, two GPL-3.0-only derived patches for AmberTools 26, and
the corresponding verbatim license texts. Dependencies are
otherwise supplied by the user from sibling checkouts or explicit runtime
paths and pinned in `config/dependencies.json`. The production build generates
private fused LAMMPS sources and a small xTBloom hardness subset only in its
build tree. Those bytes and their distribution boundary are recorded below.

## LAMMPS

- Upstream: <https://github.com/lammps/lammps>
- Initial reviewed revision: `7bd373ebc61a8028be955e6be862d5a091fd59c5`
- Stated license: GNU GPL version 2
- Classification: build headers, runtime host, and seven pinned source inputs
  transformed only in the build tree. Input and generated-output digests are
  recorded in `config/fused_lammps_sources.json`; the same input digests are
  independently checked by `config/dependencies.json`.
- Retained patch: `patches/lammps-qmmm-xtb-fused.patch`, SHA-256
  `8d4ceb3bc709f29edb068321a6b9f893aaf29d941d961b63f1d5eae6678307ba`.
  It derives from the fused fix/PPPM subset of the recovered six-file patch at
  SHA-256
  `f793e88d1ae7051eb5bbbaf7f6b9ac71e91f8ac460d4531713aa935d72f06ffd`
  based on LAMMPS revision `9ab8ca565e0f71d967587e0bca2015f7d689f19f`.
  The recovered `qmmm_xtb_ewald.cpp` optimization is excluded. The unmodified
  pinned Ewald source and header are instead compiled under the private
  `DPRCXtbEwald` name so the plugin does not require or interpose on the host's
  optional `QMMMXTBEwald` implementation.
  The retained delta additionally invalidates the direct-image Ewald cache at
  each LAMMPS initialization so a `change_box` command between runs cannot
  reuse reciprocal coefficients from the preceding cell.  It also makes the
  private fused PPPM publication hooks virtual, recognizes only the
  project-specific batched TIP4P proxy, and performs hybrid type-map checks
  through LAMMPS's common `PairHybrid` base so ordinary and Kokkos overlay
  styles preserve the same MM-only Coulomb classification.  This allows the
  verified derived class to replace per-window mesh work without registering
  or overriding an upstream LAMMPS style name. Adapter failures also preserve
  and broadcast the broker/runtime diagnostic instead of relabeling every
  failure as SCC nonconvergence.
- License text: `LICENSES/GPL-2.0-only.txt` is the verbatim license supplied by
  the pinned LAMMPS checkout. It covers the retained derived patch and the
  generated LAMMPS object code compiled into the plugin, and is installed with
  the project documentation payload.
- Generated code boundary: the seven verified outputs are compiled under
  private `FixDPRCXtbReference`, `PPPMDPRC`, `PPPMTIP4PDPRC`,
  `DPRCXtbEwald`, helper, and adapter names. No generated source is installed,
  but its object code is present in the plugin.
- ABI boundary: the plugin depends on the exact LAMMPS version, MPI library,
  integer-size configuration, compiler ABI, and enabled packages.

## xTBloom

- Upstream: <https://github.com/jinzhezenggroup/xtbloom>
- Initial reviewed revision: `3c474e1c1b639098f72ae7523472bd5f65ad3ab5`
- License: `GPL-3.0-or-later`, with its separately scoped CUDA/MKL additional
  permission
- Classification: public C ABI header at build time and an explicitly selected,
  runtime-provided shared library for the production force style and numerical
  integration tests; not vendored, installed, or bundled by this repository.
  Configuration records its resolved path and SHA-256 and can require a caller
  supplied expected digest. A production plugin linked this way has a direct
  xTBloom `DT_NEEDED` entry; the no-link diagnostic build does not.
- Generated build-tree subset: `tools/generate_xtb_hardness.py` reads only the
  per-element `gam` values from the canonical `gfn1.json` and `gfn2.json`
  parameter exports at SHA-256
  `0ecdc3f5f12990c5a7e0f0bd7e6fe931ecf72d7630e6a6a3cc396c51766a40a0`
  and `de0f20e90b592b7b92f107eb672bd3dd29c1096f904d7a472b05693f9238ed1a`.
  The primary source for those hardness values is tblite under
  `LGPL-3.0-or-later`. The deterministic header is generated only in the build
  tree and is incorporated into the plugin object code; it is not installed as
  a standalone parameter table.

## NVIDIA cuFFT

- Upstream: <https://developer.nvidia.com/cufft>
- Reviewed runtime version: cuFFT `11.4.1.4`, supplied externally from the
  NVIDIA CUDA distribution under the NVIDIA Software License Agreement.
- Classification: build header plus runtime-provided shared library for the
  opt-in batched classical CUDA backend. This repository does not download,
  vendor, install, or bundle NVIDIA headers or binaries.
- Reviewed external bytes: `cufft.h` SHA-256
  `949742cc832e5f966d8626b0cc39d0faa978f17898443a61d257b5dc9c1eb1e1` and
  `libcufft.so.11.4.1.4` SHA-256
  `8615db8574ae57b490200964ac25147dd231b3bf82a2bb19c91b619c29034d94`.
  CUDA builds require the caller to supply both expected hashes; they are not
  inferred from a toolkit path or SONAME.
- Binary boundary: a plugin built with `DPRC_ENABLE_CLASSICAL_CUDA=ON` has a
  direct `DT_NEEDED` entry for `libcufft.so.11`. The installed plugin has no
  retained RPATH, so deployment must supply the reviewed compatible runtime.

## DeePMD-kit compact evaluation and C API runtime

- Upstream: <https://github.com/deepmodeling/deepmd-kit>
- Reference PR: <https://github.com/deepmodeling/deepmd-kit/pull/5972>
- Pinned PR head: `bdb40072d57109e25e17842d24e1f0a926c00632`
- License: `LGPL-3.0-or-later`
- Classification: compact-selection design reference plus a caller-supplied
  public C API v30+ header, shared library, and frozen model. No DeePMD source,
  model, library, or license text is copied, installed, or bundled by this
  repository. The optional CMake path requires exact source revision, C API
  library SHA-256, and model SHA-256 values.
- Binary boundary: a DeePMD-enabled `dprcplugin` has a direct `libdeepmd_c`
  `DT_NEEDED` entry. It includes only `deepmd/c_api.h`; symbol validation
  rejects DeePMD C++ ABI symbols and standalone DeePMD LAMMPS-plugin
  dependencies. This C API boundary does not by itself decide derivative-work
  or redistribution questions.
- Status at initial review: the DPA4c graph revision is pinned explicitly and
  is not represented as a separately released LAMMPS plugin. The plugin uses
  only the public C API; its artifact manifest binds the source, header, and
  shared library before configuration succeeds.

## DPRc tutorial workload

- Upstream: <https://github.com/njzjz/dprc-tutorial>
- Initial reference revision: `f8716f28b03ef09734b74ae7f2ca67ab45c3d40f`
- Classification: external benchmark input and workflow reference; not vendored
- The local checkout was dirty during repository creation. Measurements from
  that checkout are diagnostic until repeated from clean, pinned inputs.
- `workloads/etpeth/manifest.json` records the exact SHA-256 of every inspected
  source and generated runtime input. `tools/etpeth_workload.py` reads those
  files only from the caller-selected external checkout and writes generated
  window controls and simulation outputs outside this source tree. It requires
  explicit `--allow-unqualified-source` for the current dirty, license-
  unresolved checkout; this opt-in permits private diagnostic execution but
  does not grant redistribution rights or qualify final benchmark evidence.

## AmberTools 26, QUICK 25.03, and xTB qualification runtimes

- Upstreams: <https://ambermd.org/AmberTools.php> and
  <https://quick-docs.readthedocs.io/>
- Reviewed source release: AmberTools 26.0.0 archive at SHA-256
  `5d46eef3c2bb7d5bf9e8c0c38add34406ea67e3f0e4097ac9d11d8a544538c9c`,
  followed by official `update.1` at SHA-256
  `8a2406339cafef5730eabc4fb0d39d4a47b29655048fb7887625c4b3edd134d9`.
- Components and stated licenses: AmberTools defaults to GPL-3.0-only with
  component-specific exceptions documented upstream; QUICK 25.03 states
  MPL-2.0. The upstream `AmberTools/LICENSE` and QUICK `license` files are
  pinned there at SHA-256
  `912af0215a173b10e44c254b1ee2ed844393298a503bea4596216afcb42ec509`
  and `1f256ecad192880510e84ad60474eab7589218784b9a50bc7ceee34c2b91f1d5`.
  The exact engine, basis, build, and runtime artifact identities are recorded
  in `config/quick_pbe0_engine.json`.
- Retained patch: `patches/ambertools26-cuda-12.9.patch`, SHA-256
  `02e4a34705b10359306680349c6da5ddeea1f1d0e4de4a2257a8a5af52a31154`.
  It changes only AmberTools' existing CUDA 12.7 branch upper bound from 12.9
  to 13.0 so CUDA 12.9 selects the already-defined SM120 flags. The patch is
  derived from AmberTools build-system source and is covered here by the
  verbatim `LICENSES/GPL-3.0-only.txt`, which is installed with the project
  documentation payload.
- Retained label patch: `patches/ambertools26-dprc-binary64-label.patch`,
  SHA-256
  `31eb58e3b11a2ddd8f42928c3aa8405d9b3ff7d5e85fd5aa9a37fc32eb323496`.
  It derives from the same reviewed AmberTools 26 update.1 Sander sources and
  adds a private, versioned binary64 trajectory/label channel, fail-closed
  single-point mode checks, exact PBE0/QM-region identity, successful
  QUICK-call accounting, positive-volume cell validation, and atomic
  no-overwrite output publication. The patch is covered by the same retained
  `LICENSES/GPL-3.0-only.txt`; no patched AmberTools source or resulting binary
  is installed or bundled.
- Retained xTB label extension:
  `patches/ambertools26-dprc-xtb-label.patch`, SHA-256
  `c80d5aa39c8b9cd6fb081e17867de38ff14e36c07aeecfafd113d4edf005e58d`.
  It applies only after the retained binary64 label patch, preserves the
  historical unset-engine behavior as the qualified `QUICK-PBE0` mode, and
  adds an explicit fail-closed `XTB-GFN2` mode with exact namelist and
  one-call-per-frame guards. It is derived from the same AmberTools Sander
  sources and is covered by `LICENSES/GPL-3.0-only.txt`; no patched source or
  executable is installed or bundled.
- The independent low-level oracle links AmberTools Sander against upstream
  xTB commit `edcfbbe39d411edc225e27315fbda3a204ddb023`, whose source reports
  version 6.7.0 and is licensed LGPL-3.0-or-later. The external xTB build uses
  fortran-lang/test-drive commit
  `9c3401e30dbd2da1add77aaa252a4c6928fe39a1` (MIT OR Apache-2.0) as a
  build/test dependency. Neither source tree, library, executable, parameter
  payload, nor test-drive artifact is copied into this repository or its
  install payload.
- Classification: retained source patch plus a hash-pinned external private
  qualification runtime. AmberTools, QUICK, and xTB executables/libraries
  remain external private runtime artifacts; no source archive, executable,
  shared library, basis/parameter file, or raw QM/MM trajectory is installed
  or bundled by this repository.
- Evidence: `workloads/etpeth/evidence/quick-pbe0-qualification.json` records
  two real CUDA PBE0/6-31G* Sander/QUICK force calls, their converged SCF data,
  complete QM/link and point-charge gradient extents, runtime hashes, loader
  contract, and GPU telemetry. It qualifies this engine installation only.
  The fixture is nonperiodic and therefore does not qualify ETP/ETH periodic
  embedding, TIP4P M-site force redistribution, or an independent PBE0 oracle.
- Periodic label evidence:
  `workloads/etpeth/evidence/quick-pbe0-binary64-label-qualification.json`
  records only compact hashes and derived numerical results. The complete
  11,912-site labels, input trajectories, executables, QUICK outputs, and GPU
  telemetry remain external private artifacts and are not redistributed.

## Distribution blocker

LAMMPS states GPL version 2 while xTBloom states GPL-3.0-or-later. Dynamic
loading does not itself establish compatibility. The repository therefore does
not grant redistribution rights for combined binaries. An owner/legal decision
must be recorded before public binary distribution.
