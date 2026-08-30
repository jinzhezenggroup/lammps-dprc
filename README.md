# LAMMPS-DPRc

LAMMPS-DPRc is the execution layer for correctness-qualified, high-throughput
umbrella sampling with batched xTB QM/MM and an optional DPRc correction. Its
primary target is aggregate free-energy sampling throughput on one GPU, not the
latency of an isolated force component.

The reference workload is the ETP/ETH system from `../dprc-tutorial`:

- multiple independent umbrella windows;
- 8,938 atoms with a 16-atom QM region;
- triclinic TIP4P-Ew solvent and a 50 x 50 x 50 PPPM mesh;
- xTBloom GFN2-xTB QM/MM followed by one DPA4c DPRc primary every step.

## Start here

- [User guide](docs/user-guide.md): build, runtime compatibility, smoke tests,
  production stages, resume, and model admission.
- [LAMMPS build and launch guide](docs/lammps-build-and-run.md): exact packages,
  Kokkos/CUDA/MPI compatibility, plugin verification, and direct `lmp` and
  multi-partition commands.
- [Diagnostic benchmark results](docs/benchmark-results.md): current MM,
  xTB QM/MM, and QM/MM+DPA4c measurements and their evidence limits.
- [Production DPA4c model plan](docs/dpa4c-model-plan.md): the required
  PBE0-minus-xTB data, training, qualification, and release program.
- [Project naming and packaging](docs/project-packaging.md): repository,
  package, benchmark, paper-artifact, and model-artifact boundaries.

No production DPA4c model is distributed with this repository. The current
QM/MM+DPA4c performance rows use an explicitly unqualified diagnostic graph
and are not scientific DPRc results.

## Repository boundary

- xTBloom remains the scientific engine and stable public C ABI.
- This repository owns LAMMPS-version-specific plugins, multi-partition batch
  coordination, fused QM/MM electrostatics, DPRc scheduling, and end-to-end
  evidence.
- Generic LAMMPS fixes should ultimately be proposed upstream. Temporary
  compatibility code here is pinned to exact LAMMPS revisions.
- No complete upstream source tree, model, dataset, or binary is vendored. A
  reviewed GPL-2.0-only fused delta is retained, then applied to five of seven
  hash-pinned LAMMPS inputs in the build tree. The build also generates a
  hash-pinned hardness subset there; see [third-party notices](THIRD_PARTY_NOTICES.md).

LAMMPS does not promise plugin ABI compatibility across versions, MPI
implementations, or build options. Build one plugin for each supported LAMMPS
configuration; do not reuse an arbitrary prebuilt plugin.

## Target execution model

```text
LAMMPS partition/window 0 --\
LAMMPS partition/window 1 ---+--> one node/GPU-local broker
LAMMPS partition/window 2 ---+       |
...                         --/       +--> one xTBloom ragged CUDA plan
                                    +--> one optional DeePMD canonical batch
                                         stable slot identity per window
```

Only independent windows or trajectories can be batched. Consecutive future
steps of one trajectory cannot be batched because step `t+1` depends on the
force at step `t`.

See [architecture](docs/architecture.md), [performance protocol](docs/performance.md),
[precision policy](docs/precision.md), and the [DeePMD compact-evaluation
reference](docs/deepmd-reference.md). The legacy-data audit and production xTB
correction requirements are documented in [DPRc labels](docs/dprc-labels.md).

## Current state

M1 now includes a force-producing reference integration. The repository
provides:

- exact dependency pins and a local pin checker;
- `dprc/info`, which collectively creates one communicator containing only
  partition roots and proves that `iworld`, root rank, and the dense stable
  window slot are identical, including under LAMMPS universe reordering;
- a plugin dynamic-symbol allowlist that exposes only `lammpsplugin_init`, so
  LAMMPS's global plugin loading cannot interpose project C++ implementation
  symbols;
- an allocation-stable ragged batch layout whose physical MM point-charge
  membership is a cutoff-plus-neighbor-skin superset refreshed by LAMMPS,
  while xTBloom uses permanent gamma-compatible padded slots across neighbor
  epochs; shell and unused slots carry exact zero charge, and synchronized
  timestep staging preserves whole-batch strict FRESH/WARM state;
- an xTBloom public-C-ABI fixed-topology plan owner;
- a root-to-root MPI broker in which only broker rank zero owns the xTBloom
  context/plan, while every other umbrella window contributes one ragged frame
  and receives only its own energy, force, charge, and status slices;
- a second root-to-root broker in which the same stable window slot feeds one
  GPU-local classical CUDA plan.  GPU-local roots exchange large frame and
  result slices through one persistent MPI shared-memory window instead of
  per-step Gather/Scatter copies.  Its `cufftPlanMany` handles execute the MM
  and QM reciprocal transforms for all synchronized windows, while one fused
  full-neighbor traversal computes full LJ plus MM-only TIP4P real-space
  Coulomb;
- the production styles `fix ... qmmm/xtb/dprc`, `pppm/dprc`, and
  `pppm/tip4p/dprc`, implemented by generating the reviewed fused LAMMPS
  fix/PPPM source under private class, helper, and adapter names, then routing
  its world-root numerical call through the partition broker and xTBloom
  public C ABI;
- explicit LAMMPS-real/atomic-unit, gradient/force, electronic-temperature,
  SCC-tolerance, charge/spin, and legacy MM-hardness translations;
- collective fixed-plan rebuild only when a window introduces a new screening
  gamma class or exceeds that class's monotonic padded capacity, with the
  previous whole-batch WARM checkpoint discarded;
- a two-window CPU QM/MM smoke test with screened point charges and the
  periodic `b + A q` operator, including serial-versus-batch and FRESH/WARM
  comparisons;
- orthogonal and triclinic PPPM and TIP4P comparisons against the pinned
  LAMMPS/libxTB style covering energy, correction, QM charges, every atom's
  force, and all six configurational pressure components; a two-step-size
  central-force-difference check; repeated and failure-recovery pending-field
  consumption; orthogonal and triclinic same-fix cell-change cache
  invalidation; explicit ik/per-atom/compute-no/mismatched-style rejection;
  two-rank numerical parity for orthogonal/triclinic PPPM and TIP4P; a
  cutoff-crossing test proving zero-charge shell equivalence plus FRESH-to-WARM
  reuse without a plan rebuild; and a two-partition ragged-topology rebuild
  test;
- an optional DeePMD C API v30+ integration inside `dprcplugin.so`, with
  center-mask, batch-2-versus-batch-1, symbol-boundary, and QM/MM overlay
  additivity tests;
- a hash-verifying, resumable external ETP/ETH runner that generates the full
  48-window restraint grid and the missing seed states without vendoring the
  dirty, license-unresolved tutorial inputs;
- a manifest-driven audit for the legacy MNDOD-to-PBE0 HDF5 payload, plus a
  fail-closed production contract for regenerating PBE0-to-xTB labels with full
  periodic QM/MM provenance and complete-molecule compact mapping;
- a hash-pinned AmberTools 26 update.1 / QUICK 25.03 CUDA PBE0/6-31G*
  qualification on the RTX 5090, backed by two real complete QM/MM force calls,
  plus a fail-closed binary64 periodic ETP/ETH label channel qualified for
  exact TIP4P force redistribution, atomic publication, selected finite
  differences, and same-process multi-frame reuse;
- a separate exact-commit AmberTools+xTB 6.7.0 low-level label extension and
  one full-system diagnostic `PBE0 - xTB` correction record whose source
  geometries/cells match bitwise and whose shared classical energy cancels
  exactly; xTBloom parity remains required before it becomes training data;
- the end-to-end benchmark contract and implementation roadmap.

The optional CUDA classical path registers
`lj/cut/dprc/batch`, `tip4p/long/dprc/batch`, and
`pppm/tip4p/dprc/batch`.  QM/MM performs one batched MM forward FFT plus four
inverse transforms, followed by one batched QM forward FFT plus three inverse
transforms.  Terminal pure classical execution skips the unused scalar field
and performs only one forward plus three inverse transforms.  Retained MM/QM
spectra provide the full reciprocal
energy and virial without a third charge-assignment FFT.  The pair proxies and
ordinary production KSpace call only consume the prepared publication and do
not launch per-window neighbor traversals or FFTs.  A GPU-built full Verlet
list removes rejected bin candidates from steady-state pair evaluation and is
reused until any synchronized window exceeds the conservative `skin/2`
displacement threshold.

The CUDA path is qualified against the pinned triclinic LAMMPS/libxTB
reference on an RTX 5090.  A two-partition end-to-end regression uses different
QM geometries in the two stable slots while sharing one classical CUDA and one
xTBloom owner.  The classical kernels are clean under memcheck, racecheck,
initcheck, and synccheck.  The complete LAMMPS chain is clean under the first
three tools; its only synccheck reports are the exact owner-disposed xTBloom
issue #279 Blackwell device-Graph signature, while the direct/host-Graph control
remains clean.  Nsight Systems on the RTX 5090 identified the original
atom-serial real-space traversal and contended spectral reductions; the path
now uses warp-owned full-neighbor evaluation, block-local PPPM reductions, a
persistent GPU Verlet list, and shared-memory MPI staging.  A dirty-tree
25-step warmup plus five 100-step diagnostic measured median aggregate rates
of `475.2, 686.3, 890.9, 1032.4, 1121.1, 1193.1, 1114.1` accepted steps/s/GPU
at batches `1, 2, 4, 8, 16, 32, 48`.  These are not release evidence because
the tutorial and this worktree remain unqualified/dirty.  A test-only cuFFT
linker trace keeps the transform contract executable; clean production
measurements and a qualified DPRc model remain required.

The corresponding padded-topology xTB QM/MM diagnostic measured median rates
of `26.92, 53.72, 99.32, 177.36, 281.43, 409.52, 427.58` accepted
steps/s/GPU. Every coordinate performed one initial plan creation, zero
capacity growths, and 525 strict WARM calls after the first FRESH call. This
removes the earlier cross-window neighbor-rebuild outliers but remains
development evidence under the same dirty/unqualified limitations.

The production styles use the distinct names `qmmm/xtb/dprc`, `pppm/dprc`,
and `pppm/tip4p/dprc`. They do not replace or override LAMMPS's existing
`qmmm/xtb`, `pppm/xtb`, or `pppm/tip4p/xtb` registrations. Native/private
style mismatches are rejected rather than silently losing the fused cache.
`pair_modify compute no`, `kspace_modify compute no`, and disabling only the
hybrid Coulomb sub-style are also rejected because they would remove a force
term required by the QM/MM subtraction. A prepared mesh is committed only
after the complete fix transaction succeeds, then bound to one timestep and
run/minimize setup phase; a caught exception discards that token locally and
the next base PPPM solve overwrites the abandoned field. A new DPRC pre-force
transaction also discards any token orphaned by a later force-pipeline failure
before the production KSpace call. Each new LAMMPS run/minimization also
invalidates the cached direct-image Ewald operator, so a command-driven cell
change between runs rebuilds its reciprocal coefficients.

The plugin does not define the original `FixQMMMXTB`, `PPPMXTB`,
`PPPMTIP4PXTB`, or `lammps_qmmm_xtb_*` symbols. The final ELF export allowlist
still contains only `lammpsplugin_init`, which prevents the private renamed
implementation from interposing on LAMMPS or another plugin under
`RTLD_GLOBAL` loading.

The DPRc runtime path is implemented inside `dprcplugin.so` through DeePMD's
public C API v30+. It registers `dprc/deepmd/batch` and the `/kk` alias, uses
compact group-only input, and requires `partition_batch yes`. One GPU-local
owner loads the model and evaluates a block-diagonal graph across synchronized
windows. The plugin has a direct `libdeepmd_c` dependency but no DeePMD C++ or
standalone LAMMPS-plugin dependency. The current path supports one primary
model; model deviation is rejected until it is implemented through the same
reviewed C API boundary.

The earlier standalone-adapter diagnostic measured median aggregate rates of
`23.56, 43.70, 74.66, 116.41, 156.30, 194.23` accepted steps/s/GPU at batches
`1, 2, 4, 8, 16, 32`. Batch 48 failed in xTBloom CUDA SCC plan creation before
a valid DPA4c timing sample. Those rows do not measure the current in-plugin C
API path. See the [benchmark report](docs/benchmark-results.md) for the full
evidence boundary and batch-32 timing decomposition.

## Configure and test

Use the same C++ compiler and MPI implementation as the selected LAMMPS
binary. A stale xTBloom binary is intentionally not auto-discovered: supply the
exact library built from the pinned source revision. For a CPU development
build, first build only the xTBloom shared-library target with the LAMMPS
toolchain (replace the compiler and LP64 provider paths as appropriate):

```bash
cmake -S ../xtbloom -B ../xtbloom/build/lammps-dprc-cpu -G Ninja \
  -DCMAKE_C_COMPILER=/path/from/lammps/CMakeCache/to/cc \
  -DCMAKE_CXX_COMPILER=/path/from/lammps/CMakeCache/to/c++ \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/absolute/path/to/lp64/lapacke-cblas.so \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ../xtbloom/build/lammps-dprc-cpu --target xtbloom --parallel
sha256sum ../xtbloom/build/lammps-dprc-cpu/libxtbloom.so
```

Then configure this repository:

```bash
cmake -S . -B build -G Ninja \
  -DLAMMPS_SOURCE_DIR="$PWD/../lammps/src" \
  -DXTBLOOM_SOURCE_DIR="$PWD/../xtbloom" \
  -DXTBLOOM_GENERATED_INCLUDE_DIR="$PWD/../xtbloom/build/lammps-dprc-cpu/generated/include" \
  -DDPRC_XTBLOOM_LIBRARY="$PWD/../xtbloom/build/lammps-dprc-cpu/libxtbloom.so" \
  -DDPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256=<sha256-printed-above> \
  -DDPRC_REQUIRE_XTBLOOM_LIBRARY=ON \
  -DDPRC_LAMMPS_EXECUTABLE="$PWD/../lammps/build-qmmm-xtb-gpu-cuda/lmp"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 tools/check_dependency_pins.py --required-only
```

Enable the batched classical CUDA backend only with an explicitly selected
CUDA compiler and reviewed cuFFT header/library pair.  The expected hashes are
mandatory because the header and runtime may come from separate external
toolkit payloads:

```bash
sha256sum /path/to/cufft/include/cufft.h /path/to/lib/libcufft.so.<exact>
cmake -S . -B build/cuda-integration -G Ninja \
  <the LAMMPS and xTBloom options above> \
  -DDPRC_ENABLE_CLASSICAL_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<actual-gpu-architecture> \
  -DDPRC_CUFFT_INCLUDE_DIR=/path/to/cufft/include \
  -DDPRC_CUFFT_LIBRARY=/path/to/lib/libcufft.so.<exact> \
  -DDPRC_EXPECTED_CUFFT_HEADER_SHA256=<cufft.h-sha256> \
  -DDPRC_EXPECTED_CUFFT_LIBRARY_SHA256=<libcufft-sha256>
```

When the production executable enables Kokkos but intentionally omits the
upstream QMMM-XTB package, provide a second ABI-compatible executable for the
pinned libxTB oracle.  The opt-in Kokkos regression then compares that host
reference with `full/kk + verlet/kk + hybrid/overlay/kk` and the batched
LJ/TIP4P/PPPM implementation, while also requiring the MM-only routing message
that proves redundant pair reference captures were skipped:

```bash
cmake -S . -B build/cuda-integration \
  <the CUDA integration options above> \
  -DDPRC_LAMMPS_EXECUTABLE=/path/to/kokkos/lmp \
  -DDPRC_REFERENCE_LAMMPS_EXECUTABLE=/path/to/qmmm-xtb/lmp \
  -DDPRC_ENABLE_KOKKOS_RUNTIME_TESTS=ON
```

Both executables must come from the pinned LAMMPS revision and use a plugin-
compatible compiler, MPI, integer-size, and shared-library configuration.

This backend currently requires one MPI rank per window, a fixed fully
periodic three-dimensional box, restricted triclinic or orthogonal geometry,
explicit mesh/order/gewald, ik differentiation, and global-only pair/KSpace
tallies.  It rejects slab, staggered/ad PPPM, r-RESPA, per-atom tallies, and
continuously changing cells rather than falling back to duplicate per-window
work.

To enable compact DeePMD batching, provide a DeePMD C API v30+ header and
shared library plus a declared, content-addressed artifact cohort:

```bash
sha256sum /path/to/libdeepmd_c.so /path/to/dprc-model.pt2
python3 tools/deepmd_artifact_manifest.py write \
  --source "$PWD/../deepmd-kit" \
  --include-dir /path/to/deepmd/include \
  --library /path/to/libdeepmd_c.so \
  --output /path/to/deepmd-artifact-manifest.json
cmake -S . -B build -G Ninja \
  <the xTBloom and LAMMPS options above> \
  -DDEEPMD_SOURCE_DIR="$PWD/../deepmd-kit" \
  -DDPRC_EXPECTED_DEEPMD_REVISION=<reviewed-api-v30-revision> \
  -DDPRC_DEEPMD_INCLUDE_DIR=/path/to/deepmd/include \
  -DDPRC_DEEPMD_C_LIBRARY=/path/to/libdeepmd_c.so \
  -DDPRC_DEEPMD_ARTIFACT_MANIFEST=/path/to/deepmd-artifact-manifest.json \
  -DDPRC_DEEPMD_MODEL=/path/to/dprc-model.pt2 \
  -DDPRC_EXPECTED_DEEPMD_C_LIBRARY_SHA256=<c-api-library-sha256> \
  -DDPRC_EXPECTED_DEEPMD_MODEL_SHA256=<model-sha256> \
  -DDPRC_REQUIRE_DEEPMD_C_API=ON \
  -DDPRC_ENABLE_KOKKOS_RUNTIME_TESTS=ON
cmake --build build --parallel
ctest --test-dir build -R deepmd --output-on-failure
```

The manifest proves that the declared checkout, public header, and library
bytes match the declared content identities. It is not build-system attestation and does
not by itself prove that the library was compiled from that checkout.

The tests record executable, plugin, model, revision, units, parity results,
and overlay additivity. A diagnostic model validates software integration only;
it cannot support a scientific DPRc claim. See the complete build and launch
instructions in [the LAMMPS guide](docs/lammps-build-and-run.md).

The exact compiler and MPI settings used to build LAMMPS may need to be passed
to CMake explicitly. `DPRC_XTBLOOM_BACKEND` selects `AUTO`, `CPU`, or `CUDA`;
`DPRC_XTBLOOM_DEVICE_ID` selects a CUDA ordinal (`-1` leaves selection to the
runtime), and `DPRC_XTBLOOM_CPU_THREADS` controls the one broker-owned CPU
context. Omitting `DPRC_XTBLOOM_LIBRARY` omits the xTB QM/MM styles and the
xTBloom `DT_NEEDED` entry; `dprc/info` and any explicitly enabled DeePMD C API
styles remain available. The dependency checker rejects dirty revisions,
mismatched pins, and mismatched reviewed artifact hashes by default because
such builds cannot support final correctness or performance claims.

The installed plugin intentionally has no build-machine RPATH. Run it in the
same LAMMPS/MPI toolchain environment and make the explicitly selected
`libxtbloom` available through the deployment's normal dynamic-loader search
path. A CUDA xTBloom build also lazy-loads the CUDA runtime/math cohort by
SONAME; put the matching toolkit's `lib64` directory ahead of any older CUDA
installation and expose the intended GPU. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/path/to/the-matching-cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  /path/to/lmp ...
```

Using a CUDA 12.9 xTBloom library with an older `libcudart.so.12` can fail even
though the major SONAME matches, because the older runtime may lack symbols
required by the build.

The complete umbrella runner and its staged smoke, anchor, seed,
equilibration, and production commands are documented in
[`examples/umbrella/README.md`](examples/umbrella/README.md). Current tutorial
runs are explicitly labeled private/diagnostic because the upstream checkout
is dirty, contains untracked generated LAMMPS inputs, and has no asserted
license grant. Long stages use hash-chained 5,000-step checkpoints and a
workspace lock so one failed batch does not silently mix old and new window
states or force a complete production restart.

## Performance objective

The headline metric is correctness-qualified aggregate MD steps per second per
GPU across all active umbrella windows. Secondary outputs include simulated
nanoseconds per day per GPU, time per force component, SCC iterations, GPU
memory, transfer volume, and the statistical uncertainty of the reconstructed
free-energy curve.

RTX 5090 FP32 is an explicit experiment axis. DPRc may use an FP32 model path;
xTBloom continues to accept and publish IEEE binary64 values. Any mixed-
precision xTB kernel remains experimental until force, charge, trajectory, and
free-energy evidence qualifies it.

## Distribution status

This repository currently grants no redistribution license. The intended
runtime crosses a GPLv2 LAMMPS and GPL-3.0-or-later xTBloom boundary that needs
an explicit owner decision before combined binaries are distributed. Private
research builds can proceed; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The verbatim GPL-2.0-only
text covering the retained patch and compiled generated LAMMPS material is in
[LICENSES/GPL-2.0-only.txt](LICENSES/GPL-2.0-only.txt) and is installed with
the documentation payload. The retained AmberTools build-system and binary64
label patches are accompanied by
[LICENSES/GPL-3.0-only.txt](LICENSES/GPL-3.0-only.txt); no AmberTools or QUICK
binary is distributed here.
