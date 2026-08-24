# User guide

LAMMPS-DPRc provides a source-built LAMMPS plugin and an external workflow
runner for synchronized umbrella sampling with batched xTB QM/MM and an
optional DPA4c DPRc correction. The supported production shape is one MPI
partition per independent umbrella window and one physical GPU per synchronized
batch.

This repository does not ship a production DPA4c model. A QM/MM+DPA4c run is
scientifically meaningful only when the user supplies a model that has passed
the qualification program in [the DPA4c model plan](dpa4c-model-plan.md).

## Compatibility boundary

Build all runtime components for one exact environment. A plugin binary is not
portable across arbitrary LAMMPS versions, MPI implementations, LAMMPS integer
sizes, C++ compilers, accelerator configurations, or DeePMD builds.

The required runtime is:

- the pinned LAMMPS revision and a GPU-enabled Kokkos build;
- the pinned xTBloom revision, called only through its public C ABI;
- `dprcplugin.so` built against the matching LAMMPS headers, MPI, compiler,
  integer-size mode, xTBloom headers, CUDA architecture, and cuFFT artifacts;
- for QM/MM+DPA4c, a separately loaded DeePMD LAMMPS plugin that supports
  compact `center_group` input and `partition_batch yes`;
- one qualified DPA4c model for ordinary production, or four independently
  trained qualified models when model deviation is enabled;
- the external ETP/ETH tutorial checkout at the revision and artifact hashes in
  `workloads/etpeth/manifest.json`.

The current multi-partition DeePMD implementation has not yet been published as
a clean, hash-pinned release. Until that happens, QM/MM+DPA4c deployment is a
source-build workflow and cannot support a public release claim.

Run the dependency check before configuring:

```bash
python3 tools/check_dependency_pins.py --required-only
```

## Build LAMMPS

Follow the complete [LAMMPS build and launch guide](lammps-build-and-run.md).
It covers the pinned source checkout, shared Kokkos runtime required by
DeePMD, CUDA architecture, MPI and integer-size boundary, enabled LAMMPS
packages, xTBloom and plugin builds, ABI verification, direct single-window
`lmp` invocation, and multi-partition `mpiexec` command.

## Build the plugin

The complete build explanation and CPU reference configuration are in the
root [README](../README.md#configure-and-test). A production GPU configuration
must explicitly select the xTBloom library, CUDA compiler, GPU architecture,
cuFFT header and library, and their reviewed hashes. The essential shape is:

```bash
sha256sum /path/to/libxtbloom.so
sha256sum /path/to/cufft/include/cufft.h /path/to/libcufft.so

cmake -S . -B build/cuda-integration -G Ninja \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -DXTBLOOM_SOURCE_DIR=/path/to/xtbloom \
  -DXTBLOOM_GENERATED_INCLUDE_DIR=/path/to/xtbloom-build/generated/include \
  -DDPRC_XTBLOOM_LIBRARY=/path/to/libxtbloom.so \
  -DDPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256=<xtbloom-sha256> \
  -DDPRC_REQUIRE_XTBLOOM_LIBRARY=ON \
  -DDPRC_LAMMPS_EXECUTABLE=/path/to/kokkos/lmp \
  -DDPRC_ENABLE_CLASSICAL_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<gpu-architecture> \
  -DDPRC_CUFFT_INCLUDE_DIR=/path/to/cufft/include \
  -DDPRC_CUFFT_LIBRARY=/path/to/libcufft.so \
  -DDPRC_EXPECTED_CUFFT_HEADER_SHA256=<cufft-header-sha256> \
  -DDPRC_EXPECTED_CUFFT_LIBRARY_SHA256=<cufft-library-sha256>

cmake --build build/cuda-integration --parallel
ctest --test-dir build/cuda-integration --output-on-failure
```

Keep the LAMMPS executable, plugin, xTBloom library, DeePMD plugin, and models
unchanged for the lifetime of one resumable run directory.

## Verify and prepare the external workload

The runner never vendors tutorial inputs. It verifies the external source and
copies only generated runtime material into a separate run directory.

```bash
python3 tools/etpeth_workload.py verify \
  --tutorial ../dprc-tutorial

python3 tools/etpeth_workload.py prepare \
  --tutorial ../dprc-tutorial \
  --output ../lammps-dprc-runs/etpeth
```

If the tutorial checkout is dirty or its license is still unresolved, the
runner fails closed. `--allow-unqualified-source` permits only a private
diagnostic run; it does not qualify the source for publication or release.

## Smoke-test xTB QM/MM

Run one short window first:

```bash
python3 tools/etpeth_workload.py run \
  --tutorial ../dprc-tutorial \
  --output ../lammps-dprc-runs/etpeth-qmmm \
  --lammps /path/to/kokkos/lmp \
  --plugin /path/to/dprcplugin.so \
  --xtbloom-library /path/to/libxtbloom.so \
  --library-dir /path/to/cuda/lib64 \
  --mode qmmm \
  --stage smoke
```

Then exercise the synchronized broker with at least two windows:

```bash
<same command> --stage batch-smoke --smoke-window-count 2 --smoke-steps 25
```

Before timing, confirm that the one-window and batched correctness gates pass
and that every xTB slot retains its stable replica identity.

## Smoke-test QM/MM+DPA4c

For a scientifically qualified primary model, add the separately loaded
DeePMD plugin and an explicit qualification assertion:

```bash
python3 tools/etpeth_workload.py run \
  --tutorial ../dprc-tutorial \
  --output ../lammps-dprc-runs/etpeth-dpa4c \
  --lammps /path/to/kokkos/lmp \
  --plugin /path/to/dprcplugin.so \
  --xtbloom-library /path/to/libxtbloom.so \
  --deepmd-plugin /path/to/libdeepmd_lmp.so \
  --deepmd-model /path/to/qualified-dpa4c.pt2 \
  --mode qmmm-dpa4c \
  --dpa4c-models-qualified \
  --library-dir /path/to/cuda/lib64 \
  --stage batch-smoke \
  --smoke-window-count 2 \
  --smoke-steps 25
```

The generated input uses compact group-only DPA4c input and
`partition_batch yes`. One GPU-local owner evaluates the block-diagonal graph
for all synchronized windows; it does not load one independent model per
partition.

An unqualified or randomly initialized model may be used only for private
software and performance diagnostics. Replace the qualification assertion
with the explicit diagnostic opt-in:

```text
--allow-unqualified-source --allow-unqualified-dpa4c-models
```

The runner records that boundary in every invocation and refuses ambiguous
model status. Never present such a run as DPRc scientific evidence.

## Optional four-model deviation

A positive deviation frequency requires exactly four model arguments. Model
zero is the production primary and the other independent seeds are evaluated
on deviation steps:

```text
--deepmd-model model-0.pt2 \
--deepmd-model model-1.pt2 \
--deepmd-model model-2.pt2 \
--deepmd-model model-3.pt2 \
--model-deviation-frequency 100 \
--dpa4c-models-qualified
```

The current ensemble path uses DeePMD's generic adapter because the Kokkos pair
does not accept an ensemble. Treat its performance as a separate coordinate
from the single-model `deepmd/kk` partition-batched path.

## Run the resumable umbrella protocol

After both smoke tests pass, use the same immutable runtime arguments for each
stage:

```bash
<same command> --stage anchor
<same command> --stage seeds
<same command> --stage equilibrate
<same command> --stage production
```

The stages prepare the anchor, walk two seed branches across all 48 centers,
equilibrate every window, and run three production trials. The default
long-running stages are split into 5,000-step chunks. Tune `--chunk-steps`
only after measuring a representative batch and retain the chosen value in the
run provenance.

Each successful invocation publishes an atomic JSON record under `records/`.
Resume is accepted only if the full dependency chain still matches, including
the rendered input, step range, start state, LAMMPS executable, both plugins,
xTBloom library, every model, selected environment, manifest, provenance, and
all declared outputs. Positive-frequency model deviation also makes each
window's deviation output part of the resume contract.

If a launcher dies while holding the workspace lock, inspect the run first.
Use `--recover-stale-lock` only when the recorded process identity is proven
dead on the same host.

## Precision and scientific acceptance

LAMMPS and xTBloom exchange IEEE binary64 values in atomic units at the public
xTBloom C ABI. Any FP32 or mixed-precision experiment is opt-in and must be
qualified independently for energy, forces, charges, SCC behavior,
trajectories, and the final free-energy result against the FP64 reference.

One-window and batched results must agree within the declared scientific
tolerances before their timings are eligible. A model file being loadable is
not evidence that it represents the required PBE0-minus-xTB correction.

## Distribution limitations

The initial GPL-2.0-only LAMMPS and GPL-3.0-or-later xTBloom distribution
boundary remains unresolved. Do not distribute a combined binary until the
owner records a licensing decision. DeePMD and model artifacts have separate
license and provenance requirements. A public release must therefore contain
source and build instructions unless every binary and model boundary has been
reviewed explicitly.
