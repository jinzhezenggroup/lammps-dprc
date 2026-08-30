# Build and run LAMMPS

This guide builds a Kokkos/CUDA LAMMPS host, xTBloom, the DeePMD public C API,
and `dprcplugin.so`. It also shows direct one-window and multi-partition LAMMPS
commands. Every `/path/to` value is a user-supplied placeholder.

The supported QM/MM plus DPRc workflow loads only `dprcplugin.so`. DeePMD is a
linked C API dependency of that plugin, not a separately loaded LAMMPS plugin.

## 1. Create the source workspace

```bash
export DPRC_WORKSPACE=/path/to/dprc-workspace
mkdir -p "$DPRC_WORKSPACE"
cd "$DPRC_WORKSPACE"

git clone https://github.com/lammps/lammps.git
git -C lammps checkout 7bd373ebc61a8028be955e6be862d5a091fd59c5

git clone https://github.com/jinzhezenggroup/xtbloom.git
git -C xtbloom checkout 3c474e1c1b639098f72ae7523472bd5f65ad3ab5

git clone https://github.com/jinzhezenggroup/lammps-dprc.git
```

For QM/MM plus DPA4c, also obtain a clean DeePMD-kit revision that provides
public C API version 30 or newer. API v30 supplies the canonical-graph entry
point used by the packed fallback; API v31 or newer additionally supplies the
explicit frame-axis batch entry point:

```bash
git clone https://github.com/<publisher>/deepmd-kit.git deepmd-kit
git -C deepmd-kit checkout <reviewed-api-v30-or-newer-revision>
```

The currently recorded DeePMD design-reference pin provides the v30
canonical-graph entry point but predates the v31 explicit frame-axis
extension. Replace the placeholder only with a clean, immutable revision after
that API has been published and reviewed. A dirty local API implementation is
suitable for development tests, not a public reproducibility or performance
claim.

Verify the required public pins:

```bash
python3 lammps-dprc/tools/check_dependency_pins.py --required-only
```

## 2. Select one toolchain

The build requires CMake 3.24 or newer, Ninja, Python 3.11 or newer, one MPI
implementation, a CUDA toolkit accepted by the host compiler, and a matching
cuFFT runtime. Example variables for an RTX 5090 are:

```bash
export DPRC_CUDA_ROOT=/path/to/cuda
export DPRC_HOST_CC=/path/to/gcc
export DPRC_HOST_CXX=/path/to/g++
export DPRC_MPI_CC=/path/to/mpicc
export DPRC_MPI_CXX=/path/to/mpicxx
export DPRC_CUDA_ARCH=120
export DPRC_KOKKOS_ARCH=BLACKWELL120
export DPRC_CUFFT_INCLUDE_DIR="$DPRC_CUDA_ROOT/include"
export DPRC_CUFFT_LIBRARY="$DPRC_CUDA_ROOT/lib64/libcufft.so"
export PATH="$DPRC_CUDA_ROOT/bin:$PATH"
```

Change both architecture values for another GPU. LAMMPS and the plugin must
use the same MPI implementation, compiler ABI, and `LAMMPS_SIZES` mode. Do not
mix Open MPI and MPICH artifacts or reuse a plugin built for another LAMMPS
revision. The partition-batched DeePMD transport also requires MPI-3 shared
windows with the unified memory model (`MPI_WIN_UNIFIED`).

## 3. Build a shared Kokkos runtime

Build Kokkos from the pinned LAMMPS checkout so LAMMPS can use a shared CUDA
runtime:

```bash
cd "$DPRC_WORKSPACE"
export NVCC_WRAPPER_DEFAULT_COMPILER="$DPRC_HOST_CXX"

cmake -S lammps/lib/kokkos -B kokkos-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$DPRC_WORKSPACE/lammps/lib/kokkos/bin/nvcc_wrapper" \
  -DCMAKE_INSTALL_PREFIX="$DPRC_WORKSPACE/kokkos-install" \
  -DBUILD_SHARED_LIBS=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_${DPRC_KOKKOS_ARCH}=ON

cmake --build kokkos-build --parallel
cmake --install kokkos-build
```

## 4. Build the LAMMPS executable

The plugin supplies its own private QM/MM styles, so disable the upstream
`QMMM` and `QMMM-XTB` packages. `PKG_PLUGIN=ON` provides `plugin load`.
`EXTERNAL_KOKKOS=ON` is required when using the shared Kokkos installation.

```bash
cd "$DPRC_WORKSPACE"

cmake -S lammps/cmake -B lammps-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$DPRC_HOST_CC" \
  -DCMAKE_CXX_COMPILER="$DPRC_WORKSPACE/kokkos-install/bin/nvcc_wrapper" \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCUDAToolkit_ROOT="$DPRC_CUDA_ROOT" \
  -DMPI_C_COMPILER="$DPRC_MPI_CC" \
  -DMPI_CXX_COMPILER="$DPRC_MPI_CXX" \
  -DBUILD_MPI=ON \
  -DBUILD_OMP=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLAMMPS_SIZES=smallbig \
  -DPKG_PLUGIN=ON \
  -DPKG_KOKKOS=ON \
  -DEXTERNAL_KOKKOS=ON \
  -DKokkos_DIR="$DPRC_WORKSPACE/kokkos-install/lib/cmake/Kokkos" \
  -DKOKKOS_PREC=double \
  -DFFT_KOKKOS=CUFFT \
  -DPKG_KSPACE=ON \
  -DPKG_MOLECULE=ON \
  -DPKG_RIGID=ON \
  -DPKG_COLVARS=ON \
  -DPKG_QMMM=OFF \
  -DPKG_QMMM-XTB=OFF \
  -DPKG_GPU=OFF

cmake --build lammps-build --target lmp --parallel
```

Verify the executable and enabled packages:

```bash
sha256sum lammps-build/lmp
lammps-build/lmp -h > lammps-build/lmp-help.txt
rg 'KOKKOS|KSPACE|MOLECULE|RIGID|COLVARS|PLUGIN' \
  lammps-build/lmp-help.txt
ldd lammps-build/lmp
```

`KOKKOS_PREC=double` is the default. FP32 or mixed-precision variants are
separate experiments and require the qualification sequence in
`docs/precision.md`.

## 5. Build xTBloom

```bash
cd "$DPRC_WORKSPACE"

cmake -S xtbloom -B xtbloom/build/lammps-dprc-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$DPRC_HOST_CC" \
  -DCMAKE_CXX_COMPILER="$DPRC_HOST_CXX" \
  -DCMAKE_CUDA_COMPILER="$DPRC_CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$DPRC_HOST_CXX" \
  -DCMAKE_CUDA_ARCHITECTURES="$DPRC_CUDA_ARCH" \
  -DCUDAToolkit_ROOT="$DPRC_CUDA_ROOT" \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DXTBLOOM_ENABLE_TORCH_EXT=OFF \
  -DXTBLOOM_BUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=ON

cmake --build xtbloom/build/lammps-dprc-cuda \
  --target xtbloom --parallel

export DPRC_XTBLOOM_LIBRARY="$DPRC_WORKSPACE/xtbloom/build/lammps-dprc-cuda/libxtbloom.so"
export DPRC_XTBLOOM_SHA256="$(sha256sum "$DPRC_XTBLOOM_LIBRARY" | awk '{print $1}')"
```

If the selected environment requires an explicit CPU eigensolver provider,
add `-DXTBLOOM_CPU_LINALG_LIBRARY=/path/to/lp64/lapacke-cblas.so`.

## 6. Build the DeePMD C API

This step is required only for QM/MM plus DPA4c. Build the public C API and a
CUDA-capable PyTorch backend from the reviewed API-v30-or-newer revision. The exact
PyTorch prefix is installation-specific.

```bash
cd "$DPRC_WORKSPACE"
export DPRC_DEEPMD_PREFIX="$DPRC_WORKSPACE/deepmd-install"

cmake -S deepmd-kit/source -B deepmd-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$DPRC_HOST_CC" \
  -DCMAKE_CXX_COMPILER="$DPRC_HOST_CXX" \
  -DCMAKE_CUDA_ARCHITECTURES="$DPRC_CUDA_ARCH" \
  -DCUDAToolkit_ROOT="$DPRC_CUDA_ROOT" \
  -DCMAKE_PREFIX_PATH=/path/to/pytorch/prefix \
  -DCMAKE_INSTALL_PREFIX="$DPRC_DEEPMD_PREFIX" \
  -DBUILD_CPP_IF=ON \
  -DBUILD_PY_IF=OFF \
  -DBUILD_TESTING=OFF \
  -DDP_USING_C_API=ON \
  -DUSE_CUDA_TOOLKIT=TRUE \
  -DUSE_PT_PYTHON_LIBS=ON

cmake --build deepmd-build --parallel
cmake --install deepmd-build

export DPRC_DEEPMD_INCLUDE_DIR="$DPRC_DEEPMD_PREFIX/include"
export DPRC_DEEPMD_C_LIBRARY="$DPRC_DEEPMD_PREFIX/lib/libdeepmd_c.so"
export DPRC_DEEPMD_C_SHA256="$(sha256sum "$DPRC_DEEPMD_C_LIBRARY" | awk '{print $1}')"
export DPRC_DEEPMD_ARTIFACT_MANIFEST="$DPRC_DEEPMD_PREFIX/lammps-dprc-artifacts.json"

python3 "$DPRC_WORKSPACE/lammps-dprc/tools/deepmd_artifact_manifest.py" write \
  --source "$DPRC_WORKSPACE/deepmd-kit" \
  --include-dir "$DPRC_DEEPMD_INCLUDE_DIR" \
  --library "$DPRC_DEEPMD_C_LIBRARY" \
  --output "$DPRC_DEEPMD_ARTIFACT_MANIFEST"

rg -n '^#define[[:space:]]+DP_C_API_VERSION' \
  "$DPRC_DEEPMD_INCLUDE_DIR/deepmd/c_api.h"
nm -D "$DPRC_DEEPMD_C_LIBRARY" | \
  rg 'DP_DeepPotComputeCanonicalGraph(GPU|BatchGPU)'
```

Stop if the C API version is below 30 or neither canonical-graph symbol is
present. With API v30, the plugin validates and evaluates a packed
block-diagonal graph through `DP_DeepPotComputeCanonicalGraphGPU`; with API v31
or newer it selects the explicit frame-axis batch entry point.

## 7. Build LAMMPS-DPRc

Hash cuFFT and, when enabled, the diagnostic or production model:

```bash
export DPRC_CUFFT_HEADER_SHA256="$(sha256sum "$DPRC_CUFFT_INCLUDE_DIR/cufft.h" | awk '{print $1}')"
export DPRC_CUFFT_LIBRARY_SHA256="$(sha256sum "$DPRC_CUFFT_LIBRARY" | awk '{print $1}')"
export DPRC_DEEPMD_MODEL=/path/to/dpa4c-model.pt2
export DPRC_DEEPMD_MODEL_SHA256="$(sha256sum "$DPRC_DEEPMD_MODEL" | awk '{print $1}')"
```

Configure the complete plugin:

```bash
cd "$DPRC_WORKSPACE"

cmake -S lammps-dprc -B lammps-dprc/build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$DPRC_HOST_CC" \
  -DCMAKE_CXX_COMPILER="$DPRC_HOST_CXX" \
  -DCMAKE_CUDA_COMPILER="$DPRC_CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$DPRC_HOST_CXX" \
  -DCMAKE_CUDA_ARCHITECTURES="$DPRC_CUDA_ARCH" \
  -DMPI_CXX_COMPILER="$DPRC_MPI_CXX" \
  -DLAMMPS_SOURCE_DIR="$DPRC_WORKSPACE/lammps/src" \
  -DXTBLOOM_SOURCE_DIR="$DPRC_WORKSPACE/xtbloom" \
  -DXTBLOOM_GENERATED_INCLUDE_DIR="$DPRC_WORKSPACE/xtbloom/build/lammps-dprc-cuda/generated/include" \
  -DDPRC_XTBLOOM_LIBRARY="$DPRC_XTBLOOM_LIBRARY" \
  -DDPRC_EXPECTED_XTBLOOM_LIBRARY_SHA256="$DPRC_XTBLOOM_SHA256" \
  -DDPRC_REQUIRE_XTBLOOM_LIBRARY=ON \
  -DDPRC_XTBLOOM_BACKEND=CUDA \
  -DDPRC_XTBLOOM_DEVICE_ID=0 \
  -DDEEPMD_SOURCE_DIR="$DPRC_WORKSPACE/deepmd-kit" \
  -DDPRC_EXPECTED_DEEPMD_REVISION=<reviewed-api-v30-or-newer-revision> \
  -DDPRC_DEEPMD_INCLUDE_DIR="$DPRC_DEEPMD_INCLUDE_DIR" \
  -DDPRC_DEEPMD_C_LIBRARY="$DPRC_DEEPMD_C_LIBRARY" \
  -DDPRC_DEEPMD_ARTIFACT_MANIFEST="$DPRC_DEEPMD_ARTIFACT_MANIFEST" \
  -DDPRC_EXPECTED_DEEPMD_C_LIBRARY_SHA256="$DPRC_DEEPMD_C_SHA256" \
  -DDPRC_REQUIRE_DEEPMD_C_API=ON \
  -DDPRC_DEEPMD_MODEL="$DPRC_DEEPMD_MODEL" \
  -DDPRC_EXPECTED_DEEPMD_MODEL_SHA256="$DPRC_DEEPMD_MODEL_SHA256" \
  -DDPRC_LAMMPS_EXECUTABLE="$DPRC_WORKSPACE/lammps-build/lmp" \
  -DDPRC_LAMMPS_SIZES=smallbig \
  -DDPRC_BUILD_TESTING=ON \
  -DDPRC_ENABLE_KOKKOS_RUNTIME_TESTS=ON \
  -DDPRC_ENABLE_CLASSICAL_CUDA=ON \
  -DDPRC_CUFFT_INCLUDE_DIR="$DPRC_CUFFT_INCLUDE_DIR" \
  -DDPRC_CUFFT_LIBRARY="$DPRC_CUFFT_LIBRARY" \
  -DDPRC_EXPECTED_CUFFT_HEADER_SHA256="$DPRC_CUFFT_HEADER_SHA256" \
  -DDPRC_EXPECTED_CUFFT_LIBRARY_SHA256="$DPRC_CUFFT_LIBRARY_SHA256"

cmake --build lammps-dprc/build/cuda --parallel
ctest --test-dir lammps-dprc/build/cuda --output-on-failure
python3 lammps-dprc/tools/check_dependency_pins.py --required-only
```

The plugin is `lammps-dprc/build/cuda/dprcplugin.so`. To build xTB QM/MM
without DPRc, omit all `DEEPMD` options. To compile the DeePMD styles without
runtime model tests, omit only `DPRC_DEEPMD_MODEL` and its hash.

## 8. Install and inspect the plugin

```bash
cmake --install lammps-dprc/build/cuda \
  --prefix "$DPRC_WORKSPACE/lammps-dprc-install"

export DPRC_PLUGIN="$DPRC_WORKSPACE/lammps-dprc-install/lib/lammps/plugins/dprcplugin.so"
sha256sum "$DPRC_PLUGIN"
ldd "$DPRC_PLUGIN"
readelf -d "$DPRC_PLUGIN"
nm -D --defined-only "$DPRC_PLUGIN"
```

The installed plugin must export only `lammpsplugin_init`. A DeePMD-enabled
build must name `libdeepmd_c` as a dependency and must not name a standalone
DeePMD LAMMPS plugin. The install step removes build-machine RPATH entries.

## 9. Set the runtime environment

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export HYDRA_LAUNCHER=fork
export LD_LIBRARY_PATH="$DPRC_WORKSPACE/xtbloom/build/lammps-dprc-cuda:$DPRC_DEEPMD_PREFIX/lib:$DPRC_WORKSPACE/kokkos-install/lib:$DPRC_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Record the effective environment and `ldd` output for every correctness or
performance result. Do not combine libraries from incompatible CUDA cohorts.

## 10. Verify plugin loading

```bash
cd "$DPRC_WORKSPACE"
mkdir -p plugin-check

lammps-build/lmp \
  -log plugin-check/log.lammps \
  -screen plugin-check/screen.txt \
  -var dprc_plugin "$DPRC_PLUGIN" \
  -var dprc_marker "$DPRC_WORKSPACE/plugin-check/partition.txt" \
  -in "$DPRC_WORKSPACE/lammps-dprc/tests/in.dprc_info"
```

Stop after any ABI, MPI, integer-size, or style-registration error.

## 11. Direct LAMMPS launch commands

For one window:

```bash
/path/to/lmp \
  -log /path/to/run/log.lammps \
  -screen none \
  -in /path/to/run/input.lammps
```

For 32 independent windows sharing one GPU:

```bash
mpiexec -n 32 /path/to/lmp \
  -partition 32x1 \
  -plog /path/to/run/log.lammps \
  -pscreen none \
  -in /path/to/run/input.lammps
```

Every `variable ... world` command in the input must provide one value per
partition. Each partition must be an independent trajectory or umbrella
window; future dependent timesteps from one trajectory cannot be batched.

## Essential QM/MM plus DPA4c LAMMPS commands

The workflow runner should be preferred, but the essential force path is:

```lammps
plugin load /path/to/dprcplugin.so

units real
dimension 3
boundary p p p
atom_style full
atom_modify map array
newton on

variable start_data world /path/to/window.data
read_data ${start_data}
run_style verlet

group qm id 1:16
group water type 6 7
bond_style harmonic
angle_style harmonic

pair_style hybrid/overlay &
  lj/cut/dprc/batch 9.0 &
  tip4p/long/dprc/batch 6 7 1 1 0.125 9.0 &
  dprc/deepmd/batch /path/to/qualified-dpa4c.pt2 &
    partition_batch yes &
    center_group qm &
    environment_cutoff 6.0 &
    include_molecule yes

include /path/to/generated/forcefield_dprc_batch.inc
pair_coeff 6*7 6*7 tip4p/long/dprc/batch
pair_coeff * * dprc/deepmd/batch P O O C H OW HW
pair_modify pair lj/cut/dprc/batch tail yes
special_bonds amber

kspace_style pppm/tip4p/dprc/batch 1.0e-6
kspace_modify mesh 50 50 50 order 4 gewald 0.348831617901729

fix qmmm qm qmmm/xtb/dprc &
  elements P O O C H O H &
  cutoff 9.0 charge -2 uhf 0 &
  method gfn2 accuracy 0.001 maxiter 250 etemp 300.0 &
  mmhardness 0.0 kmax 8 8 8 ksqmax 100
fix_modify qmmm energy yes

fix water_shake water shake 1.0e-6 200 0 b 1 a 1
fix integrate all nve
fix thermostat all langevin 300.0 300.0 100.0 12345

timestep 0.001
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 100
run 1000
```

`center_group` must be static. `environment_cutoff` is expressed in LAMMPS
distance units. `include_molecule yes` requires positive molecule IDs for
selected environment atoms. The style requires atom IDs, an atom map, one MPI
rank per partition, synchronized timesteps, and no `neigh_modify exclude`.

For xTB QM/MM without DPA4c, remove the `dprc/deepmd/batch` sub-style and
its `pair_coeff`; keep `dprcplugin.so`, the batched classical styles, KSpace,
and `fix qmmm`.

These direct commands deliberately keep ordinary LAMMPS work on the host.
Use `-k on g 1 -pk kokkos newton on neigh half` together with the `/kk` style
aliases only for a separately qualified Kokkos run; otherwise every partition
would create an additional CUDA context alongside the GPU-local brokers.

## 12. Recommended production runner

```bash
python3 "$DPRC_WORKSPACE/lammps-dprc/tools/etpeth_workload.py" run \
  --tutorial /path/to/dprc-tutorial \
  --output /path/to/run-directory \
  --lammps "$DPRC_WORKSPACE/lammps-build/lmp" \
  --plugin "$DPRC_PLUGIN" \
  --xtbloom-library "$DPRC_XTBLOOM_LIBRARY" \
  --deepmd-model /path/to/qualified-dpa4c.pt2 \
  --mode qmmm-dpa4c \
  --lammps-execution-backend host \
  --dpa4c-models-qualified \
  --library-dir "$DPRC_DEEPMD_PREFIX/lib" \
  --library-dir "$DPRC_WORKSPACE/kokkos-install/lib" \
  --library-dir "$DPRC_CUDA_ROOT/lib64" \
  --stage batch-smoke \
  --smoke-window-count 2 \
  --smoke-steps 25
```

Proceed to `anchor`, `seeds`, `equilibrate`, and `production` only after the
one-window and batched correctness checks pass with the exact production
artifacts. The host execution backend keeps ordinary LAMMPS work on the CPU
and avoids one unused Kokkos CUDA context per umbrella partition; xTBloom and
DeePMD remain owned by their GPU-local brokers.
