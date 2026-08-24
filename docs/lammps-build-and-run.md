# Build and run LAMMPS

This guide builds the exact LAMMPS host required by LAMMPS-DPRc and shows the
direct `lmp` commands for one-window and multi-partition execution. Every path
beginning with `/path/to` is a user-supplied placeholder.

The public source stack is sufficient for batched xTB QM/MM. QM/MM+DPA4c also
requires a separately loaded DeePMD plugin that implements compact
`center_group` input and `partition_batch yes`, plus a scientifically qualified
DPA4c DPRc model. Those two artifacts are not distributed by this repository;
the current multi-partition DeePMD implementation is not yet available as a
clean, immutable public release.

## Create the source workspace

Choose an empty parent directory and clone the three repositories as siblings:

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

The revisions are also recorded in
`lammps-dprc/config/dependencies.json`. Keep the LAMMPS and xTBloom worktrees
clean for any correctness or performance claim:

```bash
python3 lammps-dprc/tools/check_dependency_pins.py --required-only
```

## Select one toolchain

Provide:

- CMake 3.24 or newer and Ninja;
- a C++20-capable host compiler accepted by the selected CUDA toolkit;
- one MPI implementation with C and C++ wrappers;
- an NVIDIA CUDA toolkit compatible with the target GPU;
- cuFFT from the same reviewed CUDA runtime cohort used by LAMMPS-DPRc;
- Python 3.11 or newer for the repository tools.

Set machine-specific paths outside the repositories. The following example
uses an RTX 5090, whose CUDA compute capability is 12.0 and whose Kokkos
architecture name is `BLACKWELL120`:

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

Replace both architecture values for another GPU. The LAMMPS executable,
Kokkos, DeePMD plugin, and LAMMPS-DPRc plugin must use the same host C++ ABI,
MPI implementation, LAMMPS integer-size mode, and CUDA architecture. Do not
mix an Open MPI plugin with an MPICH executable or reuse a plugin built for a
different LAMMPS revision.

## Build the shared Kokkos runtime

QM/MM+DPA4c loads DeePMD separately. LAMMPS and the DeePMD plugin must
therefore share one Kokkos runtime. Build that runtime from the Kokkos snapshot
contained in the pinned LAMMPS checkout:

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

Build the separately loaded DeePMD LAMMPS plugin against this same
`kokkos-install` prefix. For xTB QM/MM without DPA4c, a shared Kokkos runtime
is not intrinsically required, but this route keeps one LAMMPS build compatible
with both execution modes.

## Build the production LAMMPS executable

The plugin supplies private QM/MM styles, so leave both upstream `QMMM` and
`QMMM-XTB` packages disabled. `EXTERNAL_KOKKOS=ON` is required; setting only
`Kokkos_DIR` would otherwise leave LAMMPS on its bundled static Kokkos build.
`PKG_PLUGIN=ON` provides the `plugin load` command.

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

`KOKKOS_PREC=double` is the production default. It is independent of the
xTBloom C ABI, which exchanges IEEE binary64 values in atomic units. Any
Kokkos, xTBloom, or DPA4c FP32 or mixed-precision variant is a separate opt-in
experiment requiring independent scientific qualification.

Do not change `LAMMPS_SIZES` when building the plugin. If `bigbig` is required,
configure both LAMMPS and LAMMPS-DPRc explicitly with `bigbig` and rerun all
correctness tests.

### Optional upstream GPU-package benchmark executable

The production batched path does not use the LAMMPS GPU package. To reproduce
the retained upstream-GPU classical MM reference, configure a separate LAMMPS
build with the same options above except:

```text
-DPKG_GPU=ON
-DGPU_API=cuda
-DGPU_ARCH=sm_120
-DGPU_PREC=mixed
```

The `GPU_PREC=mixed` result applies only to that upstream MM benchmark
coordinate. It does not change xTBloom or LAMMPS-DPRc precision.

## Verify the LAMMPS executable

Record the executable identity and verify the required packages:

```bash
cd "$DPRC_WORKSPACE"
sha256sum lammps-build/lmp
lammps-build/lmp -h > lammps-build/lmp-help.txt
rg 'KOKKOS|KSPACE|MOLECULE|RIGID|COLVARS|PLUGIN' lammps-build/lmp-help.txt
```

Check that execution resolves the intended MPI, C++ runtime, CUDA cohort, and
shared Kokkos libraries:

```bash
ldd lammps-build/lmp
ldd /path/to/libdeepmd_lmp.so
```

The DeePMD command is required only for QM/MM+DPA4c. Do not continue if the
two components resolve different MPI or Kokkos libraries.

## Build xTBloom for CUDA

Build the public shared library from the pinned xTBloom checkout. The Torch
extension and xTBloom's own test suite are not required for this runtime
artifact:

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

cmake --build xtbloom/build/lammps-dprc-cuda --target xtbloom --parallel

export DPRC_XTBLOOM_LIBRARY="$DPRC_WORKSPACE/xtbloom/build/lammps-dprc-cuda/libxtbloom.so"
export DPRC_XTBLOOM_SHA256="$(sha256sum "$DPRC_XTBLOOM_LIBRARY" | awk '{print $1}')"
printf '%s  %s\n' "$DPRC_XTBLOOM_SHA256" "$DPRC_XTBLOOM_LIBRARY"
```

If xTBloom requires an explicit CPU eigensolver provider in the selected
environment, add
`-DXTBLOOM_CPU_LINALG_LIBRARY=/path/to/lp64/lapacke-cblas.so`. Do not allow
CMake to select an unintended LP64/ILP64 or threaded provider silently.

## Build LAMMPS-DPRc

Hash the exact cuFFT header and shared library before configuring:

```bash
export DPRC_CUFFT_HEADER_SHA256="$(sha256sum "$DPRC_CUFFT_INCLUDE_DIR/cufft.h" | awk '{print $1}')"
export DPRC_CUFFT_LIBRARY_SHA256="$(sha256sum "$DPRC_CUFFT_LIBRARY" | awk '{print $1}')"
printf '%s  %s\n' "$DPRC_CUFFT_HEADER_SHA256" "$DPRC_CUFFT_INCLUDE_DIR/cufft.h"
printf '%s  %s\n' "$DPRC_CUFFT_LIBRARY_SHA256" "$DPRC_CUFFT_LIBRARY"
```

Use the same compiler, MPI, integer-size mode, CUDA architecture, and cuFFT
cohort as the LAMMPS host:

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
```

The resulting plugin is
`$DPRC_WORKSPACE/lammps-dprc/build/cuda/dprcplugin.so`.

## Set the runtime environment

Expose one physical GPU and the exact runtime libraries. Add DeePMD library
directories only for QM/MM+DPA4c:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export HYDRA_LAUNCHER=fork
export LD_LIBRARY_PATH="$DPRC_WORKSPACE/xtbloom/build/lammps-dprc-cuda:$DPRC_WORKSPACE/kokkos-install/lib:/path/to/deepmd/lib:$DPRC_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Do not inherit an unreviewed loader path from a login shell for a qualified
run. Record the effective environment and `ldd` output with every performance
or scientific result.

Run the plugin tests after establishing this environment:

```bash
ctest --test-dir "$DPRC_WORKSPACE/lammps-dprc/build/cuda" --output-on-failure
```

## Verify plugin loading

Run the topology and ABI diagnostic before a scientific input:

```bash
cd "$DPRC_WORKSPACE"
mkdir -p plugin-check

lammps-build/lmp \
  -log plugin-check/log.lammps \
  -screen plugin-check/screen.txt \
  -var dprc_plugin "$DPRC_WORKSPACE/lammps-dprc/build/cuda/dprcplugin.so" \
  -var dprc_marker "$DPRC_WORKSPACE/plugin-check/partition.txt" \
  -in "$DPRC_WORKSPACE/lammps-dprc/tests/in.dprc_info"
```

The command must load the plugin and write the expected partition identity.
Never continue after an ABI, integer-size, MPI, or style-registration error.

## Direct single-window LAMMPS command

For a generated input containing one value for each `variable ... world`
command:

```bash
mkdir -p /path/to/run-directory/logs

"$DPRC_WORKSPACE/lammps-build/lmp" \
  -k on g 1 \
  -pk kokkos newton on neigh half \
  -log /path/to/run-directory/logs/log.lammps \
  -screen none \
  -in /path/to/run-directory/input.lammps
```

Kokkos must be initialized before `read_data`. The generated input uses
`atom_style full/kk`, places `newton on` before `read_data`, and selects
`run_style verlet/kk` after the simulation box exists.

## Direct multi-partition LAMMPS command

Use one MPI rank per independent window. For 32 synchronized windows on one
GPU:

```bash
mkdir -p /path/to/run-directory/logs

mpiexec -n 32 "$DPRC_WORKSPACE/lammps-build/lmp" \
  -k on g 1 \
  -pk kokkos newton on neigh half \
  -partition 32x1 \
  -plog /path/to/run-directory/logs/log.lammps \
  -pscreen none \
  -in /path/to/run-directory/input.lammps
```

The input must contain exactly 32 values for every world variable. LAMMPS
creates partition logs such as `log.lammps.0` through `log.lammps.31`. The
synchronized timing denominator is the slowest partition loop, not the average
of those logs.

Do not batch future dependent timesteps from one trajectory. Each partition
must represent an independent window or trajectory and retain its stable slot
identity for the entire run.

## Essential QM/MM+DPA4c LAMMPS commands

The workflow runner generates the complete input and should be preferred over
manual authoring. The essential force-path commands have this shape:

```lammps
plugin load /path/to/libdeepmd_lmp.so
plugin load /path/to/dprcplugin.so

units real
boundary p p p
atom_style full/kk
atom_modify map array
newton on

variable start_data world /path/to/window.data
read_data ${start_data}
run_style verlet/kk

group qm id 1:16
group water type 6 7
bond_style harmonic/kk
angle_style harmonic/kk

pair_style hybrid/overlay/kk &
  lj/cut/dprc/batch 9.0 &
  tip4p/long/dprc/batch 6 7 1 1 0.125 9.0 &
  deepmd/kk /path/to/qualified-dpa4c.pt2 &
  partition_batch yes out_freq 0 atomic center_group qm &
  environment_cutoff 6.0 include_molecule yes

include /path/to/generated/forcefield_dprc_batch.inc
pair_coeff 6*7 6*7 tip4p/long/dprc/batch
pair_coeff * * deepmd/kk P O O C H OW HW
pair_modify pair lj/cut/dprc/batch tail yes
special_bonds amber

kspace_style pppm/tip4p/dprc/batch 1.0e-6
kspace_modify mesh 50 50 50 order 4 gewald 0.348831617901729

fix qmmm qm qmmm/xtb/dprc &
  elements P O O C H O H cutoff 9.0 charge -2 uhf 0 &
  method gfn2 accuracy 0.001 maxiter 250 etemp 300.0 &
  mmhardness 0.0 kmax 8 8 8 ksqmax 100
fix_modify qmmm energy yes
```

The complete input must also define all force-field coefficients, TIP4P SHAKE,
the integrator, thermostat, Colvars restraints, neighbor policy, thermo output,
run length, and final data/restart writes. These are generated from
`workloads/etpeth/manifest.json` by `tools/etpeth_workload.py`.

For xTB QM/MM without DPA4c, omit the DeePMD plugin, the `deepmd/kk` sub-style,
and its `pair_coeff`; keep the LAMMPS-DPRc plugin, batched classical styles,
KSpace style, and `fix qmmm`.

## Recommended production command

The direct commands are useful for debugging. For production, use the runner
so input rendering, model qualification, artifact hashes, stable world
ordering, resume validation, and output publication remain fail-closed:

```bash
python3 "$DPRC_WORKSPACE/lammps-dprc/tools/etpeth_workload.py" run \
  --tutorial /path/to/dprc-tutorial \
  --output /path/to/run-directory \
  --lammps "$DPRC_WORKSPACE/lammps-build/lmp" \
  --plugin "$DPRC_WORKSPACE/lammps-dprc/build/cuda/dprcplugin.so" \
  --xtbloom-library "$DPRC_XTBLOOM_LIBRARY" \
  --deepmd-plugin /path/to/libdeepmd_lmp.so \
  --deepmd-model /path/to/qualified-dpa4c.pt2 \
  --mode qmmm-dpa4c \
  --dpa4c-models-qualified \
  --library-dir "$DPRC_WORKSPACE/kokkos-install/lib" \
  --library-dir /path/to/deepmd/lib \
  --library-dir "$DPRC_CUDA_ROOT/lib64" \
  --stage batch-smoke \
  --smoke-window-count 2 \
  --smoke-steps 25
```

Proceed to `anchor`, `seeds`, `equilibrate`, and `production` only after the
one-window and batch smoke checks pass with the exact production artifacts.
