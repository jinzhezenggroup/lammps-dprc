# End-to-end performance protocol

## Headline claim

The project optimizes correctness-qualified aggregate umbrella MD steps per
second per physical GPU for the complete xTB QM/MM plus optional DPRc workflow.
No force component is removed from timing, and setup is reported separately
rather than subtracted from unavoidable runtime work.

Secondary metrics are:

- simulated ns/day/GPU summed across active windows;
- p50, p95, and maximum timestep wall time;
- per-component pair, KSpace, QMMM, xTB SCC, DPRc, MPI, and idle time;
- SCC iteration distributions and per-window convergence state;
- GPU utilization, memory high-water mark, allocation count, and transfer
  volume;
- reconstructed free-energy uncertainty and time to a declared effective
  sample target.

## Reference workload

The initial workload is supplied externally by the clean, pinned
`dprc-tutorial` revision recorded in `config/dependencies.json`. The starting
coordinate is the ETP/ETH triclinic system with 8,938 atoms, a 16-atom QM
region, TIP4P-Ew solvent, 9 angstrom real-space cutoff, and 50 x 50 x 50 PPPM
mesh.

Final evidence must record all umbrella centers, force constants, seeds,
thermostat/integrator settings, equilibration, production length, output
frequency, and free-energy estimator settings. One window is not evidence for
the multi-window claim.

## Required axes

Unless unavailable with a documented reason, measure:

- batch size: `1, 2, 4, 8, 16, 32, 48`;
- ranks per window: `1` for the batched classical path; `1, 2, 4` remain
  reference axes for paths that support domain decomposition;
- xTB SCC start: `FRESH`, then steady `WARM`;
- xTB precision: public FP64 reference and each named mixed-precision
  experiment;
- classical path: CPU reference, LAMMPS GPU pair path, and each fused/device
  implementation;
- DPRc: disabled, then one compact primary model evaluated every step;
- model deviation: disabled for the production throughput curve, then enabled
  at explicit low-frequency strides with four qualified models to measure its
  amortized monitoring cost;
- descriptor memory: host and CUDA-device wherever the implementation supports
  both without changing semantics.

For `pppm/tip4p/dprc/batch`, separately record setup and steady-state transform
counts.  One synchronized step must contain exactly two forward transform
batches, one four-field MM inverse batch, and one three-field QM inverse batch;
there is no full-charge forward transform.  Pair timing must likewise show one
fused LJ/TIP4P real-space traversal, with cell-list rebuilds reported
separately from reuse steps.

`dprc.classical_batch` enforces the structural cuFFT contract at link time for
GNU/LLVM Linux test builds: for `B=2` it requires planMany batches `2`, `8`, and
`6`, followed by exactly `forward(2)`, `inverse(8)`, `forward(2)`, and
`inverse(6)`.  This catches a reintroduced third FFT, but it is not a timing
profile; final time attribution still requires a profiler that supports the
target GPU and driver.

Setting `LAMMPS_QMMM_XTB_PROFILE=1` adds two diagnostics at fix destruction.
The phase record reports average preparation, MM potential/KSpace, direct-image
response, xTB, QM KSpace, and periodic-force time per step plus MM-list refresh
and topology-change counts. The broker record reports batch calls, actual
xTBloom plan rebuilds, FRESH/WARM calls, system calls, and SCC iteration
statistics. It also reports padded-slot capacity growths and mean/maximum
physical versus plan point counts. Profiling is opt-in and its output is not
itself a timing claim.

Do not compare resource-unequal baselines without naming the difference.

The number of loaded model artifacts is not the per-step inference multiplier.
With DeePMD's model-deviation schedule, ordinary timesteps evaluate only the
first model.  Every `f` steps, the deviation timestep evaluates all four and
uses model zero for dynamics.  The idealized forward count is therefore
`1 + 3/f` per timestep (`1.03` at `f=100`, `1.003` at `f=1000`).  Report the
single-primary production curve and each monitored stride separately; do not
describe the complete trajectory as a four-model or four-times calculation.

The single-primary curve must use the explicit device-resident
`deepmd/kk`/`full/kk`/`verlet/kk` path together with the available Kokkos
bonded, SHAKE, integration, thermostat, momentum, and Colvars styles.  Until
DeePMD's Kokkos pair supports ensembles, positive model-deviation strides use
its generic GPU adapter.  The device path also pins Kokkos to `newton on neigh
half`, which preserves the single-owner accumulation contract required by the
batched classical broker.
The forward schedule remains `1 + 3/f`, but deviation timesteps include host
staging and must be named `deepmd-generic-sparse-deviation` rather than
device-resident Kokkos evidence.

## Correctness qualification

Before a timing row is eligible, require:

1. single-window reference energy, forces, virial, QM charges, point-charge
   forces, and SCC state;
2. serial versus batch agreement for every window;
3. CPU/CUDA and FP64/mixed-precision agreement under predeclared tolerances;
4. stable trajectory behavior under the production integrator and thermostat;
5. an umbrella free-energy curve statistically compatible with the FP64
   reference at the same sampling length.

Precision tolerances must be chosen from scientific error requirements before
examining favorable performance results. Free-energy compatibility is assessed
with uncertainty, not visual similarity alone.

## Measurement record

Retain:

- clean Git revisions and dirty bits for this repository and every dependency;
- absolute paths and SHA-256 for executable, plugin, xTBloom, and DeepMD
  libraries/models;
- compiler, flags, CMake caches, CUDA toolkit/driver, GPU/CPU, clocks/power
  policy, process affinity, MPI, BLAS/eigensolver, and thread environment;
- exact input hashes, batch topology, point-charge capacity, requested outputs,
  warmup, synchronization boundary, repetitions, and every raw timing sample;
- convergence/failure status and correctness error for every coordinate.

The current development target is an NVIDIA GeForce RTX 5090 with compute
capability 12.0 and 32,607 MiB reported memory. This identifies the test target;
it is not a performance result.

The current dirty-tree development diagnostic uses 25 warmup steps and five
100-step samples of the 8,938-atom ETP/ETH workload.  Median aggregate
classical rates at batches `1, 2, 4, 8, 16, 32, 48` are respectively `475.2,
686.3, 890.9, 1032.4, 1121.1, 1193.1, 1114.1` accepted steps/s/GPU.  The exact
runner output is outside the repository at
`../lammps-dprc-runs/classical-optimized-long-diagnostic-20260823`; it remains
diagnostic because the worktree is dirty, correctness evidence was not
supplied to the runner, and the tutorial source is unqualified.

Under the same timing protocol, the fixed-capacity xTB QM/MM path measured
median aggregate rates of `26.92, 53.72, 99.32, 177.36, 281.43, 409.52,
427.58` accepted steps/s/GPU. The B1/B32/B48 output is at
`../lammps-dprc-runs/qmmm-padded-point-slots-tuned-b1-b32-b48-diagnostic-20260823`
and the B2/B4/B8/B16 output is at
`../lammps-dprc-runs/qmmm-padded-point-slots-tuned-b2-b16-diagnostic-20260823`.
Every coordinate used 624 plan slots for at most 612 physical points, created
one plan, performed no capacity growth, and used 525 WARM calls after the
initial FRESH call. These rows are likewise dirty, correctness-unqualified
runner diagnostics and are not publication evidence.

## Evidence layout

```text
benchmarks/evidence/issue-<N>/<date>-<machine>/
  README.md
  SHA256SUMS
  environment.json
  correctness.json
  samples.csv
  summary.json
  derived-profiler-reports
```

Do not commit raw Nsight captures. Store only sanitized derived summaries with
the profiler version and extraction command.
