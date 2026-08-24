# Diagnostic benchmark results

This report records the current development measurements for the ETP/ETH
umbrella workload. It is a reviewed derived summary, not publication-qualified
evidence. The tutorial checkout and LAMMPS-DPRc worktree were dirty, tutorial
licensing was unresolved, and the DPA4c graph was an unqualified diagnostic
artifact rather than a real PBE0-minus-xTB DPRc model.

## Measurement contract

- Metric: aggregate accepted MD steps/s/GPU over all synchronized windows.
- System: 8,938 atoms, 16 QM atoms, TIP4P-Ew, and a 50 x 50 x 50 PPPM mesh.
- Hardware: AMD EPYC 7K62 with 48 CPU cores and one NVIDIA GeForce RTX 5090
  with 32 GB memory.
- Software environment: NVIDIA driver 580.95.05 and MPICH/Hydra 5.0.1.
- Layout: one MPI rank per umbrella window and one physical GPU.
- Sampling: 25 warmup steps followed by 100-step synchronized samples.
- Repetitions: five for MM and xTB QM/MM; three for QM/MM+DPA4c.
- Reported value: median aggregate throughput, using the slowest partition's
  loop time for each synchronized repetition.

## Aggregate throughput

| Batch | MM steps/s/GPU | xTB QM/MM steps/s/GPU | QM/MM+DPA4c steps/s/GPU | xTB QM/MM / Sander xTB |
|---:|---:|---:|---:|---:|
| 1 | 475.161 | 26.924 | 23.559 | 3.893x |
| 2 | 686.318 | 53.717 | 43.699 | 7.767x |
| 4 | 890.946 | 99.318 | 74.663 | 14.361x |
| 8 | 1032.434 | 177.363 | 116.406 | 25.645x |
| 16 | 1121.076 | 281.433 | 156.305 | 40.693x |
| 32 | 1193.099 | 409.518 | 194.225 | 59.213x |
| 48 | 1114.087 | 427.575 | failed | 61.824x |

The Sander reference is 6.916 xTB QM/MM steps/s for one window on one CPU
core. Batch 1 is the closest latency comparison. Ratios above batch 1 compare
aggregate multi-window GPU throughput with one Sander window and must be
described as aggregate throughput ratios, not single-trajectory speedups.

The older Sander QM/MM+DPRc result of 6.414 steps/s is not a DPA4c reference.
It evaluated four copies of an older DeePMD test graph, so it must not be
compared with the current DPA4c row.

## Batch-32 end-to-end decomposition

The median synchronized loop time for 100 steps was:

| Incremental component | Seconds | Share of QM/MM+DPA4c loop |
|---|---:|---:|
| MM base | 2.68209 | 16.28% |
| xTB QM/MM increment | 5.13198 | 31.15% |
| DPA4c increment | 8.66163 | 52.57% |
| Total QM/MM+DPA4c | 16.47570 | 100.00% |

This subtraction is the most useful high-level attribution because all three
coordinates use the same workload and batch size.

The median LAMMPS timing categories over 32 windows and three repetitions were:

| LAMMPS category | Seconds | Share |
|---|---:|---:|
| Modify | 9.81045 | 59.58% |
| Pair | 6.21955 | 37.78% |
| Other | 0.24230 | 1.47% |
| Kspace | 0.14273 | 0.87% |
| Comm | 0.03338 | 0.20% |
| Bond | 0.01218 | 0.07% |

These categories are not a direct force-component decomposition. `Modify`
contains fix work such as the QM/MM transaction, while `Pair` includes the
DPA4c overlay and real-space pair work. Profiler evidence is required before
assigning a narrower kernel-level interpretation.

## Batch-48 failure

QM/MM+DPA4c at batch 48 failed before a valid timing sample. The failure came
from xTBloom plan creation, not from the DPA4c forward:

```text
xtbloom_plan_create
status=4 error=11 field=2
```

The observed failure was in the CUDA SCC initializer. The batch-48 coordinate
must remain visible as failed until xTBloom can create and qualify that plan;
it must not be omitted from a paper table.

## Artifact boundary

The diagnostic DeePMD plugin SHA-256 was
`0cf9e3b8a1c30b03c0778fcd24507ba26bf83f2720af84bdaa9a96b47bd931d0`.
The diagnostic model SHA-256 was
`b44d9eef44009739fee9ef98d22328b7fb29e298e82ae77b8824e361d335dfde`.
That model was randomly initialized and is explicitly unqualified. The
xTBloom library SHA-256 was
`c19e0376100b283f933da896d09fab26f578367b3460f2e79c70daf03b8efa6c`.

The derived CSV and metadata are stored under
`benchmarks/results/2026-08-24-rtx5090-diagnostic/`. Raw run directories,
trajectories, logs, and profiler captures remain external.

## Paper-ready rerun checklist

A publication table requires a new frozen run rather than relabeling these
diagnostics:

1. Use clean, immutable revisions for LAMMPS-DPRc and every dependency.
2. Resolve and record tutorial, model, dataset, and binary distribution
   licenses.
3. Replace the random graph with a scientifically qualified DPA4c ensemble.
4. Record binary hashes, compilers, toolkit, driver, CPU/GPU, affinity, MPI
   layout, batch size, descriptors, SCC state, warmup, synchronization, and raw
   samples.
5. Run correctness gates before timing every coordinate.
6. Measure batches 1, 2, 4, 8, 16, and 32, and retain batch 48 as either a
   passed result or an explicitly explained failure.
7. Repeat MM, xTB QM/MM, and QM/MM+DPA4c with identical protocol settings.
8. Archive reviewed derived CSV/JSON and immutable raw evidence separately;
   do not commit raw profiler captures.
