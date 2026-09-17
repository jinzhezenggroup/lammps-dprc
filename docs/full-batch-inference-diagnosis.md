# Full-batch DPA4c inference qualification

Status: 2026-09-17. These are diagnostic correctness observations, not accepted
performance results or a new model-accuracy assessment.

## Required experiment

For B simultaneous umbrella windows, both xTB and DeePMD must process all B
windows in each inference request. A disconnected graph with all windows is
a full batch even though the canonical ABI stores one flat node/edge axis.
Two-frame DeePMD subcalls are not a full-batch benchmark. The main runner
rejects the subcall cap and legacy chunked plans rather than relabeling them.

The model and deployment arithmetic are unchanged: binary64 xTB and the
existing FP32 DPA4c descriptor/fitting kernels without AMP or TF32. Binary64
energy/force output buffers do not make their internal arithmetic binary64.

## Observations on the production primary model

Full-batch probes on an NVIDIA RTX PRO 6000 GPU used the same equilibrated
configurations as the matched singleton references. The force gate remained
1e-5 kcal/mol/A; the plugin-energy gate remained 1e-8 kcal/mol plus a relative
1e-12. No thresholds or reference values were changed.

| B | Maximum force difference (kcal/mol/A) | Outcome |
| --- | --- | --- |
| 1 | 1.52e-12 | Pass |
| 2 | 1.13e-11 | Pass |
| 4 | 1.10e-5 | Force gate fails |
| 8 | 1.10e-5 | Force and some energy gates fail |
| 16 | 1.10e-5 | Force and some energy gates fail |
| 32 | 1.10e-5 | Force and some energy gates fail |
| 48 | 1.20e-5 | Force and some energy gates fail |

Only 24 matching singleton references were available for these probes. The
B=32 and B=48 comparisons therefore cover that subset, not every window. This
is enough to disqualify those layouts but not a complete acceptance test.

The initial chunked DPRc timing job was cancelled. Its already completed
singleton results are preserved, but no chunked matrix can represent the main
experiment. The independent QM/MM matrix can continue without neural inference.

## First differing operation

Diagnostic-only C ABI interception captured four singleton graphs and their
corresponding slices in full B=4 and B=48 requests. After rebasing indices,
all six input arrays were bit-identical: types, source indices, edge vectors,
destination CSR, source CSR, and source order. Energy and force output already
differed at the GPU C ABI, before plugin unit conversion or LAMMPS force scatter.

Replaying those immutable graphs and capturing each fitting GEMM located the
first discrepancy at the **first FP32 cuBLAS matrix multiplication**:

- Both operands (descriptor values and weights) are bit-identical.
- The first GEMM has column-major dimensions M=192, K=144, with N equal to
  the compact node count: 268 for the first singleton, 1,087 for B=4, and
  12,867 for B=48.
- Corresponding output rows differ by up to 2.38e-6 across the four checked
  windows, then propagate through subsequent fitting layers and force gradients.
- The same operands replayed through `cublasGemmEx` reproduce the discrepancy.
  Trying the default and legacy algorithm IDs 0 through 23 with pedantic FP32
  does not remove it on this device/runtime. Merely setting an algorithm ID is
  therefore not a demonstrated fix.

This establishes batch-shape-dependent FP32 GEMM numerics as the first observed
source. It does not establish a particular undocumented cuBLAS internal kernel
or reduction schedule. Pedantic math and a fixed workspace do not, by
themselves, guarantee identical results across different matrix dimensions.
The evidence excludes graph packing and descriptor generation for the captured
windows; it is not a statement about every other configuration.

## Remaining acceptance gate

A candidate fix must retain full-B inference and the existing precision
contract. A shape-stable full-matrix GEMM implementation or qualified fixed
cuBLASLt configuration is a possible route, not yet a validated solution.
Compare against the preserved singleton outputs as well as against the
candidate's own singleton path. Do not replace the reference just to remove a
failure. Check all 48 windows, energies, forces, charges, and virials, followed
by the relevant trajectory/NVE qualification before claiming accepted speed.
Any precision change requires separate scientific qualification.

The small discrepancy is a failed declared deployment-parity contract; it does
not, by itself, establish a material error in a completed free-energy curve.
No existing trajectory, label, model, or restart was replaced by this diagnosis.

Raw graphs, profiler/interception captures, models, and binaries remain private.
The primary model SHA-256 is
`3a15edc4ba0f764abe73b3d5aeff2df52eddc9aab1c34e3c523adef6bd0fcb6f`.
The diagnostic plugin SHA-256 is
`21006b7c6d724e2ab94adce526c5e452f54856d4a1a082ac1f82d74a6258c091`.
Clean source-to-binary qualification remains necessary for publication timing.
