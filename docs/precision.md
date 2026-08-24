# Precision policy for RTX 5090

RTX 5090 makes FP32 and tensor-oriented execution attractive, while the QM/MM
scientific contract still requires binary64 public values. Precision is
therefore split by component instead of controlled by one global switch.

## Supported experiment names

| Name | DPRc | xTBloom | PPPM and classical | Status |
| --- | --- | --- | --- | --- |
| `fp64-reference` | model-native reference | public and internal FP64 | FP64 reference | required baseline |
| `dprc-fp32` | FP32 artifact/inference | FP64 | reference | planned production candidate |
| `xtb-mixed-safe` | selected independently | FP64 SCC/eigensolver and double accumulation | reference | experimental |
| `full-mixed-experimental` | FP32 candidate | measured mixed precision | measured mesh/pair precision | research only |

The xTBloom public C ABI remains double for positions, energy, forces, charges,
periodic operators, and point charges in every mode.

## Plausible first xTB mixed-precision candidates

- distance screening, neighbor/pair-list construction, and bucket metadata;
- geometry-derived short-lived intermediates whose final contribution is
  recomputed or accumulated in double;
- selected pair kernels with FP32 evaluation and FP64 accumulation, after
  term-level force and invariance tests;
- batched matrix operations only when the provider exposes a deterministic
  mixed-input/double-output path and the SCC/eigensolver evidence remains
  qualified.

These are candidates, not approved conversions.

## Keep in FP64 until separately proven

- SCC charge, multipole, residual, mixer, and convergence state;
- Hamiltonian/overlap matrices and generalized eigensolvers;
- occupation and free-energy decisions near degeneracies;
- periodic `b + Aq` response and operator-derivative accounting;
- final energy, force, virial, charge, and reduction accumulation.

A whole-library `double -> float` replacement is not an acceptable experiment.

## DPRc and DeePMD

DeePMD models may explicitly use FP32 inference. The selected model artifact,
backend, compact environment, and output precision must be recorded. FP32 is
not inferred from the GPU name alone, and FP16/TF32 is a separate experiment.

The in-plugin DeePMD path requires public C API version 31 or newer and exposes
a deliberate mixed-representation boundary:

- `DP_DeepPotComputeCanonicalGraphBatchGPU` consumes a block-diagonal,
  device-resident compact graph with FP32 edge vectors and publishes FP64
  atomic energies, forces, and virials;
- `DP_DeepPotUsesFP32EdgeVectors` and the related device/canonical-graph
  capability queries allow runtime selection instead of assuming a model
  representation from its filename.

This is the preferred first RTX 5090 experiment because it reduces graph
geometry traffic without weakening the LAMMPS force interface. It still needs
model-specific correctness and end-to-end umbrella qualification; the API
shape alone is not evidence that FP32 inference is accurate enough.

LAMMPS-DPRc supplies compact selection and cross-window batching through its
own `dprc/deepmd/batch[/kk]` styles. One GPU-local owner executes the C API
batch after collective graph validation. The current API does not expose the
backend CUDA stream, so a device synchronization remains part of the
correctness boundary before host publication.

## Acceptance sequence

1. Term-level energy/force and finite-difference qualification.
2. Full single-step QM/MM and DPRc comparison on all windows.
3. Serial versus batched comparison.
4. Short trajectory statistical and stability checks.
5. Full umbrella free-energy comparison with uncertainty.
6. Only then publish throughput or speedup.
