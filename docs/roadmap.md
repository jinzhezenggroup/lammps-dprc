# Roadmap

## M0: repository and measurement contract

- Build/load a version-pinned diagnostic plugin.
- Record dependency and distribution boundaries.
- Define the complete umbrella throughput and precision matrix.

## M1: no-DPRc batched xTB QM/MM

- [implemented CPU reference] Port the existing QMMM semantics to the xTBloom
  C ABI and publish forces through `fix qmmm/xtb/dprc`.
- [implemented CPU reference] Gather synchronized partition roots into one
  ragged xTBloom plan owned by broker rank zero.
- [implemented core] Maintain stable replica slots and strict WARM behavior.
- [remaining] Prove real-GPU CUDA parity and aggregate multi-window scaling.

## M2: remove duplicate electrostatics work

- Cache fixed-cell Ewald setup and steady buffers.
- Replace repeated MM projection/MM-only/QM-only/production PPPM work with the
  qualified two-pass design.
- Reuse LAMMPS full-neighbor information for the explicit MM environment.

## M3: GPU classical path

- Profile upstream LJ/TIP4P GPU paths after M2.
- Decide between triclinic KOKKOS PPPM work and a broker-oriented batched CUDA
  PPPM implementation.
- Keep CPU reference paths available as correctness oracles.

## M4: DPRc

- [implemented C API gate] Link `dprcplugin.so` to DeePMD's public C API,
  build compact canonical graphs for `dprc/deepmd/batch[/kk]`, batch stable
  partition slots through one GPU-local owner, and prove host/alias parity,
  batch-2 parity, and QM/MM overlay additivity.
- [implemented label gate] Audit the legacy MNDOD-to-PBE0 archive and define
  the fail-closed full-periodic provenance required to regenerate PBE0-to-xTB
  correction labels.
- [implemented engine gate] Build and hash-qualify AmberTools 26 update.1 with
  QUICK 25.03 CUDA PBE0/6-31G* on the RTX 5090 using two complete QM/MM force
  calls; retain only reviewed compact evidence and the CUDA 12.9 build patch.
- [implemented periodic high-level gate] Retain a fail-closed AmberTools patch
  for atomic binary64 ETP/ETH labels, prove one QUICK call per frame, exact
  TIP4P M-site force redistribution, selected finite differences, and
  same-process multi-frame reuse on the RTX 5090.
- [implemented diagnostic] Build a clean, exact-commit upstream xTB 6.7.0 and
  AmberTools 26 update.1 Sander oracle, add a separate fail-closed binary64 xTB
  label mode without changing the PBE0 guard, and form one bitwise-matched
  full-system `PBE0 - xTB` correction record with exact classical cancellation.
- [remaining] Prove the independent AmberTools+xTB periodic result matches
  xTBloom's operator and full-force convention, then generate the first
  production xTBloom-based correction corpus.
- [remaining] Publish and pin the clean DeePMD C API v31 implementation used by
  the broker, then replace the diagnostic graph with a qualified model.
- Enable one qualified FP32 DPRc primary as the production mode. Keep model
  deviation offline until a reviewed in-plugin ensemble schedule exists.
- Add exact DPRc masking support for the selected DPA4/DPA4C architecture.
- Explore broker-level GPU stream scheduling after context and transfer
  profiling.

## M5: complete free-energy evidence

- [implemented runner] Verify external ETP/ETH input hashes, generate the
  48-window grid, relax the `-1.5 Angstrom` anchor, walk two adjacent-center
  seed branches, and resume synchronized equilibration/production from
  SHA-256-qualified checkpoints.
- [implemented analysis] Validate the production checkpoint DAG and reconstruct
  a predeclared histogram-WHAM PMF with adjacent overlap, time-correlation ESS,
  and trial-separated block-bootstrap uncertainty.
- [remaining] Generate and accept all 48 equilibrated states on the CUDA
  xTBloom path, then complete three production trials and run the analysis.

- Run all umbrella windows from clean pinned inputs.
- Archive correctness, raw timing samples, hardware/software identity, and PMF
  uncertainty.
- Select the production batch/rank/precision configuration from aggregate
  steps/s/GPU, not a component microbenchmark.
