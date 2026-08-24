# AGENTS.md

This file applies to the whole repository.

## Mission

LAMMPS-DPRc maximizes correctness-qualified aggregate umbrella-sampling
throughput for multi-window xTB QM/MM plus optional DPRc on GPU systems. The
complete free-energy workflow is the product; microbenchmarks are supporting
evidence only.

## Start of material work

1. Read the active GitHub issue and blocking issues once a remote exists.
2. Inspect the branch, HEAD, worktree status, dependency pins, and relevant CI.
3. Run `python3 tools/check_dependency_pins.py --required-only`.
4. Record the intended correctness and performance claim before timing.
5. Preserve unrelated changes in this and all sibling repositories.

## Architectural invariants

- Call xTBloom only through its public C ABI. Never depend on xTBloom C++
  implementation symbols.
- Treat the LAMMPS C++ plugin boundary as version-, MPI-, integer-size-, and
  compiler-specific. Never promise one plugin binary works across builds.
- One GPU-local broker owns the batched xTBloom CUDA context. Do not create one
  tiny xTBloom CUDA context per umbrella window.
- A batch slot has stable replica identity so SCC WARM state cannot migrate to
  another trajectory.
- Validate the full request before modifying LAMMPS forces, energies, or
  virials. A failed window must not publish partial force slices.
- Do not batch future dependent timesteps from one trajectory.
- Avoid steady-state allocation, repeated neighbor scans, redundant FFTs,
  device-wide synchronization, and avoidable host/device copies.

## Scientific and precision invariants

- LAMMPS/xTBloom exchange uses IEEE binary64 and atomic units at the xTBloom C
  ABI. Convert LAMMPS units explicitly at one documented boundary.
- FP32 and mixed precision are opt-in experiments, never silent defaults.
- A precision change requires independent energy, force, charge, SCC,
  trajectory, and free-energy qualification against the FP64 reference.
- Never weaken a tolerance or regenerate a reference to make a precision or
  performance experiment pass.
- One-window and batched results must agree within the declared scientific
  tolerance before their timings are eligible.

## Performance evidence

- The primary metric is aggregate accepted MD steps/s/GPU over all windows.
- Record clean revisions, binary hashes, compiler/toolkit/driver, GPU/CPU,
  affinity, MPI layout, batch size, descriptors, warmup, synchronization,
  sample count, raw samples, SCC state, and correctness results.
- Compare batch sizes `1, 2, 4, 8, 16, 32` unless a coordinate is explicitly
  unavailable. Do not silently remove an inconvenient result.
- Dirty dependency measurements are diagnostic only and cannot support a final
  claim.
- Raw Nsight captures may contain environment data and must not be committed.
  Store only reviewed derived CSV, JSON, or text summaries.

## Dependencies and licensing

- Do not vendor external source, models, datasets, patches, or binaries without
  an exact revision, SHA-256, license record, and explicit review.
- The initial GPLv2 LAMMPS / GPL-3.0-or-later xTBloom distribution boundary is
  unresolved. Do not distribute combined binaries until the owner records a
  licensing decision.
- DeePMD PR #5943 is an open, hash-pinned reference, not an upstream release.
- Do not infer that `dlopen` resolves license compatibility.

## Code and validation

- Use C++17, focused comments, and public API documentation.
- Search with `rg` before editing and register new sources/tests in CMake.
- Run the smallest focused checks while iterating, then at minimum:

```bash
cmake -S . -B build -G Ninja <required dependency options>
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 tools/check_dependency_pins.py --required-only
git diff --check
```

## AI attribution

Every AI-authored GitHub comment, PR body, review, and Git commit must identify
the coding agent, actual client version, exact configured model, and configured
reasoning effort. For Codex, read them immediately before each write with:

```bash
codex --version
rg -n '^(model|model_reasoning_effort)\s*=' ~/.codex/config.toml
```

Codex commit trailers are:

```text
Coding-Agent: Codex
Codex-Version: <actual output>
Model: <configured model>
Reasoning-Effort: <configured effort>
```
