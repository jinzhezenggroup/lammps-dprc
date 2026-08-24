---
name: Performance work
about: Track a correctness-qualified end-to-end performance change
title: "perf: "
labels: performance
---

## Claim and threshold

State the exact aggregate steps/s/GPU, memory, transfer, or component claim and
the acceptance threshold before timing.

## Workload and baseline

Record input revision/hash, windows, batch/rank layout, precision, SCC start,
DPRc mode, and baseline semantics.

## Correctness gates

- [ ] Energy/forces/virial
- [ ] QM charges and point-charge forces
- [ ] Serial/batch parity
- [ ] CPU/CUDA or FP64/mixed parity
- [ ] Free-energy curve with uncertainty

## Evidence matrix

List every required coordinate and mark pass, fail, or unavailable.

## Exact commands and artifacts

Include clean revisions, binary hashes, hardware/software identity, raw sample
count, and evidence paths.
