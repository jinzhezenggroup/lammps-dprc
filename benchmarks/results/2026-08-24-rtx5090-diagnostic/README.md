# RTX 5090 diagnostic benchmark snapshot

This directory contains reviewed derived results only. The measurements are
private diagnostics and are not eligible for a publication claim because the
LAMMPS-DPRc worktree and tutorial source were unqualified and the DPA4c model
was a randomly initialized test artifact.
The DPA4c coordinate used the earlier standalone DeePMD adapter and is not a
measurement of the current in-plugin C API path.

- `summary.csv` contains median aggregate throughput for the three execution
  modes and every requested batch coordinate.
- `metadata.json` records the workload, machine, protocol, runtime hashes,
  source-summary hashes, Sander comparison boundary, and batch-48 failure.
- The interpretation and paper-ready rerun checklist are in the
  [benchmark report](../../../docs/benchmark-results.md).

Raw summaries, samples, trajectories, logs, and profiler captures remain in
external run storage and are not committed here.
