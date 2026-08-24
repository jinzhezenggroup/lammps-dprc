# Benchmarks

Benchmark evidence answers one question: how quickly one physical GPU produces
scientifically accepted umbrella-sampling trajectory across all active windows.

The declarative initial matrix is in `matrix.json`. Runners added later must
preserve every requested coordinate as pass, fail, or unavailable and emit raw
samples plus correctness metadata. They must not replace the matrix with a
smaller ad hoc timing script.

`run.py` is the authoritative ETP/ETH comparison runner.  It measures the
complete classical, xTB QM/MM, and xTB QM/MM plus DPA4c modes at batch sizes
`1, 2, 4, 8, 16, 32, 48`.  Warmup and repeated samples execute in one LAMMPS
process.  For a partitioned sample, the denominator is the maximum synchronized
LAMMPS loop time across windows, so startup and fast-but-idle partitions cannot
inflate aggregate throughput.

An inventory that proves matrix completeness without launching LAMMPS is:

```bash
python3 benchmarks/run.py \
  --tutorial ../dprc-tutorial \
  --output /absolute/path/out \
  --lammps /absolute/path/lmp \
  --inventory-only \
  --allow-unqualified-source
```

For execution, additionally provide the exact `--plugin`,
`--xtbloom-library`, CUDA/DeePMD `--library-dir` values, and one model artifact.
The production `qmmm-dpa4c` curve evaluates one primary model every timestep
through `dprcplugin.so` (`--model-deviation-frequency 0`). The runner rejects
positive model-deviation frequencies and multiple models until an in-plugin
ensemble schedule is implemented. `--dpa4c-models-qualified` must explicitly
assert that the supplied artifact
passed the xTB-based DPRc scientific gates.  This prevents a convenient
pretrained absolute potential from being benchmarked under the DPRc label.
For GPU-path development before such a model exists,
`--allow-unqualified-dpa4c-models` admits random, pretrained, or otherwise
unqualified artifacts for diagnostic timing only.  The runner records the
model hash and schedule but makes every resulting row evidence-ineligible;
the diagnostic switch and the qualification assertion are mutually exclusive.

The production path is rendered explicitly as `atom_style full/kk`,
`run_style verlet/kk`, `dprc/deepmd/batch/kk`, and the available Kokkos bonded,
SHAKE, integration, thermostat, momentum, and Colvars variants. The launcher
initializes Kokkos with one GPU plus `newton on neigh half`; the latter pair is
required by the shared batched classical broker instead of Kokkos GPU's
full-list, Newton-off defaults. Evidence records
`dprcplugin-deepmd-c-api-batch`, the exact `libdeepmd_c` identity inherited by
the plugin under `loaded_deepmd_c` (resolved path, SONAME, and SHA-256), and the
model hash.

Each output contains `environment.json`, `summary.json`, `samples.csv`, the
generated inputs, launcher logs, per-partition LAMMPS logs, and final state
hashes.  Timings from dirty or license-unresolved sources remain diagnostic;
`summary.json` records every reason that a passed timing is not evidence
eligible. DPA4c runs pin both DeePMD operator thread pools to one thread; the
exact values are included in every coordinate's selected
environment record.

The reviewed 2026-08-24 RTX 5090 diagnostic snapshot is under
`results/2026-08-24-rtx5090-diagnostic/`. Its interpretation, Sander comparison
boundary, batch-32 decomposition, and paper-ready rerun checklist are in
`docs/benchmark-results.md`. Only derived CSV/JSON and explanatory prose are
tracked; the raw run directories remain external.

The classical mode defaults to the shared `lj/cut/dprc/batch` +
`tip4p/long/dprc/batch` + `pppm/tip4p/dprc/batch` GPU path. Use
`--classical-backend upstream-gpu` only for the retained LAMMPS GPU-pair plus
CPU-PPPM reference. Every classical row records which backend produced it.

Final compact evidence belongs under `evidence/issue-<N>/<date>-<machine>/`.
Raw profiler captures, trajectories, restart files, and model artifacts are not
tracked here.
