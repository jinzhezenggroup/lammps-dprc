# Project naming and packaging

## Naming decision

Keep the research project and paper-facing name:

```text
LAMMPS-DPRc
GPU-batched xTB QM/MM + DPRc for LAMMPS
```

The name is concise, preserves the established DPRc identity, and does not
bind the project to one model architecture. DPA4c is the current correction
architecture, while the repository's durable boundary is the LAMMPS execution
and workflow layer.

Use `lammps-qmmm-dprc` as the future source distribution or package-manager
name. It is explicit enough for users searching for a LAMMPS QM/MM plugin and
avoids claiming that the package contains LAMMPS itself.

Keep the current internal names during this productization pass:

- repository directory: `lammps-dprc`;
- CMake project: `lammps_dprc`;
- plugin binary: `dprcplugin.so`;
- LAMMPS styles: the existing `/dprc` and `/dprc/batch` names.

Renaming compiled targets or styles in a dirty development tree would create
compatibility churn without improving the user contract. Such a rename should
occur only as a dedicated release migration with aliases or clear breakage
notes.

## Repository boundary

Keep these items together in the main repository:

- the LAMMPS plugin source and version-specific build logic;
- the synchronized xTBloom and classical brokers;
- production input generation and resumable umbrella workflow;
- correctness tests and dependency pins;
- benchmark matrix, runner, protocol, and reviewed derived summaries;
- model admission policy and model-card schema.

The benchmark runner belongs here because timing eligibility depends on the
same runtime semantics, stable-slot identity, correctness gates, dependency
pins, and generated inputs as production. Moving it to an independent source
repository would make it easier for the paper benchmark to drift from the
software being measured.

## Artifacts that should remain separate

Do not commit raw trajectories, label calculations, profiler captures, large
datasets, model binaries, or combined runtime binaries to this repository.
They have independent size, privacy, provenance, and licensing boundaries.

Use separate immutable artifacts when they exist:

- a system-specific model release, recommended name
  `dpa4c-etpeth-pbe0-xtb-dprc`;
- a DOI-backed dataset archive for source frames and paired labels;
- a frozen paper artifact or data archive containing raw benchmark samples and
  the exact clean revisions used for the publication.

A paper artifact can be named after the eventual article and release tag, for
example `lammps-dprc-paper-artifact-<year>`. It should be created only when the
paper benchmark is frozen. There is no need for a separate live benchmark code
repository now.

## Suggested release layout

The future `lammps-qmmm-dprc` source release should expose:

```text
README.md
docs/user-guide.md
docs/benchmark-results.md
docs/dpa4c-model-plan.md
config/dependencies.json
src/
tools/etpeth_workload.py
benchmarks/run.py
benchmarks/matrix.json
benchmarks/results/
examples/umbrella/
```

Release notes must state the exact LAMMPS, MPI, integer-size, compiler, CUDA,
xTBloom, and DeePMD compatibility boundary. Do not promise one plugin binary
for multiple LAMMPS builds.

## Distribution decision still required

The GPL-2.0-only LAMMPS and GPL-3.0-or-later xTBloom boundary has not been
resolved. `dlopen` does not by itself establish license compatibility. Until
the owner records a decision, publish source instructions and hashes but do
not distribute a combined binary bundle.
