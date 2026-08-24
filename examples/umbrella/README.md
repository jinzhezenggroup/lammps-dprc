# Umbrella workflow

Inputs remain in the external, pinned `dprc-tutorial` repository until their
license and canonical-generation status are explicitly recorded. Do not copy a
dirty local input into this directory.

The integration runner is `tools/etpeth_workload.py`. It accepts:

- a clean tutorial checkout;
- the manifest-defined umbrella grid and deterministic seed policy;
- the LAMMPS executable and this plugin;
- xTBloom and optional DeePMD C API/model runtime paths;
- the execution mode, ranks per window, and smoke-test batch size;
- a separate output directory outside the source tree.

It must emit the exact input hashes and dependency revisions before starting
production sampling.

The local tutorial checkout currently has an unresolved license and only one
initial window. Private diagnostic generation therefore needs an explicit
opt-in; no tutorial bytes are copied into this repository:

```bash
python3 tools/etpeth_workload.py verify \
  --tutorial ../dprc-tutorial \
  --allow-unqualified-source

python3 tools/etpeth_workload.py prepare \
  --tutorial ../dprc-tutorial \
  --output ../lammps-dprc-runs/etpeth \
  --allow-unqualified-source
```

After building a CUDA xTBloom library and a CUDA-forced plugin from stable
bytes, run the five-step end-to-end smoke:

```bash
python3 tools/etpeth_workload.py run \
  --tutorial ../dprc-tutorial \
  --output ../lammps-dprc-runs/etpeth \
  --allow-unqualified-source \
  --lammps ../lammps/build-dprc-gpu-cuda/lmp \
  --plugin build/cuda-integration/dprcplugin.so \
  --xtbloom-library ../xtbloom/build/lammps-dprc-cuda/libxtbloom.so \
  --library-dir /path/to/cuda/lib64 \
  --mode qmmm \
  --stage smoke
```

Before committing a long stage, exercise the real partition broker (for
example two windows and 25 steps) without treating the startup-dominated row
as final performance evidence:

```bash
<same options> --stage batch-smoke --smoke-window-count 2 --smoke-steps 25
```

To exercise QM/MM+DPA4c, use a `dprcplugin.so` built with DeePMD C API v31+
and one qualified DPA4c model. The model must represent the periodic
PBE0-minus-xTBloom correction; this repository does not ship one:

```bash
python3 tools/etpeth_workload.py run \
  --tutorial ../dprc-tutorial \
  --output ../lammps-dprc-runs/etpeth-dpa4c \
  --lammps /path/to/kokkos/lmp \
  --plugin /path/to/dprcplugin.so \
  --xtbloom-library /path/to/libxtbloom.so \
  --deepmd-model /path/to/qualified-dpa4c.pt2 \
  --mode qmmm-dpa4c \
  --dpa4c-models-qualified \
  --library-dir /path/to/deepmd/lib \
  --library-dir /path/to/cuda/lib64 \
  --stage batch-smoke \
  --smoke-window-count 2 \
  --smoke-steps 25
```

The current dirty or license-unresolved tutorial also requires
`--allow-unqualified-source`, which makes the entire run a private diagnostic.
If the model itself is an unqualified software fixture, replace
`--dpa4c-models-qualified` with `--allow-unqualified-dpa4c-models`. The runner
records this choice and refuses to start without one explicit model-status
boundary.

The single-model path emits `dprc/deepmd/batch/kk`, compact `center_group qm`
input, and `partition_batch yes`, so one GPU-local DeePMD owner evaluates the
synchronized windows as one block-diagonal batch. Model deviation is currently
rejected by the in-plugin path.

The scientific stages are resumable and deliberately separate:

```bash
# 20 ps relaxation of the available -1.5 Angstrom anchor.
<same options> --stage anchor

# Two outward branches, 3 ps per adjacent center, yielding all 48 seeds.
# Seed generation uses a transient 1000 kcal/mol/Angstrom^2 restraint; the
# 200 kcal/mol/Angstrom^2 sampling restraint is restored for equilibration.
<same options> --stage seeds

# One synchronized 48-partition, 100 ps/window equilibration.
<same options> --stage equilibrate

# Three 250 ps/window production trials by default.
<same options> --stage production
```

Each completed invocation has a JSON checkpoint under `records/`. Resume is
accepted only when the start data, rendered input, step range, launcher,
LAMMPS, plugin, resolved xTBloom library, manifest, provenance, and every final
data/restart/Colvars/trajectory file still match their recorded SHA-256.
Seed stages additionally require `|reaction_coordinate-center| <= 0.15
Angstrom`, attack angle within 30 degrees, finite Colvars values, successful
xTB SCC, and zero dangerous neighbor builds. The stronger seed-only restraint
is a state-preparation aid, not part of the umbrella Hamiltonian used for
equilibration, production, or WHAM.

Anchor, 48-window equilibration, and production are split into 5,000-step
chunks by default. Use `--chunk-steps` to tune the target chunk wall time after
measuring the real batch throughput. Each chunk keeps an absolute timestep
offset and a distinct deterministic Langevin seed, and the completion ledger
retains the ordered Colvars and trajectory artifacts needed by later PMF
analysis. This is a valid thermostatted process but not bitwise identical to a
single monolithic Langevin run.

One workspace-wide lock covers generation and all requested stages. A killed
launcher leaves the lock fail-closed. `--recover-stale-lock` archives it only
after verifying on the same host that the recorded PID/start identity is no
longer active; it never removes a lock merely because it is old.

The generated production input uses `lj/cut/dprc/batch`,
`tip4p/long/dprc/batch`, and `pppm/tip4p/dprc/batch`.  All partition roots
contribute to one GPU-local classical owner, so the run does not create one
PPPM/cuFFT plan or one GPU-package pair runtime per window.  The external
tutorial force-field include is copied only into the run workspace and only
the `lj/cut` style token is changed; all coefficient bytes remain unchanged.

After all three production ledgers pass, reconstruct the reaction-coordinate
PMF with the separately hash-pinned analysis contract:

```bash
python3 tools/analyze_etpeth_pmf.py \
  --run ../lammps-dprc-runs/etpeth \
  --output-prefix ../lammps-dprc-runs/etpeth/analysis/no-dprc-pmf
```

The analyzer first validates the equilibration chunk DAG, then requires every
production trial to start from its exact final `{path, sha256}` states. It
fails closed if a runtime, generated input, chunk record, trajectory, or
Colvars file changed. Duplicate chunk-boundary samples may differ only within
the manifest's explicit `write_data` decimal-round-trip tolerances.

The estimator uses the predeclared histogram-WHAM grid, reports neighboring
empirical overlap and time-correlation ESS for every window, and constructs
pointwise uncertainty by resampling correlation-length blocks separately
within each of the three trials. Every trial must support the complete pooled
low-free-energy region. The reported curve is the reaction-coordinate PMF
under the common attack-angle restraint; it does not claim that the angle
restraint was removed. The JSON records the analyzer bytes and repository
state; only clean analysis of publication-qualified inputs is marked `final`.
