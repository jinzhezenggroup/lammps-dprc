# DeePMD compact-evaluation reference

The compact DPA4c evaluation reference is deepmodeling/deepmd-kit PR
[#5972](https://github.com/deepmodeling/deepmd-kit/pull/5972), pinned at
`bdb40072d57109e25e17842d24e1f0a926c00632`.

That PR introduced the useful `center_group`, `environment_cutoff`, and
`include_molecule` concepts together with the public canonical-graph C entry
point. It did not provide multi-window GPU batching; the broker and the
block-diagonal composition are implemented here.

## Implemented C API boundary

LAMMPS-DPRc now implements compact partition batching inside `dprcplugin.so`.
The production code includes only `deepmd/c_api.h` and requires C API version
30 or newer. Its v30 numerical entry point is:

```c
DP_DeepPotComputeCanonicalGraphGPU(...)
```

When a v31-or-newer header exposes the explicit frame-axis entry point, the
executor selects it after the same validation. The v30 path is equivalent for
the supported atomwise DPA4c model because disconnected block-diagonal graphs
cannot exchange messages.

The integration registers:

```text
dprc/deepmd/batch
dprc/deepmd/batch/kk
```

The unsuffixed style is the production default. The `/kk` name is an opt-in
LAMMPS style alias for a separately qualified Kokkos workflow; in both cases
the adapter builds the compact graph on the host while DeePMD inference uses
CUDA through the public C API.

One partition-root communicator is created across independent LAMMPS worlds.
Rank zero alone loads the model and owns the CUDA context. Every rank contributes
one compact canonical graph; the owner assembles one block-diagonal batch,
executes a single C API call, and scatters complete result slices only after the
whole call succeeds.

```lammps
plugin load /path/to/dprcplugin.so

group qm id 1:16
pair_style hybrid/overlay &
  lj/cut/dprc/batch 9.0 &
  dprc/deepmd/batch /path/to/dprc.pt2 &
    partition_batch yes center_group qm &
    environment_cutoff 6.0 include_molecule yes
pair_coeff * * dprc/deepmd/batch P O O C H OW HW
```

This host-style form is the recommended multi-window deployment because it
does not initialize one Kokkos CUDA context per partition. The `/kk` alias is
available only for a separately qualified Kokkos execution backend.

No separate DeePMD LAMMPS plugin is loaded. A DeePMD-enabled build has a direct
`libdeepmd_c` dependency, but the symbol gate rejects DeePMD C++ ABI symbols and
standalone DeePMD LAMMPS-plugin dependencies.

## Compact graph semantics

The LAMMPS adapter:

- requires one MPI rank per partition and a static center group;
- walks full-neighbor rows for center atoms;
- selects mapped environment atoms within `environment_cutoff`;
- optionally promotes selected environment atoms to complete molecules;
- recovers bonded atoms hidden by zero-valued `special_bonds` exclusions;
- folds periodic ghost copies to stable local atom identities;
- builds destination and source CSR indices for the model cutoff;
- converts LAMMPS distances to angstrom and publishes eV-based model outputs
  back in the active LAMMPS unit style;
- adds forces, global energy and virial, per-atom energy, and nine-component
  centroid atomic virial only after collective success.

`neigh_modify exclude`, dynamic center groups, reduced LJ units, r-RESPA, and
more than one MPI rank per partition are rejected rather than silently changing
the correction semantics.

## Current verification

The runtime test suite covers:

- C API header and version compatibility;
- plugin export and dynamic-dependency boundaries;
- host-style and `/kk`-style parity for energy, atomic energy, and force;
- zero atomic energy outside the center mask and nonzero reaction force on
  selected environment atoms;
- batch-2 equality with two independent batch-1 calls;
- QM/MM plus compact DPRc overlay additivity and exclusion behavior.

The current diagnostic model is not a real PBE0-minus-xTB correction model.
These tests qualify software behavior only.

## DPA4 and DPA4c as DPRc models

DPRc labels are corrections, not absolute potential labels:

```text
target = high-level QM/MM - xTB QM/MM
```

The model must preserve the required structure: QM/QM and QM/MM corrections
are represented, MM/MM correction is zero, and an isolated MM environment atom
has zero correction energy. Compact selection alone does not create those
properties in a generic pretrained potential.

DPA4 can serve as a scientific teacher or direct reference model. DPA4c is the
preferred throughput-oriented student because it is one-hop, local, and suited
to compressed CUDA inference. DPA4 weights cannot be converted into DPA4c
weights; the student must be trained or distilled from real xTB-based correction
labels.

The runtime type roles must distinguish chemically different uses where the
correction requires them, for example `O` versus `OW` and `H` versus `HW`.
Ordinary elemental pretrained checkpoints do not satisfy that contract merely
because their files load successfully.

## C API artifact provenance

The build requires a manifest that binds the exact DeePMD source revision, the
installed public header, and the linked `libdeepmd_c` bytes. Generate it with
`tools/deepmd_artifact_manifest.py` and pass it as
`DPRC_DEEPMD_ARTIFACT_MANIFEST`. CMake verifies the manifest before compiling
the plugin, so a header/library mix-up cannot silently enter a run. Dirty
development checkouts require the explicit
`DPRC_REQUIRE_CLEAN_DEPENDENCIES=OFF` diagnostic override and remain ineligible
for release evidence.

## Precision boundary

The canonical model must report device-edge, FP32-edge-vector, and canonical
graph capabilities. Atom types, CSR indices, frame sizes, atomic energies,
forces, and virials cross the C API in their declared integer or FP64 forms;
edge vectors are FP32. The plugin synchronizes before host publication because
the current C API does not expose the backend stream.

This mixed representation is an explicit experiment boundary. A model still
requires independent energy, force, trajectory, and free-energy qualification
against the FP64 reference.

## Model deviation

The in-plugin path currently supports exactly one primary model. Periodic
multi-model deviation is rejected by both the workflow and benchmark runners.
Model ensembles should be evaluated offline until a reviewed batched C API
schedule, memory policy, result semantics, and qualification protocol are
implemented.
