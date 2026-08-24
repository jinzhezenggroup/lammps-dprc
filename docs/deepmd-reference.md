# DeePMD compact-evaluation reference

The design reference is deepmodeling/deepmd-kit PR
[#5943](https://github.com/deepmodeling/deepmd-kit/pull/5943), pinned at
`835a42f001b41e1bee0646b7e1403855b7fe6340`.

The PR introduces `center_group`, `environment_cutoff`, and
`include_molecule` for `pair_style deepmd`. Its reusable ideas are:

- walk full-neighbor rows for center atoms instead of scanning every center
  against all local and ghost atoms each step;
- promote selected atoms to complete molecules when the correction requires a
  chemically complete environment;
- synchronize the selected subsystem across MPI ranks;
- compact inputs by marking excluded model atoms and reuse existing inference,
  communication, neighbor remapping, and output scattering;
- invalidate cached neighbor mappings whenever subsystem membership changes;
- keep excluded atomic outputs zero and compute model-deviation summaries only
  over selected atoms;
- test orthogonal and restricted triclinic cells, cutoff crossings, MPI ranks
  with no selected atoms, and neighbor-cache transitions.

LAMMPS-DPRc should share the center/environment selection with construction of
the xTB explicit point-charge environment where possible. Two independent
neighbor scans and molecule-promotion implementations would create both wasted
work and cutoff inconsistencies.

The PR does not solve:

- multi-window or multi-frame DeepMD batching;
- `pair_style deepmd/kk` compact evaluation, which it diagnoses as unsupported;
- xTB QM/MM or fused PPPM;
- GPU scheduling between xTBloom and DPRc.

## Implemented host composition boundary

The first runtime connection intentionally keeps DeePMD out of
`dprcplugin`.  A matching DeePMD LAMMPS plugin is loaded separately and its
compact correction is overlaid with the classical pair style:

```lammps
plugin load /path/to/libdeepmd_lmp.so
plugin load /path/to/dprcplugin.so

group qm id 1:16
pair_style hybrid/overlay &
  lj/cut/tip4p/long ... &
  deepmd dprc.pt2 center_group qm environment_cutoff 6.0 include_molecule yes
pair_coeff * * lj/cut/tip4p/long ...
pair_coeff * * deepmd C H HW O OW P
fix qmmm qm qmmm/xtb/dprc ...
```

This boundary avoids a DeePMD `DT_NEEDED` edge and avoids embedding DeePMD's
private C++ API in the LAMMPS-DPRc plugin.  It is still tied to the exact
LAMMPS compiler, MPI, integer-size, DeePMD plugin, and model artifacts used at
runtime.  Configuration of the optional composition test therefore requires
SHA-256 values for both the DeePMD plugin and model.

Both plugins necessarily define `lammpsplugin_init`; LAMMPS resolves that
entry point on each individual plugin handle. The optional symbol-overlap gate
requires every other dynamic implementation symbol to be disjoint between
`dprcplugin` and the selected DeePMD plugin. DeePMD may still define symbols
also present in the host LAMMPS executable, which is why the exact host and
DeePMD build ABI remains a hard pin rather than a cross-version guarantee.

The pinned QMMM-XTB reference fix calls only the selected long-range Coulomb
sub-style during its MM-only and full-charge pair captures.  It does not call
the complete `hybrid/overlay` pair object, so the DeePMD sub-style should run
only in LAMMPS's ordinary production pair evaluation. This execution-count
statement comes from inspection of the pinned source; numerical additivity
alone cannot count calls whose outputs cancel.  MM-only type routing is
queried through LAMMPS's common `PairHybrid` base, which keeps this behavior
identical for `hybrid/overlay` and `hybrid/overlay/kk`. The
`lammps.qmmm_xtb.deepmd_overlay` test independently proves the net correction
is neither omitted nor double-counted by checking

```text
energy/force(overlay) = energy/force(xTB QM/MM) + energy/force(compact DeePMD)
```

and also verifies zero DeePMD force on an atom excluded by compact selection.
This is a correctness and coexistence gate, not yet the multi-window batch
implementation.

## DPA4 and DPA4C as DPRc models

DPRc labels are correction labels, not absolute potential labels:

```text
target = high-level QM/MM - xTB QM/MM
```

The model must also preserve the original DPRc structural semantics: QM/QM
and QM/MM contributions are represented, MM/MM interaction is exactly zero,
and an isolated MM atom has zero correction energy.  The standard DeePMD
energy fitting path can enforce the last condition through its atomic-energy
vacuum subtraction, but a generic pretrained absolute potential does not gain
that property merely by being evaluated on a compact subsystem.

DPA4 is usable as a research teacher or direct DPRc model because it supports
type-pair exclusions and conservative energy/force training.  Its message
passing and lack of compression may make it inefficient for the small compact
subsystems targeted here.

DPA4C is the more attractive throughput-oriented student: it is one-hop,
strictly local, FP32, and supports compressed CUDA inference.  The current
upstream interfaces nevertheless have two independent blockers:

- compact `center_group` evaluation is rejected by `pair_style deepmd/kk`;
- compressed DPA4C rejects nonempty `exclude_types`, while exact DPRc needs an
  MM/MM exclusion or an independently proven equivalent mask.

Consequently, the pinned compact host composition can be used first with an
uncompressed DPA4 graph artifact for scientific closure. DPA4C requires a
separately reviewed commit that combines its merged implementation with the
still-open compact PR before even the host route is reproducible. Neither route
may be described as the final device-resident DPA4C path. The final path needs
compact selection in `deepmd/kk` plus either compressed type-pair exclusion or
another structurally exact DPRc masking scheme.

## Pretrained-model boundary

DPA4 weights cannot be converted into DPA4C weights.  DPA4 uses message
passing, whereas DPA4C uses one-hop degree-wise moments.  The supported route
is training or distilling a DPA4C student from a DPA4 teacher and/or the real
xTB-based correction labels.

Same-architecture checkpoint fine-tuning is an experiment only when the type
map is already identical.  Neither DPA4 nor DPA4C currently supports the
type-map change needed to turn ordinary elemental O/H embeddings into the
distinct DPRc `O/OW` and `H/HW` roles.  The upstream DPA4 pretrained registry
therefore provides, at most, a teacher or initialization candidate; it is not
an xTB DPRc model and cannot replace the missing xTB-labelled dataset.

The tutorial archive contains 3,600 MNDOD-to-PBE0 correction frames and no
xTB labels, boxes, virials, atom parameters, frame parameters, or periodic
charge-response data.  A production model still requires an exporter and
labeling workflow whose low-level QM/MM Hamiltonian, charges, cutoff, periodic
operator, and subsystem membership exactly match the xTBloom runtime.

## Device-edge FP32 follow-up

The inspected local follow-up branch contains API-version-28 device entry
points that are newer than the pinned PR head used as the compact-selection
design reference. In particular, `DP_DeepPotComputeEdgesGPUFloat32` and
`DP_DeepPotComputeCanonicalGraphGPU` keep coordinates and force-like outputs
in FP64 while storing edge vectors in FP32. The LAMMPS `deepmd/kk` path queries
this capability through `DP_DeepPotUsesFP32EdgeVectors` and builds the matching
device graph without a host round trip.

These APIs are promising for the RTX 5090 target, but they do not remove the
current incompatibility between `center_group` compact evaluation and
`pair_style deepmd/kk`, and they do not batch independent umbrella windows.
LAMMPS-DPRc should therefore treat them as a reviewed integration candidate,
not as part of the immutable PR #5943 reference pin. Before adopting them,
record the exact upstream commit and audit the final API, model support,
stream synchronization, accuracy, and packaging boundary.

The PR is still open. This repository references its immutable head and does
not copy its code. If it merges, update the pin to an upstream commit only after
reviewing the final diff and validation.

For four-model model deviation, the current host implementation evaluates the
models sequentially.  It does not batch models or umbrella windows.  Loading
four models independently in every partition also duplicates weights and GPU
runtime state.  The performance end state remains one GPU-local DPRc broker
that batches compatible frames across windows and owns model/context lifetime;
the host overlay is the reference implementation against which that broker
must be qualified.
