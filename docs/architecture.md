# Architecture

## End state

LAMMPS-DPRc coordinates multiple independent umbrella windows while preserving
one ordinary LAMMPS state machine per window. A node-local broker owns the GPU
batch resources.

```text
per window/partition                         per GPU
--------------------                        -------
classical MM + restraints ---- descriptors ---> batch broker
local/fused PPPM ------------ b, A, fields ---> xTBloom CUDA plan
                                                   |
forces, charges, status <--------------------------+
optional compact DPRc ----------------------> scheduled GPU inference
```

The first implementation should use synchronized LAMMPS partitions. Every
partition root joins a communicator derived from `Universe::uworld`; non-root
ranks remain inside their normal LAMMPS `world`. This is a private LAMMPS C++
boundary and is therefore compiled and tested per supported LAMMPS revision.

An external asynchronous broker is a later option. It is not needed until
window imbalance or independently stopping replicas makes the synchronous
collective a measured bottleneck.

## Batch identity

- One stable batch slot corresponds to one umbrella window for the life of a
  fixed xTBloom plan.
- QM topology, molecular charge, spin, point-charge offsets, response shape,
  and compute policy stay fixed.
- Geometry, MM point positions, periodic inputs, and converged state change per
  step according to xTBloom's public plan contract.
- Near-MM lists use cutoff plus skin. The adapter groups their screening gamma
  values into permanent padded slot ranges; current points are remapped into
  compatible slots and every unused entry carries exact zero charge.
- Slot capacities grow monotonically. A new gamma class or real capacity
  overflow rebuilds the collective plan and restarts SCC from FRESH; an
  ordinary membership or tag change remains WARM and never reuses an
  incompatible gamma topology.

Use several measured micro-batches instead of assuming every window belongs in
one maximum-size batch. This limits the effect of a slow/nonconvergent SCC peer
and may improve occupancy.

The implemented synchronized all-window M1 core requires the immutable window
index, `Universe::iworld`, partition-root rank, and xTBloom slot to be the same
dense integer in `[0, nworlds)`. It preallocates the complete ragged input image
and requires every slot to stage the same MD timestep before compute. Its
software WARM state mirrors xTBloom's
native whole-plan checkpoint: a successful call with one failed peer preserves
that peer-local result isolation but makes the next call FRESH for the complete
micro-batch. This is intentionally stricter than independently marking the
successful slots WARM, which the current public plan ABI cannot represent.

`dprc/info` also exercises the synchronized communication boundary. Every
universe rank participates in `MPI_Comm_split`; only world rank zero joins the
root communicator, and `Universe::iworld` is the key. Therefore root rank is
the stable slot even when LAMMPS reorders universe ranks. This command is a
topology diagnostic, not the production descriptor collective.

Future micro-batches that contain a subset or permutation of global windows
must add and test an explicit permanent broker-rank-to-plan-slot map. They must
not weaken the current identity rule by silently sorting arbitrary window IDs.

The root-to-root numerical broker now makes this ownership concrete. Every
partition root contributes one fixed topology during collective construction.
Only root-communicator rank zero creates `XtbloomPlanExecutor`; steady calls
gather the ragged positions, point charges, and periodic `b + A q` data to that
rank and scatter only the corresponding result slices back. Rejected local
frames and timestep mismatches are detected collectively before the previous
result is invalidated. A native call-level failure is reported to every root
and revokes the whole-plan WARM checkpoint.

## Plugin and symbol isolation

LAMMPS plugins necessarily export the C entry point `lammpsplugin_init`. All
other project implementation types use either the `DPRC` namespace or unique
LAMMPS class names. The production force style is reserved as
`qmmm/xtb/dprc` / `LAMMPS_NS::FixDPRCXtb`; it must never register under the
built-in `qmmm/xtb` name, because LAMMPS explicitly allows a plugin to override
an existing same-name style.

LAMMPS loads plugins with `RTLD_NOW | RTLD_GLOBAL` on Unix. Namespace and class
names alone are therefore insufficient isolation: every implementation symbol
is compiled with hidden visibility, and platform export controls allow only
`lammpsplugin_init` into the dynamic symbol table. An automated ELF symbol test
guards this allowlist before the numerical broker is linked into the production
plugin.

The fused force implementation registers only `qmmm/xtb/dprc`, `pppm/dprc`,
and `pppm/tip4p/dprc`. At configure time, seven exact LAMMPS QMMM-XTB inputs are
copied into the build tree and checked by SHA-256; the five patched inputs are
checked again after applying the retained delta.
Compilation renames the generated classes to `FixDPRCXtbReference`,
`PPPMDPRC`, `PPPMTIP4PDPRC`, and `DPRCXtbEwald`, and also renames the helper
and adapter entry points. The plugin therefore neither registers native style
names nor defines or consumes the existing `FixQMMMXTB`, `PPPMXTB`,
`PPPMTIP4PXTB`, `QMMMXTBEwald`, or `lammps_qmmm_xtb_*` symbols. None of these
private C++ boundaries is a stable cross-version ABI.

Each world root translates the legacy adapter request into one local broker
frame. xTBloom forces are converted back to the gradients expected by the
pinned fix, while the fix retains ownership of LAMMPS unit conversion and the
coordinate derivatives of its periodic `b + A q` operator. The adapter commits
energy, gradients, and charges to the fix only after the collective call
succeeds and every partition root confirms that its local SCC result is usable.
This all-window publication gate prevents one failed slot from returning while
successful roots advance into a later collective. The derived fix also keeps a
reusable rollback image of atom charges/forces and the top-level pair/KSpace
energies and virials around the reference `pre_force` transaction; library-mode
exceptions therefore do not expose the temporary zero-charge or cleared-force
state used before SCC. Rollback itself performs no MPI collective so a
rank-local `error->one()` can still escape without deadlocking its peers. The
fused field is only prepared inside the reference transaction; the derived
wrapper commits it after that transaction returns successfully. The commit is
bound to one timestep, run/minimize mode, setup phase, and virial flags. A
failure clears both prepared and pending state without allocation or
collectives; the next ordinary `poisson_ik()` overwrites every owned field
element before grid forward communication rebuilds ghosts. Every new DPRC
pre-force transaction also clears an orphan token first, covering failures in
later PRE_FORCE, bonded, pair, or setup work after this fix has returned but
before the production KSpace call consumed its commit.

At the first force call and every LAMMPS neighbor rebuild, the reference fix
selects an MM point superset out to `cutoff + neighbor skin`. Between those
boundaries it updates only the stable tags in that superset, avoiding the full
MM-by-QM selection scan. A site outside the physical cutoff remains present
with exact zero charge; the explicit xTB embedding, matching Ewald subtraction,
point force, virtual-site redistribution, and virial contribution are all
linear in that charge and therefore remain zero. The adapter compares the
exact binary64 screening gammas with permanent sorted slot ranges, maps each
current point into a compatible slot, and returns only those mapped
point-force slices to LAMMPS. Unused slots retain finite positions and exact
zero charge. A neighbor rebuild that replaces tags or changes physical point
count therefore keeps the same plan and strict WARM checkpoint. Only a new
gamma class or class-capacity overflow grows the monotonic topology; every
root then recreates the broker collectively and intentionally invalidates the
whole-batch checkpoint. Geometry, point values, and periodic response fields
remain per-step data.

## Fused two-solve path

The native QMMM-XTB reference performs separate mesh work for MM potential
projection, MM-only energy/forces, QM-only subtraction, and normal production
PPPM. The private fused path now uses two mesh solves per timestep:

1. Before SCC, the MM-only solve produces MM-MM energy/forces. Its retained
   reciprocal potential is inverse-transformed once for projection at QM atoms,
   then its spectral array, fully communicated field, energy, charge sum, and
   virial are cached.
2. After SCC, the QM-only solve produces the QM field. Bilinear spectral terms
   supply the MM/QM cross energy and virial, including the uniform-background
   cross term; the cached MM field is added in place.
3. The immediately following production KSpace call consumes the assembled
   full field once, invokes the inherited orthogonal/triclinic/TIP4P
   `fieldforce()` path, and publishes the assembled global energy and virial
   without another mesh solve.

The path is restricted to ik differentiation and rejects per-atom KSpace
output. It also requires the production pair and KSpace computations plus the
selected Coulomb pair sub-style to remain enabled. Native/private style
mismatches fail during initialization. Focused CPU tests qualify orthogonal
and triclinic PPPM/TIP4P energy, charges, all forces, and global
virial-equivalent pressure against the native reference, plus central finite
differences and same-instance exception recovery.  The batched CUDA derived
style has additionally passed a real RTX 5090 triclinic reference comparison;
long production trajectories and archived performance evidence remain open.

The direct QM-image Ewald coefficients are cached within a run. Every LAMMPS
`init()` boundary invalidates that cache, so an explicit `change_box` between
runs cannot reuse coefficients from the old cell. Continuously changing-box
fixes remain rejected.

Other steady-state work to remove:

- the neighbor-epoch MM-by-QM membership scan when a suitable full neighbor
  list can supply the same cutoff-plus-skin candidates;
- repeated temporary vector allocation;
- redundant MPI collectives for related point-charge fields;
- host gather/scatter copies once the xTBloom and classical adapters can accept
  a common device-resident frame layout.

## Implementation phases

### Phase 0: boundary scaffold

- Exact dependency pins and clean-tree checks.
- `dprc/info` plugin load/ABI diagnostic.
- Benchmark and precision contracts.

### Phase 1: xTBloom batch broker

- One rank per window initially.
- One xTBloom context and fixed ragged plan per GPU micro-batch.
- Root-to-root ragged frame gather and result scatter with one native owner.
- Existing CPU TIP4P/PPPM remains the correctness reference.
- Compare serial xTBloom calls with batched FRESH/WARM calls before timing.

### Phase 2: fused QM/MM electrostatics

- Two-pass private PPPM/TIP4P implementation and CPU focused qualification are
  complete.
- Cache point-charge buffers; fixed-cell direct-Ewald structures are already
  cached within each run and refreshed at the next `init()` boundary.
- The custom batched CUDA derived style has a real-GPU one-window reference
  comparison and a two-window, distinct-geometry end-to-end partition
  regression.  Long trajectories, profiler evidence, and clean 48-window
  production measurements remain.

### Phase 3: device-resident classical path

- One GPU-local `ClassicalBatchPlan` now owns persistent FP64 CUDA buffers,
  pinned staging, shared cuFFT workspace, and batch-size `cufftPlanMany`
  handles; no partition creates an upstream PPPM grid or cuFFT plan.
- Retained QM/MM `begin_mm()` performs one batched forward plus four inverse
  transforms; terminal pure-MM execution performs one forward plus three
  inverses.  `finish_qm()` performs one forward plus three inverse transforms
  and forms full energy/virial from retained MM/QM spectra without a third FFT.
- The CUDA regression wraps only the test executable's cuFFT calls and asserts
  planMany batch shapes `B`, `4B`, and `3B` plus execution order forward,
  inverse, forward, inverse.  The production plugin carries no counter ABI.
- One warp owns each atom's full (Newton-off) Verlet neighborhood.  Repeating
  pair arithmetic from the partner warp avoids contended partner-force
  atomics; energy and virial remain single-counted.  The GPU-built list is
  reused while every window stays within `skin/2`; a larger displacement
  rebuilds the synchronized batch once.
- GPU-local roots write stable slots in one MPI shared-memory window, removing
  the large host Gather/Scatter copies from the classical path.  KOKKOS is not
  used and should be reconsidered only if a later device-resident LAMMPS
  adapter can also eliminate the remaining per-window pack/publication copies.

### Phase 4: DPRc

- Reuse the compact center/environment selection semantics from DeePMD PR
  #5943.
- Share one cached selection/neighbor mapping between QM embedding and DPRc
  where their cutoffs permit it.
- Schedule DPRc and xTBloom on the GPU using measured stream/context policies;
  do not assume concurrency improves throughput.
