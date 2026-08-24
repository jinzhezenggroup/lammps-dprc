# Batched classical CUDA contract

## Scope

The production target is the fixed-cell ETP/ETH umbrella workload: up to 48
independent one-rank LAMMPS partitions, 8,938 atoms per partition, a restricted
triclinic cell, TIP4P-Ew water, a 9 A real-space cutoff, and an order-4
50-by-50-by-50 PPPM mesh.  One process on the GPU-local partition-root
communicator owns the CUDA stream, cuFFT plans, and persistent workspaces.
Other partition roots contribute frames and receive only their result slices.

The first CUDA implementation is binary64 throughout.  A mixed-precision path
is a separate, opt-in experiment and cannot replace this reference until it is
qualified for energies, forces, charges, trajectories, and the final PMF.

## One-step execution

For one synchronized timestep the owner performs the following transaction:

1. Each GPU-local root writes wrapped coordinates and charges into its stable
   MPI shared-memory slot.  Validate the timestep, cell epoch, atom order,
   fixed topology, and result extents before the owner reads the batch or
   changes any LAMMPS output.
2. Construct TIP4P M sites and assign the MM-only charge density for all
   windows.  Execute one batched forward FFT, form the MM spectral potential,
   and execute batched inverse transforms for the scalar potential and the
   three electric-field components.
3. Project the MM reciprocal potential at the QM atoms.  Add LAMMPS's uniform
   background term and subtract the real-space `erf(alpha*r)/r` complement so
   the xTBloom periodic operator has the same meaning as
   `pppm/tip4p/dprc`.
4. Call one xTBloom ragged-plan computation for the synchronized windows.
   Stable window identity is also stable xTBloom plan-slot identity.
5. Assign the QM-only charge density from the converged SCC charges.  Execute
   the second batched forward FFT and the three inverse field transforms.
   Obtain the MM/QM bilinear energy and virial from the retained MM and QM
   spectra; do not perform a third forward FFT.
6. Interpolate the QM-only field for the QM/QM subtraction, add the retained MM
   field in-place, and interpolate the assembled full field for production
   PPPM publication.  Redistribute every M-site force to O/H/H with weights
   `(1-alpha, alpha/2, alpha/2)`.
7. Evaluate full LJ plus MM-only real-space Ewald Coulomb from a GPU-built full
   Verlet list.  Reuse the coordinates already resident from step 1 and reuse
   the list itself until a synchronized window exceeds `skin/2`.
8. Commit force, energy, and all six virial components for a window only after
   every required result slice for that window is complete.  A call-level
   failure commits nothing; a future peer-local policy must explicitly fill a
   failed window with NaNs rather than publish a partial classical result.

The immediately following ordinary LAMMPS pair and KSpace calls are consumers
of the prepared result.  They must not launch another neighbor traversal,
charge assignment, or FFT.

## Real-space Hamiltonian

The ETP/ETH input currently uses

```text
pair_style hybrid/overlay lj/cut 9.0 tip4p/long 6 7 1 1 0.125 9.0
pair_modify pair lj/cut tail yes
special_bonds amber
```

The fused backend therefore computes two distinct terms in one traversal:

- 12-6 LJ for every mapped type pair, using LAMMPS's `lj1`--`lj4`, cutoff,
  offset, and special-LJ scale;
- real-space Ewald Coulomb only for the MM mapping represented by the TIP4P
  sub-style.  An oxygen charge is evaluated at its implicit M site, while the
  LJ displacement remains the true atom-atom displacement.  The Coulomb force
  and energy include LAMMPS's special-Coulomb complement.

The backend consumes an explicit fixed-topology special-pair CSR.  It must not
infer exclusions from atom indices or molecule layout.  LJ tail energy and
pressure retain LAMMPS's existing type-count/volume semantics and are not
included in the pair kernel's force or six-component configurational virial.

Neighbor pairs use the restricted-triclinic minimum image.  A production GPU
cell list is rebuilt according to the same cutoff-plus-skin displacement rule
as LAMMPS.  The correctness oracle may use an all-pairs traversal, but timings
from that oracle are never eligible for a performance claim.

## TIP4P geometry

For an oxygen at `rO` and its two bonded hydrogens in the closest periodic
images, the implicit charge site is

```text
rM = rO + alpha/2 * ((rH1-rO) + (rH2-rO))
alpha = qdist / (cos(theta0/2) * r0)
```

The topology records the O/H/H atom indices explicitly.  QM atoms may not be
members of an implicit TIP4P molecule, matching the current fix restriction.
Charge-site force redistribution is linear and must be used consistently for
real-space Coulomb, reciprocal-space interpolation, xTB point-charge forces,
and virial construction.

## PPPM numerical meaning

The reference is the pinned LAMMPS ik-differentiated PPPM implementation:

- the same mesh dimensions, interpolation order, `g_ewald`, spline
  coefficients, Hockney-Eastwood influence function, FFT normalization, and
  reciprocal-index convention;
- restricted-triclinic mapping through fractional coordinates and reciprocal
  vectors `2*pi*H^-T*n`;
- binary64 charge assignment, complex FFT storage, spectral products,
  interpolation, reductions, and publication;
- global energy including the Ewald self term and uniform-background term;
- all six global virial components using LAMMPS's reciprocal virial factors;
- no slab correction, no per-atom KSpace energy/virial, and no analytic (`ad`)
  differentiation in the initial production backend.

cuFFT is a runtime-provided NVIDIA dependency.  The project does not vendor or
redistribute CUDA libraries.  The plugin remains tied to the exact pinned
LAMMPS/MPI/compiler build and continues to export only `lammpsplugin_init`.

## Acceptance gates

Before the batched backend can replace the current path it must pass:

- one-window comparisons against pinned LAMMPS for MM-only, QM-only, and full
  TIP4P PPPM energy, forces, scalar potential, and six virial components;
- one-window comparisons for LJ, TIP4P real-space Coulomb, Amber special-bond
  scaling, LJ tail values, and combined hybrid-overlay accounting;
- central finite differences and translation invariance on orthogonal and
  restricted-triclinic cells;
- batch-versus-sequential parity for sizes 1, 2, 4, 8, 16, 32, and 48;
- an end-to-end two-partition run with different geometries in the two stable
  slots and one shared classical/xTBloom owner;
- full QM/MM comparisons against `qmmm/xtb/dprc`, including SCC charges,
  xTB point-charge forces, total energy, total force, and pressure;
- repeated-step, changed-geometry, changed-cell-epoch, failure rollback, and
  prepared-result token tests;
- real RTX 5090 execution plus Compute Sanitizer memcheck, racecheck,
  initcheck, and synccheck on the production call chain;
- correctness-qualified aggregate accepted steps/s/GPU with warmups and
  repeated raw samples.  Startup and plan creation are reported separately
  from steady-state MD-loop throughput.
