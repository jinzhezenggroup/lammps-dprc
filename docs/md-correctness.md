# Corrected QM/MM dynamics and recovery boundaries

SHAKE must be defined after force-modifying thermostat and restraint fixes.
The input generator now enforces this ordering for MM, QM/MM, and QM/MM+DPRc,
with both host and Kokkos integration. Defining Langevin after SHAKE changes
forces after the constrained-position prediction and violates rigid-water
geometry. See the [LAMMPS SHAKE documentation](https://docs.lammps.org/fix_shake.html).
The copyable input also uses `timestep 1.0`: time is already in femtoseconds
under `units real`.

The ETP/ETH paper coordinate is `distance(1,12)-distance(5,1)`, or
`r(P,O5')-r(P,O2')`. Atom 5 is the attacking chain oxygen and atom 12 is the
leaving oxygen; the topology's oxygen names can be misleading. The initial
umbrella center therefore has the corresponding positive sign. Archived
opposite-sign manifests remain explicitly renderable, but are not silently
relabeled or pooled with corrected sampling.

Trajectories produced with incorrect thermostat/SHAKE ordering are diagnostic,
not qualified constrained-ensemble sampling. Preserve old labels and states;
their existence does not establish that old trajectories are valid for a PMF.
Use verified states as initialization when appropriate, then equilibrate and
sample with the corrected force ordering. Do not automatically accept an old
production ledger merely because a runner hash is known.

`tools/qualify_nve_stability.py` supports explicit QM/MM and QM/MM+DPRc
Hamiltonians. Their model identities and qualification scopes cannot be
interchanged. The production gate retains its drift/temperature limits and
requires full-start NVE evidence. Optional transient-discard diagnostics are
identified separately and cannot satisfy that production gate.

For independently initialized production trials, `--trial-scoped-lock` is
valid only with `--stage production` and exactly one `--trial`. Trial-local
locks reject duplicate workers, while a shared/exclusive POSIX workflow guard
prevents trials from overlapping initialization or whole-workflow writers.
All cooperating launchers must use this locking protocol. Existing completed
initialization records retain their own chunk policy when later stages change
chunk size; other scientific-input and artifact checks remain strict.

`--neighbor-every` and `--neighbor-check` are explicit overrides for newly
generated invocations, not edits to old manifests or records. A changed policy
requires its own numerical validation and dangerous-neighbor-build checks.
Neither an override nor source-snapshot relocation upgrades diagnostic data to
publication-qualified evidence.
