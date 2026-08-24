# DPRc label contract

## Production target

The correction model must learn the difference between two evaluations of the
same full QM/MM frame:

```text
target = PBE0 QM/MM - xTBloom GFN2-xTB QM/MM
```

Coordinates, classical MM terms, point charges, periodic treatment, QM region,
and force mapping must be identical on both sides. A model trained on an
absolute potential, or on a correction against a different semiempirical
Hamiltonian, is not an xTB DPRc model.

The machine-readable contract is
[`workloads/etpeth/dprc-labels.json`](../workloads/etpeth/dprc-labels.json).
It fixes the target semantics and the source fields needed to reproduce each
label. QUICK 25.03 through AmberTools 26 `update.1`, PBE0, and the QUICK
`6-31GD.BAS` definition of 6-31G* are now hash-pinned. The engine installation
completed two CUDA QM/MM force calls, and the target ETP/ETH topology has now
completed a separate binary64 periodic label qualification on the RTX 5090.
That qualification covers binary64 preservation of the values parsed from the
hash-pinned Amber restart without an additional binary32 round trip, periodic
QUICK/PME embedding, exact TIP4P extra-point force redistribution, atomic
no-overwrite publication, one QUICK call per frame, exact original `ntc=2` /
`ntf=1` input, positive-volume cells, bitwise frame/label geometry identity,
and selected central finite differences.
The formatted restart records coordinates to seven decimal places, so this is
not a claim that the source carried arbitrary 53-bit coordinate precision. It
does not provide an independent PBE0 oracle or the matching xTBloom periodic
operator.
In particular, every source frame must retain:

- the complete triclinic system, stable atom and molecule IDs, and type map;
- every MM point charge and per-point gamma in Hartree atomic units, plus the
  SHA-256 of the deterministic gamma-policy manifest;
- the periodic atomic shift `b`, response matrix `A`, and the complete
  coordinate-derivative force that xTBloom intentionally excludes;
- raw xTBloom QM and point-charge forces, result flags, and the final mapped
  QM/MM forces after periodic-operator and TIP4P virtual-site contributions;
- high-level and xTB energy/force results before subtraction;
- the complete-molecule compact membership and compact-to-full atom mapping.

The xTB energy at 300 K is the electronic Helmholtz free energy. xTBloom holds
`b` and `A` fixed during differentiation and must publish
`XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES`; the label
generator must add the missing operator derivatives. It must also retain the
TIP4P-Ew M-site definition and prove the M-site-to-O/H force redistribution.
The training payload may be nonperiodic after compact selection, but the
periodic source calculation may not be discarded from its provenance ledger.

The DeePMD model type map is the unique list `P O C H OW HW`. The separate
seven-entry `lammps_type_to_model_species` mapping is allowed to map both QM
oxygen LAMMPS types to the same model type `O`. The QM region is the inclusive
LAMMPS atom-ID range `1:16`, not the two-atom set `{1, 16}`; its recorded
composition is `P1 O5 C3 H7`, charge is `-2`, and UHF count is zero.

## Legacy tutorial archive

The external `dprc-tutorial` archive is a useful compatibility workload, not a
production xTB dataset. Its documented `dpamber corr` workflow first forms

```text
PBE0 QM/MM - MNDOD QM/MM
```

in the full periodic system, converts Amber kcal/mol values to DeePMD eV and
eV/Angstrom, then wraps and applies the atom-level Amber mask
`((:1-2)<@6)&!@%EP`. The compact HDF5 discards the cell and marks every system
`nopbc`.

Audit it without copying the archive into this repository:

```bash
python3 tools/audit_dprc_labels.py \
  --dataset ../lammps-dprc-runs/dprc-labels/source/init_data.hdf5 \
  --output ../lammps-dprc-runs/dprc-labels/audit.json \
  --allow-unqualified-source
```

For the pinned tutorial bytes, the audit finds 3,600 frames in 389 ragged
systems with 226--285 atoms. Only 194 frames satisfy the necessary
stoichiometric condition `HW == 2*OW`; atom IDs and molecule IDs are absent, so
even those 194 frames do not prove complete waters. The other 3,406 definitely
contain atom-cut water shells rather than complete water molecules. The
payload also lacks separate high/low labels, the box, charges, atom/molecule
IDs, compact-to-full mapping, and periodic operator data. Those omissions make
a rigorous MNDOD-to-xTB baseline conversion impossible.

Although the arrays are stored as binary64, every coordinate and force value
round-trips exactly through binary32 while the energies do not. This describes
the precision of this legacy payload only; it is not evidence that an FP32
DPA4/DPA4C trajectory or PMF is accurate.

The tutorial repository has no asserted redistribution license and does not
pin the producer versions. Keep its bytes and any model trained from them
outside the repository and label the result `private-diagnostic`.

## Label regeneration

Production labels should be generated from the complete 48-window trajectory:

1. Select frames by window, trial, and absolute timestep before looking at
   force errors. Hash the source trajectory, data/restart state, and selection
   ledger.
2. Re-evaluate the xTB low-level side in binary64 with the exact production
   xTBloom method, SCC settings, point charges, `b + A q` operator, and force
   convention. Thermostat and umbrella forces must not contaminate the label.
3. Evaluate PBE0 on the same coordinates with the same MM Hamiltonian and
   periodic embedding. Record the electronic-structure engine, revision,
   basis, functional, convergence, charge/spin, and every failed frame.
4. Subtract energy and forces before compact selection. Do not silently drop a
   nonconverged peer or replace it with a partial force slice.
5. Select the QM region plus complete molecules within 6 Angstrom, retain the
   compact-to-full map, and prove that excluded MM/MM correction is exactly
   zero under the chosen model mask.
6. Split train, validation, and test data by trajectory/window blocks, not by
   randomly mixing neighboring frames from the same trajectory.

The current host has a hash-qualified external AmberTools/QUICK CUDA runtime.
Its manifest is
[`config/quick_pbe0_engine.json`](../config/quick_pbe0_engine.json), and the
compact report is
[`workloads/etpeth/evidence/quick-pbe0-qualification.json`](../workloads/etpeth/evidence/quick-pbe0-qualification.json).
Their `periodic-labeler-pending` scope names the earlier nonperiodic engine
fixture; the separate target-topology periodic qualification below supersedes
that pending gate without rewriting the historical engine evidence.
The report was generated from preserved raw bytes outside the repository with:

```bash
python3 tools/qualify_quick_pbe0.py \
  --run-directory /path/to/pbe0-6-31g-star \
  --sander-executable /path/to/sander.quick.cuda \
  --quick-executable /path/to/quick.cuda \
  --quick-library /path/to/libquick_cuda.so \
  --source-archive /path/to/AmberTools26.tar.bz2 \
  --update-patch /path/to/update.1 \
  --ambertools-license /path/to/AmberTools/LICENSE \
  --quick-license /path/to/AmberTools/src/quick/license \
  --cuda-config /path/to/AmberTools/cmake/CudaConfig.cmake \
  --cmake-cache /path/to/AmberTools/build/CMakeCache.txt \
  --basis-directory /path/to/quick/basis \
  --output workloads/etpeth/evidence/quick-pbe0-qualification.json
```

The periodic label interface is retained as
[`patches/ambertools26-dprc-binary64-label.patch`](../patches/ambertools26-dprc-binary64-label.patch),
with its stream generator and parser in
[`tools/dprc_binary64_io.py`](../tools/dprc_binary64_io.py). `DPRCFRM1` and
`DPRCLBL1` are little-endian binary64 streams terminated by an exact-size
`DPRCEND1` record. Labels are first written as `.partial`, checked, and then
atomically hard-linked to a previously absent final name before the partial
name is removed. Ordinary Amber NetCDF input cannot activate the writer,
and special force/Hessian/bias paths are rejected before QUICK is called. The
compact stream header intentionally does not identify a parm7 or atom map;
production generation must therefore verify their recorded SHA-256 values in
the external run manifest before accepting any label.

The low-level oracle is a second patch applied after that immutable PBE0
interface:
[`patches/ambertools26-dprc-xtb-label.patch`](../patches/ambertools26-dprc-xtb-label.patch).
An unset `SANDER_DPRC_LABEL64_ENGINE` retains the historical `QUICK-PBE0`
contract. `SANDER_DPRC_LABEL64_ENGINE=XTB-GFN2` instead requires the exact
GFN2-xTB, 300 K, accuracy `0.001`, 250-iteration, default-MM-hardness settings
and proves exactly one successful xTB call per frame while also proving QUICK
was not called. Unknown or empty engine tags, a mismatched QM theory, or a
changed xTB namelist fail before force evaluation and publish no label.

`tools/dprc_correction_io.py` subtracts two independently published labels
only after frame index, complete binary64 geometry, triclinic cell, TIP4P
policy, and zero extra-point force slots agree. It also requires the total-
potential and QM/MM-energy differences to agree within a declared tolerance,
which proves the shared classical Hamiltonian cancels. The output `DPRCCOR1`
record stores `PBE0 - xTB` energy and full-system force corrections in
binary64 and uses the same atomic no-overwrite publication rule.

An external one-frame diagnostic now exists for the reference ETP/ETH frame.
The upstream xTB source is clean at commit
`edcfbbe39d411edc225e27315fbda3a204ddb023` (source version 6.7.0), and the
linked Sander executable resolves that exact build rather than the conda xTB
library. The diagnostic produced one 11,912-site xTB label and one
`PBE0 - xTB` correction with bitwise-identical coordinates/cell, exactly zero
TIP4P extra-point forces, and zero recorded classical-energy cancellation
residual. This closes the label transport and subtraction mechanism only; it
does not yet prove AmberTools xTB and the production xTBloom periodic operator
are numerically equivalent, so it is not yet a production training corpus.

Compact retained evidence is in
[`workloads/etpeth/evidence/quick-pbe0-binary64-label-qualification.json`](../workloads/etpeth/evidence/quick-pbe0-binary64-label-qualification.json).
The reference label contains 11,912 sites and 2,974 TIP4P extra points; all
extra-point force slots are exactly zero after redistribution. Twelve labels
in one process produced twelve successful QUICK calls. The retained single
diagnostic samples were 15.27 seconds for the one-frame run and 73.28 seconds
for the twelve-frame run (6.11 seconds per frame amortized). They demonstrate
same-process execution without duplicate QUICK calls, but are not a statistical
performance comparison or the final 48-window xTB QM/MM+DPRc throughput metric.

Selected central differences agree for a QM hydrogen and TIP4P O/H atoms. The
retained failed diagnostic on the QM phosphorus also shows that QUICK's PBE0
atom-centered numerical integration can make heavy-atom energy differences
strongly step-size dependent; the evidence does not hide that result or widen
a tolerance to admit it. The next scientific gate is to compare the independent
AmberTools+xTB label against xTBloom on the same periodic point-charge/operator
and full-force convention; only the qualified xTBloom result may become the
production low-level side before compact selection.

## DPA4 and DPA4C design candidate

DPA4 is the leading candidate for the first scientific teacher. Before that is
treated as an implemented training path, a pinned DeePMD revision and real
configuration must prove that the model preserves distinct `O/OW` and `H/HW`
roles, excludes all MM--MM neighbor pairs, fixes the isolated MM atomic-energy
baseline to zero, and supports the required compact inference. A generic
pretrained DPA4 checkpoint is at most an initialization or teacher because its
ordinary element type map and absolute-potential target do not satisfy this
contract.

DPA4C is a throughput-oriented student rather than a checkpoint conversion.
Distill or train it on the same xTB correction labels only after compressed
inference supports the exact MM--MM exclusion (or an independently equivalent
mask) and compact `deepmd/kk` evaluation. FP32 is eligible only after the
energy, force, trajectory, and full-PMF qualification sequence in
[`precision.md`](precision.md).
