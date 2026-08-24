# Production DPA4c DPRc model plan

LAMMPS-DPRc does not yet ship a real DPA4c correction model. The production
target is the paired periodic correction

```text
periodic PBE0 QM/MM - production xTBloom GFN2-xTB QM/MM
```

Both terms must use the same complete frame, periodic operator, classical
environment, charge convention, and force publication convention. A generic
absolute potential or an older MNDOD-based correction is not a substitute.

## Phase 1: freeze the scientific contract

Record the contract in a machine-readable manifest before generating labels:

- unique DeePMD type map: `P O C H OW HW`;
- LAMMPS types may map both QM oxygen types to the single DeePMD type `O`;
- QM atoms: IDs 1 through 16;
- QM charge: -2; UHF: 0;
- compact environment: QM atoms plus complete molecules within 6 Angstrom;
- exact zero MM/MM contribution by construction;
- xTB reference: production xTBloom GFN2-xTB with the 300 K Helmholtz free
  energy convention;
- explicit periodic `b + A q` operator, point-charge gamma values, and TIP4P
  force redistribution;
- binary64 label generation and immutable unit conversions.

Changes to this contract create a new dataset and model lineage rather than a
silent update.

## Phase 2: close the xTB oracle boundary

The low-level AmberTools/libxTB label path and production xTBloom must agree on
identical periodic inputs before subtraction is admitted as training data.
Compare energy, QM charges, QM forces, MM point-charge forces, SCC status, and
the complete periodic operator derivative convention.

Do not correct a mismatch by loosening tolerances or regenerating a reference.
Classify and resolve the physical convention first.

## Phase 3: generate source configurations

Sample all 48 umbrella windows with multiple independent trajectories or
trials. Include ordinary equilibrium frames and deliberately enriched cases:

- reaction and transition regions;
- high-force configurations;
- SCC-hard or slowly converging frames;
- compact-cutoff molecule crossings;
- perturbed cells, coordinates, and solvent environments within the intended
  domain.

Split train, validation, and test data by complete trajectory and window
blocks. Never randomly split neighboring MD frames across sets.

Use an uncertainty-driven acquisition loop only after an initial physically
diverse seed set exists. Every acquisition decision must be reproducible from
the model ensemble, frame identity, and threshold recorded in the ledger.

## Phase 4: produce paired binary64 labels

For each candidate, evaluate PBE0/QUICK and xTBloom on the same complete frame
and cell. The two calculations must share point charges, gamma values,
periodic `b + A q`, and TIP4P virtual-site redistribution. Publish a correction
only after both peers converge and every requested energy, charge, and force
slice is complete.

Reject partial pairs, nonconverged peers, changed geometries, and mismatched
operator metadata. Preserve hashes for the source frame, both engine inputs,
both outputs, the subtraction record, and the code that performed the
subtraction.

## Phase 5: compact after subtraction

Create the DPA4c sample only after the complete periodic correction has been
formed. Retain the QM atoms and every complete molecule within 6 Angstrom,
along with an immutable compact-to-full atom mapping.

Enforce MM/MM exclusion structurally. Test isolated MM atoms, complete
MM-only systems, and cutoff-crossing molecules for exact zero correction and
force. Do not depend on the training process to learn this invariant
approximately.

## Phase 6: train an ensemble

If scientific closure requires it, first train a DPA4 teacher or reference
model on the real paired labels. Train or distill the DPA4c student from those
real labels; do not convert DPA4 weights into a DPA4c model and assume the
architectural contract survived.

Train four independent DPA4c seeds with fixed, published configurations. The
primary model runs every MD step, and the ensemble supplies uncertainty and
model-deviation evidence. Record data selection, random seeds, optimizer
state, precision, code revision, dependencies, and all intermediate model
hashes.

Start with FP64 label handling and a scientifically conservative inference
configuration. FP32 or mixed precision is a separate optimization experiment,
not a default model property.

## Phase 7: qualification gates

A release candidate must pass all of the following without weakening a
tolerance:

1. Held-out energy and force errors, reported by species, QM region, compact
   environment, and distance from the compact boundary.
2. Exact isolated-MM and MM-only zero tests.
3. One-window versus partition-batched parity for energy, atomic energy, and
   forces.
4. Charge and SCC response checks against the paired FP64 reference.
5. Short NVE and NVT stability tests, including cutoff crossings and SCC-hard
   cases.
6. Umbrella-window overlap and effective-sample checks.
7. A complete PMF with uncertainty compared with the FP64/reference workflow.
8. Independent energy, force, charge, SCC, trajectory, and PMF qualification
   for any FP32 or mixed-precision variant.

Performance measurements become eligible only after the exact model artifact
used for timing passes these gates.

## Phase 8: release the model lineage

Publish a model card and immutable manifest containing:

- model SHA-256 and byte size;
- architecture, type map, cutoff, precision, and correction target;
- exact training code and configuration revisions;
- label-engine revisions and artifact hashes;
- source-frame and paired-label provenance hashes;
- train/validation/test split ledger;
- qualification results and known scientific limitations;
- dataset, engine, model, and redistribution licenses.

Do not place model bytes in this repository until the revision, hash, license,
and distribution review is complete. The recommended release shape is a
separate, system-specific model artifact named
`dpa4c-etpeth-pbe0-xtb-dprc`, with large labels and raw calculations archived
in an immutable data repository. This repository should retain only the model
manifest, expected hash, public download reference, and user-facing
compatibility contract.

## Execution milestones

The shortest credible path is:

1. close xTBloom/libxTB oracle parity;
2. generate and audit a small paired-label pilot spanning all configuration
   classes;
3. train four pilot DPA4c seeds and run the zero/parity/stability gates;
4. expand labels through block-split active learning;
5. freeze the dataset and train release candidates;
6. qualify the full umbrella PMF;
7. publish the model artifact and rerun the paper benchmark from clean
   revisions.
