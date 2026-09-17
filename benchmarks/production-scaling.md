# Scaling the corrected production workload

`production_scaling.py` supplements the original comparison-matrix runner. It
measures QM/MM and QM/MM+DPRc at **every integer batch size from 1 through 48**,
without starting another free-energy production campaign. Pure MM is excluded.
It reuses the original runner's slowest-partition loop-time calculation.

## Scientific and timing contract

The input campaign must supply the corrected 48-window paper coordinate,
completed 100 ps QM/MM equilibration, and passed NVE records for both methods.
Every initial state is checked against its recorded SHA-256 and water
constraints. Both methods start from the same QM/MM configurations and
velocities; this is a matched throughput experiment, not additional equilibrium
sampling from either Hamiltonian.

The workload retains 8,938 atoms, 16 QM atoms, TIP4P-Ew, 298 K, 1 fs timesteps,
a 4 A neighbor skin with displacement checks every 10 steps, the production
9 A cutoffs/50-cubed PME grid, and the corrected thermostat-before-SHAKE order.
The model is the production primary DPA4c DPRc model; no ensemble inference is
included in the timed loop. xTB uses binary64, while the model uses its existing
float32 arithmetic without AMP. The workload is not an all-FP64 experiment.

Each method gets one scheduler-exclusive GPU, 48 physical CPU cores, one MPI
rank per window, and one thread per rank. Request `--hint=nomultithread`:
on n2 this reserves both SMT threads of each core (96 logical CPUs), while
using only one thread per core. Ranks are bound to distinct CPUs from the
Slurm-assigned allocation. The runner maps Slurm's socket/core/thread CPU
numbering to Linux CPU IDs; it does not assume the inherited OS affinity mask
is restricted, because n2 uses `task/none`. The host LAMMPS path uses the shared GPU brokers;
unused CPUs remain idle at small B. GPU visibility is never overridden.

The main experiment requires **both xTB and DeePMD inference batch sizes to
equal B**. DeePMD receives one disconnected graph containing all B windows;
it is not invoked on two-frame subbatches. The runner rejects legacy chunked
plans and any `DPRC_DEEPMD_MAX_FRAMES_PER_CALL` environment setting. Chunked
experiments, if performed separately, are diagnostics and must not be labeled
full-batch scaling. When full-batch parity fails, retain the failed matrix row
and diagnose it before admitting its timing; never relax the gate or replace
the reference to make a failed coordinate pass.

The deterministic window order starts near xi = 0 and repeatedly adds the
most distant uncovered window. Its prefixes are nested and cover both ends
early. There is a separate singleton measurement for every one of the 48
configurations. Work-matched speedup therefore compares a batch with the sum
of its own members' singleton times, not exclusively with the possibly cheaper
xi = 0 singleton. The larger batch sizes run in a fixed shuffled order.

Before a multi-window coordinate is timed, a separate 25-step run compares
every slot's initial energies, atomic forces, and charges with its matching
singleton reference. Thresholds are frozen in `plan.json` before GPU work:

- Plugin energy: absolute 1e-8 kcal/mol plus relative 1e-12.
- Other logged reductions: absolute 2e-5 in the logged units.
- Force components: absolute 1e-5 kcal/mol/A.
- Charges: absolute 1e-10 electron charges.
- Water bonds/angles: 2e-6 A and 5e-4 degrees, respectively.

The B=48 preflight is also checked against every singleton as it becomes
available. A failure, including one outside the first slot, blocks the entire
main matrix with an explicit error before more timing is attempted. CSV rows
record the requested xTB and DeePMD batch sizes separately; neither is a claim
of numerical acceptance when the row is blocked or failed.

Each timing process warms up for 25 steps, then executes five successive
100-step sections without reconstructing the simulation. The context starts
fresh, and each stable slot retains its own SCC WARM history. For section r,

\[
R_r = \frac{B\,100}{\max_w t_{w,r}},\qquad
S_B = \frac{\sum_{w\in B}\sum_r t_{w,r}^{(1)}}{\sum_r \max_w t_{w,r}^{(B)}}.
\]

The main metric is accepted window steps/s/GPU. At 1 fs, aggregate ns/day is
`0.0864 * steps_per_second`. Five successive sections are **not** five
independent statistical replicas. Report raw samples and their median/range;
do not treat their spread as an independent-replica confidence interval.

All final states must exist and satisfy the water constraints, finite thermo,
expected step count, and zero dangerous neighbor builds. Every accepted timing
process records its wall time separately from its loop times. Initialization,
qualification, and failed work remain in the raw records; none is silently
called successful steady-state throughput. Initial parity is a deployment
test, not an independent electronic-structure oracle or proof of long-time
trajectory identity.

## Running and inspecting

The `prepare` subcommand takes explicit paths for the completed campaign,
external tutorial, LAMMPS executable, plugin, xTBloom library, MPI launcher,
and primary model. See `python3 benchmarks/production_scaling.py prepare --help`.
It writes `plan.json` and refuses to overwrite an existing plan.

Run each mode in its own finite Slurm GPU allocation with 48 CPUs and the
qualified production library environment:

```sh
python3 benchmarks/production_scaling.py run --output RUN_DIRECTORY --mode qmmm
python3 benchmarks/production_scaling.py run --output RUN_DIRECTORY --mode qmmm-dpa4c
```

The commands above must execute inside the allocation, not on a login node.
The runner checks Slurm/GPU visibility, CPU allocation, and the full-batch
DeePMD policy. It never clears old outputs or automatically retries failed
coordinates. Each mode writes `runtime.json`, `summary.json`, `summary.csv`,
and the full singleton/batch gate and timing records. All 48 matrix rows
remain present, including unstarted and failed coordinates.

## Publication status

This runner deliberately keeps `publication_qualified: false`. The current
production plugin contains an unpublished executor patch, the source-to-binary
audit is incomplete, and the existing full-range free-energy acceptance gates
did not all pass. Numerical batch parity and successful MD timing must not be
misrepresented as satisfying those separate requirements. The resulting data
characterize the specified runtime, but are not yet a final clean-revision
performance claim under the repository's publication policy.

The corrected-affinity 2026-09-17 n2 QM/MM campaign (job 1803) continues.
The initial QM/MM+DPRc matrix (job 1804) was cancelled because its two-frame
inference policy does not qualify for the main full-batch experiment. Its
completed singleton references remain diagnostic evidence. Full-batch probes
reproduced energy/force parity failures for B >= 4. Captured per-window input
graphs match bit-for-bit, but GPU C API outputs differ before plugin conversion
or force scatter. Operand capture localizes the first discrepancy to the first
FP32 cuBLAS matrix multiplication of the fitting network: weights and descriptor
inputs are identical, while corresponding output rows differ with matrix size.
TF32 is already disabled and the operator uses pedantic FP32 math. A diagnostic
scan of legacy GEMM algorithm choices did not eliminate the difference. No
failed full-batch coordinate is eligible for qualified throughput. See
`docs/full-batch-inference-diagnosis.md` for the evidence and remaining gate.

The initial jobs 1800/1801 were cancelled after discovering that the inherited
affinity masks exposed all CPUs. Their rank wrappers therefore selected
overlapping CPUs despite independent Slurm allocations. Those timings are
invalid and are retained only as diagnostics, including the separate early
preflight 1802; they must not enter the performance table. This was a benchmark
launcher defect, not a newly discovered defect in completed production MD.
No production trajectory, model, restart, or active source snapshot is replaced.
