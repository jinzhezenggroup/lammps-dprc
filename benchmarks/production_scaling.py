#!/usr/bin/env python3
"""Measure all integer batch sizes using hash-checked production states.

This supplemental runner does not relax the publication gates in ``run.py``.
It separates numerical acceptance from source/publication qualification and
never promotes an exported or dirty runtime to publication-ready evidence.
The plan is written before timing; existing production files are read-only.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import etpeth_workload as W
import qualify_nve_stability as N


def load_timing_module():
    """Reuse the public slowest-partition timing definition without copying it."""
    spec = importlib.util.spec_from_file_location("scaling_timing", ROOT / "benchmarks/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": W.sha256(path)}


def checked(record):
    path = Path(record["path"])
    if W.sha256(path) != record["sha256"]:
        raise ValueError(f"Changed input or evidence: {path}")
    return path


def nested_windows(windows):
    """Start near xi=0, then add the farthest uncovered window deterministically.

    Prefixes are nested, span the reaction coordinate, and preserve slot
    identity. Matched per-window singleton timings correct for their different
    SCC costs when computing speedup; B=1 alone is not a universal baseline.
    """
    remaining = list(windows)
    selected = [min(remaining, key=lambda w: (abs(w.center), w.index))]
    remaining.remove(selected[0])
    while remaining:
        item = max(remaining, key=lambda w: (min(abs(w.center - x.center) for x in selected), -w.index))
        selected.append(item)
        remaining.remove(item)
    return selected


def require_full_deepmd_batch(plan, mode):
    """Reject legacy chunked protocols instead of silently relabeling them.

    B denotes simultaneous windows and the complete disconnected-graph neural
    inference request. An unset cap selects that public C API path; even a cap
    that happens not to split one particular B is forbidden in this protocol.
    QM/MM has no neural inference and is unaffected by the optional cap.
    """
    if mode != "qmmm-dpa4c":
        return
    layout = plan["layout"]
    if (layout.get("deepmd_batch_policy") != "full"
            or layout.get("deepmd_max_frames_per_call") is not None):
        raise ValueError("Main scaling requires a full DeePMD batch plan; chunked plans are diagnostic only")
    if "DPRC_DEEPMD_MAX_FRAMES_PER_CALL" in os.environ:
        raise ValueError("Unset DPRC_DEEPMD_MAX_FRAMES_PER_CALL: DeePMD inference batch must equal B")


class FullBatchParityError(ValueError):
    """Stop the main matrix while a largest-batch numerical mismatch is diagnosed."""


def check_preflight(root, summary, tag, check):
    """Check every available singleton, failing before more timing is attempted."""
    summary.setdefault("preflight", {})[tag] = check
    if not check["passed"]:
        message = f"Full-batch initial parity failed for {tag}; diagnose before timing"
        summary["status"] = "blocked-full-batch-parity"
        summary["references"][tag] = dict(status="failed", error=message)
        for coordinate in summary["coordinates"].values():
            coordinate.update(status="blocked", error=message)
        write_summary(root, summary)
        raise FullBatchParityError(message)
    write_summary(root, summary)


def geometry(path):
    """Check every water's periodic SHAKE geometry, not just mean temperature."""
    import numpy as np
    from lammps_data_to_dprc_frames import read_lammps_data

    data = read_lammps_data(path)
    molecules = {}
    for atom, kind in data.type_by_id.items():
        if kind in (6, 7):
            molecules.setdefault(data.molecule_by_id[atom], []).append(atom)
    vectors = []
    for atoms in molecules.values():
        oxygen = [i for i in atoms if data.type_by_id[i] == 6]
        hydrogen = [i for i in atoms if data.type_by_id[i] == 7]
        if len(oxygen) != 1 or len(hydrogen) != 2:
            raise ValueError("Incomplete TIP4P water")
        vectors.append([data.coordinates_by_id[h] - data.coordinates_by_id[oxygen[0]] for h in hydrogen])
    bonds = np.asarray(vectors)
    fraction = bonds @ np.linalg.inv(data.cell).T
    bonds = (fraction - np.rint(fraction)) @ data.cell.T
    lengths = np.linalg.norm(bonds, axis=2)
    angles = np.degrees(np.arccos(np.clip(np.sum(bonds[:, 0] * bonds[:, 1], axis=1) / np.prod(lengths, axis=1), -1., 1.)))
    bond_error = float(np.max(np.abs(lengths - 0.9572000000000001)))
    angle_error = float(np.max(np.abs(angles - 104.4906026584604)))
    return dict(waters=len(vectors), bond_error_angstrom=bond_error,
                angle_error_degree=angle_error,
                passed=len(vectors) == 2974 and bond_error <= 2e-6 and angle_error <= 5e-4)


def prepare(args):
    """Freeze the scientific contract, inputs, thresholds, and complete matrix."""
    output, campaign = args.output.resolve(), args.campaign.resolve()
    if (output / "plan.json").exists():
        raise ValueError("Do not replace an existing benchmark plan")
    manifest_path = campaign / "paper-manifest.json"
    manifest = W.load_manifest(manifest_path)
    if (manifest["dynamics"]["timestep_fs"] != 1.0
            or manifest["dynamics"]["neighbor_every"] != 10
            or not manifest["dynamics"]["neighbor_check"]
            or manifest["umbrella"]["reaction_coordinate"]["expression"] != "distance(1,12)-distance(5,1)"):
        raise ValueError("This protocol requires the corrected 1 fs/every-10 paper-coordinate inputs")
    windows = nested_windows(W.windows_from_manifest(manifest))
    if len(windows) != 48:
        raise ValueError("Expected all 48 paper windows")
    ledger_path = campaign / "qmmm/records/equilibrate-complete.json"
    ledger = read(ledger_path)
    if ledger["status"] != "passed" or ledger["total_steps_per_window"] != 100000:
        raise ValueError("The common initial states must have completed corrected equilibration")
    starts = {}
    for window in windows:
        path = checked(ledger["outputs"][window.tag]["data"])
        if not geometry(path)["passed"]:
            raise ValueError(f"Initial water geometry failed: {window.tag}")
        starts[window.tag] = identity(path)
    gates = {}
    for method in ("qmmm", "dprc"):
        path = campaign / method / "qualification/nve-stability.json"
        if read(path)["status"] != "passed":
            raise ValueError(f"Prior NVE qualification failed: {method}")
        gates[method] = identity(path)
    order = list(range(2, 49))
    random.Random(20260917).shuffle(order)
    plan = dict(schema_version=2,
        claim="Numerically checked aggregate accepted umbrella MD steps/s/GPU for the specified production runtime; not model accuracy or PMF convergence",
        publication_qualified=False,
        publication_blockers=["Production runtime source qualification remains incomplete",
                              "Runtime source-to-binary clean-revision audit remains incomplete",
                              "The existing full-range free-energy convergence gates did not pass"],
        modes=["qmmm", "qmmm-dpa4c"], batch_sizes=list(range(1, 49)), execution_order=[1, *order],
        window_order=[w.tag for w in windows], starts=starts,
        common_initial_states="Corrected-order QM/MM 100 ps equilibrated states, identical for both methods",
        measurement=dict(warmup_steps=25, sample_steps=100, repetitions=5,
                         samples="Five successive segments of one trajectory, not independent replica statistics",
                         boundary="maximum LAMMPS loop time over all partitions; setup and failures reported separately"),
        qualification=dict(steps=25, energy_atol=1e-8, energy_rtol=1e-12,
                           reduction_atol=2e-5, force_atol=1e-5, charge_atol=1e-10,
                           bond_atol_angstrom=2e-6, angle_atol_degree=5e-4,
                           maximum_postwarmup_bond_energy_kcal_mol=1e-5),
        layout=dict(gpus=1, allocated_cpus=48, mpi_ranks_per_window=1, threads_per_rank=1,
                    deepmd_batch_policy="full", deepmd_max_frames_per_call=None,
                    deepmd_frames_per_call="B", model_deviation_frequency=0,
                    xtb_precision="binary64", model_precision="float32, AMP disabled",
                    scc="fresh context per invocation; WARM history retained per stable slot after initialization"),
        manifest=identity(manifest_path), equilibration_ledger=identity(ledger_path), nve_gates=gates,
        tutorial=str(args.tutorial.resolve()), tutorial_source=W.verify_source(args.tutorial, manifest, allow_unqualified_source=True),
        runtime={name: identity(getattr(args, name)) for name in ("lammps", "plugin", "xtbloom", "model", "mpiexec")},
        code={str(p.relative_to(ROOT)): identity(p) for p in [Path(__file__), ROOT / "benchmarks/run.py",
              *[ROOT / "tools" / name for name in ("etpeth_workload.py", "qualify_nve_stability.py",
                                                  "lammps_data_to_dprc_frames.py", "dprc_binary64_io.py")]]})
    W.write_json_atomic(output / "plan.json", plan)
    print(output / "plan.json", flush=True)


def first_frame(path):
    """Read ID-sorted, full-precision initial forces and atomic charges."""
    with path.open() as handle:
        if handle.readline().strip() != "ITEM: TIMESTEP" or int(handle.readline()) != 0:
            raise ValueError("Initial parity frame is missing")
        if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
            raise ValueError("Missing atom count")
        count = int(handle.readline())
        for _ in range(4):
            handle.readline()
        fields = handle.readline().split()[2:]
        indices = [fields.index(x) for x in ("id", "q", "fx", "fy", "fz")]
        rows = [[float(row[i]) for i in indices] for row in (handle.readline().split() for _ in range(count))]
    if count != 8938 or len({row[0] for row in rows}) != count or not all(math.isfinite(x) for row in rows for x in row):
        raise ValueError("Malformed/nonfinite force frame")
    return sorted(rows)


def parity(reference, candidate, limits):
    """Compare a batch slot to the same initial single-window state."""
    left = N.thermo_rows(checked(reference["log"]))[0]
    right = N.thermo_rows(checked(candidate["log"]))[0]
    if set(left) != set(right) or left["Step"] != 0 or right["Step"] != 0:
        raise ValueError("Initial thermo fields differ")
    energy = {}
    for field in left:
        atol = limits["energy_atol"] if field.startswith("f_") else limits["reduction_atol"]
        rtol = limits["energy_rtol"] if field.startswith("f_") else 0.
        delta = abs(left[field] - right[field])
        allowed = atol + rtol * max(abs(left[field]), abs(right[field]))
        energy[field] = dict(difference=delta, allowed=allowed,
            passed=delta <= allowed + math.ulp(left[field]) + math.ulp(right[field]))
    a, b = first_frame(checked(reference["trajectory"])), first_frame(checked(candidate["trajectory"]))
    if [r[0] for r in a] != [r[0] for r in b]:
        raise ValueError("Atom identity differs")
    forces = max(abs(x-y) for ra, rb in zip(a, b) for x, y in zip(ra[2:], rb[2:]))
    charges = max(abs(ra[1]-rb[1]) for ra, rb in zip(a, b))
    return dict(energy=energy, maximum_force_difference=forces, maximum_charge_difference=charges,
        passed=all(x["passed"] for x in energy.values()) and forces <= limits["force_atol"] and charges <= limits["charge_atol"])


def invocation(root, plan, mode, windows, name, timing):
    """Run once, retain failed wall time, and accept only complete valid output."""
    require_full_deepmd_batch(plan, mode)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    manifest = W.load_manifest(checked(plan["manifest"]))
    # Full-precision logs are necessary for the unchanged parity thresholds.
    for window in windows:
        W.write_generated(root / "generated/colvars" / f"{window.tag}.conf", W.render_colvars(manifest, window))
    runs = [W.RunWindow(w, checked(plan["starts"][w.tag]), directory / w.tag, root, W.seed_for(manifest, w, 1)) for w in windows]
    commands = ["run 25"] + (["run 100 pre no"] * 5 if timing else [])
    text = W.render_lammps_input(manifest, Path(plan["tutorial"]), checked(plan["runtime"]["plugin"]), runs,
        steps=525 if timing else 25, trajectory_frequency=0 if timing else 25,
        mode=mode, lammps_execution_backend="host",
        deepmd_models=(checked(plan["runtime"]["model"]),) if mode == "qmmm-dpa4c" else (), run_commands=commands)
    text = text.replace("thermo_modify format float %.12g", "thermo_modify format float %.17g")
    text = text.replace("dump_modify trajectory sort id pbc yes", "dump_modify trajectory sort id pbc yes format float %.17g")
    input_path = directory / "input.in"
    W.write_generated(input_path, text)
    command = ["timeout", "--kill-after=10s", "900s", str(checked(plan["runtime"]["mpiexec"])), "-n", str(len(windows)),
        sys.executable, str(Path(__file__).resolve()), "rank", str(directory), str(checked(plan["runtime"]["lammps"])),
        "-partition", f"{len(windows)}x1", "-plog", str(directory / "log.lammps"), "-pscreen", "none", "-in", str(input_path)]
    record = dict(status="running", command=command, input=identity(input_path), timing=timing,
                  windows=[w.tag for w in windows], started_utc=W.dt.datetime.now(W.dt.timezone.utc).isoformat())
    W.write_json_atomic(directory / "record.json", record)
    start = time.monotonic()
    try:
        with (directory / "launcher.log").open("w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=directory)
        record.update(returncode=process.returncode, wall_seconds=time.monotonic()-start)
        if process.returncode:
            raise ValueError(f"LAMMPS return code {process.returncode}")
        outputs, logs = {}, []
        for index, run in enumerate(runs):
            log = directory / f"log.lammps.{index}"
            logs.append(log)
            rows = N.thermo_rows(log)
            expected = 525 if timing else 25
            if rows[-1]["Step"] != expected or any(row["Temp"] <= 0 or row["Temp"] > 1000 for row in rows):
                raise ValueError(f"Incomplete or unstable dynamics: {run.window.tag}")
            bond_energy = max(abs(row["E_bond"]) for row in rows if row["Step"] >= 25)
            if bond_energy > plan["qualification"]["maximum_postwarmup_bond_energy_kcal_mol"]:
                raise ValueError(f"Water bond energy failed: {run.window.tag}: {bond_energy}")
            dangerous = W.DANGEROUS_BUILDS.findall(log.read_text())
            if len(dangerous) != len(commands) or any(int(x) for x in dangerous):
                raise ValueError("Dangerous neighbor builds or missing neighbor statistics")
            constraint = geometry(run.final_data)
            if not constraint["passed"]:
                raise ValueError(f"Final water geometry failed: {run.window.tag}")
            outputs[run.window.tag] = dict(log=identity(log), data=identity(run.final_data), restart=identity(run.final_restart),
                geometry=constraint, maximum_postwarmup_bond_energy=bond_energy,
                final_step=rows[-1]["Step"], dangerous_builds=[int(x) for x in dangerous])
            if not timing:
                outputs[run.window.tag]["trajectory"] = identity(run.trajectory)
        record["outputs"] = outputs
        if timing:
            record["measurement"] = load_timing_module().collect_samples(logs, batch_size=len(windows),
                warmup_steps=25, sample_steps=100, repetitions=5, timestep_fs=1.)
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=str(error), wall_seconds=time.monotonic()-start)
    W.write_json_atomic(directory / "record.json", record)
    if record["status"] != "passed":
        raise ValueError(f"{name}: {record['error']}")
    return record


def command_record(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return dict(command=command, returncode=result.returncode, output=result.stdout)


def physical_allocation(job_text, topology_text, hostname, required=48):
    """Map Slurm abstract CPU IDs to one Linux PU per allocated physical core.

    task/none does not narrow sched_getaffinity(): it can expose the whole
    machine to every job. Slurm numbers CPUs by socket/core/thread, whereas
    Linux often puts SMT siblings in separate ID ranges. Never treat either
    the inherited mask or Slurm's abstract IDs as a ready-to-use Linux mask.
    The job must reserve at least ``required`` complete, distinct cores.
    """
    allocations = re.findall(r"\bNodes=(\S+) CPU_IDs=([0-9,-]+)", job_text)
    if len(allocations) != 1 or allocations[0][0] != hostname:
        raise ValueError("Expected one explicitly allocated local Slurm node")
    abstract = set()
    for item in allocations[0][1].split(","):
        ends = [int(x) for x in item.split("-")]
        if len(ends) == 1:
            abstract.add(ends[0])
        elif len(ends) == 2 and ends[0] <= ends[1]:
            abstract.update(range(ends[0], ends[1]+1))
        else:
            raise ValueError("Malformed Slurm CPU allocation")
    rows = [tuple(map(int, line.split(","))) for line in topology_text.splitlines() if line and not line.startswith("#")]
    ordered = sorted(rows, key=lambda x: (x[2], x[1], x[0]))
    if not abstract or max(abstract) >= len(ordered):
        raise ValueError("Slurm abstract CPU IDs exceed the hardware topology")
    cores = {}
    for index, (cpu, core, socket) in enumerate(ordered):
        cores.setdefault((socket, core), []).append((index, cpu))
    complete = [members for members in cores.values() if all(index in abstract for index, cpu in members)]
    if len(complete) != required:
        raise ValueError(f"Expected {required} reserved physical cores, found {len(complete)}; request --hint=nomultithread")
    return sorted(min(cpu for index, cpu in members) for members in complete)


def bind_allocation():
    """Bind the launcher before Hydra forks ranks; retain scheduler evidence."""
    job = command_record(["scontrol", "show", "job", "-dd", os.environ["SLURM_JOB_ID"]])
    topology = command_record(["lscpu", "-p=CPU,CORE,SOCKET"])
    if job["returncode"] or topology["returncode"]:
        raise ValueError("Cannot verify scheduler CPU allocation")
    selected = physical_allocation(job["output"], topology["output"], os.uname().nodename)
    inherited = sorted(os.sched_getaffinity(0))
    if not set(selected) <= set(inherited):
        raise ValueError("Allocated physical CPUs are outside the inherited OS mask")
    os.sched_setaffinity(0, selected)
    return dict(job=job, topology=topology, inherited_affinity=inherited, selected_physical_cpus=selected)


def write_summary(root, summary):
    """Keep every requested coordinate visible, including failed/unstarted rows."""
    W.write_json_atomic(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="") as handle:
        fields = ["batch_size", "xtb_batch_size", "deepmd_frames_per_call", "status", "median_steps_per_s_per_gpu", "min_steps_per_s_per_gpu", "max_steps_per_s_per_gpu",
                  "aggregate_ns_per_day", "matched_singleton_speedup", "publication_qualified", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for size in range(1, 49):
            neural_batch = size if summary.get("method") == "qmmm-dpa4c" else ""
            writer.writerow(dict(batch_size=size, xtb_batch_size=size, deepmd_frames_per_call=neural_batch,
                                 publication_qualified=False, **summary["coordinates"][str(size)]))


def run(args):
    """Execute one method under its own finite, exclusive-GPU Slurm allocation."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("A Slurm GPU allocation is required; do not override device visibility")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 48 or len(os.sched_getaffinity(0)) < 48:
        raise ValueError("The plan requires an allocation of 48 CPUs")
    allocation = bind_allocation()
    base = args.output.resolve()
    plan = read(base / "plan.json")
    require_full_deepmd_batch(plan, args.mode)
    for group in (plan["runtime"], plan["code"], plan["starts"], plan["nve_gates"]):
        for record in group.values():
            checked(record)
    root = base / args.mode
    root.mkdir(exist_ok=True)
    lock = (root / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "summary.json").exists():
        raise ValueError("Refusing to rerun/replace an existing matrix; recover explicitly")
    plugin = checked(plan["runtime"]["plugin"])
    metadata = dict(plan=identity(base / "plan.json"), affinity=sorted(os.sched_getaffinity(0)), allocation=allocation,
        environment={k: v for k, v in os.environ.items() if k.startswith(("SLURM_", "DPRC_", "DP_", "OMP_", "MKL_", "OPENBLAS_")) or k in
                     ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "HYDRA_LAUNCHER")},
        xtbloom=W.verify_loaded_xtbloom(plugin, checked(plan["runtime"]["xtbloom"]), dict(os.environ)),
        deepmd=W.verify_loaded_deepmd_c(plugin, dict(os.environ)),
        hardware=[command_record(c) for c in (["nvidia-smi", "-q"], ["lscpu"], ["ldd", str(plugin)], ["c++", "--version"])])
    W.write_json_atomic(root / "runtime.json", metadata)
    manifest = W.load_manifest(checked(plan["manifest"]))
    windows = nested_windows(W.windows_from_manifest(manifest))
    summary = dict(status="running", method=args.mode, plan=identity(base / "plan.json"), references={},
                   coordinates={str(b): dict(status="not-started") for b in plan["batch_sizes"]})
    write_summary(root, summary)
    # Exercise the largest allocation immediately, before paying for all 48
    # singleton timing references. This also provides an early cost estimate.
    preflight = invocation(root, plan, args.mode, windows, "preflight/batch48", False)
    references = {}
    # Singleton timing every member permits a work-matched speedup denominator,
    # rather than silently comparing a cheap one-window state to a harder batch.
    for window in windows:
        print(f"REFERENCE {args.mode} {window.tag}", flush=True)
        try:
            qualification = invocation(root, plan, args.mode, [window], f"reference/{window.tag}/gate", False)
            early = parity(qualification["outputs"][window.tag], preflight["outputs"][window.tag], plan["qualification"])
            check_preflight(root, summary, window.tag, early)
            timing = invocation(root, plan, args.mode, [window], f"reference/{window.tag}/timing", True)
            references[window.tag] = dict(qualification=qualification, timing=timing)
            summary["references"][window.tag] = dict(status="passed")
        except FullBatchParityError:
            # This is not a failed singleton. It blocks interpretation of the
            # entire full-batch experiment until the cause has been resolved.
            raise
        except Exception as error:
            summary["references"][window.tag] = dict(status="failed", error=str(error))
        write_summary(root, summary)
    for size in plan["execution_order"]:
        selected = windows[:size]
        print(f"COORDINATE {args.mode} B={size}", flush=True)
        result = summary["coordinates"][str(size)]
        result["status"] = "running"
        write_summary(root, summary)
        try:
            missing = [w.tag for w in selected if w.tag not in references]
            if missing:
                raise ValueError(f"Singleton reference failed: {missing}")
            if size == 1:
                timing = references[selected[0].tag]["timing"]
            else:
                batch = invocation(root, plan, args.mode, selected, f"batch-{size:02d}/gate", False)
                checks = {w.tag: parity(references[w.tag]["qualification"]["outputs"][w.tag], batch["outputs"][w.tag], plan["qualification"]) for w in selected}
                W.write_json_atomic(root / f"batch-{size:02d}/parity.json", checks)
                if not all(x["passed"] for x in checks.values()):
                    raise ValueError("Single/batch parity failed; no timed dynamics admitted")
                timing = invocation(root, plan, args.mode, selected, f"batch-{size:02d}/timing", True)
            samples = timing["measurement"]["samples"]
            rate = [x["aggregate_window_steps_per_second"] for x in samples]
            serial_seconds = sum(sum(s["synchronized_loop_seconds"] for s in references[w.tag]["timing"]["measurement"]["samples"]) for w in selected)
            result.update(status="passed", median_steps_per_s_per_gpu=statistics.median(rate),
                min_steps_per_s_per_gpu=min(rate), max_steps_per_s_per_gpu=max(rate), aggregate_ns_per_day=statistics.median(rate)*.0864,
                matched_singleton_speedup=serial_seconds/sum(s["synchronized_loop_seconds"] for s in samples),
                windows=[w.tag for w in selected])
        except Exception as error:
            result.update(status="failed", error=str(error))
        write_summary(root, summary)
        print(json.dumps(dict(batch_size=size, **result)), flush=True)
    summary["status"] = "completed" if all(x["status"] == "passed" for x in summary["coordinates"].values()) else "completed-with-failures"
    for name, record in plan["runtime"].items():
        checked(record)
    write_summary(root, summary)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rank":
        # Hydra's local ranks share the srun cpuset. Bind one rank per granted
        # CPU without changing CUDA_VISIBLE_DEVICES or creating extra contexts.
        directory, executable = Path(sys.argv[2]), sys.argv[3]
        rank = int(os.environ["PMI_RANK"])
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[rank]})
        W.write_json_atomic(directory / f"rank-{rank}.json", dict(rank=rank, affinity=sorted(os.sched_getaffinity(0)),
                                                               cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]))
        os.execv(executable, [executable, *sys.argv[4:]])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--campaign", type=Path, required=True)
    prep.add_argument("--tutorial", type=Path, required=True)
    for name in ("lammps", "plugin", "xtbloom", "model", "mpiexec"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--mode", choices=("qmmm", "qmmm-dpa4c"), required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
