#include "xtbloom_lammps_adapter.h"

#include "point_charge_slots.h"
#include "xtb_legacy_hardness.h"
#include "xtbloom_partition_broker.h"

#include "lammps.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef DPRC_XTBLOOM_BACKEND_VALUE
#define DPRC_XTBLOOM_BACKEND_VALUE 0
#endif
#ifndef DPRC_XTBLOOM_DEVICE_ID
#define DPRC_XTBLOOM_DEVICE_ID -1
#endif
#ifndef DPRC_XTBLOOM_CPU_THREADS
#define DPRC_XTBLOOM_CPU_THREADS 0
#endif

namespace DPRC {
namespace {

// CODATA conversion used by xTBloom's public ABI. The LAMMPS command retains
// the upstream xTB convention of accepting electronic temperature in kelvin.
constexpr double kKelvinToHartree = 3.166808578545117e-6;

struct AdapterState {
  LAMMPS_NS::LAMMPS *lmp = nullptr;
  MPI_Comm roots = MPI_COMM_NULL;
  int stable_slot = -1;
  bool bound = false;
  bool active = false;
  int method = 0;
  WindowTopology local_topology;
  XtbloomExecutorOptions options;
  PointChargeSlots point_slots;
  std::vector<double> candidate_point_charge_gammas;
  std::vector<int> request_records;
  WindowFrame frame;
  std::unique_ptr<XtbloomPartitionBroker> broker;
  std::string last_error;
  bool profile_enabled = false;
  std::uint64_t batch_calls = 0;
  std::uint64_t plan_rebuilds = 0;
  std::uint64_t capacity_growths = 0;
  std::uint64_t fresh_calls = 0;
  std::uint64_t warm_calls = 0;
  std::uint64_t system_calls = 0;
  std::uint64_t total_scc_iterations = 0;
  std::uint64_t max_scc_iterations = 0;
  std::uint64_t total_actual_points = 0;
  std::uint64_t total_plan_points = 0;
  std::uint64_t max_actual_points = 0;
  std::uint64_t max_plan_points = 0;
};

AdapterState state;

bool collectively_all(MPI_Comm communicator, bool local_value) noexcept {
  const int local = local_value ? 1 : 0;
  int global = 0;
  return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      global != 0;
}

bool collective_request_state(MPI_Comm communicator, bool local_valid,
                              bool local_topology_changed,
                              std::vector<int> &records, bool &all_valid,
                              bool &any_topology_changed) noexcept {
  const int local[2] = {local_valid ? 1 : 0,
                        local_topology_changed ? 1 : 0};
  if (records.empty() || records.size() % 2u != 0u ||
      MPI_Allgather(local, 2, MPI_INT, records.data(), 2, MPI_INT,
                    communicator) != MPI_SUCCESS) {
    return false;
  }
  all_valid = true;
  any_topology_changed = false;
  for (std::size_t offset = 0; offset < records.size(); offset += 2u) {
    all_valid = all_valid && records[offset] != 0;
    any_topology_changed =
        any_topology_changed || records[offset + 1u] != 0;
  }
  return true;
}

void clear_calculator_state(bool clear_binding) noexcept {
  state.broker.reset();
  state.point_slots.clear();
  state.candidate_point_charge_gammas.clear();
  state.frame = {};
  state.local_topology = {};
  state.options = {};
  state.method = 0;
  state.active = false;
  state.last_error.clear();
  if (clear_binding) {
    state.request_records.clear();
    state.lmp = nullptr;
    state.roots = MPI_COMM_NULL;
    state.stable_slot = -1;
    state.bound = false;
    state.profile_enabled = false;
    state.batch_calls = 0;
    state.plan_rebuilds = 0;
    state.capacity_growths = 0;
    state.fresh_calls = 0;
    state.warm_calls = 0;
    state.system_calls = 0;
    state.total_scc_iterations = 0;
    state.max_scc_iterations = 0;
    state.total_actual_points = 0;
    state.total_plan_points = 0;
    state.max_actual_points = 0;
    state.max_plan_points = 0;
  }
}

bool same_options(const XtbloomExecutorOptions &lhs,
                  const XtbloomExecutorOptions &rhs) noexcept {
  return lhs.backend == rhs.backend && lhs.device_id == rhs.device_id &&
      lhs.cpu_threads == rhs.cpu_threads && lhs.model == rhs.model &&
      lhs.compute_flags == rhs.compute_flags &&
      lhs.max_scc_iterations == rhs.max_scc_iterations &&
      lhs.charge_tolerance == rhs.charge_tolerance &&
      lhs.energy_tolerance == rhs.energy_tolerance &&
      lhs.electronic_temperature == rhs.electronic_temperature &&
      lhs.scc_mixer == rhs.scc_mixer &&
      lhs.scc_mixer_history == rhs.scc_mixer_history &&
      lhs.scc_mixer_damping == rhs.scc_mixer_damping &&
      lhs.determinism == rhs.determinism;
}

bool same_static_topology(const WindowTopology &lhs,
                          const WindowTopology &rhs) noexcept {
  return lhs.window_index == rhs.window_index &&
      lhs.atomic_numbers == rhs.atomic_numbers &&
      lhs.molecular_charge == rhs.molecular_charge &&
      lhs.unpaired_electrons == rhs.unpaired_electrons &&
      lhs.spin_channels == rhs.spin_channels &&
      lhs.charge_response_enabled == rhs.charge_response_enabled;
}

void resize_frame(WindowFrame &frame, int stable_slot, std::size_t atoms,
                  std::size_t points) {
  if (atoms > std::numeric_limits<std::size_t>::max() / atoms)
    throw std::overflow_error("QM charge-response matrix is too large");
  frame.window_index = stable_slot;
  frame.positions.resize(3u * atoms);
  frame.point_charge_positions.resize(3u * points);
  frame.point_charge_values.resize(points);
  frame.atomic_potential_shifts.resize(atoms);
  frame.charge_response_matrix.resize(atoms * atoms);
}

bool pointer_request_is_valid(
    int nqm, const double *qm_xyz_bohr, int npoint,
    const double *point_xyz_bohr, const double *point_charge,
    const int *point_atomic_numbers, const double *mm_shift_hartree,
    const double *image_response_hartree, double *energy_hartree,
    double *qm_gradient_hartree_bohr, double *mulliken_charge,
    double *point_gradient_hartree_bohr) noexcept {
  return nqm > 0 && npoint >= 0 && qm_xyz_bohr != nullptr &&
      mm_shift_hartree != nullptr && image_response_hartree != nullptr &&
      energy_hartree != nullptr && qm_gradient_hartree_bohr != nullptr &&
      mulliken_charge != nullptr &&
      (npoint == 0 ||
       (point_xyz_bohr != nullptr && point_charge != nullptr &&
        point_atomic_numbers != nullptr &&
        point_gradient_hartree_bohr != nullptr));
}

} // namespace

int bind_lammps_xtbloom_adapter(LAMMPS_NS::LAMMPS *lmp, MPI_Comm roots,
                               int stable_slot) noexcept {
  if (lmp == nullptr || roots == MPI_COMM_NULL || stable_slot < 0)
    return 1;
  if (state.bound &&
      (state.lmp != lmp || state.roots != roots ||
       state.stable_slot != stable_slot)) {
    return 1;
  }
  state.lmp = lmp;
  state.roots = roots;
  state.stable_slot = stable_slot;
  int roots_size = 0;
  if (MPI_Comm_size(roots, &roots_size) != MPI_SUCCESS || roots_size <= 0)
    return 1;
  try {
    state.request_records.resize(2u * static_cast<std::size_t>(roots_size));
  } catch (...) {
    return 1;
  }
  // Profiling is a collective property so destruction never enters an MPI
  // reduction on only a subset of the GPU-local roots.
  state.profile_enabled = collectively_all(
      roots, std::getenv("LAMMPS_QMMM_XTB_PROFILE") != nullptr);
  state.bound = true;
  return 0;
}

} // namespace DPRC

extern "C" int dprc_lammps_xtb_create(
    int nqm, const int *atomic_numbers, const double *qm_xyz_bohr, int method,
    int charge, int uhf, double accuracy, int maxiter,
    double electronic_temperature_kelvin) {
  using namespace DPRC;

  if (!state.bound)
    return 1;

  WindowTopology topology;
  XtbloomExecutorOptions options;
  bool local_valid = nqm > 0 && atomic_numbers != nullptr &&
      qm_xyz_bohr != nullptr && (method == 1 || method == 2) && uhf >= 0 &&
      std::isfinite(accuracy) && accuracy > 0.0 && maxiter > 0 &&
      std::isfinite(electronic_temperature_kelvin) &&
      electronic_temperature_kelvin >= 0.0;
  try {
    if (local_valid) {
      topology.window_index = state.stable_slot;
      topology.atomic_numbers.reserve(static_cast<std::size_t>(nqm));
      for (int atom = 0; atom < nqm; ++atom) {
        if (atomic_numbers[atom] <= 0) {
          local_valid = false;
          break;
        }
        topology.atomic_numbers.push_back(atomic_numbers[atom]);
      }
      topology.molecular_charge = static_cast<double>(charge);
      topology.unpaired_electrons = uhf;
      topology.spin_channels = uhf == 0 ? 1 : 2;
      topology.charge_response_enabled = true;

      options.backend = DPRC_XTBLOOM_BACKEND_VALUE;
      options.device_id = DPRC_XTBLOOM_DEVICE_ID;
      options.cpu_threads = DPRC_XTBLOOM_CPU_THREADS;
      options.model = method == 1 ? XTBLOOM_MODEL_GFN1_XTB
                                  : XTBLOOM_MODEL_GFN2_XTB;
      options.max_scc_iterations = maxiter;
      // These are the exact convergence multipliers used by xTB's SCC driver.
      options.energy_tolerance = 1.0e-6 * accuracy;
      options.charge_tolerance =
          (method == 1 ? 2.0e-5 : 1.0e-4) * accuracy;
      options.electronic_temperature =
          electronic_temperature_kelvin * kKelvinToHartree;
    }
  } catch (...) {
    local_valid = false;
  }

  if (!collectively_all(state.roots, local_valid))
    return 1;

  const bool local_reusable = state.active && state.method == method &&
      same_static_topology(state.local_topology, topology) &&
      same_options(state.options, options);
  if (collectively_all(state.roots, local_reusable)) {
    // LAMMPS calls Fix::init() again for a subsequent run command. An
    // unchanged fixed topology/policy can keep both the plan allocation and
    // its strict whole-batch WARM checkpoint.
    state.last_error.clear();
    return 0;
  }

  // Reinitialization is collective because an existing broker owns a
  // duplicated root communicator. Preserve the binding supplied by the fix.
  clear_calculator_state(false);
  state.method = method;
  state.local_topology = std::move(topology);
  state.options = options;
  state.active = true;
  return 0;
}

extern "C" int dprc_lammps_xtb_calculate(
    int nqm, const double *qm_xyz_bohr, int npoint,
    const double *point_xyz_bohr, const double *point_charge,
    const int *point_atomic_numbers, double mm_hardness,
    const double *mm_shift_hartree,
    const double *image_response_hartree, double *energy_hartree,
    double *qm_gradient_hartree_bohr, double *mulliken_charge,
    double *point_gradient_hartree_bohr) {
  using namespace DPRC;

  if (!state.bound || !state.active)
    return 1;
  bool local_valid = pointer_request_is_valid(
      nqm, qm_xyz_bohr, npoint, point_xyz_bohr, point_charge,
      point_atomic_numbers, mm_shift_hartree, image_response_hartree,
      energy_hartree, qm_gradient_hartree_bohr, mulliken_charge,
      point_gradient_hartree_bohr);
  local_valid = local_valid &&
      static_cast<std::size_t>(nqm) == state.local_topology.atomic_numbers.size();

  try {
    if (local_valid) {
      fill_legacy_point_charge_gammas(
          state.method, mm_hardness, point_atomic_numbers,
          static_cast<std::size_t>(npoint),
          state.candidate_point_charge_gammas);
    }
  } catch (const std::exception &exception) {
    state.last_error = exception.what();
    local_valid = false;
  } catch (...) {
    state.last_error = "unknown MM-hardness conversion failure";
    local_valid = false;
  }
  bool local_slots_fit = false;
  try {
    local_slots_fit = local_valid &&
        state.point_slots.assign(state.candidate_point_charge_gammas);
  } catch (const std::exception &exception) {
    state.last_error = exception.what();
    local_valid = false;
  } catch (...) {
    state.last_error = "unknown point-charge slot assignment failure";
    local_valid = false;
  }
  const bool local_topology_changed =
      local_valid && (state.broker == nullptr || !local_slots_fit);
  bool all_valid = false;
  bool topology_changed = false;
  if (!collective_request_state(state.roots, local_valid,
                                local_topology_changed,
                                state.request_records, all_valid,
                                topology_changed) ||
      !all_valid) {
    return 1;
  }

  try {
    if (topology_changed) {
      const bool growing_existing_plan = state.broker != nullptr;
      state.point_slots.grow(state.candidate_point_charge_gammas);
      const std::size_t plan_points = state.point_slots.capacity();
      if (plan_points >
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 3)) {
        throw std::overflow_error(
            "padded point-charge topology exceeds broker MPI limits");
      }
      WindowTopology next_topology = state.local_topology;
      next_topology.point_charge_gammas =
          state.point_slots.topology_gammas();
      WindowFrame next_frame;
      resize_frame(next_frame, state.stable_slot,
                   static_cast<std::size_t>(nqm), plan_points);

      // Every root takes this branch together. Ordinary neighbor-epoch
      // membership changes fit zero-charge padding and stay WARM; only a real
      // gamma-class capacity overflow invalidates the whole native checkpoint.
      state.broker.reset();
      state.broker = std::make_unique<XtbloomPartitionBroker>(
          state.roots, std::move(next_topology), state.options);
      ++state.plan_rebuilds;
      if (growing_existing_plan)
        ++state.capacity_growths;
      state.frame = std::move(next_frame);
    }

    const std::size_t atoms = static_cast<std::size_t>(nqm);
    const std::size_t points = static_cast<std::size_t>(npoint);
    const std::size_t plan_points = state.point_slots.capacity();
    const std::vector<std::size_t> &point_slots =
        state.point_slots.assignments();
    if (point_slots.size() != points)
      throw std::logic_error("point-charge slot assignment extent changed");
    WindowFrame &frame = state.frame;
    frame.timestep = static_cast<std::int64_t>(state.lmp->update->ntimestep);
    std::copy_n(qm_xyz_bohr, 3u * atoms, frame.positions.data());
    // Unused permanent slots retain any finite previous position but carry an
    // exact zero charge. Every explicit xTB point term and point force is
    // linear in that charge, so padding cannot change the physical result.
    std::fill(frame.point_charge_values.begin(),
              frame.point_charge_values.end(), 0.0);
    if (points != 0u) {
      for (std::size_t point = 0; point < points; ++point) {
        const std::size_t slot = point_slots[point];
        std::copy_n(point_xyz_bohr + 3u * point, 3u,
                    frame.point_charge_positions.data() + 3u * slot);
        frame.point_charge_values[slot] = point_charge[point];
      }
    }
    std::copy_n(mm_shift_hartree, atoms,
                frame.atomic_potential_shifts.data());
    std::copy_n(image_response_hartree, atoms * atoms,
                frame.charge_response_matrix.data());

    const PartitionBrokerOutcome outcome = state.broker->compute(frame);
    if (outcome.call_status != XTBLOOM_STATUS_SUCCESS) {
      state.last_error = state.broker->last_error();
      return 1;
    }
    if ((outcome.result_flags &
         XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES) == 0u) {
      state.last_error =
          "xTBloom did not report the required external-operator force semantics";
      return 1;
    }
    if (!outcome.all_systems_succeeded) {
      state.last_error =
          "one or more xTBloom QM/MM systems did not converge";
      return 1;
    }

    const WindowResultView result = state.broker->result_for_local_window();
    const bool local_result_usable =
        result.status == XTBLOOM_STATUS_SUCCESS && result.scc_converged &&
        result.atom_count == atoms &&
        result.point_charge_count == plan_points;
    if (!local_result_usable) {
      state.last_error =
          "xTBloom broker published an inconsistent local QM/MM result";
      return 1;
    }

    ++state.batch_calls;
    if (outcome.start_policy == SccStartPolicy::Fresh)
      ++state.fresh_calls;
    else
      ++state.warm_calls;
    ++state.system_calls;
    state.total_actual_points += static_cast<std::uint64_t>(points);
    state.total_plan_points += static_cast<std::uint64_t>(plan_points);
    state.max_actual_points =
        std::max(state.max_actual_points, static_cast<std::uint64_t>(points));
    state.max_plan_points = std::max(
        state.max_plan_points, static_cast<std::uint64_t>(plan_points));
    const std::uint64_t scc_iterations =
        static_cast<std::uint64_t>(std::max(result.scc_iterations, 0));
    state.total_scc_iterations += scc_iterations;
    state.max_scc_iterations =
        std::max(state.max_scc_iterations, scc_iterations);

    *energy_hartree = result.energy;
    std::copy_n(result.atomic_charges, atoms, mulliken_charge);
    // The pinned LAMMPS fix consumes gradients and negates them during force
    // publication; xTBloom publishes forces directly.
    for (std::size_t coordinate = 0; coordinate < 3u * atoms; ++coordinate)
      qm_gradient_hartree_bohr[coordinate] = -result.forces[coordinate];
    for (std::size_t point = 0; point < points; ++point) {
      const std::size_t slot = point_slots[point];
      for (std::size_t dimension = 0; dimension < 3u; ++dimension) {
        point_gradient_hartree_bohr[3u * point + dimension] =
            -result.point_charge_forces[3u * slot + dimension];
      }
    }
    state.last_error.clear();
    return 0;
  } catch (const std::exception &exception) {
    state.last_error = exception.what();
  } catch (...) {
    state.last_error = "unknown xTBloom LAMMPS adapter failure";
  }
  return 1;
}

extern "C" const char *dprc_lammps_xtb_last_error() {
  using namespace DPRC;
  return state.last_error.empty() ? nullptr : state.last_error.c_str();
}

extern "C" void dprc_lammps_xtb_destroy() {
  using namespace DPRC;
  if (!state.bound)
    return;
  if (state.profile_enabled) {
    const std::uint64_t local_maxima[] = {
        state.batch_calls,       state.plan_rebuilds,
        state.fresh_calls,       state.warm_calls,
        state.max_scc_iterations, state.capacity_growths,
        state.max_actual_points, state.max_plan_points};
    std::uint64_t global_maxima[8] = {};
    const std::uint64_t local_sums[] = {
        state.system_calls, state.total_scc_iterations,
        state.total_actual_points, state.total_plan_points};
    std::uint64_t global_sums[4] = {};
    int rank = -1;
    const bool reduced =
        MPI_Comm_rank(state.roots, &rank) == MPI_SUCCESS &&
        MPI_Reduce(local_maxima, global_maxima, 8, MPI_UINT64_T, MPI_MAX, 0,
                   state.roots) == MPI_SUCCESS &&
        MPI_Reduce(local_sums, global_sums, 4, MPI_UINT64_T, MPI_SUM, 0,
                   state.roots) == MPI_SUCCESS;
    if (reduced && rank == 0) {
      const double mean_iterations = global_sums[0] == 0
          ? 0.0
          : static_cast<double>(global_sums[1]) /
              static_cast<double>(global_sums[0]);
      const double mean_actual_points = global_sums[0] == 0
          ? 0.0
          : static_cast<double>(global_sums[2]) /
              static_cast<double>(global_sums[0]);
      const double mean_plan_points = global_sums[0] == 0
          ? 0.0
          : static_cast<double>(global_sums[3]) /
              static_cast<double>(global_sums[0]);
      std::fprintf(
          stderr,
          "dprc xTB broker profile: batch_calls=%llu plan_rebuilds=%llu "
          "fresh=%llu warm=%llu system_calls=%llu mean_scc_iterations=%.3f "
          "max_scc_iterations=%llu capacity_growths=%llu "
          "mean_actual_points=%.3f mean_plan_points=%.3f "
          "max_actual_points=%llu max_plan_points=%llu\n",
          static_cast<unsigned long long>(global_maxima[0]),
          static_cast<unsigned long long>(global_maxima[1]),
          static_cast<unsigned long long>(global_maxima[2]),
          static_cast<unsigned long long>(global_maxima[3]),
          static_cast<unsigned long long>(global_sums[0]), mean_iterations,
          static_cast<unsigned long long>(global_maxima[4]),
          static_cast<unsigned long long>(global_maxima[5]),
          mean_actual_points, mean_plan_points,
          static_cast<unsigned long long>(global_maxima[6]),
          static_cast<unsigned long long>(global_maxima[7]));
    }
  }
  clear_calculator_state(true);
}

#undef DPRC_XTBLOOM_BACKEND_VALUE
#undef DPRC_XTBLOOM_DEVICE_ID
#undef DPRC_XTBLOOM_CPU_THREADS
