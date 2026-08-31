#ifndef LAMMPS_DPRC_XTBLOOM_PLAN_EXECUTOR_H
#define LAMMPS_DPRC_XTBLOOM_PLAN_EXECUTOR_H

#include "stable_batch.h"

#include <xtbloom/xtbloom.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace DPRC {

inline constexpr std::uint32_t kDprcQmmmComputeFlags =
    XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
    XTBLOOM_COMPUTE_ATOMIC_CHARGES | XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;

struct XtbloomExecutorOptions {
  xtbloom_backend_t backend = XTBLOOM_BACKEND_CPU;
  std::int32_t device_id = -1;
  std::int32_t cpu_threads = 0;
  xtbloom_model_t model = XTBLOOM_MODEL_GFN2_XTB;
  std::uint32_t compute_flags = kDprcQmmmComputeFlags;
  std::int32_t max_scc_iterations = 250;
  double charge_tolerance = 1.0e-6;
  double energy_tolerance = 1.0e-8;
  // xTBloom's native ABI accepts k_B*T in Hartree, not kelvin.
  double electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  xtbloom_scc_mixer_t scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  std::int32_t scc_mixer_history = 8;
  double scc_mixer_damping = 0.4;
  xtbloom_determinism_t determinism = XTBLOOM_DETERMINISM_DEFAULT;
};

struct XtbloomComputeOutcome {
  xtbloom_status_t call_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::int64_t timestep = -1;
  SccStartPolicy start_policy = SccStartPolicy::Fresh;
  std::uint32_t result_flags = 0;
  // A SUCCESS call may contain per-system SCC/eigensolver failures.  Such a
  // batch is deliberately not publishable to callers, because all windows
  // must advance from one consistent xTBloom checkpoint.
  bool all_systems_succeeded = false;
};

struct WindowResultView {
  std::int32_t window_index = -1;
  std::int64_t timestep = -1;
  xtbloom_status_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::int32_t scc_iterations = 0;
  bool scc_converged = false;
  double energy = 0.0;
  const double *forces = nullptr;
  const double *atomic_charges = nullptr;
  const double *point_charge_forces = nullptr;
  std::size_t atom_count = 0;
  std::size_t point_charge_count = 0;
};

struct XtbloomWorkspaceInfo {
  std::uint64_t host_bytes = 0;
  std::uint32_t host_alignment = 1;
  std::uint64_t device_bytes = 0;
  std::uint32_t device_alignment = 1;
};

// Public-C-ABI-only owner for one xTBloom context and fixed ragged plan. The
// class deliberately knows nothing about LAMMPS objects so its serial/batch
// scientific contract can be qualified before force publication is wired in.
class XtbloomPlanExecutor {
public:
  XtbloomPlanExecutor(std::vector<WindowTopology> topologies,
                      const XtbloomExecutorOptions &options = {});
  ~XtbloomPlanExecutor();

  XtbloomPlanExecutor(const XtbloomPlanExecutor &) = delete;
  XtbloomPlanExecutor &operator=(const XtbloomPlanExecutor &) = delete;

  const StableBatch &batch() const noexcept { return batch_; }
  void stage(const WindowFrame &frame);
  void stage(const WindowFrameView &frame);

  XtbloomComputeOutcome compute();
  WindowResultView result_for_window(std::int32_t window_index) const;
  XtbloomWorkspaceInfo workspace() const;

  // Invalidate a successfully computed native result when a downstream
  // publication validator rejects its shape or ownership.  This also forces
  // the next compute to use FRESH SCC state.
  void invalidate_result() noexcept {
    result_valid_ = false;
    batch_.invalidate_warm_checkpoint();
  }

  bool has_result() const noexcept { return result_valid_; }
  const std::string &last_error() const noexcept { return last_error_; }
  xtbloom_backend_t backend() const noexcept;
  std::int32_t device_id() const noexcept;

private:
  void bind_descriptors();

  StableBatch batch_;
  XtbloomExecutorOptions executor_options_;
  xtbloom_context_t *context_ = nullptr;
  xtbloom_plan_t *plan_ = nullptr;
  xtbloom_batch_t descriptor_{};
  xtbloom_compute_options_t compute_options_{};
  xtbloom_batch_result_t result_{};

  std::vector<double> energies_;
  std::vector<double> forces_;
  std::vector<double> atomic_charges_;
  std::vector<double> point_charge_forces_;
  std::vector<std::int32_t> scc_iterations_;
  std::vector<std::uint8_t> scc_converged_;
  std::vector<std::int32_t> per_system_status_;

  std::int64_t result_timestep_ = -1;
  bool result_valid_ = false;
  std::string last_error_;
};

} // namespace DPRC

#endif
