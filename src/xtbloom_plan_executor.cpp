#include "xtbloom_plan_executor.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace DPRC {
namespace {

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T> &values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T),
          XTBLOOM_MEMORY_HOST, 0};
}

template <typename T> xtbloom_buffer_t output_buffer(std::vector<T> &values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T),
          XTBLOOM_MEMORY_HOST, 0};
}

std::runtime_error api_error(const char *operation) {
  const char *diagnostic = xtbloom_get_last_error();
  return std::runtime_error(
      std::string(operation) + ": " +
      (diagnostic == nullptr ? "unknown xTBloom error" : diagnostic));
}

std::size_t range_size(std::int64_t begin, std::int64_t end) {
  if (begin < 0 || end < begin ||
      static_cast<std::uint64_t>(end - begin) >
          std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("xTBloom result range does not fit size_t");
  }
  return static_cast<std::size_t>(end - begin);
}

const char *status_name(xtbloom_status_t status) {
  const char *name = xtbloom_status_string(status);
  return name == nullptr ? "UNKNOWN_STATUS" : name;
}

std::string per_system_failure_diagnostic(const StableBatch &batch,
                                         const std::vector<std::int32_t> &statuses,
                                         const std::vector<std::uint8_t> &converged,
                                         const std::vector<std::int32_t> &iterations) {
  std::ostringstream message;
  message << "xTBloom batch rejected because one or more systems failed";
  for (std::size_t slot = 0; slot < batch.size(); ++slot) {
    const auto status = static_cast<xtbloom_status_t>(statuses[slot]);
    if (status == XTBLOOM_STATUS_SUCCESS && converged[slot] != 0u)
      continue;
    message << "; window " << batch.topology(slot).window_index
            << " status=" << status_name(status) << " ("
            << statuses[slot] << ")"
            << " scc_converged=" << (converged[slot] != 0u ? "true" : "false")
            << " scc_iterations=" << iterations[slot];
  }
  return message.str();
}

} // namespace

XtbloomPlanExecutor::XtbloomPlanExecutor(std::vector<WindowTopology> topologies,
                                         const XtbloomExecutorOptions &options)
    : batch_(std::move(topologies)), executor_options_(options) {
  if (executor_options_.compute_flags != kDprcQmmmComputeFlags) {
    throw std::invalid_argument(
        "DPRc xTBloom plans require energy, QM forces, atomic charges, and "
        "point-charge forces as one fixed publication policy");
  }
  xtbloom_context_options_t context_options{};
  if (xtbloom_context_options_init(&context_options, sizeof(context_options)) !=
      XTBLOOM_STATUS_SUCCESS) {
    throw api_error("xtbloom_context_options_init failed");
  }
  context_options.backend = executor_options_.backend;
  context_options.device_id = executor_options_.device_id;
  context_options.cpu_threads = executor_options_.cpu_threads;
  if (xtbloom_context_create(&context_options, &context_) !=
      XTBLOOM_STATUS_SUCCESS) {
    context_ = nullptr;
    throw api_error("xtbloom_context_create failed");
  }

  try {
    if (xtbloom_batch_init(&descriptor_, sizeof(descriptor_)) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom_compute_options_init(&compute_options_,
                                     sizeof(compute_options_)) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom_batch_result_init(&result_, sizeof(result_)) !=
            XTBLOOM_STATUS_SUCCESS) {
      throw api_error("xTBloom descriptor initialization failed");
    }
    compute_options_.model = executor_options_.model;
    compute_options_.flags = executor_options_.compute_flags;
    compute_options_.max_scc_iterations = executor_options_.max_scc_iterations;
    compute_options_.charge_tolerance = executor_options_.charge_tolerance;
    compute_options_.energy_tolerance = executor_options_.energy_tolerance;
    compute_options_.electronic_temperature =
        executor_options_.electronic_temperature;
    compute_options_.scc_mixer = executor_options_.scc_mixer;
    compute_options_.scc_mixer_history = executor_options_.scc_mixer_history;
    compute_options_.scc_mixer_damping = executor_options_.scc_mixer_damping;
    compute_options_.determinism = executor_options_.determinism;

    const std::size_t systems = batch_.size();
    const std::size_t atoms = batch_.atomic_numbers().size();
    energies_.resize(systems);
    forces_.resize(batch_.positions().size());
    atomic_charges_.resize(atoms);
    point_charge_forces_.resize(batch_.point_charge_positions().size());
    scc_iterations_.resize(systems);
    scc_converged_.resize(systems);
    per_system_status_.resize(systems);
    bind_descriptors();

    if (xtbloom_plan_create(context_, &descriptor_, &compute_options_,
                            &plan_) != XTBLOOM_STATUS_SUCCESS) {
      plan_ = nullptr;
      throw api_error("xtbloom_plan_create failed");
    }
  } catch (...) {
    xtbloom_plan_destroy(plan_);
    plan_ = nullptr;
    xtbloom_context_destroy(context_);
    context_ = nullptr;
    throw;
  }
}

XtbloomPlanExecutor::~XtbloomPlanExecutor() {
  xtbloom_plan_destroy(plan_);
  xtbloom_context_destroy(context_);
}

void XtbloomPlanExecutor::stage(const WindowFrame &frame) {
  batch_.stage(frame);
  // Once any next-step input is accepted, callers must not accidentally
  // publish a result from the preceding timestep.
  result_valid_ = false;
}

void XtbloomPlanExecutor::stage(const WindowFrameView &frame) {
  batch_.stage(frame);
  result_valid_ = false;
}

void XtbloomPlanExecutor::bind_descriptors() {
  descriptor_.batch_size = static_cast<std::int64_t>(batch_.size());
  descriptor_.total_atoms = batch_.atom_offsets().back();
  descriptor_.total_point_charges = batch_.point_charge_offsets().back();
  descriptor_.total_charge_response_elements =
      batch_.charge_response_offsets().empty()
          ? 0
          : batch_.charge_response_offsets().back();
  descriptor_.atom_offsets = input_buffer(batch_.atom_offsets());
  descriptor_.atomic_numbers = input_buffer(batch_.atomic_numbers());
  descriptor_.positions = input_buffer(batch_.positions());
  descriptor_.molecular_charges = input_buffer(batch_.molecular_charges());
  descriptor_.unpaired_electrons = input_buffer(batch_.unpaired_electrons());
  descriptor_.spin_channels = input_buffer(batch_.spin_channels());
  descriptor_.point_charge_offsets =
      input_buffer(batch_.point_charge_offsets());
  descriptor_.point_charge_positions =
      input_buffer(batch_.point_charge_positions());
  descriptor_.point_charge_values = input_buffer(batch_.point_charge_values());
  descriptor_.point_charge_gammas = input_buffer(batch_.point_charge_gammas());
  descriptor_.atomic_potential_shifts =
      input_buffer(batch_.atomic_potential_shifts());
  descriptor_.charge_response_offsets =
      input_buffer(batch_.charge_response_offsets());
  descriptor_.charge_response_matrix =
      input_buffer(batch_.charge_response_matrix());

  result_.energies = output_buffer(energies_);
  result_.forces = output_buffer(forces_);
  result_.atomic_charges = output_buffer(atomic_charges_);
  result_.point_charge_forces = output_buffer(point_charge_forces_);
  result_.scc_iterations = output_buffer(scc_iterations_);
  result_.scc_converged = output_buffer(scc_converged_);
  result_.per_system_status = output_buffer(per_system_status_);
}

XtbloomComputeOutcome XtbloomPlanExecutor::compute() {
  const std::int64_t timestep = batch_.timestep();
  const SccStartPolicy start_policy = batch_.prepare_compute();
  compute_options_.scc_start_mode = start_policy == SccStartPolicy::Warm
                                        ? XTBLOOM_SCC_START_WARM
                                        : XTBLOOM_SCC_START_FRESH;
  result_valid_ = false;
  last_error_.clear();
  bool all_systems_succeeded = false;
  const xtbloom_status_t status =
      xtbloom_plan_compute(plan_, &descriptor_, &compute_options_, &result_);
  if (status == XTBLOOM_STATUS_SUCCESS) {
    all_systems_succeeded = true;
    for (std::size_t slot = 0; slot < batch_.size(); ++slot) {
      all_systems_succeeded =
          all_systems_succeeded &&
          per_system_status_[slot] == XTBLOOM_STATUS_SUCCESS &&
          scc_converged_[slot] != 0u;
    }
    // A data-level failure is still a failed transaction for the strict WARM
    // contract.  Passing call_succeeded=false revokes the checkpoint while
    // retaining the native status arrays for the diagnostic below.
    if (all_systems_succeeded) {
      batch_.complete_compute(true, per_system_status_.data(),
                              per_system_status_.size(),
                              XTBLOOM_STATUS_SUCCESS);
    } else {
      batch_.complete_compute(false, nullptr, 0u, XTBLOOM_STATUS_SUCCESS);
      last_error_ = per_system_failure_diagnostic(
          batch_, per_system_status_, scc_converged_, scc_iterations_);
    }
    result_timestep_ = timestep;
    // Keep failed native buffers private.  In particular, xTBloom intentionally
    // fills failed slices with NaNs; exposing them to the broker would permit a
    // partial force publication and could make a later LAMMPS callback unsafe.
    result_valid_ = all_systems_succeeded;
  } else {
    const char *diagnostic = xtbloom_get_last_error();
    if (diagnostic != nullptr)
      last_error_ = diagnostic;
    batch_.complete_compute(false, nullptr, 0u, XTBLOOM_STATUS_SUCCESS);
  }
  return {status, timestep, start_policy,
          status == XTBLOOM_STATUS_SUCCESS ? result_.flags : 0u,
          all_systems_succeeded};
}

WindowResultView
XtbloomPlanExecutor::result_for_window(std::int32_t window_index) const {
  if (!result_valid_)
    throw std::logic_error("no completed xTBloom batch result is available");
  const std::size_t slot = batch_.slot_for_window(window_index);
  const SlotRange &slot_range = batch_.range(slot);
  const std::size_t atom_begin =
      static_cast<std::size_t>(slot_range.atom_begin);
  const std::size_t point_begin =
      static_cast<std::size_t>(slot_range.point_charge_begin);
  return {
      window_index,
      result_timestep_,
      static_cast<xtbloom_status_t>(per_system_status_[slot]),
      scc_iterations_[slot],
      scc_converged_[slot] != 0u,
      energies_[slot],
      forces_.data() + 3u * atom_begin,
      atomic_charges_.data() + atom_begin,
      point_charge_forces_.empty()
          ? nullptr
          : point_charge_forces_.data() + 3u * point_begin,
      range_size(slot_range.atom_begin, slot_range.atom_end),
      range_size(slot_range.point_charge_begin, slot_range.point_charge_end)};
}

XtbloomWorkspaceInfo XtbloomPlanExecutor::workspace() const {
  xtbloom_workspace_query_t query{};
  if (xtbloom_workspace_query_init(&query, sizeof(query)) !=
      XTBLOOM_STATUS_SUCCESS) {
    throw api_error("xtbloom_workspace_query_init failed");
  }
  query.compute_flags = executor_options_.compute_flags;
  if (xtbloom_plan_query_workspace(plan_, &query) != XTBLOOM_STATUS_SUCCESS)
    throw api_error("xtbloom_plan_query_workspace failed");
  return {query.host_required_bytes, query.host_required_alignment,
          query.device_required_bytes, query.device_required_alignment};
}

xtbloom_backend_t XtbloomPlanExecutor::backend() const noexcept {
  return xtbloom_context_get_backend(context_);
}

std::int32_t XtbloomPlanExecutor::device_id() const noexcept {
  return xtbloom_context_get_device_id(context_);
}

} // namespace DPRC
