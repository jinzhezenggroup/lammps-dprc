#include "xtbloom_partition_broker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace DPRC {
namespace {

constexpr int kBrokerRoot = 0;

void check_mpi(int status, const char *operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed");
}

int checked_scale(int value, int factor, const char *label) {
  if (value < 0 || factor < 0 ||
      (factor != 0 && value > std::numeric_limits<int>::max() / factor)) {
    throw std::overflow_error(std::string(label) + " exceeds MPI int count");
  }
  return value * factor;
}

int checked_square(int value, const char *label) {
  return checked_scale(value, value, label);
}

void build_displacements(const std::vector<int> &counts,
                         std::vector<int> &displacements, const char *label) {
  displacements.assign(counts.size(), 0);
  for (std::size_t index = 1; index < counts.size(); ++index) {
    if (counts[index - 1] < 0 ||
        displacements[index - 1] >
            std::numeric_limits<int>::max() - counts[index - 1]) {
      throw std::overflow_error(std::string(label) +
                                " displacements exceed MPI int count");
    }
    displacements[index] = displacements[index - 1] + counts[index - 1];
  }
}

std::size_t total_count(const std::vector<int> &counts,
                        const std::vector<int> &displacements) {
  if (counts.empty())
    return 0;
  return static_cast<std::size_t>(displacements.back()) +
         static_cast<std::size_t>(counts.back());
}

std::size_t checked_sum(std::size_t left, std::size_t right,
                        const char *label) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left + right;
}

bool topology_locally_valid(const WindowTopology &topology, int rank) {
  if (topology.window_index != rank || topology.atomic_numbers.empty() ||
      !std::isfinite(topology.molecular_charge) ||
      topology.unpaired_electrons < 0 ||
      (topology.spin_channels != 1 && topology.spin_channels != 2)) {
    return false;
  }
  if (topology.atomic_numbers.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 3) ||
      topology.point_charge_gammas.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 3)) {
    return false;
  }
  const std::size_t atoms = topology.atomic_numbers.size();
  if (topology.charge_response_enabled &&
      atoms >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) / atoms) {
    return false;
  }
  for (const std::int32_t atomic_number : topology.atomic_numbers)
    if (atomic_number <= 0)
      return false;
  for (const double gamma : topology.point_charge_gammas)
    if (!std::isfinite(gamma) || gamma <= 0.0)
      return false;
  return true;
}

void require_collectively(MPI_Comm communicator, bool local_condition,
                          const char *message) {
  const int local = local_condition ? 1 : 0;
  int global = 0;
  check_mpi(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator),
            "MPI_Allreduce");
  if (global == 0)
    throw std::invalid_argument(message);
}

void broadcast_string(MPI_Comm communicator, int rank, std::string &value) {
  std::uint64_t length =
      rank == kBrokerRoot ? static_cast<std::uint64_t>(value.size()) : 0u;
  check_mpi(MPI_Bcast(&length, 1, MPI_UINT64_T, kBrokerRoot, communicator),
            "MPI_Bcast diagnostic length");
  if (length >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      length > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("broker diagnostic is too large");
  }
  if (rank != kBrokerRoot)
    value.resize(static_cast<std::size_t>(length));
  check_mpi(MPI_Bcast(value.empty() ? nullptr : value.data(),
                      static_cast<int>(length), MPI_CHAR, kBrokerRoot,
                      communicator),
            "MPI_Bcast diagnostic");
}

DoubleArrayView slot_view(const double *values,
                          const std::vector<int> &counts,
                          const std::vector<int> &displacements,
                          std::size_t slot) {
  const int count = counts.at(slot);
  return {count == 0 ? nullptr : values + displacements.at(slot),
          static_cast<std::size_t>(count)};
}

} // namespace

XtbloomPartitionBroker::XtbloomPartitionBroker(
    MPI_Comm roots, WindowTopology local_topology,
    const XtbloomExecutorOptions &options)
    : local_topology_(std::move(local_topology)) {
  if (roots == MPI_COMM_NULL)
    throw std::invalid_argument(
        "xTBloom partition broker requires a root communicator");
  check_mpi(MPI_Comm_dup(roots, &communicator_), "MPI_Comm_dup");
  try {
    check_mpi(MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler");
    check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(communicator_, &size_), "MPI_Comm_size");
    initialize_collective(options);
  } catch (...) {
    release_shared_storage();
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
    throw;
  }
}

XtbloomPartitionBroker::~XtbloomPartitionBroker() {
  executor_.reset();
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized)
    return;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    release_shared_storage();
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
  }
}

void XtbloomPartitionBroker::initialize_collective(
    const XtbloomExecutorOptions &options) {
  require_collectively(communicator_, size_ > 0, "empty broker communicator");
  require_collectively(communicator_,
                       topology_locally_valid(local_topology_, rank_),
                       "invalid or non-dense broker window topology");

  const std::array<std::int64_t, 6> local_metadata = {
      local_topology_.window_index,
      static_cast<std::int64_t>(local_topology_.atomic_numbers.size()),
      static_cast<std::int64_t>(local_topology_.point_charge_gammas.size()),
      local_topology_.unpaired_electrons,
      local_topology_.spin_channels,
      local_topology_.charge_response_enabled ? 1 : 0};
  std::vector<std::int64_t> all_metadata(static_cast<std::size_t>(size_) *
                                         local_metadata.size());
  check_mpi(MPI_Allgather(local_metadata.data(), local_metadata.size(),
                          MPI_INT64_T, all_metadata.data(),
                          local_metadata.size(), MPI_INT64_T, communicator_),
            "MPI_Allgather topology metadata");

  std::vector<double> molecular_charges(static_cast<std::size_t>(size_));
  check_mpi(MPI_Allgather(&local_topology_.molecular_charge, 1, MPI_DOUBLE,
                          molecular_charges.data(), 1, MPI_DOUBLE,
                          communicator_),
            "MPI_Allgather molecular charges");

  const std::array<std::int64_t, 9> local_integer_options = {
      options.backend,
      options.device_id,
      options.cpu_threads,
      options.model,
      static_cast<std::int64_t>(options.compute_flags),
      options.max_scc_iterations,
      options.scc_mixer,
      options.scc_mixer_history,
      options.determinism};
  const std::array<double, 4> local_real_options = {
      options.charge_tolerance, options.energy_tolerance,
      options.electronic_temperature, options.scc_mixer_damping};
  std::vector<std::int64_t> all_integer_options(
      static_cast<std::size_t>(size_) * local_integer_options.size());
  std::vector<double> all_real_options(static_cast<std::size_t>(size_) *
                                       local_real_options.size());
  check_mpi(
      MPI_Allgather(local_integer_options.data(), local_integer_options.size(),
                    MPI_INT64_T, all_integer_options.data(),
                    local_integer_options.size(), MPI_INT64_T, communicator_),
      "MPI_Allgather integer compute options");
  check_mpi(MPI_Allgather(local_real_options.data(), local_real_options.size(),
                          MPI_DOUBLE, all_real_options.data(),
                          local_real_options.size(), MPI_DOUBLE, communicator_),
            "MPI_Allgather real compute options");
  bool matching_options = true;
  for (int peer = 1; peer < size_; ++peer) {
    matching_options =
        matching_options &&
        std::equal(all_integer_options.begin(),
                   all_integer_options.begin() + local_integer_options.size(),
                   all_integer_options.begin() +
                       static_cast<std::size_t>(peer) *
                           local_integer_options.size()) &&
        std::equal(all_real_options.begin(),
                   all_real_options.begin() + local_real_options.size(),
                   all_real_options.begin() + static_cast<std::size_t>(peer) *
                                                  local_real_options.size());
  }
  require_collectively(communicator_, matching_options,
                       "broker compute options must match on every root");

  atom_counts_.resize(size_);
  position_counts_.resize(size_);
  point_counts_.resize(size_);
  point_position_counts_.resize(size_);
  shift_counts_.resize(size_);
  response_counts_.resize(size_);
  for (int slot = 0; slot < size_; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * 6u;
    if (all_metadata[base] != slot || all_metadata[base + 1] <= 0 ||
        all_metadata[base + 1] > std::numeric_limits<int>::max() ||
        all_metadata[base + 2] < 0 ||
        all_metadata[base + 2] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("invalid gathered broker topology metadata");
    }
    atom_counts_[slot] = static_cast<int>(all_metadata[base + 1]);
    point_counts_[slot] = static_cast<int>(all_metadata[base + 2]);
    position_counts_[slot] =
        checked_scale(atom_counts_[slot], 3, "QM position");
    point_position_counts_[slot] =
        checked_scale(point_counts_[slot], 3, "point-charge position");
    const bool response = all_metadata[base + 5] != 0;
    shift_counts_[slot] = response ? atom_counts_[slot] : 0;
    response_counts_[slot] =
        response ? checked_square(atom_counts_[slot], "charge response") : 0;
  }
  build_displacements(atom_counts_, atom_displacements_, "atom");
  build_displacements(position_counts_, position_displacements_, "position");
  build_displacements(point_counts_, point_displacements_, "point charge");
  build_displacements(point_position_counts_, point_position_displacements_,
                      "point-charge position");
  build_displacements(shift_counts_, shift_displacements_, "potential shift");
  build_displacements(response_counts_, response_displacements_,
                      "charge response");

  std::vector<std::int32_t> gathered_atomic_numbers;
  std::vector<double> gathered_gammas;
  if (rank_ == kBrokerRoot) {
    gathered_atomic_numbers.resize(
        total_count(atom_counts_, atom_displacements_));
    gathered_gammas.resize(total_count(point_counts_, point_displacements_));
  }
  check_mpi(MPI_Gatherv(local_topology_.atomic_numbers.data(),
                        atom_counts_[rank_], MPI_INT32_T,
                        rank_ == kBrokerRoot ? gathered_atomic_numbers.data()
                                             : nullptr,
                        atom_counts_.data(), atom_displacements_.data(),
                        MPI_INT32_T, kBrokerRoot, communicator_),
            "MPI_Gatherv atomic numbers");
  check_mpi(MPI_Gatherv(local_topology_.point_charge_gammas.empty()
                            ? nullptr
                            : local_topology_.point_charge_gammas.data(),
                        point_counts_[rank_], MPI_DOUBLE,
                        rank_ == kBrokerRoot && !gathered_gammas.empty()
                            ? gathered_gammas.data()
                            : nullptr,
                        point_counts_.data(), point_displacements_.data(),
                        MPI_DOUBLE, kBrokerRoot, communicator_),
            "MPI_Gatherv point-charge gammas");

  initialize_shared_storage();
  preflight_records_.resize(2u * static_cast<std::size_t>(size_));

  int setup_failed = 0;
  if (rank_ == kBrokerRoot) {
    try {
      topologies_.resize(size_);
      for (int slot = 0; slot < size_; ++slot) {
        const std::size_t base = static_cast<std::size_t>(slot) * 6u;
        WindowTopology &topology = topologies_[slot];
        topology.window_index = slot;
        topology.atomic_numbers.assign(
            gathered_atomic_numbers.begin() + atom_displacements_[slot],
            gathered_atomic_numbers.begin() + atom_displacements_[slot] +
                atom_counts_[slot]);
        topology.molecular_charge = molecular_charges[slot];
        topology.unpaired_electrons =
            static_cast<std::int32_t>(all_metadata[base + 3]);
        topology.spin_channels =
            static_cast<std::int32_t>(all_metadata[base + 4]);
        topology.point_charge_gammas.assign(
            gathered_gammas.begin() + point_displacements_[slot],
            gathered_gammas.begin() + point_displacements_[slot] +
                point_counts_[slot]);
        topology.charge_response_enabled = all_metadata[base + 5] != 0;
      }

      executor_ = std::make_unique<XtbloomPlanExecutor>(topologies_, options);
    } catch (const std::exception &exception) {
      setup_failed = 1;
      last_error_ = exception.what();
    } catch (...) {
      setup_failed = 1;
      last_error_ = "unknown exception while creating xTBloom broker plan";
    }
  }
  check_mpi(MPI_Bcast(&setup_failed, 1, MPI_INT, kBrokerRoot, communicator_),
            "MPI_Bcast broker setup status");
  if (setup_failed != 0) {
    broadcast_string(communicator_, rank_, last_error_);
    throw std::runtime_error(last_error_);
  }
}

void XtbloomPartitionBroker::initialize_shared_storage() {
  static_assert(std::is_trivially_copyable_v<SharedResultMetadata>);

  check_mpi(MPI_Comm_split_type(communicator_, MPI_COMM_TYPE_SHARED, rank_,
                                MPI_INFO_NULL, &shared_communicator_),
            "MPI_Comm_split_type shared xTB roots");
  check_mpi(MPI_Comm_set_errhandler(shared_communicator_, MPI_ERRORS_RETURN),
            "MPI_Comm_set_errhandler shared xTB roots");
  int shared_rank = -1;
  int shared_size = 0;
  check_mpi(MPI_Comm_rank(shared_communicator_, &shared_rank),
            "MPI_Comm_rank shared xTB roots");
  check_mpi(MPI_Comm_size(shared_communicator_, &shared_size),
            "MPI_Comm_size shared xTB roots");
  require_collectively(
      communicator_, shared_rank == rank_ && shared_size == size_,
      "xTBloom GPU broker roots must occupy one shared-memory node");

  const std::size_t positions =
      total_count(position_counts_, position_displacements_);
  const std::size_t point_positions =
      total_count(point_position_counts_, point_position_displacements_);
  const std::size_t points =
      total_count(point_counts_, point_displacements_);
  const std::size_t shifts =
      total_count(shift_counts_, shift_displacements_);
  const std::size_t responses =
      total_count(response_counts_, response_displacements_);
  const std::size_t atoms = total_count(atom_counts_, atom_displacements_);

  std::size_t doubles = 0;
  auto reserve = [&](std::size_t count) {
    const std::size_t offset = doubles;
    doubles = checked_sum(doubles, count, "shared xTB storage");
    return offset;
  };
  const std::size_t positions_offset = reserve(positions);
  const std::size_t point_positions_offset = reserve(point_positions);
  const std::size_t point_values_offset = reserve(points);
  const std::size_t shifts_offset = reserve(shifts);
  const std::size_t responses_offset = reserve(responses);
  const std::size_t forces_offset = reserve(positions);
  const std::size_t charges_offset = reserve(atoms);
  const std::size_t point_forces_offset = reserve(point_positions);

  if (doubles > static_cast<std::size_t>(
                    std::numeric_limits<MPI_Aint>::max() / sizeof(double))) {
    throw std::overflow_error("shared xTB storage exceeds MPI_Aint");
  }
  const MPI_Aint data_bytes =
      static_cast<MPI_Aint>(doubles * sizeof(double));
  void *local_data = nullptr;
  check_mpi(MPI_Win_allocate_shared(rank_ == kBrokerRoot ? data_bytes : 0,
                                    sizeof(double), MPI_INFO_NULL,
                                    shared_communicator_, &local_data,
                                    &shared_data_window_),
            "MPI_Win_allocate_shared xTB data");
  MPI_Aint root_data_bytes = 0;
  int data_displacement = 0;
  void *root_data = nullptr;
  check_mpi(MPI_Win_shared_query(shared_data_window_, kBrokerRoot,
                                 &root_data_bytes, &data_displacement,
                                 &root_data),
            "MPI_Win_shared_query xTB data");
  require_collectively(
      communicator_, root_data != nullptr && root_data_bytes == data_bytes &&
                         data_displacement == static_cast<int>(sizeof(double)),
      "xTBloom shared data mapping is inconsistent");
  check_mpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, shared_data_window_),
            "MPI_Win_lock_all xTB data");
  shared_data_locked_ = true;
  shared_data_base_ = static_cast<double *>(root_data);
  shared_positions_ = shared_data_base_ + positions_offset;
  shared_point_positions_ = shared_data_base_ + point_positions_offset;
  shared_point_values_ = shared_data_base_ + point_values_offset;
  shared_potential_shifts_ = shared_data_base_ + shifts_offset;
  shared_response_matrices_ = shared_data_base_ + responses_offset;
  shared_forces_ = shared_data_base_ + forces_offset;
  shared_atomic_charges_ = shared_data_base_ + charges_offset;
  shared_point_charge_forces_ = shared_data_base_ + point_forces_offset;

  if (static_cast<std::size_t>(size_) >
      static_cast<std::size_t>(std::numeric_limits<MPI_Aint>::max()) /
          sizeof(SharedResultMetadata)) {
    throw std::overflow_error("shared xTB metadata exceeds MPI_Aint");
  }
  const MPI_Aint metadata_bytes = static_cast<MPI_Aint>(
      static_cast<std::size_t>(size_) * sizeof(SharedResultMetadata));
  void *local_metadata = nullptr;
  check_mpi(
      MPI_Win_allocate_shared(rank_ == kBrokerRoot ? metadata_bytes : 0, 1,
                              MPI_INFO_NULL, shared_communicator_,
                              &local_metadata, &shared_metadata_window_),
      "MPI_Win_allocate_shared xTB metadata");
  MPI_Aint root_metadata_bytes = 0;
  int metadata_displacement = 0;
  void *root_metadata = nullptr;
  check_mpi(MPI_Win_shared_query(shared_metadata_window_, kBrokerRoot,
                                 &root_metadata_bytes, &metadata_displacement,
                                 &root_metadata),
            "MPI_Win_shared_query xTB metadata");
  const bool metadata_aligned =
      reinterpret_cast<std::uintptr_t>(root_metadata) %
          alignof(SharedResultMetadata) ==
      0u;
  require_collectively(
      communicator_, root_metadata != nullptr &&
                         root_metadata_bytes == metadata_bytes &&
                         metadata_displacement == 1 && metadata_aligned,
      "xTBloom shared result metadata mapping is inconsistent");
  check_mpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, shared_metadata_window_),
            "MPI_Win_lock_all xTB metadata");
  shared_metadata_locked_ = true;
  shared_metadata_ = static_cast<SharedResultMetadata *>(root_metadata);
}

void XtbloomPartitionBroker::release_shared_storage() noexcept {
  if (shared_metadata_locked_ && shared_metadata_window_ != MPI_WIN_NULL)
    MPI_Win_unlock_all(shared_metadata_window_);
  shared_metadata_locked_ = false;
  if (shared_data_locked_ && shared_data_window_ != MPI_WIN_NULL)
    MPI_Win_unlock_all(shared_data_window_);
  shared_data_locked_ = false;
  if (shared_metadata_window_ != MPI_WIN_NULL)
    MPI_Win_free(&shared_metadata_window_);
  if (shared_data_window_ != MPI_WIN_NULL)
    MPI_Win_free(&shared_data_window_);
  if (shared_communicator_ != MPI_COMM_NULL)
    MPI_Comm_free(&shared_communicator_);
  shared_data_base_ = nullptr;
  shared_metadata_ = nullptr;
  shared_positions_ = nullptr;
  shared_point_positions_ = nullptr;
  shared_point_values_ = nullptr;
  shared_potential_shifts_ = nullptr;
  shared_response_matrices_ = nullptr;
  shared_forces_ = nullptr;
  shared_atomic_charges_ = nullptr;
  shared_point_charge_forces_ = nullptr;
}

bool XtbloomPartitionBroker::local_frame_is_valid(
    const WindowFrame &frame) const noexcept {
  const std::size_t atoms = local_topology_.atomic_numbers.size();
  const std::size_t points = local_topology_.point_charge_gammas.size();
  const std::size_t response =
      local_topology_.charge_response_enabled ? atoms * atoms : 0u;
  return frame.window_index == rank_ && frame.timestep >= 0 &&
         frame.positions.size() == 3u * atoms &&
         frame.point_charge_positions.size() == 3u * points &&
         frame.point_charge_values.size() == points &&
         frame.atomic_potential_shifts.size() ==
             (response == 0u ? 0u : atoms) &&
         frame.charge_response_matrix.size() == response;
}

void XtbloomPartitionBroker::stage_shared_frame(const WindowFrame &frame) {
  const std::size_t slot = static_cast<std::size_t>(rank_);
  std::copy(frame.positions.begin(), frame.positions.end(),
            shared_positions_ + position_displacements_[slot]);
  std::copy(frame.point_charge_positions.begin(),
            frame.point_charge_positions.end(),
            shared_point_positions_ + point_position_displacements_[slot]);
  std::copy(frame.point_charge_values.begin(), frame.point_charge_values.end(),
            shared_point_values_ + point_displacements_[slot]);
  std::copy(frame.atomic_potential_shifts.begin(),
            frame.atomic_potential_shifts.end(),
            shared_potential_shifts_ + shift_displacements_[slot]);
  std::copy(frame.charge_response_matrix.begin(),
            frame.charge_response_matrix.end(),
            shared_response_matrices_ + response_displacements_[slot]);
}

PartitionBrokerOutcome
XtbloomPartitionBroker::compute(const WindowFrame &local_frame) {
  const bool local_valid = local_frame_is_valid(local_frame);
  if (local_valid)
    stage_shared_frame(local_frame);
  // The preflight Allgather doubles as the process rendezvous for the shared
  // input mapping. Win_sync publishes each stable slot before the collective
  // and refreshes the root's local view after every peer has entered it.
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync xTB input publication");
  const std::array<std::int64_t, 2> local_preflight = {
      local_valid ? 1 : 0, local_frame.timestep};
  check_mpi(MPI_Allgather(local_preflight.data(), local_preflight.size(),
                          MPI_INT64_T, preflight_records_.data(),
                          local_preflight.size(), MPI_INT64_T, communicator_),
            "MPI_Allgather xTB frame preflight");
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync xTB input observation");
  const std::int64_t timestep = preflight_records_[1];
  bool frames_valid = true;
  bool timesteps_match = true;
  for (int slot = 0; slot < size_; ++slot) {
    const std::size_t base = 2u * static_cast<std::size_t>(slot);
    frames_valid = frames_valid && preflight_records_[base] == 1;
    timesteps_match =
        timesteps_match && preflight_records_[base + 1u] == timestep;
  }
  if (!frames_valid)
    throw std::invalid_argument("invalid local broker frame");
  if (!timesteps_match)
    throw std::invalid_argument(
        "all broker windows must submit the same timestep");

  result_valid_ = false;
  result_timestep_ = -1;
  last_error_.clear();

  XtbloomComputeOutcome native_outcome;
  if (rank_ == kBrokerRoot) {
    try {
      for (int slot = 0; slot < size_; ++slot) {
        const WindowFrameView frame{
            slot,
            timestep,
            slot_view(shared_positions_, position_counts_,
                      position_displacements_, slot),
            slot_view(shared_point_positions_, point_position_counts_,
                      point_position_displacements_, slot),
            slot_view(shared_point_values_, point_counts_,
                      point_displacements_, slot),
            slot_view(shared_potential_shifts_, shift_counts_,
                      shift_displacements_, slot),
            slot_view(shared_response_matrices_, response_counts_,
                      response_displacements_, slot)};
        executor_->stage(frame);
      }
      native_outcome = executor_->compute();
      if (native_outcome.call_status != XTBLOOM_STATUS_SUCCESS)
        last_error_ = executor_->last_error();
    } catch (const std::exception &exception) {
      native_outcome = {XTBLOOM_STATUS_INTERNAL_ERROR, timestep,
                        SccStartPolicy::Fresh, 0u, false};
      last_error_ = exception.what();
    } catch (...) {
      native_outcome = {XTBLOOM_STATUS_INTERNAL_ERROR, timestep,
                        SccStartPolicy::Fresh, 0u, false};
      last_error_ = "unknown exception during xTBloom broker compute";
    }
  }

  bool root_all_systems_succeeded = false;
  if (rank_ == kBrokerRoot &&
      native_outcome.call_status == XTBLOOM_STATUS_SUCCESS)
    root_all_systems_succeeded = publish_success(native_outcome);
  // Root publication is made visible before the existing outcome broadcast;
  // the broadcast is also the rendezvous that lets peers safely observe their
  // result slices without a second standalone barrier.
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync xTB output publication");
  check_mpi(MPI_Win_sync(shared_metadata_window_),
            "MPI_Win_sync xTB metadata publication");
  std::array<std::uint64_t, 4> publication = {
      rank_ == kBrokerRoot
          ? static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(native_outcome.call_status))
          : static_cast<std::uint64_t>(XTBLOOM_STATUS_INTERNAL_ERROR),
      rank_ == kBrokerRoot &&
              native_outcome.start_policy == SccStartPolicy::Warm
          ? 1u
          : 0u,
      rank_ == kBrokerRoot
          ? static_cast<std::uint64_t>(native_outcome.result_flags)
          : 0u,
      rank_ == kBrokerRoot && root_all_systems_succeeded ? 1u : 0u};
  check_mpi(MPI_Bcast(publication.data(), publication.size(), MPI_UINT64_T,
                      kBrokerRoot, communicator_),
            "MPI_Bcast xTBloom publication metadata");
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync xTB output observation");
  check_mpi(MPI_Win_sync(shared_metadata_window_),
            "MPI_Win_sync xTB metadata observation");

  const PartitionBrokerOutcome outcome{
      static_cast<xtbloom_status_t>(
          static_cast<std::uint32_t>(publication[0])),
      timestep,
      publication[1] == 0 ? SccStartPolicy::Fresh : SccStartPolicy::Warm,
      static_cast<std::uint32_t>(publication[2]), publication[3] != 0u};
  if (outcome.call_status == XTBLOOM_STATUS_SUCCESS &&
      outcome.all_systems_succeeded) {
    result_timestep_ = outcome.timestep;
    result_valid_ = true;
  } else {
    broadcast_error();
  }
  return outcome;
}

bool XtbloomPartitionBroker::publish_success(
    const XtbloomComputeOutcome &outcome) {
  if (rank_ != kBrokerRoot)
    return false;
  if (!outcome.all_systems_succeeded || !executor_->has_result()) {
    if (last_error_.empty())
      last_error_ = executor_->last_error();
    return false;
  }

  try {
    // First pass: validate every extent and pointer without touching shared
    // output.  A malformed native view therefore cannot leave a partial batch
    // in the shared mapping.  Keep this inside the root-side try block so an
    // unexpected executor state is converted into a collective failure rather
    // than throwing before peers reach the metadata broadcast.
    for (int slot = 0; slot < size_; ++slot) {
      const std::size_t index = static_cast<std::size_t>(slot);
      const WindowResultView view = executor_->result_for_window(slot);
      const bool valid =
          view.window_index == slot &&
          view.status == XTBLOOM_STATUS_SUCCESS && view.scc_converged &&
          view.atom_count == static_cast<std::size_t>(atom_counts_[index]) &&
          view.point_charge_count ==
              static_cast<std::size_t>(point_counts_[index]) &&
          (position_counts_[index] == 0 || view.forces != nullptr) &&
          (atom_counts_[index] == 0 || view.atomic_charges != nullptr) &&
          (point_position_counts_[index] == 0 ||
           view.point_charge_forces != nullptr);
      if (!valid) {
        executor_->invalidate_result();
        std::ostringstream message;
        message << "xTBloom result validation failed for window " << slot
                << ": status=" << view.status
                << " scc_converged="
                << (view.scc_converged ? "true" : "false")
                << " atom_count=" << view.atom_count << "/"
                << atom_counts_[index] << " point_charge_count="
                << view.point_charge_count << "/" << point_counts_[index];
        last_error_ = message.str();
        return false;
      }
    }

    // Second pass: publication is now all-or-nothing with respect to the
    // validation above.  No allocation occurs in this steady-state path.
    for (int slot = 0; slot < size_; ++slot) {
      const std::size_t index = static_cast<std::size_t>(slot);
      const WindowResultView view = executor_->result_for_window(slot);
      shared_metadata_[index].energy = view.energy;
      shared_metadata_[index].status = view.status;
      shared_metadata_[index].scc_iterations = view.scc_iterations;
      shared_metadata_[index].scc_converged = view.scc_converged ? 1u : 0u;
      std::copy_n(view.forces, position_counts_[index],
                  shared_forces_ + position_displacements_[index]);
      std::copy_n(view.atomic_charges, atom_counts_[index],
                  shared_atomic_charges_ + atom_displacements_[index]);
      if (point_position_counts_[index] != 0) {
        std::copy_n(view.point_charge_forces, point_position_counts_[index],
                    shared_point_charge_forces_ +
                        point_position_displacements_[index]);
      }
    }
  } catch (const std::exception &exception) {
    executor_->invalidate_result();
    last_error_ = std::string("xTBloom result publication failed: ") +
                  exception.what();
    return false;
  } catch (...) {
    executor_->invalidate_result();
    last_error_ = "xTBloom result publication failed: unknown exception";
    return false;
  }
  return true;
}

void XtbloomPartitionBroker::broadcast_error() {
  broadcast_string(communicator_, rank_, last_error_);
}

WindowResultView XtbloomPartitionBroker::result_for_local_window() const {
  if (!result_valid_)
    throw std::logic_error("no completed partition-broker result is available");
  const std::size_t slot = static_cast<std::size_t>(rank_);
  const SharedResultMetadata &metadata = shared_metadata_[slot];
  return {local_topology_.window_index,
          result_timestep_,
          static_cast<xtbloom_status_t>(metadata.status),
          metadata.scc_iterations,
          metadata.scc_converged != 0u,
          metadata.energy,
          shared_forces_ + position_displacements_[slot],
          shared_atomic_charges_ + atom_displacements_[slot],
          point_position_counts_[slot] == 0
              ? nullptr
              : shared_point_charge_forces_ +
                    point_position_displacements_[slot],
          local_topology_.atomic_numbers.size(),
          local_topology_.point_charge_gammas.size()};
}

} // namespace DPRC
