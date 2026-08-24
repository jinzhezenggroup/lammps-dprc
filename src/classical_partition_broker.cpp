#include "classical_partition_broker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace DPRC {
namespace {

constexpr int kBrokerRoot = 0;
constexpr std::size_t kCollectiveByteChunk = 1u << 20u;

[[nodiscard]] std::string mpi_diagnostic(int status, const char *operation) {
  char buffer[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  if (MPI_Error_string(status, buffer, &length) != MPI_SUCCESS || length <= 0)
    return std::string(operation) + " failed";
  return std::string(operation) +
         " failed: " + std::string(buffer, static_cast<std::size_t>(length));
}

void check_mpi(int status, const char *operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(mpi_diagnostic(status, operation));
}

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          const char *label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left * right;
}

[[nodiscard]] std::size_t checked_sum(std::size_t left, std::size_t right,
                                      const char *label) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string(label) + " size overflows");
  return left + right;
}

// Canonical little-endian field serialization avoids all dependence on C++
// object padding, enum layout, host endianness, or textual double formatting.
class CanonicalBytes {
public:
  void append_u8(std::uint8_t value) { bytes_.push_back(value); }

  void append_u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
      append_u8(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }

  void append_u64(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
      append_u8(static_cast<std::uint8_t>((value >> shift) & 0xffu));
  }

  void append_i32(std::int32_t value) {
    append_u32(static_cast<std::uint32_t>(value));
  }

  void append_size(std::size_t value) {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    append_u64(static_cast<std::uint64_t>(value));
  }

  void append_double(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(bits);
  }

  void append_i32_vector(const std::vector<std::int32_t> &values) {
    append_size(values.size());
    for (const std::int32_t value : values)
      append_i32(value);
  }

  void append_u8_vector(const std::vector<std::uint8_t> &values) {
    append_size(values.size());
    bytes_.insert(bytes_.end(), values.begin(), values.end());
  }

  void append_double_vector(const std::vector<double> &values) {
    append_size(values.size());
    for (const double value : values)
      append_double(value);
  }

  [[nodiscard]] std::vector<std::uint8_t> finish() && {
    return std::move(bytes_);
  }

private:
  std::vector<std::uint8_t> bytes_;
};

[[nodiscard]] std::vector<std::uint8_t>
serialize_topology(const ClassicalTopology &topology) {
  CanonicalBytes bytes;
  bytes.append_size(topology.atom_count);
  bytes.append_i32(topology.type_count);
  bytes.append_i32_vector(topology.atom_types);

  bytes.append_size(topology.tip4p_sites.size());
  for (const Tip4pSite &site : topology.tip4p_sites) {
    bytes.append_i32(site.oxygen);
    bytes.append_i32(site.hydrogen1);
    bytes.append_i32(site.hydrogen2);
  }

  bytes.append_size(topology.special_pairs.size());
  for (const SpecialPair &pair : topology.special_pairs) {
    bytes.append_i32(pair.atom1);
    bytes.append_i32(pair.atom2);
    bytes.append_double(pair.lj_scale);
    bytes.append_double(pair.coulomb_scale);
  }

  bytes.append_size(topology.lj.size());
  for (const LennardJonesParameters &entry : topology.lj) {
    bytes.append_double(entry.lj1);
    bytes.append_double(entry.lj2);
    bytes.append_double(entry.lj3);
    bytes.append_double(entry.lj4);
    bytes.append_double(entry.offset);
    bytes.append_double(entry.cutoff);
  }
  bytes.append_u8_vector(topology.coulomb_type_pairs);

  for (const double value : topology.cell.boxlo)
    bytes.append_double(value);
  for (const double value : topology.cell.h)
    bytes.append_double(value);
  for (const std::int32_t value : topology.pppm.mesh)
    bytes.append_i32(value);
  bytes.append_i32(topology.pppm.order);
  bytes.append_double(topology.pppm.g_ewald);

  const CoulombLookupTable &table = topology.coulomb_table;
  bytes.append_i32(table.bits);
  bytes.append_i32(table.shift_bits);
  bytes.append_i32(table.mask);
  bytes.append_double(table.inner_squared);
  bytes.append_double_vector(table.r);
  bytes.append_double_vector(table.dr);
  bytes.append_double_vector(table.force);
  bytes.append_double_vector(table.dforce);
  bytes.append_double_vector(table.coulomb);
  bytes.append_double_vector(table.dcoulomb);
  bytes.append_double_vector(table.energy);
  bytes.append_double_vector(table.denergy);

  bytes.append_double(topology.tip4p_alpha);
  bytes.append_double(topology.tip4p_qdist);
  bytes.append_double(topology.real_space_cutoff);
  bytes.append_double(topology.neighbor_skin);
  bytes.append_double(topology.qqrd2e);
  return std::move(bytes).finish();
}

[[nodiscard]] std::vector<std::uint8_t>
serialize_options(const ClassicalPlanOptions &options) {
  CanonicalBytes bytes;
  bytes.append_i32(static_cast<std::int32_t>(options.backend));
  bytes.append_size(options.max_batch_count);
  bytes.append_i32(options.cuda_device);
  return std::move(bytes).finish();
}

[[nodiscard]] bool finite_values(const double *values,
                                 std::size_t count) noexcept {
  if (!values)
    return false;
  for (std::size_t index = 0; index < count; ++index)
    if (!std::isfinite(values[index]))
      return false;
  return true;
}

} // namespace

ClassicalPartitionBroker::ClassicalPartitionBroker(
    MPI_Comm roots, ClassicalTopology local_topology,
    const ClassicalPlanOptions &options)
    : local_topology_(std::move(local_topology)) {
  if (roots == MPI_COMM_NULL)
    throw std::invalid_argument(
        "classical partition broker requires a roots communicator");
  check_mpi(MPI_Comm_dup(roots, &communicator_), "MPI_Comm_dup");
  try {
    check_mpi(MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler");
    check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(communicator_, &size_), "MPI_Comm_size");
    initialize_collective(options);
  } catch (...) {
    if (shared_window_locked_)
      MPI_Win_unlock_all(shared_window_);
    if (shared_window_ != MPI_WIN_NULL)
      MPI_Win_free(&shared_window_);
    if (shared_communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&shared_communicator_);
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
    throw;
  }
}

ClassicalPartitionBroker::~ClassicalPartitionBroker() {
  plan_.reset();
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || communicator_ == MPI_COMM_NULL)
    return;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    if (shared_window_locked_)
      MPI_Win_unlock_all(shared_window_);
    if (shared_window_ != MPI_WIN_NULL)
      MPI_Win_free(&shared_window_);
    if (shared_communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&shared_communicator_);
    MPI_Comm_free(&communicator_);
  }
}

void ClassicalPartitionBroker::require_collectively(bool local_condition,
                                                    const char *message) {
  const int local = local_condition ? 1 : 0;
  int global = 0;
  check_mpi(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator_),
            "MPI_Allreduce validation");
  if (global == 0) {
    last_error_ = message;
    throw std::invalid_argument(last_error_);
  }
}

void ClassicalPartitionBroker::require_matching_bytes(
    const std::vector<std::uint8_t> &local_bytes, const char *message) {
  std::uint64_t root_length =
      rank_ == kBrokerRoot ? static_cast<std::uint64_t>(local_bytes.size())
                           : 0u;
  check_mpi(
      MPI_Bcast(&root_length, 1, MPI_UINT64_T, kBrokerRoot, communicator_),
      "MPI_Bcast canonical length");
  const int local_length_match =
      root_length == static_cast<std::uint64_t>(local_bytes.size()) ? 1 : 0;
  int global_length_match = 0;
  check_mpi(MPI_Allreduce(&local_length_match, &global_length_match, 1,
                          MPI_INT, MPI_MIN, communicator_),
            "MPI_Allreduce canonical length match");
  if (global_length_match == 0) {
    last_error_ = std::string(message) + " (serialized lengths differ)";
    throw std::invalid_argument(last_error_);
  }

  std::uint64_t first_mismatch = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint8_t> received;
  if (rank_ != kBrokerRoot)
    received.resize(
        std::min<std::size_t>(local_bytes.size(), kCollectiveByteChunk));
  for (std::size_t offset = 0; offset < local_bytes.size();) {
    const std::size_t remaining = local_bytes.size() - offset;
    const int count = static_cast<int>(
        std::min<std::size_t>(remaining, kCollectiveByteChunk));
    void *buffer = rank_ == kBrokerRoot
                       ? static_cast<void *>(const_cast<std::uint8_t *>(
                             local_bytes.data() + offset))
                       : static_cast<void *>(received.data());
    check_mpi(MPI_Bcast(buffer, count, MPI_BYTE, kBrokerRoot, communicator_),
              "MPI_Bcast canonical bytes");
    if (rank_ != kBrokerRoot &&
        first_mismatch == std::numeric_limits<std::uint64_t>::max()) {
      const auto local_begin =
          local_bytes.begin() + static_cast<std::ptrdiff_t>(offset);
      const auto mismatch = std::mismatch(received.begin(),
                                          received.begin() + count,
                                          local_begin);
      if (mismatch.first != received.begin() + count)
        first_mismatch = static_cast<std::uint64_t>(offset) +
            static_cast<std::uint64_t>(
                std::distance(received.begin(), mismatch.first));
    }
    offset += static_cast<std::size_t>(count);
  }
  std::uint64_t global_mismatch = 0;
  check_mpi(MPI_Allreduce(&first_mismatch, &global_mismatch, 1, MPI_UINT64_T,
                          MPI_MIN, communicator_),
            "MPI_Allreduce canonical mismatch offset");
  if (global_mismatch != std::numeric_limits<std::uint64_t>::max()) {
    last_error_ = std::string(message) + " (first differing byte " +
        std::to_string(global_mismatch) + ")";
    throw std::invalid_argument(last_error_);
  }
}

void ClassicalPartitionBroker::initialize_shared_storage() {
  check_mpi(MPI_Comm_split_type(communicator_, MPI_COMM_TYPE_SHARED, rank_,
                                MPI_INFO_NULL, &shared_communicator_),
            "MPI_Comm_split_type shared roots");
  check_mpi(MPI_Comm_set_errhandler(shared_communicator_, MPI_ERRORS_RETURN),
            "MPI_Comm_set_errhandler shared roots");
  int shared_rank = -1;
  int shared_size = 0;
  check_mpi(MPI_Comm_rank(shared_communicator_, &shared_rank),
            "MPI_Comm_rank shared roots");
  check_mpi(MPI_Comm_size(shared_communicator_, &shared_size),
            "MPI_Comm_size shared roots");
  require_collectively(
      shared_rank == rank_ && shared_size == size_,
      "classical GPU broker roots must occupy one shared-memory node");

  const std::size_t batch = static_cast<std::size_t>(size_);
  const std::size_t batch_atoms =
      checked_product(batch, atom_count_, "shared batch atoms");
  const std::size_t batch_coordinates =
      checked_product(batch_atoms, std::size_t{3},
                      "shared batch coordinates");
  const std::size_t batch_virials =
      checked_product(batch, std::size_t{6}, "shared batch virials");
  std::size_t doubles = 0;
  auto reserve = [&](std::size_t count) {
    const std::size_t offset = doubles;
    doubles = checked_sum(doubles, count, "shared classical storage");
    return offset;
  };
  const std::size_t positions_offset = reserve(batch_coordinates);
  const std::size_t mm_charges_offset = reserve(batch_atoms);
  const std::size_t qm_charges_offset = reserve(batch_atoms);
  const std::size_t pair_forces_offset = reserve(batch_coordinates);
  const std::size_t lj_energy_offset = reserve(batch);
  const std::size_t coulomb_energy_offset = reserve(batch);
  const std::size_t pair_virial_offset = reserve(batch_virials);
  const std::size_t mm_energy_offset = reserve(batch);
  const std::size_t mm_virial_offset = reserve(batch_virials);
  const std::size_t mm_potential_offset = reserve(batch_atoms);
  const std::size_t mm_forces_offset = reserve(batch_coordinates);
  const std::size_t qm_forces_offset = reserve(batch_coordinates);
  const std::size_t full_forces_offset = reserve(batch_coordinates);
  const std::size_t qm_energy_offset = reserve(batch);
  const std::size_t full_energy_offset = reserve(batch);
  const std::size_t qm_virial_offset = reserve(batch_virials);
  const std::size_t full_virial_offset = reserve(batch_virials);

  if (doubles > static_cast<std::size_t>(
                    std::numeric_limits<MPI_Aint>::max() / sizeof(double)))
    throw std::overflow_error("shared classical storage exceeds MPI_Aint");
  const MPI_Aint bytes = static_cast<MPI_Aint>(doubles * sizeof(double));
  void *local_base = nullptr;
  check_mpi(MPI_Win_allocate_shared(rank_ == kBrokerRoot ? bytes : 0,
                                    sizeof(double), MPI_INFO_NULL,
                                    shared_communicator_, &local_base,
                                    &shared_window_),
            "MPI_Win_allocate_shared classical storage");
  MPI_Aint root_bytes = 0;
  int displacement = 0;
  void *root_base = nullptr;
  check_mpi(MPI_Win_shared_query(shared_window_, kBrokerRoot, &root_bytes,
                                 &displacement, &root_base),
            "MPI_Win_shared_query classical storage");
  require_collectively(root_base != nullptr && root_bytes == bytes &&
                           displacement == static_cast<int>(sizeof(double)),
                       "classical shared-memory mapping is inconsistent");
  check_mpi(MPI_Win_lock_all(MPI_MODE_NOCHECK, shared_window_),
            "MPI_Win_lock_all classical storage");
  shared_window_locked_ = true;
  shared_base_ = static_cast<double *>(root_base);
  gathered_positions_ = shared_base_ + positions_offset;
  gathered_mm_charges_ = shared_base_ + mm_charges_offset;
  gathered_qm_charges_ = shared_base_ + qm_charges_offset;
  root_pair_forces_ = shared_base_ + pair_forces_offset;
  root_lj_energy_ = shared_base_ + lj_energy_offset;
  root_coulomb_energy_ = shared_base_ + coulomb_energy_offset;
  root_pair_virial_ = shared_base_ + pair_virial_offset;
  root_mm_pppm_energy_ = shared_base_ + mm_energy_offset;
  root_mm_pppm_virial_ = shared_base_ + mm_virial_offset;
  root_mm_pppm_potential_ = shared_base_ + mm_potential_offset;
  root_mm_pppm_forces_ = shared_base_ + mm_forces_offset;
  root_qm_pppm_forces_ = shared_base_ + qm_forces_offset;
  root_full_pppm_forces_ = shared_base_ + full_forces_offset;
  root_qm_pppm_energy_ = shared_base_ + qm_energy_offset;
  root_full_pppm_energy_ = shared_base_ + full_energy_offset;
  root_qm_pppm_virial_ = shared_base_ + qm_virial_offset;
  root_full_pppm_virial_ = shared_base_ + full_virial_offset;
  synchronize_shared_storage();
}

void ClassicalPartitionBroker::synchronize_shared_storage() {
  check_mpi(MPI_Win_sync(shared_window_), "MPI_Win_sync classical storage");
  check_mpi(MPI_Barrier(shared_communicator_),
            "MPI_Barrier classical shared storage");
  check_mpi(MPI_Win_sync(shared_window_), "MPI_Win_sync classical storage");
}

void ClassicalPartitionBroker::initialize_collective(
    const ClassicalPlanOptions &options) {
  require_collectively(size_ > 0, "classical broker communicator is empty");

  std::vector<std::uint8_t> topology_bytes;
  std::vector<std::uint8_t> option_bytes;
  bool serialized = true;
  try {
    topology_bytes = serialize_topology(local_topology_);
    option_bytes = serialize_options(options);
  } catch (...) {
    serialized = false;
  }
  require_collectively(
      serialized, "classical broker topology or options cannot be serialized");
  require_matching_bytes(option_bytes,
                         "classical plan options must match on every root");
  require_matching_bytes(topology_bytes,
                         "classical topology must match exactly on every root");

  const bool counts_fit =
      local_topology_.atom_count > 0 &&
      local_topology_.atom_count <=
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 3);
  const bool capacity_fits =
      options.max_batch_count > 0 && size_ > 0 &&
      static_cast<std::size_t>(size_) <= options.max_batch_count;
  require_collectively(
      counts_fit, "classical atom count exceeds the MPI broker count range");
  require_collectively(capacity_fits,
                       "classical broker size exceeds plan batch capacity");

  atom_count_ = local_topology_.atom_count;
  atom_count_mpi_ = static_cast<int>(atom_count_);
  coordinate_count_mpi_ = static_cast<int>(3u * atom_count_);
  initialize_shared_storage();

  bool storage_ready = true;
  try {
    mm_pair_forces_.resize(atom_count_ * 3u);
    candidate_mm_pair_forces_.resize(atom_count_ * 3u);
    mm_pppm_potential_.resize(atom_count_);
    candidate_mm_pppm_potential_.resize(atom_count_);
    mm_pppm_forces_.resize(atom_count_ * 3u);
    candidate_mm_pppm_forces_.resize(atom_count_ * 3u);
    qm_pppm_forces_.resize(atom_count_ * 3u);
    candidate_qm_pppm_forces_.resize(atom_count_ * 3u);
    full_pppm_forces_.resize(atom_count_ * 3u);
    candidate_full_pppm_forces_.resize(atom_count_ * 3u);
    preflight_records_.resize(4u * static_cast<std::size_t>(size_));
  } catch (...) {
    storage_ready = false;
  }
  require_collectively(
      storage_ready, "classical broker could not allocate fixed batch storage");

  ErrorKind root_error = ErrorKind::None;
  std::string diagnostic;
  if (rank_ == kBrokerRoot) {
    try {
      plan_ = create_classical_batch_plan(local_topology_, options);
      if (!plan_)
        throw std::runtime_error("classical plan factory returned null");
    } catch (const std::invalid_argument &error) {
      root_error = ErrorKind::InvalidArgument;
      diagnostic = error.what();
    } catch (const std::overflow_error &error) {
      root_error = ErrorKind::Overflow;
      diagnostic = error.what();
    } catch (const std::logic_error &error) {
      root_error = ErrorKind::Logic;
      diagnostic = error.what();
    } catch (const std::exception &error) {
      root_error = ErrorKind::Runtime;
      diagnostic = error.what();
    } catch (...) {
      root_error = ErrorKind::Runtime;
      diagnostic =
          "classical plan construction failed with an unknown exception";
    }
  }
  broadcast_root_error(root_error, diagnostic);
  if (root_error != ErrorKind::None)
    throw_error(root_error, diagnostic);
  last_error_.clear();
}

bool ClassicalPartitionBroker::mm_frame_is_valid(
    const ClassicalMmFrameView &frame) const noexcept {
  return frame.timestep >= 0 && frame.position_count == 3u * atom_count_ &&
         frame.charge_count == atom_count_ &&
         finite_values(frame.positions, frame.position_count) &&
         finite_values(frame.mm_charges, frame.charge_count);
}

bool ClassicalPartitionBroker::qm_frame_is_valid(
    const ClassicalQmFrameView &frame) const noexcept {
  return frame.timestep >= 0 && frame.charge_count == atom_count_ &&
         finite_values(frame.qm_charges, frame.charge_count);
}

void ClassicalPartitionBroker::broadcast_root_error(ErrorKind &kind,
                                                    std::string &diagnostic) {
  int encoded = rank_ == kBrokerRoot ? static_cast<int>(kind) : 0;
  check_mpi(MPI_Bcast(&encoded, 1, MPI_INT, kBrokerRoot, communicator_),
            "MPI_Bcast classical owner status");
  kind = static_cast<ErrorKind>(encoded);
  if (kind == ErrorKind::None)
    return;

  std::uint64_t length =
      rank_ == kBrokerRoot ? static_cast<std::uint64_t>(diagnostic.size()) : 0u;
  check_mpi(MPI_Bcast(&length, 1, MPI_UINT64_T, kBrokerRoot, communicator_),
            "MPI_Bcast classical diagnostic length");
  if (length >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("classical broker diagnostic is too large");
  if (rank_ != kBrokerRoot)
    diagnostic.resize(static_cast<std::size_t>(length));
  for (std::size_t offset = 0; offset < diagnostic.size();) {
    const int count = static_cast<int>(std::min<std::size_t>(
        diagnostic.size() - offset, kCollectiveByteChunk));
    check_mpi(MPI_Bcast(diagnostic.data() + offset, count, MPI_CHAR,
                        kBrokerRoot, communicator_),
              "MPI_Bcast classical diagnostic");
    offset += static_cast<std::size_t>(count);
  }
}

[[noreturn]] void
ClassicalPartitionBroker::throw_error(ErrorKind kind,
                                      const std::string &diagnostic) {
  last_error_ =
      diagnostic.empty() ? "classical broker owner failed" : diagnostic;
  switch (kind) {
  case ErrorKind::InvalidArgument:
    throw std::invalid_argument(last_error_);
  case ErrorKind::Logic:
    throw std::logic_error(last_error_);
  case ErrorKind::Overflow:
    throw std::overflow_error(last_error_);
  case ErrorKind::None:
  case ErrorKind::Runtime:
    throw std::runtime_error(last_error_);
  }
  throw std::runtime_error(last_error_);
}

void ClassicalPartitionBroker::begin_mm(
    const ClassicalMmFrameView &local_frame,
    ClassicalMmResultRequest request) {
  last_error_.clear();
  const int local_request = (request.pppm_potential ? 1 : 0) |
      (request.pppm_forces ? 2 : 0) | (request.retain_for_qm ? 4 : 0);
  const std::array<std::int64_t, 4> local_preflight{
      local_frame.timestep, static_cast<std::int64_t>(local_request),
      active_mm_ ? 0 : 1, mm_frame_is_valid(local_frame) ? 1 : 0};
  check_mpi(MPI_Allgather(local_preflight.data(),
                          static_cast<int>(local_preflight.size()),
                          MPI_INT64_T, preflight_records_.data(),
                          static_cast<int>(local_preflight.size()), MPI_INT64_T,
                          communicator_),
            "MPI_Allgather classical MM preflight");
  const std::int64_t timestep = preflight_records_[0];
  const std::int64_t publication_request = preflight_records_[1];
  for (int root = 0; root < size_; ++root) {
    const std::size_t offset = 4u * static_cast<std::size_t>(root);
    if (preflight_records_[offset + 2] == 0) {
      last_error_ = "classical broker already has an active MM epoch";
      throw std::invalid_argument(last_error_);
    }
    if (preflight_records_[offset] != timestep) {
      last_error_ = "classical broker timesteps are not synchronized";
      throw std::invalid_argument(last_error_);
    }
    if (preflight_records_[offset + 1] != publication_request) {
      last_error_ = "classical MM publication requests do not match";
      throw std::invalid_argument(last_error_);
    }
    if (preflight_records_[offset + 3] == 0) {
      last_error_ = "classical MM frame has invalid extents or values";
      throw std::invalid_argument(last_error_);
    }
  }

  const std::size_t rank_atoms =
      static_cast<std::size_t>(rank_) * atom_count_;
  const std::size_t rank_coordinates = 3u * rank_atoms;
  std::copy_n(local_frame.positions, static_cast<std::size_t>(coordinate_count_mpi_),
              gathered_positions_ + rank_coordinates);
  std::copy_n(local_frame.mm_charges, atom_count_,
              gathered_mm_charges_ + rank_atoms);
  synchronize_shared_storage();

  ErrorKind root_error = ErrorKind::None;
  std::string diagnostic;
  if (rank_ == kBrokerRoot) {
    try {
      ClassicalBatchInput input{static_cast<std::size_t>(size_),
                                gathered_positions_, gathered_mm_charges_};
      ClassicalMmBatchOutput output{
          static_cast<std::size_t>(size_), root_pair_forces_,
          root_lj_energy_,                  root_coulomb_energy_,
          root_pair_virial_,                root_mm_pppm_energy_,
          root_mm_pppm_virial_,
          request.pppm_potential ? root_mm_pppm_potential_ : nullptr,
          request.pppm_forces ? root_mm_pppm_forces_ : nullptr,
          request.retain_for_qm};
      plan_->begin_mm(input, output);
    } catch (const std::invalid_argument &error) {
      root_error = ErrorKind::InvalidArgument;
      diagnostic = error.what();
    } catch (const std::overflow_error &error) {
      root_error = ErrorKind::Overflow;
      diagnostic = error.what();
    } catch (const std::logic_error &error) {
      root_error = ErrorKind::Logic;
      diagnostic = error.what();
    } catch (const std::exception &error) {
      root_error = ErrorKind::Runtime;
      diagnostic = error.what();
    } catch (...) {
      root_error = ErrorKind::Runtime;
      diagnostic =
          "classical MM owner execution failed with an unknown exception";
    }
    if (root_error != ErrorKind::None)
      plan_->cancel();
  }
  broadcast_root_error(root_error, diagnostic);
  if (root_error != ErrorKind::None)
    throw_error(root_error, diagnostic);
  synchronize_shared_storage();
  std::copy_n(root_pair_forces_ + rank_coordinates,
              static_cast<std::size_t>(coordinate_count_mpi_),
              candidate_mm_pair_forces_.data());
  candidate_lj_energy_ = root_lj_energy_[rank_];
  candidate_coulomb_energy_ = root_coulomb_energy_[rank_];
  std::copy_n(root_pair_virial_ + 6u * static_cast<std::size_t>(rank_), 6,
              candidate_pair_virial_.data());
  candidate_mm_pppm_energy_ = root_mm_pppm_energy_[rank_];
  std::copy_n(root_mm_pppm_virial_ + 6u * static_cast<std::size_t>(rank_), 6,
              candidate_mm_pppm_virial_.data());
  if (request.pppm_potential)
    std::copy_n(root_mm_pppm_potential_ + rank_atoms, atom_count_,
                candidate_mm_pppm_potential_.data());
  if (request.pppm_forces)
    std::copy_n(root_mm_pppm_forces_ + rank_coordinates,
                static_cast<std::size_t>(coordinate_count_mpi_),
                candidate_mm_pppm_forces_.data());

  mm_pair_forces_.swap(candidate_mm_pair_forces_);
  if (request.pppm_potential)
    mm_pppm_potential_.swap(candidate_mm_pppm_potential_);
  if (request.pppm_forces)
    mm_pppm_forces_.swap(candidate_mm_pppm_forces_);
  pair_virial_.swap(candidate_pair_virial_);
  mm_pppm_virial_.swap(candidate_mm_pppm_virial_);
  std::swap(lj_energy_, candidate_lj_energy_);
  std::swap(coulomb_energy_, candidate_coulomb_energy_);
  std::swap(mm_pppm_energy_, candidate_mm_pppm_energy_);
  mm_result_timestep_ = timestep;
  mm_result_valid_ = true;
  mm_potential_valid_ = request.pppm_potential;
  mm_forces_valid_ = request.pppm_forces;
  active_timestep_ = request.retain_for_qm ? timestep : -1;
  active_mm_ = request.retain_for_qm;
}

void ClassicalPartitionBroker::finish_qm(
    const ClassicalQmFrameView &local_frame) {
  last_error_.clear();
  const std::array<std::int64_t, 4> local_preflight{
      local_frame.timestep, active_timestep_, active_mm_ ? 1 : 0,
      qm_frame_is_valid(local_frame) ? 1 : 0};
  check_mpi(MPI_Allgather(local_preflight.data(),
                          static_cast<int>(local_preflight.size()),
                          MPI_INT64_T, preflight_records_.data(),
                          static_cast<int>(local_preflight.size()), MPI_INT64_T,
                          communicator_),
            "MPI_Allgather classical QM preflight");
  const std::int64_t timestep = preflight_records_[0];
  for (int root = 0; root < size_; ++root) {
    const std::size_t offset = 4u * static_cast<std::size_t>(root);
    if (preflight_records_[offset + 2] == 0) {
      last_error_ = "classical broker has no active MM epoch";
      throw std::invalid_argument(last_error_);
    }
    if (preflight_records_[offset] != timestep) {
      last_error_ = "classical broker timesteps are not synchronized";
      throw std::invalid_argument(last_error_);
    }
    if (preflight_records_[offset + 1] != timestep) {
      last_error_ =
          "classical QM timestep does not match the active MM epoch";
      throw std::invalid_argument(last_error_);
    }
    if (preflight_records_[offset + 3] == 0) {
      last_error_ = "classical QM frame has invalid extents or values";
      throw std::invalid_argument(last_error_);
    }
  }

  const std::size_t rank_atoms =
      static_cast<std::size_t>(rank_) * atom_count_;
  const std::size_t rank_coordinates = 3u * rank_atoms;
  std::copy_n(local_frame.qm_charges, atom_count_,
              gathered_qm_charges_ + rank_atoms);
  synchronize_shared_storage();

  ErrorKind root_error = ErrorKind::None;
  std::string diagnostic;
  if (rank_ == kBrokerRoot) {
    try {
      ClassicalQmBatchInput input{static_cast<std::size_t>(size_),
                                  gathered_qm_charges_};
      ClassicalQmBatchOutput output{
          static_cast<std::size_t>(size_), root_qm_pppm_forces_,
          root_full_pppm_forces_,          root_qm_pppm_energy_,
          root_full_pppm_energy_,          root_qm_pppm_virial_,
          root_full_pppm_virial_};
      plan_->finish_qm(input, output);
    } catch (const std::invalid_argument &error) {
      root_error = ErrorKind::InvalidArgument;
      diagnostic = error.what();
    } catch (const std::overflow_error &error) {
      root_error = ErrorKind::Overflow;
      diagnostic = error.what();
    } catch (const std::logic_error &error) {
      root_error = ErrorKind::Logic;
      diagnostic = error.what();
    } catch (const std::exception &error) {
      root_error = ErrorKind::Runtime;
      diagnostic = error.what();
    } catch (...) {
      root_error = ErrorKind::Runtime;
      diagnostic =
          "classical QM owner execution failed with an unknown exception";
    }
    if (root_error != ErrorKind::None)
      plan_->cancel();
  }
  broadcast_root_error(root_error, diagnostic);
  if (root_error != ErrorKind::None) {
    active_mm_ = false;
    active_timestep_ = -1;
    throw_error(root_error, diagnostic);
  }
  synchronize_shared_storage();
  std::copy_n(root_qm_pppm_forces_ + rank_coordinates,
              static_cast<std::size_t>(coordinate_count_mpi_),
              candidate_qm_pppm_forces_.data());
  std::copy_n(root_full_pppm_forces_ + rank_coordinates,
              static_cast<std::size_t>(coordinate_count_mpi_),
              candidate_full_pppm_forces_.data());
  candidate_qm_pppm_energy_ = root_qm_pppm_energy_[rank_];
  candidate_full_pppm_energy_ = root_full_pppm_energy_[rank_];
  std::copy_n(root_qm_pppm_virial_ + 6u * static_cast<std::size_t>(rank_), 6,
              candidate_qm_pppm_virial_.data());
  std::copy_n(root_full_pppm_virial_ + 6u * static_cast<std::size_t>(rank_), 6,
              candidate_full_pppm_virial_.data());
  active_mm_ = false;
  active_timestep_ = -1;

  qm_pppm_forces_.swap(candidate_qm_pppm_forces_);
  full_pppm_forces_.swap(candidate_full_pppm_forces_);
  qm_pppm_virial_.swap(candidate_qm_pppm_virial_);
  full_pppm_virial_.swap(candidate_full_pppm_virial_);
  std::swap(qm_pppm_energy_, candidate_qm_pppm_energy_);
  std::swap(full_pppm_energy_, candidate_full_pppm_energy_);
  qm_result_timestep_ = timestep;
  qm_result_valid_ = true;
}

void ClassicalPartitionBroker::clear_active_epoch() noexcept {
  if (rank_ == kBrokerRoot && plan_)
    plan_->cancel();
  active_mm_ = false;
  active_timestep_ = -1;
}

void ClassicalPartitionBroker::cancel() noexcept { clear_active_epoch(); }

ClassicalMmResultView ClassicalPartitionBroker::mm_result() const {
  if (!mm_result_valid_)
    throw std::logic_error("classical broker has no published MM result");
  return {mm_result_timestep_,      atom_count_,
          mm_pair_forces_.data(),   lj_energy_,
          coulomb_energy_,          pair_virial_.data(),
          mm_pppm_energy_,          mm_pppm_virial_.data(),
          mm_potential_valid_ ? mm_pppm_potential_.data() : nullptr,
          mm_forces_valid_ ? mm_pppm_forces_.data() : nullptr};
}

ClassicalQmResultView ClassicalPartitionBroker::qm_result() const {
  if (!qm_result_valid_)
    throw std::logic_error("classical broker has no published QM result");
  return {qm_result_timestep_,    atom_count_,
          qm_pppm_forces_.data(), full_pppm_forces_.data(),
          qm_pppm_energy_,        full_pppm_energy_,
          qm_pppm_virial_.data(), full_pppm_virial_.data()};
}

} // namespace DPRC
