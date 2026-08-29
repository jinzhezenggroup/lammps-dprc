#include "deepmd_partition_broker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace DPRC {
namespace {

constexpr int kOwner = 0;

void check_mpi(int status, const char *operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed");
}

int checked_count(std::size_t count, const char *label) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) + " exceeds MPI int count");
  return static_cast<int>(count);
}

void build_displacements(const std::vector<int> &counts,
                         std::vector<int> &displacements,
                         const char *label) {
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

std::size_t grown_capacity(std::size_t required, const char *label) {
  const std::size_t slack = required / 8 + 64;
  if (required > std::numeric_limits<std::size_t>::max() - slack)
    throw std::overflow_error(std::string(label) + " capacity overflows");
  return required + slack;
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char *label) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error(std::string(label) + " byte size overflows");
  return left * right;
}

std::size_t align_offset(std::size_t offset, std::size_t alignment,
                         const char *label) {
  const std::size_t remainder = offset % alignment;
  const std::size_t padding = remainder == 0 ? 0 : alignment - remainder;
  if (offset > std::numeric_limits<std::size_t>::max() - padding)
    throw std::overflow_error(std::string(label) + " alignment overflows");
  return offset + padding;
}

struct SharedDataLayout {
  std::size_t bytes = 0;
  std::size_t atom_types = 0;
  std::size_t sources = 0;
  std::size_t edge_vectors = 0;
  std::size_t destination_row_ptr = 0;
  std::size_t source_row_ptr = 0;
  std::size_t source_order = 0;
  std::size_t atom_energy = 0;
  std::size_t force = 0;
  std::size_t atom_virial = 0;
};

template <typename T>
std::size_t reserve_array(std::size_t &offset, std::size_t count,
                          const char *label) {
  offset = align_offset(offset, alignof(T), label);
  const std::size_t begin = offset;
  const std::size_t bytes = checked_product(count, sizeof(T), label);
  if (offset > std::numeric_limits<std::size_t>::max() - bytes)
    throw std::overflow_error(std::string(label) + " storage overflows");
  offset += bytes;
  return begin;
}

SharedDataLayout build_shared_layout(std::size_t node_capacity,
                                     std::size_t edge_capacity) {
  SharedDataLayout layout;
  std::size_t offset = 0;
  layout.atom_types =
      reserve_array<std::int64_t>(offset, node_capacity, "atom types");
  layout.sources =
      reserve_array<std::uint32_t>(offset, edge_capacity, "edge sources");
  layout.edge_vectors = reserve_array<float>(
      offset, checked_product(edge_capacity, 3, "edge vectors"),
      "edge vectors");
  layout.destination_row_ptr = reserve_array<std::int64_t>(
      offset, node_capacity + 1, "destination CSR");
  layout.source_row_ptr =
      reserve_array<std::int64_t>(offset, node_capacity + 1, "source CSR");
  layout.source_order =
      reserve_array<std::uint32_t>(offset, edge_capacity, "source order");
  layout.atom_energy =
      reserve_array<double>(offset, node_capacity, "atomic energy");
  layout.force = reserve_array<double>(
      offset, checked_product(node_capacity, 3, "force"), "force");
  layout.atom_virial = reserve_array<double>(
      offset, checked_product(node_capacity, 9, "atomic virial"),
      "atomic virial");
  layout.bytes = offset;
  return layout;
}

void broadcast_string(MPI_Comm communicator, int rank, std::string &value) {
  int length = rank == kOwner ? checked_count(value.size(), "string length") : 0;
  check_mpi(MPI_Bcast(&length, 1, MPI_INT, kOwner, communicator),
            "MPI_Bcast string length");
  if (rank != kOwner)
    value.resize(static_cast<std::size_t>(length));
  check_mpi(MPI_Bcast(value.empty() ? nullptr : value.data(), length, MPI_CHAR,
                      kOwner, communicator),
            "MPI_Bcast string payload");
}

}  // namespace

DeepmdPartitionBroker::DeepmdPartitionBroker(MPI_Comm roots,
                                             std::string model_path,
                                             int gpu_rank) {
  if (roots == MPI_COMM_NULL)
    throw std::invalid_argument(
        "DeePMD partition broker requires a root communicator");
  check_mpi(MPI_Comm_dup(roots, &communicator_), "MPI_Comm_dup");
  try {
    check_mpi(MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler");
    check_mpi(MPI_Comm_rank(communicator_, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(communicator_, &size_), "MPI_Comm_size");
    if (size_ <= 0)
      throw std::invalid_argument("DeePMD partition broker is empty");

#if defined(MPI_VERSION) && MPI_VERSION >= 3
    check_mpi(MPI_Comm_split_type(communicator_, MPI_COMM_TYPE_SHARED, rank_,
                                  MPI_INFO_NULL, &shared_communicator_),
              "MPI_Comm_split_type");
    check_mpi(MPI_Comm_set_errhandler(shared_communicator_, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler shared");
    int shared_rank = -1;
    int shared_size = 0;
    check_mpi(MPI_Comm_rank(shared_communicator_, &shared_rank),
              "MPI_Comm_rank shared");
    check_mpi(MPI_Comm_size(shared_communicator_, &shared_size),
              "MPI_Comm_size shared");
    if (shared_rank != rank_ || shared_size != size_)
      throw std::invalid_argument(
          "dprc/deepmd/batch requires all partition roots on one GPU-local "
          "shared-memory node");
#else
    throw std::invalid_argument(
        "dprc/deepmd/batch requires an MPI-3 implementation");
#endif
    initialize_model(model_path, gpu_rank);
  } catch (...) {
    if (shared_communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&shared_communicator_);
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
    throw;
  }
}

DeepmdPartitionBroker::~DeepmdPartitionBroker() {
  executor_.reset();
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized)
    return;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    release_shared_storage();
    if (shared_communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&shared_communicator_);
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
  }
}

void DeepmdPartitionBroker::initialize_model(const std::string &model_path,
                                             int gpu_rank) {
  int failed = 0;
  std::string diagnostic;
  if (rank_ == kOwner) {
    try {
      executor_ = std::make_unique<DeepmdBatchExecutor>(model_path, gpu_rank);
      metadata_ = executor_->metadata();
    } catch (const std::exception &exception) {
      failed = 1;
      diagnostic = exception.what();
    }
  }
  check_mpi(MPI_Bcast(&failed, 1, MPI_INT, kOwner, communicator_),
            "MPI_Bcast DeePMD model status");
  if (failed) {
    broadcast_string(communicator_, rank_, diagnostic);
    throw std::runtime_error(diagnostic);
  }

  double cutoff = rank_ == kOwner ? metadata_.cutoff : 0.0;
  int integer_metadata[5] = {
      rank_ == kOwner ? metadata_.type_count : 0,
      rank_ == kOwner ? metadata_.spin_type_count : 0,
      rank_ == kOwner ? metadata_.frame_parameter_width : 0,
      rank_ == kOwner ? metadata_.atom_parameter_width : 0,
      rank_ == kOwner ? metadata_.charge_spin_width : 0};
  check_mpi(MPI_Bcast(&cutoff, 1, MPI_DOUBLE, kOwner, communicator_),
            "MPI_Bcast DeePMD cutoff");
  check_mpi(MPI_Bcast(integer_metadata, 5, MPI_INT, kOwner, communicator_),
            "MPI_Bcast DeePMD metadata");
  broadcast_string(communicator_, rank_, metadata_.type_map);
  metadata_.cutoff = cutoff;
  metadata_.type_count = integer_metadata[0];
  metadata_.spin_type_count = integer_metadata[1];
  metadata_.frame_parameter_width = integer_metadata[2];
  metadata_.atom_parameter_width = integer_metadata[3];
  metadata_.charge_spin_width = integer_metadata[4];
}

bool DeepmdPartitionBroker::local_graph_valid(
    const DeepmdCanonicalGraph &graph, std::string &diagnostic) const {
  const std::size_t nodes = graph.atom_types.size();
  const std::size_t edges = graph.sources.size();
  if (graph.timestep < 0 || nodes == 0) {
    diagnostic = "canonical graph timestep and node count must be positive";
    return false;
  }
  if (nodes > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      edges > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    diagnostic = "canonical graph exceeds MPI int count";
    return false;
  }
  if (graph.edge_vectors.size() != 3 * edges ||
      graph.destination_row_ptr.size() != nodes + 1 ||
      graph.source_row_ptr.size() != nodes + 1 ||
      graph.source_order.size() != edges) {
    diagnostic = "canonical graph array extents are inconsistent";
    return false;
  }
  if (graph.destination_row_ptr.front() != 0 ||
      graph.source_row_ptr.front() != 0 ||
      graph.destination_row_ptr.back() != static_cast<std::int64_t>(edges) ||
      graph.source_row_ptr.back() != static_cast<std::int64_t>(edges)) {
    diagnostic = "canonical graph CSR terminal entries are invalid";
    return false;
  }
  const std::int64_t edge_count = static_cast<std::int64_t>(edges);
  for (std::size_t node = 0; node < nodes; ++node) {
    const std::int64_t destination_begin = graph.destination_row_ptr[node];
    const std::int64_t destination_end =
        graph.destination_row_ptr[node + 1];
    const std::int64_t source_begin = graph.source_row_ptr[node];
    const std::int64_t source_end = graph.source_row_ptr[node + 1];
    if (graph.atom_types[node] < 0 ||
        graph.atom_types[node] >= metadata_.type_count ||
        destination_begin < 0 || destination_end < destination_begin ||
        destination_end > edge_count || source_begin < 0 ||
        source_end < source_begin || source_end > edge_count) {
      diagnostic = "canonical graph contains an invalid type or CSR row";
      return false;
    }
  }
  for (std::size_t edge = 0; edge < edges; ++edge) {
    if (graph.sources[edge] >= nodes || graph.source_order[edge] >= edges ||
        !std::isfinite(graph.edge_vectors[3 * edge + 0]) ||
        !std::isfinite(graph.edge_vectors[3 * edge + 1]) ||
        !std::isfinite(graph.edge_vectors[3 * edge + 2])) {
      diagnostic = "canonical graph contains an invalid edge";
      return false;
    }
  }
  std::vector<unsigned char> source_order_seen(edges, 0);
  for (std::size_t source = 0; source < nodes; ++source) {
    for (std::int64_t slot = graph.source_row_ptr[source];
         slot < graph.source_row_ptr[source + 1]; ++slot) {
      const std::uint32_t edge =
          graph.source_order[static_cast<std::size_t>(slot)];
      if (source_order_seen[edge] || graph.sources[edge] != source) {
        diagnostic =
            "canonical source order is not a source-grouped permutation";
        return false;
      }
      source_order_seen[edge] = 1;
    }
  }
  return true;
}

void DeepmdPartitionBroker::release_shared_storage() noexcept {
  if (shared_data_locked_ && shared_data_window_ != MPI_WIN_NULL)
    MPI_Win_unlock_all(shared_data_window_);
  shared_data_locked_ = false;
  if (shared_data_window_ != MPI_WIN_NULL)
    MPI_Win_free(&shared_data_window_);
  shared_node_capacity_ = 0;
  shared_edge_capacity_ = 0;
  shared_atom_types_ = nullptr;
  shared_sources_ = nullptr;
  shared_edge_vectors_ = nullptr;
  shared_destination_row_ptr_ = nullptr;
  shared_source_row_ptr_ = nullptr;
  shared_source_order_ = nullptr;
  shared_atom_energy_ = nullptr;
  shared_force_ = nullptr;
  shared_atom_virial_ = nullptr;
}

void DeepmdPartitionBroker::ensure_shared_capacity(
    std::size_t nodes, std::size_t edge_storage) {
  if (nodes <= shared_node_capacity_ &&
      edge_storage <= shared_edge_capacity_)
    return;

  const std::size_t node_capacity =
      nodes > shared_node_capacity_
          ? grown_capacity(nodes, "shared DeePMD node")
          : shared_node_capacity_;
  const std::size_t edge_capacity =
      edge_storage > shared_edge_capacity_
          ? grown_capacity(edge_storage, "shared DeePMD edge")
          : shared_edge_capacity_;
  const SharedDataLayout layout =
      build_shared_layout(node_capacity, edge_capacity);
  if (layout.bytes > static_cast<std::size_t>(
                         std::numeric_limits<MPI_Aint>::max()))
    throw std::overflow_error("shared DeePMD storage exceeds MPI_Aint");

  // Window allocation and release are collective. All ranks derive identical
  // capacities from the all-gathered graph extents before entering here.
  release_shared_storage();
  void *local_base = nullptr;
  const MPI_Aint root_bytes = static_cast<MPI_Aint>(layout.bytes);
  check_mpi(MPI_Win_allocate_shared(rank_ == kOwner ? root_bytes : 0, 1,
                                    MPI_INFO_NULL, shared_communicator_,
                                    &local_base, &shared_data_window_),
            "MPI_Win_allocate_shared DeePMD data");
  int setup_valid = 1;
  std::string setup_diagnostic;
  const auto record_setup_status = [&](int status, const char *operation) {
    if (status != MPI_SUCCESS && setup_valid) {
      setup_valid = 0;
      setup_diagnostic = std::string(operation) + " failed";
    }
  };
  record_setup_status(
      MPI_Win_set_errhandler(shared_data_window_, MPI_ERRORS_RETURN),
      "MPI_Win_set_errhandler DeePMD data");

  // Direct C++ loads and stores through another rank's shared mapping are
  // portable only when MPI exposes one coherent public/private copy. Reject a
  // separate-model implementation collectively before publishing pointers.
  int *window_model = nullptr;
  int model_available = 0;
  record_setup_status(
      MPI_Win_get_attr(shared_data_window_, MPI_WIN_MODEL, &window_model,
                       &model_available),
      "MPI_Win_get_attr DeePMD memory model");
  const bool local_unified =
      (model_available && window_model != nullptr &&
       *window_model == MPI_WIN_UNIFIED);
  if (!local_unified && setup_valid) {
    setup_valid = 0;
    setup_diagnostic =
        "dprc/deepmd/batch requires the MPI unified shared-window memory "
        "model";
  }

  MPI_Aint queried_bytes = 0;
  int displacement_unit = 0;
  void *root_base = nullptr;
  record_setup_status(
      MPI_Win_shared_query(shared_data_window_, kOwner, &queried_bytes,
                           &displacement_unit, &root_base),
      "MPI_Win_shared_query DeePMD data");
  if (root_base == nullptr || queried_bytes != root_bytes ||
      displacement_unit != 1) {
    if (setup_valid) {
      setup_valid = 0;
      setup_diagnostic = "shared DeePMD data mapping is inconsistent";
    }
  }
  const int lock_status =
      MPI_Win_lock_all(MPI_MODE_NOCHECK, shared_data_window_);
  record_setup_status(lock_status, "MPI_Win_lock_all DeePMD data");
  shared_data_locked_ = lock_status == MPI_SUCCESS;

  int all_setup_valid = 0;
  check_mpi(MPI_Allreduce(&setup_valid, &all_setup_valid, 1, MPI_INT, MPI_MIN,
                          shared_communicator_),
            "MPI_Allreduce DeePMD shared-window setup");
  if (!all_setup_valid) {
    const int candidate = setup_valid ? size_ : rank_;
    int diagnostic_rank = size_;
    check_mpi(MPI_Allreduce(&candidate, &diagnostic_rank, 1, MPI_INT, MPI_MIN,
                            shared_communicator_),
              "MPI_Allreduce DeePMD setup diagnostic rank");
    if (rank_ != diagnostic_rank)
      setup_diagnostic.clear();
    int diagnostic_length =
        rank_ == diagnostic_rank
            ? checked_count(setup_diagnostic.size(), "setup diagnostic")
            : 0;
    check_mpi(MPI_Bcast(&diagnostic_length, 1, MPI_INT, diagnostic_rank,
                        shared_communicator_),
              "MPI_Bcast DeePMD setup diagnostic length");
    if (rank_ != diagnostic_rank)
      setup_diagnostic.resize(static_cast<std::size_t>(diagnostic_length));
    check_mpi(MPI_Bcast(setup_diagnostic.empty()
                            ? nullptr
                            : setup_diagnostic.data(),
                        diagnostic_length, MPI_CHAR, diagnostic_rank,
                        shared_communicator_),
              "MPI_Bcast DeePMD setup diagnostic");
    release_shared_storage();
    throw std::runtime_error(setup_diagnostic);
  }

  auto *bytes = static_cast<unsigned char *>(root_base);
  shared_atom_types_ =
      reinterpret_cast<std::int64_t *>(bytes + layout.atom_types);
  shared_sources_ =
      reinterpret_cast<std::uint32_t *>(bytes + layout.sources);
  shared_edge_vectors_ =
      reinterpret_cast<float *>(bytes + layout.edge_vectors);
  shared_destination_row_ptr_ =
      reinterpret_cast<std::int64_t *>(bytes + layout.destination_row_ptr);
  shared_source_row_ptr_ =
      reinterpret_cast<std::int64_t *>(bytes + layout.source_row_ptr);
  shared_source_order_ =
      reinterpret_cast<std::uint32_t *>(bytes + layout.source_order);
  shared_atom_energy_ =
      reinterpret_cast<double *>(bytes + layout.atom_energy);
  shared_force_ = reinterpret_cast<double *>(bytes + layout.force);
  shared_atom_virial_ =
      reinterpret_cast<double *>(bytes + layout.atom_virial);
  shared_node_capacity_ = node_capacity;
  shared_edge_capacity_ = edge_capacity;
}

void DeepmdPartitionBroker::compute(
    const DeepmdCanonicalGraph &local_graph) {
  result_valid_ = false;
  std::string local_diagnostic;
  const int local_valid = local_graph_valid(local_graph, local_diagnostic) ? 1 : 0;
  int all_valid = 0;
  check_mpi(MPI_Allreduce(&local_valid, &all_valid, 1, MPI_INT, MPI_MIN,
                          communicator_),
            "MPI_Allreduce canonical graph validity");
  if (!all_valid) {
    const int local_rank = local_valid ? size_ : rank_;
    int diagnostic_rank = size_;
    check_mpi(MPI_Allreduce(&local_rank, &diagnostic_rank, 1, MPI_INT, MPI_MIN,
                            communicator_),
              "MPI_Allreduce canonical diagnostic rank");
    if (rank_ != diagnostic_rank)
      local_diagnostic.clear();
    int length = rank_ == diagnostic_rank
                     ? checked_count(local_diagnostic.size(), "diagnostic")
                     : 0;
    check_mpi(MPI_Bcast(&length, 1, MPI_INT, diagnostic_rank, communicator_),
              "MPI_Bcast canonical diagnostic length");
    if (rank_ != diagnostic_rank)
      local_diagnostic.resize(static_cast<std::size_t>(length));
    check_mpi(MPI_Bcast(local_diagnostic.empty() ? nullptr
                                                 : local_diagnostic.data(),
                        length, MPI_CHAR, diagnostic_rank, communicator_),
              "MPI_Bcast canonical diagnostic");
    throw std::invalid_argument(local_diagnostic);
  }

  const std::int64_t local_metadata[3] = {
      static_cast<std::int64_t>(local_graph.atom_types.size()),
      static_cast<std::int64_t>(local_graph.sources.size()),
      local_graph.timestep};
  metadata_records_.resize(static_cast<std::size_t>(size_) * 3);
  check_mpi(MPI_Allgather(local_metadata, 3, MPI_INT64_T,
                          metadata_records_.data(), 3, MPI_INT64_T,
                          communicator_),
            "MPI_Allgather canonical graph metadata");

  node_counts_.resize(size_);
  edge_counts_.resize(size_);
  local_nodes_per_frame_.resize(size_);
  all_nodes_per_frame_.resize(size_);
  int metadata_failed = 0;
  std::string metadata_diagnostic;
  try {
    const std::int64_t timestep = metadata_records_[2];
    for (int frame = 0; frame < size_; ++frame) {
      const std::int64_t nodes = metadata_records_[3 * frame + 0];
      const std::int64_t edges = metadata_records_[3 * frame + 1];
      const std::int64_t frame_timestep = metadata_records_[3 * frame + 2];
      if (nodes <= 0 || edges < 0 || frame_timestep != timestep)
        throw std::invalid_argument(
            "DeePMD partitions reached inconsistent graph extents or "
            "timesteps");
      node_counts_[frame] = checked_count(static_cast<std::size_t>(nodes),
                                          "canonical node count");
      edge_counts_[frame] = checked_count(static_cast<std::size_t>(edges),
                                          "canonical edge count");
      local_nodes_per_frame_[frame] = nodes;
      all_nodes_per_frame_[frame] = nodes;
    }
    build_displacements(node_counts_, node_displacements_, "node");
    build_displacements(edge_counts_, edge_displacements_, "edge");
  } catch (const std::exception &exception) {
    metadata_failed = 1;
    metadata_diagnostic = exception.what();
  }
  int any_metadata_failed = 0;
  check_mpi(MPI_Allreduce(&metadata_failed, &any_metadata_failed, 1, MPI_INT,
                          MPI_MAX, communicator_),
            "MPI_Allreduce canonical metadata status");
  if (any_metadata_failed) {
    if (!metadata_failed)
      metadata_diagnostic.clear();
    const int candidate = metadata_failed ? rank_ : size_;
    int diagnostic_rank = size_;
    check_mpi(MPI_Allreduce(&candidate, &diagnostic_rank, 1, MPI_INT, MPI_MIN,
                            communicator_),
              "MPI_Allreduce metadata diagnostic rank");
    if (rank_ != diagnostic_rank)
      metadata_diagnostic.clear();
    int length = rank_ == diagnostic_rank
                     ? checked_count(metadata_diagnostic.size(), "diagnostic")
                     : 0;
    check_mpi(MPI_Bcast(&length, 1, MPI_INT, diagnostic_rank, communicator_),
              "MPI_Bcast metadata diagnostic length");
    if (rank_ != diagnostic_rank)
      metadata_diagnostic.resize(static_cast<std::size_t>(length));
    check_mpi(MPI_Bcast(metadata_diagnostic.empty()
                            ? nullptr
                            : metadata_diagnostic.data(),
                        length, MPI_CHAR, diagnostic_rank, communicator_),
              "MPI_Bcast metadata diagnostic");
    throw std::invalid_argument(metadata_diagnostic);
  }

  const std::size_t total_nodes =
      total_count(node_counts_, node_displacements_);
  const std::size_t total_edges =
      total_count(edge_counts_, edge_displacements_);
  if (total_nodes > std::numeric_limits<std::uint32_t>::max() ||
      total_edges > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("DeePMD batch exceeds uint32 graph index range");
  const std::size_t edge_storage = std::max<std::size_t>(total_edges, 2);
  ensure_shared_capacity(total_nodes, edge_storage);

  const std::size_t local_node_offset =
      static_cast<std::size_t>(node_displacements_[rank_]);
  const std::size_t local_edge_offset =
      static_cast<std::size_t>(edge_displacements_[rank_]);
  const std::size_t local_nodes =
      static_cast<std::size_t>(node_counts_[rank_]);
  std::copy(local_graph.atom_types.begin(), local_graph.atom_types.end(),
            shared_atom_types_ + local_node_offset);
  std::copy(local_graph.sources.begin(), local_graph.sources.end(),
            shared_sources_ + local_edge_offset);
  std::copy(local_graph.edge_vectors.begin(), local_graph.edge_vectors.end(),
            shared_edge_vectors_ + 3 * local_edge_offset);
  // The terminal CSR entry for the concatenated graph is written once by the
  // owner. Each rank publishes only its node rows into its disjoint slot.
  std::copy_n(local_graph.destination_row_ptr.begin(), local_nodes,
              shared_destination_row_ptr_ + local_node_offset);
  std::copy_n(local_graph.source_row_ptr.begin(), local_nodes,
              shared_source_row_ptr_ + local_node_offset);
  std::copy(local_graph.source_order.begin(), local_graph.source_order.end(),
            shared_source_order_ + local_edge_offset);

  // Direct stores into an MPI shared window require a local sync before the
  // rendezvous and another sync before rank zero observes peer mappings.
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync DeePMD input publication");
  check_mpi(MPI_Barrier(shared_communicator_),
            "MPI_Barrier DeePMD input publication");
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync DeePMD input observation");

  int owner_failed = 0;
  std::string owner_diagnostic;
  if (rank_ == kOwner) {
    try {
      for (int frame = 0; frame < size_; ++frame) {
        const std::int64_t node_offset = node_displacements_[frame];
        const std::int64_t edge_offset = edge_displacements_[frame];
        const std::int64_t nodes = node_counts_[frame];
        const std::int64_t edges = edge_counts_[frame];
        for (std::int64_t node = 0; node < nodes; ++node) {
          const std::size_t index =
              static_cast<std::size_t>(node_offset + node);
          shared_destination_row_ptr_[index] += edge_offset;
          shared_source_row_ptr_[index] += edge_offset;
        }
        for (std::int64_t edge = 0; edge < edges; ++edge) {
          const std::size_t index =
              static_cast<std::size_t>(edge_offset + edge);
          shared_sources_[index] += static_cast<std::uint32_t>(node_offset);
          shared_source_order_[index] +=
              static_cast<std::uint32_t>(edge_offset);
        }
      }
      shared_destination_row_ptr_[total_nodes] =
          static_cast<std::int64_t>(total_edges);
      shared_source_row_ptr_[total_nodes] =
          static_cast<std::int64_t>(total_edges);
      for (std::size_t edge = total_edges; edge < edge_storage; ++edge) {
        shared_sources_[edge] = 0;
        shared_edge_vectors_[3 * edge + 0] = 0.0f;
        shared_edge_vectors_[3 * edge + 1] = 0.0f;
        shared_edge_vectors_[3 * edge + 2] = 0.0f;
        shared_source_order_[edge] = static_cast<std::uint32_t>(edge);
      }

      const DeepmdCanonicalBatchView batch{
          shared_atom_types_,
          shared_sources_,
          shared_edge_vectors_,
          shared_destination_row_ptr_,
          shared_source_row_ptr_,
          shared_source_order_,
          local_nodes_per_frame_.data(),
          all_nodes_per_frame_.data(),
          total_nodes,
          static_cast<std::int64_t>(total_edges),
          static_cast<std::int64_t>(edge_storage),
          size_};
      executor_->compute(batch, batch_result_);
      if (batch_result_.atom_energy.size() != total_nodes ||
          batch_result_.force.size() != 3 * total_nodes ||
          batch_result_.atom_virial.size() != 9 * total_nodes) {
        throw std::runtime_error(
            "DeePMD batch returned inconsistent result extents");
      }
      std::copy(batch_result_.atom_energy.begin(),
                batch_result_.atom_energy.end(), shared_atom_energy_);
      std::copy(batch_result_.force.begin(), batch_result_.force.end(),
                shared_force_);
      std::copy(batch_result_.atom_virial.begin(),
                batch_result_.atom_virial.end(), shared_atom_virial_);
    } catch (const std::exception &exception) {
      owner_failed = 1;
      owner_diagnostic = exception.what();
    }
  }
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync DeePMD output publication");
  check_mpi(MPI_Bcast(&owner_failed, 1, MPI_INT, kOwner, communicator_),
            "MPI_Bcast DeePMD batch status");
  check_mpi(MPI_Win_sync(shared_data_window_),
            "MPI_Win_sync DeePMD output observation");
  if (owner_failed) {
    broadcast_string(communicator_, rank_, owner_diagnostic);
    throw std::runtime_error(owner_diagnostic);
  }

  result_valid_ = true;
}

DeepmdWindowResultView
DeepmdPartitionBroker::result_for_local_window() const {
  if (!result_valid_)
    throw std::logic_error("DeePMD broker result is not available");
  const std::size_t node_offset =
      static_cast<std::size_t>(node_displacements_[rank_]);
  const std::size_t nodes = static_cast<std::size_t>(node_counts_[rank_]);
  return {shared_atom_energy_ + node_offset,
          shared_force_ + 3 * node_offset,
          shared_atom_virial_ + 9 * node_offset, nodes};
}

}  // namespace DPRC
