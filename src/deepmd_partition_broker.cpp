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

// MPI_ERRORS_RETURN reports a failed collective only to the rank that
// observed it. Reduce the status before any rank can advance to the next
// collective, otherwise one rank can throw while its peers continue with a
// different operation and deadlock.
void check_collective_mpi(int status, MPI_Comm communicator,
                          const char *operation) {
  const int local_failed = status == MPI_SUCCESS ? 0 : 1;
  int any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                    communicator) != MPI_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             " status reduction failed");
  if (any_failed != 0)
    throw std::runtime_error(std::string(operation) + " failed");
}

int checked_count(std::size_t count, const char *label) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error(std::string(label) + " exceeds MPI int count");
  return static_cast<int>(count);
}

int checked_scale(int count, int factor, const char *label) {
  if (count < 0 || factor < 0 ||
      (factor != 0 && count > std::numeric_limits<int>::max() / factor))
    throw std::overflow_error(std::string(label) + " exceeds MPI int count");
  return count * factor;
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

void broadcast_string_from(MPI_Comm communicator, int rank, int root,
                           std::string &value) {
  int length = 0;
  int root_failed = 0;
  if (rank == root) {
    try {
      length = checked_count(value.size(), "string length");
    } catch (...) {
      // The diagnostic itself is not allowed to break the status rendezvous.
      // All ranks receive the failure flag and unwind together.
      root_failed = 1;
    }
  }
  check_mpi(MPI_Bcast(&root_failed, 1, MPI_INT, root, communicator),
            "MPI_Bcast string status");
  if (root_failed != 0)
    throw std::runtime_error("diagnostic string exceeds MPI int count");
  check_mpi(MPI_Bcast(&length, 1, MPI_INT, root, communicator),
            "MPI_Bcast string length");

  int local_resize_failed = 0;
  if (rank != root) {
    try {
      value.resize(static_cast<std::size_t>(length));
    } catch (...) {
      local_resize_failed = 1;
    }
  }
  int any_resize_failed = 0;
  if (MPI_Allreduce(&local_resize_failed, &any_resize_failed, 1, MPI_INT,
                    MPI_MAX, communicator) != MPI_SUCCESS)
    throw std::runtime_error("MPI_Allreduce diagnostic allocation status "
                             "failed");
  if (any_resize_failed != 0)
    throw std::runtime_error("could not allocate a diagnostic string on all "
                             "MPI ranks");

  check_mpi(MPI_Bcast(value.empty() ? nullptr : value.data(), length, MPI_CHAR,
                      root, communicator),
            "MPI_Bcast string payload");
}

void broadcast_string(MPI_Comm communicator, int rank, std::string &value) {
  broadcast_string_from(communicator, rank, kOwner, value);
}

// Synchronize local preparation failures before entering the next collective.
// This covers host-vector allocation and result-buffer allocation, both of
// which can otherwise make only one rank leave the broker call.
void throw_if_collective_failure(MPI_Comm communicator, int rank, int size,
                                 int local_failed, std::string diagnostic,
                                 const char *operation) {
  int any_failed = 0;
  check_mpi(MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                          communicator),
            operation);
  if (any_failed == 0)
    return;

  const int candidate = local_failed != 0 ? rank : size;
  int diagnostic_rank = size;
  check_mpi(MPI_Allreduce(&candidate, &diagnostic_rank, 1, MPI_INT, MPI_MIN,
                          communicator),
            "MPI_Allreduce collective diagnostic rank");
  if (rank != diagnostic_rank)
    diagnostic.clear();
  broadcast_string_from(communicator, rank, diagnostic_rank, diagnostic);
  if (diagnostic.empty())
    diagnostic = std::string(operation) + " failed";
  throw std::runtime_error(diagnostic);
}

template <typename T>
T *data_or_null(std::vector<T> &values) {
  return values.empty() ? nullptr : values.data();
}

template <typename T>
const T *data_or_null(const std::vector<T> &values) {
  return values.empty() ? nullptr : values.data();
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
    int shared_size = 0;
    check_mpi(MPI_Comm_size(shared_communicator_, &shared_size),
              "MPI_Comm_size shared");
    if (shared_size != size_)
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
  for (std::size_t node = 1; node <= nodes; ++node) {
    if (graph.destination_row_ptr[node] <
            graph.destination_row_ptr[node - 1] ||
        graph.source_row_ptr[node] < graph.source_row_ptr[node - 1]) {
      diagnostic = "canonical graph CSR row pointers are not monotonic";
      return false;
    }
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
      if (edge >= edges || source_order_seen[edge] ||
          graph.sources[edge] != source) {
        diagnostic =
            "canonical source order is not a source-grouped permutation";
        return false;
      }
      source_order_seen[edge] = 1;
    }
  }
  if (std::find(source_order_seen.begin(), source_order_seen.end(), 0) !=
      source_order_seen.end()) {
    diagnostic = "canonical source order is not a complete permutation";
    return false;
  }
  return true;
}

void DeepmdPartitionBroker::ensure_host_capacity(std::size_t nodes,
                                                 std::size_t edge_storage) {
  if (rank_ != kOwner)
    return;
  batch_atom_types_.resize(nodes);
  batch_sources_.resize(edge_storage);
  batch_edge_vectors_.resize(3 * edge_storage);
  batch_destination_row_ptr_.resize(nodes + 1);
  batch_source_row_ptr_.resize(nodes + 1);
  batch_source_order_.resize(edge_storage);
}

void DeepmdPartitionBroker::compute(
    const DeepmdCanonicalGraph &local_graph) {
  result_valid_ = false;
  std::string local_diagnostic;
  int local_valid = 0;
  try {
    local_valid = local_graph_valid(local_graph, local_diagnostic) ? 1 : 0;
  } catch (const std::exception &exception) {
    local_diagnostic = exception.what();
  } catch (...) {
    local_diagnostic = "unknown exception while validating canonical graph";
  }
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
    broadcast_string_from(communicator_, rank_, diagnostic_rank,
                          local_diagnostic);
    throw std::invalid_argument(local_diagnostic);
  }

  const std::int64_t local_metadata[3] = {
      static_cast<std::int64_t>(local_graph.atom_types.size()),
      static_cast<std::int64_t>(local_graph.sources.size()),
      local_graph.timestep};
  int metadata_storage_failed = 0;
  std::string metadata_storage_diagnostic;
  try {
    metadata_records_.resize(static_cast<std::size_t>(size_) * 3);
    node_counts_.resize(size_);
    edge_counts_.resize(size_);
    local_nodes_per_frame_.resize(size_);
    all_nodes_per_frame_.resize(size_);
  } catch (const std::exception &exception) {
    metadata_storage_failed = 1;
    metadata_storage_diagnostic = exception.what();
  } catch (...) {
    metadata_storage_failed = 1;
    metadata_storage_diagnostic =
        "unknown exception while allocating DeePMD metadata";
  }
  throw_if_collective_failure(
      communicator_, rank_, size_, metadata_storage_failed,
      std::move(metadata_storage_diagnostic),
      "DeePMD metadata allocation status");
  check_mpi(MPI_Allgather(local_metadata, 3, MPI_INT64_T,
                          metadata_records_.data(), 3, MPI_INT64_T,
                          communicator_),
            "MPI_Allgather canonical graph metadata");

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
  } catch (...) {
    metadata_failed = 1;
    metadata_diagnostic =
        "unknown exception while validating DeePMD graph metadata";
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
    broadcast_string_from(communicator_, rank_, diagnostic_rank,
                          metadata_diagnostic);
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
  std::vector<int> edge_vector_counts;
  std::vector<int> edge_vector_displacements;
  int staging_failed = 0;
  std::string staging_diagnostic;
  try {
    ensure_host_capacity(total_nodes, edge_storage);
    edge_vector_counts.resize(size_);
    edge_vector_displacements.resize(size_);
    for (int frame = 0; frame < size_; ++frame)
      edge_vector_counts[frame] =
          checked_scale(edge_counts_[frame], 3, "edge vector count");
    build_displacements(edge_vector_counts, edge_vector_displacements,
                        "edge vector");
  } catch (const std::exception &exception) {
    staging_failed = 1;
    staging_diagnostic = exception.what();
  } catch (...) {
    staging_failed = 1;
    staging_diagnostic =
        "unknown exception while allocating DeePMD batch staging";
  }
  throw_if_collective_failure(
      communicator_, rank_, size_, staging_failed, std::move(staging_diagnostic),
      "DeePMD batch staging allocation status");

  check_collective_mpi(
      MPI_Gatherv(data_or_null(local_graph.atom_types), node_counts_[rank_],
                  MPI_INT64_T, data_or_null(batch_atom_types_),
                  node_counts_.data(), node_displacements_.data(), MPI_INT64_T,
                  kOwner, communicator_),
      communicator_, "MPI_Gatherv DeePMD atom types");
  check_collective_mpi(
      MPI_Gatherv(data_or_null(local_graph.sources), edge_counts_[rank_],
                  MPI_UINT32_T, data_or_null(batch_sources_),
                  edge_counts_.data(), edge_displacements_.data(),
                  MPI_UINT32_T, kOwner, communicator_),
      communicator_, "MPI_Gatherv DeePMD sources");
  check_collective_mpi(
      MPI_Gatherv(data_or_null(local_graph.edge_vectors),
                  edge_vector_counts[rank_], MPI_FLOAT,
                  data_or_null(batch_edge_vectors_),
                  edge_vector_counts.data(), edge_vector_displacements.data(),
                  MPI_FLOAT, kOwner, communicator_),
      communicator_, "MPI_Gatherv DeePMD edge vectors");
  check_collective_mpi(
      MPI_Gatherv(data_or_null(local_graph.destination_row_ptr),
                  node_counts_[rank_], MPI_INT64_T,
                  data_or_null(batch_destination_row_ptr_),
                  node_counts_.data(), node_displacements_.data(), MPI_INT64_T,
                  kOwner, communicator_),
      communicator_, "MPI_Gatherv DeePMD destination CSR");
  check_collective_mpi(
      MPI_Gatherv(data_or_null(local_graph.source_row_ptr),
                  node_counts_[rank_], MPI_INT64_T,
                  data_or_null(batch_source_row_ptr_), node_counts_.data(),
                  node_displacements_.data(), MPI_INT64_T, kOwner,
                  communicator_),
      communicator_, "MPI_Gatherv DeePMD source CSR");
  check_collective_mpi(
      MPI_Gatherv(data_or_null(local_graph.source_order), edge_counts_[rank_],
                  MPI_UINT32_T, data_or_null(batch_source_order_),
                  edge_counts_.data(), edge_displacements_.data(),
                  MPI_UINT32_T, kOwner, communicator_),
      communicator_, "MPI_Gatherv DeePMD source order");

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
          batch_destination_row_ptr_[index] += edge_offset;
          batch_source_row_ptr_[index] += edge_offset;
        }
        for (std::int64_t edge = 0; edge < edges; ++edge) {
          const std::size_t index =
              static_cast<std::size_t>(edge_offset + edge);
          batch_sources_[index] += static_cast<std::uint32_t>(node_offset);
          batch_source_order_[index] +=
              static_cast<std::uint32_t>(edge_offset);
        }
      }
      batch_destination_row_ptr_[total_nodes] =
          static_cast<std::int64_t>(total_edges);
      batch_source_row_ptr_[total_nodes] =
          static_cast<std::int64_t>(total_edges);
      for (std::size_t edge = total_edges; edge < edge_storage; ++edge) {
        batch_sources_[edge] = 0;
        batch_edge_vectors_[3 * edge + 0] = 0.0f;
        batch_edge_vectors_[3 * edge + 1] = 0.0f;
        batch_edge_vectors_[3 * edge + 2] = 0.0f;
        batch_source_order_[edge] = static_cast<std::uint32_t>(edge);
      }

      const DeepmdCanonicalBatchView batch{
          batch_atom_types_.data(),
          batch_sources_.data(),
          batch_edge_vectors_.data(),
          batch_destination_row_ptr_.data(),
          batch_source_row_ptr_.data(),
          batch_source_order_.data(),
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
    } catch (const std::exception &exception) {
      owner_failed = 1;
      owner_diagnostic = exception.what();
    } catch (...) {
      owner_failed = 1;
      owner_diagnostic = "unknown exception during DeePMD batch execution";
    }
  }
  check_mpi(MPI_Bcast(&owner_failed, 1, MPI_INT, kOwner, communicator_),
            "MPI_Bcast DeePMD batch status");
  if (owner_failed) {
    broadcast_string(communicator_, rank_, owner_diagnostic);
    throw std::runtime_error(owner_diagnostic);
  }

  int result_storage_failed = 0;
  std::string result_storage_diagnostic;
  try {
    force_counts_.resize(size_);
    virial_counts_.resize(size_);
    for (int frame = 0; frame < size_; ++frame) {
      force_counts_[frame] =
          checked_scale(node_counts_[frame], 3, "force count");
      virial_counts_[frame] =
          checked_scale(node_counts_[frame], 9, "virial count");
    }
    build_displacements(force_counts_, force_displacements_, "force");
    build_displacements(virial_counts_, virial_displacements_, "virial");
    local_atom_energy_.resize(static_cast<std::size_t>(node_counts_[rank_]));
    local_force_.resize(static_cast<std::size_t>(force_counts_[rank_]));
    local_atom_virial_.resize(
        static_cast<std::size_t>(virial_counts_[rank_]));
  } catch (const std::exception &exception) {
    result_storage_failed = 1;
    result_storage_diagnostic = exception.what();
  } catch (...) {
    result_storage_failed = 1;
    result_storage_diagnostic =
        "unknown exception while allocating DeePMD result buffers";
  }
  if (rank_ == kOwner && !result_storage_failed &&
      (batch_result_.atom_energy.size() != total_nodes ||
       batch_result_.force.size() != 3 * total_nodes ||
       batch_result_.atom_virial.size() != 9 * total_nodes)) {
    result_storage_failed = 1;
    result_storage_diagnostic =
        "DeePMD batch returned inconsistent result extents";
  }
  throw_if_collective_failure(
      communicator_, rank_, size_, result_storage_failed,
      std::move(result_storage_diagnostic),
      "DeePMD result allocation status");

  check_collective_mpi(
      MPI_Scatterv(data_or_null(batch_result_.atom_energy),
                   node_counts_.data(), node_displacements_.data(), MPI_DOUBLE,
                   data_or_null(local_atom_energy_), node_counts_[rank_],
                   MPI_DOUBLE, kOwner, communicator_),
      communicator_, "MPI_Scatterv DeePMD atomic energy");
  check_collective_mpi(
      MPI_Scatterv(data_or_null(batch_result_.force), force_counts_.data(),
                   force_displacements_.data(), MPI_DOUBLE,
                   data_or_null(local_force_), force_counts_[rank_], MPI_DOUBLE,
                   kOwner, communicator_),
      communicator_, "MPI_Scatterv DeePMD force");
  check_collective_mpi(
      MPI_Scatterv(data_or_null(batch_result_.atom_virial),
                   virial_counts_.data(), virial_displacements_.data(),
                   MPI_DOUBLE, data_or_null(local_atom_virial_),
                   virial_counts_[rank_], MPI_DOUBLE, kOwner, communicator_),
      communicator_, "MPI_Scatterv DeePMD atomic virial");
  result_valid_ = true;
}

DeepmdWindowResultView
DeepmdPartitionBroker::result_for_local_window() const {
  if (!result_valid_)
    throw std::logic_error("DeePMD broker result is not available");
  return {data_or_null(local_atom_energy_), data_or_null(local_force_),
          data_or_null(local_atom_virial_), local_atom_energy_.size()};
}

}  // namespace DPRC
