#include "deepmd_partition_broker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  if (graph.destination_row_ptr.front() != 0 ||
      graph.source_row_ptr.front() != 0 ||
      graph.destination_row_ptr.back() != static_cast<std::int64_t>(edges) ||
      graph.source_row_ptr.back() != static_cast<std::int64_t>(edges)) {
    diagnostic = "canonical graph CSR terminal entries are invalid";
    return false;
  }
  for (std::size_t node = 0; node < nodes; ++node) {
    if (graph.atom_types[node] < 0 ||
        graph.atom_types[node] >= metadata_.type_count ||
        graph.destination_row_ptr[node] >
            graph.destination_row_ptr[node + 1] ||
        graph.source_row_ptr[node] > graph.source_row_ptr[node + 1]) {
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
  ensure_host_capacity(total_nodes, edge_storage);

  std::vector<int> edge_vector_counts(size_);
  std::vector<int> edge_vector_displacements(size_);
  for (int frame = 0; frame < size_; ++frame)
    edge_vector_counts[frame] =
        checked_scale(edge_counts_[frame], 3, "edge vector count");
  build_displacements(edge_vector_counts, edge_vector_displacements,
                      "edge vector");

  check_mpi(MPI_Gatherv(data_or_null(local_graph.atom_types),
                        node_counts_[rank_], MPI_INT64_T,
                        data_or_null(batch_atom_types_), node_counts_.data(),
                        node_displacements_.data(), MPI_INT64_T, kOwner,
                        communicator_),
            "MPI_Gatherv DeePMD atom types");
  check_mpi(MPI_Gatherv(data_or_null(local_graph.sources), edge_counts_[rank_],
                        MPI_UINT32_T, data_or_null(batch_sources_),
                        edge_counts_.data(), edge_displacements_.data(),
                        MPI_UINT32_T, kOwner, communicator_),
            "MPI_Gatherv DeePMD sources");
  check_mpi(MPI_Gatherv(data_or_null(local_graph.edge_vectors),
                        edge_vector_counts[rank_], MPI_FLOAT,
                        data_or_null(batch_edge_vectors_),
                        edge_vector_counts.data(),
                        edge_vector_displacements.data(), MPI_FLOAT, kOwner,
                        communicator_),
            "MPI_Gatherv DeePMD edge vectors");
  check_mpi(MPI_Gatherv(data_or_null(local_graph.destination_row_ptr),
                        node_counts_[rank_], MPI_INT64_T,
                        data_or_null(batch_destination_row_ptr_),
                        node_counts_.data(), node_displacements_.data(),
                        MPI_INT64_T, kOwner, communicator_),
            "MPI_Gatherv DeePMD destination CSR");
  check_mpi(MPI_Gatherv(data_or_null(local_graph.source_row_ptr),
                        node_counts_[rank_], MPI_INT64_T,
                        data_or_null(batch_source_row_ptr_),
                        node_counts_.data(), node_displacements_.data(),
                        MPI_INT64_T, kOwner, communicator_),
            "MPI_Gatherv DeePMD source CSR");
  check_mpi(MPI_Gatherv(data_or_null(local_graph.source_order),
                        edge_counts_[rank_], MPI_UINT32_T,
                        data_or_null(batch_source_order_), edge_counts_.data(),
                        edge_displacements_.data(), MPI_UINT32_T, kOwner,
                        communicator_),
            "MPI_Gatherv DeePMD source order");

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
    } catch (const std::exception &exception) {
      owner_failed = 1;
      owner_diagnostic = exception.what();
    }
  }
  check_mpi(MPI_Bcast(&owner_failed, 1, MPI_INT, kOwner, communicator_),
            "MPI_Bcast DeePMD batch status");
  if (owner_failed) {
    broadcast_string(communicator_, rank_, owner_diagnostic);
    throw std::runtime_error(owner_diagnostic);
  }

  force_counts_.resize(size_);
  virial_counts_.resize(size_);
  for (int frame = 0; frame < size_; ++frame) {
    force_counts_[frame] = checked_scale(node_counts_[frame], 3, "force count");
    virial_counts_[frame] =
        checked_scale(node_counts_[frame], 9, "virial count");
  }
  build_displacements(force_counts_, force_displacements_, "force");
  build_displacements(virial_counts_, virial_displacements_, "virial");
  local_atom_energy_.resize(static_cast<std::size_t>(node_counts_[rank_]));
  local_force_.resize(static_cast<std::size_t>(force_counts_[rank_]));
  local_atom_virial_.resize(static_cast<std::size_t>(virial_counts_[rank_]));

  check_mpi(MPI_Scatterv(data_or_null(batch_result_.atom_energy),
                         node_counts_.data(), node_displacements_.data(),
                         MPI_DOUBLE, data_or_null(local_atom_energy_),
                         node_counts_[rank_], MPI_DOUBLE, kOwner,
                         communicator_),
            "MPI_Scatterv DeePMD atomic energy");
  check_mpi(MPI_Scatterv(data_or_null(batch_result_.force),
                         force_counts_.data(), force_displacements_.data(),
                         MPI_DOUBLE, data_or_null(local_force_),
                         force_counts_[rank_], MPI_DOUBLE, kOwner,
                         communicator_),
            "MPI_Scatterv DeePMD force");
  check_mpi(MPI_Scatterv(data_or_null(batch_result_.atom_virial),
                         virial_counts_.data(), virial_displacements_.data(),
                         MPI_DOUBLE, data_or_null(local_atom_virial_),
                         virial_counts_[rank_], MPI_DOUBLE, kOwner,
                         communicator_),
            "MPI_Scatterv DeePMD atomic virial");
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
