#ifndef LAMMPS_DPRC_DEEPMD_PARTITION_BROKER_H
#define LAMMPS_DPRC_DEEPMD_PARTITION_BROKER_H

#include "deepmd_batch_executor.h"

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DPRC {

// One folded, single-rank compact graph contributed by a LAMMPS partition.
// Row-pointer arrays include their terminal entry and all indices are local to
// this graph.  The broker validates and offsets them before the model sees the
// concatenated block-diagonal batch.
struct DeepmdCanonicalGraph {
  std::int64_t timestep = -1;
  std::vector<std::int64_t> atom_types;
  std::vector<std::uint32_t> sources;
  std::vector<float> edge_vectors;
  std::vector<std::int64_t> destination_row_ptr;
  std::vector<std::int64_t> source_row_ptr;
  std::vector<std::uint32_t> source_order;
};

struct DeepmdWindowResultView {
  const double *atom_energy = nullptr;
  const double *force = nullptr;
  const double *atom_virial = nullptr;
  std::size_t node_count = 0;
};

// Collective root-to-root broker for one synchronized all-window DeePMD call.
// Communicator rank is the stable replica slot.  Rank zero alone loads the
// model and owns its CUDA context; every other rank contributes a host compact
// graph and receives its result slice only after the complete batch succeeds.
class DeepmdPartitionBroker {
 public:
  DeepmdPartitionBroker(MPI_Comm roots, std::string model_path, int gpu_rank);
  ~DeepmdPartitionBroker();

  DeepmdPartitionBroker(const DeepmdPartitionBroker &) = delete;
  DeepmdPartitionBroker &operator=(const DeepmdPartitionBroker &) = delete;

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  bool owns_executor() const noexcept { return executor_ != nullptr; }
  const DeepmdModelMetadata &metadata() const noexcept { return metadata_; }

  void compute(const DeepmdCanonicalGraph &local_graph);
  DeepmdWindowResultView result_for_local_window() const;

 private:
  void initialize_model(const std::string &model_path, int gpu_rank);
  void ensure_host_capacity(std::size_t nodes, std::size_t edge_storage);
  bool local_graph_valid(const DeepmdCanonicalGraph &graph,
                         std::string &diagnostic) const;

  MPI_Comm communicator_ = MPI_COMM_NULL;
  MPI_Comm shared_communicator_ = MPI_COMM_NULL;
  int rank_ = -1;
  int size_ = 0;
  std::unique_ptr<DeepmdBatchExecutor> executor_;
  DeepmdModelMetadata metadata_;

  std::vector<std::int64_t> metadata_records_;
  std::vector<int> node_counts_;
  std::vector<int> edge_counts_;
  std::vector<int> node_displacements_;
  std::vector<int> edge_displacements_;
  std::vector<int> force_counts_;
  std::vector<int> force_displacements_;
  std::vector<int> virial_counts_;
  std::vector<int> virial_displacements_;
  std::vector<std::int64_t> local_nodes_per_frame_;
  std::vector<std::int64_t> all_nodes_per_frame_;

  std::vector<std::int64_t> batch_atom_types_;
  std::vector<std::uint32_t> batch_sources_;
  std::vector<float> batch_edge_vectors_;
  std::vector<std::int64_t> batch_destination_row_ptr_;
  std::vector<std::int64_t> batch_source_row_ptr_;
  std::vector<std::uint32_t> batch_source_order_;
  DeepmdCanonicalBatchResult batch_result_;

  std::vector<double> local_atom_energy_;
  std::vector<double> local_force_;
  std::vector<double> local_atom_virial_;
  bool result_valid_ = false;
};

}  // namespace DPRC

#endif
