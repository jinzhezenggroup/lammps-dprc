#ifndef LAMMPS_DPRC_PARTITION_ROOTS_H
#define LAMMPS_DPRC_PARTITION_ROOTS_H

#include <mpi.h>

namespace DPRC {

// Collective bridge from LAMMPS's universe communicator to one rank per
// partition. The root communicator is ordered by Universe::iworld, making its
// rank the stable xTBloom batch slot for the synchronized all-window plan.
class PartitionRoots {
public:
  PartitionRoots(MPI_Comm universe, int world_rank, int world_index,
                 int world_count);
  ~PartitionRoots();

  PartitionRoots(const PartitionRoots &) = delete;
  PartitionRoots &operator=(const PartitionRoots &) = delete;

  bool is_root() const noexcept { return communicator_ != MPI_COMM_NULL; }
  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  int stable_slot() const noexcept { return world_index_; }
  MPI_Comm communicator() const noexcept { return communicator_; }

private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = -1;
  int size_ = 0;
  int world_index_ = -1;
};

} // namespace DPRC

#endif
