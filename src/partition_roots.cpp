#include "partition_roots.h"

#include <stdexcept>

namespace DPRC {

PartitionRoots::PartitionRoots(MPI_Comm universe, int world_rank,
                               int world_index, int world_count)
    : world_index_(world_index) {
  if (universe == MPI_COMM_NULL || world_count <= 0)
    throw std::invalid_argument("invalid LAMMPS partition topology");

  const int local_input_valid =
      world_rank >= 0 && world_index >= 0 && world_index < world_count ? 1 : 0;
  int global_input_valid = 0;
  if (MPI_Allreduce(&local_input_valid, &global_input_valid, 1, MPI_INT,
                    MPI_MIN, universe) != MPI_SUCCESS ||
      global_input_valid == 0) {
    throw std::invalid_argument("invalid LAMMPS partition topology");
  }

  // Every universe rank participates. Non-root ranks receive MPI_COMM_NULL;
  // using iworld as the key fixes root-rank ordering independently of any
  // Universe::uworld rank reordering selected on the LAMMPS command line.
  const int color = world_rank == 0 ? 0 : MPI_UNDEFINED;
  if (MPI_Comm_split(universe, color, world_index, &communicator_) !=
      MPI_SUCCESS) {
    communicator_ = MPI_COMM_NULL;
    throw std::runtime_error(
        "could not create DPRc partition-root communicator");
  }

  int local_valid = 1;
  if (communicator_ != MPI_COMM_NULL) {
    if (MPI_Comm_rank(communicator_, &rank_) != MPI_SUCCESS ||
        MPI_Comm_size(communicator_, &size_) != MPI_SUCCESS) {
      local_valid = 0;
    } else if (rank_ != world_index || size_ != world_count) {
      local_valid = 0;
    }
  }

  int globally_valid = 0;
  if (MPI_Allreduce(&local_valid, &globally_valid, 1, MPI_INT, MPI_MIN,
                    universe) != MPI_SUCCESS) {
    globally_valid = 0;
  }
  if (globally_valid == 0) {
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
    communicator_ = MPI_COMM_NULL;
    throw std::runtime_error(
        "DPRc partition roots do not form the expected stable slot order");
  }
}

PartitionRoots::~PartitionRoots() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && communicator_ != MPI_COMM_NULL)
    MPI_Comm_free(&communicator_);
}

} // namespace DPRC
