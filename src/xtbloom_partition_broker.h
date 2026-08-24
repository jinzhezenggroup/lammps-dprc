#ifndef LAMMPS_DPRC_XTBLOOM_PARTITION_BROKER_H
#define LAMMPS_DPRC_XTBLOOM_PARTITION_BROKER_H

#include "xtbloom_plan_executor.h"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DPRC {

// Outcome shared by every partition root after one collective broker call.
// A SUCCESS call may still contain a peer-local numerical failure, exposed by
// result_for_local_window().status without invalidating successful peers.
struct PartitionBrokerOutcome {
  xtbloom_status_t call_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::int64_t timestep = -1;
  SccStartPolicy start_policy = SccStartPolicy::Fresh;
  std::uint32_t result_flags = 0;
  bool all_systems_succeeded = false;
};

// Collective root-to-root owner for one synchronized all-window xTBloom plan.
// Only roots rank zero creates the native context and executes the ragged
// batch. All roots on the GPU-local node exchange steady-state frame and result
// slices through one shared mapping; one-time immutable topology construction
// may still use ordinary collectives. The communicator rank is the permanent
// window/plan slot and must be dense in [0, size).
class XtbloomPartitionBroker {
public:
  XtbloomPartitionBroker(MPI_Comm roots, WindowTopology local_topology,
                         const XtbloomExecutorOptions &options = {});
  ~XtbloomPartitionBroker();

  XtbloomPartitionBroker(const XtbloomPartitionBroker &) = delete;
  XtbloomPartitionBroker &operator=(const XtbloomPartitionBroker &) = delete;

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  bool owns_executor() const noexcept { return executor_ != nullptr; }
  bool uses_shared_storage() const noexcept {
    return shared_data_window_ != MPI_WIN_NULL &&
           shared_metadata_window_ != MPI_WIN_NULL;
  }
  bool has_result() const noexcept { return result_valid_; }
  const std::string &last_error() const noexcept { return last_error_; }

  PartitionBrokerOutcome compute(const WindowFrame &local_frame);
  WindowResultView result_for_local_window() const;

private:
  void initialize_collective(const XtbloomExecutorOptions &options);
  void initialize_shared_storage();
  void release_shared_storage() noexcept;
  bool local_frame_is_valid(const WindowFrame &frame) const noexcept;
  void stage_shared_frame(const WindowFrame &frame);
  bool publish_success(const XtbloomComputeOutcome &outcome);
  void broadcast_error();

  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = -1;
  int size_ = 0;
  WindowTopology local_topology_;

  std::unique_ptr<XtbloomPlanExecutor> executor_;
  std::vector<WindowTopology> topologies_;

  std::vector<int> atom_counts_;
  std::vector<int> atom_displacements_;
  std::vector<int> position_counts_;
  std::vector<int> position_displacements_;
  std::vector<int> point_counts_;
  std::vector<int> point_displacements_;
  std::vector<int> point_position_counts_;
  std::vector<int> point_position_displacements_;
  std::vector<int> shift_counts_;
  std::vector<int> shift_displacements_;
  std::vector<int> response_counts_;
  std::vector<int> response_displacements_;

  // Steady-state frame and result payloads live in one root-allocated shared
  // mapping. Each partition root writes and reads only its stable ragged slot;
  // broker rank zero stages directly from the complete mapping and publishes
  // completed xTBloom slices back into it. This removes large per-step
  // Gather/Scatter collectives while retaining one ordinary LAMMPS process per
  // umbrella window.
  struct SharedResultMetadata {
    double energy = 0.0;
    std::int32_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
    std::int32_t scc_iterations = 0;
    std::uint8_t scc_converged = 0;
    std::uint8_t reserved[7]{};
  };

  MPI_Comm shared_communicator_ = MPI_COMM_NULL;
  MPI_Win shared_data_window_ = MPI_WIN_NULL;
  MPI_Win shared_metadata_window_ = MPI_WIN_NULL;
  bool shared_data_locked_ = false;
  bool shared_metadata_locked_ = false;
  double *shared_data_base_ = nullptr;
  SharedResultMetadata *shared_metadata_ = nullptr;
  double *shared_positions_ = nullptr;
  double *shared_point_positions_ = nullptr;
  double *shared_point_values_ = nullptr;
  double *shared_potential_shifts_ = nullptr;
  double *shared_response_matrices_ = nullptr;
  double *shared_forces_ = nullptr;
  double *shared_atomic_charges_ = nullptr;
  double *shared_point_charge_forces_ = nullptr;
  std::vector<std::int64_t> preflight_records_;

  std::int64_t result_timestep_ = -1;
  bool result_valid_ = false;
  std::string last_error_;
};

} // namespace DPRC

#endif
