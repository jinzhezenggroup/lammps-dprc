#ifndef LAMMPS_DPRC_CLASSICAL_PARTITION_BROKER_H
#define LAMMPS_DPRC_CLASSICAL_PARTITION_BROKER_H

#include "classical_batch.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DPRC {

// Borrowed one-window input for the first half of a classical QM/MM epoch.
// Counts are explicit so malformed LAMMPS views are rejected collectively
// before the owner executes or any previously published result is modified.
struct ClassicalMmFrameView {
  std::int64_t timestep = -1;
  const double *positions = nullptr;
  std::size_t position_count = 0;
  const double *mm_charges = nullptr;
  std::size_t charge_count = 0;
};

// Borrowed QM charge contribution for the epoch retained by begin_mm().
struct ClassicalQmFrameView {
  std::int64_t timestep = -1;
  const double *qm_charges = nullptr;
  std::size_t charge_count = 0;
};

// One partition's successfully published real-space and MM reciprocal result.
// The pointers remain valid until the next successful begin_mm() or broker
// destruction.  A failed call and cancel() leave this publication unchanged.
struct ClassicalMmResultView {
  std::int64_t timestep = -1;
  std::size_t atom_count = 0;
  const double *pair_forces = nullptr;
  double lj_energy = 0.0;
  double coulomb_energy = 0.0;
  const double *pair_virial = nullptr;
  double mm_pppm_energy = 0.0;
  const double *mm_pppm_virial = nullptr;
  const double *mm_pppm_potential = nullptr;
  const double *mm_pppm_forces = nullptr;
};

// Select the mutually independent MM reciprocal publications needed by one
// consumer.  QM/MM requests the scalar potential; pure classical LAMMPS
// requests atom forces.  Requesting only one avoids the other's interpolation
// and, on CUDA, avoids its unused inverse-transform batch.
struct ClassicalMmResultRequest {
  bool pppm_potential = true;
  bool pppm_forces = false;
  bool retain_for_qm = true;
};

// One partition's successfully published QM-only and assembled-full reciprocal
// result.  A later successful begin_mm() deliberately preserves this view
// until finish_qm() publishes the replacement, so an incomplete new epoch
// cannot erase the last complete result.
struct ClassicalQmResultView {
  std::int64_t timestep = -1;
  std::size_t atom_count = 0;
  const double *qm_pppm_forces = nullptr;
  const double *full_pppm_forces = nullptr;
  double qm_pppm_energy = 0.0;
  double full_pppm_energy = 0.0;
  const double *qm_pppm_virial = nullptr;
  const double *full_pppm_virial = nullptr;
};

// Collective root-to-root broker for one synchronized classical batch.
//
// Communicator rank is the permanent batch slot.  Every root must provide the
// exact same immutable topology and plan options, but only roots rank zero owns
// ClassicalBatchPlan (and therefore the single CUDA/cuFFT context).  Calls to
// begin_mm() and finish_qm() are collective and must occur in the same order on
// every root.  Inputs are gathered in rank order and outputs are scattered into
// candidate storage before a collective commit, providing strict publication
// transactionality across partitions.
class ClassicalPartitionBroker {
public:
  ClassicalPartitionBroker(MPI_Comm roots, ClassicalTopology local_topology,
                           const ClassicalPlanOptions &options = {});
  ~ClassicalPartitionBroker();

  ClassicalPartitionBroker(const ClassicalPartitionBroker &) = delete;
  ClassicalPartitionBroker &
  operator=(const ClassicalPartitionBroker &) = delete;

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }
  [[nodiscard]] bool owns_plan() const noexcept { return plan_ != nullptr; }
  [[nodiscard]] bool has_active_mm_epoch() const noexcept { return active_mm_; }
  [[nodiscard]] bool has_mm_result() const noexcept { return mm_result_valid_; }
  [[nodiscard]] bool has_qm_result() const noexcept { return qm_result_valid_; }
  [[nodiscard]] const std::string &last_error() const noexcept {
    return last_error_;
  }

  // Starts one synchronized timestep.  A second begin is rejected until the
  // active epoch is consumed by finish_qm() or abandoned with cancel().
  void begin_mm(const ClassicalMmFrameView &local_frame,
                ClassicalMmResultRequest request = {});

  // Completes and consumes the active MM epoch.  The supplied timestep must be
  // identical on every root and equal to the timestep bound by begin_mm().
  void finish_qm(const ClassicalQmFrameView &local_frame);

  // Locally abandons the in-progress epoch and asks the owner plan to release
  // retained workspaces.  Call this on every root to preserve collective state.
  // It is intentionally non-collective and noexcept so surrounding failure
  // cleanup cannot deadlock.  Prior MM and QM publications are not modified.
  void cancel() noexcept;

  [[nodiscard]] ClassicalMmResultView mm_result() const;
  [[nodiscard]] ClassicalQmResultView qm_result() const;

private:
  enum class ErrorKind : int {
    None = 0,
    InvalidArgument = 1,
    Logic = 2,
    Overflow = 3,
    Runtime = 4,
  };

  void initialize_collective(const ClassicalPlanOptions &options);
  void initialize_shared_storage();
  void synchronize_shared_storage();
  void require_collectively(bool local_condition, const char *message);
  void require_matching_bytes(const std::vector<std::uint8_t> &local_bytes,
                              const char *message);
  [[nodiscard]] bool
  mm_frame_is_valid(const ClassicalMmFrameView &frame) const noexcept;
  [[nodiscard]] bool
  qm_frame_is_valid(const ClassicalQmFrameView &frame) const noexcept;
  void broadcast_root_error(ErrorKind &kind, std::string &diagnostic);
  [[noreturn]] void throw_error(ErrorKind kind, const std::string &diagnostic);
  void clear_active_epoch() noexcept;

  MPI_Comm communicator_ = MPI_COMM_NULL;
  MPI_Comm shared_communicator_ = MPI_COMM_NULL;
  MPI_Win shared_window_ = MPI_WIN_NULL;
  bool shared_window_locked_ = false;
  int rank_ = -1;
  int size_ = 0;
  ClassicalTopology local_topology_;
  std::unique_ptr<ClassicalBatchPlan> plan_;

  std::size_t atom_count_ = 0;
  int atom_count_mpi_ = 0;
  int coordinate_count_mpi_ = 0;

  // All GPU-local roots map one contiguous MPI shared-memory block.  Each
  // root writes only its stable input slot; the owner writes batched outputs
  // directly into the same block.  Barriers provide the transaction boundary
  // without copying the large frames through MPI_Gather/MPI_Scatter.
  double *shared_base_ = nullptr;
  double *gathered_positions_ = nullptr;
  double *gathered_mm_charges_ = nullptr;
  double *gathered_qm_charges_ = nullptr;
  double *root_pair_forces_ = nullptr;
  double *root_lj_energy_ = nullptr;
  double *root_coulomb_energy_ = nullptr;
  double *root_pair_virial_ = nullptr;
  double *root_mm_pppm_energy_ = nullptr;
  double *root_mm_pppm_virial_ = nullptr;
  double *root_mm_pppm_potential_ = nullptr;
  double *root_mm_pppm_forces_ = nullptr;
  double *root_qm_pppm_forces_ = nullptr;
  double *root_full_pppm_forces_ = nullptr;
  double *root_qm_pppm_energy_ = nullptr;
  double *root_full_pppm_energy_ = nullptr;
  double *root_qm_pppm_virial_ = nullptr;
  double *root_full_pppm_virial_ = nullptr;

  // Candidate buffers receive every scatter.  They are swapped with the
  // published buffers only after all roots report that every scatter passed.
  std::vector<double> mm_pair_forces_;
  std::vector<double> candidate_mm_pair_forces_;
  std::vector<double> mm_pppm_potential_;
  std::vector<double> candidate_mm_pppm_potential_;
  std::vector<double> mm_pppm_forces_;
  std::vector<double> candidate_mm_pppm_forces_;
  std::array<double, 6> pair_virial_{};
  std::array<double, 6> candidate_pair_virial_{};
  std::array<double, 6> mm_pppm_virial_{};
  std::array<double, 6> candidate_mm_pppm_virial_{};
  double lj_energy_ = 0.0;
  double candidate_lj_energy_ = 0.0;
  double coulomb_energy_ = 0.0;
  double candidate_coulomb_energy_ = 0.0;
  double mm_pppm_energy_ = 0.0;
  double candidate_mm_pppm_energy_ = 0.0;

  std::vector<double> qm_pppm_forces_;
  std::vector<double> candidate_qm_pppm_forces_;
  std::vector<double> full_pppm_forces_;
  std::vector<double> candidate_full_pppm_forces_;
  std::array<double, 6> qm_pppm_virial_{};
  std::array<double, 6> candidate_qm_pppm_virial_{};
  std::array<double, 6> full_pppm_virial_{};
  std::array<double, 6> candidate_full_pppm_virial_{};
  double qm_pppm_energy_ = 0.0;
  double candidate_qm_pppm_energy_ = 0.0;
  double full_pppm_energy_ = 0.0;
  double candidate_full_pppm_energy_ = 0.0;

  // Four fixed-width fields per root are reused for one small preflight
  // Allgather, replacing several latency-dominated min/max/validity reductions
  // on every force evaluation.
  std::vector<std::int64_t> preflight_records_;

  std::int64_t active_timestep_ = -1;
  std::int64_t mm_result_timestep_ = -1;
  std::int64_t qm_result_timestep_ = -1;
  bool active_mm_ = false;
  bool mm_result_valid_ = false;
  bool qm_result_valid_ = false;
  bool mm_potential_valid_ = false;
  bool mm_forces_valid_ = false;
  std::string last_error_;
};

} // namespace DPRC

#endif
