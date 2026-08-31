#ifndef LAMMPS_DPRC_STABLE_BATCH_H
#define LAMMPS_DPRC_STABLE_BATCH_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace DPRC {

// Immutable xTB topology owned by one umbrella window. The index is the
// plan-local replica identity. The synchronized all-window M1 plan requires it
// to equal the dense LAMMPS iworld and xTBloom batch slot; it must never be
// reassigned while a WARM checkpoint can exist.
struct WindowTopology {
  std::int32_t window_index = -1;
  std::vector<std::int32_t> atomic_numbers;
  double molecular_charge = 0.0;
  std::int32_t unpaired_electrons = 0;
  std::int32_t spin_channels = 1;
  std::vector<double> point_charge_gammas;
  bool charge_response_enabled = false;
};

// Geometry and caller-owned periodic data for one window at one MD timestep.
// All real values already use xTBloom's public atomic-unit convention.
struct WindowFrame {
  std::int32_t window_index = -1;
  std::int64_t timestep = -1;
  std::vector<double> positions;
  std::vector<double> point_charge_positions;
  std::vector<double> point_charge_values;
  std::vector<double> atomic_potential_shifts;
  std::vector<double> charge_response_matrix;
};

struct DoubleArrayView {
  const double *data = nullptr;
  std::size_t size = 0;
};

// Non-owning counterpart used by the MPI broker to stage directly from its
// preallocated ragged receive buffers without constructing per-window vectors.
struct WindowFrameView {
  std::int32_t window_index = -1;
  std::int64_t timestep = -1;
  DoubleArrayView positions;
  DoubleArrayView point_charge_positions;
  DoubleArrayView point_charge_values;
  DoubleArrayView atomic_potential_shifts;
  DoubleArrayView charge_response_matrix;
};

struct SlotRange {
  std::size_t slot = 0;
  std::int64_t atom_begin = 0;
  std::int64_t atom_end = 0;
  std::int64_t point_charge_begin = 0;
  std::int64_t point_charge_end = 0;
  std::int64_t charge_response_begin = 0;
  std::int64_t charge_response_end = 0;
};

enum class SccStartPolicy { Fresh, Warm };

// StableBatch owns the allocation-stable ragged input image used by one
// fixed-topology xTBloom plan. It also mirrors xTBloom's whole-batch strict
// WARM contract: one failed peer invalidates readiness for the complete next
// batch.  The corresponding output transaction must also be all-or-nothing;
// successful peer slices are not independently publishable.
class StableBatch {
public:
  explicit StableBatch(std::vector<WindowTopology> topologies);

  std::size_t size() const noexcept { return topologies_.size(); }
  std::int64_t timestep() const noexcept { return timestep_; }
  bool ready() const noexcept;
  bool compute_in_flight() const noexcept { return compute_in_flight_; }
  bool warm_ready() const noexcept { return warm_ready_; }

  const WindowTopology &topology(std::size_t slot) const;
  const SlotRange &range(std::size_t slot) const;
  std::size_t slot_for_window(std::int32_t window_index) const;

  const std::vector<std::int64_t> &atom_offsets() const noexcept {
    return atom_offsets_;
  }
  const std::vector<std::int32_t> &atomic_numbers() const noexcept {
    return atomic_numbers_;
  }
  const std::vector<double> &molecular_charges() const noexcept {
    return molecular_charges_;
  }
  const std::vector<std::int32_t> &unpaired_electrons() const noexcept {
    return unpaired_electrons_;
  }
  const std::vector<std::int32_t> &spin_channels() const noexcept {
    return spin_channels_;
  }
  const std::vector<std::int64_t> &point_charge_offsets() const noexcept {
    return point_charge_offsets_;
  }
  const std::vector<double> &point_charge_gammas() const noexcept {
    return point_charge_gammas_;
  }
  const std::vector<std::int64_t> &charge_response_offsets() const noexcept {
    return charge_response_offsets_;
  }

  const std::vector<double> &positions() const noexcept { return positions_; }
  const std::vector<double> &point_charge_positions() const noexcept {
    return point_charge_positions_;
  }
  const std::vector<double> &point_charge_values() const noexcept {
    return point_charge_values_;
  }
  const std::vector<double> &atomic_potential_shifts() const noexcept {
    return atomic_potential_shifts_;
  }
  const std::vector<double> &charge_response_matrix() const noexcept {
    return charge_response_matrix_;
  }

  // Validate the complete per-window frame before copying any of its bytes.
  // Every slot in one batch transaction must carry the same MD timestep.
  void stage(const WindowFrame &frame);
  void stage(const WindowFrameView &frame);

  // Seal a complete staged timestep and return the strict xTB SCC start mode.
  // Call complete_compute() exactly once after the native call is settled.
  SccStartPolicy prepare_compute();

  // Record the native call and per-system statuses. A call-level failure or
  // any failed system revokes the whole plan checkpoint, matching xTBloom.
  void complete_compute(bool call_succeeded, const std::int32_t *statuses,
                        std::size_t status_count, std::int32_t success_status);

  // Revoke a checkpoint after a post-native publication validation failure.
  // This is noexcept because it is used on an error path that must remain
  // recoverable without reopening a compute transaction.
  void invalidate_warm_checkpoint() noexcept { warm_ready_ = false; }

private:
  void clear_staging() noexcept;

  std::vector<WindowTopology> topologies_;
  std::vector<SlotRange> ranges_;
  std::vector<std::int64_t> atom_offsets_;
  std::vector<std::int32_t> atomic_numbers_;
  std::vector<double> molecular_charges_;
  std::vector<std::int32_t> unpaired_electrons_;
  std::vector<std::int32_t> spin_channels_;
  std::vector<std::int64_t> point_charge_offsets_;
  std::vector<double> point_charge_gammas_;
  std::vector<std::int64_t> charge_response_offsets_;

  std::vector<double> positions_;
  std::vector<double> point_charge_positions_;
  std::vector<double> point_charge_values_;
  std::vector<double> atomic_potential_shifts_;
  std::vector<double> charge_response_matrix_;

  std::vector<unsigned char> staged_;
  std::size_t staged_count_ = 0;
  std::int64_t timestep_ = -1;
  bool compute_in_flight_ = false;
  bool warm_ready_ = false;
};

} // namespace DPRC

#endif
