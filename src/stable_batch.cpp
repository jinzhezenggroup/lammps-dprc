#include "stable_batch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace DPRC {
namespace {

std::int64_t checked_add(std::int64_t lhs, std::int64_t rhs,
                         const char *label) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
    throw std::overflow_error(std::string(label) + " extent overflows int64");
  }
  return lhs + rhs;
}

std::int64_t checked_square(std::int64_t value, const char *label) {
  if (value < 0 || (value != 0 &&
                    value > std::numeric_limits<std::int64_t>::max() / value)) {
    throw std::overflow_error(std::string(label) + " extent overflows int64");
  }
  return value * value;
}

std::size_t as_size(std::int64_t value, const char *label) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(std::string(label) + " does not fit size_t");
  }
  return static_cast<std::size_t>(value);
}

std::int64_t as_int64(std::size_t value, const char *label) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(std::string(label) + " does not fit int64");
  }
  return static_cast<std::int64_t>(value);
}

std::size_t checked_scale(std::size_t value, std::size_t factor,
                          const char *label) {
  if (factor != 0u && value > std::numeric_limits<std::size_t>::max() / factor)
    throw std::overflow_error(std::string(label) + " extent overflows size_t");
  return value * factor;
}

void require_size(std::size_t actual, std::size_t expected, const char *label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " has extent " +
                                std::to_string(actual) + ", expected " +
                                std::to_string(expected));
  }
}

void require_view(const DoubleArrayView &view, std::size_t expected,
                  const char *label) {
  require_size(view.size, expected, label);
  if (view.size != 0u && view.data == nullptr)
    throw std::invalid_argument(std::string(label) + " data is null");
}

void copy_view(const DoubleArrayView &view, std::vector<double> &destination,
               std::size_t offset) {
  if (view.size != 0u)
    std::copy_n(view.data, view.size, destination.begin() + offset);
}

} // namespace

StableBatch::StableBatch(std::vector<WindowTopology> topologies)
    : topologies_(std::move(topologies)) {
  if (topologies_.empty())
    throw std::invalid_argument("stable batch requires at least one window");

  std::sort(topologies_.begin(), topologies_.end(),
            [](const WindowTopology &lhs, const WindowTopology &rhs) {
              return lhs.window_index < rhs.window_index;
            });

  std::int64_t total_atoms = 0;
  std::int64_t total_points = 0;
  std::int64_t total_response = 0;
  bool any_response = false;
  bool all_response = true;

  atom_offsets_.push_back(0);
  point_charge_offsets_.push_back(0);
  charge_response_offsets_.push_back(0);
  ranges_.reserve(topologies_.size());
  molecular_charges_.reserve(topologies_.size());
  unpaired_electrons_.reserve(topologies_.size());
  spin_channels_.reserve(topologies_.size());

  for (std::size_t slot = 0; slot < topologies_.size(); ++slot) {
    const WindowTopology &topology = topologies_[slot];
    if (topology.window_index < 0)
      throw std::invalid_argument("window indices must be non-negative");
    if (slot != 0 &&
        topologies_[slot - 1].window_index == topology.window_index) {
      throw std::invalid_argument("window indices must be unique");
    }
    if (static_cast<std::size_t>(topology.window_index) != slot) {
      throw std::invalid_argument(
          "window indices must be the dense stable slots [0, batch_size)");
    }
    if (topology.atomic_numbers.empty())
      throw std::invalid_argument("each window requires at least one QM atom");
    if (!std::isfinite(topology.molecular_charge))
      throw std::invalid_argument("molecular charge must be finite");
    if (topology.unpaired_electrons < 0)
      throw std::invalid_argument("unpaired electrons must be non-negative");
    if (topology.spin_channels != 1 && topology.spin_channels != 2)
      throw std::invalid_argument("spin channels must be one or two");
    for (const std::int32_t atomic_number : topology.atomic_numbers) {
      if (atomic_number <= 0)
        throw std::invalid_argument("atomic numbers must be positive");
    }
    for (const double gamma : topology.point_charge_gammas) {
      if (!std::isfinite(gamma) || gamma <= 0.0)
        throw std::invalid_argument(
            "point-charge gamma values must be finite and positive");
    }

    const std::int64_t atoms =
        as_int64(topology.atomic_numbers.size(), "window atom count");
    const std::int64_t points =
        as_int64(topology.point_charge_gammas.size(), "point-charge count");
    const std::int64_t response = topology.charge_response_enabled
                                      ? checked_square(atoms, "response")
                                      : 0;
    const SlotRange range{slot,
                          total_atoms,
                          checked_add(total_atoms, atoms, "atom"),
                          total_points,
                          checked_add(total_points, points, "point-charge"),
                          total_response,
                          checked_add(total_response, response, "response")};
    ranges_.push_back(range);
    total_atoms = range.atom_end;
    total_points = range.point_charge_end;
    total_response = range.charge_response_end;
    atom_offsets_.push_back(total_atoms);
    point_charge_offsets_.push_back(total_points);
    charge_response_offsets_.push_back(total_response);

    atomic_numbers_.insert(atomic_numbers_.end(),
                           topology.atomic_numbers.begin(),
                           topology.atomic_numbers.end());
    molecular_charges_.push_back(topology.molecular_charge);
    unpaired_electrons_.push_back(topology.unpaired_electrons);
    spin_channels_.push_back(topology.spin_channels);
    point_charge_gammas_.insert(point_charge_gammas_.end(),
                                topology.point_charge_gammas.begin(),
                                topology.point_charge_gammas.end());
    any_response = any_response || topology.charge_response_enabled;
    all_response = all_response && topology.charge_response_enabled;
  }

  // xTBloom's b/A descriptors are one optional batch-level attachment. Mixed
  // enablement would need an explicit documented zero-operator convention;
  // reject it now rather than silently changing a window's SCC identity.
  if (any_response && !all_response) {
    throw std::invalid_argument(
        "charge-response enablement must match across one fixed batch");
  }
  if (!any_response)
    charge_response_offsets_.clear();

  positions_.resize(
      checked_scale(as_size(total_atoms, "atom"), 3u, "position"));
  point_charge_positions_.resize(checked_scale(
      as_size(total_points, "point-charge"), 3u, "point-charge position"));
  point_charge_values_.resize(as_size(total_points, "point-charge"));
  if (any_response) {
    atomic_potential_shifts_.resize(as_size(total_atoms, "atom"));
    charge_response_matrix_.resize(as_size(total_response, "response"));
  }
  staged_.assign(topologies_.size(), 0u);
}

const WindowTopology &StableBatch::topology(std::size_t slot) const {
  if (slot >= topologies_.size())
    throw std::out_of_range("batch slot is out of range");
  return topologies_[slot];
}

const SlotRange &StableBatch::range(std::size_t slot) const {
  if (slot >= ranges_.size())
    throw std::out_of_range("batch slot is out of range");
  return ranges_[slot];
}

std::size_t StableBatch::slot_for_window(std::int32_t window_index) const {
  const auto found =
      std::lower_bound(topologies_.begin(), topologies_.end(), window_index,
                       [](const WindowTopology &topology, std::int32_t index) {
                         return topology.window_index < index;
                       });
  if (found == topologies_.end() || found->window_index != window_index)
    throw std::out_of_range("window is not registered in this batch");
  return static_cast<std::size_t>(found - topologies_.begin());
}

bool StableBatch::ready() const noexcept {
  return !compute_in_flight_ && staged_count_ == topologies_.size();
}

void StableBatch::stage(const WindowFrame &frame) {
  stage(WindowFrameView{
      frame.window_index,
      frame.timestep,
      {frame.positions.data(), frame.positions.size()},
      {frame.point_charge_positions.data(),
       frame.point_charge_positions.size()},
      {frame.point_charge_values.data(), frame.point_charge_values.size()},
      {frame.atomic_potential_shifts.data(),
       frame.atomic_potential_shifts.size()},
      {frame.charge_response_matrix.data(),
       frame.charge_response_matrix.size()}});
}

void StableBatch::stage(const WindowFrameView &frame) {
  if (compute_in_flight_)
    throw std::logic_error("cannot stage while a batch compute is in flight");
  if (frame.timestep < 0)
    throw std::invalid_argument("timestep must be non-negative");

  const std::size_t slot = slot_for_window(frame.window_index);
  if (staged_[slot] != 0u)
    throw std::logic_error("window already staged for this timestep");
  if (staged_count_ != 0 && frame.timestep != timestep_)
    throw std::invalid_argument(
        "all windows in one batch must stage the same timestep");

  const SlotRange &slot_range = ranges_[slot];
  const std::size_t atoms =
      as_size(slot_range.atom_end - slot_range.atom_begin, "window atom");
  const std::size_t points =
      as_size(slot_range.point_charge_end - slot_range.point_charge_begin,
              "window point-charge");
  const std::size_t response =
      as_size(slot_range.charge_response_end - slot_range.charge_response_begin,
              "window response");

  // Validate every extent before changing the shared ragged image. This makes
  // a malformed window retryable without leaving a half-copied slot.
  require_view(frame.positions, checked_scale(atoms, 3u, "positions"),
               "positions");
  require_view(frame.point_charge_positions,
               checked_scale(points, 3u, "point-charge positions"),
               "point-charge positions");
  require_view(frame.point_charge_values, points, "point-charge values");
  require_view(frame.atomic_potential_shifts, response == 0 ? 0u : atoms,
               "atomic potential shifts");
  require_view(frame.charge_response_matrix, response,
               "charge-response matrix");

  const std::size_t atom_begin = as_size(slot_range.atom_begin, "atom offset");
  const std::size_t point_begin =
      as_size(slot_range.point_charge_begin, "point-charge offset");
  const std::size_t response_begin =
      as_size(slot_range.charge_response_begin, "response offset");
  copy_view(frame.positions, positions_, 3u * atom_begin);
  copy_view(frame.point_charge_positions, point_charge_positions_,
            3u * point_begin);
  copy_view(frame.point_charge_values, point_charge_values_, point_begin);
  if (response != 0u) {
    copy_view(frame.atomic_potential_shifts, atomic_potential_shifts_,
              atom_begin);
    copy_view(frame.charge_response_matrix, charge_response_matrix_,
              response_begin);
  }

  if (staged_count_ == 0)
    timestep_ = frame.timestep;
  staged_[slot] = 1u;
  ++staged_count_;
}

SccStartPolicy StableBatch::prepare_compute() {
  if (!ready())
    throw std::logic_error("all batch windows must be staged before compute");
  compute_in_flight_ = true;
  return warm_ready_ ? SccStartPolicy::Warm : SccStartPolicy::Fresh;
}

void StableBatch::complete_compute(bool call_succeeded,
                                   const std::int32_t *statuses,
                                   std::size_t status_count,
                                   std::int32_t success_status) {
  if (!compute_in_flight_)
    throw std::logic_error("no batch compute is in flight");
  if (call_succeeded && (statuses == nullptr || status_count != size()))
    throw std::invalid_argument(
        "successful batch completion requires one status per window");

  bool all_systems_succeeded = call_succeeded;
  if (all_systems_succeeded) {
    for (std::size_t slot = 0; slot < status_count; ++slot)
      all_systems_succeeded =
          all_systems_succeeded && statuses[slot] == success_status;
  }
  warm_ready_ = all_systems_succeeded;
  clear_staging();
}

void StableBatch::clear_staging() noexcept {
  std::fill(staged_.begin(), staged_.end(), 0u);
  staged_count_ = 0;
  timestep_ = -1;
  compute_in_flight_ = false;
}

} // namespace DPRC
