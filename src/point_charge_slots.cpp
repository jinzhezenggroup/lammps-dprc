#include "point_charge_slots.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace DPRC {
namespace {

// The production ETP/ETH shell contains about 612 sites and changed by one
// site over the diagnostic trajectory. Two-percent-class initial headroom is
// enough to absorb ordinary neighbor-epoch count noise without making every
// SCC iteration process a large zero-charge tail. A genuine overflow grows
// monotonically and never drops a physical point.
constexpr std::size_t kCapacityAlignment = 8u;
constexpr std::size_t kMinimumHeadroom = 8u;
constexpr std::size_t kHeadroomDivisor = 64u;

void require_valid_gamma(double gamma) {
  if (!std::isfinite(gamma) || gamma <= 0.0)
    throw std::invalid_argument(
        "point-charge slot gamma must be finite and positive");
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char *label) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(label);
  return left + right;
}

std::size_t padded_capacity(std::size_t required) {
  if (required == 0u)
    return 0u;
  const std::size_t proportional =
      checked_add(required, kHeadroomDivisor - 1u,
                  "point-charge headroom overflows") /
      kHeadroomDivisor;
  const std::size_t headroom = std::max(kMinimumHeadroom, proportional);
  const std::size_t with_headroom =
      checked_add(required, headroom, "point-charge capacity overflows");
  const std::size_t rounded =
      checked_add(with_headroom, kCapacityAlignment - 1u,
                  "point-charge capacity alignment overflows");
  return rounded / kCapacityAlignment * kCapacityAlignment;
}

} // namespace

bool PointChargeSlots::assign(const std::vector<double> &point_gammas) {
  assignments_.resize(point_gammas.size());
  for (GammaRange &range : ranges_)
    range.used = 0u;

  for (std::size_t point = 0; point < point_gammas.size(); ++point) {
    const double gamma = point_gammas[point];
    require_valid_gamma(gamma);
    const auto found = std::lower_bound(
        ranges_.begin(), ranges_.end(), gamma,
        [](const GammaRange &range, double value) {
          return range.gamma < value;
        });
    if (found == ranges_.end() || found->gamma != gamma ||
        found->used == found->capacity) {
      assignments_.clear();
      return false;
    }
    assignments_[point] = found->begin + found->used;
    ++found->used;
  }
  return true;
}

void PointChargeSlots::grow(const std::vector<double> &point_gammas) {
  std::vector<double> sorted(point_gammas);
  for (const double gamma : sorted)
    require_valid_gamma(gamma);
  std::sort(sorted.begin(), sorted.end());

  std::vector<std::pair<double, std::size_t>> required;
  for (const double gamma : sorted) {
    if (required.empty() || required.back().first != gamma)
      required.emplace_back(gamma, 1u);
    else
      ++required.back().second;
  }

  std::vector<GammaRange> next_ranges;
  next_ranges.reserve(ranges_.size() + required.size());
  std::size_t old_index = 0u;
  std::size_t required_index = 0u;
  std::size_t total = 0u;
  while (old_index < ranges_.size() || required_index < required.size()) {
    const bool take_old =
        required_index == required.size() ||
        (old_index < ranges_.size() &&
         ranges_[old_index].gamma < required[required_index].first);
    const bool take_required =
        old_index == ranges_.size() ||
        (required_index < required.size() &&
         required[required_index].first < ranges_[old_index].gamma);

    double gamma = 0.0;
    std::size_t capacity = 0u;
    if (take_old) {
      gamma = ranges_[old_index].gamma;
      capacity = ranges_[old_index].capacity;
      ++old_index;
    } else if (take_required) {
      gamma = required[required_index].first;
      capacity = padded_capacity(required[required_index].second);
      ++required_index;
    } else {
      gamma = ranges_[old_index].gamma;
      capacity = std::max(ranges_[old_index].capacity,
                          padded_capacity(required[required_index].second));
      ++old_index;
      ++required_index;
    }
    next_ranges.push_back({gamma, total, capacity, 0u});
    total = checked_add(total, capacity,
                        "point-charge topology capacity overflows");
  }

  std::vector<double> next_topology;
  next_topology.reserve(total);
  for (const GammaRange &range : next_ranges)
    next_topology.insert(next_topology.end(), range.capacity, range.gamma);

  ranges_.swap(next_ranges);
  topology_gammas_.swap(next_topology);
  if (!assign(point_gammas))
    throw std::logic_error("grown point-charge topology cannot fit request");
}

void PointChargeSlots::clear() noexcept {
  topology_gammas_.clear();
  ranges_.clear();
  assignments_.clear();
}

} // namespace DPRC
