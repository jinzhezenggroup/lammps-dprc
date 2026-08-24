#ifndef LAMMPS_DPRC_POINT_CHARGE_SLOTS_H
#define LAMMPS_DPRC_POINT_CHARGE_SLOTS_H

#include <cstddef>
#include <vector>

namespace DPRC {

// Owns the immutable point-charge gamma topology used by one xTBloom plan.
//
// LAMMPS may replace the physical MM sites selected at a neighbor rebuild,
// while xTBloom WARM state requires the point-charge gamma sequence to remain
// unchanged.  This class assigns each current site to a compatible permanent
// slot and pads every gamma class with zero-charge capacity.  Geometry and
// charge values remain per-call data; only a real class-capacity overflow
// requires grow() and therefore a native plan rebuild.
class PointChargeSlots {
 public:
  // Assign current point gammas to the existing immutable topology. Exact
  // binary64 comparison is intentional: gamma values are part of xTBloom's
  // strict plan identity. Returns false when a class is absent or full.
  bool assign(const std::vector<double> &point_gammas);

  // Grow each required gamma class with bounded headroom while retaining old
  // classes and capacities. Growth is an epoch operation and may allocate.
  void grow(const std::vector<double> &point_gammas);

  void clear() noexcept;

  const std::vector<double> &topology_gammas() const noexcept {
    return topology_gammas_;
  }
  const std::vector<std::size_t> &assignments() const noexcept {
    return assignments_;
  }
  std::size_t capacity() const noexcept { return topology_gammas_.size(); }

 private:
  struct GammaRange {
    double gamma = 0.0;
    std::size_t begin = 0;
    std::size_t capacity = 0;
    std::size_t used = 0;
  };

  std::vector<double> topology_gammas_;
  std::vector<GammaRange> ranges_;
  std::vector<std::size_t> assignments_;
};

} // namespace DPRC

#endif
