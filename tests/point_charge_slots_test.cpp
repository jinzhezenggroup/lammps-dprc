#include "point_charge_slots.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
  DPRC::PointChargeSlots slots;
  assert(slots.assign({}));
  assert(slots.capacity() == 0u);
  assert(!slots.assign({0.4}));

  slots.grow({0.4, 0.4, 0.4});
  const std::size_t first_capacity = slots.capacity();
  assert(first_capacity >= 3u);
  assert(first_capacity % 8u == 0u);
  assert(slots.assign({0.4}));
  assert(slots.assign({0.4, 0.4, 0.4, 0.4}));
  assert(slots.capacity() == first_capacity);

  // Point order is per-frame data. Gamma classes select permanent compatible
  // slots without changing the immutable topology ordering.
  slots.grow({0.2, 0.4, 0.2, 0.4});
  const std::vector<double> topology = slots.topology_gammas();
  assert(std::is_sorted(topology.begin(), topology.end()));
  assert(slots.assign({0.4, 0.2, 0.4, 0.2}));
  const auto assignment = slots.assignments();
  assert(topology[assignment[0]] == 0.4);
  assert(topology[assignment[1]] == 0.2);
  assert(topology[assignment[2]] == 0.4);
  assert(topology[assignment[3]] == 0.2);

  // A class-capacity overflow is detected transactionally, then grow retains
  // every previously seen class so a temporarily absent element can return.
  const std::size_t gamma_02_capacity =
      static_cast<std::size_t>(std::count(topology.begin(), topology.end(), 0.2));
  std::vector<double> overflow(gamma_02_capacity + 1u, 0.2);
  assert(!slots.assign(overflow));
  slots.grow(overflow);
  assert(slots.assign(overflow));
  assert(std::find(slots.topology_gammas().begin(),
                   slots.topology_gammas().end(), 0.4) !=
         slots.topology_gammas().end());

  bool rejected = false;
  try {
    slots.assign({0.0});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

  slots.clear();
  assert(slots.capacity() == 0u);
  assert(slots.assign({}));
  return 0;
}
