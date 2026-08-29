#include "stable_local_indices.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  using Tag = std::int64_t;
  const std::vector<Tag> stable{1, 2, 3};

  // The first three entries are owned atoms in a post-sort order.  The final
  // entries model periodic ghosts whose duplicate IDs may shadow a global
  // atom-map lookup; the reconstruction deliberately receives only nlocal.
  const std::vector<Tag> owned_then_ghosts{3, 1, 2, 2, 1};
  const auto indices =
      DPRC::stable_local_indices(stable, owned_then_ghosts.data(), 3u);
  assert(indices);
  assert(*indices == std::vector<int>({1, 2, 0}));
  for (std::size_t slot = 0; slot < stable.size(); ++slot)
    assert(owned_then_ghosts[static_cast<std::size_t>((*indices)[slot])] ==
           stable[slot]);

  const std::vector<Tag> duplicate_owned{1, 1, 3};
  assert(!DPRC::stable_local_indices(stable, duplicate_owned.data(), 3u));
  const std::vector<Tag> unknown_owned{1, 2, 4};
  assert(!DPRC::stable_local_indices(stable, unknown_owned.data(), 3u));
  assert(!DPRC::stable_local_indices(stable, owned_then_ghosts.data(), 2u));
  const std::vector<Tag> unsorted_stable{2, 1, 3};
  assert(!DPRC::stable_local_indices(unsorted_stable,
                                     owned_then_ghosts.data(), 3u));
  return 0;
}
