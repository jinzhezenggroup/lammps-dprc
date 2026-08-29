#ifndef LAMMPS_DPRC_STABLE_LOCAL_INDICES_H
#define LAMMPS_DPRC_STABLE_LOCAL_INDICES_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <vector>

namespace DPRC {

// Reconstruct stable-slot indices from the owned-atom prefix only.  LAMMPS
// may store periodic ghost images with the same atom ID after nlocal; those
// images must never replace the owned atom selected for force publication.
template <class Tag>
[[nodiscard]] std::optional<std::vector<int>> stable_local_indices(
    const std::vector<Tag> &stable_tags, const Tag *local_tags,
    std::size_t local_count) {
  if (local_count != stable_tags.size() ||
      (local_count != 0u && local_tags == nullptr) ||
      !std::is_sorted(stable_tags.begin(), stable_tags.end()) ||
      std::adjacent_find(stable_tags.begin(), stable_tags.end()) !=
          stable_tags.end())
    return std::nullopt;

  std::vector<int> indices(stable_tags.size(), -1);
  for (std::size_t local = 0; local < local_count; ++local) {
    const auto stable =
        std::lower_bound(stable_tags.begin(), stable_tags.end(), local_tags[local]);
    if (stable == stable_tags.end() || *stable != local_tags[local])
      return std::nullopt;
    const std::size_t slot =
        static_cast<std::size_t>(std::distance(stable_tags.begin(), stable));
    if (indices[slot] != -1)
      return std::nullopt;
    indices[slot] = static_cast<int>(local);
  }

  if (std::find(indices.begin(), indices.end(), -1) != indices.end())
    return std::nullopt;
  return indices;
}

} // namespace DPRC

#endif
