#ifndef LAMMPS_DPRC_GRAPH_POLICY_H
#define LAMMPS_DPRC_GRAPH_POLICY_H

#include <cstdint>

namespace DPRC {

// DPRc represents the correction involving the QM center group. Environment
// atoms remain graph nodes so QM--MM forces are retained, but an edge whose
// two endpoints are both outside the center group would introduce an MM--MM
// correction and must never enter the compact canonical graph.
constexpr bool keep_dprc_canonical_edge(bool destination_is_center,
                                        bool source_is_center) noexcept {
  return destination_is_center || source_is_center;
}

// Stable key for one already validated, finite canonical edge. LAMMPS GPU
// neighbor lists do not promise a reproducible slot order, while the DPA4c
// descriptor accumulates neighbors in the supplied CSR order. Sorting by atom
// identity and then by the periodic-image displacement makes the same physical
// graph byte-identical across one-window and partition-batched launches.
struct CanonicalEdgeOrderKey {
  std::int64_t source_tag = 0;
  std::uint32_t source_node = 0;
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
};

inline bool canonical_edge_order_less(const CanonicalEdgeOrderKey &left,
                                      const CanonicalEdgeOrderKey &right) noexcept {
  if (left.source_tag != right.source_tag)
    return left.source_tag < right.source_tag;
  if (left.dx != right.dx) return left.dx < right.dx;
  if (left.dy != right.dy) return left.dy < right.dy;
  if (left.dz != right.dz) return left.dz < right.dz;
  return left.source_node < right.source_node;
}

}  // namespace DPRC

#endif
