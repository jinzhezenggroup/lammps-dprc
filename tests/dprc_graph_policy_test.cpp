#include "dprc_graph_policy.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

int main() {
  using DPRC::keep_dprc_canonical_edge;
  if (!keep_dprc_canonical_edge(true, true))
    return EXIT_FAILURE;
  if (!keep_dprc_canonical_edge(true, false))
    return EXIT_FAILURE;
  if (!keep_dprc_canonical_edge(false, true))
    return EXIT_FAILURE;
  if (keep_dprc_canonical_edge(false, false))
    return EXIT_FAILURE;

  std::vector<DPRC::CanonicalEdgeOrderKey> edges{
      {9, 3, 1.0f, 0.0f, 0.0f},
      {4, 1, 0.0f, 2.0f, 0.0f},
      {4, 1, 0.0f, -2.0f, 0.0f},
      {4, 2, 0.0f, -2.0f, 0.0f},
  };
  std::sort(edges.begin(), edges.end(), DPRC::canonical_edge_order_less);
  if (edges[0].source_tag != 4 || edges[0].dy != -2.0f ||
      edges[0].source_node != 1)
    return EXIT_FAILURE;
  if (edges[1].source_tag != 4 || edges[1].dy != -2.0f ||
      edges[1].source_node != 2)
    return EXIT_FAILURE;
  if (edges[2].source_tag != 4 || edges[2].dy != 2.0f ||
      edges[3].source_tag != 9)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
