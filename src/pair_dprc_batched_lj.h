#ifndef LAMMPS_DPRC_PAIR_DPRC_BATCHED_LJ_H
#define LAMMPS_DPRC_PAIR_DPRC_BATCHED_LJ_H

#include "classical_batch.h"
#include "pair_lj_cut.h"

#include <cstdint>
#include <vector>

namespace LAMMPS_NS {

// Publication-only LJ proxy.  Coefficient parsing, mixing, offset, and tail
// accounting remain the pinned PairLJCut implementation, while compute()
// consumes the force/energy/virial produced by the GPU-local classical broker.
class PairDPRCBatchedLJ final : public PairLJCut {
 public:
  explicit PairDPRCBatchedLJ(class LAMMPS *);

  void compute(int, int) override;
  void init_style() override;

  // Export the expanded LAMMPS coefficient matrices after Pair::init() has
  // completed mixing and offset construction.
  void export_parameters(DPRC::ClassicalTopology &,
                         const std::vector<std::uint8_t> &enabled_pairs) const;
};

}  // namespace LAMMPS_NS

#endif
