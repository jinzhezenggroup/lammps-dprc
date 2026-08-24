#ifndef LAMMPS_DPRC_PAIR_DPRC_BATCHED_TIP4P_H
#define LAMMPS_DPRC_PAIR_DPRC_BATCHED_TIP4P_H

#include "classical_batch.h"
#include "pair_tip4p_long.h"

namespace LAMMPS_NS {

// Coulomb-account publication proxy for the fused batched TIP4P real-space
// kernel.  The combined atomic force and virial are published by the LJ proxy
// once; this proxy contributes only eng_coul so PairHybrid does not double-add
// the fused force slice.
class PairDPRCBatchedTIP4PLong final : public PairTIP4PLong {
 public:
  explicit PairDPRCBatchedTIP4PLong(class LAMMPS *);

  void compute(int, int) override;
  void init_style() override;

  void export_parameters(DPRC::ClassicalTopology &) const;
  [[nodiscard]] int oxygen_type() const noexcept { return typeO; }
  [[nodiscard]] int hydrogen_type() const noexcept { return typeH; }
  [[nodiscard]] double alpha_value() const noexcept { return alpha; }
};

}  // namespace LAMMPS_NS

#endif
