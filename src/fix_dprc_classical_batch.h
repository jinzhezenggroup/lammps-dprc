#ifdef FIX_CLASS
// clang-format off
FixStyle(dprc/classical/batch,FixDPRCClassicalBatch);
// clang-format on
#else

#ifndef LAMMPS_DPRC_FIX_DPRC_CLASSICAL_BATCH_H
#define LAMMPS_DPRC_FIX_DPRC_CLASSICAL_BATCH_H

#include "fix.h"

namespace LAMMPS_NS {

// PRE_FORCE coordinator for the pure-classical shared GPU backend.  LAMMPS
// evaluates Pair::compute before the first ordinary KSpace::setup, so a fix is
// required to build and prepare the single-owner broker transaction before
// either publication proxy runs.
class FixDPRCClassicalBatch final : public Fix {
 public:
  FixDPRCClassicalBatch(class LAMMPS *, int, char **);

  int setmask() override;
  void init() override;
  void setup_pre_force(int) override;
  void pre_force(int) override;
  void min_pre_force(int) override;

 private:
  void validate_configuration() const;
  void prepare(int);
};

} // namespace LAMMPS_NS

#endif
#endif
