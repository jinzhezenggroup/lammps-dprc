#ifndef LAMMPS_DPRC_FIX_DPRC_XTB_H
#define LAMMPS_DPRC_FIX_DPRC_XTB_H

#include "partition_roots.h"

// Compile the generated fused LAMMPS reference under private fix and KSpace
// class names. The same macros are applied to both generated translation
// units, so no original QMMM-XTB implementation class is defined by the
// RTLD_GLOBAL plugin.
#define FixQMMMXTB FixDPRCXtbReference
#define PPPMXTB PPPMDPRC
#define PPPMTIP4PXTB PPPMTIP4PDPRC
#include "fix_qmmm_xtb.h"
#undef PPPMTIP4PXTB
#undef PPPMXTB
#undef FixQMMMXTB

#include <memory>
#include <vector>

namespace LAMMPS_NS {

class FixDPRCXtb final : public FixDPRCXtbReference {
public:
  FixDPRCXtb(class LAMMPS *lmp, int narg, char **arg) :
      FixDPRCXtbReference(lmp, narg, arg) {}
  ~FixDPRCXtb() override;

  // The Kokkos Verlet loop builds legacy host neighbor lists before
  // PRE_FORCE.  Publish the device-integrated coordinates at that boundary
  // so the legacy QMMM pair/neighbor path never observes stale positions.
  int setmask() override;
  void init() override;
  void pre_neighbor() override;
  void pre_force(int vflag) override;
  void post_force(int vflag) override;

#ifdef DPRC_ENABLE_TEST_HOOKS
  // Arm the test-only one-shot exception used to prove that a prepared fused
  // field cannot leak into the next library-mode force evaluation.
  void arm_failure_after_fused_prepare() noexcept {
    fail_after_fused_prepare_ = true;
  }

  // Emulate a later PRE_FORCE failure after this fix has committed its token
  // but before the production KSpace evaluation can consume it.
  void arm_failure_after_fused_commit() noexcept {
    fail_after_fused_commit_ = true;
  }
#endif

protected:
#ifdef DPRC_ENABLE_TEST_HOOKS
  void after_fused_full_solve_prepared() override;
#endif

private:
  std::unique_ptr<DPRC::PartitionRoots> roots_;
  bool adapter_bound_ = false;
#ifdef DPRC_ENABLE_TEST_HOOKS
  bool fail_after_fused_prepare_ = false;
  bool fail_after_fused_commit_ = false;
#endif
  // The pinned reference path temporarily clears forces and QM charges before
  // the external SCC call. Keep a reusable transaction image so a library-mode
  // LAMMPS exception cannot expose those intermediate values to the caller.
  std::vector<double> rollback_charges_;
  std::vector<double> rollback_forces_;
};

} // namespace LAMMPS_NS

#endif
