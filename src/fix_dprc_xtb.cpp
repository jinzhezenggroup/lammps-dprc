#include "fix_dprc_xtb.h"

#include "xtbloom_lammps_adapter.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "modify.h"
#include "pair.h"
#include "universe.h"

#include <exception>
#include <memory>

using namespace LAMMPS_NS;

FixDPRCXtb::~FixDPRCXtb() {
  // The renamed reference base destructor calls the same function again. The
  // first call is deliberately here so the collective broker is released
  // before roots_ frees its communicator.
  if (adapter_bound_ && roots_ && roots_->is_root())
    dprc_lammps_xtb_destroy();
}

void FixDPRCXtb::init() {
  if (!modify->get_fix_by_style("^qmmm/xtb$").empty())
    error->all(FLERR,
               "Fix qmmm/xtb/dprc cannot be used together with fix qmmm/xtb");
  if (modify->get_fix_by_style("^qmmm/xtb/dprc$").size() > 1)
    error->all(FLERR,
               "Only one instance of fix qmmm/xtb/dprc is supported per partition");

  if (!roots_) {
    try {
      roots_ = std::make_unique<DPRC::PartitionRoots>(
          universe->uworld, comm->me, universe->iworld, universe->nworlds);
    } catch (const std::exception &exception) {
      error->universe_all(FLERR, exception.what());
    }

    int bind_status = 0;
    if (roots_->is_root()) {
      bind_status = DPRC::bind_lammps_xtbloom_adapter(
          lmp, roots_->communicator(), roots_->stable_slot());
      adapter_bound_ = bind_status == 0;
    }
    MPI_Bcast(&bind_status, 1, MPI_INT, 0, world);
    if (bind_status != 0)
      error->all(FLERR, "Could not bind the qmmm/xtb/dprc partition broker");
  }

  FixDPRCXtbReference::init();
}

void FixDPRCXtb::pre_force(int vflag) {
  // A later PRE_FORCE/pair/bond failure can abandon a token after this fix
  // has committed it but before the production KSpace call consumes it. A
  // same-timestep library retry enters here first; clear that stale token so
  // the MM reference capture below must execute a complete base PPPM solve.
  FixDPRCXtbReference::discard_fused_full_solve();

  const int atom_count = atom->nlocal + atom->nghost;
  rollback_charges_.resize(static_cast<std::size_t>(atom_count));
  rollback_forces_.resize(static_cast<std::size_t>(3) * atom_count);
  for (int atom_index = 0; atom_index < atom_count; ++atom_index) {
    rollback_charges_[atom_index] = atom->q[atom_index];
    for (int dimension = 0; dimension < 3; ++dimension) {
      rollback_forces_[3 * atom_index + dimension] =
          atom->f[atom_index][dimension];
    }
  }

  const double pair_vdwl = force->pair->eng_vdwl;
  const double pair_coulomb = force->pair->eng_coul;
  const double kspace_energy = force->kspace->energy;
  double pair_virial[6];
  double kspace_virial[6];
  for (int component = 0; component < 6; ++component) {
    pair_virial[component] = force->pair->virial[component];
    kspace_virial[component] = force->kspace->virial[component];
  }

  try {
    FixDPRCXtbReference::pre_force(vflag);
    // Only a completely successful pre_force transaction may arm the exact
    // timestep/run-phase/vflag token consumed by the production KSpace phase.
    FixDPRCXtbReference::commit_fused_full_solve(vflag);
  } catch (...) {
    // prepare_fused_full_solve() adds the cached MM mesh field in place. If
    // any later operation throws, ensure the next KSpace call takes the full
    // base PPPM path, which overwrites owned field values and rebuilds ghosts.
    // This local noexcept reset must precede every other rollback operation.
    FixDPRCXtbReference::discard_fused_full_solve();
    // Executable LAMMPS normally aborts on error, but the library interface
    // converts error->all() into an exception. Restore the public atom and
    // force-style state before allowing that exception to escape.
    for (int atom_index = 0; atom_index < atom_count; ++atom_index) {
      atom->q[atom_index] = rollback_charges_[atom_index];
      for (int dimension = 0; dimension < 3; ++dimension) {
        atom->f[atom_index][dimension] =
            rollback_forces_[3 * atom_index + dimension];
      }
    }
    force->pair->eng_vdwl = pair_vdwl;
    force->pair->eng_coul = pair_coulomb;
    force->kspace->energy = kspace_energy;
    for (int component = 0; component < 6; ++component) {
      force->pair->virial[component] = pair_virial[component];
      force->kspace->virial[component] = kspace_virial[component];
    }
    // Do not call a world collective while unwinding: rank-local LAMMPS
    // exceptions (for example error->one()) must escape immediately instead
    // of leaving the failing rank stuck in a rollback collective.
    throw;
  }

#ifdef DPRC_ENABLE_TEST_HOOKS
  // Deliberately throw outside the transaction catch to emulate a later
  // PRE_FORCE failure after this fix returned with a committed token.
  if (fail_after_fused_commit_) {
    fail_after_fused_commit_ = false;
    error->all(FLERR,
               "DPRC test hook: failure after fused full-solve commit");
  }
#endif
}

#ifdef DPRC_ENABLE_TEST_HOOKS
void FixDPRCXtb::after_fused_full_solve_prepared() {
  if (!fail_after_fused_prepare_)
    return;
  fail_after_fused_prepare_ = false;
  error->all(FLERR,
             "DPRC test hook: failure after fused full-solve preparation");
}
#endif
