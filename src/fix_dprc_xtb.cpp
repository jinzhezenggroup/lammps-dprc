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
#include <cstdint>
#include <memory>

#ifdef DPRC_HAVE_LAMMPS_KOKKOS_BRIDGE
#include "atom_kokkos.h"
#include "atom_masks.h"
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {

// The reference QMMM-XTB implementation is intentionally a legacy LAMMPS
// fix: it uses atom->x/q/f pointers and calls the legacy PPPM routines.  A
// KOKKOS Verlet loop keeps the authoritative integration state in dual views,
// so the wrapper must make that boundary explicit.  This bridge is optional
// because a host-only LAMMPS build does not provide AtomKokkos.
void sync_legacy_atom_views(LAMMPS *lmp, std::uint64_t mask) {
#ifdef DPRC_HAVE_LAMMPS_KOKKOS_BRIDGE
  if (lmp && lmp->atomKK)
    lmp->atomKK->sync(Host, mask);
#else
  (void)lmp;
  (void)mask;
#endif
}

void publish_legacy_atom_views(LAMMPS *lmp, std::uint64_t mask) {
#ifdef DPRC_HAVE_LAMMPS_KOKKOS_BRIDGE
  if (lmp && lmp->atomKK)
    lmp->atomKK->modified(Host, mask);
#else
  (void)lmp;
  (void)mask;
#endif
}

#ifdef DPRC_HAVE_LAMMPS_KOKKOS_BRIDGE
// The legacy QMMM fix reads coordinates, charges, and forces before its
// pre-force calculation.  It publishes only charges and forces; narrowing the
// masks avoids needlessly invalidating velocities, topology, and custom data
// on every timestep.
constexpr std::uint64_t kLegacyReadMask = X_MASK | Q_MASK | F_MASK;
constexpr std::uint64_t kLegacyPublicationMask = Q_MASK | F_MASK;
// Neighbor construction runs before PRE_FORCE in a Kokkos Verlet step.  The
// legacy neighbor builder only needs current coordinates at this boundary;
// keeping this mask narrow avoids synchronizing charge/force views twice.
constexpr std::uint64_t kLegacyNeighborMask = X_MASK;
#else
constexpr std::uint64_t kLegacyReadMask = 0;
constexpr std::uint64_t kLegacyPublicationMask = 0;
constexpr std::uint64_t kLegacyNeighborMask = 0;
#endif

} // namespace

int FixDPRCXtb::setmask() {
  int mask = FixDPRCXtbReference::setmask();
#ifdef DPRC_HAVE_LAMMPS_KOKKOS_BRIDGE
  // ModifyKokkos invokes PRE_NEIGHBOR immediately before the legacy neighbor
  // builder.  This is the earliest safe point at which device coordinates can
  // be made visible through atom->x.
  mask |= PRE_NEIGHBOR;
#endif
  return mask;
}

FixDPRCXtb::~FixDPRCXtb() {
  // The renamed reference base destructor calls the same function again. The
  // first call is deliberately here so the collective broker is released
  // before roots_ frees its communicator.
  if (adapter_bound_ && roots_ && roots_->is_root())
    dprc_lammps_xtb_destroy();
}

void FixDPRCXtb::pre_neighbor() {
  sync_legacy_atom_views(lmp, kLegacyNeighborMask);
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
  // VerletKokkos has just integrated on the device.  Pull the current state
  // into the legacy arrays before the reference fix reads or modifies them.
  sync_legacy_atom_views(lmp, kLegacyReadMask);

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
    publish_legacy_atom_views(lmp, kLegacyPublicationMask);
    // Do not call a world collective while unwinding: rank-local LAMMPS
    // exceptions (for example error->one()) must escape immediately instead
    // of leaving the failing rank stuck in a rollback collective.
    throw;
  }

  // Make qmmm's legacy force/charge publication visible to the next KOKKOS
  // pair, KSpace, reverse-communication, and integration phases.
  publish_legacy_atom_views(lmp, kLegacyPublicationMask);

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

void FixDPRCXtb::post_force(int vflag) {
  // Pair/KSpace and reverse communication have completed on the device by
  // this point; the reference post_force routine consumes and modifies the
  // legacy force array.
  sync_legacy_atom_views(lmp, kLegacyReadMask);
  try {
    FixDPRCXtbReference::post_force(vflag);
  } catch (...) {
    publish_legacy_atom_views(lmp, kLegacyPublicationMask);
    throw;
  }
  publish_legacy_atom_views(lmp, kLegacyPublicationMask);
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
