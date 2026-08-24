#include "fix_dprc_classical_batch.h"

#include "kspace_dprc_batched.h"

#include "domain.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "pair.h"

using namespace LAMMPS_NS;

FixDPRCClassicalBatch::FixDPRCClassicalBatch(LAMMPS *lmp, int narg,
                                             char **arg)
    : Fix(lmp, narg, arg) {
  if (narg != 3)
    error->all(FLERR, "Illegal fix dprc/classical/batch command");
  if (igroup != 0)
    error->all(FLERR,
               "Fix dprc/classical/batch must use the all group");
}

int FixDPRCClassicalBatch::setmask() {
  int mask = 0;
  mask |= FixConst::PRE_FORCE;
  mask |= FixConst::MIN_PRE_FORCE;
  return mask;
}

void FixDPRCClassicalBatch::init() {
  if (modify->get_fix_by_style("^dprc/classical/batch$").size() != 1)
    error->all(FLERR,
               "Exactly one fix dprc/classical/batch is required");
  if (!modify->get_fix_by_style("^qmmm/xtb/dprc$").empty())
    error->all(
        FLERR,
        "Fix dprc/classical/batch cannot be combined with qmmm/xtb/dprc");
  validate_configuration();

  // Domain::init() has already reduced all fix and shrink-wrap flags for this
  // run.  Subdomain-only changes are harmless for the required one-rank
  // windows, but any cell size/shape change would invalidate the shared Green
  // function and reciprocal vectors.
  if (domain->box_change_size || domain->box_change_shape)
    error->all(
        FLERR,
        "Fix dprc/classical/batch does not support cells that change within a run");
}

void FixDPRCClassicalBatch::validate_configuration() const {
  if (!force->pair || !force->pair->compute_flag || !force->kspace ||
      !force->kspace->compute_flag ||
      !dynamic_cast<PPPMTIP4PDPRCBatched *>(force->kspace))
    error->all(
        FLERR,
        "Fix dprc/classical/batch requires enabled batched pair proxies and pppm/tip4p/dprc/batch");
}

void FixDPRCClassicalBatch::prepare(int vflag) {
  validate_configuration();
  auto *batched = dynamic_cast<PPPMTIP4PDPRCBatched *>(force->kspace);
  if (!batched)
    error->all(
        FLERR,
        "Fix dprc/classical/batch lost pppm/tip4p/dprc/batch");
  // Release an orphaned token from a caught force-pipeline failure before
  // starting the next atomic publication transaction.
  batched->discard_fused_full_solve();
  batched->prepare_classical_publication(vflag);
}

void FixDPRCClassicalBatch::setup_pre_force(int vflag) {
  // Verlet and Min run this hook before the first ordinary KSpace::setup(),
  // while Pair::compute follows immediately afterwards.
  force->kspace->setup();
  prepare(vflag);
}

void FixDPRCClassicalBatch::pre_force(int vflag) { prepare(vflag); }

void FixDPRCClassicalBatch::min_pre_force(int vflag) { prepare(vflag); }
