#include "pair_dprc_batched_tip4p.h"

#include "kspace_dprc_batched.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"

#include <cmath>
#include <cstddef>

using namespace LAMMPS_NS;

PairDPRCBatchedTIP4PLong::PairDPRCBatchedTIP4PLong(LAMMPS *lmp)
    : PairTIP4PLong(lmp) {
  no_virial_fdotr_compute = 1;
}

void PairDPRCBatchedTIP4PLong::init_style() {
  if (atom->tag_enable == 0)
    error->all(FLERR,
               "Pair style tip4p/long/dprc/batch requires atom IDs");
  if (!force->newton_pair)
    error->all(FLERR,
               "Pair style tip4p/long/dprc/batch requires newton pair on");
  if (!atom->q_flag)
    error->all(FLERR,
               "Pair style tip4p/long/dprc/batch requires per-atom charge");
  if (!force->bond || !force->angle)
    error->all(FLERR,
               "Pair style tip4p/long/dprc/batch requires bond and angle styles");
  if (!force->kspace)
    error->all(FLERR,
               "Pair style tip4p/long/dprc/batch requires a KSpace style");

  cut_coulsq = cut_coul * cut_coul;
  g_ewald = force->kspace->g_ewald;
  if (ncoultablebits) init_tables(cut_coul, nullptr);

  const double theta = force->angle->equilibrium_angle(typeA);
  const double bond_length = force->bond->equilibrium_distance(typeB);
  alpha = qdist / (std::cos(0.5 * theta) * bond_length);
  const double minimum_comm = cut_coul + qdist + bond_length + neighbor->skin;
  if (comm->get_comm_cutoff() < minimum_comm) comm->cutghostuser = minimum_comm;
  // Deliberately do not call PairCoulLong::init_style(): its neighbor request
  // would retain a redundant per-window real-space list.
}

void PairDPRCBatchedTIP4PLong::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (eflag_atom || vflag_atom || cvflag_atom || num_tally_compute > 0)
    error->all(
        FLERR,
        "Pair style tip4p/long/dprc/batch supports only global pair tallies");
  auto *batched = dynamic_cast<PPPMTIP4PDPRCBatched *>(force->kspace);
  if (!batched)
    error->all(
        FLERR,
        "Pair style tip4p/long/dprc/batch requires pppm/tip4p/dprc/batch");
  batched->consume_coulomb_publication(eflag, vflag, eng_coul);
}

void PairDPRCBatchedTIP4PLong::export_parameters(
    DPRC::ClassicalTopology &topology) const {
  topology.tip4p_alpha = alpha;
  topology.tip4p_qdist = qdist;
  topology.real_space_cutoff = cut_coul;

  auto &table = topology.coulomb_table;
  table.bits = ncoultablebits;
  table.shift_bits = ncoulshiftbits;
  table.mask = ncoulmask;
  table.inner_squared = tabinnersq;
  if (!ncoultablebits) return;
  const std::size_t count = std::size_t{1} << ncoultablebits;
  table.r.assign(rtable, rtable + count);
  table.dr.assign(drtable, drtable + count);
  table.force.assign(ftable, ftable + count);
  table.dforce.assign(dftable, dftable + count);
  table.coulomb.assign(ctable, ctable + count);
  table.dcoulomb.assign(dctable, dctable + count);
  table.energy.assign(etable, etable + count);
  table.denergy.assign(detable, detable + count);
}
