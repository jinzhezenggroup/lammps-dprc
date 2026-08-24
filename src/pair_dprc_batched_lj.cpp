#include "pair_dprc_batched_lj.h"

#include "kspace_dprc_batched.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cstddef>

using namespace LAMMPS_NS;

PairDPRCBatchedLJ::PairDPRCBatchedLJ(LAMMPS *lmp) : PairLJCut(lmp) {
  // TIP4P redistribution means the complete pair virial cannot in general be
  // reconstructed from the final atomic force array.  Force PairHybrid to use
  // the explicit six-component publication supplied by the broker.
  no_virial_fdotr_compute = 1;
}

void PairDPRCBatchedLJ::init_style() {
  if (update->whichflag == 1 &&
      utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Pair style lj/cut/dprc/batch does not support r-RESPA");
  // No neighbor request is made: the broker rebuilds one fused GPU cell list
  // for LJ and MM-only real-space Coulomb across all windows.
  cut_respa = nullptr;
}

void PairDPRCBatchedLJ::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (eflag_atom || vflag_atom || cvflag_atom || num_tally_compute > 0)
    error->all(FLERR,
               "Pair style lj/cut/dprc/batch supports only global pair tallies");
  auto *batched = dynamic_cast<PPPMTIP4PDPRCBatched *>(force->kspace);
  if (!batched)
    error->all(FLERR,
               "Pair style lj/cut/dprc/batch requires pppm/tip4p/dprc/batch");
  batched->consume_lj_publication(eflag, vflag, eng_vdwl, virial);
}

void PairDPRCBatchedLJ::export_parameters(
    DPRC::ClassicalTopology &topology,
    const std::vector<std::uint8_t> &enabled_pairs) const {
  const std::size_t types = static_cast<std::size_t>(atom->ntypes);
  if (enabled_pairs.size() != types * types)
    error->all(FLERR,
               "Invalid lj/cut/dprc/batch type-pair export mask");
  topology.type_count = atom->ntypes;
  topology.lj.assign(types * types, DPRC::LennardJonesParameters{});
  for (int itype = 1; itype <= atom->ntypes; ++itype)
    for (int jtype = 1; jtype <= atom->ntypes; ++jtype) {
      const std::size_t index =
          static_cast<std::size_t>(itype - 1) * types +
          static_cast<std::size_t>(jtype - 1);
      if (!enabled_pairs[index])
        continue;
      auto &entry = topology.lj[index];
      // PairLJCut mirrors its expanded force/energy coefficients into the
      // lower triangle, but intentionally leaves cut[j][i] untouched.  Read
      // the initialized upper triangle for every exported entry so canonical
      // topology bytes do not depend on allocator contents.
      const int lower_type = std::min(itype, jtype);
      const int upper_type = std::max(itype, jtype);
      entry.lj1 = lj1[lower_type][upper_type];
      entry.lj2 = lj2[lower_type][upper_type];
      entry.lj3 = lj3[lower_type][upper_type];
      entry.lj4 = lj4[lower_type][upper_type];
      entry.offset = offset[lower_type][upper_type];
      entry.cutoff = cut[lower_type][upper_type];
    }
}
