#include "kspace_dprc_batched.h"

#include "pair_dprc_batched_lj.h"
#include "pair_dprc_batched_tip4p.h"
#include "stable_local_indices.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "pair_hybrid.h"
#include "universe.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef DPRC_XTBLOOM_DEVICE_ID
#define DPRC_XTBLOOM_DEVICE_ID -1
#endif

using namespace LAMMPS_NS;

namespace {

[[nodiscard]] std::size_t checked_atom_count(bigint count) {
  if (count <= 0 || static_cast<unsigned long long>(count) >
                        std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("LAMMPS atom count is outside the classical batch range");
  return static_cast<std::size_t>(count);
}

} // namespace

PPPMTIP4PDPRCBatched::PPPMTIP4PDPRCBatched(LAMMPS *lmp)
    : PPPMTIP4PDPRC(lmp) {
  // The backend implements the pinned ik equations for restricted triclinic
  // boxes, but not PPPM's group/group or per-atom auxiliary interfaces.
  group_group_enable = 0;
  triclinic_support = 1;
}

PPPMTIP4PDPRCBatched::~PPPMTIP4PDPRCBatched() {
  broker_.reset();
  roots_.reset();
}

void PPPMTIP4PDPRCBatched::init() {
  if (comm->nprocs != 1)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires exactly one MPI rank per window");
  if (domain->dimension != 3 || !domain->xperiodic || !domain->yperiodic ||
      !domain->zperiodic)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires fully periodic 3-D boxes");
  if (!atom->q_flag || !atom->tag_enable || atom->map_style == 0)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires charges, atom IDs, and an atom map");
  if (!force->newton || !force->newton_pair)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires newton and newton pair on");
  if (differentiation_flag != 0 || slabflag != 0 || stagger_flag != 0)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch supports only ik, non-staggered, non-slab PPPM");
  if (!gridflag || !gewaldflag || nx_pppm <= 1 || ny_pppm <= 1 ||
      nz_pppm <= 1 || order < 2 || order > 8 || !(g_ewald > 0.0))
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires explicit mesh, order, and gewald values");

  triclinic = domain->triclinic;
  qdist = 0.0;
  init_tip4p();
  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;
  volume = domain->xprd * domain->yprd * domain->zprd;

  try {
    if (!roots_)
      roots_ = std::make_unique<DPRC::PartitionRoots>(
          universe->uworld, comm->me, universe->iworld, universe->nworlds);
  } catch (const std::exception &exception) {
    error->universe_all(FLERR, exception.what());
  }
  if (!roots_ || !roots_->is_root())
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires every window rank to be a partition root");

  ++init_epoch_;
  discard_fused_full_solve();
  if (comm->me == 0)
    utils::logmesg(
        lmp,
        "Batched DPRc classical backend: one rank/window, {} synchronized GPU slots, "
        "mesh {} {} {}, order {}, gewald {:.16g}\n",
        roots_->size(), nx_pppm, ny_pppm, nz_pppm, order, g_ewald);
}

void PPPMTIP4PDPRCBatched::setup() {
  if (broker_epoch_ == init_epoch_)
    return;
  build_broker();
  broker_epoch_ = init_epoch_;
}

void PPPMTIP4PDPRCBatched::require_roots_collectively(
    bool condition, const char *diagnostic) const {
  const int local = condition ? 1 : 0;
  int global = 0;
  if (!roots_ ||
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                    roots_->communicator()) != MPI_SUCCESS)
    error->universe_all(
        FLERR,
        "Batched DPRc classical validation collective failed");
  if (global == 0)
    error->universe_all(FLERR, diagnostic);
}

void PPPMTIP4PDPRCBatched::validate_proxy_configuration() const {
  auto *lj = dynamic_cast<PairDPRCBatchedLJ *>(
      force->pair_match("lj/cut/dprc/batch", 1, 0));
  auto *tip4p = dynamic_cast<PairDPRCBatchedTIP4PLong *>(
      force->pair_match("tip4p/long/dprc/batch", 1, 0));
  require_roots_collectively(
      lj && tip4p && lj->compute_flag && tip4p->compute_flag,
      "KSpace style pppm/tip4p/dprc/batch requires enabled lj/cut/dprc/batch and tip4p/long/dprc/batch proxies in every window");
}

std::vector<std::uint8_t>
PPPMTIP4PDPRCBatched::pair_mapping(Pair *substyle) const {
  const std::size_t types = static_cast<std::size_t>(atom->ntypes);
  std::vector<std::uint8_t> enabled(types * types, 0);
  if (force->pair == substyle) {
    std::fill(enabled.begin(), enabled.end(), 1);
    return enabled;
  }
  auto *hybrid = dynamic_cast<PairHybrid *>(force->pair);
  char *keyword = force->pair_match_ptr(substyle);
  if (!hybrid || !keyword)
    error->all(FLERR,
               "Batched DPRc pair proxies require a direct or hybrid LAMMPS pair mapping");
  for (int itype = 1; itype <= atom->ntypes; ++itype)
    for (int jtype = 1; jtype <= atom->ntypes; ++jtype)
      enabled[static_cast<std::size_t>(itype - 1) * types +
              static_cast<std::size_t>(jtype - 1)] =
          hybrid->check_ijtype(itype, jtype, keyword) > 0 ? 1 : 0;
  return enabled;
}

void PPPMTIP4PDPRCBatched::build_stable_atom_order() {
  const std::size_t atoms = checked_atom_count(atom->natoms);
  if (atom->nlocal < 0 || static_cast<std::size_t>(atom->nlocal) != atoms)
    error->all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires every atom to be local on its one-rank window");

  stable_tags_.resize(atoms);
  for (std::size_t atom_index = 0; atom_index < atoms; ++atom_index)
    stable_tags_[atom_index] = atom->tag[atom_index];
  std::sort(stable_tags_.begin(), stable_tags_.end());
  if (stable_tags_.front() <= 0 ||
      std::adjacent_find(stable_tags_.begin(), stable_tags_.end()) !=
          stable_tags_.end())
    error->all(FLERR,
               "KSpace style pppm/tip4p/dprc/batch requires unique positive atom IDs");
  const bool mapping_ok = refresh_stable_local_indices();
  // Keep setup failures collective: a rank-local mapping failure must not
  // enter an error barrier while its peers are still entering a validation
  // reduction below.
  require_roots_collectively(
      mapping_ok,
      "KSpace style pppm/tip4p/dprc/batch could not map every stable atom ID to one owned atom");
}

bool PPPMTIP4PDPRCBatched::refresh_stable_local_indices() const {
  // LAMMPS's global atom map includes periodic ghosts.  Rebuild the stable
  // slot map from the owned prefix after atom sorting instead of relying on
  // which equal-tag image the global map currently exposes.
  auto rebuilt = DPRC::stable_local_indices(
      stable_tags_, atom->tag, static_cast<std::size_t>(atom->nlocal));
  if (!rebuilt) {
    stable_local_indices_.clear();
    return false;
  }
  stable_local_indices_ = std::move(*rebuilt);
  return true;
}

void PPPMTIP4PDPRCBatched::append_special_pairs(
    DPRC::ClassicalTopology &topology) const {
  using PairKey = std::pair<tagint, tagint>;
  std::map<PairKey, std::pair<double, double>> special;
  for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
    const int index = atom_index_for_slot(slot);
    if (!atom->nspecial || !atom->special)
      continue;
    for (int entry = 0; entry < atom->nspecial[index][2]; ++entry) {
      const tagint partner = atom->special[index][entry];
      const PairKey key = std::minmax(stable_tags_[slot], partner);
      if (key.first == key.second)
        error->all(FLERR, "Invalid self special pair in batched topology");
      int category = 3;
      if (entry < atom->nspecial[index][0])
        category = 1;
      else if (entry < atom->nspecial[index][1])
        category = 2;
      const std::pair<double, double> scales = {
          force->special_lj[category], force->special_coul[category]};
      const auto [position, inserted] = special.emplace(key, scales);
      if (!inserted && position->second != scales)
        error->all(FLERR,
                   "Inconsistent special-pair scaling in batched topology");
    }
  }

  topology.special_pairs.clear();
  topology.special_pairs.reserve(special.size());
  for (const auto &[tags, scales] : special) {
    const auto first =
        std::lower_bound(stable_tags_.begin(), stable_tags_.end(), tags.first);
    const auto second =
        std::lower_bound(stable_tags_.begin(), stable_tags_.end(), tags.second);
    if (first == stable_tags_.end() || *first != tags.first ||
        second == stable_tags_.end() || *second != tags.second)
      error->all(FLERR,
                 "Special-pair atom is missing from batched topology");
    topology.special_pairs.push_back(
        {static_cast<std::int32_t>(
             std::distance(stable_tags_.begin(), first)),
         static_cast<std::int32_t>(
             std::distance(stable_tags_.begin(), second)),
         scales.first, scales.second});
  }
}

DPRC::ClassicalTopology PPPMTIP4PDPRCBatched::build_topology() const {
  validate_proxy_configuration();
  auto *lj = dynamic_cast<PairDPRCBatchedLJ *>(
      force->pair_match("lj/cut/dprc/batch", 1, 0));
  auto *tip4p = dynamic_cast<PairDPRCBatchedTIP4PLong *>(
      force->pair_match("tip4p/long/dprc/batch", 1, 0));

  DPRC::ClassicalTopology topology;
  // Stable tags are initialized by build_broker() before this const helper is
  // used; repeated setup validation never changes that ordering.
  topology.atom_count = stable_tags_.size();
  topology.atom_types.resize(stable_tags_.size());
  for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
    const int index = atom_index_for_slot(slot);
    topology.atom_types[slot] = atom->type[index] - 1;
  }
  topology.type_count = atom->ntypes;
  topology.coulomb_type_pairs = pair_mapping(tip4p);
  const std::vector<std::uint8_t> lj_mapping = pair_mapping(lj);
  lj->export_parameters(topology, lj_mapping);
  tip4p->export_parameters(topology);

  topology.cell.boxlo = {domain->boxlo[0], domain->boxlo[1],
                         domain->boxlo[2]};
  topology.cell.h = {domain->h[0], domain->triclinic ? domain->h[5] : 0.0,
                     domain->triclinic ? domain->h[4] : 0.0, 0.0,
                     domain->h[1], domain->triclinic ? domain->h[3] : 0.0,
                     0.0, 0.0, domain->h[2]};
  topology.pppm.mesh = {nx_pppm, ny_pppm, nz_pppm};
  topology.pppm.order = order;
  topology.pppm.g_ewald = g_ewald;
  topology.neighbor_skin = neighbor->skin;
  topology.qqrd2e = force->qqrd2e;

  topology.tip4p_sites.clear();
  for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
    if (topology.atom_types[slot] + 1 != tip4p->oxygen_type())
      continue;
    const tagint oxygen = stable_tags_[slot];
    const auto first = std::lower_bound(stable_tags_.begin(), stable_tags_.end(),
                                        oxygen + 1);
    const auto second = std::lower_bound(stable_tags_.begin(), stable_tags_.end(),
                                         oxygen + 2);
    if (first == stable_tags_.end() || *first != oxygen + 1 ||
        second == stable_tags_.end() || *second != oxygen + 2)
      error->all(FLERR, "TIP4P hydrogen is missing from batched topology");
    const std::size_t first_slot = static_cast<std::size_t>(
        std::distance(stable_tags_.begin(), first));
    const std::size_t second_slot = static_cast<std::size_t>(
        std::distance(stable_tags_.begin(), second));
    if (topology.atom_types[first_slot] + 1 != tip4p->hydrogen_type() ||
        topology.atom_types[second_slot] + 1 != tip4p->hydrogen_type())
      error->all(FLERR,
                 "TIP4P hydrogen has incorrect atom type in batched topology");
    topology.tip4p_sites.push_back(
        {static_cast<std::int32_t>(slot),
         static_cast<std::int32_t>(first_slot),
         static_cast<std::int32_t>(second_slot)});
  }
  append_special_pairs(topology);
  return topology;
}

void PPPMTIP4PDPRCBatched::build_broker() {
  build_stable_atom_order();
  DPRC::ClassicalTopology topology = build_topology();

  // The CUDA plan owns cell-, type-, and special-topology-dependent data.
  // Preserve a canonical snapshot so an in-run mutation fails closed instead
  // of silently using the old Green function or pair exclusions.  An explicit
  // change_box/set/delete_bonds command between runs is allowed because the
  // next KSpace::init/setup epoch rebuilds this snapshot and the broker.
  stable_atom_types_ = topology.atom_types;
  stable_special_pairs_ = topology.special_pairs;
  stable_cell_ = topology.cell;
  stable_type_count_ = topology.type_count;

  frame_positions_.resize(3u * topology.atom_count);
  frame_charges_.resize(topology.atom_count);
  DPRC::ClassicalPlanOptions options;
  options.backend = DPRC::ClassicalBackend::CUDA;
  options.max_batch_count = static_cast<std::size_t>(roots_->size());
  options.cuda_device = DPRC_XTBLOOM_DEVICE_ID;

  try {
    broker_.reset();
    broker_ = std::make_unique<DPRC::ClassicalPartitionBroker>(
        roots_->communicator(), std::move(topology), options);
  } catch (const std::exception &exception) {
    error->universe_all(FLERR, exception.what());
  }
}

int PPPMTIP4PDPRCBatched::atom_index_for_slot(std::size_t slot) const {
  const int index = stable_local_indices_.at(slot);
  if (index < 0 || index >= atom->nlocal ||
      atom->tag[index] != stable_tags_.at(slot))
    error->universe_all(
        FLERR,
        "Stable batched atom-to-local-index cache is stale within a neighbor epoch");
  return index;
}

void PPPMTIP4PDPRCBatched::validate_fixed_state() const {
  std::array<int, 4> local{};
  local[0] = atom->natoms == static_cast<bigint>(stable_tags_.size()) &&
          atom->nlocal == static_cast<int>(stable_tags_.size()) &&
          atom->ntypes == stable_type_count_
      ? 1
      : 0;

  // LAMMPS rebuilds its neighbor data whenever an in-run fix changes atom
  // identity, type, or bonded special topology.  Reconstructing the complete
  // canonical special-pair map on every force call was therefore redundant
  // O(N log N) work.  Keep the cheap count/cell checks on every call and run
  // the full immutable-topology audit at each neighbor rebuild boundary.
  const bool audit_topology = neighbor->ago == 0;
  bool tags_and_types_match = true;
  if (audit_topology) {
    tags_and_types_match = refresh_stable_local_indices();
    if (tags_and_types_match) {
      for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
        const int index = atom_index_for_slot(slot);
        if (atom->type[index] - 1 != stable_atom_types_[slot]) {
          tags_and_types_match = false;
          break;
        }
      }
    }
  }
  local[1] = tags_and_types_match ? 1 : 0;

  const DPRC::RestrictedTriclinicCell current_cell{
      {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2]},
      {domain->h[0], domain->triclinic ? domain->h[5] : 0.0,
       domain->triclinic ? domain->h[4] : 0.0, 0.0, domain->h[1],
       domain->triclinic ? domain->h[3] : 0.0, 0.0, 0.0, domain->h[2]}};
  local[2] = current_cell.boxlo == stable_cell_.boxlo &&
          current_cell.h == stable_cell_.h
      ? 1
      : 0;

  bool special_pairs_match = true;
  if (audit_topology) {
    // Do not dereference the cache after a local mapping failure.  Every rank
    // must reach the validation reduction before the collective error path;
    // otherwise one rank can enter LAMMPS's error barrier while its peers are
    // still inside MPI_Allreduce, which manifests as a silent MPI hang.
    if (!tags_and_types_match) {
      special_pairs_match = false;
    } else {
      DPRC::ClassicalTopology current;
      append_special_pairs(current);
      special_pairs_match =
          current.special_pairs.size() == stable_special_pairs_.size();
      for (std::size_t index = 0;
           special_pairs_match && index < stable_special_pairs_.size(); ++index) {
        const auto &expected = stable_special_pairs_[index];
        const auto &actual = current.special_pairs[index];
        special_pairs_match =
            actual.atom1 == expected.atom1 && actual.atom2 == expected.atom2 &&
            actual.lj_scale == expected.lj_scale &&
            actual.coulomb_scale == expected.coulomb_scale;
      }
    }
  }
  local[3] = special_pairs_match ? 1 : 0;

  std::array<int, 4> global{};
  if (!roots_ ||
      MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()),
                    MPI_INT, MPI_MIN, roots_->communicator()) != MPI_SUCCESS)
    error->universe_all(
        FLERR, "Batched DPRc classical validation collective failed");
  if (global[0] == 0)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires a fixed atom and type topology within each run");
  if (global[1] == 0)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires fixed atom IDs and types within each run");
  if (global[2] == 0)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires a fixed cell within each run");
  if (global[3] == 0)
    error->universe_all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch requires fixed special-bond topology within each run");
}

void PPPMTIP4PDPRCBatched::pack_frame(std::vector<double> &positions,
                                     std::vector<double> &charges) const {
  validate_fixed_state();
  if (positions.size() != 3u * stable_tags_.size() ||
      charges.size() != stable_tags_.size())
    error->all(FLERR, "Batched classical frame storage has invalid extent");
  for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
    const int index = atom_index_for_slot(slot);
    for (int dim = 0; dim < 3; ++dim)
      positions[3u * slot + static_cast<std::size_t>(dim)] =
          atom->x[index][dim];
    charges[slot] = atom->q[index];
  }
}

void PPPMTIP4PDPRCBatched::publish_forces(const double *forces) const {
  for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
    const int index = atom_index_for_slot(slot);
    for (int dim = 0; dim < 3; ++dim)
      atom->f[index][dim] +=
          forces[3u * slot + static_cast<std::size_t>(dim)];
  }
}

void PPPMTIP4PDPRCBatched::publish_kspace(
    const double *forces, double published_energy,
    const double *published_virial, int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (evflag_atom)
    error->all(
        FLERR,
        "KSpace style pppm/tip4p/dprc/batch supports only global energy and virial");
  // Private QM/MM capture reads these values on every step.  Pure-classical
  // publication follows ordinary LAMMPS tally semantics and only writes
  // energy/virial when their global flags were requested.
  const bool pure_classical = publication_kind_ == PublicationKind::MmOnly;
  if (!pure_classical || eflag_global)
    energy = published_energy;
  if (!pure_classical || vflag_global)
    for (int component = 0; component < 6; ++component)
      virial[component] = published_virial[component];
  if (forces)
    publish_forces(forces);
}

void PPPMTIP4PDPRCBatched::compute(int eflag, int vflag) {
  if (!broker_ || broker_epoch_ != init_epoch_)
    error->universe_all(
        FLERR,
        "Batched DPRc KSpace compute reached an uninitialized classical broker");

  try {
    if (stage_ == CaptureStage::Pending) {
      if (kspace_consumed_)
        throw std::logic_error(
            "batched full PPPM publication was consumed more than once");
      validate_publication_phase(vflag, false);
      if (publication_kind_ == PublicationKind::FullQmMm) {
        const DPRC::ClassicalQmResultView result = broker_->qm_result();
        publish_kspace(result.full_pppm_forces, result.full_pppm_energy,
                       result.full_pppm_virial, eflag, vflag);
      } else if (publication_kind_ == PublicationKind::MmOnly) {
        const DPRC::ClassicalMmResultView result = broker_->mm_result();
        if (!result.mm_pppm_forces)
          throw std::logic_error(
              "batched pure-classical publication lacks MM PPPM forces");
        publish_kspace(result.mm_pppm_forces, result.mm_pppm_energy,
                       result.mm_pppm_virial, eflag, vflag);
      } else {
        throw std::logic_error(
            "batched PPPM publication has no result kind");
      }
      kspace_consumed_ = true;
      maybe_complete_publication();
      return;
    }

    if (!capture_armed_)
      throw std::logic_error(
          "batched PPPM compute was not armed by the QM/MM pre-force transaction");

    if (stage_ == CaptureStage::Idle) {
      pack_frame(frame_positions_, frame_charges_);
      broker_->begin_mm({static_cast<std::int64_t>(update->ntimestep),
                         frame_positions_.data(), frame_positions_.size(),
                         frame_charges_.data(), frame_charges_.size()});
      const DPRC::ClassicalMmResultView result = broker_->mm_result();
      publish_kspace(nullptr, result.mm_pppm_energy, result.mm_pppm_virial,
                     eflag, vflag);
      stage_ = CaptureStage::MmReady;
      return;
    }

    if (stage_ == CaptureStage::MmCached) {
      pack_frame(frame_positions_, frame_charges_);
      broker_->finish_qm({static_cast<std::int64_t>(update->ntimestep),
                          frame_charges_.data(), frame_charges_.size()});
      const DPRC::ClassicalQmResultView result = broker_->qm_result();
      publish_kspace(result.qm_pppm_forces, result.qm_pppm_energy,
                     result.qm_pppm_virial, eflag, vflag);
      stage_ = CaptureStage::QmReady;
      return;
    }

    throw std::logic_error(
        "batched PPPM capture reached an invalid MM/QM stage");
  } catch (const std::exception &exception) {
    broker_->cancel();
    error->universe_all(FLERR, exception.what());
  }
}

void PPPMTIP4PDPRCBatched::compute_group_potential(double *, int, int, bool) {
  error->all(
      FLERR,
      "KSpace style pppm/tip4p/dprc/batch requires reuse of the retained batched MM solve");
}

void PPPMTIP4PDPRCBatched::project_last_solve_potential(
    double *potential, int sensor_groupbit) {
  if (!potential || stage_ != CaptureStage::MmReady)
    error->all(FLERR,
               "No retained batched MM potential is available for projection");
  const DPRC::ClassicalMmResultView result = broker_->mm_result();
  if (result.timestep != static_cast<std::int64_t>(update->ntimestep))
    error->all(FLERR,
               "Retained batched MM potential belongs to another timestep");
  for (std::size_t slot = 0; slot < stable_tags_.size(); ++slot) {
    const int index = atom_index_for_slot(slot);
    if (atom->mask[index] & sensor_groupbit)
      potential[index] = result.mm_pppm_potential[slot];
  }
}

void PPPMTIP4PDPRCBatched::cache_last_solve_as_mm() {
  if (stage_ != CaptureStage::MmReady)
    error->all(FLERR, "Cannot cache an unavailable batched MM PPPM solve");
  stage_ = CaptureStage::MmCached;
}

void PPPMTIP4PDPRCBatched::prepare_fused_full_solve() {
  if (stage_ != CaptureStage::QmReady)
    error->all(FLERR,
               "Cannot prepare an unavailable batched full PPPM result");
  stage_ = CaptureStage::Prepared;
}

void PPPMTIP4PDPRCBatched::commit_fused_full_solve(
    bigint timestep, int vflag, int whichflag, int setupflag) {
  if (stage_ != CaptureStage::Prepared || (whichflag != 1 && whichflag != 2))
    error->all(FLERR,
               "Cannot commit an unprepared batched full PPPM result");
  token_ = {timestep, vflag, whichflag, setupflag};
  lj_consumed_ = false;
  coulomb_consumed_ = false;
  kspace_consumed_ = false;
  publication_kind_ = PublicationKind::FullQmMm;
  stage_ = CaptureStage::Pending;
}

void PPPMTIP4PDPRCBatched::discard_fused_full_solve() noexcept {
  if (broker_)
    broker_->cancel();
  stage_ = CaptureStage::Idle;
  publication_kind_ = PublicationKind::None;
  token_ = {};
  lj_consumed_ = false;
  coulomb_consumed_ = false;
  kspace_consumed_ = false;
  capture_armed_ = true;
}

void PPPMTIP4PDPRCBatched::prepare_classical_publication(int vflag) {
  validate_proxy_configuration();
  if (stage_ == CaptureStage::Pending)
    return;
  if (stage_ != CaptureStage::Idle)
    error->universe_all(
        FLERR,
        "Pure-classical pair publication collided with an active QM/MM transaction");

  try {
    pack_frame(frame_positions_, frame_charges_);
    broker_->begin_mm(
        {static_cast<std::int64_t>(update->ntimestep),
        frame_positions_.data(), frame_positions_.size(), frame_charges_.data(),
         frame_charges_.size()},
        {/*pppm_potential=*/false, /*pppm_forces=*/true,
         /*retain_for_qm=*/false});
  } catch (const std::exception &exception) {
    broker_->cancel();
    error->universe_all(FLERR, exception.what());
  }

  token_ = {update->ntimestep, vflag, update->whichflag, update->setupflag};
  lj_consumed_ = false;
  coulomb_consumed_ = false;
  kspace_consumed_ = false;
  publication_kind_ = PublicationKind::MmOnly;
  stage_ = CaptureStage::Pending;
  capture_armed_ = false;
}

int PPPMTIP4PDPRCBatched::get_charge_site(int index, double *site,
                                          int *indices, double *weights) {
  return PPPMTIP4PDPRC::get_charge_site(index, site, indices, weights);
}

void PPPMTIP4PDPRCBatched::validate_publication_phase(
    int vflag, bool pair_phase) const {
  if (update->ntimestep != token_.timestep ||
      update->whichflag != token_.whichflag ||
      update->setupflag != token_.setupflag)
    error->all(FLERR,
               "Batched classical publication reached an unexpected LAMMPS phase");
  if (!pair_phase && vflag != token_.vflag)
    error->all(FLERR,
               "Batched PPPM publication reached an unexpected virial request");
}

void PPPMTIP4PDPRCBatched::consume_lj_publication(
    int eflag, int vflag, double &published_energy, double *published_virial) {
  if (stage_ != CaptureStage::Pending || lj_consumed_)
    error->all(FLERR,
               "No unconsumed batched LJ publication is available");
  validate_publication_phase(vflag, true);
  const DPRC::ClassicalMmResultView result = broker_->mm_result();
  if (result.timestep != static_cast<std::int64_t>(token_.timestep))
    error->all(FLERR, "Batched LJ publication belongs to another timestep");
  publish_forces(result.pair_forces);
  if (eflag)
    published_energy = result.lj_energy;
  if (vflag)
    for (int component = 0; component < 6; ++component)
      published_virial[component] = result.pair_virial[component];
  lj_consumed_ = true;
  maybe_complete_publication();
}

void PPPMTIP4PDPRCBatched::consume_coulomb_publication(
    int eflag, int vflag, double &published_energy) {
  if (stage_ != CaptureStage::Pending || coulomb_consumed_)
    error->all(FLERR,
               "No unconsumed batched TIP4P Coulomb publication is available");
  validate_publication_phase(vflag, true);
  const DPRC::ClassicalMmResultView result = broker_->mm_result();
  if (result.timestep != static_cast<std::int64_t>(token_.timestep))
    error->all(FLERR,
               "Batched TIP4P Coulomb publication belongs to another timestep");
  if (eflag)
    published_energy = result.coulomb_energy;
  coulomb_consumed_ = true;
  maybe_complete_publication();
}

void PPPMTIP4PDPRCBatched::maybe_complete_publication() noexcept {
  if (!lj_consumed_ || !coulomb_consumed_ || !kspace_consumed_)
    return;
  const bool pure_classical = publication_kind_ == PublicationKind::MmOnly;
  stage_ = CaptureStage::Idle;
  publication_kind_ = PublicationKind::None;
  // A pure-classical run has no fix transaction that calls discard before the
  // next timestep.  Re-arm it here; QM/MM deliberately remains disarmed until
  // its wrapper starts the next transaction.
  capture_armed_ = pure_classical;
}

double PPPMTIP4PDPRCBatched::memory_usage() {
  return static_cast<double>(stable_tags_.capacity() * sizeof(tagint) +
                             stable_local_indices_.capacity() * sizeof(int) +
                             (frame_positions_.capacity() +
                              frame_charges_.capacity()) *
                                 sizeof(double));
}

#undef DPRC_XTBLOOM_DEVICE_ID
