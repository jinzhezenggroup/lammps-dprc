#include "pair_dprc_deepmd_batch.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "universe.h"
#include "utils.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

namespace {

bool is_keyword(const char *token) {
  return std::strcmp(token, "center_group") == 0 ||
         std::strcmp(token, "environment_cutoff") == 0 ||
         std::strcmp(token, "include_molecule") == 0 ||
         std::strcmp(token, "partition_batch") == 0 ||
         std::strcmp(token, "device") == 0;
}

std::vector<std::string> split_words(const std::string &input) {
  std::istringstream stream(input);
  std::vector<std::string> words;
  for (std::string word; stream >> word;)
    words.push_back(std::move(word));
  return words;
}

}  // namespace

PairDPRCDeepMDBatch::PairDPRCDeepMDBatch(LAMMPS *lmp) : Pair(lmp) {
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  restartinfo = 0;
  respa_enable = 0;
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_AVAIL;
}

PairDPRCDeepMDBatch::~PairDPRCDeepMDBatch() {
  broker_.reset();
  roots_.reset();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale_);
  }
}

void PairDPRCDeepMDBatch::allocate() {
  allocated = 1;
  const int extent = atom->ntypes + 1;
  memory->create(setflag, extent, extent, "dprc/deepmd:setflag");
  memory->create(cutsq, extent, extent, "dprc/deepmd:cutsq");
  memory->create(scale_, extent, extent, "dprc/deepmd:scale");
  for (int i = 1; i < extent; ++i)
    for (int j = i; j < extent; ++j) {
      setflag[i][j] = 0;
      scale_[i][j] = 1.0;
    }
}

void PairDPRCDeepMDBatch::settings(int narg, char **arg) {
  if (narg < 1 || is_keyword(arg[0]))
    utils::missing_cmd_args(FLERR, "pair_style dprc/deepmd/batch", error);
  if (comm->nprocs != 1)
    error->universe_all(
        FLERR,
        "dprc/deepmd/batch requires exactly one MPI rank per LAMMPS partition");
  if (std::strcmp(update->unit_style, "lj") == 0)
    error->universe_all(FLERR,
                        "dprc/deepmd/batch does not support reduced LJ units");

  model_path_ = arg[0];
  center_group_id_.clear();
  environment_cutoff_ = 0.0;
  include_molecule_ = true;
  gpu_rank_ = 0;
  bool partition_batch = true;
  bool center_group_set = false;
  bool environment_cutoff_set = false;
  int index = 1;
  while (index < narg) {
    const std::string keyword = arg[index];
    if (keyword == "center_group") {
      if (center_group_set || index + 1 >= narg || is_keyword(arg[index + 1]))
        error->universe_all(
            FLERR, "center_group requires exactly one static group ID");
      center_group_id_ = arg[index + 1];
      center_group_set = true;
      index += 2;
    } else if (keyword == "environment_cutoff") {
      if (environment_cutoff_set || index + 1 >= narg)
        error->universe_all(
            FLERR, "environment_cutoff requires exactly one value");
      environment_cutoff_ = utils::numeric(FLERR, arg[index + 1], false, lmp);
      if (!std::isfinite(environment_cutoff_) || environment_cutoff_ <= 0.0)
        error->universe_all(
            FLERR, "environment_cutoff must be finite and greater than zero");
      environment_cutoff_set = true;
      index += 2;
    } else if (keyword == "include_molecule") {
      if (index + 1 >= narg)
        error->universe_all(FLERR,
                            "include_molecule requires an explicit yes/no value");
      include_molecule_ =
          utils::logical(FLERR, arg[index + 1], false, lmp) != 0;
      index += 2;
    } else if (keyword == "partition_batch") {
      if (index + 1 >= narg)
        error->universe_all(FLERR,
                            "partition_batch requires an explicit yes/no value");
      partition_batch =
          utils::logical(FLERR, arg[index + 1], false, lmp) != 0;
      index += 2;
    } else if (keyword == "device") {
      if (index + 1 >= narg)
        error->universe_all(FLERR, "device requires a non-negative GPU rank");
      gpu_rank_ = utils::inumeric(FLERR, arg[index + 1], false, lmp);
      if (gpu_rank_ < 0)
        error->universe_all(FLERR, "device must be non-negative");
      index += 2;
    } else {
      error->universe_all(FLERR,
                          "Unknown dprc/deepmd/batch keyword " + keyword);
    }
  }
  if (!center_group_set || !environment_cutoff_set)
    error->universe_all(
        FLERR,
        "dprc/deepmd/batch requires center_group and environment_cutoff");
  if (!partition_batch)
    error->universe_all(
        FLERR,
        "dprc/deepmd/batch is intrinsically partition-batched; use "
        "partition_batch yes");

  distance_unit_factor_ = force->angstrom;
  energy_unit_factor_ = force->boltz / 8.617343e-5;
  force_unit_factor_ = energy_unit_factor_ / distance_unit_factor_;

  broker_.reset();
  roots_.reset();
  try {
    roots_ = std::make_unique<DPRC::PartitionRoots>(
        universe->uworld, comm->me, universe->iworld, universe->nworlds);
    broker_ = std::make_unique<DPRC::DeepmdPartitionBroker>(
        roots_->communicator(), model_path_, gpu_rank_);
  } catch (const std::exception &exception) {
    error->universe_all(FLERR, exception.what());
  }
  model_cutoff_ = broker_->metadata().cutoff * distance_unit_factor_;
}

std::vector<std::string> PairDPRCDeepMDBatch::model_type_names() const {
  return split_words(broker_->metadata().type_map);
}

void PairDPRCDeepMDBatch::coeff(int narg, char **arg) {
  if (broker_ == nullptr)
    error->all(FLERR,
               "pair_style dprc/deepmd/batch must be set before pair_coeff");
  if (narg < 2)
    utils::missing_cmd_args(FLERR, "pair_coeff dprc/deepmd/batch", error);
  int ilo = 0;
  int ihi = 0;
  int jlo = 0;
  int jhi = 0;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  if (ilo != 1 || ihi != atom->ntypes || jlo != 1 || jhi != atom->ntypes)
    error->all(FLERR,
               "dprc/deepmd/batch requires pair_coeff * * for all atom types");
  if (narg != atom->ntypes + 2)
    error->all(FLERR,
               "dprc/deepmd/batch pair_coeff requires one element or NULL "
               "for every LAMMPS atom type");
  if (!allocated)
    allocate();

  const std::vector<std::string> model_types = model_type_names();
  if (static_cast<int>(model_types.size()) != broker_->metadata().type_count)
    error->all(FLERR, "DeePMD type_map metadata is internally inconsistent");
  type_index_map_.assign(atom->ntypes, -1);
  for (int type = 0; type < atom->ntypes; ++type) {
    const std::string requested = arg[type + 2];
    if (requested == "NULL")
      continue;
    const auto found =
        std::find(model_types.begin(), model_types.end(), requested);
    if (found == model_types.end())
      error->all(FLERR, "Element " + requested +
                            " is absent from the DeePMD model type_map");
    type_index_map_[type] =
        static_cast<int>(std::distance(model_types.begin(), found));
  }
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) {
      setflag[i][j] = 1;
      scale_[i][j] = 1.0;
    }
}

void PairDPRCDeepMDBatch::init_style() {
  if (type_index_map_.size() != static_cast<std::size_t>(atom->ntypes))
    error->all(FLERR,
               "dprc/deepmd/batch requires a complete pair_coeff type map");
  if (neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol)
    error->all(FLERR,
               "dprc/deepmd/batch does not support neigh_modify exclude");
  if (!atom->tag_enable || atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,
               "dprc/deepmd/batch requires atom IDs and an atom map; add "
               "'atom_modify map yes'");
  const int group_index = group->find(center_group_id_.c_str());
  if (group_index < 0)
    error->all(FLERR, "center_group " + center_group_id_ + " does not exist");
  if (group->dynamic[group_index])
    error->all(FLERR,
               "dprc/deepmd/batch currently requires a static center_group");
  center_group_bit_ = group->bitmask[group_index];
  if (include_molecule_ && !atom->molecule_flag)
    error->all(FLERR,
               "include_molecule yes requires an atom style with molecule IDs");
  if (update->whichflag == 1 &&
      utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "dprc/deepmd/batch does not support r-RESPA");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairDPRCDeepMDBatch::init_one(int i, int j) {
  scale_[j][i] = scale_[i][j];
  return std::max(model_cutoff_, environment_cutoff_);
}

std::vector<unsigned char> PairDPRCDeepMDBatch::select_model_atoms() const {
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  std::vector<unsigned char> is_center(static_cast<std::size_t>(nall), 0);
  std::vector<unsigned char> selected(static_cast<std::size_t>(nall), 0);
  int center_count = 0;
  for (int atom_index = 0; atom_index < nall; ++atom_index) {
    is_center[atom_index] =
        (atom->mask[atom_index] & center_group_bit_) != 0;
    if (atom_index < nlocal && is_center[atom_index]) {
      ++center_count;
      if (type_index_map_[atom->type[atom_index] - 1] < 0)
        throw std::invalid_argument(
            "center_group contains an atom mapped to NULL in pair_coeff");
    }
  }
  if (center_count == 0)
    throw std::invalid_argument("center_group is empty");

  const double cutoff_sq = environment_cutoff_ * environment_cutoff_;
  std::vector<tagint> selection_keys;
  const auto select_environment = [&](int center, int environment,
                                      bool minimum_image) {
    if (is_center[environment] ||
        type_index_map_[atom->type[environment] - 1] < 0)
      return;
    double dx = atom->x[environment][0] - atom->x[center][0];
    double dy = atom->x[environment][1] - atom->x[center][1];
    double dz = atom->x[environment][2] - atom->x[center][2];
    if (minimum_image)
      domain->minimum_image(FLERR, dx, dy, dz);
    if (dx * dx + dy * dy + dz * dz >= cutoff_sq)
      return;
    if (include_molecule_) {
      if (atom->molecule[environment] <= 0)
        throw std::invalid_argument(
            "include_molecule yes requires positive molecule IDs for selected "
            "environment atoms");
      selection_keys.push_back(atom->molecule[environment]);
    } else {
      selection_keys.push_back(atom->tag[environment]);
    }
  };

  for (int row = 0; row < list->inum; ++row) {
    const int center = list->ilist[row];
    if (!is_center[center])
      continue;
    const int neighbors = list->numneigh[center];
    int *neighbor_atoms = list->firstneigh[center];
    for (int slot = 0; slot < neighbors; ++slot)
      select_environment(center, neighbor_atoms[slot] & NEIGHMASK, false);
  }

  // special_bonds can remove a bonded environment atom from the pair list.
  // Recover those bounded topology entries so compact membership does not
  // depend on the classical force-field exclusion factors.
  if (atom->molecular != Atom::ATOMIC && atom->special && atom->nspecial) {
    std::unordered_map<tagint, int> by_tag;
    const auto find_by_tag = [&](tagint tag) {
      if (by_tag.empty()) {
        by_tag.reserve(static_cast<std::size_t>(nall));
        for (int index = 0; index < nall; ++index)
          by_tag.emplace(atom->tag[index], index);
      }
      const auto found = by_tag.find(tag);
      return found == by_tag.end() ? -1 : found->second;
    };
    for (int center = 0; center < nlocal; ++center) {
      if (!is_center[center])
        continue;
      for (int level = 1; level <= 3; ++level) {
        if (force->special_lj[level] != 0.0 ||
            force->special_coul[level] != 0.0)
          continue;
        const int begin = level == 1 ? 0 : atom->nspecial[center][level - 2];
        const int end = atom->nspecial[center][level - 1];
        for (int slot = begin; slot < end; ++slot) {
          const int environment = find_by_tag(atom->special[center][slot]);
          if (environment >= 0)
            select_environment(center, environment, true);
        }
      }
    }
  }

  std::sort(selection_keys.begin(), selection_keys.end());
  selection_keys.erase(
      std::unique(selection_keys.begin(), selection_keys.end()),
      selection_keys.end());
  int selected_local = 0;
  for (int index = 0; index < nall; ++index) {
    const tagint key =
        include_molecule_ ? atom->molecule[index] : atom->tag[index];
    const bool environment =
        std::binary_search(selection_keys.begin(), selection_keys.end(), key);
    const bool active = type_index_map_[atom->type[index] - 1] >= 0 &&
                        (is_center[index] || environment);
    selected[index] = active;
    if (index < nlocal && active)
      ++selected_local;
  }
  if (!include_molecule_) {
    for (int index = nlocal; index < nall; ++index) {
      const int owner = atom->map(atom->tag[index]);
      if (owner >= 0 && selected[index])
        selected[owner] = 1;
    }
    for (int index = nlocal; index < nall; ++index) {
      const int owner = atom->map(atom->tag[index]);
      if (owner >= 0 && selected[owner])
        selected[index] = 1;
    }
  }
  if (selected_local == 0)
    throw std::invalid_argument("compact DeePMD selection contains no atoms");
  return selected;
}

void PairDPRCDeepMDBatch::build_compact_graph(
    DPRC::DeepmdCanonicalGraph &graph,
    std::vector<int> &node_to_atom) const {
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  const std::vector<unsigned char> selected = select_model_atoms();
  std::vector<int> atom_to_node(static_cast<std::size_t>(nlocal), -1);
  node_to_atom.clear();
  graph = DPRC::DeepmdCanonicalGraph{};
  graph.timestep = update->ntimestep;
  for (int index = 0; index < nlocal; ++index) {
    if (!selected[index])
      continue;
    atom_to_node[index] = static_cast<int>(node_to_atom.size());
    node_to_atom.push_back(index);
    graph.atom_types.push_back(type_index_map_[atom->type[index] - 1]);
  }

  const double cutoff_sq = model_cutoff_ * model_cutoff_;
  std::vector<std::uint32_t> source_counts(node_to_atom.size(), 0);
  graph.destination_row_ptr.reserve(node_to_atom.size() + 1);
  graph.destination_row_ptr.push_back(0);
  for (std::size_t node = 0; node < node_to_atom.size(); ++node) {
    const int center = node_to_atom[node];
    const int neighbors = list->numneigh[center];
    int *neighbor_atoms = list->firstneigh[center];
    for (int slot = 0; slot < neighbors; ++slot) {
      const int candidate = neighbor_atoms[slot] & NEIGHMASK;
      const int owner =
          candidate < nlocal ? candidate : atom->map(atom->tag[candidate]);
      if (owner < 0 || owner >= nlocal || !selected[owner])
        continue;
      const int source_node = atom_to_node[owner];
      if (source_node < 0)
        continue;
      const double dx = atom->x[candidate][0] - atom->x[center][0];
      const double dy = atom->x[candidate][1] - atom->x[center][1];
      const double dz = atom->x[candidate][2] - atom->x[center][2];
      if (dx * dx + dy * dy + dz * dz >= cutoff_sq)
        continue;
      graph.sources.push_back(static_cast<std::uint32_t>(source_node));
      graph.edge_vectors.push_back(
          static_cast<float>(dx / distance_unit_factor_));
      graph.edge_vectors.push_back(
          static_cast<float>(dy / distance_unit_factor_));
      graph.edge_vectors.push_back(
          static_cast<float>(dz / distance_unit_factor_));
      if (source_counts[static_cast<std::size_t>(source_node)] ==
          std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("canonical source degree exceeds uint32");
      ++source_counts[static_cast<std::size_t>(source_node)];
    }
    graph.destination_row_ptr.push_back(
        static_cast<std::int64_t>(graph.sources.size()));
  }

  graph.source_row_ptr.resize(node_to_atom.size() + 1, 0);
  for (std::size_t node = 0; node < node_to_atom.size(); ++node)
    graph.source_row_ptr[node + 1] =
        graph.source_row_ptr[node] + source_counts[node];
  std::vector<std::uint32_t> cursor(node_to_atom.size(), 0);
  for (std::size_t node = 0; node < node_to_atom.size(); ++node)
    cursor[node] = static_cast<std::uint32_t>(graph.source_row_ptr[node]);
  graph.source_order.resize(graph.sources.size());
  for (std::size_t edge = 0; edge < graph.sources.size(); ++edge) {
    const std::uint32_t source = graph.sources[edge];
    graph.source_order[cursor[source]++] = static_cast<std::uint32_t>(edge);
  }
  (void)nall;
}

void PairDPRCDeepMDBatch::check_collective_build_status(
    bool failed, std::string diagnostic) const {
  const MPI_Comm roots = roots_->communicator();
  const int local_failed = failed ? 1 : 0;
  int any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, roots) !=
      MPI_SUCCESS)
    error->universe_all(FLERR, "DeePMD graph failure reduction failed");
  if (!any_failed)
    return;
  const int candidate = failed ? roots_->rank() : roots_->size();
  int diagnostic_rank = roots_->size();
  if (MPI_Allreduce(&candidate, &diagnostic_rank, 1, MPI_INT, MPI_MIN, roots) !=
      MPI_SUCCESS)
    error->universe_all(FLERR, "DeePMD diagnostic-rank reduction failed");
  int length = roots_->rank() == diagnostic_rank
                   ? static_cast<int>(diagnostic.size())
                   : 0;
  if (MPI_Bcast(&length, 1, MPI_INT, diagnostic_rank, roots) != MPI_SUCCESS)
    error->universe_all(FLERR, "DeePMD diagnostic length broadcast failed");
  if (roots_->rank() != diagnostic_rank)
    diagnostic.resize(static_cast<std::size_t>(length));
  if (MPI_Bcast(diagnostic.empty() ? nullptr : diagnostic.data(), length,
                MPI_CHAR, diagnostic_rank, roots) != MPI_SUCCESS)
    error->universe_all(FLERR, "DeePMD diagnostic broadcast failed");
  error->universe_all(FLERR, diagnostic);
}

void PairDPRCDeepMDBatch::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (vflag_atom)
    error->universe_all(
        FLERR,
        "dprc/deepmd/batch does not provide the six-component stress/atom; "
        "use compute centroid/stress/atom for the nine-component virial");

  DPRC::DeepmdCanonicalGraph graph;
  std::vector<int> node_to_atom;
  bool build_failed = false;
  std::string build_diagnostic;
  try {
    build_compact_graph(graph, node_to_atom);
  } catch (const std::exception &exception) {
    build_failed = true;
    build_diagnostic = exception.what();
  }
  check_collective_build_status(build_failed, std::move(build_diagnostic));

  try {
    broker_->compute(graph);
  } catch (const std::exception &exception) {
    error->universe_all(FLERR, exception.what());
  }
  const DPRC::DeepmdWindowResultView result =
      broker_->result_for_local_window();
  if (result.node_count != node_to_atom.size())
    error->universe_all(FLERR,
                        "DeePMD broker returned an invalid result extent");

  const double scale = scale_[1][1];
  for (std::size_t node = 0; node < node_to_atom.size(); ++node) {
    const int atom_index = node_to_atom[node];
    atom->f[atom_index][0] +=
        scale * force_unit_factor_ * result.force[3 * node + 0];
    atom->f[atom_index][1] +=
        scale * force_unit_factor_ * result.force[3 * node + 1];
    atom->f[atom_index][2] +=
        scale * force_unit_factor_ * result.force[3 * node + 2];
    if (eflag_atom)
      eatom[atom_index] +=
          scale * energy_unit_factor_ * result.atom_energy[node];
  }

  if (eflag_global) {
    double energy = 0.0;
    for (std::size_t node = 0; node < result.node_count; ++node)
      energy += result.atom_energy[node];
    eng_vdwl += scale * energy_unit_factor_ * energy;
  }
  if (vflag_global) {
    constexpr int component[6] = {0, 4, 8, 3, 6, 7};
    for (int output = 0; output < 6; ++output) {
      double value = 0.0;
      for (std::size_t node = 0; node < result.node_count; ++node)
        value += result.atom_virial[9 * node + component[output]];
      virial[output] += scale * energy_unit_factor_ * value;
    }
  }
  if (cvflag_atom) {
    constexpr int component[9] = {0, 4, 8, 3, 6, 7, 1, 2, 5};
    for (std::size_t node = 0; node < node_to_atom.size(); ++node) {
      const int atom_index = node_to_atom[node];
      for (int output = 0; output < 9; ++output)
        cvatom[atom_index][output] +=
            scale * energy_unit_factor_ *
            result.atom_virial[9 * node + component[output]];
    }
  }
}

void *PairDPRCDeepMDBatch::extract(const char *name, int &dimension) {
  if (std::strcmp(name, "scale") == 0) {
    dimension = 2;
    return scale_;
  }
  return nullptr;
}
