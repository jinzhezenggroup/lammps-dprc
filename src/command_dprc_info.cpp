#include "command_dprc_info.h"

#include "partition_roots.h"

#include "comm.h"
#include "error.h"
#include "universe.h"
#include "utils.h"

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace LAMMPS_NS;

void CommandDPRCInfo::command(int argc, char **argv) {
  if (argc > 1)
    error->all(FLERR, "Illegal dprc/info command");

  std::unique_ptr<DPRC::PartitionRoots> roots;
  try {
    roots = std::make_unique<DPRC::PartitionRoots>(
        universe->uworld, comm->me, universe->iworld, universe->nworlds);
  } catch (const std::exception &exception) {
    error->universe_all(FLERR, exception.what());
  }

  const std::string message = fmt::format(
      "LAMMPS-DPRC {}: universe_worlds={} world_index={} world_ranks={} "
      "universe_rank={} broker_root={} broker_rank={} broker_size={} "
      "stable_slot={}\n",
      DPRC_PROJECT_VERSION, universe->nworlds, universe->iworld, comm->nprocs,
      universe->me, roots->is_root() ? 1 : 0, roots->rank(), roots->size(),
      roots->stable_slot());

  if (comm->me == 0) {
    utils::logmesg(lmp, message);
    // An optional marker makes the plugin load test independent of LAMMPS
    // screen/log routing, which differs between single and multi-partition
    // runs.
    if (argc == 1) {
      std::string marker_path = argv[0];
      if (universe->nworlds > 1)
        marker_path += fmt::format(".{}", universe->iworld);
      std::ofstream output(marker_path, std::ios::out | std::ios::trunc);
      if (!output)
        error->one(FLERR, "Could not open dprc/info marker file");
      output << message;
    }
  }
}
