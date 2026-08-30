#include "command_dprc_info.h"

#include "partition_roots.h"

#include "comm.h"
#include "error.h"
#include "universe.h"
#include "utils.h"

#include <fstream>
#include <memory>
#include <sstream>
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

  // Keep the plugin boundary independent of LAMMPS's bundled fmt ABI.  Some
  // valid static LAMMPS hosts do not export fmt's inline-namespace symbols,
  // so formatting through the standard library is required for dlopen to
  // succeed across supported host builds.
  std::ostringstream message_stream;
  message_stream << "LAMMPS-DPRC " << DPRC_PROJECT_VERSION
                 << ": universe_worlds=" << universe->nworlds
                 << " world_index=" << universe->iworld
                 << " world_ranks=" << comm->nprocs
                 << " universe_rank=" << universe->me
                 << " broker_root=" << (roots->is_root() ? 1 : 0)
                 << " broker_rank=" << roots->rank()
                 << " broker_size=" << roots->size()
                 << " stable_slot=" << roots->stable_slot() << '\n';
  const std::string message = message_stream.str();

  if (comm->me == 0) {
    utils::logmesg(lmp, message);
    // An optional marker makes the plugin load test independent of LAMMPS
    // screen/log routing, which differs between single and multi-partition
    // runs.
    if (argc == 1) {
      std::string marker_path = argv[0];
      if (universe->nworlds > 1)
        marker_path += "." + std::to_string(universe->iworld);
      std::ofstream output(marker_path, std::ios::out | std::ios::trunc);
      if (!output)
        error->one(FLERR, "Could not open dprc/info marker file");
      output << message;
    }
  }
}
