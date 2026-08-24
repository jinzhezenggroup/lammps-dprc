#ifndef LAMMPS_DPRC_COMMAND_DPRC_INFO_H
#define LAMMPS_DPRC_COMMAND_DPRC_INFO_H

#include "command.h"

namespace LAMMPS_NS {

// Diagnostic command used to prove that the plugin matches the selected
// LAMMPS process topology before force-producing styles are introduced.
class CommandDPRCInfo final : public Command {
public:
  explicit CommandDPRCInfo(class LAMMPS *lmp) : Command(lmp) {}
  void command(int argc, char **argv) override;
};

} // namespace LAMMPS_NS

#endif
