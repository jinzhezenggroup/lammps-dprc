#ifndef LAMMPS_DPRC_COMMAND_DPRC_TEST_RECOVERY_H
#define LAMMPS_DPRC_COMMAND_DPRC_TEST_RECOVERY_H

#include "command.h"

namespace LAMMPS_NS {

// Test-only command that proves a caught post-prepare exception does not leave
// a stale fused PPPM field pending in the same LAMMPS instance.
class CommandDPRCTestRecovery final : public Command {
public:
  explicit CommandDPRCTestRecovery(class LAMMPS *lmp) : Command(lmp) {}
  void command(int argc, char **argv) override;
};

} // namespace LAMMPS_NS

#endif
