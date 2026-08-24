#ifndef LAMMPS_DPRC_KSPACE_DPRC_H
#define LAMMPS_DPRC_KSPACE_DPRC_H

// The generated headers retain upstream filenames and source structure, but
// every implementation type is compiled under a project-specific name. This
// prevents the RTLD_GLOBAL plugin from defining the host's PPPMXTB or
// PPPMTIP4PXTB C++ symbols while preserving the pinned implementation.
#define PPPMXTB PPPMDPRC
#define PPPMTIP4PXTB PPPMTIP4PDPRC
#define QMMMXTBPPPMHelper DPRCQMMMReferencePPPMHelper
#include "pppm_xtb.h"
#include "pppm_tip4p_xtb.h"
#undef QMMMXTBPPPMHelper
#undef PPPMTIP4PXTB
#undef PPPMXTB

#endif
