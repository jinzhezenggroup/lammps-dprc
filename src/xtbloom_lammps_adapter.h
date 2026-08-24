#ifndef LAMMPS_DPRC_XTBLOOM_LAMMPS_ADAPTER_H
#define LAMMPS_DPRC_XTBLOOM_LAMMPS_ADAPTER_H

#include <mpi.h>

namespace LAMMPS_NS {
class LAMMPS;
}

namespace DPRC {

// Bind the rank-zero adapter calls made by the pinned reference fix to the
// permanent communicator of LAMMPS partition roots. The communicator remains
// borrowed from PartitionRoots for the lifetime of the fix.
int bind_lammps_xtbloom_adapter(LAMMPS_NS::LAMMPS *lmp, MPI_Comm roots,
                               int stable_slot) noexcept;

} // namespace DPRC

extern "C" {

int dprc_lammps_xtb_create(int nqm, const int *atomic_numbers,
                           const double *qm_xyz_bohr, int method, int charge,
                           int uhf, double accuracy, int maxiter,
                           double electronic_temperature_kelvin);

int dprc_lammps_xtb_calculate(
    int nqm, const double *qm_xyz_bohr, int npoint,
    const double *point_xyz_bohr, const double *point_charge,
    const int *point_atomic_numbers, double mm_hardness,
    const double *mm_shift_hartree,
    const double *image_response_hartree, double *energy_hartree,
    double *qm_gradient_hartree_bohr, double *mulliken_charge,
    double *point_gradient_hartree_bohr);

// Return the diagnostic from the most recent failed adapter operation in this
// process. The borrowed pointer remains valid until the next adapter call.
const char *dprc_lammps_xtb_last_error();

void dprc_lammps_xtb_destroy();
}

#endif
