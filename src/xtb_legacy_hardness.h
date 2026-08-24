#ifndef LAMMPS_DPRC_XTB_LEGACY_HARDNESS_H
#define LAMMPS_DPRC_XTB_LEGACY_HARDNESS_H

#include <cstddef>
#include <vector>

namespace DPRC {

// Translate LAMMPS QMMM-XTB's historical mmhardness convention into the
// positive per-point gamma values required by xTBloom's public C ABI. The
// output overload reuses existing capacity on the steady-state adapter path.
void fill_legacy_point_charge_gammas(
    int method, double mm_hardness, const int *atomic_numbers,
    std::size_t point_count, std::vector<double> &gammas);

std::vector<double> legacy_point_charge_gammas(
    int method, double mm_hardness, const int *atomic_numbers,
    std::size_t point_count);

} // namespace DPRC

#endif
