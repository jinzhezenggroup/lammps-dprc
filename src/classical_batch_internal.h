#ifndef LAMMPS_DPRC_CLASSICAL_BATCH_INTERNAL_H
#define LAMMPS_DPRC_CLASSICAL_BATCH_INTERNAL_H

#include "classical_batch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace DPRC {

// Private host-side image shared by the CPU oracle and CUDA plan setup.  It is
// not installed and is deliberately independent of LAMMPS C++ implementation
// types.  All reciprocal arrays use the full-complex x-fastest mesh layout.
struct PreparedClassicalData {
  std::array<double, 9> hinv{};
  double volume = 0.0;
  double delvolinv = 0.0;
  std::array<std::int32_t, 3> mesh{};
  std::size_t mesh_count = 0;
  std::int32_t spline_lower = 0;
  std::int32_t spline_upper = 0;
  double spline_shift = 0.0;
  double spline_shift_one = 0.0;
  std::vector<double> spline_coefficients;
  std::vector<double> green;
  std::vector<double> kvector;
  std::vector<double> virial_factor;
  std::vector<std::int32_t> oxygen_site;

  // Symmetric per-atom CSR for special LJ/Coulomb scales.
  std::vector<std::int32_t> special_offsets;
  std::vector<std::int32_t> special_partners;
  std::vector<double> special_lj;
  std::vector<double> special_coulomb;

  // Fractional cell-list topology and a unique periodic neighbor-bin CSR.
  std::array<std::int32_t, 3> bin_count{};
  std::vector<std::int32_t> neighbor_bin_offsets;
  std::vector<std::int32_t> neighbor_bins;
  double neighbor_cutoff = 0.0;
};

[[nodiscard]] PreparedClassicalData
prepare_classical_data(const ClassicalTopology &topology);

}  // namespace DPRC

#endif
