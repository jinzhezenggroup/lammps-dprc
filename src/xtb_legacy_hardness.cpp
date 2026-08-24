#include "xtb_legacy_hardness.h"

#include "dprc_xtb_hardness.h"

#include <cmath>
#include <stdexcept>

namespace DPRC {

void fill_legacy_point_charge_gammas(
    int method, double mm_hardness, const int *atomic_numbers,
    std::size_t point_count, std::vector<double> &gammas) {
  if ((method != 1 && method != 2) || !std::isfinite(mm_hardness) ||
      (point_count != 0u && atomic_numbers == nullptr)) {
    throw std::invalid_argument("invalid legacy MM-hardness request");
  }

  const auto &hardness = method == 1 ? Generated::kGfn1ChemicalHardness
                                     : Generated::kGfn2ChemicalHardness;
  gammas.resize(point_count);
  for (std::size_t point = 0; point < point_count; ++point) {
    if (mm_hardness > 1.0e-6) {
      gammas[point] = mm_hardness;
    } else if (mm_hardness > -1.0e-6) {
      // Zero/near-zero requests select hydrogen hardness for every real or
      // virtual MM site, exactly matching the legacy libxTB adapter.
      gammas[point] = hardness.front();
    } else {
      const int atomic_number = atomic_numbers[point] <= 0
          ? 1
          : atomic_numbers[point];
      if (static_cast<std::size_t>(atomic_number) > hardness.size()) {
        throw std::out_of_range(
            "MM element is outside the selected xTB parameter range");
      }
      gammas[point] =
          std::abs(mm_hardness) * hardness[atomic_number - 1];
    }
  }
}

std::vector<double> legacy_point_charge_gammas(
    int method, double mm_hardness, const int *atomic_numbers,
    std::size_t point_count) {
  std::vector<double> gammas;
  fill_legacy_point_charge_gammas(method, mm_hardness, atomic_numbers,
                                  point_count, gammas);
  return gammas;
}

} // namespace DPRC
