#include "xtb_legacy_hardness.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition     \
                << '\n';                                                       \
      return __LINE__;                                                         \
    }                                                                          \
  } while (false)

bool near(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= 1.0e-15 *
      std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

int run_test() {
  const int elements[] = {0, 1, 8, 86};

  const std::vector<double> explicit_gammas =
      DPRC::legacy_point_charge_gammas(2, 0.75, elements, 4);
  CHECK(explicit_gammas == std::vector<double>({0.75, 0.75, 0.75, 0.75}));

  const std::vector<double> gfn1_default =
      DPRC::legacy_point_charge_gammas(1, 0.0, elements, 4);
  const std::vector<double> gfn2_default =
      DPRC::legacy_point_charge_gammas(2, 0.0, elements, 4);
  for (double value : gfn1_default) CHECK(near(value, 0.470099));
  for (double value : gfn2_default) CHECK(near(value, 0.405771));

  const std::vector<double> scaled =
      DPRC::legacy_point_charge_gammas(2, -2.0, elements, 4);
  CHECK(near(scaled[0], 2.0 * 0.405771));
  CHECK(near(scaled[1], 2.0 * 0.405771));
  CHECK(near(scaled[2], 2.0 * 0.451896));
  CHECK(near(scaled[3], 2.0 * 0.3034));

  std::vector<double> reused;
  reused.reserve(4);
  const double *const allocation = reused.data();
  DPRC::fill_legacy_point_charge_gammas(2, 0.0, elements, 4, reused);
  CHECK(reused.data() == allocation);
  DPRC::fill_legacy_point_charge_gammas(2, -2.0, elements, 4, reused);
  CHECK(reused.data() == allocation);
  CHECK(reused == scaled);

  bool rejected_range = false;
  const int unsupported[] = {87};
  try {
    static_cast<void>(
        DPRC::legacy_point_charge_gammas(2, -1.0, unsupported, 1));
  } catch (const std::out_of_range &) {
    rejected_range = true;
  }
  CHECK(rejected_range);
  return 0;
}

} // namespace

int main() { return run_test(); }
