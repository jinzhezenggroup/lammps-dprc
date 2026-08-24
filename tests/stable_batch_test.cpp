#include "stable_batch.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " #condition      \
                << '\n';                                                       \
      return __LINE__;                                                         \
    }                                                                          \
  } while (false)

namespace {

template <class Exception, class Function> bool throws(Function &&function) {
  try {
    function();
  } catch (const Exception &) {
    return true;
  } catch (...) {
  }
  return false;
}

DPRC::WindowTopology topology(std::int32_t window, std::int32_t element,
                              std::size_t point_count, bool response = true) {
  DPRC::WindowTopology value;
  value.window_index = window;
  value.atomic_numbers = {element, 1};
  value.point_charge_gammas.assign(point_count,
                                   0.8 + 0.01 * static_cast<double>(window));
  value.charge_response_enabled = response;
  return value;
}

DPRC::WindowFrame frame(std::int32_t window, std::int64_t timestep,
                        std::size_t point_count, double marker,
                        bool response = true) {
  DPRC::WindowFrame value;
  value.window_index = window;
  value.timestep = timestep;
  value.positions = {marker, 0.0, 0.0, marker + 1.4, 0.0, 0.0};
  value.point_charge_positions.resize(3u * point_count, marker + 2.0);
  value.point_charge_values.resize(point_count, 0.1 * marker);
  if (response) {
    value.atomic_potential_shifts = {0.001 * marker, -0.001 * marker};
    value.charge_response_matrix = {0.02, 0.001, 0.001, 0.018};
  }
  return value;
}

DPRC::WindowFrameView view(const DPRC::WindowFrame &frame) {
  return {frame.window_index,
          frame.timestep,
          {frame.positions.data(), frame.positions.size()},
          {frame.point_charge_positions.data(),
           frame.point_charge_positions.size()},
          {frame.point_charge_values.data(), frame.point_charge_values.size()},
          {frame.atomic_potential_shifts.data(),
           frame.atomic_potential_shifts.size()},
          {frame.charge_response_matrix.data(),
           frame.charge_response_matrix.size()}};
}

int test_stable_slot_order_and_ragged_offsets() {
  DPRC::StableBatch batch({topology(1, 8, 2), topology(0, 6, 1)});
  CHECK(batch.size() == 2u);
  CHECK(batch.topology(0).window_index == 0);
  CHECK(batch.topology(1).window_index == 1);
  CHECK(batch.slot_for_window(0) == 0u);
  CHECK(batch.slot_for_window(1) == 1u);
  CHECK(batch.atom_offsets() == std::vector<std::int64_t>({0, 2, 4}));
  CHECK(batch.point_charge_offsets() == std::vector<std::int64_t>({0, 1, 3}));
  CHECK(batch.charge_response_offsets() ==
        std::vector<std::int64_t>({0, 4, 8}));
  CHECK(batch.atomic_numbers() == std::vector<std::int32_t>({6, 1, 8, 1}));
  CHECK(batch.point_charge_gammas().size() == 3u);
  CHECK(throws<std::out_of_range>([&] { batch.slot_for_window(2); }));
  return 0;
}

int test_staging_is_transactional_and_timestep_aligned() {
  DPRC::StableBatch batch({topology(0, 6, 1), topology(1, 8, 2)});

  DPRC::WindowFrame malformed = frame(0, 100, 1, 3.0);
  malformed.charge_response_matrix.pop_back();
  CHECK(throws<std::invalid_argument>([&] { batch.stage(malformed); }));
  CHECK(batch.timestep() == -1);
  CHECK(batch.positions()[0] == 0.0);

  DPRC::WindowFrame valid_borrowed = frame(0, 100, 1, 2.0);
  DPRC::WindowFrameView null_positions = view(valid_borrowed);
  null_positions.positions.data = nullptr;
  CHECK(throws<std::invalid_argument>([&] { batch.stage(null_positions); }));
  CHECK(batch.timestep() == -1);

  batch.stage(frame(1, 100, 2, 7.0));
  CHECK(batch.timestep() == 100);
  CHECK(!batch.ready());
  CHECK(throws<std::invalid_argument>(
      [&] { batch.stage(frame(0, 101, 1, 2.0)); }));
  CHECK(throws<std::logic_error>([&] { batch.stage(frame(1, 100, 2, 7.0)); }));

  batch.stage(view(valid_borrowed));
  CHECK(batch.ready());
  // Window 0 is slot 0 even though it staged second; window 1 occupies slot 1.
  CHECK(batch.positions()[0] == 2.0);
  CHECK(batch.positions()[6] == 7.0);
  CHECK(batch.point_charge_values()[0] == 0.2);
  CHECK(std::abs(batch.point_charge_values()[1] - 0.7) < 1.0e-15);
  return 0;
}

int test_strict_whole_batch_warm_state() {
  DPRC::StableBatch batch({topology(0, 1, 0, false), topology(1, 2, 0, false)});
  batch.stage(frame(0, 0, 0, 1.0, false));
  batch.stage(frame(1, 0, 0, 2.0, false));
  CHECK(batch.prepare_compute() == DPRC::SccStartPolicy::Fresh);
  CHECK(batch.compute_in_flight());
  CHECK(throws<std::logic_error>(
      [&] { batch.stage(frame(0, 1, 0, 1.1, false)); }));

  const std::int32_t success[] = {0, 0};
  batch.complete_compute(true, success, 2u, 0);
  CHECK(batch.warm_ready());
  CHECK(batch.timestep() == -1);

  batch.stage(frame(1, 1, 0, 2.1, false));
  batch.stage(frame(0, 1, 0, 1.1, false));
  CHECK(batch.prepare_compute() == DPRC::SccStartPolicy::Warm);
  const std::int32_t peer_failure[] = {0, 9};
  batch.complete_compute(true, peer_failure, 2u, 0);
  CHECK(!batch.warm_ready());

  batch.stage(frame(0, 2, 0, 1.2, false));
  batch.stage(frame(1, 2, 0, 2.2, false));
  CHECK(batch.prepare_compute() == DPRC::SccStartPolicy::Fresh);
  batch.complete_compute(false, nullptr, 0u, 0);
  CHECK(!batch.warm_ready());
  return 0;
}

int test_invalid_topologies_rejected() {
  CHECK(throws<std::invalid_argument>([] { DPRC::StableBatch batch({}); }));
  CHECK(throws<std::invalid_argument>(
      [] { DPRC::StableBatch batch({topology(1, 1, 0), topology(1, 1, 0)}); }));
  CHECK(throws<std::invalid_argument>([] {
    DPRC::StableBatch batch(
        {topology(0, 1, 0, true), topology(1, 1, 0, false)});
  }));
  CHECK(throws<std::invalid_argument>([] {
    DPRC::WindowTopology invalid = topology(0, 1, 1);
    invalid.point_charge_gammas[0] = 0.0;
    DPRC::StableBatch batch({invalid});
  }));
  CHECK(throws<std::invalid_argument>(
      [] { DPRC::StableBatch batch({topology(2, 1, 0)}); }));
  return 0;
}

} // namespace

int main() {
  if (const int result = test_stable_slot_order_and_ragged_offsets())
    return result;
  if (const int result = test_staging_is_transactional_and_timestep_aligned())
    return result;
  if (const int result = test_strict_whole_batch_warm_state())
    return result;
  return test_invalid_topologies_rejected();
}
