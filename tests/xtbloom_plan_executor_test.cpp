#include "xtbloom_plan_executor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>
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

double max_serial_batch_energy_error = 0.0;
double max_serial_batch_force_error = 0.0;
double max_serial_batch_charge_error = 0.0;
double max_serial_batch_point_force_error = 0.0;
double oracle_energy_error = 0.0;
double oracle_force_error = 0.0;
double oracle_point_force_error = 0.0;

bool near(double lhs, double rhs, double tolerance) {
  const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
  return std::abs(lhs - rhs) <= tolerance * scale;
}

bool absolute_near(double lhs, double rhs, double tolerance) {
  return std::abs(lhs - rhs) <= tolerance;
}

DPRC::WindowTopology make_topology(std::int32_t window) {
  DPRC::WindowTopology topology;
  topology.window_index = window;
  topology.atomic_numbers = {1, 1};
  topology.point_charge_gammas = {0.82};
  topology.charge_response_enabled = true;
  return topology;
}

DPRC::WindowFrame make_frame(std::int32_t window, std::int64_t timestep,
                             double displacement) {
  DPRC::WindowFrame frame;
  frame.window_index = window;
  frame.timestep = timestep;
  frame.positions = {-0.71 - displacement, 0.0, 0.0,
                     0.71 + displacement,  0.0, 0.0};
  frame.point_charge_positions = {0.2, 1.8 + displacement, -0.7};
  frame.point_charge_values = {0.25 - 0.1 * displacement};
  frame.atomic_potential_shifts = {0.003 + 0.01 * displacement,
                                   -0.002 - 0.01 * displacement};
  frame.charge_response_matrix = {0.02 + 0.01 * displacement, 0.001, 0.001,
                                  0.018};
  return frame;
}

DPRC::WindowTopology make_gas_topology(std::int32_t window) {
  DPRC::WindowTopology topology;
  topology.window_index = window;
  topology.atomic_numbers = {1, 1};
  return topology;
}

DPRC::WindowFrame make_gas_frame(std::int32_t window, std::int64_t timestep,
                                 double bond_length = 1.4) {
  DPRC::WindowFrame frame;
  frame.window_index = window;
  frame.timestep = timestep;
  frame.positions = {-0.5 * bond_length, 0.0, 0.0, 0.5 * bond_length, 0.0, 0.0};
  return frame;
}

int compare_window(const DPRC::WindowResultView &batch,
                   const DPRC::WindowResultView &serial) {
  CHECK(batch.status == XTBLOOM_STATUS_SUCCESS);
  CHECK(serial.status == XTBLOOM_STATUS_SUCCESS);
  CHECK(batch.scc_converged);
  CHECK(serial.scc_converged);
  CHECK(batch.atom_count == serial.atom_count);
  CHECK(batch.point_charge_count == serial.point_charge_count);
  max_serial_batch_energy_error = std::max(
      max_serial_batch_energy_error, std::abs(batch.energy - serial.energy));
  CHECK(near(batch.energy, serial.energy, 2.0e-10));
  for (std::size_t coordinate = 0; coordinate < 3u * batch.atom_count;
       ++coordinate) {
    max_serial_batch_force_error = std::max(
        max_serial_batch_force_error,
        std::abs(batch.forces[coordinate] - serial.forces[coordinate]));
    CHECK(near(batch.forces[coordinate], serial.forces[coordinate], 2.0e-8));
  }
  for (std::size_t atom = 0; atom < batch.atom_count; ++atom) {
    max_serial_batch_charge_error = std::max(
        max_serial_batch_charge_error,
        std::abs(batch.atomic_charges[atom] - serial.atomic_charges[atom]));
    CHECK(
        near(batch.atomic_charges[atom], serial.atomic_charges[atom], 2.0e-8));
  }
  for (std::size_t coordinate = 0; coordinate < 3u * batch.point_charge_count;
       ++coordinate) {
    max_serial_batch_point_force_error =
        std::max(max_serial_batch_point_force_error,
                 std::abs(batch.point_charge_forces[coordinate] -
                          serial.point_charge_forces[coordinate]));
    CHECK(near(batch.point_charge_forces[coordinate],
               serial.point_charge_forces[coordinate], 2.0e-8));
  }
  return 0;
}

int run_serial_reference(const DPRC::WindowFrame &frame,
                         const DPRC::WindowResultView &batch_result) {
  DPRC::XtbloomExecutorOptions options;
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 1;
  // A standalone one-system plan has one plan-local dense slot (zero). The
  // global window identity remains only on the all-window batch being tested.
  DPRC::XtbloomPlanExecutor serial({make_topology(0)}, options);
  DPRC::WindowFrame serial_frame = frame;
  serial_frame.window_index = 0;
  serial.stage(serial_frame);
  const DPRC::XtbloomComputeOutcome outcome = serial.compute();
  CHECK(outcome.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(outcome.start_policy == DPRC::SccStartPolicy::Fresh);
  return compare_window(batch_result, serial.result_for_window(0));
}

int test_pinned_xtb_water_point_charge_oracle() {
  // xTB 6.7.1 revision edcfbbe39d411edc225e27315fbda3a204ddb023,
  // --acc 0.0001. The values and 5e-7 absolute public-conformance tolerance
  // are pinned in xTBloom's docs/theory/qmmm.md and conformance manifest.
  DPRC::WindowTopology topology;
  topology.window_index = 0;
  topology.atomic_numbers = {8, 1, 1};
  topology.point_charge_gammas = {0.405771};

  DPRC::WindowFrame frame;
  frame.window_index = 0;
  frame.timestep = 0;
  frame.positions = {0.0,        0.0,         0.0, 1.43233673, 0.0,
                     1.10715266, -1.43233673, 0.0, 1.10715266};
  frame.point_charge_positions = {4.0, 0.0, 0.0};
  frame.point_charge_values = {0.5};

  DPRC::XtbloomExecutorOptions options;
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 1;
  DPRC::XtbloomPlanExecutor executor({topology}, options);
  executor.stage(frame);
  const DPRC::XtbloomComputeOutcome outcome = executor.compute();
  CHECK(outcome.call_status == XTBLOOM_STATUS_SUCCESS);
  const DPRC::WindowResultView result = executor.result_for_window(0);
  CHECK(result.status == XTBLOOM_STATUS_SUCCESS);
  CHECK(result.scc_converged);
  constexpr double tolerance = 5.0e-7;
  oracle_energy_error = std::abs(result.energy - (-5.0730682804123326));
  CHECK(absolute_near(result.energy, -5.0730682804123326, tolerance));
  const double expected_forces[] = {
      0.0112621578336112,  0.0, 0.0033988314289940,
      -0.0013479723470460, 0.0, 0.0012180034891371,
      -0.0074467401247049, 0.0, -0.0012859430397343};
  for (std::size_t coordinate = 0; coordinate < 9u; ++coordinate) {
    oracle_force_error =
        std::max(oracle_force_error, std::abs(result.forces[coordinate] -
                                              expected_forces[coordinate]));
    CHECK(absolute_near(result.forces[coordinate], expected_forces[coordinate],
                        tolerance));
  }
  const double expected_point_force[] = {-0.0024674453618603, 0.0,
                                         -0.0033308918783968};
  for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
    oracle_point_force_error =
        std::max(oracle_point_force_error,
                 std::abs(result.point_charge_forces[coordinate] -
                          expected_point_force[coordinate]));
    CHECK(absolute_near(result.point_charge_forces[coordinate],
                        expected_point_force[coordinate], tolerance));
  }
  return 0;
}

int test_native_failure_is_atomic_and_fresh_recovery() {
  DPRC::XtbloomExecutorOptions options;
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 2;
  DPRC::XtbloomPlanExecutor executor(
      {make_gas_topology(1), make_gas_topology(0)}, options);

  executor.stage(make_gas_frame(0, 0));
  executor.stage(make_gas_frame(1, 0));
  const DPRC::XtbloomComputeOutcome initial = executor.compute();
  CHECK(initial.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(initial.start_policy == DPRC::SccStartPolicy::Fresh);
  CHECK(executor.batch().warm_ready());

  executor.stage(make_gas_frame(0, 1));
  executor.stage(make_gas_frame(1, 1, 1.1e-6));
  const DPRC::XtbloomComputeOutcome peer_failure = executor.compute();
  CHECK(peer_failure.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(!peer_failure.all_systems_succeeded);
  CHECK(peer_failure.start_policy == DPRC::SccStartPolicy::Warm);
  CHECK(!executor.has_result());
  CHECK(!executor.last_error().empty());
  CHECK(executor.last_error().find("window 1") != std::string::npos);
  for (const int window : {0, 1}) {
    bool rejected_result = false;
    try {
      static_cast<void>(executor.result_for_window(window));
    } catch (const std::logic_error &) {
      rejected_result = true;
    }
    CHECK(rejected_result);
  }
  CHECK(!executor.batch().warm_ready());

  executor.stage(make_gas_frame(0, 2));
  executor.stage(make_gas_frame(1, 2));
  const DPRC::XtbloomComputeOutcome recovered = executor.compute();
  CHECK(recovered.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(recovered.start_policy == DPRC::SccStartPolicy::Fresh);
  CHECK(executor.batch().warm_ready());

  DPRC::WindowFrame invalid = make_gas_frame(0, 3);
  invalid.positions[0] = std::numeric_limits<double>::quiet_NaN();
  executor.stage(invalid);
  executor.stage(make_gas_frame(1, 3));
  const DPRC::XtbloomComputeOutcome rejected = executor.compute();
  CHECK(rejected.call_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected.start_policy == DPRC::SccStartPolicy::Warm);
  CHECK(!executor.has_result());
  CHECK(!executor.last_error().empty());
  CHECK(!executor.batch().warm_ready());
  bool rejected_result_publication = false;
  try {
    executor.result_for_window(0);
  } catch (const std::logic_error &) {
    rejected_result_publication = true;
  }
  CHECK(rejected_result_publication);

  executor.stage(make_gas_frame(0, 4));
  executor.stage(make_gas_frame(1, 4));
  const DPRC::XtbloomComputeOutcome after_rejection = executor.compute();
  CHECK(after_rejection.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(after_rejection.start_policy == DPRC::SccStartPolicy::Fresh);
  return 0;
}

} // namespace

int main() {
  if (const int result = test_pinned_xtb_water_point_charge_oracle())
    return result;
  if (const int result = test_native_failure_is_atomic_and_fresh_recovery())
    return result;

  DPRC::XtbloomExecutorOptions incomplete_options;
  incomplete_options.compute_flags = XTBLOOM_COMPUTE_ENERGY;
  bool rejected_incomplete_policy = false;
  try {
    DPRC::XtbloomPlanExecutor invalid({make_topology(0)}, incomplete_options);
  } catch (const std::invalid_argument &) {
    rejected_incomplete_policy = true;
  }
  CHECK(rejected_incomplete_policy);

  DPRC::XtbloomExecutorOptions invalid_tolerance;
  invalid_tolerance.charge_tolerance = 0.0;
  bool rejected_invalid_tolerance = false;
  try {
    DPRC::XtbloomPlanExecutor invalid({make_topology(0)}, invalid_tolerance);
  } catch (const std::runtime_error &) {
    rejected_invalid_tolerance = true;
  }
  CHECK(rejected_invalid_tolerance);

  DPRC::XtbloomExecutorOptions options;
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 2;
  // Register in reverse order to prove window identity, not arrival order,
  // determines the native ragged slot and its WARM checkpoint.
  DPRC::XtbloomPlanExecutor batch({make_topology(1), make_topology(0)},
                                  options);
  CHECK(batch.backend() == XTBLOOM_BACKEND_CPU);
  CHECK(batch.device_id() == -1);
  const DPRC::XtbloomWorkspaceInfo workspace = batch.workspace();
  CHECK(workspace.host_bytes > 0u);
  CHECK(workspace.host_alignment >= 8u);
  CHECK(workspace.device_bytes == 0u);
  CHECK(workspace.device_alignment == 1u);

  const DPRC::WindowFrame first0 = make_frame(0, 10, 0.000);
  const DPRC::WindowFrame first1 = make_frame(1, 10, 0.006);
  batch.stage(first1);
  batch.stage(first0);
  const DPRC::XtbloomComputeOutcome first = batch.compute();
  CHECK(first.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(first.timestep == 10);
  CHECK(first.start_policy == DPRC::SccStartPolicy::Fresh);
  CHECK((first.result_flags &
         XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES) != 0u);
  CHECK(run_serial_reference(first0, batch.result_for_window(0)) == 0);
  CHECK(run_serial_reference(first1, batch.result_for_window(1)) == 0);

  const DPRC::WindowFrame second0 = make_frame(0, 11, 0.002);
  const DPRC::WindowFrame second1 = make_frame(1, 11, 0.008);
  batch.stage(second0);
  bool rejected_stale_result = false;
  try {
    batch.result_for_window(0);
  } catch (const std::logic_error &) {
    rejected_stale_result = true;
  }
  CHECK(rejected_stale_result);
  batch.stage(second1);
  const DPRC::XtbloomComputeOutcome second = batch.compute();
  CHECK(second.call_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(second.timestep == 11);
  CHECK(second.start_policy == DPRC::SccStartPolicy::Warm);
  CHECK(run_serial_reference(second0, batch.result_for_window(0)) == 0);
  CHECK(run_serial_reference(second1, batch.result_for_window(1)) == 0);
  std::cout << std::scientific << std::setprecision(6)
            << "oracle_max_errors: energy_Eh=" << oracle_energy_error
            << " force_Eh_per_bohr=" << oracle_force_error
            << " point_force_Eh_per_bohr=" << oracle_point_force_error << '\n'
            << "serial_batch_max_errors: energy_Eh="
            << max_serial_batch_energy_error
            << " force_Eh_per_bohr=" << max_serial_batch_force_error
            << " charge_e=" << max_serial_batch_charge_error
            << " point_force_Eh_per_bohr=" << max_serial_batch_point_force_error
            << '\n';
  return 0;
}
