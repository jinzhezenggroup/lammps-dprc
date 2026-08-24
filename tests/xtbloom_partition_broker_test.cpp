#include "xtbloom_partition_broker.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool collectively_true(bool condition, const char *expression, int line,
                       MPI_Comm communicator) {
  const int local = condition ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, communicator);
  if (global == 0 && local == 0) {
    int rank = -1;
    MPI_Comm_rank(communicator, &rank);
    std::cerr << "rank " << rank << " CHECK failed at line " << line << ": "
              << expression << '\n';
  }
  return global != 0;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!collectively_true((condition), #condition, __LINE__, MPI_COMM_WORLD)) \
      return __LINE__;                                                         \
  } while (false)

bool near(double lhs, double rhs, double tolerance) {
  const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
  return std::abs(lhs - rhs) <= tolerance * scale;
}

DPRC::WindowTopology make_topology(int rank) {
  DPRC::WindowTopology topology;
  topology.window_index = rank;
  topology.atomic_numbers = {1, 1};
  topology.point_charge_gammas =
      rank == 0 ? std::vector<double>{0.82} : std::vector<double>{0.83, 0.91};
  topology.charge_response_enabled = true;
  return topology;
}

DPRC::WindowFrame make_frame(int rank, std::int64_t timestep,
                             double displacement) {
  DPRC::WindowFrame frame;
  frame.window_index = rank;
  frame.timestep = timestep;
  frame.positions = {-0.71 - displacement, 0.0, 0.0,
                     0.71 + displacement,  0.0, 0.0};
  if (rank == 0) {
    frame.point_charge_positions = {0.2, 1.8 + displacement, -0.7};
    frame.point_charge_values = {0.25 - 0.1 * displacement};
  } else {
    frame.point_charge_positions = {0.2,  1.8 + displacement,  -0.7,
                                    -0.4, -1.6 - displacement, 0.9};
    frame.point_charge_values = {0.25 - 0.1 * displacement,
                                 -0.18 + 0.05 * displacement};
  }
  frame.atomic_potential_shifts = {0.003 + 0.01 * displacement,
                                   -0.002 - 0.01 * displacement};
  frame.charge_response_matrix = {0.02 + 0.01 * displacement, 0.001, 0.001,
                                  0.018};
  return frame;
}

bool all_finite(const double *values, std::size_t count) {
  return count == 0u || std::all_of(values, values + count, [](double value) {
           return std::isfinite(value);
         });
}

bool all_nan(const double *values, std::size_t count) {
  return count == 0u || std::all_of(values, values + count, [](double value) {
           return std::isnan(value);
         });
}

int compare_with_serial(const DPRC::WindowTopology &local_topology,
                        const DPRC::WindowFrame &local_frame,
                        const DPRC::WindowResultView &broker_result) {
  DPRC::WindowTopology serial_topology = local_topology;
  serial_topology.window_index = 0;
  DPRC::WindowFrame serial_frame = local_frame;
  serial_frame.window_index = 0;

  DPRC::XtbloomExecutorOptions serial_options;
  serial_options.backend = XTBLOOM_BACKEND_CPU;
  serial_options.cpu_threads = 1;
  DPRC::XtbloomPlanExecutor serial({serial_topology}, serial_options);
  serial.stage(serial_frame);
  const DPRC::XtbloomComputeOutcome outcome = serial.compute();
  if (outcome.call_status != XTBLOOM_STATUS_SUCCESS)
    return __LINE__;
  const DPRC::WindowResultView reference = serial.result_for_window(0);
  if (broker_result.status != XTBLOOM_STATUS_SUCCESS ||
      reference.status != XTBLOOM_STATUS_SUCCESS ||
      broker_result.atom_count != reference.atom_count ||
      broker_result.point_charge_count != reference.point_charge_count ||
      !near(broker_result.energy, reference.energy, 2.0e-10)) {
    std::cerr << "energy/status mismatch: broker=" << broker_result.energy
              << " serial=" << reference.energy << '\n';
    return __LINE__;
  }
  for (std::size_t coordinate = 0; coordinate < 3u * broker_result.atom_count;
       ++coordinate)
    if (!near(broker_result.forces[coordinate], reference.forces[coordinate],
              2.0e-8)) {
      std::cerr << "force mismatch: broker=" << broker_result.forces[coordinate]
                << " serial=" << reference.forces[coordinate] << '\n';
      return __LINE__;
    }
  for (std::size_t atom = 0; atom < broker_result.atom_count; ++atom)
    if (!near(broker_result.atomic_charges[atom],
              reference.atomic_charges[atom], 2.0e-8)) {
      std::cerr << "charge mismatch: broker="
                << broker_result.atomic_charges[atom]
                << " serial=" << reference.atomic_charges[atom] << '\n';
      return __LINE__;
    }
  for (std::size_t coordinate = 0;
       coordinate < 3u * broker_result.point_charge_count; ++coordinate)
    if (!near(broker_result.point_charge_forces[coordinate],
              reference.point_charge_forces[coordinate], 2.0e-8)) {
      std::cerr << "point-force mismatch: broker="
                << broker_result.point_charge_forces[coordinate]
                << " serial=" << reference.point_charge_forces[coordinate]
                << '\n';
      return __LINE__;
    }
  return 0;
}

template <class Function>
bool collectively_throws_invalid_argument(Function &&function) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  const int local = threw ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global != 0;
}

int run_test() {
  int rank = -1;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  CHECK(size == 2);

  const DPRC::WindowTopology topology = make_topology(rank);
  DPRC::XtbloomExecutorOptions options;
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 2;

  {
    DPRC::XtbloomPartitionBroker broker(MPI_COMM_WORLD, topology, options);
    const int local_owner = broker.owns_executor() ? 1 : 0;
    int owner_count = 0;
    MPI_Allreduce(&local_owner, &owner_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    CHECK(owner_count == 1);
    CHECK(broker.owns_executor() == (rank == 0));
    CHECK(broker.uses_shared_storage());

    const DPRC::WindowFrame first_frame =
        make_frame(rank, 10, 0.004 * static_cast<double>(rank));
    const DPRC::PartitionBrokerOutcome first = broker.compute(first_frame);
    CHECK(first.call_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(first.all_systems_succeeded);
    CHECK(first.start_policy == DPRC::SccStartPolicy::Fresh);
    CHECK((first.result_flags &
           XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES) != 0u);
    CHECK(broker.has_result());
    const DPRC::WindowResultView first_result =
        broker.result_for_local_window();
    CHECK(first_result.window_index == rank);
    CHECK(first_result.timestep == 10);
    CHECK(compare_with_serial(topology, first_frame, first_result) == 0);
    const double *force_storage = first_result.forces;
    const double *charge_storage = first_result.atomic_charges;

    DPRC::WindowFrame malformed = make_frame(rank, 11, 0.002);
    if (rank == 1)
      malformed.point_charge_values.pop_back();
    CHECK(collectively_throws_invalid_argument(
        [&] { static_cast<void>(broker.compute(malformed)); }));
    CHECK(broker.has_result());

    DPRC::WindowFrame misaligned = make_frame(rank, 11 + rank, 0.002);
    CHECK(collectively_throws_invalid_argument(
        [&] { static_cast<void>(broker.compute(misaligned)); }));
    CHECK(broker.has_result());

    const DPRC::WindowFrame second_frame =
        make_frame(rank, 11, 0.006 + 0.003 * static_cast<double>(rank));
    const DPRC::PartitionBrokerOutcome second = broker.compute(second_frame);
    CHECK(second.call_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(second.all_systems_succeeded);
    CHECK(second.start_policy == DPRC::SccStartPolicy::Warm);
    const DPRC::WindowResultView second_result =
        broker.result_for_local_window();
    CHECK(second_result.forces == force_storage);
    CHECK(second_result.atomic_charges == charge_storage);
    CHECK(second_result.status == XTBLOOM_STATUS_SUCCESS);
    CHECK(second_result.scc_converged);
    CHECK(std::isfinite(second_result.energy));

    DPRC::WindowFrame rejected_frame = make_frame(rank, 12, 0.008);
    if (rank == 1)
      rejected_frame.positions[0] = std::numeric_limits<double>::quiet_NaN();
    const DPRC::PartitionBrokerOutcome rejected =
        broker.compute(rejected_frame);
    CHECK(rejected.call_status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(!rejected.all_systems_succeeded);
    CHECK(rejected.start_policy == DPRC::SccStartPolicy::Warm);
    CHECK(!broker.has_result());
    CHECK(!broker.last_error().empty());

    const DPRC::WindowFrame recovery_frame =
        make_frame(rank, 13, 0.010 + 0.002 * static_cast<double>(rank));
    const DPRC::PartitionBrokerOutcome recovered =
        broker.compute(recovery_frame);
    CHECK(recovered.call_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(recovered.all_systems_succeeded);
    CHECK(recovered.start_policy == DPRC::SccStartPolicy::Fresh);
    CHECK(broker.has_result());
    CHECK(compare_with_serial(topology, recovery_frame,
                              broker.result_for_local_window()) == 0);
  }

  {
    DPRC::XtbloomPartitionBroker broker(MPI_COMM_WORLD, topology, options);
    const DPRC::PartitionBrokerOutcome initial =
        broker.compute(make_frame(rank, 20, 0.0));
    CHECK(initial.call_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(initial.start_policy == DPRC::SccStartPolicy::Fresh);

    DPRC::WindowFrame peer_frame = make_frame(rank, 21, 0.0);
    if (rank == 1)
      peer_frame.positions = {-0.55e-6, 0.0, 0.0, 0.55e-6, 0.0, 0.0};
    const DPRC::PartitionBrokerOutcome peer_failure =
        broker.compute(peer_frame);
    CHECK(peer_failure.call_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(!peer_failure.all_systems_succeeded);
    CHECK(peer_failure.start_policy == DPRC::SccStartPolicy::Warm);
    CHECK(broker.has_result());
    const DPRC::WindowResultView local_result =
        broker.result_for_local_window();
    if (rank == 0) {
      CHECK(local_result.status == XTBLOOM_STATUS_SUCCESS);
      CHECK(local_result.scc_converged);
      CHECK(std::isfinite(local_result.energy));
      CHECK(all_finite(local_result.forces, 3u * local_result.atom_count));
      CHECK(all_finite(local_result.atomic_charges, local_result.atom_count));
      CHECK(all_finite(local_result.point_charge_forces,
                       3u * local_result.point_charge_count));
    } else {
      CHECK(local_result.status == XTBLOOM_STATUS_EIGENSOLVER_FAILED ||
            local_result.status == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
      CHECK(!local_result.scc_converged);
      CHECK(std::isnan(local_result.energy));
      CHECK(all_nan(local_result.forces, 3u * local_result.atom_count));
      CHECK(all_nan(local_result.atomic_charges, local_result.atom_count));
      CHECK(all_nan(local_result.point_charge_forces,
                    3u * local_result.point_charge_count));
    }

    const DPRC::PartitionBrokerOutcome recovered =
        broker.compute(make_frame(rank, 22, 0.0));
    CHECK(recovered.call_status == XTBLOOM_STATUS_SUCCESS);
    CHECK(recovered.all_systems_succeeded);
    CHECK(recovered.start_policy == DPRC::SccStartPolicy::Fresh);
    CHECK(broker.result_for_local_window().status == XTBLOOM_STATUS_SUCCESS);
  }

  DPRC::WindowTopology invalid_topology = topology;
  if (rank == 1)
    invalid_topology.window_index = 0;
  CHECK(collectively_throws_invalid_argument([&] {
    DPRC::XtbloomPartitionBroker invalid(MPI_COMM_WORLD, invalid_topology,
                                         options);
  }));

  DPRC::XtbloomExecutorOptions mismatched_options = options;
  if (rank == 1)
    mismatched_options.cpu_threads = 1;
  CHECK(collectively_throws_invalid_argument([&] {
    DPRC::XtbloomPartitionBroker invalid(MPI_COMM_WORLD, topology,
                                         mismatched_options);
  }));

  if (rank == 0)
    std::cout << "partition broker: one native owner, shared ragged QM/MM staging, "
                 "peer-local failure isolation, WARM recovery, and collective "
                 "rejection passed\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  const int result = run_test();
  MPI_Finalize();
  return result;
}
