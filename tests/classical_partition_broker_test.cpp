#include "classical_partition_broker.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
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

constexpr std::size_t kAtoms = 4;
static_assert(
    noexcept(std::declval<DPRC::ClassicalPartitionBroker &>().cancel()));

[[nodiscard]] bool near(double left, double right, double absolute = 2.0e-9,
                        double relative = 2.0e-9) {
  return std::abs(left - right) <=
         absolute + relative * std::max(std::abs(left), std::abs(right));
}

[[nodiscard]] bool vectors_near(const std::vector<double> &left,
                                const std::vector<double> &right) {
  if (left.size() != right.size())
    return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!near(left[index], right[index]))
      return false;
  return true;
}

[[nodiscard]] DPRC::ClassicalTopology make_topology() {
  DPRC::ClassicalTopology topology;
  topology.atom_count = kAtoms;
  topology.type_count = 3;
  topology.atom_types = {0, 1, 1, 2};
  topology.tip4p_sites = {{0, 1, 2}};
  topology.special_pairs = {
      {0, 1, 0.0, 0.0}, {0, 2, 0.0, 0.0}, {1, 2, 0.0, 0.0}};
  topology.lj.resize(9);
  topology.coulomb_type_pairs.assign(9, 1);
  for (int type1 = 0; type1 < 3; ++type1) {
    for (int type2 = 0; type2 < 3; ++type2) {
      auto &entry = topology.lj[static_cast<std::size_t>(3 * type1 + type2)];
      const double epsilon = type1 == 1 || type2 == 1 ? 0.0 : 0.08;
      const double sigma = type1 == 2 || type2 == 2 ? 3.1 : 3.2;
      entry.lj1 = 48.0 * epsilon * std::pow(sigma, 12.0);
      entry.lj2 = 24.0 * epsilon * std::pow(sigma, 6.0);
      entry.lj3 = 4.0 * epsilon * std::pow(sigma, 12.0);
      entry.lj4 = 4.0 * epsilon * std::pow(sigma, 6.0);
      entry.cutoff = 6.0;
    }
  }
  topology.cell.boxlo = {-1.0, 0.5, -0.25};
  topology.cell.h = {12.0, 1.7, -0.8, 0.0, 11.0, 1.2, 0.0, 0.0, 10.0};
  topology.pppm.mesh = {24, 24, 24};
  topology.pppm.order = 4;
  topology.pppm.g_ewald = 0.31;
  topology.tip4p_alpha = 0.21328275680467643;
  topology.tip4p_qdist = 0.125;
  topology.real_space_cutoff = 6.0;
  topology.neighbor_skin = 0.5;
  topology.qqrd2e = 332.06371;
  return topology;
}

[[nodiscard]] std::vector<double> make_positions(int rank) {
  std::vector<double> positions = {1.0, 2.0, 2.5, 1.9, 2.1, 2.6,
                                   0.7, 2.8, 2.4, 5.1, 4.4, 3.8};
  positions[9] += 0.017 * static_cast<double>(rank);
  positions[10] -= 0.008 * static_cast<double>(rank);
  return positions;
}

struct ReferenceResult {
  std::vector<double> pair_forces;
  std::vector<double> mm_forces;
  std::vector<double> mm_potential;
  std::vector<double> pair_virial;
  std::vector<double> mm_virial;
  std::vector<double> qm_forces;
  std::vector<double> full_forces;
  std::vector<double> qm_virial;
  std::vector<double> full_virial;
  double lj_energy = 0.0;
  double coulomb_energy = 0.0;
  double mm_energy = 0.0;
  double qm_energy = 0.0;
  double full_energy = 0.0;
};

[[nodiscard]] ReferenceResult
run_reference(const DPRC::ClassicalTopology &topology,
              const std::vector<double> &positions,
              const std::vector<double> &mm_charges,
              const std::vector<double> &qm_charges) {
  DPRC::ClassicalPlanOptions options;
  options.backend = DPRC::ClassicalBackend::CPU;
  options.max_batch_count = 1;
  auto plan = DPRC::create_classical_batch_plan(topology, options);

  ReferenceResult result;
  result.pair_forces.resize(3 * kAtoms);
  result.mm_forces.resize(3 * kAtoms);
  result.mm_potential.resize(kAtoms);
  result.pair_virial.resize(6);
  result.mm_virial.resize(6);
  result.qm_forces.resize(3 * kAtoms);
  result.full_forces.resize(3 * kAtoms);
  result.qm_virial.resize(6);
  result.full_virial.resize(6);

  DPRC::ClassicalBatchInput mm_input{1, positions.data(), mm_charges.data()};
  DPRC::ClassicalMmBatchOutput mm_output{1,
                                         result.pair_forces.data(),
                                         &result.lj_energy,
                                         &result.coulomb_energy,
                                         result.pair_virial.data(),
                                         &result.mm_energy,
                                         result.mm_virial.data(),
                                         result.mm_potential.data(),
                                         result.mm_forces.data()};
  plan->begin_mm(mm_input, mm_output);

  DPRC::ClassicalQmBatchInput qm_input{1, qm_charges.data()};
  DPRC::ClassicalQmBatchOutput qm_output{1,
                                         result.qm_forces.data(),
                                         result.full_forces.data(),
                                         &result.qm_energy,
                                         &result.full_energy,
                                         result.qm_virial.data(),
                                         result.full_virial.data()};
  plan->finish_qm(qm_input, qm_output);
  return result;
}

struct MmSnapshot {
  std::int64_t timestep = -1;
  std::vector<double> pair_forces;
  std::vector<double> pair_virial;
  std::vector<double> mm_virial;
  std::vector<double> potential;
  double lj = 0.0;
  double coulomb = 0.0;
  double mm = 0.0;
};

[[nodiscard]] MmSnapshot snapshot(const DPRC::ClassicalMmResultView &view) {
  return {view.timestep,
          {view.pair_forces, view.pair_forces + 3 * view.atom_count},
          {view.pair_virial, view.pair_virial + 6},
          {view.mm_pppm_virial, view.mm_pppm_virial + 6},
          {view.mm_pppm_potential, view.mm_pppm_potential + view.atom_count},
          view.lj_energy,
          view.coulomb_energy,
          view.mm_pppm_energy};
}

struct QmSnapshot {
  std::int64_t timestep = -1;
  std::vector<double> qm_forces;
  std::vector<double> full_forces;
  std::vector<double> qm_virial;
  std::vector<double> full_virial;
  double qm = 0.0;
  double full = 0.0;
};

[[nodiscard]] QmSnapshot snapshot(const DPRC::ClassicalQmResultView &view) {
  return {view.timestep,
          {view.qm_pppm_forces, view.qm_pppm_forces + 3 * view.atom_count},
          {view.full_pppm_forces, view.full_pppm_forces + 3 * view.atom_count},
          {view.qm_pppm_virial, view.qm_pppm_virial + 6},
          {view.full_pppm_virial, view.full_pppm_virial + 6},
          view.qm_pppm_energy,
          view.full_pppm_energy};
}

[[nodiscard]] bool same(const MmSnapshot &left, const MmSnapshot &right) {
  return left.timestep == right.timestep &&
         vectors_near(left.pair_forces, right.pair_forces) &&
         vectors_near(left.pair_virial, right.pair_virial) &&
         vectors_near(left.mm_virial, right.mm_virial) &&
         vectors_near(left.potential, right.potential) &&
         near(left.lj, right.lj) && near(left.coulomb, right.coulomb) &&
         near(left.mm, right.mm);
}

[[nodiscard]] bool same(const QmSnapshot &left, const QmSnapshot &right) {
  return left.timestep == right.timestep &&
         vectors_near(left.qm_forces, right.qm_forces) &&
         vectors_near(left.full_forces, right.full_forces) &&
         vectors_near(left.qm_virial, right.qm_virial) &&
         vectors_near(left.full_virial, right.full_virial) &&
         near(left.qm, right.qm) && near(left.full, right.full);
}

template <class Exception, class Function>
bool collectively_throws(Function &&function) {
  bool threw = false;
  try {
    function();
  } catch (const Exception &) {
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
  CHECK(size > 0);

  const DPRC::ClassicalTopology topology = make_topology();
  const std::vector<double> positions = make_positions(rank);
  const std::vector<double> mm_charges = {-1.04, 0.52, 0.52, 0.0};
  const std::vector<double> qm_charges = {
      0.0, 0.0, 0.0, -0.35 - 0.01 * static_cast<double>(rank)};
  const ReferenceResult reference =
      run_reference(topology, positions, mm_charges, qm_charges);

  DPRC::ClassicalPlanOptions options;
  options.backend = DPRC::ClassicalBackend::CPU;
  options.max_batch_count = static_cast<std::size_t>(size);

  {
    DPRC::ClassicalPartitionBroker broker(MPI_COMM_WORLD, topology, options);
    const int local_owner = broker.owns_plan() ? 1 : 0;
    int owner_count = 0;
    MPI_Allreduce(&local_owner, &owner_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    CHECK(owner_count == 1);
    CHECK(broker.owns_plan() == (rank == 0));
    CHECK(!broker.has_mm_result());
    CHECK(!broker.has_qm_result());

    broker.begin_mm({10, positions.data(), positions.size(), mm_charges.data(),
                     mm_charges.size()});
    CHECK(broker.has_active_mm_epoch());
    CHECK(broker.has_mm_result());
    const MmSnapshot first_mm = snapshot(broker.mm_result());
    CHECK(first_mm.timestep == 10);
    CHECK(vectors_near(first_mm.pair_forces, reference.pair_forces));
    CHECK(vectors_near(first_mm.pair_virial, reference.pair_virial));
    CHECK(vectors_near(first_mm.mm_virial, reference.mm_virial));
    CHECK(vectors_near(first_mm.potential, reference.mm_potential));
    CHECK(near(first_mm.lj, reference.lj_energy));
    CHECK(near(first_mm.coulomb, reference.coulomb_energy));
    CHECK(near(first_mm.mm, reference.mm_energy));

    broker.finish_qm({10, qm_charges.data(), qm_charges.size()});
    CHECK(!broker.has_active_mm_epoch());
    CHECK(broker.has_qm_result());
    const QmSnapshot first_qm = snapshot(broker.qm_result());
    CHECK(first_qm.timestep == 10);
    CHECK(vectors_near(first_qm.qm_forces, reference.qm_forces));
    CHECK(vectors_near(first_qm.full_forces, reference.full_forces));
    CHECK(vectors_near(first_qm.qm_virial, reference.qm_virial));
    CHECK(vectors_near(first_qm.full_virial, reference.full_virial));
    CHECK(near(first_qm.qm, reference.qm_energy));
    CHECK(near(first_qm.full, reference.full_energy));

    std::vector<double> malformed_positions = positions;
    if (rank == size - 1)
      malformed_positions[2] = std::numeric_limits<double>::quiet_NaN();
    CHECK(collectively_throws<std::invalid_argument>([&] {
      broker.begin_mm({11, malformed_positions.data(),
                       malformed_positions.size(), mm_charges.data(),
                       mm_charges.size()});
    }));
    CHECK(!broker.has_active_mm_epoch());
    CHECK(same(snapshot(broker.mm_result()), first_mm));
    CHECK(same(snapshot(broker.qm_result()), first_qm));

    CHECK(collectively_throws<std::invalid_argument>(
        [&] { broker.finish_qm({11, qm_charges.data(), qm_charges.size()}); }));
    CHECK(same(snapshot(broker.mm_result()), first_mm));
    CHECK(same(snapshot(broker.qm_result()), first_qm));

    if (size >= 2) {
      const DPRC::ClassicalMmResultRequest mismatched_request =
          rank == 0 ? DPRC::ClassicalMmResultRequest{true, false, true}
                    : DPRC::ClassicalMmResultRequest{false, true, false};
      CHECK(collectively_throws<std::invalid_argument>([&] {
        broker.begin_mm({25, positions.data(), positions.size(),
                         mm_charges.data(), mm_charges.size()},
                        mismatched_request);
      }));
      CHECK(!broker.has_active_mm_epoch());
      CHECK(same(snapshot(broker.mm_result()), first_mm));

      CHECK(collectively_throws<std::invalid_argument>([&] {
        broker.begin_mm({20 + rank, positions.data(), positions.size(),
                         mm_charges.data(), mm_charges.size()});
      }));
      CHECK(!broker.has_active_mm_epoch());
      CHECK(same(snapshot(broker.mm_result()), first_mm));
      CHECK(same(snapshot(broker.qm_result()), first_qm));
    }

    broker.begin_mm({30, positions.data(), positions.size(), mm_charges.data(),
                     mm_charges.size()});
    CHECK(broker.has_active_mm_epoch());
    CHECK(collectively_throws<std::invalid_argument>(
        [&] { broker.finish_qm({31, qm_charges.data(), qm_charges.size()}); }));
    CHECK(broker.has_active_mm_epoch());
    CHECK(same(snapshot(broker.qm_result()), first_qm));
    broker.cancel();
    CHECK(!broker.has_active_mm_epoch());
    CHECK(same(snapshot(broker.qm_result()), first_qm));
    CHECK(collectively_throws<std::invalid_argument>(
        [&] { broker.finish_qm({30, qm_charges.data(), qm_charges.size()}); }));
    CHECK(same(snapshot(broker.qm_result()), first_qm));

    broker.begin_mm({40, positions.data(), positions.size(), mm_charges.data(),
                     mm_charges.size()},
                    {/*pppm_potential=*/false, /*pppm_forces=*/true,
                     /*retain_for_qm=*/false});
    CHECK(!broker.has_active_mm_epoch());
    const DPRC::ClassicalMmResultView pure_mm = broker.mm_result();
    CHECK(pure_mm.timestep == 40);
    CHECK(pure_mm.mm_pppm_potential == nullptr);
    CHECK(pure_mm.mm_pppm_forces != nullptr);
    CHECK(vectors_near(
        std::vector<double>(pure_mm.mm_pppm_forces,
                            pure_mm.mm_pppm_forces +
                                3 * pure_mm.atom_count),
        reference.mm_forces));
    CHECK(collectively_throws<std::invalid_argument>(
        [&] { broker.finish_qm({40, qm_charges.data(), qm_charges.size()}); }));
  }

  if (size >= 2) {
    DPRC::ClassicalTopology mismatched = topology;
    if (rank == 1)
      mismatched.real_space_cutoff += 0.125;
    CHECK(collectively_throws<std::invalid_argument>([&] {
      DPRC::ClassicalPartitionBroker rejected(MPI_COMM_WORLD, mismatched,
                                              options);
    }));

    DPRC::ClassicalPlanOptions mismatched_options = options;
    if (rank == 1)
      mismatched_options.cuda_device = 0;
    CHECK(collectively_throws<std::invalid_argument>([&] {
      DPRC::ClassicalPartitionBroker rejected(MPI_COMM_WORLD, topology,
                                              mismatched_options);
    }));
  }

  DPRC::ClassicalPlanOptions insufficient_capacity = options;
  insufficient_capacity.max_batch_count = static_cast<std::size_t>(size - 1);
  CHECK(collectively_throws<std::invalid_argument>([&] {
    DPRC::ClassicalPartitionBroker rejected(MPI_COMM_WORLD, topology,
                                            insufficient_capacity);
  }));

  if (rank == 0)
    std::cout << "classical partition broker: one owner, stable rank slots, "
                 "batched LJ/TIP4P/PPPM parity, and transactional publication "
                 "passed\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    result = run_test();
  } catch (const std::exception &error) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "rank " << rank << " unexpected exception: " << error.what()
              << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
