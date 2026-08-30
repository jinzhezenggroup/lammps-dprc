#include "classical_batch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(DPRC_TEST_CUFFT_WRAP)
#include <cufft.h>

namespace {

// The production plan deliberately exposes no profiling API.  GNU/LLVM
// linker wrapping lets this test observe only the public cuFFT calls emitted
// by the statically linked backend, without adding counters or symbols to the
// plugin itself.
struct CufftCallTrace {
  bool enabled = false;
  std::vector<int> planned_batches;
  std::vector<std::pair<int, int>> executions;
  std::vector<std::pair<cufftHandle, int>> handle_batches;

  void reset() {
    planned_batches.clear();
    executions.clear();
    handle_batches.clear();
  }

  [[nodiscard]] int batch_for(cufftHandle handle) const {
    const auto match = std::find_if(
        handle_batches.begin(), handle_batches.end(),
        [handle](const auto &entry) { return entry.first == handle; });
    return match == handle_batches.end() ? -1 : match->second;
  }
};

CufftCallTrace cufft_trace;

}  // namespace

extern "C" cufftResult __real_cufftMakePlanMany(
    cufftHandle, int, int *, int *, int, int, int *, int, int, cufftType, int,
    std::size_t *);
extern "C" cufftResult __real_cufftExecZ2Z(
    cufftHandle, cufftDoubleComplex *, cufftDoubleComplex *, int);

extern "C" cufftResult __wrap_cufftMakePlanMany(
    cufftHandle plan, int rank, int *dimensions, int *input_embedding,
    int input_stride, int input_distance, int *output_embedding,
    int output_stride, int output_distance, cufftType type, int batch,
    std::size_t *workspace_bytes) {
  const cufftResult result = __real_cufftMakePlanMany(
      plan, rank, dimensions, input_embedding, input_stride, input_distance,
      output_embedding, output_stride, output_distance, type, batch,
      workspace_bytes);
  if (cufft_trace.enabled && result == CUFFT_SUCCESS) {
    cufft_trace.planned_batches.push_back(batch);
    cufft_trace.handle_batches.emplace_back(plan, batch);
  }
  return result;
}

extern "C" cufftResult __wrap_cufftExecZ2Z(
    cufftHandle plan, cufftDoubleComplex *input, cufftDoubleComplex *output,
    int direction) {
  if (cufft_trace.enabled)
    cufft_trace.executions.emplace_back(cufft_trace.batch_for(plan), direction);
  return __real_cufftExecZ2Z(plan, input, output, direction);
}
#endif

namespace {

#define CHECK(condition)                                                         \
  do {                                                                           \
    if (!(condition)) {                                                          \
      std::cerr << "check failed at line " << __LINE__ << ": " #condition       \
                << '\n';                                                         \
      return 1;                                                                  \
    }                                                                            \
  } while (false)

constexpr std::size_t kAtoms = 4;

[[nodiscard]] bool near(double left, double right, double absolute = 2.0e-9,
                        double relative = 2.0e-9) {
  return std::abs(left - right) <=
      absolute + relative * std::max(std::abs(left), std::abs(right));
}

[[nodiscard]] DPRC::ClassicalTopology make_topology() {
  DPRC::ClassicalTopology topology;
  topology.atom_count = kAtoms;
  topology.type_count = 3;
  topology.atom_types = {0, 1, 1, 2};
  topology.tip4p_sites = {{0, 1, 2}};
  topology.special_pairs = {{0, 1, 0.0, 0.0}, {0, 2, 0.0, 0.0},
                            {1, 2, 0.0, 0.0}};
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

[[nodiscard]] DPRC::ClassicalTopology make_table_topology() {
  DPRC::ClassicalTopology topology = make_topology();
  auto &table = topology.coulomb_table;
  table.bits = 1;
  table.shift_bits = 0;
  table.mask = 1;
  table.inner_squared = 0.0;
  // Both bitmap entries intentionally carry the same affine value.  This
  // isolates the exact float-bit lookup, special-pair complement, and unit
  // handling without making the fixture depend on one host's float payload.
  table.r.assign(2, 0.0);
  table.dr.assign(2, 0.0);
  table.force.assign(2, 83.0);
  table.dforce.assign(2, 0.0);
  table.coulomb.assign(2, 71.0);
  table.dcoulomb.assign(2, 0.0);
  table.energy.assign(2, 37.0);
  table.denergy.assign(2, 0.0);
  return topology;
}

struct StagedResult {
  std::vector<double> pair_forces;
  std::vector<double> mm_forces;
  std::vector<double> full_forces;
  std::vector<double> qm_forces;
  std::vector<double> potential;
  std::vector<double> lj;
  std::vector<double> coulomb;
  std::vector<double> mm_energy;
  std::vector<double> qm_energy;
  std::vector<double> full_energy;
  std::vector<double> pair_virial;
  std::vector<double> mm_virial;
  std::vector<double> qm_virial;
  std::vector<double> full_virial;
};

[[nodiscard]] StagedResult run(DPRC::ClassicalBatchPlan &plan, std::size_t batch,
                               const std::vector<double> &positions,
                               const std::vector<double> &mm_charges,
                               const std::vector<double> &qm_charges) {
  StagedResult result;
  result.pair_forces.assign(3 * batch * kAtoms, 0.0);
  result.mm_forces.assign(3 * batch * kAtoms, 0.0);
  result.full_forces.assign(3 * batch * kAtoms, 0.0);
  result.qm_forces.assign(3 * batch * kAtoms, 0.0);
  result.potential.assign(batch * kAtoms, 0.0);
  result.lj.assign(batch, 0.0);
  result.coulomb.assign(batch, 0.0);
  result.mm_energy.assign(batch, 0.0);
  result.qm_energy.assign(batch, 0.0);
  result.full_energy.assign(batch, 0.0);
  result.pair_virial.assign(6 * batch, 0.0);
  result.mm_virial.assign(6 * batch, 0.0);
  result.qm_virial.assign(6 * batch, 0.0);
  result.full_virial.assign(6 * batch, 0.0);

  DPRC::ClassicalBatchInput mm_input{batch, positions.data(), mm_charges.data()};
  DPRC::ClassicalMmBatchOutput mm_output{
      batch, result.pair_forces.data(), result.lj.data(), result.coulomb.data(),
      result.pair_virial.data(), result.mm_energy.data(), result.mm_virial.data(),
      result.potential.data(), result.mm_forces.data()};
  plan.begin_mm(mm_input, mm_output);

  DPRC::ClassicalQmBatchInput qm_input{batch, qm_charges.data()};
  DPRC::ClassicalQmBatchOutput qm_output{
      batch, result.qm_forces.data(), result.full_forces.data(),
      result.qm_energy.data(), result.full_energy.data(), result.qm_virial.data(),
      result.full_virial.data()};
  plan.finish_qm(qm_input, qm_output);
  return result;
}

[[nodiscard]] double total_energy(const StagedResult &result) {
  return result.lj[0] + result.coulomb[0] + result.full_energy[0];
}

[[nodiscard]] bool vectors_near(const std::vector<double> &left,
                                const std::vector<double> &right,
                                double absolute = 2.0e-8,
                                double relative = 2.0e-8) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!near(left[index], right[index], absolute, relative)) {
      std::cerr << "vector mismatch at " << index << ": CPU=" << left[index]
                << " CUDA=" << right[index] << '\n';
      return false;
    }
  return true;
}

[[nodiscard]] bool results_near(const StagedResult &left,
                                const StagedResult &right) {
  return vectors_near(left.pair_forces, right.pair_forces) &&
      vectors_near(left.mm_forces, right.mm_forces) &&
      vectors_near(left.full_forces, right.full_forces) &&
      vectors_near(left.qm_forces, right.qm_forces) &&
      vectors_near(left.potential, right.potential) &&
      vectors_near(left.lj, right.lj) &&
      vectors_near(left.coulomb, right.coulomb) &&
      vectors_near(left.mm_energy, right.mm_energy) &&
      vectors_near(left.qm_energy, right.qm_energy) &&
      vectors_near(left.full_energy, right.full_energy) &&
      vectors_near(left.pair_virial, right.pair_virial) &&
      vectors_near(left.mm_virial, right.mm_virial) &&
      vectors_near(left.qm_virial, right.qm_virial) &&
      vectors_near(left.full_virial, right.full_virial);
}

[[nodiscard]] bool frame_near(const StagedResult &batch, std::size_t frame,
                              const StagedResult &expected) {
  const auto slice_near = [frame](const std::vector<double> &values,
                                  const std::vector<double> &reference,
                                  std::size_t width) {
    for (std::size_t index = 0; index < width; ++index)
      if (!near(values[frame * width + index], reference[index], 2.0e-8,
                2.0e-8))
        return false;
    return true;
  };
  return slice_near(batch.pair_forces, expected.pair_forces, 3 * kAtoms) &&
      slice_near(batch.mm_forces, expected.mm_forces, 3 * kAtoms) &&
      slice_near(batch.full_forces, expected.full_forces, 3 * kAtoms) &&
      slice_near(batch.qm_forces, expected.qm_forces, 3 * kAtoms) &&
      slice_near(batch.potential, expected.potential, kAtoms) &&
      slice_near(batch.lj, expected.lj, 1) &&
      slice_near(batch.coulomb, expected.coulomb, 1) &&
      slice_near(batch.mm_energy, expected.mm_energy, 1) &&
      slice_near(batch.qm_energy, expected.qm_energy, 1) &&
      slice_near(batch.full_energy, expected.full_energy, 1) &&
      slice_near(batch.pair_virial, expected.pair_virial, 6) &&
      slice_near(batch.mm_virial, expected.mm_virial, 6) &&
      slice_near(batch.qm_virial, expected.qm_virial, 6) &&
      slice_near(batch.full_virial, expected.full_virial, 6);
}

[[nodiscard]] bool pair_scalars_equal(const StagedResult &batch,
                                      std::size_t frame,
                                      const StagedResult &expected) {
  if (batch.lj[frame] != expected.lj[0] ||
      batch.coulomb[frame] != expected.coulomb[0])
    return false;
  for (int component = 0; component < 6; ++component)
    if (batch.pair_virial[6 * frame + component] !=
        expected.pair_virial[component])
      return false;
  return true;
}

}  // namespace

int main() {
  try {
    DPRC::ClassicalPlanOptions options;
    options.backend = DPRC::ClassicalBackend::CPU;
    options.max_batch_count = 2;
    auto plan = DPRC::create_classical_batch_plan(make_topology(), options);
    CHECK(plan->backend() == DPRC::ClassicalBackend::CPU);
    CHECK(plan->max_batch_count() == 2);

    const std::vector<double> frame0 = {
        1.0, 2.0, 2.5, 1.9, 2.1, 2.6, 0.7, 2.8, 2.4, 5.1, 4.4, 3.8};
    std::vector<double> frame1 = frame0;
    frame1[9] += 0.17;
    frame1[10] -= 0.08;
    std::vector<double> positions = frame0;
    positions.insert(positions.end(), frame1.begin(), frame1.end());

    const std::vector<double> mm_one = {-1.04, 0.52, 0.52, 0.0};
    const std::vector<double> qm_one = {0.0, 0.0, 0.0, -0.35};
    std::vector<double> mm = mm_one;
    mm.insert(mm.end(), mm_one.begin(), mm_one.end());
    std::vector<double> qm = qm_one;
    qm.insert(qm.end(), qm_one.begin(), qm_one.end());

    const StagedResult batch = run(*plan, 2, positions, mm, qm);

    options.max_batch_count = 1;
    auto sequential_plan = DPRC::create_classical_batch_plan(make_topology(), options);
    const StagedResult first = run(*sequential_plan, 1, frame0, mm_one, qm_one);
    const StagedResult second = run(*sequential_plan, 1, frame1, mm_one, qm_one);
    for (std::size_t coordinate = 0; coordinate < 3 * kAtoms; ++coordinate) {
      CHECK(near(batch.pair_forces[coordinate], first.pair_forces[coordinate]));
      CHECK(near(batch.mm_forces[coordinate], first.mm_forces[coordinate]));
      CHECK(near(batch.full_forces[coordinate], first.full_forces[coordinate]));
      CHECK(near(batch.pair_forces[3 * kAtoms + coordinate],
                 second.pair_forces[coordinate]));
      CHECK(near(batch.mm_forces[3 * kAtoms + coordinate],
                 second.mm_forces[coordinate]));
      CHECK(near(batch.full_forces[3 * kAtoms + coordinate],
                 second.full_forces[coordinate]));
    }
    CHECK(near(batch.full_energy[0], first.full_energy[0]));
    CHECK(near(batch.full_energy[1], second.full_energy[0]));

    // The staged MM+QM reciprocal assembly must equal a direct full-charge
    // solve.  This independently exercises the bilinear energy/virial term.
    std::vector<double> full_charge = mm_one;
    for (std::size_t atom = 0; atom < kAtoms; ++atom) full_charge[atom] += qm_one[atom];
    const std::vector<double> zero_charge(kAtoms, 0.0);
    const StagedResult direct =
        run(*sequential_plan, 1, frame0, full_charge, zero_charge);
    CHECK(near(first.full_energy[0], direct.full_energy[0], 5.0e-9, 5.0e-9));
    for (int component = 0; component < 6; ++component)
      CHECK(near(first.full_virial[component], direct.full_virial[component],
                 5.0e-9, 5.0e-9));
    for (std::size_t coordinate = 0; coordinate < 3 * kAtoms; ++coordinate)
      CHECK(near(first.full_forces[coordinate], direct.full_forces[coordinate],
                 5.0e-9, 5.0e-9));

    // Central finite difference for a non-TIP4P atom qualifies the composed
    // pair plus reciprocal force without constrained-water complications.
    const double step = 2.0e-5;
    std::vector<double> plus = frame0;
    std::vector<double> minus = frame0;
    plus[9] += step;
    minus[9] -= step;
    const StagedResult plus_result = run(*sequential_plan, 1, plus, mm_one, qm_one);
    const StagedResult minus_result = run(*sequential_plan, 1, minus, mm_one, qm_one);
    const double numerical_force =
        -(total_energy(plus_result) - total_energy(minus_result)) / (2.0 * step);
    const double analytic_force = first.pair_forces[9] + first.full_forces[9];
    // ik differentiation is not the exact derivative of the discretized mesh
    // energy.  This 24^3 mesh keeps the expected interpolation discrepancy
    // below 5e-4 while still detecting sign, unit, and composition errors.
    if (!near(numerical_force, analytic_force, 5.0e-4, 2.0e-5))
      std::cerr << "finite-difference force mismatch: numerical=" << numerical_force
                << " analytic=" << analytic_force << '\n';
    CHECK(near(numerical_force, analytic_force, 5.0e-4, 2.0e-5));

#if defined(DPRC_TEST_CLASSICAL_CUDA)
    // The production CUDA path is eligible for timing only after every
    // staged output agrees with the readable CPU oracle.  This comparison
    // includes the pair accounts, raw MM potential needed by xTBloom, and the
    // distinct QM-only/full reciprocal publications used by the correction
    // algebra.
    DPRC::ClassicalPlanOptions cuda_options;
    cuda_options.backend = DPRC::ClassicalBackend::CUDA;
    cuda_options.max_batch_count = 2;
#if defined(DPRC_TEST_CUFFT_WRAP)
    cufft_trace.reset();
    cufft_trace.enabled = true;
#endif
    auto cuda_plan = DPRC::create_classical_batch_plan(make_topology(), cuda_options);
    CHECK(cuda_plan->backend() == DPRC::ClassicalBackend::CUDA);
    const StagedResult cuda_batch = run(*cuda_plan, 2, positions, mm, qm);
#if defined(DPRC_TEST_CUFFT_WRAP)
    cufft_trace.enabled = false;
    CHECK(cufft_trace.planned_batches ==
          (std::vector<int>{2, 8, 6}));
    CHECK(cufft_trace.executions ==
          (std::vector<std::pair<int, int>>{
              {2, CUFFT_FORWARD}, {8, CUFFT_INVERSE},
              {2, CUFFT_FORWARD}, {6, CUFFT_INVERSE}}));
#endif
    CHECK(vectors_near(batch.pair_forces, cuda_batch.pair_forces));
    CHECK(vectors_near(batch.mm_forces, cuda_batch.mm_forces));
    CHECK(vectors_near(batch.full_forces, cuda_batch.full_forces));
    CHECK(vectors_near(batch.qm_forces, cuda_batch.qm_forces));
    CHECK(vectors_near(batch.potential, cuda_batch.potential));
    CHECK(vectors_near(batch.lj, cuda_batch.lj));
    CHECK(vectors_near(batch.coulomb, cuda_batch.coulomb));
    CHECK(vectors_near(batch.mm_energy, cuda_batch.mm_energy));
    CHECK(vectors_near(batch.qm_energy, cuda_batch.qm_energy));
    CHECK(vectors_near(batch.full_energy, cuda_batch.full_energy));
    CHECK(vectors_near(batch.pair_virial, cuda_batch.pair_virial));
    CHECK(vectors_near(batch.mm_virial, cuda_batch.mm_virial));
    CHECK(vectors_near(batch.qm_virial, cuda_batch.qm_virial));
    CHECK(vectors_near(batch.full_virial, cuda_batch.full_virial));

    cuda_options.max_batch_count = 1;
    auto cuda_sequential =
        DPRC::create_classical_batch_plan(make_topology(), cuda_options);
    const StagedResult cuda_first =
        run(*cuda_sequential, 1, frame0, mm_one, qm_one);
    const StagedResult cuda_second =
        run(*cuda_sequential, 1, frame1, mm_one, qm_one);
    CHECK(results_near(first, cuda_first));
    CHECK(results_near(second, cuda_second));

    // Pure classical publication requests the three reciprocal force fields
    // without the scalar potential needed only by QM embedding.  It must
    // match the combined-output oracle while executing no fourth inverse FFT.
    std::vector<double> pure_pair_forces(3 * 2 * kAtoms, 0.0);
    std::vector<double> pure_mm_forces(3 * 2 * kAtoms, 0.0);
    std::vector<double> pure_lj(2, 0.0), pure_coulomb(2, 0.0),
        pure_pair_virial(12, 0.0), pure_mm_energy(2, 0.0),
        pure_mm_virial(12, 0.0);
    DPRC::ClassicalBatchInput pure_input{2, positions.data(), mm.data()};
    DPRC::ClassicalMmBatchOutput pure_output{
        2, pure_pair_forces.data(), pure_lj.data(), pure_coulomb.data(),
        pure_pair_virial.data(), pure_mm_energy.data(), pure_mm_virial.data(),
        nullptr, pure_mm_forces.data(), /*retain_for_qm=*/false};
    DPRC::ClassicalPlanOptions pure_cpu_options = options;
    pure_cpu_options.max_batch_count = 2;
    auto pure_cpu =
        DPRC::create_classical_batch_plan(make_topology(), pure_cpu_options);
    pure_cpu->begin_mm(pure_input, pure_output);
    bool terminal_rejected_qm = false;
    try {
      DPRC::ClassicalQmBatchInput terminal_qm_input{2, qm.data()};
      std::vector<double> terminal_qm_forces(3 * 2 * kAtoms, 0.0);
      std::vector<double> terminal_qm_energy(2, 0.0);
      std::vector<double> terminal_qm_virial(12, 0.0);
      DPRC::ClassicalQmBatchOutput terminal_qm_output{
          2, terminal_qm_forces.data(), terminal_qm_forces.data(),
          terminal_qm_energy.data(), terminal_qm_energy.data(),
          terminal_qm_virial.data(), terminal_qm_virial.data()};
      pure_cpu->finish_qm(terminal_qm_input, terminal_qm_output);
    } catch (const std::logic_error &) {
      terminal_rejected_qm = true;
    }
    CHECK(terminal_rejected_qm);
    CHECK(vectors_near(batch.mm_forces, pure_mm_forces));

#if defined(DPRC_TEST_CUFFT_WRAP)
    cufft_trace.reset();
    cufft_trace.enabled = true;
#endif
    DPRC::ClassicalPlanOptions pure_cuda_options = cuda_options;
    pure_cuda_options.max_batch_count = 2;
    auto pure_cuda =
        DPRC::create_classical_batch_plan(make_topology(), pure_cuda_options);
    std::fill(pure_mm_forces.begin(), pure_mm_forces.end(), 0.0);
    pure_cuda->begin_mm(pure_input, pure_output);
    terminal_rejected_qm = false;
    try {
      DPRC::ClassicalQmBatchInput terminal_qm_input{2, qm.data()};
      std::vector<double> terminal_qm_forces(3 * 2 * kAtoms, 0.0);
      std::vector<double> terminal_qm_energy(2, 0.0);
      std::vector<double> terminal_qm_virial(12, 0.0);
      DPRC::ClassicalQmBatchOutput terminal_qm_output{
          2, terminal_qm_forces.data(), terminal_qm_forces.data(),
          terminal_qm_energy.data(), terminal_qm_energy.data(),
          terminal_qm_virial.data(), terminal_qm_virial.data()};
      pure_cuda->finish_qm(terminal_qm_input, terminal_qm_output);
    } catch (const std::logic_error &) {
      terminal_rejected_qm = true;
    }
    CHECK(terminal_rejected_qm);
#if defined(DPRC_TEST_CUFFT_WRAP)
    cufft_trace.enabled = false;
    CHECK(cufft_trace.planned_batches == (std::vector<int>{2, 8, 6}));
    CHECK(cufft_trace.executions ==
          (std::vector<std::pair<int, int>>{
              {2, CUFFT_FORWARD}, {6, CUFFT_INVERSE}}));
#endif
    CHECK(vectors_near(batch.mm_forces, pure_mm_forces));

    // The production capacity is 48 synchronized windows.  Exercise every
    // required power-of-two coordinate plus 48 itself against independently
    // computed one-frame CPU results.  Alternating geometries make frame
    // cross-talk visible while preserving a compact oracle.
    DPRC::ClassicalPlanOptions scaling_options;
    scaling_options.backend = DPRC::ClassicalBackend::CUDA;
    scaling_options.max_batch_count = 48;
    auto cuda_scaling =
        DPRC::create_classical_batch_plan(make_topology(), scaling_options);
    for (const std::size_t scaling_batch :
         std::array<std::size_t, 7>{1, 2, 4, 8, 16, 32, 48}) {
      std::vector<double> scaling_positions;
      std::vector<double> scaling_mm;
      std::vector<double> scaling_qm;
      scaling_positions.reserve(3 * scaling_batch * kAtoms);
      scaling_mm.reserve(scaling_batch * kAtoms);
      scaling_qm.reserve(scaling_batch * kAtoms);
      for (std::size_t frame = 0; frame < scaling_batch; ++frame) {
        const std::vector<double> &geometry = frame % 2 == 0 ? frame0 : frame1;
        scaling_positions.insert(scaling_positions.end(), geometry.begin(),
                                 geometry.end());
        scaling_mm.insert(scaling_mm.end(), mm_one.begin(), mm_one.end());
        scaling_qm.insert(scaling_qm.end(), qm_one.begin(), qm_one.end());
      }
      const StagedResult scaling = run(*cuda_scaling, scaling_batch,
                                       scaling_positions, scaling_mm, scaling_qm);
      for (std::size_t frame = 0; frame < scaling_batch; ++frame) {
        CHECK(frame_near(scaling, frame, frame % 2 == 0 ? first : second));
        // Publication gates compare one-window and batched global energies.
        // Require exact pair scalar publication here so CUDA block scheduling
        // cannot create a tolerance-dependent pass or fail.
        CHECK(pair_scalars_equal(
            scaling, frame, frame % 2 == 0 ? cuda_first : cuda_second));
      }
    }

    // Reuse the cell list below skin/2, then force a rebuild above skin/2.
    // Both paths must remain identical to the all-pairs CPU oracle.
    std::vector<double> small_move = frame0;
    small_move[9] += 0.10;
    const StagedResult small_move_cpu =
        run(*sequential_plan, 1, small_move, mm_one, qm_one);
    const StagedResult small_move_cuda =
        run(*cuda_sequential, 1, small_move, mm_one, qm_one);
    CHECK(results_near(small_move_cpu, small_move_cuda));
    std::vector<double> large_move = frame0;
    large_move[9] += 0.40;
    const StagedResult large_move_cpu =
        run(*sequential_plan, 1, large_move, mm_one, qm_one);
    const StagedResult large_move_cuda =
        run(*cuda_sequential, 1, large_move, mm_one, qm_one);
    CHECK(results_near(large_move_cpu, large_move_cuda));

    std::vector<double> cuda_sentinels(3 * kAtoms, 41.0);
    std::vector<double> cuda_bad_positions = frame0;
    cuda_bad_positions[2] = std::numeric_limits<double>::infinity();
    std::vector<double> cuda_energy_sentinel(1, 43.0);
    std::vector<double> cuda_virial_sentinel(6, 47.0);
    DPRC::ClassicalBatchInput cuda_bad_input{
        1, cuda_bad_positions.data(), mm_one.data()};
    DPRC::ClassicalMmBatchOutput cuda_bad_output{
        1, cuda_sentinels.data(), cuda_energy_sentinel.data(),
        cuda_energy_sentinel.data(), cuda_virial_sentinel.data(),
        cuda_energy_sentinel.data(), cuda_virial_sentinel.data(), nullptr};
    bool cuda_rejected = false;
    try {
      cuda_sequential->begin_mm(cuda_bad_input, cuda_bad_output);
    } catch (const std::invalid_argument &) {
      cuda_rejected = true;
    }
    CHECK(cuda_rejected);
    CHECK(std::all_of(cuda_sentinels.begin(), cuda_sentinels.end(),
                      [](double value) { return value == 41.0; }));
    CHECK(cuda_energy_sentinel[0] == 43.0);
    CHECK(std::all_of(cuda_virial_sentinel.begin(), cuda_virial_sentinel.end(),
                      [](double value) { return value == 47.0; }));

    DPRC::ClassicalPlanOptions table_cpu_options;
    table_cpu_options.backend = DPRC::ClassicalBackend::CPU;
    table_cpu_options.max_batch_count = 1;
    auto table_cpu = DPRC::create_classical_batch_plan(make_table_topology(),
                                                        table_cpu_options);
    DPRC::ClassicalPlanOptions table_cuda_options = table_cpu_options;
    table_cuda_options.backend = DPRC::ClassicalBackend::CUDA;
    auto table_cuda = DPRC::create_classical_batch_plan(make_table_topology(),
                                                         table_cuda_options);
    const StagedResult table_reference =
        run(*table_cpu, 1, frame0, mm_one, qm_one);
    const StagedResult table_gpu =
        run(*table_cuda, 1, frame0, mm_one, qm_one);
    CHECK(vectors_near(table_reference.pair_forces, table_gpu.pair_forces));
    CHECK(vectors_near(table_reference.coulomb, table_gpu.coulomb));
    CHECK(vectors_near(table_reference.pair_virial, table_gpu.pair_virial));
#endif

    // Validation must be transactional and a stray finish must not create an
    // implicit epoch.
    std::vector<double> sentinels(3 * kAtoms, 17.0);
    std::vector<double> bad_positions = frame0;
    bad_positions[0] = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> energy_sentinel(1, 23.0);
    std::vector<double> virial_sentinel(6, 29.0);
    DPRC::ClassicalBatchInput bad_input{1, bad_positions.data(), mm_one.data()};
    DPRC::ClassicalMmBatchOutput bad_output{
        1, sentinels.data(), energy_sentinel.data(), energy_sentinel.data(),
        virial_sentinel.data(), energy_sentinel.data(), virial_sentinel.data(), nullptr};
    bool rejected = false;
    try {
      plan->begin_mm(bad_input, bad_output);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    CHECK(rejected);
    CHECK(std::all_of(sentinels.begin(), sentinels.end(),
                      [](double value) { return value == 17.0; }));

    rejected = false;
    DPRC::ClassicalQmBatchInput orphan_input{1, qm_one.data()};
    DPRC::ClassicalQmBatchOutput orphan_output{
        1, sentinels.data(), sentinels.data(), energy_sentinel.data(),
        energy_sentinel.data(), virial_sentinel.data(), virial_sentinel.data()};
    try {
      plan->finish_qm(orphan_input, orphan_output);
    } catch (const std::logic_error &) {
      rejected = true;
    }
    CHECK(rejected);

    std::cout << "classical batch: staged MM/QM PPPM, triclinic TIP4P, real-space "
                 "LJ/Coulomb, batch parity, and finite difference passed\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
