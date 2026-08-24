#include "classical_batch.h"
#include "classical_batch_internal.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace DPRC {

#if defined(DPRC_HAVE_CLASSICAL_CUDA)
[[nodiscard]] std::unique_ptr<ClassicalBatchPlan>
create_cuda_classical_batch_plan(const ClassicalTopology &, const ClassicalPlanOptions &);
#endif

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kHalfPi = 0.5 * kPi;
constexpr double kInvSqrtPi = 0.564189583547756286948079451560772586;
constexpr double kEwaldF = 1.12837917;
constexpr double kEwaldP = 0.3275911;
constexpr double kEwaldA1 = 0.254829592;
constexpr double kEwaldA2 = -0.284496736;
constexpr double kEwaldA3 = 1.421413741;
constexpr double kEwaldA4 = -1.453152027;
constexpr double kEwaldA5 = 1.061405429;
constexpr double kEpsHoc = 1.0e-7;
constexpr int kGridOffset = 16384;

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;
using Complex = std::complex<double>;

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          const char *what) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(what) + " size overflows");
  return left * right;
}

[[nodiscard]] double determinant(const Mat3 &h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) -
      h[1] * (h[3] * h[8] - h[5] * h[6]) +
      h[2] * (h[3] * h[7] - h[4] * h[6]);
}

[[nodiscard]] Mat3 inverse(const Mat3 &h) {
  const double det = determinant(h);
  if (!std::isfinite(det) || det <= 0.0)
    throw std::invalid_argument("classical cell must have positive finite volume");
  const double invdet = 1.0 / det;
  return {(h[4] * h[8] - h[5] * h[7]) * invdet,
          (h[2] * h[7] - h[1] * h[8]) * invdet,
          (h[1] * h[5] - h[2] * h[4]) * invdet,
          (h[5] * h[6] - h[3] * h[8]) * invdet,
          (h[0] * h[8] - h[2] * h[6]) * invdet,
          (h[2] * h[3] - h[0] * h[5]) * invdet,
          (h[3] * h[7] - h[4] * h[6]) * invdet,
          (h[1] * h[6] - h[0] * h[7]) * invdet,
          (h[0] * h[4] - h[1] * h[3]) * invdet};
}

[[nodiscard]] Vec3 multiply(const Mat3 &matrix, const Vec3 &vector) {
  return {matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
          matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
          matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]};
}

[[nodiscard]] Vec3 multiply_transpose(const Mat3 &matrix, const Vec3 &vector) {
  return {matrix[0] * vector[0] + matrix[3] * vector[1] + matrix[6] * vector[2],
          matrix[1] * vector[0] + matrix[4] * vector[1] + matrix[7] * vector[2],
          matrix[2] * vector[0] + matrix[5] * vector[1] + matrix[8] * vector[2]};
}

[[nodiscard]] Vec3 wrap_fractional(Vec3 value) {
  for (double &component : value) component -= std::floor(component);
  return value;
}

[[nodiscard]] Vec3 minimum_fractional(Vec3 value) {
  for (double &component : value) component -= std::nearbyint(component);
  return value;
}

[[nodiscard]] int periodic_index(int value, int extent) {
  value %= extent;
  return value < 0 ? value + extent : value;
}

[[nodiscard]] std::size_t mesh_index(int x, int y, int z, int nx, int ny) {
  return (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(y)) *
      static_cast<std::size_t>(nx) + static_cast<std::size_t>(x);
}

[[nodiscard]] double powsinxx(double value, int exponent) {
  const double ratio = value == 0.0 ? 1.0 : std::sin(value) / value;
  return std::pow(ratio, exponent);
}

struct SpecialScale {
  std::uint64_t key = 0;
  double lj = 1.0;
  double coulomb = 1.0;
};

[[nodiscard]] std::uint64_t pair_key(std::int32_t atom1, std::int32_t atom2) {
  const std::uint32_t low = static_cast<std::uint32_t>(std::min(atom1, atom2));
  const std::uint32_t high = static_cast<std::uint32_t>(std::max(atom1, atom2));
  return (static_cast<std::uint64_t>(low) << 32u) | high;
}

[[nodiscard]] SpecialScale find_scale(const std::vector<SpecialScale> &scales,
                                      std::int32_t atom1, std::int32_t atom2) {
  const std::uint64_t key = pair_key(atom1, atom2);
  const auto iter = std::lower_bound(scales.begin(), scales.end(), key,
                                     [](const SpecialScale &entry, std::uint64_t wanted) {
                                       return entry.key < wanted;
                                     });
  return iter != scales.end() && iter->key == key ? *iter : SpecialScale{key, 1.0, 1.0};
}

struct SplineData {
  int order = 0;
  int lower = 0;
  int upper = 0;
  double shift = 0.0;
  double shift_one = 0.0;
  std::vector<double> coefficients;

  [[nodiscard]] double coefficient(int power, int stencil) const {
    return coefficients[static_cast<std::size_t>(power) * order +
                        static_cast<std::size_t>(stencil - lower)];
  }
};

[[nodiscard]] SplineData make_spline(int order) {
  if (order < 2 || order > 8)
    throw std::invalid_argument("PPPM interpolation order must be in [2, 8]");

  const int stride = 2 * order + 1;
  std::vector<double> a(static_cast<std::size_t>(order) * stride, 0.0);
  const auto at = [&](int power, int k) -> double & {
    return a[static_cast<std::size_t>(power) * stride +
             static_cast<std::size_t>(k + order)];
  };
  at(0, 0) = 1.0;
  for (int j = 1; j < order; ++j) {
    for (int k = -j; k <= j; k += 2) {
      double sum = 0.0;
      for (int power = 0; power < j; ++power) {
        at(power + 1, k) =
            (at(power, k + 1) - at(power, k - 1)) / (power + 1.0);
        sum += std::pow(0.5, power + 1.0) *
            (at(power, k - 1) + std::pow(-1.0, power) * at(power, k + 1)) /
            (power + 1.0);
      }
      at(0, k) = sum;
    }
  }

  SplineData spline;
  spline.order = order;
  spline.lower = -(order - 1) / 2;
  spline.upper = order / 2;
  spline.shift = order % 2 ? kGridOffset + 0.5 : kGridOffset;
  spline.shift_one = order % 2 ? 0.0 : 0.5;
  spline.coefficients.assign(static_cast<std::size_t>(order) * order, 0.0);
  int packed = spline.lower;
  for (int k = -(order - 1); k < order; k += 2, ++packed)
    for (int power = 0; power < order; ++power)
      spline.coefficients[static_cast<std::size_t>(power) * order +
                          static_cast<std::size_t>(packed - spline.lower)] = at(power, k);
  return spline;
}

void evaluate_spline(const SplineData &spline, double delta, double *weights) {
  for (int stencil = spline.lower; stencil <= spline.upper; ++stencil) {
    double value = 0.0;
    for (int power = spline.order - 1; power >= 0; --power)
      value = spline.coefficient(power, stencil) + value * delta;
    weights[stencil - spline.lower] = value;
  }
}

struct MeshData {
  std::array<int, 3> mesh{};
  std::size_t count = 0;
  double volume = 0.0;
  double delvolinv = 0.0;
  std::vector<double> green;
  std::vector<Vec3> kvector;
  std::vector<std::array<double, 6>> virial_factor;
};

[[nodiscard]] std::vector<double> make_gf_coefficients(int order) {
  std::vector<double> result(static_cast<std::size_t>(order), 0.0);
  result[0] = 1.0;
  for (int m = 1; m < order; ++m) {
    int l = 0;
    for (l = m; l > 0; --l)
      result[static_cast<std::size_t>(l)] =
          4.0 * (result[static_cast<std::size_t>(l)] * (l - m) * (l - m - 0.5) -
                 result[static_cast<std::size_t>(l - 1)] * (l - m - 1) *
                     (l - m - 1));
    result[0] = 4.0 * result[0] * (l - m) * (l - m - 0.5);
  }
  std::uint64_t factorial = 1;
  for (int value = 1; value < 2 * order; ++value) factorial *= value;
  for (double &value : result) value /= static_cast<double>(factorial);
  return result;
}

[[nodiscard]] double gf_denominator(const std::vector<double> &coefficients,
                                    double x, double y, double z) {
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (auto iter = coefficients.rbegin(); iter != coefficients.rend(); ++iter) {
    sx = *iter + sx * x;
    sy = *iter + sy * y;
    sz = *iter + sz * z;
  }
  const double product = sx * sy * sz;
  return product * product;
}

[[nodiscard]] MeshData make_mesh(const ClassicalTopology &topology, const Mat3 &hinv) {
  MeshData data;
  for (int dim = 0; dim < 3; ++dim) {
    if (topology.pppm.mesh[dim] <= 1)
      throw std::invalid_argument("every PPPM mesh extent must exceed one");
    data.mesh[dim] = topology.pppm.mesh[dim];
  }
  data.count = checked_product(
      checked_product(static_cast<std::size_t>(data.mesh[0]),
                      static_cast<std::size_t>(data.mesh[1]), "PPPM mesh"),
      static_cast<std::size_t>(data.mesh[2]), "PPPM mesh");
  data.volume = determinant(topology.cell.h);
  data.delvolinv = static_cast<double>(data.count) / data.volume;
  data.green.resize(data.count);
  data.kvector.resize(data.count);
  data.virial_factor.resize(data.count);

  const std::vector<double> gf = make_gf_coefficients(topology.pppm.order);
  const int nx = data.mesh[0];
  const int ny = data.mesh[1];
  const int nz = data.mesh[2];
  const double alias_scale = std::pow(-std::log(kEpsHoc), 0.25);

  // x2lamdaT in LAMMPS applies H^-T.  Multiplying H^-T by a vector is
  // equivalent to multiplying the row-major inverse H^-1 on the left here.
  Vec3 alias_estimate = {topology.pppm.g_ewald / (kPi * nx) * alias_scale,
                         topology.pppm.g_ewald / (kPi * ny) * alias_scale,
                         topology.pppm.g_ewald / (kPi * nz) * alias_scale};
  alias_estimate = multiply_transpose(topology.cell.h, alias_estimate);
  const int nbx = static_cast<int>(alias_estimate[0]);
  const int nby = static_cast<int>(alias_estimate[1]);
  const int nbz = static_cast<int>(alias_estimate[2]);
  const int twoorder = 2 * topology.pppm.order;

  for (int z = 0; z < nz; ++z) {
    const int zper = z - nz * (2 * z / nz);
    const double snz = std::pow(std::sin(kPi * zper / nz), 2.0);
    for (int y = 0; y < ny; ++y) {
      const int yper = y - ny * (2 * y / ny);
      const double sny = std::pow(std::sin(kPi * yper / ny), 2.0);
      for (int x = 0; x < nx; ++x) {
        const int xper = x - nx * (2 * x / nx);
        const double snx = std::pow(std::sin(kPi * xper / nx), 2.0);
        const std::size_t index = mesh_index(x, y, z, nx, ny);
        const Vec3 integer_k = {kTwoPi * xper, kTwoPi * yper, kTwoPi * zper};
        const Vec3 kvector = multiply_transpose(hinv, integer_k);
        data.kvector[index] = kvector;
        const double squared_k = kvector[0] * kvector[0] + kvector[1] * kvector[1] +
            kvector[2] * kvector[2];
        if (squared_k == 0.0) {
          data.green[index] = 0.0;
          data.virial_factor[index].fill(0.0);
          continue;
        }

        double alias_sum = 0.0;
        for (int ax = -nbx; ax <= nbx; ++ax) {
          const double wx = powsinxx(kPi * xper / nx + kPi * ax, twoorder);
          for (int ay = -nby; ay <= nby; ++ay) {
            const double wy = powsinxx(kPi * yper / ny + kPi * ay, twoorder);
            for (int az = -nbz; az <= nbz; ++az) {
              const double wz = powsinxx(kPi * zper / nz + kPi * az, twoorder);
              const Vec3 integer_alias = {kTwoPi * nx * ax, kTwoPi * ny * ay,
                                          kTwoPi * nz * az};
              const Vec3 alias = multiply_transpose(hinv, integer_alias);
              const Vec3 q = {kvector[0] + alias[0], kvector[1] + alias[1],
                              kvector[2] + alias[2]};
              const double q_squared = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
              const double dot = kvector[0] * q[0] + kvector[1] * q[1] +
                  kvector[2] * q[2];
              const double gaussian =
                  std::exp(-0.25 * q_squared /
                           (topology.pppm.g_ewald * topology.pppm.g_ewald));
              alias_sum += dot / q_squared * gaussian * wx * wy * wz;
            }
          }
        }
        data.green[index] = 12.5663706 / squared_k * alias_sum /
            gf_denominator(gf, snx, sny, snz);
        const double vterm =
            -2.0 * (1.0 / squared_k +
                    0.25 / (topology.pppm.g_ewald * topology.pppm.g_ewald));
        data.virial_factor[index] = {
            1.0 + vterm * kvector[0] * kvector[0],
            1.0 + vterm * kvector[1] * kvector[1],
            1.0 + vterm * kvector[2] * kvector[2],
            vterm * kvector[0] * kvector[1], vterm * kvector[0] * kvector[2],
            vterm * kvector[1] * kvector[2]};
      }
    }
  }
  return data;
}

void dft_line(const Complex *input, Complex *output, int count, bool inverse) {
  const double sign = inverse ? 1.0 : -1.0;
  for (int k = 0; k < count; ++k) {
    Complex sum = 0.0;
    for (int n = 0; n < count; ++n) {
      const double phase = sign * kTwoPi * k * n / count;
      sum += input[n] * Complex(std::cos(phase), std::sin(phase));
    }
    output[k] = sum;
  }
}

void dft3d(std::vector<Complex> &values, const std::array<int, 3> &mesh,
           bool inverse_transform) {
  const int nx = mesh[0];
  const int ny = mesh[1];
  const int nz = mesh[2];
  const int maximum = std::max({nx, ny, nz});
  std::vector<Complex> input(static_cast<std::size_t>(maximum));
  std::vector<Complex> output(static_cast<std::size_t>(maximum));

  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) input[static_cast<std::size_t>(x)] = values[mesh_index(x, y, z, nx, ny)];
      dft_line(input.data(), output.data(), nx, inverse_transform);
      for (int x = 0; x < nx; ++x) values[mesh_index(x, y, z, nx, ny)] = output[static_cast<std::size_t>(x)];
    }
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      for (int y = 0; y < ny; ++y) input[static_cast<std::size_t>(y)] = values[mesh_index(x, y, z, nx, ny)];
      dft_line(input.data(), output.data(), ny, inverse_transform);
      for (int y = 0; y < ny; ++y) values[mesh_index(x, y, z, nx, ny)] = output[static_cast<std::size_t>(y)];
    }
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      for (int z = 0; z < nz; ++z) input[static_cast<std::size_t>(z)] = values[mesh_index(x, y, z, nx, ny)];
      dft_line(input.data(), output.data(), nz, inverse_transform);
      for (int z = 0; z < nz; ++z) values[mesh_index(x, y, z, nx, ny)] = output[static_cast<std::size_t>(z)];
    }
}

struct PppmSolve {
  std::vector<Complex> normalized_spectrum;
  std::vector<double> scalar_grid;
  std::array<std::vector<double>, 3> gradient_grid;
  double energy = 0.0;
  double qsum = 0.0;
  std::array<double, 6> virial{};
};

struct FrameSites {
  std::vector<Vec3> atoms;
  std::vector<Vec3> charges;
};

class CpuClassicalBatchPlan final : public ClassicalBatchPlan {
 public:
  CpuClassicalBatchPlan(ClassicalTopology topology, std::size_t max_batch_count)
      : topology_(std::move(topology)), max_batch_count_(max_batch_count),
        hinv_(inverse(topology_.cell.h)), spline_(make_spline(topology_.pppm.order)),
        mesh_(make_mesh(topology_, hinv_)) {
    validate_topology();
  }

  [[nodiscard]] ClassicalBackend backend() const noexcept override {
    return ClassicalBackend::CPU;
  }

  [[nodiscard]] std::size_t max_batch_count() const noexcept override {
    return max_batch_count_;
  }

  void begin_mm(const ClassicalBatchInput &input,
                const ClassicalMmBatchOutput &output) override {
    validate_begin(input, output);
    if (active_) throw std::logic_error("a classical MM epoch is already active");

    const std::size_t coordinate_count = checked_product(
        checked_product(input.batch_count, topology_.atom_count, "batch atoms"), 3,
        "batch coordinates");
    std::vector<double> pair_forces(coordinate_count, 0.0);
    std::vector<double> lj_energy(input.batch_count, 0.0);
    std::vector<double> coulomb_energy(input.batch_count, 0.0);
    std::vector<double> pair_virial(checked_product(input.batch_count, 6, "pair virial"),
                                    0.0);
    std::vector<double> mm_energy(input.batch_count, 0.0);
    std::vector<double> mm_virial(checked_product(input.batch_count, 6, "MM virial"), 0.0);
    std::vector<double> mm_potential;
    if (output.mm_pppm_potential)
      mm_potential.assign(
          checked_product(input.batch_count, topology_.atom_count, "MM potential"),
          0.0);
    std::vector<double> mm_forces;
    if (output.mm_pppm_forces)
      mm_forces.assign(coordinate_count, 0.0);
    std::vector<FrameSites> sites;
    std::vector<PppmSolve> solves;
    sites.reserve(input.batch_count);
    solves.reserve(input.batch_count);

    for (std::size_t frame = 0; frame < input.batch_count; ++frame) {
      const double *positions = input.positions + 3 * frame * topology_.atom_count;
      const double *charges = input.charges + frame * topology_.atom_count;
      sites.push_back(make_fractional_sites(positions));
      double *frame_forces = pair_forces.data() + 3 * frame * topology_.atom_count;
      compute_real_space(sites.back().atoms, sites.back().charges, charges, frame_forces,
                         lj_energy[frame], coulomb_energy[frame],
                         pair_virial.data() + 6 * frame);
      solves.push_back(solve_pppm(sites.back().charges, charges,
                                  output.mm_pppm_potential != nullptr));
      mm_energy[frame] = solves.back().energy;
      std::copy(solves.back().virial.begin(), solves.back().virial.end(),
                mm_virial.begin() + static_cast<std::ptrdiff_t>(6 * frame));
      interpolate_pppm(
          sites.back().charges, solves.back(), charges,
          output.mm_pppm_forces
              ? mm_forces.data() + 3 * frame * topology_.atom_count
              : nullptr,
          output.mm_pppm_potential
              ? mm_potential.data() + frame * topology_.atom_count
              : nullptr);
    }

    std::copy(pair_forces.begin(), pair_forces.end(), output.pair_forces);
    std::copy(lj_energy.begin(), lj_energy.end(), output.lj_energy);
    std::copy(coulomb_energy.begin(), coulomb_energy.end(), output.coulomb_energy);
    std::copy(pair_virial.begin(), pair_virial.end(), output.pair_virial);
    std::copy(mm_energy.begin(), mm_energy.end(), output.mm_pppm_energy);
    std::copy(mm_virial.begin(), mm_virial.end(), output.mm_pppm_virial);
    if (output.mm_pppm_potential)
      std::copy(mm_potential.begin(), mm_potential.end(), output.mm_pppm_potential);
    if (output.mm_pppm_forces)
      std::copy(mm_forces.begin(), mm_forces.end(), output.mm_pppm_forces);

    if (output.retain_for_qm) {
      cached_batch_count_ = input.batch_count;
      cached_sites_ = std::move(sites);
      mm_solve_ = std::move(solves);
      cached_mm_charges_.assign(
          input.charges,
          input.charges + input.batch_count * topology_.atom_count);
      active_ = true;
    } else {
      // A terminal MM publication is a successful completed epoch.  Do not
      // retain large CPU spectra or accidentally permit a later QM stage.
      cancel();
    }
  }

  void finish_qm(const ClassicalQmBatchInput &input,
                 const ClassicalQmBatchOutput &output) override {
    validate_finish(input, output);
    if (!active_) throw std::logic_error("no classical MM epoch is active");
    if (input.batch_count != cached_batch_count_)
      throw std::invalid_argument("QM batch count does not match the active MM epoch");

    const std::size_t coordinate_count = checked_product(
        checked_product(input.batch_count, topology_.atom_count, "batch atoms"), 3,
        "batch coordinates");
    std::vector<double> qm_forces(coordinate_count, 0.0);
    std::vector<double> full_forces(coordinate_count, 0.0);
    std::vector<double> qm_energy(input.batch_count, 0.0);
    std::vector<double> full_energy(input.batch_count, 0.0);
    std::vector<double> qm_virial(checked_product(input.batch_count, 6, "QM virial"), 0.0);
    std::vector<double> full_virial(checked_product(input.batch_count, 6, "full virial"),
                                    0.0);
    std::vector<double> full_charges(topology_.atom_count);

    for (std::size_t frame = 0; frame < input.batch_count; ++frame) {
      const double *qm_charges = input.qm_charges + frame * topology_.atom_count;
      const double *mm_charges =
          cached_mm_charges_.data() + frame * topology_.atom_count;
      PppmSolve qm = solve_pppm(cached_sites_[frame].charges, qm_charges, false);
      qm_energy[frame] = qm.energy;
      std::copy(qm.virial.begin(), qm.virial.end(),
                qm_virial.begin() + static_cast<std::ptrdiff_t>(6 * frame));
      interpolate_pppm(cached_sites_[frame].charges, qm, qm_charges,
                       qm_forces.data() + 3 * frame * topology_.atom_count, nullptr);

      PppmSolve full = combine_solves(mm_solve_[frame], qm);
      for (std::size_t atom = 0; atom < topology_.atom_count; ++atom)
        full_charges[atom] = mm_charges[atom] + qm_charges[atom];
      interpolate_pppm(cached_sites_[frame].charges, full, full_charges.data(),
                       full_forces.data() + 3 * frame * topology_.atom_count, nullptr);
      full_energy[frame] = full.energy;
      std::copy(full.virial.begin(), full.virial.end(),
                full_virial.begin() + static_cast<std::ptrdiff_t>(6 * frame));
    }

    std::copy(qm_forces.begin(), qm_forces.end(), output.qm_pppm_forces);
    std::copy(full_forces.begin(), full_forces.end(), output.full_pppm_forces);
    std::copy(qm_energy.begin(), qm_energy.end(), output.qm_pppm_energy);
    std::copy(full_energy.begin(), full_energy.end(), output.full_pppm_energy);
    std::copy(qm_virial.begin(), qm_virial.end(), output.qm_pppm_virial);
    std::copy(full_virial.begin(), full_virial.end(), output.full_pppm_virial);
    cancel();
  }

  void cancel() noexcept override {
    active_ = false;
    cached_batch_count_ = 0;
    cached_sites_.clear();
    mm_solve_.clear();
    cached_mm_charges_.clear();
  }

 private:
  void validate_topology() {
    if (max_batch_count_ == 0)
      throw std::invalid_argument("classical max batch count must be positive");
    if (topology_.atom_count == 0)
      throw std::invalid_argument("classical topology must contain atoms");
    if (topology_.type_count <= 0)
      throw std::invalid_argument("classical topology must contain atom types");
    if (topology_.atom_types.size() != topology_.atom_count)
      throw std::invalid_argument("atom type extent does not match atom count");
    for (std::int32_t type : topology_.atom_types)
      if (type < 0 || type >= topology_.type_count)
        throw std::invalid_argument("atom type is outside the zero-based type range");
    const std::size_t type_pairs = checked_product(
        static_cast<std::size_t>(topology_.type_count),
        static_cast<std::size_t>(topology_.type_count), "type-pair matrix");
    if (topology_.lj.size() != type_pairs ||
        topology_.coulomb_type_pairs.size() != type_pairs)
      throw std::invalid_argument("type-pair parameter extents are invalid");
    if (!std::isfinite(topology_.qqrd2e) || topology_.qqrd2e <= 0.0 ||
        !std::isfinite(topology_.pppm.g_ewald) || topology_.pppm.g_ewald <= 0.0 ||
        !std::isfinite(topology_.real_space_cutoff) ||
        topology_.real_space_cutoff <= 0.0 || !std::isfinite(topology_.tip4p_alpha) ||
        topology_.tip4p_alpha < 0.0)
      throw std::invalid_argument("classical scalar parameters are invalid");
    oxygen_site_.assign(topology_.atom_count, -1);
    for (std::size_t site = 0; site < topology_.tip4p_sites.size(); ++site) {
      const Tip4pSite &entry = topology_.tip4p_sites[site];
      if (entry.oxygen < 0 || entry.hydrogen1 < 0 || entry.hydrogen2 < 0 ||
          static_cast<std::size_t>(entry.oxygen) >= topology_.atom_count ||
          static_cast<std::size_t>(entry.hydrogen1) >= topology_.atom_count ||
          static_cast<std::size_t>(entry.hydrogen2) >= topology_.atom_count ||
          entry.oxygen == entry.hydrogen1 || entry.oxygen == entry.hydrogen2 ||
          entry.hydrogen1 == entry.hydrogen2)
        throw std::invalid_argument("invalid TIP4P parent topology");
      if (oxygen_site_[static_cast<std::size_t>(entry.oxygen)] >= 0)
        throw std::invalid_argument("one oxygen appears in multiple TIP4P sites");
      oxygen_site_[static_cast<std::size_t>(entry.oxygen)] = static_cast<int>(site);
    }
    scales_.reserve(topology_.special_pairs.size());
    for (const SpecialPair &entry : topology_.special_pairs) {
      if (entry.atom1 < 0 || entry.atom2 < 0 || entry.atom1 == entry.atom2 ||
          static_cast<std::size_t>(entry.atom1) >= topology_.atom_count ||
          static_cast<std::size_t>(entry.atom2) >= topology_.atom_count ||
          !std::isfinite(entry.lj_scale) || !std::isfinite(entry.coulomb_scale))
        throw std::invalid_argument("invalid special-pair topology");
      scales_.push_back({pair_key(entry.atom1, entry.atom2), entry.lj_scale,
                         entry.coulomb_scale});
    }
    std::sort(scales_.begin(), scales_.end(),
              [](const SpecialScale &left, const SpecialScale &right) {
                return left.key < right.key;
              });
    if (std::adjacent_find(scales_.begin(), scales_.end(),
                           [](const SpecialScale &left, const SpecialScale &right) {
                             return left.key == right.key;
                           }) != scales_.end())
      throw std::invalid_argument("duplicate special-pair topology");
    validate_coulomb_table();
  }

  void validate_coulomb_table() const {
    const CoulombLookupTable &table = topology_.coulomb_table;
    if (table.bits == 0) {
      if (!table.r.empty() || !table.dr.empty() || !table.force.empty() ||
          !table.dforce.empty() || !table.coulomb.empty() || !table.dcoulomb.empty() ||
          !table.energy.empty() || !table.denergy.empty())
        throw std::invalid_argument("disabled Coulomb table must not carry arrays");
      return;
    }
    if (table.bits < 1 || table.bits > 20)
      throw std::invalid_argument("Coulomb table bit count is invalid");
    const std::size_t count = std::size_t{1} << table.bits;
    if (table.r.size() != count || table.dr.size() != count ||
        table.force.size() != count || table.dforce.size() != count ||
        table.coulomb.size() != count || table.dcoulomb.size() != count ||
        table.energy.size() != count || table.denergy.size() != count)
      throw std::invalid_argument("Coulomb table array extent is invalid");
  }

  void validate_input(const ClassicalBatchInput &input) const {
    if (input.batch_count == 0 || input.batch_count > max_batch_count_)
      throw std::invalid_argument("classical batch count is outside plan capacity");
    if (!input.positions || !input.charges)
      throw std::invalid_argument("classical input pointers must be non-null");
    const std::size_t coordinates = checked_product(
        checked_product(input.batch_count, topology_.atom_count, "batch atoms"), 3,
        "batch coordinates");
    for (std::size_t index = 0; index < coordinates; ++index)
      if (!std::isfinite(input.positions[index]))
        throw std::invalid_argument("classical positions must be finite");
    const std::size_t charges = checked_product(input.batch_count, topology_.atom_count,
                                                "batch charges");
    for (std::size_t index = 0; index < charges; ++index)
      if (!std::isfinite(input.charges[index]))
        throw std::invalid_argument("classical charges must be finite");
  }

  void validate_begin(const ClassicalBatchInput &input,
                      const ClassicalMmBatchOutput &output) const {
    validate_input(input);
    if (output.batch_count != input.batch_count || !output.pair_forces ||
        !output.lj_energy || !output.coulomb_energy || !output.pair_virial ||
        !output.mm_pppm_energy || !output.mm_pppm_virial)
      throw std::invalid_argument("classical MM output descriptor is incomplete");
    if (output.retain_for_qm && !output.mm_pppm_potential)
      throw std::invalid_argument(
          "classical QM continuation requires the MM scalar potential");
  }

  void validate_finish(const ClassicalQmBatchInput &input,
                       const ClassicalQmBatchOutput &output) const {
    if (input.batch_count == 0 || input.batch_count > max_batch_count_ ||
        !input.qm_charges)
      throw std::invalid_argument("classical QM input descriptor is invalid");
    if (output.batch_count != input.batch_count || !output.qm_pppm_forces ||
        !output.full_pppm_forces || !output.qm_pppm_energy ||
        !output.full_pppm_energy || !output.qm_pppm_virial ||
        !output.full_pppm_virial)
      throw std::invalid_argument("classical QM output descriptor is incomplete");
    const std::size_t count =
        checked_product(input.batch_count, topology_.atom_count, "QM charges");
    for (std::size_t index = 0; index < count; ++index)
      if (!std::isfinite(input.qm_charges[index]))
        throw std::invalid_argument("classical QM charges must be finite");
  }

  [[nodiscard]] FrameSites make_fractional_sites(const double *positions) const {
    FrameSites result;
    result.atoms.resize(topology_.atom_count);
    result.charges.resize(topology_.atom_count);
    for (std::size_t atom = 0; atom < topology_.atom_count; ++atom) {
      Vec3 shifted = {positions[3 * atom] - topology_.cell.boxlo[0],
                      positions[3 * atom + 1] - topology_.cell.boxlo[1],
                      positions[3 * atom + 2] - topology_.cell.boxlo[2]};
      result.atoms[atom] = wrap_fractional(multiply(hinv_, shifted));
      result.charges[atom] = result.atoms[atom];
    }
    for (const Tip4pSite &site : topology_.tip4p_sites) {
      const Vec3 oxygen = result.atoms[static_cast<std::size_t>(site.oxygen)];
      const Vec3 h1 = result.atoms[static_cast<std::size_t>(site.hydrogen1)];
      const Vec3 h2 = result.atoms[static_cast<std::size_t>(site.hydrogen2)];
      const Vec3 d1 = minimum_fractional(
          {h1[0] - oxygen[0], h1[1] - oxygen[1], h1[2] - oxygen[2]});
      const Vec3 d2 = minimum_fractional(
          {h2[0] - oxygen[0], h2[1] - oxygen[1], h2[2] - oxygen[2]});
      result.charges[static_cast<std::size_t>(site.oxygen)] = wrap_fractional(
          {oxygen[0] + 0.5 * topology_.tip4p_alpha * (d1[0] + d2[0]),
           oxygen[1] + 0.5 * topology_.tip4p_alpha * (d1[1] + d2[1]),
           oxygen[2] + 0.5 * topology_.tip4p_alpha * (d1[2] + d2[2])});
    }
    return result;
  }

  void add_site_force(std::size_t atom, const Vec3 &force, double *forces) const {
    const int site_index = oxygen_site_[atom];
    if (site_index < 0) {
      for (int dim = 0; dim < 3; ++dim) forces[3 * atom + dim] += force[dim];
      return;
    }
    const Tip4pSite &site = topology_.tip4p_sites[static_cast<std::size_t>(site_index)];
    for (int dim = 0; dim < 3; ++dim) {
      forces[3 * static_cast<std::size_t>(site.oxygen) + dim] +=
          (1.0 - topology_.tip4p_alpha) * force[dim];
      forces[3 * static_cast<std::size_t>(site.hydrogen1) + dim] +=
          0.5 * topology_.tip4p_alpha * force[dim];
      forces[3 * static_cast<std::size_t>(site.hydrogen2) + dim] +=
          0.5 * topology_.tip4p_alpha * force[dim];
    }
  }

  [[nodiscard]] bool use_coulomb_table(double squared_distance) const {
    return topology_.coulomb_table.bits > 0 &&
        squared_distance > topology_.coulomb_table.inner_squared;
  }

  void table_coulomb(double squared_distance, double charge_product, double scale,
                     double &force, double &energy) const {
    const CoulombLookupTable &table = topology_.coulomb_table;
    float squared_float = static_cast<float>(squared_distance);
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(squared_float));
    std::memcpy(&bits, &squared_float, sizeof(bits));
    std::size_t index = static_cast<std::size_t>(bits &
                                                 static_cast<std::uint32_t>(table.mask));
    index >>= table.shift_bits;
    const double fraction = (static_cast<double>(squared_float) - table.r[index]) *
        table.dr[index];
    force = charge_product * (table.force[index] + fraction * table.dforce[index]);
    energy = charge_product * (table.energy[index] + fraction * table.denergy[index]);
    if (scale < 1.0) {
      const double prefactor =
          charge_product * (table.coulomb[index] + fraction * table.dcoulomb[index]);
      force -= (1.0 - scale) * prefactor;
      energy -= (1.0 - scale) * prefactor;
    }
  }

  void compute_real_space(const std::vector<Vec3> &atoms,
                          const std::vector<Vec3> &sites, const double *charges,
                          double *forces, double &lj_energy, double &coulomb_energy,
                          double *virial) const {
    const int types = topology_.type_count;
    for (std::size_t atom1 = 0; atom1 < topology_.atom_count; ++atom1) {
      const int type1 = topology_.atom_types[atom1];
      for (std::size_t atom2 = atom1 + 1; atom2 < topology_.atom_count; ++atom2) {
        const int type2 = topology_.atom_types[atom2];
        const std::size_t type_pair = static_cast<std::size_t>(type1) * types + type2;
        const SpecialScale scale = find_scale(scales_, static_cast<std::int32_t>(atom1),
                                              static_cast<std::int32_t>(atom2));
        Vec3 fractional_delta = minimum_fractional(
            {atoms[atom1][0] - atoms[atom2][0], atoms[atom1][1] - atoms[atom2][1],
             atoms[atom1][2] - atoms[atom2][2]});
        const Vec3 delta = multiply(topology_.cell.h, fractional_delta);
        const double squared_distance = delta[0] * delta[0] + delta[1] * delta[1] +
            delta[2] * delta[2];
        const LennardJonesParameters &lj = topology_.lj[type_pair];
        if (lj.cutoff > 0.0 && squared_distance > 0.0 &&
            squared_distance < lj.cutoff * lj.cutoff) {
          const double r2inv = 1.0 / squared_distance;
          const double r6inv = r2inv * r2inv * r2inv;
          const double force_scalar =
              scale.lj * r6inv * (lj.lj1 * r6inv - lj.lj2) * r2inv;
          const Vec3 force = {delta[0] * force_scalar, delta[1] * force_scalar,
                              delta[2] * force_scalar};
          for (int dim = 0; dim < 3; ++dim) {
            forces[3 * atom1 + dim] += force[dim];
            forces[3 * atom2 + dim] -= force[dim];
          }
          lj_energy += scale.lj *
              (r6inv * (lj.lj3 * r6inv - lj.lj4) - lj.offset);
          virial[0] += delta[0] * force[0];
          virial[1] += delta[1] * force[1];
          virial[2] += delta[2] * force[2];
          virial[3] += delta[0] * force[1];
          virial[4] += delta[0] * force[2];
          virial[5] += delta[1] * force[2];
        }

        if (!topology_.coulomb_type_pairs[type_pair] || charges[atom1] == 0.0 ||
            charges[atom2] == 0.0)
          continue;
        Vec3 site_fractional_delta = minimum_fractional(
            {sites[atom1][0] - sites[atom2][0], sites[atom1][1] - sites[atom2][1],
             sites[atom1][2] - sites[atom2][2]});
        const Vec3 site_delta = multiply(topology_.cell.h, site_fractional_delta);
        const double site_squared = site_delta[0] * site_delta[0] +
            site_delta[1] * site_delta[1] + site_delta[2] * site_delta[2];
        if (!(site_squared > 0.0 &&
              site_squared < topology_.real_space_cutoff * topology_.real_space_cutoff))
          continue;

        double force_value = 0.0;
        double energy_value = 0.0;
        const double charge_product = charges[atom1] * charges[atom2];
        if (use_coulomb_table(site_squared)) {
          table_coulomb(site_squared, charge_product, scale.coulomb, force_value,
                        energy_value);
        } else {
          const double distance = std::sqrt(site_squared);
          const double grij = topology_.pppm.g_ewald * distance;
          const double expm2 = std::exp(-grij * grij);
          const double t = 1.0 / (1.0 + kEwaldP * grij);
          const double erfc =
              t * (kEwaldA1 + t * (kEwaldA2 + t * (kEwaldA3 +
                                                    t * (kEwaldA4 + t * kEwaldA5)))) *
              expm2;
          const double prefactor = topology_.qqrd2e * charge_product / distance;
          force_value = prefactor * (erfc + kEwaldF * grij * expm2) -
              (1.0 - scale.coulomb) * prefactor;
          energy_value = prefactor * erfc - (1.0 - scale.coulomb) * prefactor;
        }
        const double force_scalar = force_value / site_squared;
        const Vec3 force = {site_delta[0] * force_scalar,
                            site_delta[1] * force_scalar,
                            site_delta[2] * force_scalar};
        add_site_force(atom1, force, forces);
        add_site_force(atom2, {-force[0], -force[1], -force[2]}, forces);
        coulomb_energy += energy_value;
        virial[0] += site_delta[0] * force[0];
        virial[1] += site_delta[1] * force[1];
        virial[2] += site_delta[2] * force[2];
        virial[3] += site_delta[0] * force[1];
        virial[4] += site_delta[0] * force[2];
        virial[5] += site_delta[1] * force[2];
      }
    }
  }

  struct Assignment {
    std::array<int, 3> center{};
    std::array<std::vector<double>, 3> weights;
  };

  [[nodiscard]] Assignment assignment(const Vec3 &fractional) const {
    Assignment result;
    for (int dim = 0; dim < 3; ++dim) {
      const double grid = fractional[dim] * mesh_.mesh[dim];
      result.center[dim] = static_cast<int>(grid + spline_.shift) - kGridOffset;
      const double delta = result.center[dim] + spline_.shift_one - grid;
      result.weights[dim].resize(static_cast<std::size_t>(spline_.order));
      evaluate_spline(spline_, delta, result.weights[dim].data());
    }
    return result;
  }

  [[nodiscard]] PppmSolve solve_pppm(const std::vector<Vec3> &sites,
                                     const double *charges,
                                     bool compute_scalar_grid) const {
    std::vector<Complex> density(mesh_.count, Complex{});
    for (std::size_t atom = 0; atom < topology_.atom_count; ++atom) {
      const Assignment map = assignment(sites[atom]);
      for (int iz = spline_.lower; iz <= spline_.upper; ++iz) {
        const int z = periodic_index(map.center[2] + iz, mesh_.mesh[2]);
        const double wz = map.weights[2][static_cast<std::size_t>(iz - spline_.lower)];
        for (int iy = spline_.lower; iy <= spline_.upper; ++iy) {
          const int y = periodic_index(map.center[1] + iy, mesh_.mesh[1]);
          const double wy = map.weights[1][static_cast<std::size_t>(iy - spline_.lower)];
          for (int ix = spline_.lower; ix <= spline_.upper; ++ix) {
            const int x = periodic_index(map.center[0] + ix, mesh_.mesh[0]);
            const double wx = map.weights[0][static_cast<std::size_t>(ix - spline_.lower)];
            density[mesh_index(x, y, z, mesh_.mesh[0], mesh_.mesh[1])] +=
                charges[atom] * mesh_.delvolinv * wx * wy * wz;
          }
        }
      }
    }

    dft3d(density, mesh_.mesh, false);
    const double scale_inverse = 1.0 / static_cast<double>(mesh_.count);
    const double energy_scale = scale_inverse * scale_inverse;
    double raw_energy = 0.0;
    std::array<double, 6> raw_virial{};
    PppmSolve result;
    result.normalized_spectrum.resize(mesh_.count);
    std::vector<Complex> scalar;
    if (compute_scalar_grid) scalar.resize(mesh_.count);
    std::array<std::vector<Complex>, 3> gradient = {
        std::vector<Complex>(mesh_.count), std::vector<Complex>(mesh_.count),
        std::vector<Complex>(mesh_.count)};
    for (std::size_t index = 0; index < mesh_.count; ++index) {
      const double contribution =
          energy_scale * mesh_.green[index] * std::norm(density[index]);
      raw_energy += contribution;
      for (int component = 0; component < 6; ++component)
        raw_virial[component] +=
            contribution * mesh_.virial_factor[index][component];
      result.normalized_spectrum[index] =
          density[index] * (scale_inverse * mesh_.green[index]);
      if (compute_scalar_grid) scalar[index] = result.normalized_spectrum[index];
      for (int dim = 0; dim < 3; ++dim)
        gradient[dim][index] =
            Complex(-mesh_.kvector[index][dim] *
                        result.normalized_spectrum[index].imag(),
                    mesh_.kvector[index][dim] *
                        result.normalized_spectrum[index].real());
    }
    if (compute_scalar_grid) {
      dft3d(scalar, mesh_.mesh, true);
      result.scalar_grid.resize(mesh_.count);
      for (std::size_t index = 0; index < mesh_.count; ++index)
        result.scalar_grid[index] = scalar[index].real();
    }
    for (int dim = 0; dim < 3; ++dim) {
      dft3d(gradient[dim], mesh_.mesh, true);
      result.gradient_grid[dim].resize(mesh_.count);
      for (std::size_t index = 0; index < mesh_.count; ++index)
        result.gradient_grid[dim][index] = gradient[dim][index].real();
    }

    double qsqsum = 0.0;
    for (std::size_t atom = 0; atom < topology_.atom_count; ++atom) {
      result.qsum += charges[atom];
      qsqsum += charges[atom] * charges[atom];
    }
    result.energy = (0.5 * mesh_.volume * raw_energy -
                     topology_.pppm.g_ewald * qsqsum * kInvSqrtPi -
                     kHalfPi * result.qsum * result.qsum /
                         (topology_.pppm.g_ewald * topology_.pppm.g_ewald *
                          mesh_.volume)) *
        topology_.qqrd2e;
    for (int component = 0; component < 6; ++component)
      result.virial[component] =
          0.5 * topology_.qqrd2e * mesh_.volume * raw_virial[component];
    return result;
  }

  void interpolate_pppm(const std::vector<Vec3> &sites, const PppmSolve &solve,
                        const double *charges, double *forces,
                        double *potential) const {
    if (potential && solve.scalar_grid.size() != mesh_.count)
      throw std::logic_error("PPPM scalar grid was not retained");
    for (std::size_t atom = 0; atom < topology_.atom_count; ++atom) {
      const Assignment map = assignment(sites[atom]);
      double scalar_value = 0.0;
      Vec3 electric{};
      for (int iz = spline_.lower; iz <= spline_.upper; ++iz) {
        const int z = periodic_index(map.center[2] + iz, mesh_.mesh[2]);
        const double wz = map.weights[2][static_cast<std::size_t>(iz - spline_.lower)];
        for (int iy = spline_.lower; iy <= spline_.upper; ++iy) {
          const int y = periodic_index(map.center[1] + iy, mesh_.mesh[1]);
          const double wy = map.weights[1][static_cast<std::size_t>(iy - spline_.lower)];
          for (int ix = spline_.lower; ix <= spline_.upper; ++ix) {
            const int x = periodic_index(map.center[0] + ix, mesh_.mesh[0]);
            const double weight = wz * wy *
                map.weights[0][static_cast<std::size_t>(ix - spline_.lower)];
            const std::size_t index = mesh_index(x, y, z, mesh_.mesh[0], mesh_.mesh[1]);
            if (potential) scalar_value += weight * solve.scalar_grid[index];
            if (forces)
              for (int dim = 0; dim < 3; ++dim)
                electric[dim] -= weight * solve.gradient_grid[dim][index];
          }
        }
      }
      if (potential) potential[atom] = scalar_value;
      if (forces) {
        const Vec3 site_force = {topology_.qqrd2e * charges[atom] * electric[0],
                                 topology_.qqrd2e * charges[atom] * electric[1],
                                 topology_.qqrd2e * charges[atom] * electric[2]};
        add_site_force(atom, site_force, forces);
      }
    }
  }

  [[nodiscard]] PppmSolve combine_solves(const PppmSolve &mm,
                                         const PppmSolve &qm) const {
    PppmSolve full;
    full.qsum = mm.qsum + qm.qsum;
    for (int dim = 0; dim < 3; ++dim) {
      full.gradient_grid[dim].resize(mesh_.count);
      for (std::size_t index = 0; index < mesh_.count; ++index)
        full.gradient_grid[dim][index] =
            mm.gradient_grid[dim][index] + qm.gradient_grid[dim][index];
    }
    double cross_energy = 0.0;
    std::array<double, 6> cross_virial{};
    for (std::size_t index = 0; index < mesh_.count; ++index) {
      if (mesh_.green[index] == 0.0) continue;
      const double product =
          (mm.normalized_spectrum[index].real() *
               qm.normalized_spectrum[index].real() +
           mm.normalized_spectrum[index].imag() *
               qm.normalized_spectrum[index].imag()) /
          mesh_.green[index];
      cross_energy += product;
      for (int component = 0; component < 6; ++component)
        cross_virial[component] +=
            product * mesh_.virial_factor[index][component];
    }
    cross_energy = mesh_.volume * topology_.qqrd2e * cross_energy -
        kPi * mm.qsum * qm.qsum /
            (topology_.pppm.g_ewald * topology_.pppm.g_ewald * mesh_.volume) *
            topology_.qqrd2e;
    full.energy = mm.energy + qm.energy + cross_energy;
    for (int component = 0; component < 6; ++component)
      full.virial[component] =
          mm.virial[component] + qm.virial[component] +
          mesh_.volume * topology_.qqrd2e * cross_virial[component];
    return full;
  }

  ClassicalTopology topology_;
  std::size_t max_batch_count_ = 0;
  Mat3 hinv_{};
  SplineData spline_;
  MeshData mesh_;
  std::vector<int> oxygen_site_;
  std::vector<SpecialScale> scales_;
  bool active_ = false;
  std::size_t cached_batch_count_ = 0;
  std::vector<FrameSites> cached_sites_;
  std::vector<PppmSolve> mm_solve_;
  std::vector<double> cached_mm_charges_;
};

}  // namespace

PreparedClassicalData prepare_classical_data(const ClassicalTopology &topology) {
  if (topology.atom_count == 0 || topology.type_count <= 0 ||
      topology.atom_types.size() != topology.atom_count)
    throw std::invalid_argument("invalid classical topology extent");
  for (std::int32_t type : topology.atom_types)
    if (type < 0 || type >= topology.type_count)
      throw std::invalid_argument("atom type is outside the zero-based type range");
  const std::size_t type_pairs = checked_product(
      static_cast<std::size_t>(topology.type_count),
      static_cast<std::size_t>(topology.type_count), "type-pair matrix");
  if (topology.lj.size() != type_pairs ||
      topology.coulomb_type_pairs.size() != type_pairs)
    throw std::invalid_argument("type-pair parameter extents are invalid");
  for (const LennardJonesParameters &entry : topology.lj)
    if (!std::isfinite(entry.lj1) || !std::isfinite(entry.lj2) ||
        !std::isfinite(entry.lj3) || !std::isfinite(entry.lj4) ||
        !std::isfinite(entry.offset) || !std::isfinite(entry.cutoff) ||
        entry.cutoff < 0.0)
      throw std::invalid_argument("Lennard-Jones parameters must be finite");
  if (std::any_of(topology.coulomb_type_pairs.begin(),
                  topology.coulomb_type_pairs.end(),
                  [](std::uint8_t enabled) { return enabled > 1; }))
    throw std::invalid_argument("Coulomb type-pair entries must be zero or one");
  if (!std::all_of(topology.cell.boxlo.begin(), topology.cell.boxlo.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(topology.cell.h.begin(), topology.cell.h.end(),
                   [](double value) { return std::isfinite(value); }) ||
      topology.cell.h[3] != 0.0 || topology.cell.h[6] != 0.0 ||
      topology.cell.h[7] != 0.0)
    throw std::invalid_argument("cell must be finite restricted-triclinic form");
  if (!std::isfinite(topology.qqrd2e) || topology.qqrd2e <= 0.0 ||
      !std::isfinite(topology.pppm.g_ewald) || topology.pppm.g_ewald <= 0.0 ||
      !std::isfinite(topology.real_space_cutoff) ||
      topology.real_space_cutoff <= 0.0 ||
      !std::isfinite(topology.neighbor_skin) || topology.neighbor_skin < 0.0 ||
      !std::isfinite(topology.tip4p_alpha) || topology.tip4p_alpha < 0.0 ||
      !std::isfinite(topology.tip4p_qdist) || topology.tip4p_qdist < 0.0)
    throw std::invalid_argument("classical scalar parameters are invalid");
  for (std::int32_t extent : topology.pppm.mesh)
    if (extent <= 0)
      throw std::invalid_argument("PPPM mesh extents must be positive");

  const CoulombLookupTable &table = topology.coulomb_table;
  if (table.bits == 0) {
    if (!table.r.empty() || !table.dr.empty() || !table.force.empty() ||
        !table.dforce.empty() || !table.coulomb.empty() ||
        !table.dcoulomb.empty() || !table.energy.empty() ||
        !table.denergy.empty())
      throw std::invalid_argument("disabled Coulomb table must not carry arrays");
  } else {
    if (table.bits < 1 || table.bits > 20 || table.shift_bits < 0 ||
        table.shift_bits > 31 || table.mask < 0 ||
        !std::isfinite(table.inner_squared) || table.inner_squared < 0.0)
      throw std::invalid_argument("Coulomb table metadata is invalid");
    const std::size_t count = std::size_t{1} << table.bits;
    if (table.r.size() != count || table.dr.size() != count ||
        table.force.size() != count || table.dforce.size() != count ||
        table.coulomb.size() != count || table.dcoulomb.size() != count ||
        table.energy.size() != count || table.denergy.size() != count)
      throw std::invalid_argument("Coulomb table array extent is invalid");
  }
  PreparedClassicalData prepared;
  prepared.hinv = inverse(topology.cell.h);
  const SplineData spline = make_spline(topology.pppm.order);
  const MeshData mesh = make_mesh(topology, prepared.hinv);
  prepared.volume = mesh.volume;
  prepared.delvolinv = mesh.delvolinv;
  prepared.mesh = {mesh.mesh[0], mesh.mesh[1], mesh.mesh[2]};
  prepared.mesh_count = mesh.count;
  prepared.spline_lower = spline.lower;
  prepared.spline_upper = spline.upper;
  prepared.spline_shift = spline.shift;
  prepared.spline_shift_one = spline.shift_one;
  prepared.spline_coefficients = spline.coefficients;
  prepared.green = mesh.green;
  prepared.kvector.resize(3 * mesh.count);
  prepared.virial_factor.resize(6 * mesh.count);
  for (std::size_t index = 0; index < mesh.count; ++index) {
    for (int dim = 0; dim < 3; ++dim)
      prepared.kvector[3 * index + dim] = mesh.kvector[index][dim];
    for (int component = 0; component < 6; ++component)
      prepared.virial_factor[6 * index + component] =
          mesh.virial_factor[index][component];
  }

  prepared.oxygen_site.assign(topology.atom_count, -1);
  for (std::size_t site = 0; site < topology.tip4p_sites.size(); ++site) {
    const Tip4pSite &entry = topology.tip4p_sites[site];
    if (entry.oxygen < 0 || entry.hydrogen1 < 0 || entry.hydrogen2 < 0 ||
        static_cast<std::size_t>(entry.oxygen) >= topology.atom_count ||
        static_cast<std::size_t>(entry.hydrogen1) >= topology.atom_count ||
        static_cast<std::size_t>(entry.hydrogen2) >= topology.atom_count ||
        entry.oxygen == entry.hydrogen1 || entry.oxygen == entry.hydrogen2 ||
        entry.hydrogen1 == entry.hydrogen2)
      throw std::invalid_argument("invalid TIP4P topology");
    if (prepared.oxygen_site[static_cast<std::size_t>(entry.oxygen)] >= 0)
      throw std::invalid_argument("one oxygen appears in multiple TIP4P sites");
    prepared.oxygen_site[static_cast<std::size_t>(entry.oxygen)] =
        static_cast<std::int32_t>(site);
  }

  struct CsrEntry {
    std::int32_t partner = -1;
    double lj = 1.0;
    double coulomb = 1.0;
  };
  std::vector<std::vector<CsrEntry>> special(topology.atom_count);
  for (const SpecialPair &entry : topology.special_pairs) {
    if (entry.atom1 < 0 || entry.atom2 < 0 || entry.atom1 == entry.atom2 ||
        static_cast<std::size_t>(entry.atom1) >= topology.atom_count ||
        static_cast<std::size_t>(entry.atom2) >= topology.atom_count ||
        !std::isfinite(entry.lj_scale) ||
        !std::isfinite(entry.coulomb_scale))
      throw std::invalid_argument("invalid special-pair topology");
    special[static_cast<std::size_t>(entry.atom1)].push_back(
        {entry.atom2, entry.lj_scale, entry.coulomb_scale});
    special[static_cast<std::size_t>(entry.atom2)].push_back(
        {entry.atom1, entry.lj_scale, entry.coulomb_scale});
  }
  prepared.special_offsets.resize(topology.atom_count + 1, 0);
  for (std::size_t atom = 0; atom < topology.atom_count; ++atom) {
    auto &entries = special[atom];
    std::sort(entries.begin(), entries.end(),
              [](const CsrEntry &left, const CsrEntry &right) {
                return left.partner < right.partner;
              });
    if (std::adjacent_find(entries.begin(), entries.end(),
                           [](const CsrEntry &left, const CsrEntry &right) {
                             return left.partner == right.partner;
                           }) != entries.end())
      throw std::invalid_argument("duplicate special-pair topology");
    for (const CsrEntry &entry : entries) {
      prepared.special_partners.push_back(entry.partner);
      prepared.special_lj.push_back(entry.lj);
      prepared.special_coulomb.push_back(entry.coulomb);
    }
    prepared.special_offsets[atom + 1] =
        static_cast<std::int32_t>(prepared.special_partners.size());
  }

  double maximum_lj_cutoff = 0.0;
  for (const LennardJonesParameters &entry : topology.lj)
    maximum_lj_cutoff = std::max(maximum_lj_cutoff, entry.cutoff);
  const double neighbor_cutoff =
      std::max(maximum_lj_cutoff,
               topology.real_space_cutoff + 2.0 * topology.tip4p_qdist) +
      topology.neighbor_skin;
  if (!std::isfinite(neighbor_cutoff) || neighbor_cutoff <= 0.0)
    throw std::invalid_argument("invalid classical neighbor cutoff");
  prepared.neighbor_cutoff = neighbor_cutoff;
  std::array<int, 3> radius{};
  for (int dim = 0; dim < 3; ++dim) {
    const double row_norm =
        std::sqrt(prepared.hinv[3 * dim] * prepared.hinv[3 * dim] +
                  prepared.hinv[3 * dim + 1] * prepared.hinv[3 * dim + 1] +
                  prepared.hinv[3 * dim + 2] * prepared.hinv[3 * dim + 2]);
    const double height = 1.0 / row_norm;
    prepared.bin_count[dim] = std::max<std::int32_t>(
        1, static_cast<std::int32_t>(std::floor(2.0 * height / neighbor_cutoff)));
    radius[dim] = std::max(
        1, static_cast<int>(std::ceil(neighbor_cutoff * row_norm *
                                     prepared.bin_count[dim])));
  }
  const int bins = prepared.bin_count[0] * prepared.bin_count[1] *
      prepared.bin_count[2];
  prepared.neighbor_bin_offsets.resize(static_cast<std::size_t>(bins) + 1, 0);
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(bins), 0);
  for (int bin = 0; bin < bins; ++bin) {
    std::fill(seen.begin(), seen.end(), 0);
    const int bx = bin % prepared.bin_count[0];
    const int by = (bin / prepared.bin_count[0]) % prepared.bin_count[1];
    const int bz = bin / (prepared.bin_count[0] * prepared.bin_count[1]);
    for (int dz = -radius[2]; dz <= radius[2]; ++dz)
      for (int dy = -radius[1]; dy <= radius[1]; ++dy)
        for (int dx = -radius[0]; dx <= radius[0]; ++dx) {
          const int nx = periodic_index(bx + dx, prepared.bin_count[0]);
          const int ny = periodic_index(by + dy, prepared.bin_count[1]);
          const int nz = periodic_index(bz + dz, prepared.bin_count[2]);
          const int neighbor =
              (nz * prepared.bin_count[1] + ny) * prepared.bin_count[0] + nx;
          if (seen[static_cast<std::size_t>(neighbor)]) continue;
          seen[static_cast<std::size_t>(neighbor)] = 1;
          prepared.neighbor_bins.push_back(neighbor);
        }
    prepared.neighbor_bin_offsets[static_cast<std::size_t>(bin) + 1] =
        static_cast<std::int32_t>(prepared.neighbor_bins.size());
  }
  return prepared;
}

std::unique_ptr<ClassicalBatchPlan>
create_classical_batch_plan(const ClassicalTopology &topology,
                            const ClassicalPlanOptions &options) {
  if (options.backend == ClassicalBackend::CPU)
    return std::make_unique<CpuClassicalBatchPlan>(topology, options.max_batch_count);
#if defined(DPRC_HAVE_CLASSICAL_CUDA)
  return create_cuda_classical_batch_plan(topology, options);
#else
  throw std::runtime_error("the classical CUDA backend was not compiled");
#endif
}

}  // namespace DPRC
