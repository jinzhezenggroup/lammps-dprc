#include "classical_batch.h"
#include "classical_batch_internal.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DPRC {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kHalfPi = 0.5 * kPi;
constexpr double kInvSqrtPi = 0.564189583547756286948079451560772586;
constexpr double kEwaldF = 1.12837917;
constexpr double kEwaldP = 0.3275911;
constexpr double kEwaldA1 = 0.254829592;
constexpr double kEwaldA2 = -0.284496736;
constexpr double kEwaldA3 = 1.421413741;
constexpr double kEwaldA4 = -1.453152027;
constexpr double kEwaldA5 = 1.061405429;
constexpr int kGridOffset = 16384;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kRealSpaceWarpsPerBlock = 8;
constexpr int kRealSpaceThreads = kWarpSize * kRealSpaceWarpsPerBlock;

[[noreturn]] void throw_cuda(cudaError_t status, const char *operation) {
  throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) throw_cuda(status, operation);
}

[[noreturn]] void throw_cufft(cufftResult status, const char *operation) {
  throw std::runtime_error(std::string(operation) + " failed with cuFFT status " +
                           std::to_string(static_cast<int>(status)));
}

void check_cufft(cufftResult status, const char *operation) {
  if (status != CUFFT_SUCCESS) throw_cufft(status, operation);
}

[[nodiscard]] bool cuda_memory_diagnostics_enabled() noexcept {
  const char *value = std::getenv("DPRC_CUDA_MEMORY_DIAGNOSTICS");
  return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

template <class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        count_(std::exchange(other.count_, 0)),
        device_(std::exchange(other.device_, -1)) {}

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = std::exchange(other.pointer_, nullptr);
      count_ = std::exchange(other.count_, 0);
      device_ = std::exchange(other.device_, -1);
    }
    return *this;
  }

  void allocate(std::size_t count, const char *label = "unnamed") {
    reset();
    if (count == 0) return;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error(std::string("classical CUDA buffer size overflows: ") +
                                label);
    check_cuda(cudaGetDevice(&device_), "cudaGetDevice before allocation");
    const std::size_t bytes = count * sizeof(T);
    std::size_t free_before = 0;
    std::size_t total = 0;
    const cudaError_t info_before = cudaMemGetInfo(&free_before, &total);
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(&pointer_), bytes);
    if (status != cudaSuccess) {
      std::string diagnostic = std::string("cudaMalloc classical workspace '") +
          label + "' requested " + std::to_string(bytes) + " bytes";
      if (info_before == cudaSuccess)
        diagnostic += ", free before " + std::to_string(free_before) +
            " of " + std::to_string(total) + " bytes";
      diagnostic += ": " + std::string(cudaGetErrorString(status));
      throw std::runtime_error(diagnostic);
    }
    count_ = count;
    if (cuda_memory_diagnostics_enabled()) {
      std::size_t free_after = 0;
      std::size_t total_after = 0;
      const cudaError_t info_after = cudaMemGetInfo(&free_after, &total_after);
      std::fprintf(stderr,
                   "DPRC_CUDA_MEMORY label=%s requested_bytes=%zu "
                   "free_before_bytes=%zu free_after_bytes=%zu total_bytes=%zu\n",
                   label, bytes,
                   info_before == cudaSuccess ? free_before : std::size_t{0},
                   info_after == cudaSuccess ? free_after : std::size_t{0},
                   info_after == cudaSuccess ? total_after : total);
      std::fflush(stderr);
    }
  }

  void reset() noexcept {
    if (pointer_) {
      int previous_device = -1;
      const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
      const bool switched =
          have_previous && previous_device != device_ && cudaSetDevice(device_) == cudaSuccess;
      cudaFree(pointer_);
      if (switched) cudaSetDevice(previous_device);
    }
    pointer_ = nullptr;
    count_ = 0;
    device_ = -1;
  }

  [[nodiscard]] T *get() noexcept { return pointer_; }
  [[nodiscard]] const T *get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T *pointer_ = nullptr;
  std::size_t count_ = 0;
  int device_ = -1;
};

template <class T> class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { reset(); }

  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  void allocate(std::size_t count) {
    reset();
    if (count == 0) return;
    check_cuda(cudaMallocHost(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
               "cudaMallocHost classical staging");
    count_ = count;
  }

  void reset() noexcept {
    if (pointer_) cudaFreeHost(pointer_);
    pointer_ = nullptr;
    count_ = 0;
  }

  [[nodiscard]] T *data() noexcept { return pointer_; }
  [[nodiscard]] const T *data() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] T &operator[](std::size_t index) noexcept { return pointer_[index]; }
  [[nodiscard]] const T &operator[](std::size_t index) const noexcept {
    return pointer_[index];
  }

 private:
  T *pointer_ = nullptr;
  std::size_t count_ = 0;
};

// Preserve the host application's current CUDA device around every public
// operation.  The broker owns one selected device, but loading this plugin
// must not silently retarget CUDA work performed by another LAMMPS package.
class ScopedDevice final {
 public:
  explicit ScopedDevice(int target) {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice before classical operation");
    if (previous_ != target) {
      check_cuda(cudaSetDevice(target), "cudaSetDevice classical operation");
      switched_ = true;
    }
  }

  ~ScopedDevice() {
    if (switched_) cudaSetDevice(previous_);
  }

  ScopedDevice(const ScopedDevice &) = delete;
  ScopedDevice &operator=(const ScopedDevice &) = delete;

 private:
  int previous_ = 0;
  bool switched_ = false;
};

struct DeviceParameters {
  int atoms = 0;
  int types = 0;
  int tip4p_sites = 0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int order = 0;
  int lower = 0;
  int upper = 0;
  int bins_x = 0;
  int bins_y = 0;
  int bins_z = 0;
  int bins = 0;
  int table_bits = 0;
  int table_shift = 0;
  int table_mask = 0;
  double table_inner_squared = 0.0;
  double shift = 0.0;
  double shift_one = 0.0;
  double alpha = 0.0;
  double real_cutoff_squared = 0.0;
  double neighbor_cutoff_squared = 0.0;
  double g_ewald = 0.0;
  double qqrd2e = 0.0;
  double volume = 0.0;
  double delvolinv = 0.0;
  double h[9]{};
  double hinv[9]{};
  double boxlo[3]{};
};

struct FftPlans {
  cufftHandle forward = 0;
  cufftHandle inverse_mm = 0;
  cufftHandle inverse_qm = 0;
  std::size_t workspace_bytes = 0;

  ~FftPlans() {
    if (forward) cufftDestroy(forward);
    if (inverse_mm) cufftDestroy(inverse_mm);
    if (inverse_qm) cufftDestroy(inverse_qm);
  }

  void bind_workspace(void *workspace) const {
    check_cufft(cufftSetWorkArea(forward, workspace),
                "bind forward cuFFT workspace");
    check_cufft(cufftSetWorkArea(inverse_mm, workspace),
                "bind MM inverse cuFFT workspace");
    check_cufft(cufftSetWorkArea(inverse_qm, workspace),
                "bind QM inverse cuFFT workspace");
  }
};

__device__ double3 matrix_vector(const double *matrix, double3 vector) {
  return make_double3(matrix[0] * vector.x + matrix[1] * vector.y + matrix[2] * vector.z,
                      matrix[3] * vector.x + matrix[4] * vector.y + matrix[5] * vector.z,
                      matrix[6] * vector.x + matrix[7] * vector.y + matrix[8] * vector.z);
}

__device__ double3 wrap_fractional(double3 value) {
  value.x -= floor(value.x);
  value.y -= floor(value.y);
  value.z -= floor(value.z);
  return value;
}

__device__ double3 minimum_fractional(double3 value) {
  value.x -= nearbyint(value.x);
  value.y -= nearbyint(value.y);
  value.z -= nearbyint(value.z);
  return value;
}

__device__ int periodic_index(int value, int extent) {
  value %= extent;
  return value < 0 ? value + extent : value;
}

__device__ std::size_t grid_index(int x, int y, int z,
                                  const DeviceParameters &parameters) {
  return (static_cast<std::size_t>(z) * parameters.ny + y) * parameters.nx + x;
}

__device__ double spline_weight(const double *coefficients,
                                const DeviceParameters &parameters, int stencil,
                                double delta) {
  double value = 0.0;
  const int packed = stencil - parameters.lower;
  for (int power = parameters.order - 1; power >= 0; --power)
    value = coefficients[power * parameters.order + packed] + value * delta;
  return value;
}

__device__ void add_site_force(int frame, int atom, double3 force, double *forces,
                               const std::int32_t *oxygen_site,
                               const Tip4pSite *tip4p_sites,
                               const DeviceParameters &parameters) {
  const int site_index = oxygen_site[atom];
  const std::size_t frame_offset =
      static_cast<std::size_t>(frame) * parameters.atoms * 3;
  if (site_index < 0) {
    atomicAdd(forces + frame_offset + 3 * atom, force.x);
    atomicAdd(forces + frame_offset + 3 * atom + 1, force.y);
    atomicAdd(forces + frame_offset + 3 * atom + 2, force.z);
    return;
  }
  const Tip4pSite site = tip4p_sites[site_index];
  const double oxygen_weight = 1.0 - parameters.alpha;
  const double hydrogen_weight = 0.5 * parameters.alpha;
  const int parents[3] = {site.oxygen, site.hydrogen1, site.hydrogen2};
  const double weights[3] = {oxygen_weight, hydrogen_weight, hydrogen_weight};
  for (int parent = 0; parent < 3; ++parent) {
    atomicAdd(forces + frame_offset + 3 * parents[parent], weights[parent] * force.x);
    atomicAdd(forces + frame_offset + 3 * parents[parent] + 1,
              weights[parent] * force.y);
    atomicAdd(forces + frame_offset + 3 * parents[parent] + 2,
              weights[parent] * force.z);
  }
}

__global__ void fractional_and_bins_kernel(
    const double *positions, double3 *atom_fractional, double3 *site_fractional,
    int *atom_bins, int *bin_counts, std::size_t total_atoms,
    bool rebuild_bins, DeviceParameters parameters) {
  const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const std::size_t coordinate = 3 * global;
  const double3 shifted =
      make_double3(positions[coordinate] - parameters.boxlo[0],
                   positions[coordinate + 1] - parameters.boxlo[1],
                   positions[coordinate + 2] - parameters.boxlo[2]);
  const double3 fractional = wrap_fractional(matrix_vector(parameters.hinv, shifted));
  atom_fractional[global] = fractional;
  site_fractional[global] = fractional;
  if (!rebuild_bins) return;
  const int bx = min(static_cast<int>(fractional.x * parameters.bins_x),
                     parameters.bins_x - 1);
  const int by = min(static_cast<int>(fractional.y * parameters.bins_y),
                     parameters.bins_y - 1);
  const int bz = min(static_cast<int>(fractional.z * parameters.bins_z),
                     parameters.bins_z - 1);
  const int local_bin = (bz * parameters.bins_y + by) * parameters.bins_x + bx;
  const int global_bin = frame * parameters.bins + local_bin;
  atom_bins[global] = global_bin;
  atomicAdd(bin_counts + global_bin, 1);
}

__global__ void tip4p_sites_kernel(double3 *site_fractional,
                                   const double3 *atom_fractional,
                                   const Tip4pSite *sites, int batch,
                                   DeviceParameters parameters) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * parameters.tip4p_sites;
  if (global >= total) return;
  const int frame = global / parameters.tip4p_sites;
  const Tip4pSite site = sites[global % parameters.tip4p_sites];
  const std::size_t offset = static_cast<std::size_t>(frame) * parameters.atoms;
  const double3 oxygen = atom_fractional[offset + site.oxygen];
  const double3 h1 = atom_fractional[offset + site.hydrogen1];
  const double3 h2 = atom_fractional[offset + site.hydrogen2];
  const double3 d1 = minimum_fractional(
      make_double3(h1.x - oxygen.x, h1.y - oxygen.y, h1.z - oxygen.z));
  const double3 d2 = minimum_fractional(
      make_double3(h2.x - oxygen.x, h2.y - oxygen.y, h2.z - oxygen.z));
  site_fractional[offset + site.oxygen] = wrap_fractional(make_double3(
      oxygen.x + 0.5 * parameters.alpha * (d1.x + d2.x),
      oxygen.y + 0.5 * parameters.alpha * (d1.y + d2.y),
      oxygen.z + 0.5 * parameters.alpha * (d1.z + d2.z)));
}

__global__ void fill_bins_kernel(const int *atom_bins, int *bin_cursor,
                                 int *bin_atoms, std::size_t total_atoms,
                                 DeviceParameters parameters) {
  const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= total_atoms) return;
  const int local_atom = static_cast<int>(global % parameters.atoms);
  const int position = atomicAdd(bin_cursor + atom_bins[global], 1);
  bin_atoms[position] = local_atom;
}

__device__ bool inside_neighbor_cutoff(
    const double3 &atom1_fractional, const double3 *atom_fractional,
    std::size_t frame_offset, int atom2, const DeviceParameters &parameters) {
  const double3 fractional = minimum_fractional(make_double3(
      atom1_fractional.x - atom_fractional[frame_offset + atom2].x,
      atom1_fractional.y - atom_fractional[frame_offset + atom2].y,
      atom1_fractional.z - atom_fractional[frame_offset + atom2].z));
  const double3 delta = matrix_vector(parameters.h, fractional);
  const double squared =
      delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
  return squared < parameters.neighbor_cutoff_squared;
}

__global__ void count_verlet_neighbors_kernel(
    const double3 *atom_fractional, const int *atom_bins,
    const int *bin_offsets, const int *bin_atoms,
    const int *neighbor_bin_offsets, const int *neighbor_bins,
    std::uint64_t *neighbor_counts, std::size_t total_atoms,
    DeviceParameters parameters) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const std::size_t global =
      static_cast<std::size_t>(blockIdx.x) * kRealSpaceWarpsPerBlock + warp;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const int atom1 = static_cast<int>(global % parameters.atoms);
  const std::size_t frame_offset =
      static_cast<std::size_t>(frame) * parameters.atoms;
  const double3 atom1_fractional = atom_fractional[frame_offset + atom1];
  const int local_bin = atom_bins[global] - frame * parameters.bins;
  unsigned int count = 0;
  for (int neighbor_index = neighbor_bin_offsets[local_bin];
       neighbor_index < neighbor_bin_offsets[local_bin + 1]; ++neighbor_index) {
    const int neighbor_global_bin =
        frame * parameters.bins + neighbor_bins[neighbor_index];
    for (int packed = bin_offsets[neighbor_global_bin] + lane;
         packed < bin_offsets[neighbor_global_bin + 1]; packed += kWarpSize) {
      const int atom2 = bin_atoms[packed];
      if (atom2 != atom1 &&
          inside_neighbor_cutoff(atom1_fractional, atom_fractional,
                                 frame_offset, atom2, parameters))
        ++count;
    }
  }
  constexpr unsigned int mask = 0xffffffffu;
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    count += __shfl_down_sync(mask, count, offset);
  if (lane == 0) neighbor_counts[global] = count;
}

__global__ void fill_verlet_neighbors_kernel(
    const double3 *atom_fractional, const int *atom_bins,
    const int *bin_offsets, const int *bin_atoms,
    const int *neighbor_bin_offsets, const int *neighbor_bins,
    const std::uint64_t *neighbor_offsets, int *neighbor_cursor,
    int *verlet_neighbors, std::size_t total_atoms,
    DeviceParameters parameters) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const std::size_t global =
      static_cast<std::size_t>(blockIdx.x) * kRealSpaceWarpsPerBlock + warp;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const int atom1 = static_cast<int>(global % parameters.atoms);
  const std::size_t frame_offset =
      static_cast<std::size_t>(frame) * parameters.atoms;
  const double3 atom1_fractional = atom_fractional[frame_offset + atom1];
  const int local_bin = atom_bins[global] - frame * parameters.bins;
  for (int neighbor_index = neighbor_bin_offsets[local_bin];
       neighbor_index < neighbor_bin_offsets[local_bin + 1]; ++neighbor_index) {
    const int neighbor_global_bin =
        frame * parameters.bins + neighbor_bins[neighbor_index];
    for (int packed = bin_offsets[neighbor_global_bin] + lane;
         packed < bin_offsets[neighbor_global_bin + 1]; packed += kWarpSize) {
      const int atom2 = bin_atoms[packed];
      if (atom2 == atom1 ||
          !inside_neighbor_cutoff(atom1_fractional, atom_fractional,
                                  frame_offset, atom2, parameters))
        continue;
      const int slot = atomicAdd(neighbor_cursor + global, 1);
      verlet_neighbors[neighbor_offsets[global] +
                       static_cast<std::uint64_t>(slot)] = atom2;
    }
  }
}

__device__ void special_scale(int atom1, int atom2, const int *offsets,
                              const int *partners, const double *lj_values,
                              const double *coulomb_values, double &lj,
                              double &coulomb) {
  lj = 1.0;
  coulomb = 1.0;
  for (int index = offsets[atom1]; index < offsets[atom1 + 1]; ++index) {
    if (partners[index] == atom2) {
      lj = lj_values[index];
      coulomb = coulomb_values[index];
      return;
    }
    if (partners[index] > atom2) return;
  }
}

__device__ void lookup_table_coulomb(
    double squared_distance, double charge_product, double scale,
    const double *table_r, const double *table_dr, const double *table_force,
    const double *table_dforce, const double *table_coulomb,
    const double *table_dcoulomb, const double *table_energy,
    const double *table_denergy, const DeviceParameters &parameters,
    double &force, double &energy) {
  unsigned int bits = __float_as_uint(static_cast<float>(squared_distance));
  int index = static_cast<int>(bits & static_cast<unsigned int>(parameters.table_mask));
  index >>= parameters.table_shift;
  const float squared_float = static_cast<float>(squared_distance);
  const double fraction = (static_cast<double>(squared_float) - table_r[index]) *
      table_dr[index];
  force = charge_product * (table_force[index] + fraction * table_dforce[index]);
  energy = charge_product * (table_energy[index] + fraction * table_denergy[index]);
  if (scale < 1.0) {
    const double prefactor = charge_product *
        (table_coulomb[index] + fraction * table_dcoulomb[index]);
    force -= (1.0 - scale) * prefactor;
    energy -= (1.0 - scale) * prefactor;
  }
}

__device__ double warp_sum(double value) {
  constexpr unsigned int mask = 0xffffffffu;
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    value += __shfl_down_sync(mask, value, offset);
  return value;
}

struct RawContribution {
  double values[7]{};
};

struct RawContributionSum {
  __device__ RawContribution operator()(const RawContribution &left,
                                        const RawContribution &right) const {
    RawContribution result;
#pragma unroll
    for (int component = 0; component < 7; ++component)
      result.values[component] =
          left.values[component] + right.values[component];
    return result;
  }
};

__device__ void publish_raw_block(RawContribution contribution, double *raw,
                                  std::size_t frame) {
  using Reduction = cub::BlockReduce<RawContribution, kThreads>;
  __shared__ typename Reduction::TempStorage storage;
  const RawContribution block =
      Reduction(storage).Reduce(contribution, RawContributionSum{});
  if (threadIdx.x == 0)
#pragma unroll
    for (int component = 0; component < 7; ++component)
      atomicAdd(raw + 7 * frame + component, block.values[component]);
}

__global__ void real_space_kernel(
    const double3 *atom_fractional, const double3 *site_fractional,
    const double *charges, const int *atom_types,
    const LennardJonesParameters *lj_parameters,
    const std::uint8_t *coulomb_type_pairs,
    const std::uint64_t *neighbor_offsets, const int *verlet_neighbors,
    const int *special_offsets, const int *special_partners,
    const double *special_lj, const double *special_coulomb,
    const std::int32_t *oxygen_site, const Tip4pSite *tip4p_sites,
    const double *table_r, const double *table_dr, const double *table_force,
    const double *table_dforce, const double *table_coulomb,
    const double *table_dcoulomb, const double *table_energy,
    const double *table_denergy, double *forces, double *scalars,
    std::size_t total_atoms, DeviceParameters parameters) {
  // A warp owns one atom and evaluates a full (Newton-off) neighbor list.  The
  // pair arithmetic is intentionally repeated from the partner's warp: doing
  // so replaces millions of contended FP64 atomics to partner forces with one
  // warp-reduced publication per atom.  Energy and virial remain single-counted
  // by accumulating them only for atom2 > atom1.
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const std::size_t global =
      static_cast<std::size_t>(blockIdx.x) * kRealSpaceWarpsPerBlock + warp;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const int atom1 = static_cast<int>(global % parameters.atoms);
  const int type1 = atom_types[atom1];
  const std::size_t frame_offset =
      static_cast<std::size_t>(frame) * parameters.atoms;
  const double3 atom1_fractional = atom_fractional[frame_offset + atom1];
  const double3 site1_fractional = site_fractional[frame_offset + atom1];
  const double charge1 = charges[frame_offset + atom1];

  double direct_force_x = 0.0;
  double direct_force_y = 0.0;
  double direct_force_z = 0.0;
  double site_force_x = 0.0;
  double site_force_y = 0.0;
  double site_force_z = 0.0;
  double lj_energy = 0.0;
  double coulomb_energy = 0.0;
  double virial_xx = 0.0;
  double virial_yy = 0.0;
  double virial_zz = 0.0;
  double virial_xy = 0.0;
  double virial_xz = 0.0;
  double virial_yz = 0.0;

  for (std::uint64_t packed = neighbor_offsets[global] + lane;
       packed < neighbor_offsets[global + 1]; packed += kWarpSize) {
      const int atom2 = verlet_neighbors[packed];
      const bool tally = atom2 > atom1;
      const int type2 = atom_types[atom2];
      const double3 atom_delta_fractional = minimum_fractional(make_double3(
          atom1_fractional.x - atom_fractional[frame_offset + atom2].x,
          atom1_fractional.y - atom_fractional[frame_offset + atom2].y,
          atom1_fractional.z - atom_fractional[frame_offset + atom2].z));
      const double3 atom_delta = matrix_vector(parameters.h, atom_delta_fractional);
      const double atom_squared = atom_delta.x * atom_delta.x +
          atom_delta.y * atom_delta.y + atom_delta.z * atom_delta.z;
      double lj_scale = 1.0;
      double coulomb_scale = 1.0;
      special_scale(atom1, atom2, special_offsets, special_partners, special_lj,
                    special_coulomb, lj_scale, coulomb_scale);
      const int pair_index = type1 * parameters.types + type2;
      const LennardJonesParameters lj = lj_parameters[pair_index];
      if (lj.cutoff > 0.0 && atom_squared > 0.0 &&
          atom_squared < lj.cutoff * lj.cutoff) {
        const double r2inv = 1.0 / atom_squared;
        const double r6inv = r2inv * r2inv * r2inv;
        const double scalar =
            lj_scale * r6inv * (lj.lj1 * r6inv - lj.lj2) * r2inv;
        const double3 force =
            make_double3(atom_delta.x * scalar, atom_delta.y * scalar,
                         atom_delta.z * scalar);
        direct_force_x += force.x;
        direct_force_y += force.y;
        direct_force_z += force.z;
        if (tally) {
          lj_energy += lj_scale *
              (r6inv * (lj.lj3 * r6inv - lj.lj4) - lj.offset);
          virial_xx += atom_delta.x * force.x;
          virial_yy += atom_delta.y * force.y;
          virial_zz += atom_delta.z * force.z;
          virial_xy += atom_delta.x * force.y;
          virial_xz += atom_delta.x * force.z;
          virial_yz += atom_delta.y * force.z;
        }
      }

      if (!coulomb_type_pairs[pair_index]) continue;
      const double charge2 = charges[frame_offset + atom2];
      if (charge1 == 0.0 || charge2 == 0.0) continue;
      const double3 site_delta_fractional = minimum_fractional(make_double3(
          site1_fractional.x - site_fractional[frame_offset + atom2].x,
          site1_fractional.y - site_fractional[frame_offset + atom2].y,
          site1_fractional.z - site_fractional[frame_offset + atom2].z));
      const double3 site_delta = matrix_vector(parameters.h, site_delta_fractional);
      const double site_squared = site_delta.x * site_delta.x +
          site_delta.y * site_delta.y + site_delta.z * site_delta.z;
      if (!(site_squared > 0.0 && site_squared < parameters.real_cutoff_squared))
        continue;
      const double charge_product = charge1 * charge2;
      double force_value = 0.0;
      double energy_value = 0.0;
      if (parameters.table_bits > 0 &&
          site_squared > parameters.table_inner_squared) {
        lookup_table_coulomb(site_squared, charge_product, coulomb_scale, table_r,
                             table_dr, table_force, table_dforce, table_coulomb,
                             table_dcoulomb, table_energy, table_denergy, parameters,
                             force_value, energy_value);
      } else {
        const double distance = sqrt(site_squared);
        const double grij = parameters.g_ewald * distance;
        const double expm2 = exp(-grij * grij);
        const double t = 1.0 / (1.0 + kEwaldP * grij);
        const double erfc =
            t * (kEwaldA1 + t * (kEwaldA2 + t * (kEwaldA3 +
                                                  t * (kEwaldA4 + t * kEwaldA5)))) *
            expm2;
        const double prefactor = parameters.qqrd2e * charge_product / distance;
        force_value = prefactor * (erfc + kEwaldF * grij * expm2) -
            (1.0 - coulomb_scale) * prefactor;
        energy_value = prefactor * erfc - (1.0 - coulomb_scale) * prefactor;
      }
      const double scalar = force_value / site_squared;
      const double3 force = make_double3(site_delta.x * scalar, site_delta.y * scalar,
                                         site_delta.z * scalar);
      site_force_x += force.x;
      site_force_y += force.y;
      site_force_z += force.z;
      if (tally) {
        coulomb_energy += energy_value;
        virial_xx += site_delta.x * force.x;
        virial_yy += site_delta.y * force.y;
        virial_zz += site_delta.z * force.z;
        virial_xy += site_delta.x * force.y;
        virial_xz += site_delta.x * force.z;
        virial_yz += site_delta.y * force.z;
      }
  }

  direct_force_x = warp_sum(direct_force_x);
  direct_force_y = warp_sum(direct_force_y);
  direct_force_z = warp_sum(direct_force_z);
  site_force_x = warp_sum(site_force_x);
  site_force_y = warp_sum(site_force_y);
  site_force_z = warp_sum(site_force_z);
  lj_energy = warp_sum(lj_energy);
  coulomb_energy = warp_sum(coulomb_energy);
  virial_xx = warp_sum(virial_xx);
  virial_yy = warp_sum(virial_yy);
  virial_zz = warp_sum(virial_zz);
  virial_xy = warp_sum(virial_xy);
  virial_xz = warp_sum(virial_xz);
  virial_yz = warp_sum(virial_yz);

  if (lane == 0) {
    const std::size_t force_offset = 3 * frame_offset + 3 * atom1;
    atomicAdd(forces + force_offset, direct_force_x);
    atomicAdd(forces + force_offset + 1, direct_force_y);
    atomicAdd(forces + force_offset + 2, direct_force_z);
    add_site_force(frame, atom1,
                   make_double3(site_force_x, site_force_y, site_force_z),
                   forces, oxygen_site, tip4p_sites, parameters);
    atomicAdd(scalars + 8 * frame, lj_energy);
    atomicAdd(scalars + 8 * frame + 1, coulomb_energy);
    atomicAdd(scalars + 8 * frame + 2, virial_xx);
    atomicAdd(scalars + 8 * frame + 3, virial_yy);
    atomicAdd(scalars + 8 * frame + 4, virial_zz);
    atomicAdd(scalars + 8 * frame + 5, virial_xy);
    atomicAdd(scalars + 8 * frame + 6, virial_xz);
    atomicAdd(scalars + 8 * frame + 7, virial_yz);
  }
}

__global__ void assign_density_kernel(const double3 *site_fractional,
                                      const double *charges,
                                      const double *coefficients,
                                      cufftDoubleComplex *density,
                                      std::size_t total_atoms,
                                      DeviceParameters parameters) {
  const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const double3 fractional = site_fractional[global];
  const double grid[3] = {fractional.x * parameters.nx,
                          fractional.y * parameters.ny,
                          fractional.z * parameters.nz};
  int center[3];
  double delta[3];
  double weights[3][8]{};
  for (int dim = 0; dim < 3; ++dim) {
    center[dim] = static_cast<int>(grid[dim] + parameters.shift) - kGridOffset;
    delta[dim] = center[dim] + parameters.shift_one - grid[dim];
    for (int stencil = parameters.lower; stencil <= parameters.upper; ++stencil)
      weights[dim][stencil - parameters.lower] =
          spline_weight(coefficients, parameters, stencil, delta[dim]);
  }
  const std::size_t frame_grid =
      static_cast<std::size_t>(frame) * parameters.nx * parameters.ny * parameters.nz;
  const double charge = charges[global] * parameters.delvolinv;
  for (int iz = parameters.lower; iz <= parameters.upper; ++iz) {
    const int z = periodic_index(center[2] + iz, parameters.nz);
    for (int iy = parameters.lower; iy <= parameters.upper; ++iy) {
      const int y = periodic_index(center[1] + iy, parameters.ny);
      for (int ix = parameters.lower; ix <= parameters.upper; ++ix) {
        const int x = periodic_index(center[0] + ix, parameters.nx);
        const double value = charge * weights[0][ix - parameters.lower] *
            weights[1][iy - parameters.lower] *
            weights[2][iz - parameters.lower];
        atomicAdd(&density[frame_grid + grid_index(x, y, z, parameters)].x, value);
      }
    }
  }
}

__global__ void prepare_mm_spectrum_kernel(
    const cufftDoubleComplex *density, const double *green, const double *kvector,
    const double *virial_factor, cufftDoubleComplex *normalized,
    cufftDoubleComplex *transforms, double *raw, std::size_t total_grid,
    std::size_t mesh_count, int fields_per_frame) {
  const std::size_t blocks_per_frame =
      (mesh_count + kThreads - 1) / kThreads;
  const std::size_t frame = blockIdx.x / blocks_per_frame;
  const std::size_t grid =
      (blockIdx.x % blocks_per_frame) * kThreads + threadIdx.x;
  const std::size_t global = frame * mesh_count + grid;
  RawContribution raw_contribution;
  if (global < total_grid && grid < mesh_count) {
    const cufftDoubleComplex rho = density[global];
    const double scale_inverse = 1.0 / static_cast<double>(mesh_count);
    const double spectral_scale = scale_inverse * green[grid];
    const cufftDoubleComplex potential =
        make_cuDoubleComplex(rho.x * spectral_scale, rho.y * spectral_scale);
    normalized[global] = potential;
    if (fields_per_frame > 0) {
      const int gradient_offset = fields_per_frame == 4 ? 1 : 0;
      if (fields_per_frame == 4)
        transforms[(4 * frame) * mesh_count + grid] = potential;
      for (int dim = 0; dim < 3; ++dim) {
        const double k = kvector[3 * grid + dim];
        transforms[(fields_per_frame * frame + gradient_offset + dim) *
                       mesh_count +
                   grid] =
            make_cuDoubleComplex(-k * potential.y, k * potential.x);
      }
    }
    const double contribution =
        scale_inverse * scale_inverse * green[grid] *
        (rho.x * rho.x + rho.y * rho.y);
    raw_contribution.values[0] = contribution;
    for (int component = 0; component < 6; ++component)
      raw_contribution.values[1 + component] =
          contribution * virial_factor[6 * grid + component];
  }
  publish_raw_block(raw_contribution, raw, frame);
}

__global__ void prepare_qm_spectrum_kernel(
    const cufftDoubleComplex *density, const double *green, const double *kvector,
    const double *virial_factor, cufftDoubleComplex *normalized,
    cufftDoubleComplex *transforms, double *raw, std::size_t total_grid,
    std::size_t mesh_count) {
  const std::size_t blocks_per_frame =
      (mesh_count + kThreads - 1) / kThreads;
  const std::size_t frame = blockIdx.x / blocks_per_frame;
  const std::size_t grid =
      (blockIdx.x % blocks_per_frame) * kThreads + threadIdx.x;
  const std::size_t global = frame * mesh_count + grid;
  RawContribution raw_contribution;
  if (global < total_grid && grid < mesh_count) {
    const cufftDoubleComplex rho = density[global];
    const double scale_inverse = 1.0 / static_cast<double>(mesh_count);
    const double spectral_scale = scale_inverse * green[grid];
    const cufftDoubleComplex potential =
        make_cuDoubleComplex(rho.x * spectral_scale, rho.y * spectral_scale);
    normalized[global] = potential;
    for (int dim = 0; dim < 3; ++dim) {
      const double k = kvector[3 * grid + dim];
      transforms[(3 * frame + dim) * mesh_count + grid] =
          make_cuDoubleComplex(-k * potential.y, k * potential.x);
    }
    const double contribution =
        scale_inverse * scale_inverse * green[grid] *
        (rho.x * rho.x + rho.y * rho.y);
    raw_contribution.values[0] = contribution;
    for (int component = 0; component < 6; ++component)
      raw_contribution.values[1 + component] =
          contribution * virial_factor[6 * grid + component];
  }
  publish_raw_block(raw_contribution, raw, frame);
}

__global__ void interpolate_mm_kernel(
    const double3 *site_fractional, const double *charges,
    const double *coefficients, const cufftDoubleComplex *mm_transforms,
    double *potential, double *forces, const std::int32_t *oxygen_site,
    const Tip4pSite *tip4p_sites, std::size_t total_atoms,
    std::size_t mesh_count, int fields_per_frame,
    DeviceParameters parameters) {
  const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const double3 fractional = site_fractional[global];
  const double grid[3] = {fractional.x * parameters.nx,
                          fractional.y * parameters.ny,
                          fractional.z * parameters.nz};
  int center[3];
  double weights[3][8]{};
  for (int dim = 0; dim < 3; ++dim) {
    center[dim] = static_cast<int>(grid[dim] + parameters.shift) - kGridOffset;
    const double delta = center[dim] + parameters.shift_one - grid[dim];
    for (int stencil = parameters.lower; stencil <= parameters.upper; ++stencil)
      weights[dim][stencil - parameters.lower] =
          spline_weight(coefficients, parameters, stencil, delta);
  }
  double value = 0.0;
  double3 electric = make_double3(0.0, 0.0, 0.0);
  const int gradient_offset = fields_per_frame == 4 ? 1 : 0;
  for (int iz = parameters.lower; iz <= parameters.upper; ++iz) {
    const int z = periodic_index(center[2] + iz, parameters.nz);
    for (int iy = parameters.lower; iy <= parameters.upper; ++iy) {
      const int y = periodic_index(center[1] + iy, parameters.ny);
      for (int ix = parameters.lower; ix <= parameters.upper; ++ix) {
        const int x = periodic_index(center[0] + ix, parameters.nx);
        const double weight = weights[0][ix - parameters.lower] *
            weights[1][iy - parameters.lower] *
            weights[2][iz - parameters.lower];
        const std::size_t mesh = grid_index(x, y, z, parameters);
        if (potential)
          value += weight *
              mm_transforms[(fields_per_frame *
                                 static_cast<std::size_t>(frame)) *
                                mesh_count +
                            mesh]
                  .x;
        if (forces) {
          double *components[3] = {&electric.x, &electric.y, &electric.z};
          for (int dim = 0; dim < 3; ++dim) {
            const double gradient =
                mm_transforms[(fields_per_frame *
                                   static_cast<std::size_t>(frame) +
                               gradient_offset + dim) *
                                      mesh_count +
                                  mesh]
                    .x;
            *components[dim] -= weight * gradient;
          }
        }
      }
    }
  }
  if (potential) potential[global] = value;
  if (forces) {
    const int atom = static_cast<int>(global % parameters.atoms);
    const double factor = parameters.qqrd2e * charges[global];
    add_site_force(frame, atom,
                   make_double3(factor * electric.x, factor * electric.y,
                                factor * electric.z),
                   forces, oxygen_site, tip4p_sites, parameters);
  }
}

__global__ void finalize_subset_kernel(const double *raw, const double *qsum,
                                       const double *qsq, double *energy,
                                       double *virial, int batch,
                                       DeviceParameters parameters) {
  const int frame = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame >= batch) return;
  energy[frame] =
      (0.5 * parameters.volume * raw[7 * frame] -
       parameters.g_ewald * qsq[frame] * kInvSqrtPi -
       kHalfPi * qsum[frame] * qsum[frame] /
           (parameters.g_ewald * parameters.g_ewald * parameters.volume)) *
      parameters.qqrd2e;
  for (int component = 0; component < 6; ++component)
    virial[6 * frame + component] =
        0.5 * parameters.qqrd2e * parameters.volume *
        raw[7 * frame + 1 + component];
}

__global__ void make_full_charges_kernel(const double *mm, const double *qm,
                                         double *full, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) full[index] = mm[index] + qm[index];
}

__global__ void interpolate_qm_full_kernel(
    const double3 *site_fractional, const double *qm_charges,
    const double *full_charges, const double *coefficients,
    const cufftDoubleComplex *mm_transforms,
    const cufftDoubleComplex *qm_transforms, double *qm_forces,
    double *full_forces, const std::int32_t *oxygen_site,
    const Tip4pSite *tip4p_sites, std::size_t total_atoms,
    std::size_t mesh_count, DeviceParameters parameters) {
  const std::size_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= total_atoms) return;
  const int frame = static_cast<int>(global / parameters.atoms);
  const int atom = static_cast<int>(global % parameters.atoms);
  const double3 fractional = site_fractional[global];
  const double grid[3] = {fractional.x * parameters.nx,
                          fractional.y * parameters.ny,
                          fractional.z * parameters.nz};
  int center[3];
  double weights[3][8]{};
  for (int dim = 0; dim < 3; ++dim) {
    center[dim] = static_cast<int>(grid[dim] + parameters.shift) - kGridOffset;
    const double delta = center[dim] + parameters.shift_one - grid[dim];
    for (int stencil = parameters.lower; stencil <= parameters.upper; ++stencil)
      weights[dim][stencil - parameters.lower] =
          spline_weight(coefficients, parameters, stencil, delta);
  }
  double3 qm_electric = make_double3(0.0, 0.0, 0.0);
  double3 full_electric = make_double3(0.0, 0.0, 0.0);
  for (int iz = parameters.lower; iz <= parameters.upper; ++iz) {
    const int z = periodic_index(center[2] + iz, parameters.nz);
    for (int iy = parameters.lower; iy <= parameters.upper; ++iy) {
      const int y = periodic_index(center[1] + iy, parameters.ny);
      for (int ix = parameters.lower; ix <= parameters.upper; ++ix) {
        const int x = periodic_index(center[0] + ix, parameters.nx);
        const double weight = weights[0][ix - parameters.lower] *
            weights[1][iy - parameters.lower] *
            weights[2][iz - parameters.lower];
        const std::size_t grid = grid_index(x, y, z, parameters);
        double *qm_components[3] = {&qm_electric.x, &qm_electric.y,
                                    &qm_electric.z};
        double *full_components[3] = {&full_electric.x, &full_electric.y,
                                      &full_electric.z};
        for (int dim = 0; dim < 3; ++dim) {
          const double qm_gradient =
              qm_transforms[(3 * static_cast<std::size_t>(frame) + dim) *
                                mesh_count +
                            grid]
                  .x;
          const double mm_gradient =
              mm_transforms[(4 * static_cast<std::size_t>(frame) + 1 + dim) *
                                mesh_count +
                            grid]
                  .x;
          *qm_components[dim] -= weight * qm_gradient;
          *full_components[dim] -= weight * (mm_gradient + qm_gradient);
        }
      }
    }
  }
  const double qm_factor = parameters.qqrd2e * qm_charges[global];
  const double full_factor = parameters.qqrd2e * full_charges[global];
  add_site_force(frame, atom,
                 make_double3(qm_factor * qm_electric.x,
                              qm_factor * qm_electric.y,
                              qm_factor * qm_electric.z),
                 qm_forces, oxygen_site, tip4p_sites, parameters);
  add_site_force(frame, atom,
                 make_double3(full_factor * full_electric.x,
                              full_factor * full_electric.y,
                              full_factor * full_electric.z),
                 full_forces, oxygen_site, tip4p_sites, parameters);
}

__global__ void cross_spectrum_kernel(const cufftDoubleComplex *mm,
                                      const cufftDoubleComplex *qm,
                                      const double *green,
                                      const double *virial_factor, double *cross,
                                      std::size_t total_grid,
                                      std::size_t mesh_count) {
  const std::size_t blocks_per_frame =
      (mesh_count + kThreads - 1) / kThreads;
  const std::size_t frame = blockIdx.x / blocks_per_frame;
  const std::size_t grid =
      (blockIdx.x % blocks_per_frame) * kThreads + threadIdx.x;
  const std::size_t global = frame * mesh_count + grid;
  RawContribution raw_contribution;
  if (global < total_grid && grid < mesh_count && green[grid] != 0.0) {
    const double product =
        (mm[global].x * qm[global].x + mm[global].y * qm[global].y) /
        green[grid];
    raw_contribution.values[0] = product;
    for (int component = 0; component < 6; ++component)
      raw_contribution.values[1 + component] =
          product * virial_factor[6 * grid + component];
  }
  publish_raw_block(raw_contribution, cross, frame);
}

__global__ void finalize_full_kernel(
    const double *mm_energy, const double *mm_virial, const double *qm_energy,
    const double *qm_virial, const double *mm_qsum, const double *qm_qsum,
    const double *cross, double *full_energy, double *full_virial, int batch,
    DeviceParameters parameters) {
  const int frame = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame >= batch) return;
  const double cross_energy =
      parameters.volume * parameters.qqrd2e * cross[7 * frame] -
      kPi * mm_qsum[frame] * qm_qsum[frame] /
          (parameters.g_ewald * parameters.g_ewald * parameters.volume) *
          parameters.qqrd2e;
  full_energy[frame] = mm_energy[frame] + qm_energy[frame] + cross_energy;
  for (int component = 0; component < 6; ++component)
    full_virial[6 * frame + component] =
        mm_virial[6 * frame + component] + qm_virial[6 * frame + component] +
        parameters.volume * parameters.qqrd2e *
            cross[7 * frame + 1 + component];
}

class CudaClassicalBatchPlan final : public ClassicalBatchPlan {
 public:
  CudaClassicalBatchPlan(ClassicalTopology topology,
                         const ClassicalPlanOptions &options)
      : topology_(std::move(topology)), prepared_(prepare_classical_data(topology_)),
        max_batch_count_(options.max_batch_count) {
    if (max_batch_count_ == 0)
      throw std::invalid_argument("classical CUDA max batch count must be positive");
    const std::size_t int_max =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (topology_.atom_count > int_max || prepared_.mesh_count > int_max ||
        max_batch_count_ > int_max / 4 ||
        max_batch_count_ > int_max / topology_.atom_count ||
        max_batch_count_ > int_max / prepared_.mesh_count)
      throw std::overflow_error("classical CUDA topology exceeds kernel index limits");
    int previous_device = 0;
    check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice");
    device_ = options.cuda_device >= 0 ? options.cuda_device : previous_device;
    ScopedDevice device_guard(device_);
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreate classical plan");
    initialize_parameters();
    allocate_topology();
    allocate_workspace();
  }

  ~CudaClassicalBatchPlan() override {
    int previous_device = -1;
    const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    const bool switched = have_previous && previous_device != device_ &&
        cudaSetDevice(device_) == cudaSuccess;
    plans_.clear();
    if (stream_) cudaStreamDestroy(stream_);
    stream_ = nullptr;
    if (switched) cudaSetDevice(previous_device);
  }

  [[nodiscard]] ClassicalBackend backend() const noexcept override {
    return ClassicalBackend::CUDA;
  }

  [[nodiscard]] std::size_t max_batch_count() const noexcept override {
    return max_batch_count_;
  }

  void begin_mm(const ClassicalBatchInput &input,
                const ClassicalMmBatchOutput &output) override {
    validate_begin(input, output);
    if (active_) throw std::logic_error("a classical CUDA MM epoch is already active");
    ScopedDevice device_guard(device_);
    const std::size_t atoms = input.batch_count * topology_.atom_count;
    const std::size_t coordinates = 3 * atoms;
    const std::size_t grid = input.batch_count * prepared_.mesh_count;
    const int global_bins = static_cast<int>(input.batch_count) * parameters_.bins;
    FftPlans &fft = plans_for(input.batch_count);
    const bool rebuild_bins = neighbor_rebuild_required(input);
    const bool need_potential = output.mm_pppm_potential != nullptr;
    const bool need_forces = output.mm_pppm_forces != nullptr;
    const int mm_fields =
        output.retain_for_qm ? 4 : (need_forces ? 3 : (need_potential ? 4 : 0));

    check_cuda(cudaMemcpyAsync(d_positions_.get(), input.positions,
                               coordinates * sizeof(double), cudaMemcpyHostToDevice,
                               stream_),
               "copy classical positions");
    check_cuda(cudaMemcpyAsync(d_mm_charges_.get(), input.charges,
                               atoms * sizeof(double), cudaMemcpyHostToDevice, stream_),
               "copy classical MM charges");
    check_cuda(cudaMemsetAsync(d_pair_forces_.get(), 0, coordinates * sizeof(double),
                               stream_),
               "clear pair forces");
    check_cuda(cudaMemsetAsync(d_pair_scalars_.get(), 0,
                               8 * input.batch_count * sizeof(double), stream_),
               "clear pair scalars");
    if (rebuild_bins)
      check_cuda(cudaMemsetAsync(
                     d_bin_counts_.get(), 0,
                     (static_cast<std::size_t>(global_bins) + 1) * sizeof(int),
                     stream_),
                 "clear bin counts");
    check_cuda(cudaMemsetAsync(d_density_.get(), 0,
                               grid * sizeof(cufftDoubleComplex), stream_),
               "clear MM density");
    check_cuda(cudaMemsetAsync(d_mm_raw_.get(), 0,
                               7 * input.batch_count * sizeof(double), stream_),
               "clear MM reductions");
    if (need_forces)
      check_cuda(cudaMemsetAsync(d_mm_forces_.get(), 0,
                                 coordinates * sizeof(double), stream_),
                 "clear MM reciprocal forces");

    fractional_and_bins_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
        d_positions_.get(), d_atom_fractional_.get(), d_site_fractional_.get(),
        d_atom_bins_.get(), d_bin_counts_.get(), atoms, rebuild_bins,
        parameters_);
    launch_check("fractional_and_bins_kernel");
    if (parameters_.tip4p_sites > 0) {
      const int sites = static_cast<int>(input.batch_count) * parameters_.tip4p_sites;
      tip4p_sites_kernel<<<blocks(sites), kThreads, 0, stream_>>>(
          d_site_fractional_.get(), d_atom_fractional_.get(), d_tip4p_sites_.get(),
          static_cast<int>(input.batch_count), parameters_);
      launch_check("tip4p_sites_kernel");
    }
    if (rebuild_bins) {
      std::size_t scan_bytes = scan_workspace_bytes_;
      check_cuda(cub::DeviceScan::ExclusiveSum(
                     d_scan_workspace_.get(), scan_bytes, d_bin_counts_.get(),
                     d_bin_offsets_.get(), global_bins + 1, stream_),
                 "CUB exclusive bin scan");
      check_cuda(cudaMemcpyAsync(
                     d_bin_cursor_.get(), d_bin_offsets_.get(),
                     static_cast<std::size_t>(global_bins) * sizeof(int),
                     cudaMemcpyDeviceToDevice, stream_),
                 "initialize bin cursors");
      fill_bins_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
          d_atom_bins_.get(), d_bin_cursor_.get(), d_bin_atoms_.get(), atoms,
          parameters_);
      launch_check("fill_bins_kernel");
      rebuild_verlet_list(atoms);
    }

    const int real_space_blocks = static_cast<int>(
        (atoms + kRealSpaceWarpsPerBlock - 1) / kRealSpaceWarpsPerBlock);
    real_space_kernel<<<real_space_blocks, kRealSpaceThreads, 0, stream_>>>(
        d_atom_fractional_.get(), d_site_fractional_.get(), d_mm_charges_.get(),
        d_atom_types_.get(), d_lj_.get(), d_coulomb_type_pairs_.get(),
        d_verlet_offsets_.get(), d_verlet_neighbors_.get(),
        d_special_offsets_.get(), d_special_partners_.get(), d_special_lj_.get(),
        d_special_coulomb_.get(), d_oxygen_site_.get(), d_tip4p_sites_.get(),
        d_table_r_.get(), d_table_dr_.get(), d_table_force_.get(),
        d_table_dforce_.get(), d_table_coulomb_.get(), d_table_dcoulomb_.get(),
        d_table_energy_.get(), d_table_denergy_.get(), d_pair_forces_.get(),
        d_pair_scalars_.get(), atoms, parameters_);
    launch_check("real_space_kernel");

    assign_density_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
        d_site_fractional_.get(), d_mm_charges_.get(), d_spline_.get(),
        d_density_.get(), atoms, parameters_);
    launch_check("assign_density_kernel MM");
    check_cufft(cufftExecZ2Z(fft.forward, d_density_.get(), d_density_.get(),
                             CUFFT_FORWARD),
                "cuFFT MM forward");
    const int mm_spectrum_blocks =
        static_cast<int>(input.batch_count) * blocks(prepared_.mesh_count);
    prepare_mm_spectrum_kernel<<<mm_spectrum_blocks, kThreads, 0, stream_>>>(
        d_density_.get(), d_green_.get(), d_kvector_.get(), d_virial_factor_.get(),
        d_mm_spectrum_.get(), d_mm_transforms_.get(), d_mm_raw_.get(), grid,
        prepared_.mesh_count, mm_fields);
    launch_check("prepare_mm_spectrum_kernel");
    if (mm_fields > 0) {
      cufftHandle inverse = mm_fields == 4 ? fft.inverse_mm : fft.inverse_qm;
      check_cufft(cufftExecZ2Z(inverse, d_mm_transforms_.get(),
                               d_mm_transforms_.get(), CUFFT_INVERSE),
                  mm_fields == 4 ? "cuFFT MM scalar/field inverse batch"
                                 : "cuFFT MM field inverse batch");
      interpolate_mm_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
          d_site_fractional_.get(), d_mm_charges_.get(), d_spline_.get(),
          d_mm_transforms_.get(), need_potential ? d_mm_potential_.get() : nullptr,
          need_forces ? d_mm_forces_.get() : nullptr, d_oxygen_site_.get(),
          d_tip4p_sites_.get(), atoms, prepared_.mesh_count, mm_fields,
          parameters_);
      launch_check("interpolate_mm_kernel");
    }

    compute_charge_moments(input.charges, input.batch_count, mm_qsum_host_,
                           mm_qsq_host_);
    copy_moments_to_device(mm_qsum_host_, mm_qsq_host_, d_mm_qsum_, d_mm_qsq_);
    finalize_subset_kernel<<<blocks(input.batch_count), kThreads, 0, stream_>>>(
        d_mm_raw_.get(), d_mm_qsum_.get(), d_mm_qsq_.get(), d_mm_energy_.get(),
        d_mm_virial_.get(), static_cast<int>(input.batch_count), parameters_);
    launch_check("finalize MM PPPM");

    stage_mm_outputs(input.batch_count, coordinates, atoms, need_potential,
                     need_forces);
    check_cuda(cudaStreamSynchronize(stream_), "synchronize classical MM stage");
    if (rebuild_bins) {
      // Commit the host-side displacement epoch only after every device use of
      // the rebuilt list has completed.  A failed CUDA stage therefore forces
      // the next begin_mm() to rebuild instead of trusting partial bin state.
      std::copy_n(input.positions, coordinates, neighbor_reference_positions_.begin());
      neighbor_batch_count_ = input.batch_count;
      neighbor_reference_valid_ = true;
    }
    publish_mm(output, input.batch_count, coordinates, atoms);
    if (output.retain_for_qm) {
      active_ = true;
      cached_batch_count_ = input.batch_count;
      mm_transform_fields_ = mm_fields;
    } else {
      // Terminal pure-MM publication completes successfully at this point;
      // retained device buffers may be reused but no QM continuation exists.
      cancel();
    }
  }

  void finish_qm(const ClassicalQmBatchInput &input,
                 const ClassicalQmBatchOutput &output) override {
    validate_finish(input, output);
    if (!active_) throw std::logic_error("no classical CUDA MM epoch is active");
    if (input.batch_count != cached_batch_count_)
      throw std::invalid_argument("QM batch count does not match active CUDA MM epoch");
    if (mm_transform_fields_ != 4)
      throw std::logic_error(
          "active CUDA MM epoch lacks retained QM/MM scalar/field transforms");
    ScopedDevice device_guard(device_);
    const std::size_t atoms = input.batch_count * topology_.atom_count;
    const std::size_t coordinates = 3 * atoms;
    const std::size_t grid = input.batch_count * prepared_.mesh_count;
    FftPlans &fft = plans_for(input.batch_count);

    check_cuda(cudaMemcpyAsync(d_qm_charges_.get(), input.qm_charges,
                               atoms * sizeof(double), cudaMemcpyHostToDevice, stream_),
               "copy classical QM charges");
    make_full_charges_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
        d_mm_charges_.get(), d_qm_charges_.get(), d_full_charges_.get(), atoms);
    launch_check("make_full_charges_kernel");
    check_cuda(cudaMemsetAsync(d_density_.get(), 0,
                               grid * sizeof(cufftDoubleComplex), stream_),
               "clear QM density");
    check_cuda(cudaMemsetAsync(d_qm_raw_.get(), 0,
                               7 * input.batch_count * sizeof(double), stream_),
               "clear QM reductions");
    check_cuda(cudaMemsetAsync(d_cross_.get(), 0,
                               7 * input.batch_count * sizeof(double), stream_),
               "clear cross reductions");
    check_cuda(cudaMemsetAsync(d_qm_forces_.get(), 0, coordinates * sizeof(double),
                               stream_),
               "clear QM forces");
    check_cuda(cudaMemsetAsync(d_full_forces_.get(), 0,
                               coordinates * sizeof(double), stream_),
               "clear full forces");

    assign_density_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
        d_site_fractional_.get(), d_qm_charges_.get(), d_spline_.get(),
        d_density_.get(), atoms, parameters_);
    launch_check("assign_density_kernel QM");
    check_cufft(cufftExecZ2Z(fft.forward, d_density_.get(), d_density_.get(),
                             CUFFT_FORWARD),
                "cuFFT QM forward");
    const int qm_spectrum_blocks =
        static_cast<int>(input.batch_count) * blocks(prepared_.mesh_count);
    prepare_qm_spectrum_kernel<<<qm_spectrum_blocks, kThreads, 0, stream_>>>(
        d_density_.get(), d_green_.get(), d_kvector_.get(), d_virial_factor_.get(),
        d_qm_spectrum_.get(), d_qm_transforms_.get(), d_qm_raw_.get(), grid,
        prepared_.mesh_count);
    launch_check("prepare_qm_spectrum_kernel");
    check_cufft(cufftExecZ2Z(fft.inverse_qm, d_qm_transforms_.get(),
                             d_qm_transforms_.get(), CUFFT_INVERSE),
                "cuFFT QM inverse batch");
    interpolate_qm_full_kernel<<<blocks(atoms), kThreads, 0, stream_>>>(
        d_site_fractional_.get(), d_qm_charges_.get(), d_full_charges_.get(),
        d_spline_.get(), d_mm_transforms_.get(), d_qm_transforms_.get(),
        d_qm_forces_.get(), d_full_forces_.get(), d_oxygen_site_.get(),
        d_tip4p_sites_.get(), atoms, prepared_.mesh_count, parameters_);
    launch_check("interpolate_qm_full_kernel");
    cross_spectrum_kernel<<<qm_spectrum_blocks, kThreads, 0, stream_>>>(
        d_mm_spectrum_.get(), d_qm_spectrum_.get(), d_green_.get(),
        d_virial_factor_.get(), d_cross_.get(), grid, prepared_.mesh_count);
    launch_check("cross_spectrum_kernel");

    compute_charge_moments(input.qm_charges, input.batch_count, qm_qsum_host_,
                           qm_qsq_host_);
    copy_moments_to_device(qm_qsum_host_, qm_qsq_host_, d_qm_qsum_, d_qm_qsq_);
    finalize_subset_kernel<<<blocks(input.batch_count), kThreads, 0, stream_>>>(
        d_qm_raw_.get(), d_qm_qsum_.get(), d_qm_qsq_.get(), d_qm_energy_.get(),
        d_qm_virial_.get(), static_cast<int>(input.batch_count), parameters_);
    launch_check("finalize QM PPPM");
    finalize_full_kernel<<<blocks(input.batch_count), kThreads, 0, stream_>>>(
        d_mm_energy_.get(), d_mm_virial_.get(), d_qm_energy_.get(),
        d_qm_virial_.get(), d_mm_qsum_.get(), d_qm_qsum_.get(), d_cross_.get(),
        d_full_energy_.get(), d_full_virial_.get(),
        static_cast<int>(input.batch_count), parameters_);
    launch_check("finalize full PPPM");

    stage_qm_outputs(input.batch_count, coordinates);
    check_cuda(cudaStreamSynchronize(stream_), "synchronize classical QM stage");
    publish_qm(output, input.batch_count, coordinates);
    cancel();
  }

  void cancel() noexcept override {
    active_ = false;
    cached_batch_count_ = 0;
    mm_transform_fields_ = 0;
  }

 private:
  static int blocks(std::size_t count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
  }

  void launch_check(const char *operation) {
    check_cuda(cudaGetLastError(), operation);
  }

  void initialize_parameters() {
    parameters_.atoms = static_cast<int>(topology_.atom_count);
    parameters_.types = topology_.type_count;
    parameters_.tip4p_sites = static_cast<int>(topology_.tip4p_sites.size());
    parameters_.nx = prepared_.mesh[0];
    parameters_.ny = prepared_.mesh[1];
    parameters_.nz = prepared_.mesh[2];
    parameters_.order = topology_.pppm.order;
    parameters_.lower = prepared_.spline_lower;
    parameters_.upper = prepared_.spline_upper;
    parameters_.bins_x = prepared_.bin_count[0];
    parameters_.bins_y = prepared_.bin_count[1];
    parameters_.bins_z = prepared_.bin_count[2];
    parameters_.bins = parameters_.bins_x * parameters_.bins_y * parameters_.bins_z;
    parameters_.table_bits = topology_.coulomb_table.bits;
    parameters_.table_shift = topology_.coulomb_table.shift_bits;
    parameters_.table_mask = topology_.coulomb_table.mask;
    parameters_.table_inner_squared = topology_.coulomb_table.inner_squared;
    parameters_.shift = prepared_.spline_shift;
    parameters_.shift_one = prepared_.spline_shift_one;
    parameters_.alpha = topology_.tip4p_alpha;
    parameters_.real_cutoff_squared =
        topology_.real_space_cutoff * topology_.real_space_cutoff;
    parameters_.neighbor_cutoff_squared =
        prepared_.neighbor_cutoff * prepared_.neighbor_cutoff;
    parameters_.g_ewald = topology_.pppm.g_ewald;
    parameters_.qqrd2e = topology_.qqrd2e;
    parameters_.volume = prepared_.volume;
    parameters_.delvolinv = prepared_.delvolinv;
    std::copy(topology_.cell.h.begin(), topology_.cell.h.end(), parameters_.h);
    std::copy(prepared_.hinv.begin(), prepared_.hinv.end(), parameters_.hinv);
    std::copy(topology_.cell.boxlo.begin(), topology_.cell.boxlo.end(),
              parameters_.boxlo);
  }

  [[nodiscard]] bool
  neighbor_rebuild_required(const ClassicalBatchInput &input) const noexcept {
    if (!neighbor_reference_valid_ || input.batch_count != neighbor_batch_count_ ||
        !(topology_.neighbor_skin > 0.0))
      return true;

    const double threshold_squared =
        0.25 * topology_.neighbor_skin * topology_.neighbor_skin;
    const std::size_t atoms = input.batch_count * topology_.atom_count;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      const std::size_t offset = 3 * atom;
      const double dx = input.positions[offset] - neighbor_reference_positions_[offset];
      const double dy =
          input.positions[offset + 1] - neighbor_reference_positions_[offset + 1];
      const double dz =
          input.positions[offset + 2] - neighbor_reference_positions_[offset + 2];
      double sx = prepared_.hinv[0] * dx + prepared_.hinv[1] * dy +
          prepared_.hinv[2] * dz;
      double sy = prepared_.hinv[3] * dx + prepared_.hinv[4] * dy +
          prepared_.hinv[5] * dz;
      double sz = prepared_.hinv[6] * dx + prepared_.hinv[7] * dy +
          prepared_.hinv[8] * dz;
      sx -= std::nearbyint(sx);
      sy -= std::nearbyint(sy);
      sz -= std::nearbyint(sz);
      const double rx = topology_.cell.h[0] * sx + topology_.cell.h[1] * sy +
          topology_.cell.h[2] * sz;
      const double ry = topology_.cell.h[3] * sx + topology_.cell.h[4] * sy +
          topology_.cell.h[5] * sz;
      const double rz = topology_.cell.h[6] * sx + topology_.cell.h[7] * sy +
          topology_.cell.h[8] * sz;
      if (rx * rx + ry * ry + rz * rz > threshold_squared) return true;
    }
    return false;
  }

  template <class T>
  void upload(DeviceBuffer<T> &device, const std::vector<T> &host,
              const char *operation) {
    device.allocate(host.size(), operation);
    if (!host.empty())
      check_cuda(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 operation);
  }

  void allocate_topology() {
    upload(d_atom_types_, topology_.atom_types, "copy atom types");
    upload(d_tip4p_sites_, topology_.tip4p_sites, "copy TIP4P sites");
    upload(d_oxygen_site_, prepared_.oxygen_site, "copy oxygen-site map");
    upload(d_lj_, topology_.lj, "copy LJ parameters");
    upload(d_coulomb_type_pairs_, topology_.coulomb_type_pairs,
           "copy Coulomb type mapping");
    upload(d_special_offsets_, prepared_.special_offsets, "copy special offsets");
    upload(d_special_partners_, prepared_.special_partners, "copy special partners");
    upload(d_special_lj_, prepared_.special_lj, "copy special LJ scales");
    upload(d_special_coulomb_, prepared_.special_coulomb,
           "copy special Coulomb scales");
    upload(d_neighbor_bin_offsets_, prepared_.neighbor_bin_offsets,
           "copy neighbor-bin offsets");
    upload(d_neighbor_bins_, prepared_.neighbor_bins, "copy neighbor bins");
    upload(d_spline_, prepared_.spline_coefficients, "copy spline coefficients");
    upload(d_green_, prepared_.green, "copy PPPM influence function");
    upload(d_kvector_, prepared_.kvector, "copy reciprocal vectors");
    upload(d_virial_factor_, prepared_.virial_factor,
           "copy reciprocal virial factors");
    upload(d_table_r_, topology_.coulomb_table.r, "copy Coulomb table r");
    upload(d_table_dr_, topology_.coulomb_table.dr, "copy Coulomb table dr");
    upload(d_table_force_, topology_.coulomb_table.force,
           "copy Coulomb force table");
    upload(d_table_dforce_, topology_.coulomb_table.dforce,
           "copy Coulomb dforce table");
    upload(d_table_coulomb_, topology_.coulomb_table.coulomb,
           "copy Coulomb complement table");
    upload(d_table_dcoulomb_, topology_.coulomb_table.dcoulomb,
           "copy Coulomb dcomplement table");
    upload(d_table_energy_, topology_.coulomb_table.energy,
           "copy Coulomb energy table");
    upload(d_table_denergy_, topology_.coulomb_table.denergy,
           "copy Coulomb denergy table");
  }

  void allocate_workspace() {
    const std::size_t atoms = max_batch_count_ * topology_.atom_count;
    const std::size_t coordinates = 3 * atoms;
    const std::size_t grid = max_batch_count_ * prepared_.mesh_count;
    const std::size_t bins = max_batch_count_ * parameters_.bins;
    d_positions_.allocate(coordinates, "positions");
    d_mm_charges_.allocate(atoms, "MM charges");
    d_qm_charges_.allocate(atoms, "QM charges");
    d_full_charges_.allocate(atoms, "full charges");
    d_atom_fractional_.allocate(atoms, "atom fractional coordinates");
    d_site_fractional_.allocate(atoms, "charge-site fractional coordinates");
    d_atom_bins_.allocate(atoms, "atom bins");
    d_bin_counts_.allocate(bins + 1, "bin counts");
    d_bin_offsets_.allocate(bins + 1, "bin offsets");
    d_bin_cursor_.allocate(bins, "bin cursors");
    d_bin_atoms_.allocate(atoms, "binned atoms");
    d_verlet_counts_.allocate(atoms + 1, "Verlet counts");
    d_verlet_offsets_.allocate(atoms + 1, "Verlet offsets");
    d_verlet_cursor_.allocate(atoms, "Verlet cursors");
    d_pair_forces_.allocate(coordinates, "pair forces");
    d_mm_forces_.allocate(coordinates, "MM reciprocal forces");
    d_qm_forces_.allocate(coordinates, "QM reciprocal forces");
    d_full_forces_.allocate(coordinates, "full reciprocal forces");
    d_pair_scalars_.allocate(8 * max_batch_count_, "pair scalars");
    d_density_.allocate(grid, "PPPM density");
    d_mm_spectrum_.allocate(grid, "MM spectrum");
    d_qm_spectrum_.allocate(grid, "QM spectrum");
    d_mm_transforms_.allocate(4 * grid, "MM scalar and field transforms");
    d_qm_transforms_.allocate(3 * grid, "QM field transforms");
    d_mm_potential_.allocate(atoms, "MM atom potentials");
    d_mm_raw_.allocate(7 * max_batch_count_, "MM reciprocal reductions");
    d_qm_raw_.allocate(7 * max_batch_count_, "QM reciprocal reductions");
    d_cross_.allocate(7 * max_batch_count_, "MM-QM cross reductions");
    d_mm_qsum_.allocate(max_batch_count_, "MM charge sums");
    d_mm_qsq_.allocate(max_batch_count_, "MM squared-charge sums");
    d_qm_qsum_.allocate(max_batch_count_, "QM charge sums");
    d_qm_qsq_.allocate(max_batch_count_, "QM squared-charge sums");
    d_mm_energy_.allocate(max_batch_count_, "MM reciprocal energies");
    d_qm_energy_.allocate(max_batch_count_, "QM reciprocal energies");
    d_full_energy_.allocate(max_batch_count_, "full reciprocal energies");
    d_mm_virial_.allocate(6 * max_batch_count_, "MM reciprocal virials");
    d_qm_virial_.allocate(6 * max_batch_count_, "QM reciprocal virials");
    d_full_virial_.allocate(6 * max_batch_count_, "full reciprocal virials");
    check_cuda(cub::DeviceScan::ExclusiveSum(
                   nullptr, scan_workspace_bytes_, d_bin_counts_.get(),
                   d_bin_offsets_.get(), static_cast<int>(bins + 1), stream_),
               "query CUB scan workspace");
    d_scan_workspace_.allocate(scan_workspace_bytes_, "CUB bin-scan workspace");
    check_cuda(cub::DeviceScan::ExclusiveSum(
                   nullptr, verlet_scan_workspace_bytes_,
                   d_verlet_counts_.get(), d_verlet_offsets_.get(), atoms + 1,
                   stream_),
               "query CUB Verlet scan workspace");
    d_verlet_scan_workspace_.allocate(verlet_scan_workspace_bytes_,
                                      "CUB Verlet-scan workspace");

    host_pair_forces_.allocate(coordinates);
    host_mm_forces_.allocate(coordinates);
    host_qm_forces_.allocate(coordinates);
    host_full_forces_.allocate(coordinates);
    host_pair_scalars_.allocate(8 * max_batch_count_);
    host_mm_potential_.allocate(atoms);
    host_mm_energy_.allocate(max_batch_count_);
    host_qm_energy_.allocate(max_batch_count_);
    host_full_energy_.allocate(max_batch_count_);
    host_mm_virial_.allocate(6 * max_batch_count_);
    host_qm_virial_.allocate(6 * max_batch_count_);
    host_full_virial_.allocate(6 * max_batch_count_);
    host_verlet_total_.allocate(1);
    mm_qsum_host_.resize(max_batch_count_);
    mm_qsq_host_.resize(max_batch_count_);
    qm_qsum_host_.resize(max_batch_count_);
    qm_qsq_host_.resize(max_batch_count_);
    neighbor_reference_positions_.resize(coordinates);
  }

  void rebuild_verlet_list(std::size_t atoms) {
    check_cuda(cudaMemsetAsync(d_verlet_counts_.get(), 0,
                               (atoms + 1) * sizeof(std::uint64_t), stream_),
               "clear Verlet neighbor counts");
    const int neighbor_blocks = static_cast<int>(
        (atoms + kRealSpaceWarpsPerBlock - 1) / kRealSpaceWarpsPerBlock);
    count_verlet_neighbors_kernel<<<neighbor_blocks, kRealSpaceThreads, 0,
                                    stream_>>>(
        d_atom_fractional_.get(), d_atom_bins_.get(), d_bin_offsets_.get(),
        d_bin_atoms_.get(), d_neighbor_bin_offsets_.get(),
        d_neighbor_bins_.get(), d_verlet_counts_.get(), atoms, parameters_);
    launch_check("count_verlet_neighbors_kernel");
    std::size_t scan_bytes = verlet_scan_workspace_bytes_;
    check_cuda(cub::DeviceScan::ExclusiveSum(
                   d_verlet_scan_workspace_.get(), scan_bytes,
                   d_verlet_counts_.get(), d_verlet_offsets_.get(), atoms + 1,
                   stream_),
               "CUB exclusive Verlet scan");
    check_cuda(cudaMemcpyAsync(host_verlet_total_.data(),
                               d_verlet_offsets_.get() + atoms,
                               sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                               stream_),
               "download Verlet neighbor count");
    // The first fixed-topology call sizes the persistent list from the actual
    // geometry.  Later rebuilds reuse that storage and grow only if the total
    // neighbor population exceeds the previous high-water mark.
    check_cuda(cudaStreamSynchronize(stream_),
               "synchronize Verlet neighbor count");
    const std::uint64_t required_u64 = host_verlet_total_[0];
    if (required_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("Verlet neighbor list exceeds host size limits");
    const std::size_t required = static_cast<std::size_t>(required_u64);
    if (required > verlet_neighbor_capacity_) {
      const std::size_t spare = std::max(atoms * std::size_t{16}, required / 4);
      if (spare > std::numeric_limits<std::size_t>::max() - required)
        throw std::overflow_error("Verlet neighbor capacity overflows");
      verlet_neighbor_capacity_ = required + spare;
      d_verlet_neighbors_.allocate(verlet_neighbor_capacity_,
                                   "Verlet neighbor list");
    }
    check_cuda(cudaMemsetAsync(d_verlet_cursor_.get(), 0,
                               atoms * sizeof(int), stream_),
               "clear Verlet neighbor cursors");
    fill_verlet_neighbors_kernel<<<neighbor_blocks, kRealSpaceThreads, 0,
                                   stream_>>>(
        d_atom_fractional_.get(), d_atom_bins_.get(), d_bin_offsets_.get(),
        d_bin_atoms_.get(), d_neighbor_bin_offsets_.get(),
        d_neighbor_bins_.get(), d_verlet_offsets_.get(),
        d_verlet_cursor_.get(), d_verlet_neighbors_.get(), atoms, parameters_);
    launch_check("fill_verlet_neighbors_kernel");
  }

  FftPlans &plans_for(std::size_t batch) {
    auto iter = plans_.find(batch);
    if (iter != plans_.end()) return *iter->second;
    auto plans = std::make_unique<FftPlans>();
    int dimensions[3] = {parameters_.nz, parameters_.ny, parameters_.nx};
    const int distance = static_cast<int>(prepared_.mesh_count);
    const auto make_plan = [&](cufftHandle &handle, int transform_batch,
                               const char *operation) {
      check_cufft(cufftCreate(&handle), operation);
      check_cufft(cufftSetAutoAllocation(handle, 0),
                  "disable automatic cuFFT workspace allocation");
      std::size_t workspace_bytes = 0;
      check_cufft(cufftMakePlanMany(handle, 3, dimensions, nullptr, 1, distance,
                                    nullptr, 1, distance, CUFFT_Z2Z,
                                    transform_batch, &workspace_bytes),
                  operation);
      plans->workspace_bytes =
          std::max(plans->workspace_bytes, workspace_bytes);
    };
    make_plan(plans->forward, static_cast<int>(batch),
              "create forward cuFFT planMany");
    make_plan(plans->inverse_mm, static_cast<int>(4 * batch),
              "create MM inverse cuFFT planMany");
    make_plan(plans->inverse_qm, static_cast<int>(3 * batch),
              "create QM inverse cuFFT planMany");
    check_cufft(cufftSetStream(plans->forward, stream_), "set forward cuFFT stream");
    check_cufft(cufftSetStream(plans->inverse_mm, stream_),
                "set MM inverse cuFFT stream");
    check_cufft(cufftSetStream(plans->inverse_qm, stream_),
                "set QM inverse cuFFT stream");
    FftPlans &result = *plans;
    plans_.emplace(batch, std::move(plans));
    if (result.workspace_bytes > d_fft_workspace_.size()) {
      // All transforms execute serially on stream_, so every batch-size plan
      // can share one broker-owned cuFFT work area.  Growing occurs only when
      // a new batch coordinate is first encountered, never in steady state.
      d_fft_workspace_.allocate(result.workspace_bytes, "shared cuFFT workspace");
      for (const auto &entry : plans_)
        entry.second->bind_workspace(d_fft_workspace_.get());
    } else {
      result.bind_workspace(d_fft_workspace_.get());
    }
    return result;
  }

  void compute_charge_moments(const double *charges, std::size_t batch,
                              std::vector<double> &qsum,
                              std::vector<double> &qsq) const {
    std::fill_n(qsum.begin(), batch, 0.0);
    std::fill_n(qsq.begin(), batch, 0.0);
    for (std::size_t frame = 0; frame < batch; ++frame)
      for (std::size_t atom = 0; atom < topology_.atom_count; ++atom) {
        const double charge = charges[frame * topology_.atom_count + atom];
        qsum[frame] += charge;
        qsq[frame] += charge * charge;
      }
  }

  void copy_moments_to_device(const std::vector<double> &qsum,
                              const std::vector<double> &qsq,
                              DeviceBuffer<double> &device_qsum,
                              DeviceBuffer<double> &device_qsq) {
    check_cuda(cudaMemcpyAsync(device_qsum.get(), qsum.data(),
                               qsum.size() * sizeof(double), cudaMemcpyHostToDevice,
                               stream_),
               "copy charge sum");
    check_cuda(cudaMemcpyAsync(device_qsq.get(), qsq.data(), qsq.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream_),
               "copy charge square sum");
  }

  void stage_mm_outputs(std::size_t batch, std::size_t coordinates,
                        std::size_t atoms, bool need_potential,
                        bool need_forces) {
    check_cuda(cudaMemcpyAsync(host_pair_forces_.data(), d_pair_forces_.get(),
                               coordinates * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download pair forces");
    check_cuda(cudaMemcpyAsync(host_pair_scalars_.data(), d_pair_scalars_.get(),
                               8 * batch * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download pair scalars");
    if (need_potential)
      check_cuda(cudaMemcpyAsync(host_mm_potential_.data(), d_mm_potential_.get(),
                                 atoms * sizeof(double), cudaMemcpyDeviceToHost,
                                 stream_),
                 "download MM potential");
    if (need_forces)
      check_cuda(cudaMemcpyAsync(host_mm_forces_.data(), d_mm_forces_.get(),
                                 coordinates * sizeof(double),
                                 cudaMemcpyDeviceToHost, stream_),
                 "download MM reciprocal forces");
    check_cuda(cudaMemcpyAsync(host_mm_energy_.data(), d_mm_energy_.get(),
                               batch * sizeof(double), cudaMemcpyDeviceToHost, stream_),
               "download MM energy");
    check_cuda(cudaMemcpyAsync(host_mm_virial_.data(), d_mm_virial_.get(),
                               6 * batch * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download MM virial");
  }

  void publish_mm(const ClassicalMmBatchOutput &output, std::size_t batch,
                  std::size_t coordinates, std::size_t atoms) const {
    std::copy_n(host_pair_forces_.data(), coordinates, output.pair_forces);
    for (std::size_t frame = 0; frame < batch; ++frame) {
      output.lj_energy[frame] = host_pair_scalars_[8 * frame];
      output.coulomb_energy[frame] = host_pair_scalars_[8 * frame + 1];
      std::copy_n(host_pair_scalars_.data() + 8 * frame + 2, 6,
                  output.pair_virial + 6 * frame);
    }
    std::copy_n(host_mm_energy_.data(), batch, output.mm_pppm_energy);
    std::copy_n(host_mm_virial_.data(), 6 * batch, output.mm_pppm_virial);
    if (output.mm_pppm_potential)
      std::copy_n(host_mm_potential_.data(), atoms, output.mm_pppm_potential);
    if (output.mm_pppm_forces)
      std::copy_n(host_mm_forces_.data(), coordinates, output.mm_pppm_forces);
  }

  void stage_qm_outputs(std::size_t batch, std::size_t coordinates) {
    check_cuda(cudaMemcpyAsync(host_qm_forces_.data(), d_qm_forces_.get(),
                               coordinates * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download QM forces");
    check_cuda(cudaMemcpyAsync(host_full_forces_.data(), d_full_forces_.get(),
                               coordinates * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download full forces");
    check_cuda(cudaMemcpyAsync(host_qm_energy_.data(), d_qm_energy_.get(),
                               batch * sizeof(double), cudaMemcpyDeviceToHost, stream_),
               "download QM energy");
    check_cuda(cudaMemcpyAsync(host_full_energy_.data(), d_full_energy_.get(),
                               batch * sizeof(double), cudaMemcpyDeviceToHost, stream_),
               "download full energy");
    check_cuda(cudaMemcpyAsync(host_qm_virial_.data(), d_qm_virial_.get(),
                               6 * batch * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download QM virial");
    check_cuda(cudaMemcpyAsync(host_full_virial_.data(), d_full_virial_.get(),
                               6 * batch * sizeof(double), cudaMemcpyDeviceToHost,
                               stream_),
               "download full virial");
  }

  void publish_qm(const ClassicalQmBatchOutput &output, std::size_t batch,
                  std::size_t coordinates) const {
    std::copy_n(host_qm_forces_.data(), coordinates, output.qm_pppm_forces);
    std::copy_n(host_full_forces_.data(), coordinates, output.full_pppm_forces);
    std::copy_n(host_qm_energy_.data(), batch, output.qm_pppm_energy);
    std::copy_n(host_full_energy_.data(), batch, output.full_pppm_energy);
    std::copy_n(host_qm_virial_.data(), 6 * batch, output.qm_pppm_virial);
    std::copy_n(host_full_virial_.data(), 6 * batch, output.full_pppm_virial);
  }

  void validate_common_batch(std::size_t batch) const {
    if (batch == 0 || batch > max_batch_count_)
      throw std::invalid_argument("classical CUDA batch count is outside capacity");
  }

  void validate_begin(const ClassicalBatchInput &input,
                      const ClassicalMmBatchOutput &output) const {
    validate_common_batch(input.batch_count);
    if (!input.positions || !input.charges || output.batch_count != input.batch_count ||
        !output.pair_forces || !output.lj_energy || !output.coulomb_energy ||
        !output.pair_virial || !output.mm_pppm_energy || !output.mm_pppm_virial)
      throw std::invalid_argument("classical CUDA MM descriptor is incomplete");
    if (output.retain_for_qm && !output.mm_pppm_potential)
      throw std::invalid_argument(
          "classical CUDA QM continuation requires the MM scalar potential");
    const std::size_t coordinates = 3 * input.batch_count * topology_.atom_count;
    for (std::size_t index = 0; index < coordinates; ++index)
      if (!std::isfinite(input.positions[index]))
        throw std::invalid_argument("classical CUDA positions must be finite");
    const std::size_t charges = input.batch_count * topology_.atom_count;
    for (std::size_t index = 0; index < charges; ++index)
      if (!std::isfinite(input.charges[index]))
        throw std::invalid_argument("classical CUDA MM charges must be finite");
  }

  void validate_finish(const ClassicalQmBatchInput &input,
                       const ClassicalQmBatchOutput &output) const {
    validate_common_batch(input.batch_count);
    if (!input.qm_charges || output.batch_count != input.batch_count ||
        !output.qm_pppm_forces || !output.full_pppm_forces ||
        !output.qm_pppm_energy || !output.full_pppm_energy ||
        !output.qm_pppm_virial || !output.full_pppm_virial)
      throw std::invalid_argument("classical CUDA QM descriptor is incomplete");
    const std::size_t charges = input.batch_count * topology_.atom_count;
    for (std::size_t index = 0; index < charges; ++index)
      if (!std::isfinite(input.qm_charges[index]))
        throw std::invalid_argument("classical CUDA QM charges must be finite");
  }

  ClassicalTopology topology_;
  PreparedClassicalData prepared_;
  std::size_t max_batch_count_ = 0;
  int device_ = 0;
  cudaStream_t stream_ = nullptr;
  DeviceParameters parameters_{};
  std::map<std::size_t, std::unique_ptr<FftPlans>> plans_;
  bool active_ = false;
  std::size_t cached_batch_count_ = 0;
  int mm_transform_fields_ = 0;
  std::vector<double> mm_qsum_host_, mm_qsq_host_, qm_qsum_host_, qm_qsq_host_;
  // One conservative Verlet epoch is shared by the synchronized batch.  A
  // displacement in any window beyond skin/2 rebuilds all bins together;
  // otherwise the old membership remains valid because neighbor-bin reach was
  // constructed with cutoff + skin.
  std::vector<double> neighbor_reference_positions_;
  std::size_t neighbor_batch_count_ = 0;
  bool neighbor_reference_valid_ = false;

  DeviceBuffer<std::int32_t> d_atom_types_, d_oxygen_site_, d_special_offsets_,
      d_special_partners_, d_neighbor_bin_offsets_, d_neighbor_bins_;
  DeviceBuffer<Tip4pSite> d_tip4p_sites_;
  DeviceBuffer<LennardJonesParameters> d_lj_;
  DeviceBuffer<std::uint8_t> d_coulomb_type_pairs_;
  DeviceBuffer<double> d_special_lj_, d_special_coulomb_, d_spline_, d_green_,
      d_kvector_, d_virial_factor_;
  DeviceBuffer<double> d_table_r_, d_table_dr_, d_table_force_, d_table_dforce_,
      d_table_coulomb_, d_table_dcoulomb_, d_table_energy_, d_table_denergy_;
  DeviceBuffer<double> d_positions_, d_mm_charges_, d_qm_charges_, d_full_charges_;
  DeviceBuffer<double3> d_atom_fractional_, d_site_fractional_;
  DeviceBuffer<int> d_atom_bins_, d_bin_counts_, d_bin_offsets_, d_bin_cursor_,
      d_bin_atoms_, d_verlet_cursor_, d_verlet_neighbors_;
  DeviceBuffer<std::uint64_t> d_verlet_counts_, d_verlet_offsets_;
  DeviceBuffer<std::uint8_t> d_scan_workspace_, d_verlet_scan_workspace_;
  DeviceBuffer<std::uint8_t> d_fft_workspace_;
  std::size_t scan_workspace_bytes_ = 0;
  std::size_t verlet_scan_workspace_bytes_ = 0;
  std::size_t verlet_neighbor_capacity_ = 0;
  DeviceBuffer<double> d_pair_forces_, d_mm_forces_, d_qm_forces_, d_full_forces_,
      d_pair_scalars_;
  DeviceBuffer<cufftDoubleComplex> d_density_, d_mm_spectrum_, d_qm_spectrum_,
      d_mm_transforms_, d_qm_transforms_;
  DeviceBuffer<double> d_mm_potential_, d_mm_raw_, d_qm_raw_, d_cross_,
      d_mm_qsum_, d_mm_qsq_, d_qm_qsum_, d_qm_qsq_, d_mm_energy_, d_qm_energy_,
      d_full_energy_, d_mm_virial_, d_qm_virial_, d_full_virial_;

  PinnedBuffer<double> host_pair_forces_, host_mm_forces_, host_qm_forces_,
      host_full_forces_, host_pair_scalars_, host_mm_potential_, host_mm_energy_,
      host_qm_energy_, host_full_energy_, host_mm_virial_, host_qm_virial_,
      host_full_virial_;
  PinnedBuffer<std::uint64_t> host_verlet_total_;
};

}  // namespace

std::unique_ptr<ClassicalBatchPlan>
create_cuda_classical_batch_plan(const ClassicalTopology &topology,
                                 const ClassicalPlanOptions &options) {
  return std::make_unique<CudaClassicalBatchPlan>(topology, options);
}

}  // namespace DPRC
