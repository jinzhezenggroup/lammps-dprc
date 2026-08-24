#include "deepmd_batch_executor.h"

#include <deepmd/c_api.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

static_assert(DP_C_API_VERSION >= 31,
              "LAMMPS-DPRC requires DeePMD C API version 31 or newer");

namespace DPRC {
namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t grown_capacity(std::size_t required) {
  const std::size_t slack = required / 50 + 64;
  if (required > std::numeric_limits<std::size_t>::max() - slack)
    throw std::overflow_error("DeePMD device buffer capacity overflows size_t");
  return required + slack;
}

template <typename T>
void allocate_device(T *&pointer, std::size_t count, const char *label) {
  void *allocation = nullptr;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::overflow_error(std::string(label) + " byte size overflows");
  check_cuda(cudaMalloc(&allocation, count * sizeof(T)), label);
  pointer = static_cast<T *>(allocation);
}

template <typename T>
void free_device(T *&pointer) noexcept {
  if (pointer != nullptr) {
    cudaFree(pointer);
    pointer = nullptr;
  }
}

template <typename T>
void copy_to_device(T *destination, const T *source, std::size_t count,
                    const char *label) {
  if (count == 0)
    return;
  check_cuda(cudaMemcpy(destination, source, count * sizeof(T),
                        cudaMemcpyHostToDevice),
             label);
}

template <typename T>
void copy_to_host(T *destination, const T *source, std::size_t count,
                  const char *label) {
  if (count == 0)
    return;
  check_cuda(cudaMemcpy(destination, source, count * sizeof(T),
                        cudaMemcpyDeviceToHost),
             label);
}

void require_pointer(const void *pointer, std::size_t count,
                     const char *label) {
  if (count != 0 && pointer == nullptr)
    throw std::invalid_argument(std::string(label) + " is null");
}

}  // namespace

DeepmdBatchExecutor::DeepmdBatchExecutor(const std::string &model_path,
                                         int gpu_rank) {
  if (model_path.empty())
    throw std::invalid_argument("DeePMD model path is empty");
  if (gpu_rank < 0)
    throw std::invalid_argument("DeePMD GPU rank must be non-negative");

  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count <= 0)
    throw std::runtime_error("DeePMD batch execution requires a CUDA device");
  device_id_ = gpu_rank % device_count;
  select_device();

  model_ = DP_NewDeepPotWithParam2(model_path.c_str(), gpu_rank, "", 0);
  if (model_ == nullptr)
    throw std::runtime_error("DP_NewDeepPotWithParam2 returned null");
  try {
    check_model("loading DeePMD model");
    metadata_.cutoff = DP_DeepPotGetCutoff(model_);
    metadata_.type_count = DP_DeepPotGetNumbTypes(model_);
    metadata_.spin_type_count = DP_DeepPotGetNumbTypesSpin(model_);
    metadata_.frame_parameter_width = DP_DeepPotGetDimFParam(model_);
    metadata_.atom_parameter_width = DP_DeepPotGetDimAParam(model_);
    metadata_.charge_spin_width = DP_DeepPotGetDimChgSpin(model_);
    const char *type_map = DP_DeepPotGetTypeMap(model_);
    metadata_.type_map = type_map == nullptr ? std::string{} : type_map;
    if (type_map != nullptr)
      DP_DeleteChar(type_map);
    check_model("querying DeePMD model metadata");

    if (!std::isfinite(metadata_.cutoff) || metadata_.cutoff <= 0.0 ||
        metadata_.type_count <= 0 || metadata_.type_map.empty()) {
      throw std::runtime_error("DeePMD model returned invalid metadata");
    }
    if (metadata_.spin_type_count != 0 ||
        metadata_.frame_parameter_width != 0 ||
        metadata_.atom_parameter_width != 0) {
      throw std::runtime_error(
          "dprc/deepmd/batch requires a non-spin model without fparam or "
          "aparam inputs");
    }

    const bool device_edges =
        DP_DeepPotSupportsDeviceEdgeInference(model_);
    const bool fp32_edges = DP_DeepPotUsesFP32EdgeVectors(model_);
    const bool canonical = DP_DeepPotUsesCanonicalGraphInference(model_);
    check_model("querying DeePMD canonical-graph capabilities");
    if (!device_edges || !fp32_edges || !canonical) {
      throw std::runtime_error(
          "dprc/deepmd/batch requires a compact canonical graph artifact "
          "with FP32 edge vectors");
    }
  } catch (...) {
    DP_DeleteDeepPot(model_);
    model_ = nullptr;
    throw;
  }
}

DeepmdBatchExecutor::~DeepmdBatchExecutor() {
  release_device_buffers();
  if (model_ != nullptr)
    DP_DeleteDeepPot(model_);
}

void DeepmdBatchExecutor::check_model(const char *operation) const {
  const char *message = DP_DeepPotCheckOK(model_);
  const std::string diagnostic = message == nullptr ? std::string{} : message;
  if (message != nullptr)
    DP_DeleteChar(message);
  if (!diagnostic.empty())
    throw std::runtime_error(std::string(operation) + ": " + diagnostic);
}

void DeepmdBatchExecutor::select_device() const {
  check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
}

void DeepmdBatchExecutor::release_device_buffers() noexcept {
  if (cudaSetDevice(device_id_) != cudaSuccess)
    return;
  free_device(device_atom_types_);
  free_device(device_sources_);
  free_device(device_edge_vectors_);
  free_device(device_destination_row_ptr_);
  free_device(device_source_row_ptr_);
  free_device(device_source_order_);
  free_device(device_atom_energy_);
  free_device(device_force_);
  free_device(device_atom_virial_);
  node_capacity_ = 0;
  edge_capacity_ = 0;
  row_capacity_ = 0;
}

void DeepmdBatchExecutor::ensure_node_capacity(std::size_t required) {
  if (node_capacity_ >= required)
    return;
  const std::size_t capacity = grown_capacity(required);
  free_device(device_atom_types_);
  free_device(device_atom_energy_);
  free_device(device_force_);
  free_device(device_atom_virial_);
  allocate_device(device_atom_types_, capacity, "cudaMalloc atom types");
  allocate_device(device_atom_energy_, capacity, "cudaMalloc atomic energy");
  allocate_device(device_force_, 3 * capacity, "cudaMalloc force");
  allocate_device(device_atom_virial_, 9 * capacity,
                  "cudaMalloc atomic virial");
  node_capacity_ = capacity;
}

void DeepmdBatchExecutor::ensure_edge_capacity(std::size_t required) {
  if (edge_capacity_ >= required)
    return;
  const std::size_t capacity = grown_capacity(required);
  free_device(device_sources_);
  free_device(device_edge_vectors_);
  free_device(device_source_order_);
  allocate_device(device_sources_, capacity, "cudaMalloc edge sources");
  allocate_device(device_edge_vectors_, 3 * capacity,
                  "cudaMalloc edge vectors");
  allocate_device(device_source_order_, capacity,
                  "cudaMalloc source order");
  edge_capacity_ = capacity;
}

void DeepmdBatchExecutor::ensure_row_capacity(std::size_t required) {
  if (row_capacity_ >= required)
    return;
  const std::size_t capacity = grown_capacity(required);
  free_device(device_destination_row_ptr_);
  free_device(device_source_row_ptr_);
  allocate_device(device_destination_row_ptr_, capacity,
                  "cudaMalloc destination CSR");
  allocate_device(device_source_row_ptr_, capacity,
                  "cudaMalloc source CSR");
  row_capacity_ = capacity;
}

void DeepmdBatchExecutor::compute(const DeepmdCanonicalBatchView &batch,
                                  DeepmdCanonicalBatchResult &result) {
  if (batch.frame_count <= 0 || batch.node_count == 0 ||
      batch.physical_edge_count < 0 ||
      batch.edge_storage < std::max<std::int64_t>(batch.physical_edge_count, 2)) {
    throw std::invalid_argument("invalid DeePMD canonical batch extents");
  }
  if (batch.node_count >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      static_cast<std::uint64_t>(batch.edge_storage) >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("DeePMD canonical batch exceeds ABI index limits");
  }
  require_pointer(batch.atom_types, batch.node_count, "atom_types");
  require_pointer(batch.sources, static_cast<std::size_t>(batch.edge_storage),
                  "sources");
  require_pointer(batch.edge_vectors,
                  3 * static_cast<std::size_t>(batch.edge_storage),
                  "edge_vectors");
  require_pointer(batch.destination_row_ptr, batch.node_count + 1,
                  "destination_row_ptr");
  require_pointer(batch.source_row_ptr, batch.node_count + 1,
                  "source_row_ptr");
  require_pointer(batch.source_order,
                  static_cast<std::size_t>(batch.edge_storage),
                  "source_order");
  require_pointer(batch.local_nodes_per_frame,
                  static_cast<std::size_t>(batch.frame_count),
                  "local_nodes_per_frame");
  require_pointer(batch.all_nodes_per_frame,
                  static_cast<std::size_t>(batch.frame_count),
                  "all_nodes_per_frame");

  // Validate every host-side index before any device copy or model call. The
  // C API accepts raw pointers and cannot recover safely from malformed CSR
  // metadata, so this scan is part of the failure-atomic publication boundary.
  std::size_t frame_node_sum = 0;
  for (int frame = 0; frame < batch.frame_count; ++frame) {
    const std::int64_t local = batch.local_nodes_per_frame[frame];
    const std::int64_t all = batch.all_nodes_per_frame[frame];
    if (local <= 0 || all < local)
      throw std::invalid_argument("invalid DeePMD frame node extents");
    if (static_cast<std::uint64_t>(all) >
        std::numeric_limits<std::size_t>::max() - frame_node_sum)
      throw std::overflow_error("DeePMD frame node sum overflows size_t");
    frame_node_sum += static_cast<std::size_t>(all);
  }
  if (frame_node_sum != batch.node_count)
    throw std::invalid_argument(
        "DeePMD frame node extents do not cover the batch node axis");

  const std::int64_t physical_edges = batch.physical_edge_count;
  if (batch.destination_row_ptr[0] != 0 || batch.source_row_ptr[0] != 0 ||
      batch.destination_row_ptr[batch.node_count] != physical_edges ||
      batch.source_row_ptr[batch.node_count] != physical_edges)
    throw std::invalid_argument("DeePMD CSR terminal entries are invalid");
  for (std::size_t node = 0; node < batch.node_count; ++node) {
    if (batch.atom_types[node] < 0 ||
        batch.atom_types[node] >= metadata_.type_count ||
        batch.destination_row_ptr[node] >
            batch.destination_row_ptr[node + 1] ||
        batch.source_row_ptr[node] > batch.source_row_ptr[node + 1] ||
        batch.destination_row_ptr[node + 1] > physical_edges ||
        batch.source_row_ptr[node + 1] > physical_edges) {
      throw std::invalid_argument(
          "DeePMD batch contains an invalid type or CSR row");
    }
  }
  std::vector<unsigned char> source_order_seen(
      static_cast<std::size_t>(physical_edges), 0);
  for (std::int64_t edge = 0; edge < physical_edges; ++edge) {
    if (batch.sources[edge] >= batch.node_count ||
        !std::isfinite(batch.edge_vectors[3 * edge + 0]) ||
        !std::isfinite(batch.edge_vectors[3 * edge + 1]) ||
        !std::isfinite(batch.edge_vectors[3 * edge + 2]))
      throw std::invalid_argument("DeePMD batch contains an invalid edge");
  }
  for (std::size_t source = 0; source < batch.node_count; ++source) {
    for (std::int64_t slot = batch.source_row_ptr[source];
         slot < batch.source_row_ptr[source + 1]; ++slot) {
      const std::uint32_t edge = batch.source_order[slot];
      if (edge >= static_cast<std::uint64_t>(physical_edges) ||
          source_order_seen[edge] || batch.sources[edge] != source) {
        throw std::invalid_argument(
            "DeePMD source order is not a valid source-grouped permutation");
      }
      source_order_seen[edge] = 1;
    }
  }

  select_device();
  ensure_node_capacity(batch.node_count);
  ensure_edge_capacity(static_cast<std::size_t>(batch.edge_storage));
  ensure_row_capacity(batch.node_count + 1);

  copy_to_device(device_atom_types_, batch.atom_types, batch.node_count,
                 "copy DeePMD atom types");
  copy_to_device(device_sources_, batch.sources,
                 static_cast<std::size_t>(batch.edge_storage),
                 "copy DeePMD edge sources");
  copy_to_device(device_edge_vectors_, batch.edge_vectors,
                 3 * static_cast<std::size_t>(batch.edge_storage),
                 "copy DeePMD edge vectors");
  copy_to_device(device_destination_row_ptr_, batch.destination_row_ptr,
                 batch.node_count + 1, "copy DeePMD destination CSR");
  copy_to_device(device_source_row_ptr_, batch.source_row_ptr,
                 batch.node_count + 1, "copy DeePMD source CSR");
  copy_to_device(device_source_order_, batch.source_order,
                 static_cast<std::size_t>(batch.edge_storage),
                 "copy DeePMD source order");

  DP_DeepPotComputeCanonicalGraphBatchGPU(
      model_, device_atom_energy_, device_force_, device_atom_virial_,
      device_atom_types_, device_sources_, device_edge_vectors_,
      device_destination_row_ptr_, device_source_row_ptr_,
      device_source_order_, batch.local_nodes_per_frame,
      batch.all_nodes_per_frame, batch.frame_count, batch.edge_storage);
  check_model("DP_DeepPotComputeCanonicalGraphBatchGPU");

  // The public API does not currently expose the backend CUDA stream.  A
  // completion fence is therefore required before host publication; this is a
  // correctness boundary, not an implicit precision or performance mode.
  check_cuda(cudaDeviceSynchronize(), "synchronize DeePMD batch result");

  DeepmdCanonicalBatchResult pending;
  pending.atom_energy.resize(batch.node_count);
  pending.force.resize(3 * batch.node_count);
  pending.atom_virial.resize(9 * batch.node_count);
  copy_to_host(pending.atom_energy.data(), device_atom_energy_, batch.node_count,
               "copy DeePMD atomic energy");
  copy_to_host(pending.force.data(), device_force_, 3 * batch.node_count,
               "copy DeePMD force");
  copy_to_host(pending.atom_virial.data(), device_atom_virial_,
               9 * batch.node_count, "copy DeePMD atomic virial");
  const auto finite = [](const std::vector<double> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (!finite(pending.atom_energy) || !finite(pending.force) ||
      !finite(pending.atom_virial)) {
    throw std::runtime_error("DeePMD batch returned a non-finite result");
  }
  result = std::move(pending);
}

}  // namespace DPRC
