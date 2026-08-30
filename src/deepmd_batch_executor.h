#ifndef LAMMPS_DPRC_DEEPMD_BATCH_EXECUTOR_H
#define LAMMPS_DPRC_DEEPMD_BATCH_EXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct DP_DeepPot;

namespace DPRC {

// Immutable metadata queried from the broker-owned DeePMD model.  Values use
// DeePMD's native units: angstrom for cutoff and eV for model outputs.
struct DeepmdModelMetadata {
  double cutoff = 0.0;
  int type_count = 0;
  int spin_type_count = 0;
  int frame_parameter_width = 0;
  int atom_parameter_width = 0;
  int charge_spin_width = 0;
  std::string type_map;
};

// Host view of one already assembled block-diagonal canonical graph batch.
// Node and edge indices are global on the concatenated axes.  The DeePMD C API
// consumes device pointers, so the executor owns reusable staging buffers and
// never exposes a DeePMD C++ object or private backend symbol.
struct DeepmdCanonicalBatchView {
  const std::int64_t *atom_types = nullptr;
  const std::uint32_t *sources = nullptr;
  const float *edge_vectors = nullptr;
  const std::int64_t *destination_row_ptr = nullptr;
  const std::int64_t *source_row_ptr = nullptr;
  const std::uint32_t *source_order = nullptr;
  const std::int64_t *local_nodes_per_frame = nullptr;
  const std::int64_t *all_nodes_per_frame = nullptr;
  std::size_t node_count = 0;
  std::int64_t physical_edge_count = 0;
  std::int64_t edge_storage = 0;
  int frame_count = 0;
};

struct DeepmdCanonicalBatchResult {
  std::vector<double> atom_energy;
  std::vector<double> force;
  std::vector<double> atom_virial;
};

// Single-device owner of a DeePMD compact canonical model.  The class uses
// only source/api_c/include/c_api.h and the CUDA runtime needed to allocate the
// raw device buffers required by canonical-graph inference. API v31+ receives
// the explicit frame axis; API v30 uses the qualified block-diagonal fallback.
class DeepmdBatchExecutor {
 public:
  DeepmdBatchExecutor(const std::string &model_path, int gpu_rank);
  ~DeepmdBatchExecutor();

  DeepmdBatchExecutor(const DeepmdBatchExecutor &) = delete;
  DeepmdBatchExecutor &operator=(const DeepmdBatchExecutor &) = delete;

  const DeepmdModelMetadata &metadata() const noexcept { return metadata_; }
  int device_id() const noexcept { return device_id_; }

  void compute(const DeepmdCanonicalBatchView &batch,
               DeepmdCanonicalBatchResult &result);

 private:
  void check_model(const char *operation) const;
  void select_device() const;
  void release_device_buffers() noexcept;
  void ensure_node_capacity(std::size_t required);
  void ensure_edge_capacity(std::size_t required);
  void ensure_row_capacity(std::size_t required);

  DP_DeepPot *model_ = nullptr;
  DeepmdModelMetadata metadata_;
  int device_id_ = 0;

  std::size_t node_capacity_ = 0;
  std::size_t edge_capacity_ = 0;
  std::size_t row_capacity_ = 0;
  std::int64_t *device_atom_types_ = nullptr;
  std::uint32_t *device_sources_ = nullptr;
  float *device_edge_vectors_ = nullptr;
  std::int64_t *device_destination_row_ptr_ = nullptr;
  std::int64_t *device_source_row_ptr_ = nullptr;
  std::uint32_t *device_source_order_ = nullptr;
  double *device_atom_energy_ = nullptr;
  double *device_force_ = nullptr;
  double *device_atom_virial_ = nullptr;
};

}  // namespace DPRC

#endif
