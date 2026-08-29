#ifndef LAMMPS_DPRC_KSPACE_DPRC_BATCHED_H
#define LAMMPS_DPRC_KSPACE_DPRC_BATCHED_H

#include "classical_batch.h"
#include "classical_partition_broker.h"
#include "kspace_dprc.h"
#include "partition_roots.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace LAMMPS_NS {

// Publication adapter for the GPU-local batched classical backend.  This
// derived KSpace style deliberately does not initialize PPPM's per-window FFT
// grids: one ClassicalPartitionBroker owns the sole CUDA/cuFFT plan for all
// synchronized one-rank windows.
class PPPMTIP4PDPRCBatched final : public PPPMTIP4PDPRC {
 public:
  explicit PPPMTIP4PDPRCBatched(class LAMMPS *);
  ~PPPMTIP4PDPRCBatched() override;

  void init() override;
  void setup() override;
  void compute(int, int) override;
  double memory_usage() override;

  void compute_group_potential(double *, int, int, bool) override;
  void project_last_solve_potential(double *, int) override;
  void cache_last_solve_as_mm() override;
  void prepare_fused_full_solve() override;
  void commit_fused_full_solve(bigint, int, int, int) override;
  void discard_fused_full_solve() noexcept override;
  int get_charge_site(int, double *, int *, double *) override;

  // The pure-classical coordinator invokes this in PRE_FORCE, before ordinary
  // pair styles run.  This is intentionally separate from the QM/MM capture
  // transaction so a pair proxy can never infer or create execution mode.
  void prepare_classical_publication(int vflag);

  // Pair proxies consume one prepared publication without launching kernels.
  void consume_lj_publication(int eflag, int vflag, double &energy,
                              double *virial);
  void consume_coulomb_publication(int eflag, int vflag, double &energy);

 private:
  struct PublicationToken {
    bigint timestep = -1;
    int vflag = 0;
    int whichflag = 0;
    int setupflag = 0;
  };

  enum class CaptureStage {
    Idle,
    MmReady,
    MmCached,
    QmReady,
    Prepared,
    Pending,
  };

  enum class PublicationKind {
    None,
    MmOnly,
    FullQmMm,
  };

  void build_broker();
  void validate_proxy_configuration() const;
  void require_roots_collectively(bool, const char *) const;
  [[nodiscard]] DPRC::ClassicalTopology build_topology() const;
  [[nodiscard]] std::vector<std::uint8_t> pair_mapping(class Pair *) const;
  void build_stable_atom_order();
  void refresh_stable_local_indices() const;
  void append_special_pairs(DPRC::ClassicalTopology &) const;
  void validate_fixed_state() const;
  void pack_frame(std::vector<double> &positions,
                  std::vector<double> &charges) const;
  void publish_forces(const double *) const;
  void publish_kspace(const double *forces, double published_energy,
                      const double *published_virial, int eflag, int vflag);
  void validate_publication_phase(int vflag, bool pair_phase) const;
  void maybe_complete_publication() noexcept;
  [[nodiscard]] int atom_index_for_slot(std::size_t slot) const;

  std::unique_ptr<DPRC::PartitionRoots> roots_;
  std::unique_ptr<DPRC::ClassicalPartitionBroker> broker_;
  std::vector<tagint> stable_tags_;
  // Atom sorting can change owned indices at neighbor rebuilds.  This cache is
  // rebuilt from atom->tag[0:nlocal], never from the ghost-inclusive atom map.
  mutable std::vector<int> stable_local_indices_;
  std::vector<std::int32_t> stable_atom_types_;
  std::vector<DPRC::SpecialPair> stable_special_pairs_;
  DPRC::RestrictedTriclinicCell stable_cell_{};
  std::int32_t stable_type_count_ = 0;
  std::vector<double> frame_positions_;
  std::vector<double> frame_charges_;

  bigint init_epoch_ = 0;
  bigint broker_epoch_ = -1;
  CaptureStage stage_ = CaptureStage::Idle;
  PublicationKind publication_kind_ = PublicationKind::None;
  PublicationToken token_{};
  bool capture_armed_ = false;
  bool lj_consumed_ = false;
  bool coulomb_consumed_ = false;
  bool kspace_consumed_ = false;
};

} // namespace LAMMPS_NS

#endif
