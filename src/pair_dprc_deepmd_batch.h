#ifndef LAMMPS_DPRC_PAIR_DPRC_DEEPMD_BATCH_H
#define LAMMPS_DPRC_PAIR_DPRC_DEEPMD_BATCH_H

#include "deepmd_partition_broker.h"
#include "pair.h"
#include "partition_roots.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

// Host-side LAMMPS publication adapter for the broker-owned DeePMD canonical
// batch.  Each partition is deliberately restricted to one MPI rank: periodic
// ghosts are folded onto their local owners, then independent partition graphs
// are evaluated together by one GPU-local C API model instance.
class PairDPRCDeepMDBatch final : public Pair {
 public:
  explicit PairDPRCDeepMDBatch(class LAMMPS *);
  ~PairDPRCDeepMDBatch() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

 private:
  void allocate();
  void build_compact_graph(DPRC::DeepmdCanonicalGraph &,
                           std::vector<int> &node_to_atom) const;
  std::vector<unsigned char> select_model_atoms() const;
  void check_collective_build_status(bool failed,
                                     std::string diagnostic) const;
  std::vector<std::string> model_type_names() const;

  std::string model_path_;
  std::string center_group_id_;
  int center_group_bit_ = 0;
  double environment_cutoff_ = 0.0;
  double model_cutoff_ = 0.0;
  double distance_unit_factor_ = 1.0;
  double energy_unit_factor_ = 1.0;
  double force_unit_factor_ = 1.0;
  bool include_molecule_ = true;
  int gpu_rank_ = 0;

  std::vector<int> type_index_map_;
  double **scale_ = nullptr;
  std::unique_ptr<DPRC::PartitionRoots> roots_;
  std::unique_ptr<DPRC::DeepmdPartitionBroker> broker_;
};

}  // namespace LAMMPS_NS

#endif
