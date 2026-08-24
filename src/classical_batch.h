#ifndef LAMMPS_DPRC_CLASSICAL_BATCH_H
#define LAMMPS_DPRC_CLASSICAL_BATCH_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace DPRC {

enum class ClassicalBackend { CPU, CUDA };

// LAMMPS restricted-triclinic cell.  H is row-major and maps fractional
// coordinates to Cartesian coordinates: x = boxlo + H*s.
struct RestrictedTriclinicCell {
  std::array<double, 3> boxlo{};
  std::array<double, 9> h{};
};

struct Tip4pSite {
  std::int32_t oxygen = -1;
  std::int32_t hydrogen1 = -1;
  std::int32_t hydrogen2 = -1;
};

struct SpecialPair {
  std::int32_t atom1 = -1;
  std::int32_t atom2 = -1;
  double lj_scale = 1.0;
  double coulomb_scale = 1.0;
};

// Pair coefficients use the exact expanded form consumed by LAMMPS pair
// kernels.  Indexing is a dense zero-based type_count-by-type_count matrix.
struct LennardJonesParameters {
  double lj1 = 0.0;
  double lj2 = 0.0;
  double lj3 = 0.0;
  double lj4 = 0.0;
  double offset = 0.0;
  double cutoff = 0.0;
};

struct PppmParameters {
  std::array<std::int32_t, 3> mesh{0, 0, 0};
  std::int32_t order = 0;
  double g_ewald = 0.0;
};

// Optional exact mirror of LAMMPS's bitmap-interpolated real-space Coulomb
// table.  Empty arrays select the direct polynomial path for every distance.
// The integration layer copies these values from the matched pinned Pair
// instance so the GPU backend does not silently change the Hamiltonian above
// pair_modify tabinner (sqrt(2) A by default).
struct CoulombLookupTable {
  std::int32_t bits = 0;
  std::int32_t shift_bits = 0;
  std::int32_t mask = 0;
  double inner_squared = 0.0;
  std::vector<double> r;
  std::vector<double> dr;
  std::vector<double> force;
  std::vector<double> dforce;
  std::vector<double> coulomb;
  std::vector<double> dcoulomb;
  std::vector<double> energy;
  std::vector<double> denergy;
};

struct ClassicalTopology {
  std::size_t atom_count = 0;
  std::int32_t type_count = 0;
  std::vector<std::int32_t> atom_types;
  std::vector<Tip4pSite> tip4p_sites;
  std::vector<SpecialPair> special_pairs;
  std::vector<LennardJonesParameters> lj;

  // A true type-pair entry enables the real-space Ewald Coulomb term.  The
  // ETP/ETH contract enables only the MM water mapping while LJ remains
  // enabled for every configured type pair.
  std::vector<std::uint8_t> coulomb_type_pairs;

  RestrictedTriclinicCell cell;
  PppmParameters pppm;
  CoulombLookupTable coulomb_table;
  double tip4p_alpha = 0.0;
  double tip4p_qdist = 0.0;
  double real_space_cutoff = 0.0;
  double neighbor_skin = 0.0;
  double qqrd2e = 0.0;
};

// All frames in one call share the immutable topology above.  Coordinates and
// charges are laid out frame-major, then atom-major, in LAMMPS Cartesian units.
struct ClassicalBatchInput {
  std::size_t batch_count = 0;
  const double *positions = nullptr;
  const double *charges = nullptr;
};

struct ClassicalMmBatchOutput {
  std::size_t batch_count = 0;
  double *pair_forces = nullptr;
  double *lj_energy = nullptr;
  double *coulomb_energy = nullptr;
  double *pair_virial = nullptr;
  double *mm_pppm_energy = nullptr;
  double *mm_pppm_virial = nullptr;

  // Optional reciprocal scalar potential at every atom charge site, before
  // multiplication by qqrd2e or the atom charge.  This is the quantity needed
  // by the fused QM/MM periodic embedding path.
  double *mm_pppm_potential = nullptr;

  // Optional MM-only reciprocal force on every real atom.  TIP4P charge-site
  // forces are redistributed to O/H/H before publication.  A pure-classical
  // caller requests this output instead of the scalar potential, allowing the
  // CUDA backend to execute only the three electric-field inverse transforms.
  double *mm_pppm_forces = nullptr;

  // Retain the MM spectrum, charge sites, and four scalar/field transforms so
  // finish_qm() can construct the bilinear QM/MM reciprocal result.  A
  // terminal pure-classical request sets this false; both CPU and CUDA then
  // reject finish_qm() instead of inferring continuation from nullable output
  // pointers.
  bool retain_for_qm = true;
};

// QM-only charges are supplied after batched SCC.  Positions and the MM state
// are retained by begin_mm(), so the second stage does not repeat coordinate
// staging or the MM charge assignment/FFT.
struct ClassicalQmBatchInput {
  std::size_t batch_count = 0;
  const double *qm_charges = nullptr;
};

struct ClassicalQmBatchOutput {
  std::size_t batch_count = 0;
  double *qm_pppm_forces = nullptr;
  double *full_pppm_forces = nullptr;
  double *qm_pppm_energy = nullptr;
  double *full_pppm_energy = nullptr;
  double *qm_pppm_virial = nullptr;
  double *full_pppm_virial = nullptr;
};

struct ClassicalPlanOptions {
  ClassicalBackend backend = ClassicalBackend::CPU;
  std::size_t max_batch_count = 1;
  std::int32_t cuda_device = -1;
};

class ClassicalBatchPlan {
 public:
  virtual ~ClassicalBatchPlan() = default;

  ClassicalBatchPlan(const ClassicalBatchPlan &) = delete;
  ClassicalBatchPlan &operator=(const ClassicalBatchPlan &) = delete;

  [[nodiscard]] virtual ClassicalBackend backend() const noexcept = 0;
  [[nodiscard]] virtual std::size_t max_batch_count() const noexcept = 0;

  // begin_mm retains one exact frame epoch until finish_qm() or cancel().  It
  // performs the coordinate staging, real-space pair traversal, MM charge
  // assignment, and one forward FFT.  Requested outputs control the inverse
  // work: potential needs one scalar inverse, while reciprocal forces need the
  // three electric-field inverses.  QM/MM and pure-classical callers therefore
  // do not compute one another's unused publication.
  virtual void begin_mm(const ClassicalBatchInput &, const ClassicalMmBatchOutput &) = 0;

  // finish_qm performs the QM-only charge assignment and four-dimensional
  // batched reciprocal work, publishes both QM-only and assembled-full PPPM
  // results, and consumes the retained MM epoch.  A third full-charge forward
  // FFT is forbidden by this contract.
  virtual void finish_qm(const ClassicalQmBatchInput &,
                         const ClassicalQmBatchOutput &) = 0;

  // Abandon retained state after an SCC or surrounding LAMMPS transaction
  // fails.  This path must not allocate or publish caller output.
  virtual void cancel() noexcept = 0;

 protected:
  ClassicalBatchPlan() = default;
};

[[nodiscard]] std::unique_ptr<ClassicalBatchPlan>
create_classical_batch_plan(const ClassicalTopology &, const ClassicalPlanOptions &);

}  // namespace DPRC

#endif
