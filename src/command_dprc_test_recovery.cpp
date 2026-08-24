#include "command_dprc_test_recovery.h"

#include "fix_dprc_xtb.h"

#include "atom.h"
#include "error.h"
#include "exceptions.h"
#include "force.h"
#include "input.h"
#include "kspace.h"
#include "modify.h"
#include "pair.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {

constexpr const char *EXPECTED_FAILURE =
    "DPRC test hook: failure after fused full-solve preparation";
constexpr const char *EXPECTED_POST_COMMIT_FAILURE =
    "DPRC test hook: failure after fused full-solve commit";
constexpr const char *EXPECTED_COMPUTE_NO_FAILURE =
    "Fix qmmm/xtb requires kspace_modify compute yes";

struct ForceSnapshot {
  std::vector<double> charges;
  std::vector<double> forces;
  double correction = 0.0;
  double pair_vdwl = 0.0;
  double pair_coulomb = 0.0;
  double kspace_energy = 0.0;
  std::array<double, 6> pair_virial{};
  std::array<double, 6> kspace_virial{};
};

ForceSnapshot capture_snapshot(Atom *atom, Force *force, FixDPRCXtb *fix) {
  ForceSnapshot snapshot;
  snapshot.charges.resize(atom->nlocal);
  snapshot.forces.resize(static_cast<std::size_t>(3) * atom->nlocal);
  for (int i = 0; i < atom->nlocal; ++i) {
    snapshot.charges[i] = atom->q[i];
    for (int dim = 0; dim < 3; ++dim)
      snapshot.forces[3 * i + dim] = atom->f[i][dim];
  }
  snapshot.correction = fix->compute_scalar();
  snapshot.pair_vdwl = force->pair->eng_vdwl;
  snapshot.pair_coulomb = force->pair->eng_coul;
  snapshot.kspace_energy = force->kspace->energy;
  for (int component = 0; component < 6; ++component) {
    snapshot.pair_virial[component] = force->pair->virial[component];
    snapshot.kspace_virial[component] = force->kspace->virial[component];
  }
  return snapshot;
}

double absolute_error(double actual, double expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected))
    return HUGE_VAL;
  return std::fabs(actual - expected);
}

std::array<double, 4> snapshot_errors(Atom *atom,
                                      const ForceSnapshot &actual,
                                      const ForceSnapshot &expected,
                                      MPI_Comm world) {
  double local_charge_error = 0.0;
  double local_force_error = 0.0;
  for (int i = 0; i < atom->nlocal; ++i) {
    local_charge_error = std::max(
        local_charge_error,
        absolute_error(actual.charges[i], expected.charges[i]));
    for (int dim = 0; dim < 3; ++dim)
      local_force_error = std::max(
          local_force_error,
          absolute_error(actual.forces[3 * i + dim],
                         expected.forces[3 * i + dim]));
  }

  double local_energy_error =
      absolute_error(actual.correction, expected.correction);
  local_energy_error = std::max(
      local_energy_error,
      absolute_error(actual.pair_vdwl, expected.pair_vdwl));
  local_energy_error = std::max(
      local_energy_error,
      absolute_error(actual.pair_coulomb, expected.pair_coulomb));
  local_energy_error = std::max(
      local_energy_error,
      absolute_error(actual.kspace_energy, expected.kspace_energy));

  double local_virial_error = 0.0;
  for (int component = 0; component < 6; ++component) {
    local_virial_error = std::max(
        local_virial_error,
        absolute_error(actual.pair_virial[component],
                       expected.pair_virial[component]));
    local_virial_error = std::max(
        local_virial_error,
        absolute_error(actual.kspace_virial[component],
                       expected.kspace_virial[component]));
  }

  const std::array<double, 4> local_errors = {
      local_energy_error, local_charge_error, local_force_error,
      local_virial_error};
  std::array<double, 4> errors{};
  MPI_Allreduce(local_errors.data(), errors.data(),
                static_cast<int>(errors.size()), MPI_DOUBLE, MPI_MAX, world);
  return errors;
}

void require_snapshot_match(Error *error, Atom *atom,
                            const ForceSnapshot &actual,
                            const ForceSnapshot &expected, MPI_Comm world,
                            const char *label) {
  const std::array<double, 4> errors =
      snapshot_errors(atom, actual, expected, world);

  // These are the same LAMMPS-real-unit tolerances used by the independent
  // reference harness. Both stale-token defects exceed them by a wide margin.
  if (errors[0] > 1.0e-5 || errors[1] > 1.0e-7 ||
      errors[2] > 1.0e-4 || errors[3] > 1.0e-4)
    error->all(
        FLERR,
        "DPRC {} retry differs from the no-failure result: "
        "energy {}, charge {}, force {}, virial {}",
        label, errors[0], errors[1], errors[2], errors[3]);
}

} // namespace

void CommandDPRCTestRecovery::command(int argc, char **argv) {
  if (argc != 1)
    error->all(FLERR,
               "Illegal dprc/test/qmmm_failure_recovery command");
  if (!force->pair || !force->kspace)
    error->all(FLERR,
               "DPRC failure-recovery test requires pair and KSpace styles");

  auto *fix = dynamic_cast<FixDPRCXtb *>(modify->get_fix_by_id(argv[0]));
  if (!fix)
    error->all(FLERR,
               "DPRC failure-recovery test requires a qmmm/xtb/dprc fix ID");

  const ForceSnapshot initial_baseline = capture_snapshot(atom, force, fix);

  // A disabled production KSpace call used to leave a prepared field waiting
  // for an unrelated later compute. Prove that init rejects it and that a
  // library caller can restore compute=yes and retry the same instance.
  input->one("kspace_modify compute no");
  int caught_compute_no = 0;
  try {
    input->one("run 0 post no");
  } catch (const LAMMPSException &exception) {
    caught_compute_no = std::string(exception.what()).find(
                            EXPECTED_COMPUTE_NO_FAILURE) != std::string::npos;
  }
  int all_caught_compute_no = 0;
  MPI_Allreduce(&caught_compute_no, &all_caught_compute_no, 1, MPI_INT,
                MPI_MIN, world);
  if (!all_caught_compute_no)
    error->all(FLERR,
               "DPRC failure-recovery test did not reject KSpace compute no");
  input->one("kspace_modify compute yes");
  input->one("run 0 post no");

  const ForceSnapshot baseline = capture_snapshot(atom, force, fix);
  double local_compute_no_retry_error = absolute_error(
      baseline.correction, initial_baseline.correction);
  for (int i = 0; i < atom->nlocal; ++i) {
    local_compute_no_retry_error = std::max(
        local_compute_no_retry_error,
        absolute_error(baseline.charges[i], initial_baseline.charges[i]));
    for (int dim = 0; dim < 3; ++dim)
      local_compute_no_retry_error = std::max(
          local_compute_no_retry_error,
          absolute_error(baseline.forces[3 * i + dim],
                         initial_baseline.forces[3 * i + dim]));
  }
  double compute_no_retry_error = 0.0;
  MPI_Allreduce(&local_compute_no_retry_error, &compute_no_retry_error, 1,
                MPI_DOUBLE, MPI_MAX, world);
  if (compute_no_retry_error > 1.0e-4)
    error->all(FLERR,
               "DPRC KSpace compute-no retry differs from the baseline: {}",
               compute_no_retry_error);

  fix->arm_failure_after_fused_prepare();

  int caught_expected = 0;
  try {
    // This command re-enters LAMMPS exactly as a library caller would. The
    // one-shot hook throws after the fused mesh is prepared; the derived fix
    // must discard that pending transaction before the exception escapes.
    input->one("run 0 post no");
  } catch (const LAMMPSException &exception) {
    caught_expected =
        std::string(exception.what()).find(EXPECTED_FAILURE) != std::string::npos;
  }

  int all_caught_expected = 0;
  MPI_Allreduce(&caught_expected, &all_caught_expected, 1, MPI_INT, MPI_MIN,
                world);
  if (!all_caught_expected)
    error->all(FLERR,
               "DPRC failure-recovery test did not catch the expected exception");

  // The wrapper snapshots charges and force-style scalars at pre_force entry.
  // Validate that state before retrying. Forces are intentionally excluded:
  // Verlet clears them before pre_force, so their transaction entry value is
  // zero rather than the preceding completed run's force.
  double local_rollback_error = 0.0;
  for (int i = 0; i < atom->nlocal; ++i)
    local_rollback_error = std::max(
        local_rollback_error,
        absolute_error(atom->q[i], baseline.charges[i]));
  local_rollback_error = std::max(
      local_rollback_error,
      absolute_error(force->pair->eng_vdwl, baseline.pair_vdwl));
  local_rollback_error = std::max(
      local_rollback_error,
      absolute_error(force->pair->eng_coul, baseline.pair_coulomb));
  local_rollback_error = std::max(
      local_rollback_error,
      absolute_error(force->kspace->energy, baseline.kspace_energy));
  for (int component = 0; component < 6; ++component) {
    local_rollback_error = std::max(
        local_rollback_error,
        absolute_error(force->pair->virial[component],
                       baseline.pair_virial[component]));
    local_rollback_error = std::max(
        local_rollback_error,
        absolute_error(force->kspace->virial[component],
                       baseline.kspace_virial[component]));
  }
  double rollback_error = 0.0;
  MPI_Allreduce(&local_rollback_error, &rollback_error, 1, MPI_DOUBLE, MPI_MAX,
                world);
  if (rollback_error != 0.0)
    error->all(FLERR,
               "DPRC failure-recovery rollback changed caller-visible state");

  // A stale pending bit would make the first MM-only capture below consume the
  // abandoned full field. A successful retry therefore proves both the local
  // discard and the base PPPM overwrite/recommunication invariant.
  input->one("run 0 post no");
  const ForceSnapshot retry = capture_snapshot(atom, force, fix);
  require_snapshot_match(error, atom, retry, baseline, world,
                         "post-prepare failure");

  fix->arm_failure_after_fused_commit();
  int caught_post_commit = 0;
  try {
    // The hook escapes after commit without running the transaction catch.
    // The next same-timestep pre_force entry must discard that orphan token
    // before its MM-only capture can call the private KSpace implementation.
    input->one("run 0 post no");
  } catch (const LAMMPSException &exception) {
    caught_post_commit =
        std::string(exception.what()).find(EXPECTED_POST_COMMIT_FAILURE) !=
        std::string::npos;
  }
  int all_caught_post_commit = 0;
  MPI_Allreduce(&caught_post_commit, &all_caught_post_commit, 1, MPI_INT,
                MPI_MIN, world);
  if (!all_caught_post_commit)
    error->all(
        FLERR,
        "DPRC failure-recovery test did not catch the post-commit exception");

  input->one("run 0 post no");
  const ForceSnapshot post_commit_retry = capture_snapshot(atom, force, fix);
  require_snapshot_match(error, atom, post_commit_retry, baseline, world,
                         "post-commit failure");
}
