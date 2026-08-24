#include "command_dprc_info.h"
#ifdef DPRC_HAVE_DEEPMD_BATCH
#include "pair_dprc_deepmd_batch.h"
#endif
#ifdef DPRC_HAVE_XTBLOOM_FIX
#include "fix_dprc_xtb.h"
#include "kspace_dprc.h"
#ifdef DPRC_HAVE_CLASSICAL_BATCH
#include "fix_dprc_classical_batch.h"
#include "kspace_dprc_batched.h"
#include "pair_dprc_batched_lj.h"
#include "pair_dprc_batched_tip4p.h"
#endif
#ifdef DPRC_ENABLE_TEST_HOOKS
#include "command_dprc_test_recovery.h"
#endif
#endif

#include "lammpsplugin.h"
#include "version.h"

using namespace LAMMPS_NS;

namespace {

Command *create_dprc_info(LAMMPS *lmp) { return new CommandDPRCInfo(lmp); }

#ifdef DPRC_HAVE_DEEPMD_BATCH
Pair *create_dprc_deepmd_batch(LAMMPS *lmp) {
  return new PairDPRCDeepMDBatch(lmp);
}
#endif

#ifdef DPRC_HAVE_XTBLOOM_FIX
Fix *create_dprc_xtb(LAMMPS *lmp, int narg, char **arg) {
  return new FixDPRCXtb(lmp, narg, arg);
}

KSpace *create_pppm_dprc(LAMMPS *lmp) { return new PPPMDPRC(lmp); }

KSpace *create_pppm_tip4p_dprc(LAMMPS *lmp) {
  return new PPPMTIP4PDPRC(lmp);
}

#ifdef DPRC_HAVE_CLASSICAL_BATCH
Fix *create_dprc_classical_batch(LAMMPS *lmp, int narg, char **arg) {
  return new FixDPRCClassicalBatch(lmp, narg, arg);
}

Pair *create_lj_dprc_batch(LAMMPS *lmp) {
  return new PairDPRCBatchedLJ(lmp);
}

Pair *create_tip4p_dprc_batch(LAMMPS *lmp) {
  return new PairDPRCBatchedTIP4PLong(lmp);
}

KSpace *create_pppm_tip4p_dprc_batch(LAMMPS *lmp) {
  return new PPPMTIP4PDPRCBatched(lmp);
}
#endif

#ifdef DPRC_ENABLE_TEST_HOOKS
Command *create_dprc_test_recovery(LAMMPS *lmp) {
  return new CommandDPRCTestRecovery(lmp);
}
#endif
#endif

} // namespace

#if defined(_WIN32)
#define DPRC_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define DPRC_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define DPRC_PLUGIN_EXPORT extern "C"
#endif

// This is the plugin's only public dynamic symbol. LAMMPS resolves it by name;
// every implementation symbol remains hidden despite RTLD_GLOBAL loading.
DPRC_PLUGIN_EXPORT void lammpsplugin_init(void *lmp, void *handle,
                                          void *regfunc) {
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);

  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "command";
  plugin.name = "dprc/info";
  plugin.info = "LAMMPS-DPRC topology and ABI diagnostic command";
  plugin.author = "Jinzhe Zeng and contributors";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_dprc_info);
  plugin.handle = handle;
  register_plugin(&plugin, lmp);

#ifdef DPRC_HAVE_DEEPMD_BATCH
  plugin.style = "pair";
  plugin.name = "dprc/deepmd/batch";
  plugin.info = "GPU-local partition-batched DPRc through the DeePMD C API";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_dprc_deepmd_batch);
  register_plugin(&plugin, lmp);

  // The implementation is a host LAMMPS adapter around one CUDA model owner,
  // but the explicit alias lets it compose naturally inside hybrid/overlay/kk
  // and under the standard -sf kk command-line suffix.
  plugin.name = "dprc/deepmd/batch/kk";
  register_plugin(&plugin, lmp);
#endif

#ifdef DPRC_HAVE_XTBLOOM_FIX
  plugin.style = "fix";
  plugin.name = "qmmm/xtb/dprc";
  plugin.info = "Batched xTBloom QM/MM with pinned LAMMPS electrostatics";
  plugin.creator.v2 =
      reinterpret_cast<lammpsplugin_factory2 *>(&create_dprc_xtb);
  register_plugin(&plugin, lmp);

  plugin.style = "kspace";
  plugin.name = "pppm/dprc";
  plugin.info = "Fused MM/QM PPPM for qmmm/xtb/dprc";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_pppm_dprc);
  register_plugin(&plugin, lmp);

  plugin.name = "pppm/tip4p/dprc";
  plugin.info = "Fused TIP4P MM/QM PPPM for qmmm/xtb/dprc";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_pppm_tip4p_dprc);
  register_plugin(&plugin, lmp);

#ifdef DPRC_HAVE_CLASSICAL_BATCH
  plugin.style = "fix";
  plugin.name = "dprc/classical/batch";
  plugin.info = "PRE_FORCE coordinator for shared batched CUDA classical forces";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(
      &create_dprc_classical_batch);
  register_plugin(&plugin, lmp);

  plugin.style = "pair";
  plugin.name = "lj/cut/dprc/batch";
  plugin.info = "Publication proxy for batched CUDA Lennard-Jones";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_lj_dprc_batch);
  register_plugin(&plugin, lmp);

  plugin.name = "tip4p/long/dprc/batch";
  plugin.info = "Publication proxy for batched CUDA TIP4P Coulomb";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_tip4p_dprc_batch);
  register_plugin(&plugin, lmp);

  plugin.style = "kspace";
  plugin.name = "pppm/tip4p/dprc/batch";
  plugin.info = "Single-owner batched CUDA triclinic TIP4P PPPM";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(
      &create_pppm_tip4p_dprc_batch);
  register_plugin(&plugin, lmp);
#endif

#ifdef DPRC_ENABLE_TEST_HOOKS
  plugin.style = "command";
  plugin.name = "dprc/test/qmmm_failure_recovery";
  plugin.info = "Test-only fused QM/MM failure transaction check";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_dprc_test_recovery);
  register_plugin(&plugin, lmp);
#endif
#endif
}

#undef DPRC_PLUGIN_EXPORT
