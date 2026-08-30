#include <deepmd/c_api.h>

#if DP_C_API_VERSION < 30
#error "LAMMPS-DPRC requires DeePMD C API version 30 or newer"
#endif

int main(void) {
#if DP_C_API_VERSION >= 31
  void (*batch_function)(DP_DeepPot *, double *, double *, double *,
                         const int64_t *, const uint32_t *, const float *,
                         const int64_t *, const int64_t *, const uint32_t *,
                         const int64_t *, const int64_t *, int, int64_t) =
      &DP_DeepPotComputeCanonicalGraphBatchGPU;
  return batch_function == 0;
#else
  void (*graph_function)(DP_DeepPot *, double *, double *, double *,
                         const int64_t *, const uint32_t *, const float *,
                         const int64_t *, const int64_t *, const uint32_t *,
                         int, int, int64_t) =
      &DP_DeepPotComputeCanonicalGraphGPU;
  return graph_function == 0;
#endif
}
