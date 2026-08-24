#include <deepmd/c_api.h>

#if DP_C_API_VERSION < 31
#error "LAMMPS-DPRC requires DeePMD C API version 31 or newer"
#endif

int main(void) {
  void (*batch_function)(DP_DeepPot *, double *, double *, double *,
                         const int64_t *, const uint32_t *, const float *,
                         const int64_t *, const int64_t *, const uint32_t *,
                         const int64_t *, const int64_t *, int, int64_t) =
      &DP_DeepPotComputeCanonicalGraphBatchGPU;
  return batch_function == 0;
}
