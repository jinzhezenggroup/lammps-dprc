#include <xtbloom/xtbloom.h>

#include <stddef.h>
#include <stdint.h>

int main(void) {
  if (XTBLOOM_API_VERSION == 0u)
    return 1;
  if (sizeof(xtbloom_backend_t) != sizeof(int32_t))
    return 2;
  if (offsetof(xtbloom_context_options_t, stream) != 24u)
    return 3;
  return 0;
}
