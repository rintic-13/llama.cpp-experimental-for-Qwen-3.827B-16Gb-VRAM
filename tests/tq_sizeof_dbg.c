#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

int main(void) {
    printf("QK_TURBO2=%d QK_TURBO3=%d QK_TURBO4=%d  QK_K=%d\n", QK_TURBO2, QK_TURBO3, QK_TURBO4, QK_K);
    printf("sizeof(block_turbo2_0)=%zu sizeof(block_turbo3_0)=%zu sizeof(block_turbo4_0)=%zu\n",
           sizeof(block_turbo2_0), sizeof(block_turbo3_0), sizeof(block_turbo4_0));
    printf("offsetof(block_turbo2_0, qs)=%zu\n", offsetof(block_turbo2_0, qs));
    printf("raw t2 (norm+qs[8])=%zu\n", sizeof(struct { ggml_half norm; uint8_t qs[8]; }));
    return 0;
}
