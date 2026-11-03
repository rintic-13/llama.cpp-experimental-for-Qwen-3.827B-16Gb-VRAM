// NVFP4-like KV cache (Stage A): -ctk nvfp4 -ctv nvfp4. Single-level NVFP4: per-16 UE4M3 scale +
// packed E2M1 nibbles, unrotated, no dither. QK_NVFP4 = 64 divides every supported head size,
// including the 64-element K row.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
