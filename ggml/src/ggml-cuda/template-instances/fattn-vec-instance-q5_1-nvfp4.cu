// NVFP4-like KV cache (Stage A): -ctk q5_1 -ctv nvfp4 (K quality dominated by q5_1, NVFP4-like V).
// V dequant: per-16 UE4M3 scale + packed E2M1 nibbles, unrotated, no dither.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_NVFP4);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_NVFP4);
