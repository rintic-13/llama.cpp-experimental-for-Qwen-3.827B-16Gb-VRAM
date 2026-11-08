// TurboQuant combo for -ctk turbo4 -ctv turbo5 (added by hand 2026-11-08, batch F).
// K: vec_dot_fattn_vec_KQ_turbo4_0 (S2 fused path), V: dequantize_V_turbo5_0 per-block inverse FHT.
// No D=64 instance: QK_TURBO4/QK_TURBO5 = 128 and turbo5 V needs full-warp FHT ownership (see fattn.cu).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO5_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO5_0);
