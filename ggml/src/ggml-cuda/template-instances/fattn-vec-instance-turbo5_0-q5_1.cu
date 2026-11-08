// TurboQuant combo for -ctk turbo5 -ctv q5_1 (added by hand 2026-11-08, batch F).
// K: vec_dot_fattn_vec_KQ_turbo5_0 raw-domain dot, V: generic Q5_1 vec KV dot.
// No D=64 instance: QK_TURBO5 = 128 (see fattn.cu turbo5 comment).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO5_0, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO5_0, GGML_TYPE_Q5_1);
