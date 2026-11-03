// TurboQuant Q_prod combo for -ctk q5_1 -ctv turbo3q (added by hand).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3Q_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3Q_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3Q_0);
