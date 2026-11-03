// TurboQuant Q_prod combo for -ctk q5_1 -ctv turbo4q (added by hand).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_TURBO4Q_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_TURBO4Q_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_TURBO4Q_0);
