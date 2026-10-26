/*
 * TurboQuant: KV cache compression via PolarQuant (WHT rotation + Lloyd-Max codebooks)
 * Port of TheTom/llama-cpp-turboquant (arXiv 2504.19874), CPU codec subset:
 *   GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit), GGML_TYPE_TURBO4_0 (4-bit).
 *
 * Deliberately dropped from the upstream port:
 *   - QJL residual correction (legacy turbo4 3-bit+QJL variant) - the upstream default
 *     is already pure 4-bit PolarQuant (TURBO4_USE_4BIT=1), so no rotation/QJL matrices
 *     or rnorm field are needed here.
 *   - innerq per-channel scale (turbo-innerq) - not used at block size 128 here.
 *   - runtime group-size global (turbo3_cpu_wht_group_size): the local fork always has
 *     head_dim % 128 == 0 (27B model: head_dim 256), so the group is fixed to 128.
 *
 * Rotation: y = D(s2) * (H / sqrt(128)) * D(s1) * x, signs seeded 42, byte-for-byte the
 * upstream turbo_cpu_s1/s2 arrays; must stay identical to the CUDA k_set_rows_turboN and
 * graph-side ggml_turbo_wht sign arrays. Dequant stays in the rotated domain (the graph
 * pre-rotates Q and applies the inverse WHT on the attention output).
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

/* ---------- centroids (Lloyd-Max for N(0, 1/128), scaled by 1/sqrt(128)) ---------- */

/* 2-bit: {±0.453, ±1.51} / sqrt(128) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

/* 4-bit: 16 optimal centroids for N(0, 1/128) */
static const float CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

/* ---------- nearest centroid (threshold search on codebook midpoints) ---------- */

static int nearest_centroid_2bit(float val) {
    /* midpoints of {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;
    if (val <  0.000000f) return 1;
    if (val <  0.086728f) return 2;
    return 3;
}

static int nearest_centroid_3bit(float val) {
    if (val < -0.154496f) return 0;
    if (val < -0.092804f) return 1;
    if (val < -0.044243f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.044243f) return 4;
    if (val <  0.092804f) return 5;
    if (val <  0.154496f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    if (val < -0.212203f) return 0;
    if (val < -0.162947f) return 1;
    if (val < -0.127026f) return 2;
    if (val < -0.097164f) return 3;
    if (val < -0.070671f) return 4;
    if (val < -0.046174f) return 5;
    if (val < -0.022824f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.022824f) return 8;
    if (val <  0.046174f) return 9;
    if (val <  0.070671f) return 10;
    if (val <  0.097164f) return 11;
    if (val <  0.127026f) return 12;
    if (val <  0.162947f) return 13;
    if (val <  0.212203f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal + graph rotation, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- sign table accessor for the graph side (llama-kv-cache.cpp) ----------
 * the graph builds signed Hadamard matrices D(s1)*H*D(s2)/sqrt(128) (inverse WHT on the
 * attention output) and D(s2)*H*D(s1)/sqrt(128) (Q pre-rotation) from the very same signs
 * that the block quantizers apply inside turbo_cpu_fwht / k_set_rows_turboN */
GGML_API void ggml_turbo_wht_signs(const float ** s1, const float ** s2) {
    *s1 = turbo_cpu_s1;
    *s2 = turbo_cpu_s2;
}

/* ---------- CPU forward WHT (in-place, 128-element groups) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- CPU inverse WHT (in-place, 128-element groups) ----------
 *
 * Forward is  y = D(s2) * N * H * D(s1) * x   (N = 1/sqrt(group_size))
 * H is the unnormalized Hadamard butterfly with H*H = group_size * I, so
 * (N*H) is self-inverse.  s1 and s2 are ±1 diagonals, also self-inverse.
 * The inverse therefore has the same structure with s1 and s2 swapped:
 *     x = D(s1) * N * H * D(s2) * y
 */
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size);
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs2 (undoes the s2 that was applied last in the forward pass)
    for (int i = 0; i < group_size; i++) x[i] *= s2[i];

    // butterfly stages (same as forward - self-inverse up to the inv_sqrt scaling below)
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs1
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s1[i];
}

/* ---------- TURBO2_0: 2-bit PolarQuant ---------- */

GGML_API void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = (int)(k / QK_TURBO2);

    for (int b = 0; b < nb; b++) {
        float buf[QK_TURBO2];
        float norm_sq, grp_norm;

        // norm + normalize + forward WHT
        norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO2; j++) {
            buf[j] = x[b * QK_TURBO2 + j];
            norm_sq += buf[j] * buf[j];
        }
        grp_norm = sqrtf(norm_sq);
        const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
        for (int j = 0; j < QK_TURBO2; j++) buf[j] *= inv_norm;
        turbo_cpu_fwht(buf, QK_TURBO2);

        // quantize + pack (4 indices per byte), collecting recon norm for correction
        float recon_sq = 0.0f;
        memset(y[b].qs, 0, QK_TURBO2 / 4);
        for (int j = 0; j < QK_TURBO2; j++) {
            const int idx = nearest_centroid_2bit(buf[j]);
            y[b].qs[j / 4] |= (uint8_t)((idx & 0x3) << ((j % 4) * 2));
            recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
        }

        // corrected norm: dequantized block rescaled by norm/sqrt(sum c^2) matches ||x||
        const float recon_norm = sqrtf(recon_sq);
        const float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        y[b].norm = GGML_FP32_TO_FP16(corrected);
    }
}

GGML_API void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = (int)(k / QK_TURBO2);
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            const uint8_t idx = (x[block].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

/* ---------- TURBO3_0: 3-bit PolarQuant (MSE-only, no QJL) ---------- */

GGML_API void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = (int)(k / QK_TURBO3);

    for (int b = 0; b < nb; b++) {
        float buf[QK_TURBO3];
        float norm_sq, grp_norm;

        norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) {
            buf[j] = x[b * QK_TURBO3 + j];
            norm_sq += buf[j] * buf[j];
        }
        grp_norm = sqrtf(norm_sq);
        const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) buf[j] *= inv_norm;
        turbo_cpu_fwht(buf, QK_TURBO3);

        // 3-bit index split: low 2 bits in qs[], high bit in signs[]
        float recon_sq = 0.0f;
        memset(y[b].qs,    0, QK_TURBO3 / 4);
        memset(y[b].signs, 0, QK_TURBO3 / 8);
        for (int j = 0; j < QK_TURBO3; j++) {
            const int idx = nearest_centroid_3bit(buf[j]);
            y[b].qs[j / 4]    |= (uint8_t)((idx & 0x3) << ((j % 4) * 2));
            y[b].signs[j / 8] |= (uint8_t)(((idx >> 2) & 0x1) << (j % 8));
            recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
        }

        const float recon_norm = sqrtf(recon_sq);
        const float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        y[b].norm = GGML_FP32_TO_FP16(corrected);
    }
}

GGML_API void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int nb = (int)(k / QK_TURBO3);
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            const uint8_t lo  = (x[block].qs[j / 4]    >> ((j % 4) * 2)) & 0x3;
            const uint8_t hi  = (x[block].signs[j / 8] >>  (j % 8))      & 0x1;
            const uint8_t idx = lo | (hi << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

/* ---------- TURBO4_0: 4-bit PolarQuant (no QJL, no rnorm) ---------- */

GGML_API void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = (int)(k / QK_TURBO4);

    for (int b = 0; b < nb; b++) {
        float buf[QK_TURBO4];
        float norm_sq, grp_norm;

        norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) {
            buf[j] = x[b * QK_TURBO4 + j];
            norm_sq += buf[j] * buf[j];
        }
        grp_norm = sqrtf(norm_sq);
        const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) buf[j] *= inv_norm;
        turbo_cpu_fwht(buf, QK_TURBO4);

        // 4-bit nibble packing: element i in low nibble of byte i/2 (upstream layout)
        float recon_sq = 0.0f;
        memset(y[b].qs, 0, QK_TURBO4 / 2);
        for (int j = 0; j < QK_TURBO4; j++) {
            const int idx = nearest_centroid_4bit(buf[j]);
            y[b].qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j % 2) * 4));
            recon_sq += CENTROIDS_4BIT[idx] * CENTROIDS_4BIT[idx];
        }

        const float recon_norm = sqrtf(recon_sq);
        const float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        y[b].norm = GGML_FP32_TO_FP16(corrected);
    }
}

GGML_API void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int nb = (int)(k / QK_TURBO4);
    // No inverse WHT here: dequant stays in the rotated domain, <Q_rot, K_rot> gives correct
    // attention scores (Q is pre-rotated in the graph); the inverse WHT is applied to the
    // attention output via GGML_OP_TURBO_WHT (stage 3).
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO4; j++) {
            const uint8_t idx = (x[block].qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            y[block * QK_TURBO4 + j] = CENTROIDS_4BIT[idx] * norm;
        }
    }
}

/* ---------- chunk quantizers (weight-style API, used by tests and CPU set_rows helpers) ---------- */

GGML_API size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                  int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    const size_t row_size = (size_t)((n_per_row / QK_TURBO2) * sizeof(block_turbo2_0));
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return (size_t)nrows * row_size;
}

GGML_API size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                  int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    const size_t row_size = (size_t)((n_per_row / QK_TURBO3) * sizeof(block_turbo3_0));
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return (size_t)nrows * row_size;
}

GGML_API size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                                  int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    const size_t row_size = (size_t)((n_per_row / QK_TURBO4) * sizeof(block_turbo4_0));
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return (size_t)nrows * row_size;
}
