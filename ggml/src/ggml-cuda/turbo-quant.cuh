/*
 * TurboQuant CUDA constants and device helpers - port of TheTom/llama-cpp-turboquant.
 *
 * Subset for this fork: TURBO3_0 (3-bit PolarQuant) and TURBO4_0 (4-bit PolarQuant) at
 * fixed block/WHT-group size 128 (head_dim 256 % 128 == 0 on Dirk-Qwen3.8-27B). Dropped
 * from upstream: turbo5/6, the legacy 3-bit+QJL turbo4 variant, innerq per-channel
 * equalization, 64-element WHT groups.
 *
 * The centroid / midpoint / WHT sign tables are bit-identical to ggml-turbo-quant.c so
 * CPU and CUDA quantization produce identical blocks. Dequant stays in the WHT-rotated
 * domain: the graph pre-rotates Q and applies the inverse WHT to the attention output.
 */

#pragma once

#include "common.cuh"

// ---- Quantization ratios for dequantize_block (2 elements per call, like q8_0) ----

#define QR_TURBO2 1
#define QR_TURBO3 1
#define QR_TURBO4 1
#define QR_TURBO5 1
#define QR_TURBO3Q 1
#define QR_TURBO4Q 1

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)), kept for the dequant path only ----

static __constant__ float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

static __constant__ float TURBO_MID_3BIT[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f
};

// ---- 4-bit centroids (16 optimal centroids for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f,
    -0.070671f, -0.046174f, -0.022824f,  0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,
     0.127026f,  0.162947f,  0.212203f
};

// ---- 5-bit centroids (32 Lloyd-Max-style centroids for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_5BIT[32] = {
    -0.262203f, -0.241529f, -0.212203f, -0.182877f,
    -0.162947f, -0.143016f, -0.127026f, -0.111036f,
    -0.097164f, -0.083292f, -0.070671f, -0.058050f,
    -0.046174f, -0.034299f, -0.022824f, -0.011349f,
     0.000000f,  0.011349f,  0.022824f,  0.034299f,
     0.046174f,  0.058050f,  0.070671f,  0.083292f,
     0.097164f,  0.111036f,  0.127026f,  0.143016f,
     0.162947f,  0.182877f,  0.212203f,  0.241529f
};

// FIX: boundaries must be midpoints of adjacent TURBO_CENTROIDS_5BIT entries
// ((T5[i]+T5[i+1])/2). The old table drifted up to 0.69 sigma at the edges
// (last boundary 0.165610 instead of 0.226866), inflating 5-bit MSE by ~7x
// and making turbo5 quantitatively worse than turbo4. 2026-10-28.
static __constant__ float TURBO_MID_5BIT[31] = {
    -0.251866f, -0.226866f, -0.197540f, -0.172912f,
    -0.152981f, -0.135021f, -0.119031f, -0.104100f,
    -0.090228f, -0.076982f, -0.064361f, -0.052112f,
    -0.040237f, -0.028562f, -0.017087f, -0.005674f,
     0.005674f,  0.017087f,  0.028562f,  0.040237f,
     0.052112f,  0.064361f,  0.076982f,  0.090228f,
     0.104100f,  0.119031f,  0.135021f,  0.152981f,
     0.172912f,  0.197540f,  0.226866f
};

// ---- WHT sign arrays (seed=42), same values as turbo_cpu_s1/turbo_cpu_s2 ----

static __constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
     1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2[128] = {
     1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- Nearest centroid index (threshold search on codebook midpoints, same as CPU) ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < -0.086728f) return 0;
    else if (val <  0.000000f) return 1;
    else if (val <  0.086728f) return 2;
    else                       return 3;
}

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
    if      (val < TURBO_MID_4BIT[ 0]) return  0;
    else if (val < TURBO_MID_4BIT[ 1]) return  1;
    else if (val < TURBO_MID_4BIT[ 2]) return  2;
    else if (val < TURBO_MID_4BIT[ 3]) return  3;
    else if (val < TURBO_MID_4BIT[ 4]) return  4;
    else if (val < TURBO_MID_4BIT[ 5]) return  5;
    else if (val < TURBO_MID_4BIT[ 6]) return  6;
    else if (val < TURBO_MID_4BIT[ 7]) return  7;
    else if (val < TURBO_MID_4BIT[ 8]) return  8;
    else if (val < TURBO_MID_4BIT[ 9]) return  9;
    else if (val < TURBO_MID_4BIT[10]) return 10;
    else if (val < TURBO_MID_4BIT[11]) return 11;
    else if (val < TURBO_MID_4BIT[12]) return 12;
    else if (val < TURBO_MID_4BIT[13]) return 13;
    else if (val < TURBO_MID_4BIT[14]) return 14;
    else                               return 15;
}

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_5bit(float val) {
    if      (val < TURBO_MID_5BIT[ 0]) return  0;
    else if (val < TURBO_MID_5BIT[ 1]) return  1;
    else if (val < TURBO_MID_5BIT[ 2]) return  2;
    else if (val < TURBO_MID_5BIT[ 3]) return  3;
    else if (val < TURBO_MID_5BIT[ 4]) return  4;
    else if (val < TURBO_MID_5BIT[ 5]) return  5;
    else if (val < TURBO_MID_5BIT[ 6]) return  6;
    else if (val < TURBO_MID_5BIT[ 7]) return  7;
    else if (val < TURBO_MID_5BIT[ 8]) return  8;
    else if (val < TURBO_MID_5BIT[ 9]) return  9;
    else if (val < TURBO_MID_5BIT[10]) return 10;
    else if (val < TURBO_MID_5BIT[11]) return 11;
    else if (val < TURBO_MID_5BIT[12]) return 12;
    else if (val < TURBO_MID_5BIT[13]) return 13;
    else if (val < TURBO_MID_5BIT[14]) return 14;
    else if (val < TURBO_MID_5BIT[15]) return 15;
    else if (val < TURBO_MID_5BIT[16]) return 16;
    else if (val < TURBO_MID_5BIT[17]) return 17;
    else if (val < TURBO_MID_5BIT[18]) return 18;
    else if (val < TURBO_MID_5BIT[19]) return 19;
    else if (val < TURBO_MID_5BIT[20]) return 20;
    else if (val < TURBO_MID_5BIT[21]) return 21;
    else if (val < TURBO_MID_5BIT[22]) return 22;
    else if (val < TURBO_MID_5BIT[23]) return 23;
    else if (val < TURBO_MID_5BIT[24]) return 24;
    else if (val < TURBO_MID_5BIT[25]) return 25;
    else if (val < TURBO_MID_5BIT[26]) return 26;
    else if (val < TURBO_MID_5BIT[27]) return 27;
    else if (val < TURBO_MID_5BIT[28]) return 28;
    else if (val < TURBO_MID_5BIT[29]) return 29;
    else if (val < TURBO_MID_5BIT[30]) return 30;
    else                               return 31;
}

// ---- Inline dequant helpers: extract one float from a block (rotated domain) ----
// The *_dithered variants are for blocks written by k_set_rows_turbo3/4 with the subtractive
// dither: the block pointer is the block itself (row-relative indexing is the caller's business)
// and `seed` must be the LINEAR block index inside the cache tensor - the exact number
// k_set_rows_turbo3/4 hashed (see the dither contract in ggml-common.h).

#ifdef GGML_TURBO_DITHER
// Subtract a PREGENERATED u (the batched hot-path pattern: turbo_dither_range fills u[] once
// per 4-coordinate group, extraction stays single-source-of-truth here).
static __device__ __forceinline__ float turbo3_dequant_element_u(
        const block_turbo3_0 * __restrict__ blk, int j, float norm, float u) {
    const uint8_t low2 = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
    return (TURBO_CENTROIDS_3BIT[low2 | (hi1 << 2)] - u) * norm;
}

static __device__ __forceinline__ float turbo4_dequant_element_u(
        const block_turbo4_0 * __restrict__ blk, int j, float norm, float u) {
    const uint8_t idx = (blk->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return (TURBO_CENTROIDS_4BIT[idx] - u) * norm;
}

static __device__ __forceinline__ float turbo3_dequant_element_dithered(
        const block_turbo3_0 * __restrict__ blk, const uint32_t seed, int j, float norm) {
    return turbo3_dequant_element_u(blk, j, norm, turbo_dither(seed, j, TURBO_DITHER_STEP_3BIT));
}

static __device__ __forceinline__ float turbo4_dequant_element_dithered(
        const block_turbo4_0 * __restrict__ blk, const uint32_t seed, int j, float norm) {
    return turbo4_dequant_element_u(blk, j, norm, turbo_dither(seed, j, TURBO_DITHER_STEP_4BIT));
}

// Batched u fetch for `ne` consecutive coordinates of the same block (FA vec kernel V-dequant
// hot path): one turbo_dither4 per 4-coordinate group, +1 only when the range crosses a group
// boundary (j0 is even and ne is 2 or 4, so that happens just for j0 % 4 == 2 && ne == 4).
static __device__ __forceinline__ void turbo_dither_range(
        const uint32_t block_idx, const int j0, const int ne, const float step, float * out) {
    float a[4];
    turbo_dither4(block_idx, j0 >> 2, step, a);
    const int r = j0 & 3;
    if (r + ne <= 4) {
#pragma unroll
        for (int l = 0; l < ne; ++l) out[l] = a[r + l];
    } else {
#pragma unroll
        for (int l = 0; l < 4 - r; ++l) out[l] = a[r + l];
        float b[4];
        turbo_dither4(block_idx, (j0 >> 2) + 1, step, b);
#pragma unroll
        for (int l = 4 - r; l < ne; ++l) out[l] = b[l - (4 - r)];
    }
}
#endif // GGML_TURBO_DITHER

static __device__ __forceinline__ float turbo2_dequant_element(
        const block_turbo2_0 * __restrict__ x, int j, float norm) {
    const uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    const uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    return TURBO_CENTROIDS_3BIT[low2 | (hi1 << 2)] * norm;
}

static __device__ __forceinline__ float turbo4_dequant_element(
        const block_turbo4_0 * __restrict__ x, int j, float norm) {
    const uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

static __device__ __forceinline__ float turbo5_dequant_element(
        const block_turbo5_0 * __restrict__ x, int j, float norm) {
    const uint8_t lo  = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    const uint8_t hi  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    return TURBO_CENTROIDS_5BIT[lo | (hi << 4)] * norm;
}

// Q_prod dequant: (b-1)-bit codebook + 1-bit QJL residual sign (see ggml-common.h).
// norm and rnorm are fp16 block fields; the QJL term is sign * TURBO_QJL_ALPHA * rnorm.

static __device__ __forceinline__ float turbo3q_dequant_element(
        const block_turbo3q_0 * __restrict__ x, int j, float norm, float qjl_scale) {
    const uint8_t idx  = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t sign = (x->signs[j / 8] >> (j % 8)) & 0x1;
    const float   qjl  = (sign ? 1.0f : -1.0f) * qjl_scale;
    return (TURBO_CENTROIDS_2BIT[idx] + qjl) * norm;
}

static __device__ __forceinline__ float turbo4q_dequant_element(
        const block_turbo4q_0 * __restrict__ x, int j, float norm, float qjl_scale) {
    const uint8_t lo   = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi   = (x->hi1[j / 8] >> (j % 8)) & 0x1;
    const uint8_t sign = (x->signs[j / 8] >> (j % 8)) & 0x1;
    const float   qjl  = (sign ? 1.0f : -1.0f) * qjl_scale;
    return (TURBO_CENTROIDS_3BIT[lo | (hi << 2)] + qjl) * norm;
}

// ---- dequantize_block-style entry points (2 consecutive elements, like q8_0) ----
// Used by convert.cu (to_fp16 / to_fp32 / FA f16-scratch fallback: both the MMA_F16 prefill
// kernel and the vec-kernel fallback decode turbo cache blocks through these entries).
// Dither scope (2026-10-31 A/B): turbo3_0 and turbo4_0 blocks carry the subtractive dither -
// remove it here using `ib`, the linear block index inside the tensor being converted, the same
// seed k_set_rows_turbo3/4 used (see ggml-common.h). turbo2/5 and Q_prod blocks are dither-free.
// The FA vector kernel's own dequant path (fattn-common.cuh) applies the same dither natively,
// so no f16 scratch fallback is needed for turbo3/turbo4 combos.

static __device__ __forceinline__ void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo2_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo2_dequant_element(x + ib, iqs + 1, norm);
}

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
#ifdef GGML_TURBO_DITHER
    v.x = turbo3_dequant_element_dithered(x + ib, (uint32_t) ib, iqs + 0, norm);
    v.y = turbo3_dequant_element_dithered(x + ib, (uint32_t) ib, iqs + 1, norm);
#else
    v.x = turbo3_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo3_dequant_element(x + ib, iqs + 1, norm);
#endif
}

static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float norm = __half2float(x[ib].norm);
#ifdef GGML_TURBO_DITHER
    v.x = turbo4_dequant_element_dithered(x + ib, (uint32_t) ib, iqs + 0, norm);
    v.y = turbo4_dequant_element_dithered(x + ib, (uint32_t) ib, iqs + 1, norm);
#else
    v.x = turbo4_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo4_dequant_element(x + ib, iqs + 1, norm);
#endif
}

static __device__ __forceinline__ void dequantize_turbo5_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo5_0 * x = (const block_turbo5_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo5_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo5_dequant_element(x + ib, iqs + 1, norm);
}

static __device__ __forceinline__ void dequantize_turbo3q_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3q_0 * x = (const block_turbo3q_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    const float qjl_scale = TURBO_QJL_ALPHA * __half2float(x[ib].rnorm);
    v.x = turbo3q_dequant_element(x + ib, iqs + 0, norm, qjl_scale);
    v.y = turbo3q_dequant_element(x + ib, iqs + 1, norm, qjl_scale);
}

static __device__ __forceinline__ void dequantize_turbo4q_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4q_0 * x = (const block_turbo4q_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    const float qjl_scale = TURBO_QJL_ALPHA * __half2float(x[ib].rnorm);
    v.x = turbo4q_dequant_element(x + ib, iqs + 0, norm, qjl_scale);
    v.y = turbo4q_dequant_element(x + ib, iqs + 1, norm, qjl_scale);
}
