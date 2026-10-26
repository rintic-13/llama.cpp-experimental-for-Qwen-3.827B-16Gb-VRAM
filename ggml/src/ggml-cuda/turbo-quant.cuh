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

// ---- Inline dequant helpers: extract one float from a block (rotated domain) ----

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

// ---- dequantize_block-style entry points (2 consecutive elements, like q8_0) ----
// Used by convert.cu (to_fp16 / to_fp32 / FA f16-scratch fallback).

static __device__ __forceinline__ void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo2_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo2_dequant_element(x + ib, iqs + 1, norm);
}

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo3_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo3_dequant_element(x + ib, iqs + 1, norm);
}

static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo4_dequant_element(x + ib, iqs + 0, norm);
    v.y = turbo4_dequant_element(x + ib, iqs + 1, norm);
}
