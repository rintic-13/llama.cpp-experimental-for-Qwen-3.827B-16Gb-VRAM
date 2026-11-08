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

// ---- Register-mux codebook decoders (S1, 2026-11-02) ----
// The FA turbo hot path read TURBO_CENTROIDS_* one element per coordinate. The constant cache
// broadcasts ONE address per cycle, so with divergent per-lane indices a single LDC replays up
// to 16x: SASS forensics on the D=256 turbo4_0-turbo3_0 vec kernel showed LDC=537 (q4_1 integer
// dot etalon: 20). All turbo codebooks are sign-symmetric with strictly monotonic magnitudes
// (4-bit: c[idx] = +/-mag[idx&7], mag ascending; 3-bit: c = +/-mag[idx&3], mag DESCENDING in
// idx&3 on the negative half hence the k^3 reflection; 2-bit: same pattern; 5-bit: +/-mag over
// 16 magnitudes plus an explicit +0 at idx 16), so each value can be selected with an SEL tree
// in registers instead. The literals below are the exact table literals and IEEE negation is an
// exact sign-bit flip (all magnitudes nonzero), so the mux decoders return BITWISE-IDENTICAL
// floats to the LUT reads they replace. The static_asserts check every index against a literal
// mirror of the tables; the tools/fa_vec_ab_probe A/B additionally diffs old/new kernel output.
// The __constant__ tables stay for the quantizer (TURBO_MID_* boundaries) and any non-hot path.

static constexpr float TURBO_C2_REF[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static constexpr float TURBO_C3_REF[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

static constexpr float TURBO_C4_REF[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

static constexpr float TURBO_C5_REF[32] = {
    -0.262203f, -0.241529f, -0.212203f, -0.182877f,
    -0.162947f, -0.143016f, -0.127026f, -0.111036f,
    -0.097164f, -0.083292f, -0.070671f, -0.058050f,
    -0.046174f, -0.034299f, -0.022824f, -0.011349f,
     0.000000f,  0.011349f,  0.022824f,  0.034299f,
     0.046174f,  0.058050f,  0.070671f,  0.083292f,
     0.097164f,  0.111036f,  0.127026f,  0.143016f,
     0.162947f,  0.182877f,  0.212203f,  0.241529f
};

// Magnitude trees (explicit host+device constexpr so the static_asserts below can evaluate them
// on the host pass and the __device__ wrappers can call them without --expt-relaxed-constexpr).
static __host__ __device__ constexpr float turbo2_centroid_mag(const unsigned idx) {
    const unsigned t = (idx & 1) ^ ((idx & 2) ? 0u : 1u);
    return (t & 1) ? 0.133462f : 0.039994f;
}

static __host__ __device__ constexpr float turbo3_centroid_mag(const unsigned idx) {
    const unsigned t = (idx & 3) ^ ((idx & 4) ? 0u : 3u);   // n[t] ascending, n0=0.021663f
    const float e = (t & 1) ? 0.066822f : 0.021663f;
    const float f = (t & 1) ? 0.190207f : 0.118786f;
    return (t & 2) ? f : e;
}

static __host__ __device__ constexpr float turbo4_centroid_mag(const unsigned idx) {
    // |c| DESCENDS with idx over the NEGATIVE half (idx 0..7), ascends over the positive half
    // (idx 8..15): reflect idx&7 on the negative half (7-k = k^7), then the ascending tree.
    const unsigned t = (idx & 7) ^ ((idx & 8) ? 0u : 7u);
    const float p0 = (t & 1) ? 0.034299f : 0.011349f;
    const float p1 = (t & 1) ? 0.083292f : 0.058050f;
    const float p2 = (t & 1) ? 0.143016f : 0.111036f;
    const float p3 = (t & 1) ? 0.241529f : 0.182877f;
    const float q0 = (t & 2) ? p1 : p0;
    const float q1 = (t & 2) ? p3 : p2;
    return (t & 4) ? q1 : q0;
}

// 5-bit: magnitudes descend with the index on the negative half (M0=0.262203f .. M15=0.011349f);
// the positive half reflects them (T5[17-k] = M[k]) and idx 16 is the explicit +0.
static __host__ __device__ constexpr float turbo5_centroid_mag(const unsigned idx) {
    const unsigned k = idx & 15;
    const unsigned m = (idx & 16) ? 16u - k : k;            // m==16 <=> idx==16 (the +0 entry)
    const float a = (m & 1) ? 0.241529f : 0.262203f;
    const float b = (m & 1) ? 0.182877f : 0.212203f;
    const float c = (m & 1) ? 0.143016f : 0.162947f;
    const float d = (m & 1) ? 0.111036f : 0.127026f;
    const float e = (m & 1) ? 0.083292f : 0.097164f;
    const float f = (m & 1) ? 0.058050f : 0.070671f;
    const float g = (m & 1) ? 0.034299f : 0.046174f;
    const float h = (m & 1) ? 0.011349f : 0.022824f;
    const float i0 = (m & 2) ? b : a;
    const float i1 = (m & 2) ? d : c;
    const float i2 = (m & 2) ? f : e;
    const float i3 = (m & 2) ? h : g;
    const float j0 = (m & 4) ? i1 : i0;
    const float j1 = (m & 4) ? i3 : i2;
    return (m == 16u) ? 0.0f : ((m & 8) ? j1 : j0);
}

static constexpr float turbo_absf_(const float x) { return x < 0.0f ? -x : x; }

static constexpr bool turbo2_mux_ok(const int idx) {
    const float ref = TURBO_C2_REF[idx];
    const bool pos = (idx & 2) != 0;
    return turbo2_centroid_mag((unsigned) idx) == turbo_absf_(ref) && ((ref < 0.0f) != pos);
}
static constexpr bool turbo2_mux_ok_all() {
    for (int i = 0; i < 4; ++i) if (!turbo2_mux_ok(i)) return false;
    return true;
}
static constexpr bool turbo3_mux_ok_all() {
    for (int i = 0; i < 8; ++i) {
        const float ref = TURBO_C3_REF[i];
        if (turbo3_centroid_mag((unsigned) i) != turbo_absf_(ref)) return false;
        if ((ref < 0.0f) != ((i & 4) == 0)) return false;   // neg half <=> hi bit clear
    }
    return true;
}
static constexpr bool turbo4_mux_ok_all() {
    for (int i = 0; i < 16; ++i) {
        const float ref = TURBO_C4_REF[i];
        if (turbo4_centroid_mag((unsigned) i) != turbo_absf_(ref)) return false;
        if ((ref < 0.0f) != ((i & 8) == 0)) return false;
    }
    return true;
}
static constexpr bool turbo5_mux_ok_all() {
    for (int i = 0; i < 32; ++i) {
        const float ref = TURBO_C5_REF[i];
        if (turbo5_centroid_mag((unsigned) i) != turbo_absf_(ref)) return false;
        if ((ref < 0.0f) != ((i & 16) == 0)) return false;
    }
    return true;
}
static_assert(turbo2_mux_ok_all(), "turbo2 mux decoder differs from TURBO_CENTROIDS_2BIT");
static_assert(turbo3_mux_ok_all(), "turbo3 mux decoder differs from TURBO_CENTROIDS_3BIT");
static_assert(turbo4_mux_ok_all(), "turbo4 mux decoder differs from TURBO_CENTROIDS_4BIT");
static_assert(turbo5_mux_ok_all(), "turbo5 mux decoder differs from TURBO_CENTROIDS_5BIT");

// ONE sign-mask helper shared by the device wrappers and the host verification (the first
// S1 probe run caught this exact bug class: the device wrapper had the sign XOR inverted for
// the 2/3/4-bit tables - negative half is where the idx sign bit is CLEAR, tables ascend).
// shift: signbit 2->30, 4->29, 8->28, 16->27 so the surviving bit lands on bit31.
static __host__ __device__ constexpr uint32_t turbo_sign_mask_(const unsigned idx, const unsigned signbit, const int shift) {
    return ((uint32_t) ((idx & signbit) ^ signbit)) << shift;
}

static __device__ __forceinline__ float turbo2_centroid(const unsigned idx) {
    return __uint_as_float(__float_as_uint(turbo2_centroid_mag(idx)) ^ turbo_sign_mask_(idx, 2u, 30));
}
static __device__ __forceinline__ float turbo3_centroid(const unsigned idx) {
    return __uint_as_float(__float_as_uint(turbo3_centroid_mag(idx)) ^ turbo_sign_mask_(idx, 4u, 29));
}
static __device__ __forceinline__ float turbo4_centroid(const unsigned idx) {
    return __uint_as_float(__float_as_uint(turbo4_centroid_mag(idx)) ^ turbo_sign_mask_(idx, 8u, 28));
}
static __device__ __forceinline__ float turbo5_centroid(const unsigned idx) {  // idx < 32
    return __uint_as_float(__float_as_uint(turbo5_centroid_mag(idx)) ^ turbo_sign_mask_(idx, 16u, 27));
}

// ---- S8: byte-indexed turbo3 decode tables for the FA V path (GGML_CUDA_FATTN_VEC_VTAB) ----
//
// dequantize_V_turbo3_0 is only ever instantiated with ne == 4, and every caller passes an i0 that
// is a multiple of 4 (fattn-vec: i0 = 2*i_VKQ_0 + lane*4; lightning-indexer: lane*4 / i_embd*4).
// Four consecutive coords therefore share ONE qs byte (four 2-bit fields) and ONE signs byte (four
// 1-bit fields), which is what a table indexes.
//
// The sign is not a post-process either: turbo3_centroid_mag() itself reflects the magnitude index
// when the sign bit is clear (t = (idx&3) ^ (idx&4 ? 0 : 3)), so the sign bit is part of the index.
// All tables are built from turbo3_centroid_c(), so every form is bit-identical to turbo3_centroid(),
// and each table is cross-checked against the others by a static_assert.
//
// GGML_CUDA_FATTN_VEC_VTAB:
//   0 off (scalar mux decode)
//   1 one 8 KB float4 table: fewest instructions, but value-indexed gathers (up to ~20 L1 lines/load)
//   2 8-value scalar table (32 B = one cache line): most instructions, no gather
//   4 signed float2 pairs, 64 entries (512 B): 2 gathers of ~4 lines, ALU ~= form 1
//   3 per-instantiation choice made by fattn-vec: 2 where the K side uses its own table, else 1
//   5 same rule but 4 instead of 1 where the K side has no table
// Measured (FA_S5_S6_ATTRIBUTION_20261103.md SS2h-SS2j): form 1 wins where the K side still carries
// its mux trees, form 2 wins where the K side is table-ised - hence 3/5.  The final pick run (SS2j)
// then showed form 4 dominates BOTH corners - best form on every probe row, T6 -40.8% with S7,
// bitwise - and the auto rules (3/5) cost 9.9% on the S7 rows.  ACCEPTED 2026-11-03: default 4;
// 3/5 stay as forensics of the rejected hypothesis.
#ifndef GGML_CUDA_FATTN_VEC_VTAB
#define GGML_CUDA_FATTN_VEC_VTAB 4
#endif

#if GGML_CUDA_FATTN_VEC_VTAB
// turbo3_centroid() as a constexpr: the magnitude is positive, so an exact multiply by +-1
// reproduces the bit-31 XOR of turbo_sign_mask_() (no rounding involved).
static __host__ __device__ constexpr float turbo3_centroid_c(const unsigned idx) {
    return turbo3_centroid_mag(idx) * ((idx & 4u) != 0 ? 1.0f : -1.0f);
}

// GGML_CUDA_FATTN_VEC_VTAB == 1: ONE table, both halves: [0..255] is the hi1 = 0 half
// (-0.190207 .. -0.021663), [256..511] the hi1 = 1 half (+0.021663 .. +0.190207).  A single object
// means a single 64-bit base register in the decode and the second read is the same address plus an
// immediate 0x1000 - two separate arrays measurably spilled (STACK:8 in the D256/turbo5-K/turbo3-V
// dot instantiation).
struct turbo3_c4_tab {
    float4 v[512];   // v[(hi1<<8) | b].{x,y,z,w} = centroids of the four 2-bit fields of byte b
    __host__ __device__ constexpr turbo3_c4_tab() : v{} {
        for (unsigned hi1 = 0; hi1 < 2; ++hi1) {
            for (unsigned b = 0; b < 256; ++b) {
                float4 & e = v[(hi1 << 8) | b];
                e.x = turbo3_centroid_c(((b >> 0) & 3u) | (hi1 << 2));
                e.y = turbo3_centroid_c(((b >> 2) & 3u) | (hi1 << 2));
                e.z = turbo3_centroid_c(((b >> 4) & 3u) | (hi1 << 2));
                e.w = turbo3_centroid_c(((b >> 6) & 3u) | (hi1 << 2));
            }
        }
    }
};

static __device__ constexpr turbo3_c4_tab TURBO3_C4_TAB{};

// GGML_CUDA_FATTN_VEC_VTAB == 2 (S8c): the same values in the shape of the value domain.  There are
// only 8 distinct turbo3 centroids, so this table is 32 B = one cache line, and every lane's read is
// a broadcast out of that line instead of a gather out of 8 KB.  Costs more instructions (4 LDG.32 +
// index math per byte instead of 2 LDG.128 + 4 SEL) and buys the L1 wavefronts back.  Built by a
// constexpr constructor like the float4 table so the same static_assert can check it.
struct turbo3_c8_tab {
    float v[8];
    __host__ __device__ constexpr turbo3_c8_tab() : v{} {
        for (unsigned i = 0; i < 8; ++i) {
            v[i] = turbo3_centroid_c(i);
        }
    }
};

static __device__ constexpr turbo3_c8_tab TURBO3_C8_TAB{};

// GGML_CUDA_FATTN_VEC_VTAB == 4 (S8e): signed float2 pairs, 64 entries = 512 B = FOUR cache lines.
// The point is that turbo3_centroid_mag() itself reflects the magnitude index when the sign bit is
// clear (t = (idx&3) ^ (idx&4 ? 0 : 3)), so the sign is part of the *index*, not a post-process -
// which lets two coords share one 6-bit index (4 magnitude bits + 2 sign bits).  Two LDG.64 per byte
// with roughly the same ALU as the float4 form, at ~4 touched lines per gather instead of ~20.
struct turbo3_c2_tab {
    float2 v[64];   // v[(s1<<5)|(s0<<4)|(f1<<2)|f0] = signed centroids of fields 0 and 1
    __host__ __device__ constexpr turbo3_c2_tab() : v{} {
        for (unsigned n = 0; n < 64; ++n) {
            const unsigned f0 = (n >> 0) & 3u, f1 = (n >> 2) & 3u;
            const unsigned s0 = (n >> 4) & 1u, s1 = (n >> 5) & 1u;
            v[n].x = turbo3_centroid_c(f0 | (s0 << 2));
            v[n].y = turbo3_centroid_c(f1 | (s1 << 2));
        }
    }
};

static __device__ constexpr turbo3_c2_tab TURBO3_C2_TAB{};

// Compile-time proof that the table IS the reference codebook.  It checks a host copy of the
// object (a __device__ constexpr object is not host-readable), which is exactly what needs
// checking: the constructor, and the index derivation that replaces the scalar one.
static constexpr bool turbo3_centroid_c_ok() {
    for (unsigned i = 0; i < 8; ++i) {
        if (turbo3_centroid_c(i) != TURBO_C3_REF[i]) return false;
    }
    return true;
}
static constexpr bool turbo3_ctab_ok(const turbo3_c4_tab & t) {
    for (unsigned hi1 = 0; hi1 < 2; ++hi1) {
        for (unsigned b = 0; b < 256; ++b) {
            const float4 & e = t.v[(hi1 << 8) | b];
            const float   got[4]  = { e.x, e.y, e.z, e.w };
            const float  want[4]  = { TURBO_C3_REF[((b >> 0) & 3u) | (hi1 << 2)],
                                      TURBO_C3_REF[((b >> 2) & 3u) | (hi1 << 2)],
                                      TURBO_C3_REF[((b >> 4) & 3u) | (hi1 << 2)],
                                      TURBO_C3_REF[((b >> 6) & 3u) | (hi1 << 2)] };
            for (int i = 0; i < 4; ++i) {
                if (got[i] != want[i]) return false;
            }
        }
    }
    return true;
}
static constexpr turbo3_c4_tab TURBO3_C4_TAB_CHECK{};
// Cross-check, not a duplicate of the assert above: the S8c scalar path and the S8 float4 path
// derive the codebook index differently (`(qb >> 2l & 3) | (sb >> l & 1) << 2` vs a byte-indexed
// entry plus a |256 half select).  Both must land on the same codebook entry, so the scalar table is
// compared against the float4 table built by a different constructor, not against TURBO_C3_REF again.
static constexpr bool turbo3_c8tab_ok(const turbo3_c8_tab & t, const turbo3_c4_tab & q) {
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned b    = i & 3u;                    // field 0 of byte b holds code i & 3
        const unsigned hi1  = (i & 4u) >> 2;             // the sign bit is the table half
        if (t.v[i] != q.v[(hi1 << 8) | b].x) return false;
    }
    return true;
}
static constexpr turbo3_c8_tab TURBO3_C8_TAB_CHECK{};
// Same cross-check idea for S8e: its 6-bit index must select the same centroid the float4 table
// selects from the byte form, including the magnitude reflection that the sign bit causes.
static constexpr bool turbo3_c2tab_ok(const turbo3_c2_tab & t, const turbo3_c4_tab & q) {
    for (unsigned n = 0; n < 64; ++n) {
        const unsigned f0 = (n >> 0) & 3u, f1 = (n >> 2) & 3u;
        const unsigned s0 = (n >> 4) & 1u, s1 = (n >> 5) & 1u;
        const unsigned b  = f0 | (f1 << 2);
        if (t.v[n].x != q.v[(s0 << 8) | b].x) return false;
        if (t.v[n].y != q.v[(s1 << 8) | b].y) return false;
    }
    return true;
}
static constexpr turbo3_c2_tab TURBO3_C2_TAB_CHECK{};
static_assert(turbo3_centroid_c_ok(),    "turbo3_centroid_c differs from TURBO_C3_REF");
static_assert(turbo3_ctab_ok(TURBO3_C4_TAB_CHECK), "S8 turbo3 V centroid table differs from TURBO_C3_REF");
static_assert(turbo3_c8tab_ok(TURBO3_C8_TAB_CHECK, TURBO3_C4_TAB_CHECK),
              "S8c scalar V table disagrees with the S8 float4 table");
static_assert(turbo3_c2tab_ok(TURBO3_C2_TAB_CHECK, TURBO3_C4_TAB_CHECK),
              "S8e float2 V table disagrees with the S8 float4 table");
#endif // GGML_CUDA_FATTN_VEC_VTAB

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
    return (turbo3_centroid(low2 | (hi1 << 2)) - u) * norm;
}

static __device__ __forceinline__ float turbo4_dequant_element_u(
        const block_turbo4_0 * __restrict__ blk, int j, float norm, float u) {
    const uint8_t idx = (blk->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return (turbo4_centroid(idx) - u) * norm;
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
    return turbo2_centroid(idx) * norm;
}

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    const uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    return turbo3_centroid(low2 | (hi1 << 2)) * norm;
}

static __device__ __forceinline__ float turbo4_dequant_element(
        const block_turbo4_0 * __restrict__ x, int j, float norm) {
    const uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return turbo4_centroid(idx) * norm;
}

// 2026-11-08: unused element-wise turbo5 helpers (rotated-domain, shared-mask era).  Kept for
// reference only - the live readers are the per-block inverse-FHT kernels (convert.cu
// dequantize_block_turbo5_cont/rowwise, fattn-common.cuh KQ dot / dequantize_V_turbo5_0).
static __device__ __forceinline__ float turbo5_dequant_element(
        const block_turbo5_0 * __restrict__ x, int j, float norm) {
    const uint8_t lo  = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    const uint8_t hi  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    return turbo5_centroid(lo | (hi << 4)) * norm;
}

// Q_prod dequant: (b-1)-bit codebook + 1-bit QJL residual sign (see ggml-common.h).
// norm and rnorm are fp16 block fields; the QJL term is sign * TURBO_QJL_ALPHA * rnorm.

static __device__ __forceinline__ float turbo3q_dequant_element(
        const block_turbo3q_0 * __restrict__ x, int j, float norm, float qjl_scale) {
    const uint8_t idx  = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t sign = (x->signs[j / 8] >> (j % 8)) & 0x1;
    const float   qjl  = (sign ? 1.0f : -1.0f) * qjl_scale;
    return (turbo2_centroid(idx) + qjl) * norm;
}

static __device__ __forceinline__ float turbo4q_dequant_element(
        const block_turbo4q_0 * __restrict__ x, int j, float norm, float qjl_scale) {
    const uint8_t lo   = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi   = (x->hi1[j / 8] >> (j % 8)) & 0x1;
    const uint8_t sign = (x->signs[j / 8] >> (j % 8)) & 0x1;
    const float   qjl  = (sign ? 1.0f : -1.0f) * qjl_scale;
    return (turbo3_centroid(lo | (hi << 2)) + qjl) * norm;
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
