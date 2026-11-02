#include "common.cuh"
#include "fattn-common.cuh"
#include "mma.cuh"

// Hardware FP4 KQ^T in the vec kernel (GGML_CUDA_FATTN_VEC_KQ_MMA=1, NVFP4 K only):
// computes QK^T with mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// instead of the integer dot. m-side = 16 K-token rows read straight from the block_nvfp4 cache
// bytes (nibble repack at load), n-side = up to 8 query rows requantized to nvfp4 in smem.
// Q uses a residual split: Q = Q_hi + Q_lo, both nvfp4 (own scales), two mma chains accumulating
// into the same f32 tile - this removes ~10x of the E2M1 Q error incl. saturation clipping.
// See NVFP4_FA_RESEARCH.md for the fragment / scale-lane layout and the numerical rationale.
static bool ggml_cuda_fattn_vec_kq_mma_enabled() {
    static const bool val = getenv("GGML_CUDA_FATTN_VEC_KQ_MMA") != nullptr;
    return val;
}

// Fused f16 KQ^T in the vec kernel (GGML_CUDA_FATTN_VEC_KQ_MMA_F16=1, NVFP4 K only):
// K is read as unmodified block_nvfp4 cache bytes and decoded E2M1+UE4M3 -> f16 fragment words
// directly in registers (the m16n8k16 map is lane-aligned, so no smem staging and no f16 KV is
// ever written to global memory), Q is converted f32 -> f16 with no quantization and kept in
// per-lane fragment registers, and a dense m16n8k16 f32-accum mma computes KQ^T (m = 16 K
// tokens, n = up to 8 query rows).  Probe-validated numerics: K stays codec-exact, Q error is
// f16 rounding only (score rms 3e-6 against the f16 reference; see NVFP4_FA_RESEARCH.md SS10).
// Takes precedence over GGML_CUDA_FATTN_VEC_KQ_MMA when both are set.
// COMPILE-TIME default since 2026-11-02: cmake option GGML_CUDA_FATTN_VEC_KQ_MMA_F16 (ON by
// default once the fused decode+mma KQ met the speed gate vs the q4_1 integer-dot etalon,
// +0.2% @8K / +0.1% @32K, fa_comp_20261102_010329, decoders probe-proven bit-exact).
// Runtime env GGML_CUDA_FATTN_VEC_KQ_MMA_F16=0/1 overrides either way (0 = integer dp4a dot
// fallback, A/B and debugging).
#ifndef GGML_CUDA_FATTN_VEC_KQ_MMA_F16_DEFAULT
#define GGML_CUDA_FATTN_VEC_KQ_MMA_F16_DEFAULT 1
#endif
static bool ggml_cuda_fattn_vec_kq_mma_f16_enabled() {
    static const bool val = [] (const char * v) {
        if (v != nullptr) return v[0] != '0' && v[0] != '\0';
        return GGML_CUDA_FATTN_VEC_KQ_MMA_F16_DEFAULT != 0;
    }(getenv("GGML_CUDA_FATTN_VEC_KQ_MMA_F16"));
    return val;
}

// S2 (2026-11-02): the same fused f16-MMA KQ^T machinery for the TURBO K types (TURBO4_0 /
// TURBO5_0).  K stays in the raw turbo cache bytes (mux centroid decode + dither + norm folded
// into the f16 A fragments), Q is the same unquantized f32->f16 B fragment set.
// Unlike nvfp4 the turbo numerics change vs the q8_1 dot path (f16 rounding of K*norm and Q),
// so the compile default was OFF until the PPL gate passed; since 2026-11-03 it is ON
// (SS12: T5==T2 exactly, dPPL(T7-T6) = -0.0052). env GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16=0/1
// overrides it at runtime (0 forces the q8_1 dot path for A/B).
// SPEED GATE (fa_turbo_ab_probe, idle GPU, §12): originally fused beat the q8_1 dot only for the
// dither-free heavy-K decoders (turbo5 -12%); TURBO4_0 K lost 35% (D256) / 71% (D128) because the
// float[4] dither out-params spilled to local memory. After S2c (register-only turbo_dither4_pair)
// that is fixed: fused now BEATS the dot on every turbo4 row (D256 t4t3 1.369 vs 1.412, t4t4 1.416
// vs 1.447; D128 0.489 vs 0.503), so the selector has no turbo4 exclusion - turbo4/turbo5 both
// follow GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16.
// turbo4q_0 K was retired from the FA path on 2026-11-02 (worse PPL class than the turbo4/turbo3
// combos - the branch was buried, see §12).
#ifndef GGML_CUDA_FATTN_TURBO_KQ_MMA_F16_DEFAULT
#define GGML_CUDA_FATTN_TURBO_KQ_MMA_F16_DEFAULT 0
#endif
static bool ggml_cuda_fattn_vec_turbo_kq_mma_f16_enabled() {
    static const bool val = [] (const char * v) {
        if (v != nullptr) return v[0] != '0' && v[0] != '\0';
        return GGML_CUDA_FATTN_TURBO_KQ_MMA_F16_DEFAULT != 0;
    }(getenv("GGML_CUDA_FATTN_VEC_TURBO_KQ_MMA_F16"));
    return val;
}

// Fused turbo KQ: build the four f16x2 A-fragment words of one m16n8k16 tile for a pair of K
// token rows (one 16-coord k-fragment f, lane sub-column tg).  Fragment word order matches the
// mma operand order below: {row0:e0, row1:e0, row0:e1, row1:e1} where e0/e1 cover coord pairs
// (2tg, 2tg+1) and (8+2tg, 9+2tg) of the 16-coord chunk cc = f & 7 inside the 128-coord turbo
// block kb = f >> 3 (QK_TURBO* = 128 = 8 chunks per block).  Value arithmetic mirrors
// turboN_dequant_element*() in turbo-quant.cuh bit-for-bit ((c - u)*norm etc.), so the fragment
// equals __float2half_rn(reference f32 value).  This function is the single source of truth for
// the coord->byte mapping and is verified coordinate-by-coordinate against the element
// reference by tools/fa_turbo_frag_probe.cu (the NVFP4_FA_RESEARCH.md SS10j mapping-bug class).
template <ggml_type type_K>
static __device__ __forceinline__ void ggml_cuda_fattn_turbo_kq_afrag(
        const char * __restrict__ Kb0, const char * __restrict__ Kb1,
        const int f, const int tg, const uint32_t seed0, const uint32_t seed1,
        uint32_t A[4]) {
    static_assert(type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TURBO5_0,
                  "turbo kq_afrag: unsupported type_K");

    const int cc = f & 7;   // 16-coord chunk inside the 128-coord block

    if constexpr (type_K == GGML_TYPE_TURBO4_0) {
        const block_turbo4_0 * b0 = (const block_turbo4_0 *) Kb0;
        const block_turbo4_0 * b1 = (const block_turbo4_0 *) Kb1;

        const float n0 = __half2float(b0->norm);
        const float n1 = __half2float(b1->norm);

        // Nibble packing: byte j/2 holds coords (j, j+1) low/high (j even) -> e0 byte 8cc+tg,
        // e1 byte 8cc+4+tg (coords +8).  Dither word w = j/4: w0 = 4cc + (tg>>1), w1 = w0+2,
        // lane j%4 = 2*(tg&1) (see the dither contract in ggml-common.h).
        const uint8_t q00 = b0->qs[8*cc + tg + 0];
        const uint8_t q01 = b0->qs[8*cc + tg + 4];
        const uint8_t q10 = b1->qs[8*cc + tg + 0];
        const uint8_t q11 = b1->qs[8*cc + tg + 4];

        // Register-only dither (S2c): each lane consumes only lanes (2hi, 2hi+1) of the group, hi = tg&1.
        // The old float[4] out-param form + runtime kl index put the groups in LOCAL MEMORY
        // (STACK:64 B/thread, 256 LDL/STL, +36%/+70% kernel time - SS12 spill forensics).
        const bool dhi = (tg & 1) != 0;
        float2 du0 = make_float2(0.0f, 0.0f), du1 = du0, dv0 = du0, dv1 = du0;
#ifdef GGML_TURBO_DITHER
        du0 = turbo_dither4_pair(seed0, 4*cc + (tg >> 1) + 0, dhi, TURBO_DITHER_STEP_4BIT);
        du1 = turbo_dither4_pair(seed0, 4*cc + (tg >> 1) + 2, dhi, TURBO_DITHER_STEP_4BIT);
        dv0 = turbo_dither4_pair(seed1, 4*cc + (tg >> 1) + 0, dhi, TURBO_DITHER_STEP_4BIT);
        dv1 = turbo_dither4_pair(seed1, 4*cc + (tg >> 1) + 2, dhi, TURBO_DITHER_STEP_4BIT);
#endif

        const half2 h00 = __floats2half2_rn((turbo4_centroid(q00 & 0xF) - du0.x) * n0,
                                            (turbo4_centroid(q00 >>  4) - du0.y) * n0);
        const half2 h01 = __floats2half2_rn((turbo4_centroid(q01 & 0xF) - du1.x) * n0,
                                            (turbo4_centroid(q01 >>  4) - du1.y) * n0);
        const half2 h10 = __floats2half2_rn((turbo4_centroid(q10 & 0xF) - dv0.x) * n1,
                                            (turbo4_centroid(q10 >>  4) - dv0.y) * n1);
        const half2 h11 = __floats2half2_rn((turbo4_centroid(q11 & 0xF) - dv1.x) * n1,
                                            (turbo4_centroid(q11 >>  4) - dv1.y) * n1);

        A[0] = *(const uint32_t *) &h00;
        A[1] = *(const uint32_t *) &h10;
        A[2] = *(const uint32_t *) &h01;
        A[3] = *(const uint32_t *) &h11;
    } else if constexpr (type_K == GGML_TYPE_TURBO5_0) {
        const block_turbo5_0 * b0 = (const block_turbo5_0 *) Kb0;
        const block_turbo5_0 * b1 = (const block_turbo5_0 *) Kb1;

        const float n0 = __half2float(b0->norm);
        const float n1 = __half2float(b1->norm);

        // 5-bit index: lo nibble like turbo4 + hi bit in signs[j/8] bit j%8.  For e0 (j=16cc+2tg,
        // +1) that is signs byte 2cc bits 2tg,2tg+1; e1 uses byte 2cc+1 at the same bit positions.
        // No dither in the turbo5 codec (vec_dot_fattn_vec_KQ_turbo5_0 has no u term).
        const uint8_t q00 = b0->qs[8*cc + tg + 0];
        const uint8_t q01 = b0->qs[8*cc + tg + 4];
        const uint8_t q10 = b1->qs[8*cc + tg + 0];
        const uint8_t q11 = b1->qs[8*cc + tg + 4];

        const uint8_t s00 = b0->signs[2*cc + 0];
        const uint8_t s01 = b0->signs[2*cc + 1];
        const uint8_t s10 = b1->signs[2*cc + 0];
        const uint8_t s11 = b1->signs[2*cc + 1];

        const half2 h00 = __floats2half2_rn(turbo5_centroid((q00 & 0xF) | (((s00 >> (2*tg    )) & 1) << 4)) * n0,
                                            turbo5_centroid((q00 >>  4) | (((s00 >> (2*tg + 1)) & 1) << 4)) * n0);
        const half2 h01 = __floats2half2_rn(turbo5_centroid((q01 & 0xF) | (((s01 >> (2*tg    )) & 1) << 4)) * n0,
                                            turbo5_centroid((q01 >>  4) | (((s01 >> (2*tg + 1)) & 1) << 4)) * n0);
        const half2 h10 = __floats2half2_rn(turbo5_centroid((q10 & 0xF) | (((s10 >> (2*tg    )) & 1) << 4)) * n1,
                                            turbo5_centroid((q10 >>  4) | (((s10 >> (2*tg + 1)) & 1) << 4)) * n1);
        const half2 h11 = __floats2half2_rn(turbo5_centroid((q11 & 0xF) | (((s11 >> (2*tg    )) & 1) << 4)) * n1,
                                            turbo5_centroid((q11 >>  4) | (((s11 >> (2*tg + 1)) & 1) << 4)) * n1);

        A[0] = *(const uint32_t *) &h00;
        A[1] = *(const uint32_t *) &h10;
        A[2] = *(const uint32_t *) &h01;
        A[3] = *(const uint32_t *) &h11;
    }
}

// UE4M3 encode matching ggml_fp32_to_ue4m3 (used to requantize Q rows for the FP4 mma).
static __device__ __forceinline__ uint8_t ggml_cuda_ue4m3_from_fp32(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits;
    memcpy(&bits, &x, 4);
    const int fp32_exp = (int) ((bits >> 23) & 0xFF) - 127;
    const int fp32_man = (int) ((bits >> 20) & 0x7);
    int ue_exp = fp32_exp + 7;
    if (ue_exp <= 0) {
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 1) return 0;
        return (uint8_t) man;
    }
    if (ue_exp >= 15) return 0x7E;
    const int round_bit = (int) ((bits >> 19) & 1);
    int ue_man = fp32_man + round_bit;
    if (ue_man > 7) {
        ue_man = 0;
        ue_exp++;
        if (ue_exp >= 15) return 0x7E;
    }
    return (uint8_t) ((ue_exp << 3) | ue_man);
}

// Nearest E2M1 code for v with hw scale dd (value = E2M1 * dd). The midpoints of the E2M1 grid
// times a 4-bit UE4M3 mantissa are exact in fp32, so these comparisons pick the same code as a
// linear scan over all 16 codes with strict '<' (ties go to the smaller magnitude, -0 stays 0).
static __device__ __forceinline__ uint32_t ggml_cuda_e2m1_code(float v, float dd) {
    const float a = fabsf(v);
    if (!(a > 0.0f) || !(dd > 0.0f) || a < 0.25f * dd) return 0;
    const uint32_t m =
        (a < 0.75f * dd) ? 1u :
        (a < 1.25f * dd) ? 2u :
        (a < 1.75f * dd) ? 3u :
        (a < 2.50f * dd) ? 4u :
        (a < 3.50f * dd) ? 5u :
        (a < 5.00f * dd) ? 6u : 7u;
    return (v < 0.0f) ? (m + 8u) : m;
}

// E2M1 dequant for the residual (Q_lo) pass: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6, sign bit 3
// (identical set to the probe's kvalues_fp4/2 table used by the host reference).
static __device__ __forceinline__ float ggml_cuda_e2m1_value(const uint32_t code) {
    const uint32_t m = code & 7u;
    const float mag = (m < 4u) ? ldexpf((float) m, -1)
                               : (m == 4u ? 2.0f : (m == 5u ? 3.0f : (m == 6u ? 4.0f : 6.0f)));
    return (code & 8u) ? -mag : mag;
}

// One ggml block_nvfp4 8-byte group (coords packed as (c, c+8) per byte) -> two
// consecutive-pair words as the tensor core expects (byte i = coords 2i, 2i+1).
static __device__ __forceinline__ void ggml_cuda_fp4_repack_pair(uint32_t W0, uint32_t W1, uint32_t & N0, uint32_t & N1) {
    const uint32_t M0 = W0 & 0x0F0F0F0Fu, M1 = W1 & 0x0F0F0F0Fu;
    const uint32_t X  = __byte_perm(M0, M1, 0x6420);
    const uint32_t Y  = __byte_perm(M0, M1, 0x7531);
    N0 = X | (Y << 4);
    const uint32_t H0 = (W0 >> 4) & 0x0F0F0F0Fu, H1 = (W1 >> 4) & 0x0F0F0F0Fu;
    const uint32_t Xh = __byte_perm(H0, H1, 0x6420);
    const uint32_t Yh = __byte_perm(H0, H1, 0x7531);
    N1 = Xh | (Yh << 4);
}

// E2M1 nibble pair -> packed half2 bits: see ggml_cuda_e2m1_pair_f16 in fattn-common.cuh
// (moved there when dequantize_V_nvfp4 started using it too; PRMT-based, table-free, bit-exact).
// On CUDA >= 12.8 the HW cvt.rn.f16x2.e2m1x2 path (ggml_cuda_e2m1x2_f16) is used instead where
// the data already sits as packed nibble pairs; bitwise-identical to the PRMT path (probe
// tools/fa_nvfp4_cvt_probe.cu: exact on all 256 pair encodings, including -0 for code 8).

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
// Register-budget knob (the analogue of Triton's BLOCK_M/N/K autotune for this kernel): minBlocks = 1
// lets ptxas spend all 255 registers/thread, which at 128 threads/block already fills the 65536 reg/SM
// file with 2 blocks - so any extra live state can only SPILL, never cost occupancy (SS12 forensics).
// Raising it caps registers: 2 -> 256, 3 -> 170 (3 blocks/SM = 12 warps instead of 8, +50% latency
// hiding). Experiments build the probe with -DGGML_CUDA_FATTN_VEC_MIN_BLOCKS=<n> (n>1 forces the cap
// on every instantiation).
#ifndef GGML_CUDA_FATTN_VEC_MIN_BLOCKS
#define GGML_CUDA_FATTN_VEC_MIN_BLOCKS 1
#endif
// Production policy (S2d gate, fa_turbo_ab_probe 2026-11-03): REG 255->168 with STACK still 0 gave
// -18..-25% on BOTH dot and fused D256 rows (T1 1.412->1.100, T6 1.369->1.065) but +1..+1.4% (worse)
// on q5_1-K and D128-dot rows, so 3 blocks/SM is enabled only where it measured faster: turbo K at
// D=256. Nothing else changes instantiation (nvfp4/q5_1/f16/D128 keep today's codegen).
// UPDATE same day: the policy additionally requires ncols==1 (the decode shape the probe actually
// measured). The ncols=2 (nq>1) turbo D256 rows spill 80-96 B/thread in cuobjdump AT BOTH register
// budgets (255 and 168), i.e. that spill is intrinsic to the shape, not caused by the cap - but the
// shape was never timed, so it keeps today's codegen (min_blocks=1) until someone measures it.
template <int D, int ncols, ggml_type type_K>
constexpr int ggml_cuda_fattn_vec_min_blocks() {
    if (GGML_CUDA_FATTN_VEC_MIN_BLOCKS > 1) {
        return GGML_CUDA_FATTN_VEC_MIN_BLOCKS;
    }
    return (D == 256 && ncols == 1 && (type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TURBO5_0)) ? 3 : 1;
}

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool KQ_mma = false, bool KQ_f16dq = false> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), ggml_cuda_fattn_vec_min_blocks<D, ncols, type_K>())
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    // Subtractive dither seed base (see the dither contract in ggml-common.h): k_set_rows_turbo3/4
    // hashed the LINEAR quantization-block index of the block inside the cache tensor. The vec
    // kernels walk K/V as base pointer + strides, so turn the row's byte offset into that block
    // index here; the division is exact because the row strides are whole blocks.
    // Non-turbo types ignore the value (their block size only makes the arithmetic well-defined).
    constexpr int64_t k_blk_bytes = ggml_cuda_fattn_vec_block_bytes<type_K>();
    constexpr int64_t v_blk_bytes = ggml_cuda_fattn_vec_block_bytes<type_V>();
    const int64_t k_blk_per_row = (int64_t) nb11 / k_blk_bytes;
    const int64_t v_blk_per_row = (int64_t) nb21 / v_blk_bytes;

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    // FP4 mma staging tiles in fragment-native word order, sized 1 when unused.
    // F4_STR = code words per row (+1 to break bank alignment), F4_raw holds the fp32 scores
    // before softmax; KQ itself may be half on some arches so it cannot hold the raw scores.
    constexpr int F4_STR = D/8 + 1;
    __shared__ int  F4_Qnc[KQ_mma ?  8*F4_STR      : 1];
    __shared__ int  F4_Qsc[KQ_mma ?  8*(D/64)      : 1];
    __shared__ int  F4_Qnc2[KQ_mma ?  8*F4_STR     : 1];  // residual-split (Q_lo) operands
    __shared__ int  F4_Qsc2[KQ_mma ?  8*(D/64)     : 1];
    __shared__ int  F4_Knc[KQ_mma ? 32*F4_STR      : 1];
    __shared__ int  F4_Ksc[KQ_mma ? 32*(D/64)      : 1];
    __shared__ float F4_raw[(KQ_mma || KQ_f16dq) ? ncols*nthreads : 1];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    // KQ_f16dq needs no prep: Q fragment words are built per-lane inside the KQ block below
    // (the m16n8k16 map is lane-aligned, see NVFP4_FA_RESEARCH.md SS10e).
    if constexpr (KQ_mma) {
        // Requantize up to 8 Q rows to nvfp4 in smem, fragment-native order, one sub-block per lane.
        constexpr int F4_NSUB  = D / QK_NVFP4_SUB;
        constexpr int F4_RPP   = WARP_SIZE / F4_NSUB; // rows per warp per pass
        const int l  = threadIdx.x % F4_NSUB;
        const int rr = threadIdx.x / F4_NSUB;
        for (int j0 = 0; j0 < 8; j0 += nwarps*F4_RPP) {
            const int j = j0 + threadIdx.y*F4_RPP + rr;

            if (j >= 8) {
                break;
            }

            const bool   active = (j < ncols) && (ncols == 1 || ic0 + j < int(ne01.z));
            const float * Q_f   = (const float *) (Q + j*nb01);

            float v[QK_NVFP4_SUB];
            float amax = 0.0f;
#pragma unroll
            for (int c = 0; c < QK_NVFP4_SUB; ++c) {
                v[c] = active ? Q_f[l*QK_NVFP4_SUB + c] : 0.0f;
                amax = fmaxf(amax, fabsf(v[c]));
            }
            const uint8_t ue = ggml_cuda_ue4m3_from_fp32(amax / 6.0f);
            const float   dd = 2.0f * ggml_cuda_ue4m3_to_fp32(ue); // matches the hw scale byte, exact doubling

            uint32_t cd[QK_NVFP4_SUB];
            uint32_t N0 = 0, N1 = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                cd[2*b + 0] = ggml_cuda_e2m1_code(v[2*b + 0], dd);
                cd[2*b + 1] = ggml_cuda_e2m1_code(v[2*b + 1], dd);
                cd[2*b + 8] = ggml_cuda_e2m1_code(v[2*b + 8], dd);
                cd[2*b + 9] = ggml_cuda_e2m1_code(v[2*b + 9], dd);
                N0 |= (uint32_t) (cd[2*b + 0] | (cd[2*b + 1] << 4)) << (8*b);
                N1 |= (uint32_t) (cd[2*b + 8] | (cd[2*b + 9] << 4)) << (8*b);
            }
            F4_Qnc[j*F4_STR + 2*l]     = (int) N0;
            F4_Qnc[j*F4_STR + 2*l + 1] = (int) N1;

            uint32_t u = ue;
            u |= __shfl_sync(0xFFFFFFFFu, ue, threadIdx.x + 1, WARP_SIZE) << 8;
            u |= __shfl_sync(0xFFFFFFFFu, ue, threadIdx.x + 2, WARP_SIZE) << 16;
            u |= __shfl_sync(0xFFFFFFFFu, ue, threadIdx.x + 3, WARP_SIZE) << 24;
            if ((l & 3) == 0) {
                F4_Qsc[j*(D/64) + (l >> 2)] = (int) u;
            }

            // Residual split Q_lo = nvfp4(Q - dequant(Q_hi)) with its own block scales, same
            // closed-form codec as the hi pass. Removes ~10x of the E2M1 Q error (incl. the
            // saturation-clipped coords); quantized exactly like the host reference in the
            // probe's --mode residual study (NVFP4_FA_RESEARCH.md §8).
            float rv[QK_NVFP4_SUB];
            float amax_r = 0.0f;
#pragma unroll
            for (int c = 0; c < QK_NVFP4_SUB; ++c) {
                rv[c] = v[c] - dd * ggml_cuda_e2m1_value(cd[c]);
                amax_r = fmaxf(amax_r, fabsf(rv[c]));
            }
            const uint8_t ue2 = ggml_cuda_ue4m3_from_fp32(amax_r / 6.0f);
            const float   dd2 = 2.0f * ggml_cuda_ue4m3_to_fp32(ue2);

            uint32_t M0 = 0, M1 = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                M0 |= (uint32_t) (ggml_cuda_e2m1_code(rv[2*b + 0], dd2) | (ggml_cuda_e2m1_code(rv[2*b + 1], dd2) << 4)) << (8*b);
                M1 |= (uint32_t) (ggml_cuda_e2m1_code(rv[2*b + 8], dd2) | (ggml_cuda_e2m1_code(rv[2*b + 9], dd2) << 4)) << (8*b);
            }
            F4_Qnc2[j*F4_STR + 2*l]     = (int) M0;
            F4_Qnc2[j*F4_STR + 2*l + 1] = (int) M1;

            uint32_t u2 = ue2;
            u2 |= __shfl_sync(0xFFFFFFFFu, ue2, threadIdx.x + 1, WARP_SIZE) << 8;
            u2 |= __shfl_sync(0xFFFFFFFFu, ue2, threadIdx.x + 2, WARP_SIZE) << 16;
            u2 |= __shfl_sync(0xFFFFFFFFu, ue2, threadIdx.x + 3, WARP_SIZE) << 24;
            if ((l & 3) == 0) {
                F4_Qsc2[j*(D/64) + (l >> 2)] = (int) u2;
            }
        }
        __syncthreads();
    } else if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        // First block of this K tile inside the cache tensor (dither seed base, see ggml-common.h):
        const int64_t k_tile_ibase = (K - K_ptr) / k_blk_bytes;

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

        if constexpr (!KQ_mma && !KQ_f16dq) {
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
                const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j], k_tile_ibase + (int64_t) i_KQ*k_blk_per_row);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);

                    if (use_logit_softcap) {
                        sum = logit_softcap*tanhf(sum);
                    }

                    if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                        sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                    }

                    KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                    if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                        KQ_reg[j] = sum;
                    }
                }
            }
        } else {
            if constexpr (KQ_f16dq) {
                // Fused decode + dense f16 MMA, fully register-resident (v2; NVFP4_FA_RESEARCH.md
                // SS10e, turbo port SS12).  No smem tiles / warp syncs: the m16n8k16 fragment map is
                // lane-aligned - lane (g, tg) owns token rows g and g+8 at coords {2tg,2tg+1,
                // 8+2tg, 9+2tg} of each 16-coord k-fragment, and query column g in the B operand.
                // The A fragments come from the raw gmem cache bytes (nvfp4: byte-permuted nibbles x
                // UE4M3 scale; turbo: ggml_cuda_fattn_turbo_kq_afrag mux decode + dither + norm),
                // and the B fragments (Q) are constant over the whole CTA run.
                const int tok0 = WARP_SIZE*threadIdx.y;
                const int g  = threadIdx.x >> 2;
                const int tg = threadIdx.x &  3;

                const bool    q_active = (g < ncols) && (ncols == 1 || ic0 + g < int(ne01.z));
                const float * Q_g      = (const float *) (Q + g*nb01);

                uint32_t Bfr[D/16][2];
#pragma unroll
                for (int f = 0; f < D/16; ++f) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int c = 16*f + 8*e + 2*tg;   // half2 word holds coords c, c+1
                        const half2 h = q_active ? __floats2half2_rn(Q_g[c + 0], Q_g[c + 1])
                                                 : __floats2half2_rn(0.0f, 0.0f);
                        Bfr[f][e] = (uint32_t) *reinterpret_cast<const uint32_t *>(&h);
                    }
                }

                const size_t   koff = (size_t) (tok0 + g)*nb11;

                if constexpr (type_K == GGML_TYPE_NVFP4) {
                const uint32_t sq   = (2*tg) | ((2*tg + 1) << 4); // byte_perm picks subblock bytes 2tg, 2tg+1

#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float Dt[4] = { 0.f, 0.f, 0.f, 0.f };
#pragma unroll
                    for (int f = 0; f < D/16; ++f) {
                        const block_nvfp4 * K_r0 = (const block_nvfp4 *) (K + koff + (size_t) (16*mt + 0)*nb11) + (f >> 2);
                        const block_nvfp4 * K_r1 = (const block_nvfp4 *) (K + koff + (size_t) (16*mt + 8)*nb11) + (f >> 2);
                        const int sb = 8*(f & 3);

                        // 8-byte subblock words -> the two bytes holding coords (2tg, 2tg+1) / (8+2tg, 9+2tg):
                        // byte b of the subblock stores coords (b, b+8) as (low, high) nibbles.
                        const uint32_t s0 = __byte_perm(*(const uint32_t *) &K_r0->qs[sb + 0],
                                                       *(const uint32_t *) &K_r0->qs[sb + 4], sq);
                        const uint32_t s1 = __byte_perm(*(const uint32_t *) &K_r1->qs[sb + 0],
                                                       *(const uint32_t *) &K_r1->qs[sb + 4], sq);

                        const half2 sh0 = ggml_cuda_ue4m3_dword_h2(*(const uint32_t *) &K_r0->d[0], f & 3);
                        const half2 sh1 = ggml_cuda_ue4m3_dword_h2(*(const uint32_t *) &K_r1->d[0], f & 3);

                        // Fragment pairs are nibbles at stride 2 (byte b of s0 holds coords (b, b+8)):
                        // pack each MMA pair (elem0 low, elem1 high) and decode via HW cvt -
                        // bitwise-identical to ggml_cuda_e2m1_pair_f16 (probe fa_nvfp4_cvt_probe),
                        // one instruction per pair instead of ~5.
                        const uint32_t q0 = ( s0        & 0x0Fu) | (( s0 >> 4) & 0xF0u); // coords (c,   c+1)
                        const uint32_t q2 = ((s0 >> 4)  & 0x0Fu) | (( s0 >> 8) & 0xF0u); // coords (c+8, c+9)
                        const uint32_t q1 = ( s1        & 0x0Fu) | (( s1 >> 4) & 0xF0u);
                        const uint32_t q3 = ((s1 >> 4)  & 0x0Fu) | (( s1 >> 8) & 0xF0u);

                        const uint32_t c0 = ggml_cuda_e2m1x2_f16(q0);
                        const uint32_t c2 = ggml_cuda_e2m1x2_f16(q2);
                        const uint32_t c1 = ggml_cuda_e2m1x2_f16(q1);
                        const uint32_t c3 = ggml_cuda_e2m1x2_f16(q3);

                        half2 hv0 = __hmul2(*reinterpret_cast<const half2 *>(&c0), sh0);
                        half2 hv1 = __hmul2(*reinterpret_cast<const half2 *>(&c1), sh1);
                        half2 hv2 = __hmul2(*reinterpret_cast<const half2 *>(&c2), sh0);
                        half2 hv3 = __hmul2(*reinterpret_cast<const half2 *>(&c3), sh1);

                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                     : "+f"(Dt[0]), "+f"(Dt[1]), "+f"(Dt[2]), "+f"(Dt[3])
                                     : "r"(*(uint32_t *) &hv0), "r"(*(uint32_t *) &hv1),
                                       "r"(*(uint32_t *) &hv2), "r"(*(uint32_t *) &hv3),
                                       "r"(Bfr[f][0]), "r"(Bfr[f][1]));
                    }
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int j = 2*tg + e;
                        if (j < ncols) {
                            F4_raw[j*nthreads + tok0 + 16*mt + g + 0] = Dt[0 + e]*scale;
                            F4_raw[j*nthreads + tok0 + 16*mt + g + 8] = Dt[2 + e]*scale;
                        }
                    }
                }
                } else {
                    // Fused TURBO decode + dense f16 MMA (S2; NVFP4_FA_RESEARCH.md SS12).  Same lane
                    // map as the nvfp4 branch above; the K side is ggml_cuda_fattn_turbo_kq_afrag
                    // (mux centroid decode + dither + norm folded into the f16 fragment words), so
                    // the only per-lane work here is the block pointer walk and one mma per fragment.
                    static_assert(type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TURBO5_0,
                                  "KQ_f16dq: unexpected fused type_K");

#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        float Dt[4] = { 0.f, 0.f, 0.f, 0.f };
#pragma unroll
                        for (int f = 0; f < D/16; ++f) {
                            const char * Kb0 = K + koff + (size_t) (16*mt + 0)*nb11 + (size_t) (f >> 3) * (size_t) k_blk_bytes;
                            const char * Kb1 = K + koff + (size_t) (16*mt + 8)*nb11 + (size_t) (f >> 3) * (size_t) k_blk_bytes;

                            // Dither seeds must match the vec_dot / set_rows contract: the LINEAR
                            // block index of this row inside the cache tensor + the block-within-row
                            // index (f>>3, QK_TURBO = 128 coords = 8 k-fragments).
                            const uint32_t seed0 = (uint32_t) (k_tile_ibase + (int64_t) (16*mt + g + 0)*k_blk_per_row + (f >> 3));
                            const uint32_t seed1 = (uint32_t) (k_tile_ibase + (int64_t) (16*mt + g + 8)*k_blk_per_row + (f >> 3));

                            uint32_t A[4];
                            ggml_cuda_fattn_turbo_kq_afrag<type_K>(Kb0, Kb1, f, tg, seed0, seed1, A);

                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                         : "+f"(Dt[0]), "+f"(Dt[1]), "+f"(Dt[2]), "+f"(Dt[3])
                                         : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                                           "r"(Bfr[f][0]), "r"(Bfr[f][1]));
                        }
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int j = 2*tg + e;
                            if (j < ncols) {
                                F4_raw[j*nthreads + tok0 + 16*mt + g + 0] = Dt[0 + e]*scale;
                                F4_raw[j*nthreads + tok0 + 16*mt + g + 8] = Dt[2 + e]*scale;
                            }
                        }
                    }
                }
            } else {
            // Each warp stages its 32 K rows into fragment-native words + scale dwords, then runs
            // 2 x m16n8k64 block-scaled mma per k-fragment (m = tokens, n = queries).
            const int tok0 = WARP_SIZE*threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < 32*(D/16); i0 += WARP_SIZE) {
                const int i  = i0 + threadIdx.x;
                const int r  = i / (D/16);
                const int pr = i % (D/16);

                const block_nvfp4 * K_b = (const block_nvfp4 *) (K + (size_t) (tok0 + r)*nb11);
                const int * words       = (const int *) &K_b[pr >> 2];

                uint32_t N0, N1;
                ggml_cuda_fp4_repack_pair((uint32_t) words[1 + 2*(pr & 3)], (uint32_t) words[2 + 2*(pr & 3)], N0, N1);
                F4_Knc[r*F4_STR + 2*pr]     = (int) N0;
                F4_Knc[r*F4_STR + 2*pr + 1] = (int) N1;
            }
            for (int i0 = 0; i0 < 32*(D/64); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                const int r = i / (D/64);
                const int f = i % (D/64);
                const block_nvfp4 * K_b = (const block_nvfp4 *) (K + (size_t) (tok0 + r)*nb11);
                F4_Ksc[r*(D/64) + f] = *(const int *) &K_b[f].d[0];
            }
            __syncwarp();

            const int g = threadIdx.x >> 2;
            const int t = threadIdx.x &  3;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                using namespace ggml_cuda_mma;
                tile<16, 8, float> Dt;
#pragma unroll
                for (int f = 0; f < D/64; ++f) {
                    tile<16, 8, int> At;
                    tile<8, 8, int>  Bt;
                    At.x[0] = F4_Knc[(16*mt + g + 0)*F4_STR + f*8 + t + 0];
                    At.x[1] = F4_Knc[(16*mt + g + 8)*F4_STR + f*8 + t + 0];
                    At.x[2] = F4_Knc[(16*mt + g + 0)*F4_STR + f*8 + t + 4];
                    At.x[3] = F4_Knc[(16*mt + g + 8)*F4_STR + f*8 + t + 4];
                    Bt.x[0] = F4_Qnc[g*F4_STR + f*8 + t + 0];
                    Bt.x[1] = F4_Qnc[g*F4_STR + f*8 + t + 4];
                    const uint32_t sa = (uint32_t) F4_Ksc[(16*mt + g + 8*(threadIdx.x & 1))*(D/64) + f];
                    const uint32_t sb = (uint32_t) F4_Qsc[g*(D/64) + f];
                    mma_block_scaled_fp4<GGML_TYPE_NVFP4>(Dt, At, Bt, sa, sb);
                    // Residual chain Q_lo: same K fragments & scales, accumulates into the same Dt.
                    tile<8, 8, int> Bt2;
                    Bt2.x[0] = F4_Qnc2[g*F4_STR + f*8 + t + 0];
                    Bt2.x[1] = F4_Qnc2[g*F4_STR + f*8 + t + 4];
                    const uint32_t sb2 = (uint32_t) F4_Qsc2[g*(D/64) + f];
                    mma_block_scaled_fp4<GGML_TYPE_NVFP4>(Dt, At, Bt2, sa, sb2);
                }
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int j = 2*t + e;
                    if (j < ncols) {
                        F4_raw[j*nthreads + tok0 + 16*mt + g + 0] = Dt.x[0 + e]*scale;
                        F4_raw[j*nthreads + tok0 + 16*mt + g + 8] = Dt.x[2 + e]*scale;
                    }
                }
            }
            }

            __syncwarp();

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = F4_raw[j*nthreads + tid];

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + tid]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], warp_reduce_max(sum + FATTN_KQ_MAX_OFFSET));
                KQ_reg[j] = sum;
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

        ggml_cuda_syncwarp();

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            // First block of this V row inside the cache tensor (dither seed base, see ggml-common.h):
            const int64_t v_row_ibase = (V - V_ptr) / v_blk_bytes + (int64_t) k*v_blk_per_row;

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, v_row_ibase);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, v_row_ibase);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread, v_row_ibase);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool kq_mma_build, bool kq_f16_build>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    if constexpr (kq_f16_build) {
        // See the S2 speed gate comment at the turbo env helpers above: after S2c fused wins both
        // turbo4 and turbo5 rows, so the switch applies to all turbo K types (turbo4q K retired
        // from this kernel - §12).  nvfp4 keeps its own (default-ON) switch.
        const bool kq_f16_sel = (type_K == GGML_TYPE_NVFP4)
            ? ggml_cuda_fattn_vec_kq_mma_f16_enabled()
            : ggml_cuda_fattn_vec_turbo_kq_mma_f16_enabled();
        if (turing_mma_available(cc) && kq_f16_sel) {
            fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, false, true>;
            launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false);
            return;
        }
    }
    if constexpr (kq_mma_build) {
        if (blackwell_mma_available(cc) && ggml_cuda_fattn_vec_kq_mma_enabled()) {
            fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, true>;
            launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false);
            return;
        }
    }
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // the mma KQ variant needs whole 64-coord k-fragments and 16-token m-tiles
    constexpr bool kq_mma_build  = (type_K == GGML_TYPE_NVFP4) && (D >= 128);
    // fused f16 KQ needs whole 16-coord k-fragments; probe-validated at D=128/256.
    // Turbo K (S2): QK_TURBO* = 128 coords per block = 8 fragments, so the same D gate applies
    // (no D=64 turbo-K instances exist).  Runtime default: nvfp4 ON, turbo OFF (SS12 PPL gate).
    constexpr bool kq_f16_build  = (D == 128 || D == 256) &&
        (type_K == GGML_TYPE_NVFP4 || type_K == GGML_TYPE_TURBO4_0 || type_K == GGML_TYPE_TURBO5_0);

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, kq_mma_build, kq_f16_build>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, kq_mma_build, kq_f16_build>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, kq_mma_build, kq_f16_build>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap, kq_mma_build, kq_f16_build>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

// TurboQuant KV cache: only the K/V pairs with a compiled template instance are dispatched
// (gated by the GGML_CUDA_FA_Q5_1_TURBO4_0 / GGML_CUDA_FA_Q5_1_TURBO3_0 constexpr flags in fattn.cu).
// The extern declarations suppress implicit instantiation of the heavy kernels in fattn.cu itself.
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0);

// turbo4 K has a native vec_dot (vec_dot_fattn_vec_KQ_turbo4_0). No 64 case: QK_TURBO4 = 128 doesn't fit a 64-element row.
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
// turbo4 K + turbo4 V (competitor gate row): native vec instance (see template-instances note re: the MMA speed gate)
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);

// Q_prod (MSE + 1-bit QJL residual sign) combos.
// turbo4q_0 combos were retired from the FA vec kernel on 2026-11-02 (worse PPL class than the
// turbo4/turbo3 combos, branch buried - §12); such combos now fall back to the f16-scratch path.
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3Q_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3Q_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_TURBO3Q_0);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3Q_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3Q_0);

// NVFP4-like KV cache (Stage A): single-level NVFP4 (per-16 UE4M3 + E2M1). Unrotated, no dither;
// the block size (64) fits a 64-element K row, so all three head sizes have instances.
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q5_1, GGML_TYPE_NVFP4);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_NVFP4);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q5_1, GGML_TYPE_NVFP4);

extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_NVFP4);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_NVFP4);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_NVFP4);
