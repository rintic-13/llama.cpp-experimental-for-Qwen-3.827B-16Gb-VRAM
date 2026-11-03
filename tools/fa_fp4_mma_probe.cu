/*
 * fa_fp4_mma_probe.cu — SM120 hardware FP4 MMA feasibility probe for NVFP4 KV in Flash Attention.
 *
 * Standalone (no llama.cpp link): nvcc -arch=sm_120a. Three modes:
 *
 *   --mode verify  : does a block-scaled FP4 MMA fed DIRECTLY from block_nvfp4 bytes
 *                    (d[4] UE4M3 + qs[32] packed E2M1, c / c+8 byte interleaving) produce the
 *                    mathematically correct QK^T tile? Tests 3 levels:
 *                      T1 uniform scales  -> code/k-coordinate mapping
 *                      T2 per-row scales  -> row<->scale-register mapping
 *                      T3 per-subblock scales -> scale_vec::4X byte-order mapping
 *   --mode bench   : QK^T tile throughput: PathVEC (current fattn-vec integer-dot semantics)
 *                    vs PathMMA (hardware kind::mxf4nvf4 block_scale) on the same tile shapes,
 *                    plus a pure tensor-core issue-rate microbench (f16 / fp8 / fp4).
 *   --mode numerics: score-matrix error stats for Q codecs: q8_1 (current), nvfp4 (E2M1+UE4M3),
 *                    e4m3+ue8m0 (mixed-kind candidate), against an exact fp32 reference.
 *   --mode residual: Q-side residual split: Q_hi=nvfp4(Q), Q_lo=nvfp4(Q-Q_hi) -> two mma chains
 *                    against the SAME K bytes; saturation share, Q-err and score-err for
 *                    pass1 vs pass1+2.  --qfile <f32 rows of D> uses dumped real activations.
 *   --mode kernelcheck: bit-exact comparison of the fattn-vec.cuh hi+lo Q-prep and K-stage +
 *                    two-chain mma loop REPLICAS against an exact host reference (localizes the
 *                    PPL-gate failure to prep / fragment wiring / kernel context).
 *
 * Everything device-side is deliberately a faithful copy of the instruction sequence and
 * fragment addressing already used by MMQ (ggml/src/ggml-cuda/mmq-vec-dot.cuh,
 * ggml_cuda_mmq_vec_dot_fp4_fp4_mma) so that what is verified here is what FA would use.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// codec constants (mirrors ggml/src/ggml-common.h)
// ---------------------------------------------------------------------------
#define QK_NVFP4     64          // elements per block
#define QK_NVFP4_SUB 16          // elements per UE4M3 scale
#define NSUB         (QK_NVFP4 / QK_NVFP4_SUB)   // 4
#define BLK_BYTES    (4 + QK_NVFP4 / 2)          // block_nvfp4 = 36 B

// kvalues_fp4 == 2 * E2M1 value, ggml_ue4m3_to_fp32 == ue4m3 * 0.5, product == E2M1 * UE4M3.
static const int kvalues_fp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

static inline float e2m1_value(uint8_t code) {
    return 0.5f * (float) kvalues_fp4[code & 0xF];
}

// Closed-form E2M1 quantizer, bit-for-bit the same comparisons as ggml_cuda_e2m1_code /
// e2m1_code_dev (ties resolve toward the lower bin - that's what the kernel does).
static inline uint32_t e2m1_code_host(float v, float dd) {
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

// UE4M3 semantics identical to ggml_ue4m3_to_fp32 in ggml/src/ggml-impl.h, but WITHOUT the
// *0.5 factor the fork's tables use (the hardware MMA reads the raw UE4M3 byte).
static inline float ue4m3_hw(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;
    if (exp == 0) return (float) mant * (1.0f / 512.0f);
    return (1.0f + (float) mant / 8.0f) * exp2f((float) (exp - 7));
}

// mirrors ggml_fp32_to_ue4m3 (round-to-nearest via the fp32 bits, saturating)
static inline uint8_t ue4m3_from_f32(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits;
    memcpy(&bits, &x, 4);
    const int fp32_exp  = (int) ((bits >> 23) & 0xFF) - 127;
    const int fp32_man  = (int) ((bits >> 20) & 0x7);
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

// exact (fp64) UE4M3 decode - used to flag scale encodings that are borderline fp32 ties
static inline double ue4m3_exact(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0;
    const int exp = (x >> 3) & 0xF, mant = x & 0x7;
    if (exp == 0) return (double) mant / 512.0;
    return (1.0 + (double) mant / 8.0) * pow(2.0, exp - 7);
}

// quantize a row of QK_NVFP4*nb blocks exactly like quantize_row_nvfp4_ref (max/6 policy)
static void nvfp4_quant_row(const float * x, int nb, uint8_t * d, uint8_t * qs) {
    for (int ib = 0; ib < nb; ++ib) {
        for (int s = 0; s < NSUB; ++s) {
            const float * xb = x + ib * QK_NVFP4 + s * QK_NVFP4_SUB;
            float amax = 0.0f;
            for (int j = 0; j < QK_NVFP4_SUB; ++j) amax = fmaxf(amax, fabsf(xb[j]));
            const uint8_t ue = ue4m3_from_f32(amax / 6.0f);
            d[ib * NSUB + s] = ue;
            const float dd = ue4m3_hw(ue);
            for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                auto nearest = [&](float v) -> uint8_t {
                    float best = 1e30f; uint8_t bc = 0;
                    for (int c = 0; c < 16; ++c) {
                        const float e = fabsf(v - e2m1_value(c) * dd);
                        if (e < best) { best = e; bc = (uint8_t) c; }
                    }
                    return bc;
                };
                const uint8_t x0 = nearest(xb[j]);
                const uint8_t x1 = nearest(xb[QK_NVFP4_SUB / 2 + j]);
                qs[ib * (QK_NVFP4 / 2) + s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

// dequantize one row (nb blocks) back to float using the stored codes/scales
static void nvfp4_dequant_row(const uint8_t * d, const uint8_t * qs, int nb, float * out) {
    for (int ib = 0; ib < nb; ++ib) {
        for (int s = 0; s < NSUB; ++s) {
            const float dd = ue4m3_hw(d[ib * NSUB + s]);
            for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                const uint8_t byte = qs[ib * (QK_NVFP4 / 2) + s * (QK_NVFP4_SUB / 2) + j];
                out[ib * QK_NVFP4 + s * QK_NVFP4_SUB + j]               = e2m1_value(byte & 0xF) * dd;
                out[ib * QK_NVFP4 + s * QK_NVFP4_SUB + QK_NVFP4_SUB / 2 + j] = e2m1_value(byte >> 4) * dd;
            }
        }
    }
}

// ggml (c,c+8) nibble packing -> fragment-native consecutive-pair words (NVIDIA nvfp4 tensor
// byte order: byte i = coords 2i, 2i+1).  Row r of nb blocks, dst row-major with `stride` words.
static void ggml_to_native_words(const uint8_t * qs, int r, int nb, int * dst, int stride) {
    for (int f = 0; f < nb; ++f) {
        uint32_t w[8] = { 0 };
        const uint8_t * q = qs + (size_t) r * nb * (QK_NVFP4 / 2) + f * (QK_NVFP4 / 2);
        for (int c = 0; c < QK_NVFP4; ++c) {
            const int byte = (c / 16) * 8 + (c % 8);
            const int hi = (c % 16) / 8;
            const uint32_t code = (q[byte] >> (hi * 4)) & 0xF;
            w[c / 8] |= code << (4 * (c % 8));
        }
        for (int t = 0; t < 8; ++t) dst[(size_t) r * stride + f * 8 + t] = (int) w[t];
    }
}

// ---------------------------------------------------------------------------
// device helpers: exactly the PTX used by MMQ
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_nvfp4(float * D, const uint32_t * A, const uint32_t * B,
                                          uint32_t a_scale, uint32_t b_scale) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(a_scale), "r"(b_scale));
}

// ldmatrix for the A operand (16 rows x 64 e2m1 = 8 words/row), MMQ addressing
__device__ __forceinline__ void ldmatrix_a(uint32_t * A, const int * xs0, int stride_words) {
    const int * xs = xs0 + (threadIdx.x % 16) * stride_words + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(xs));
}

// GGML (c,c+8) byte pair -> fragment-native consecutive-pair words, in-register.
// One GGML 8-byte group (2 words, 16 coords) -> one native word pair (same indices).
// N0 = coords (2i,2i+1) taken from LOW nibbles of g0..g7; N1 from HIGH nibbles.
__device__ __forceinline__ void ggml_pair_to_native(uint32_t W0, uint32_t W1, uint32_t & N0, uint32_t & N1) {
    const uint32_t M0 = W0 & 0x0F0F0F0Fu, M1 = W1 & 0x0F0F0F0Fu;
    const uint32_t X  = __byte_perm(M0, M1, 0x6420);   // g0 g2 g4 g6 (masked to low nibble)
    const uint32_t Y  = __byte_perm(M0, M1, 0x7531);   // g1 g3 g5 g7
    N0 = X | (Y << 4);
    const uint32_t H0 = (W0 >> 4) & 0x0F0F0F0Fu, H1 = (W1 >> 4) & 0x0F0F0F0Fu;
    const uint32_t Xh = __byte_perm(H0, H1, 0x6420);
    const uint32_t Yh = __byte_perm(H0, H1, 0x7531);
    N1 = Xh | (Yh << 4);
}

__device__ __forceinline__ float ue4m3_dev(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;
    if (exp == 0) return (float) mant * (1.0f / 512.0f);
    return (1.0f + (float) mant / 8.0f) * exp2f((float) (exp - 7));
}

// ===========================================================================
// verify kernel: one warp, A = 16 rows x 64 coords, B = 8 rows x 64 coords
// FRAGMENT-NATIVE word order: word (f*8+t) of a row = coords 64f + 8t..8t+7
// (host repacks ggml (c,c+8) bytes into consecutive-pair native words)
// ===========================================================================
#define VSTRIDE 12      // words per row in shared (8 used + pad)

__global__ void k_verify(const int * __restrict__ qsA, const int * __restrict__ scA,
                         const int * __restrict__ qsB, const int * __restrict__ scB,
                         float * __restrict__ out, int nfrag) {
    __shared__ int shA[16 * VSTRIDE];
    __shared__ int shB[8 * VSTRIDE];
    __shared__ int shSA[16 * VSTRIDE];
    __shared__ int shSB[8 * VSTRIDE];

    for (int i = threadIdx.x; i < 16 * VSTRIDE; i += blockDim.x) shA[i] = qsA[i];
    for (int i = threadIdx.x; i < 8 * VSTRIDE; i += blockDim.x) shB[i] = qsB[i];
    for (int i = threadIdx.x; i < 16 * VSTRIDE; i += blockDim.x) shSA[i] = scA[i];
    for (int i = threadIdx.x; i < 8 * VSTRIDE; i += blockDim.x) shSB[i] = scB[i];
    __syncthreads();

    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    float D[4] = { 0.f, 0.f, 0.f, 0.f };

    for (int f = 0; f < nfrag; ++f) {
        uint32_t A[4], B[2];
        A[0] = (uint32_t) shA[(g + 0) * VSTRIDE + f * 8 + t + 0];
        A[1] = (uint32_t) shA[(g + 8) * VSTRIDE + f * 8 + t + 0];
        A[2] = (uint32_t) shA[(g + 0) * VSTRIDE + f * 8 + t + 4];
        A[3] = (uint32_t) shA[(g + 8) * VSTRIDE + f * 8 + t + 4];
        B[0] = (uint32_t) shB[g * VSTRIDE + f * 8 + t + 0];
        B[1] = (uint32_t) shB[g * VSTRIDE + f * 8 + t + 4];

        const uint32_t a_scale = (uint32_t) shSA[(g + 8 * (t & 1)) * VSTRIDE + f];  // lane 4g+0 -> row g, 4g+1 -> g+8
        const uint32_t b_scale = (uint32_t) shSB[g * VSTRIDE + f];                  // lane 4g   -> col g

        mma_nvfp4(D, A, B, a_scale, b_scale);
    }

    // C fragment (m16n8 float): row = g (+8 for c2,c3), col = 2t + (i&1)
    const int cr = lane / 4;
    const int cc = (lane % 4) * 2;
    out[(cr + 0) * 8 + cc + 0] = D[0];
    out[(cr + 0) * 8 + cc + 1] = D[1];
    out[(cr + 8) * 8 + cc + 0] = D[2];
    out[(cr + 8) * 8 + cc + 1] = D[3];
}

// ===========================================================================
// PTX ISA 9.7.16.5.11 (mma.m16n8k64, e2m1) fragment model, empirically confirmed
// on this machine by k_map (one-hot sweep) + k_scale/k_scalerow:
//
//   groupID g = lane>>2, tig t = lane&3
//   A: a0 = row g,   k 8t..8t+7      a1 = row g+8, k 8t..8t+7
//      a2 = row g,   k 8t+32..+39    a3 = row g+8, k 8t+32..+39
//   B: b0 = col g,  k 8t..8t+7       b1 = col g,   k 8t+32..+39
//   C: c0,c1 = row g,   col 2t + {0,1};  c2,c3 = row g+8
//   SF_A (scale_vec::4X, one b32 = 4 UE4M3 bytes for k-slots 0..63 in 4 sub-vectors):
//        lane 4g+0 -> row g    lane 4g+1 -> row g+8   (lanes 4g+2,4g+3 ignored)
//   SF_B: lane 4g+0 -> column g (lanes 4g+1..3 ignored)
//
// Hardware k-slot s (0..63) maps to the ggml nvfp4 SUBBLOCK containing coord 16*(s/16)
// (same set; order inside the sub-vector only matters consistently between A and B).
// Fragment-native storage packs CONSECUTIVE coords in a byte (NVIDIA nvfp4 tensor format:
// byte i = coords 2i,2i+1); ggml packs (c, c+8).  We store probe tiles in native order,
// so raw LDS word loads land in exactly the right register with zero shuffle:
//   word w of a row (64-coord block f) = coords 64f + 8w .. 8w+7
//   A: a0=row[g] word f*8+t;  a1=row[g+8] f*8+t;  a2=row[g] f*8+t+4; a3=row[g+8] f*8+t+4
//   B: b0=row[g] f*8+t;       b1=row[g]   f*8+t+4
// ===========================================================================

// map probe v2: one-hot A (row 0, native coord c1) x one-hot B (row r2, native coord c2)
// with the REAL fragment loader.  Correct behavior: exactly one hit per (c1,r2) at
// C[row 0][col r2], and only when c2 == c1.
__global__ void k_map(float * __restrict__ out) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    for (int c1 = 0; c1 < 64; ++c1) {
        for (int c2 = 0; c2 < 64; ++c2) {
            for (int r2 = 0; r2 < 8; ++r2) {
                uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
                const int wa = c1 >> 3, wb = c2 >> 3;
                const uint32_t va = (uint32_t) 2u << (4 * (c1 & 7));
                const uint32_t vb = (uint32_t) 2u << (4 * (c2 & 7));
                if (g == 0) {                     // one-hot A row 0 (rows 1..15 empty)
                    if (wa == t)      a0 = va;    // wa in 0..3 -> a0
                    if (wa == t + 4)  a2 = va;    // wa in 4..7 -> a2
                }
                if (g == r2) {                    // one-hot B in column r2
                    if (wb == t)      b0 = vb;
                    if (wb == t + 4)  b1 = vb;
                }
                float D[4] = { 0.f, 0.f, 0.f, 0.f };
                const uint32_t Afr[4] = { a0, a1, a2, a3 };
                const uint32_t Bfr[2] = { b0, b1 };
                mma_nvfp4(D, Afr, Bfr, 0x38383838u, 0x38383838u);
                float s = fabsf(D[0]) + fabsf(D[1]) + fabsf(D[2]) + fabsf(D[3]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFF, s, off, 32);
                if (lane != 0) continue;
                const float exp = (c1 == c2) ? 1.f : 0.f;
                const int base = ((c1 * 64 + c2) * 8 + r2) * 4;
                out[base] = fabsf(s - exp) < 0.5f ? 1.f : 2.f;   // 1 = ok, 2 = WRONG
            }
        }
    }
}

// Scale-semantics tests with constant fills (all coords = E2M1 1.0 -> every word 0x22222222).
//   t=1: sa=sb=1.0 everywhere        -> C == 64 for every cell if the MMA depth really is k=64
//   t=2: sa=[1,2,4,8] sb=[1,2,4,8]   -> aligned sub-vectors: C == 16*(1+4+16+64) = 1360
//   t=3: sa=1.0,      sb=[8,4,2,1]   -> aligned: C == 16*(8+8+8+8) = 512
__global__ void k_scale(int t, float * __restrict__ out) {
    const int lane = threadIdx.x;
    const uint32_t fill = 0x22222222u;
    uint32_t sa = 0x38383838u, sb = 0x38383838u;
    if (t == 2) { sa = 0x50484038u; sb = 0x50484038u; }
    if (t == 3) { sa = 0x38383838u; sb = 0x50484038u; }
    float D[4] = { 0.f, 0.f, 0.f, 0.f };
    const uint32_t Afr[4] = { fill, fill, fill, fill };
    const uint32_t Bfr[2] = { fill, fill };
    mma_nvfp4(D, Afr, Bfr, sa, sb);
#pragma unroll
    for (int m = 0; m < 4; ++m) out[lane * 4 + m] = D[m];
}

// Scale row-binding test: sa/sb depend on the LANE, so the C rows/cols that scale up reveal
// which fragment rows (A) and columns (B) take their scale from which lane.
// t=4: sa = 2.0 only for lanes 0..3   -> which C rows double?
// t=5: sb = 2.0 only for lanes 0..3   -> which C cols double?
__global__ void k_scalerow(int t, float * __restrict__ out) {
    const int lane = threadIdx.x;
    const uint32_t fill = 0x22222222u;
    const uint32_t two = 0x40404040u, one = 0x38383838u;
    uint32_t sa = one, sb = one;
    if (t == 4 && lane < 4) sa = two;
    if (t == 5 && lane < 4) sb = two;
    float D[4] = { 0.f, 0.f, 0.f, 0.f };
    const uint32_t Afr[4] = { fill, fill, fill, fill };
    const uint32_t Bfr[2] = { fill, fill };
    mma_nvfp4(D, Afr, Bfr, sa, sb);
#pragma unroll
    for (int m = 0; m < 4; ++m) out[lane * 4 + m] = D[m];
}

static void set_code(uint8_t * qs, int coord, uint8_t code, int sub_off) {
    const int s = coord / QK_NVFP4_SUB;
    const int c = coord % QK_NVFP4_SUB;
    uint8_t & byte = qs[s * (QK_NVFP4_SUB / 2) + (c % 8)];
    if (c < 8) byte = (byte & 0xF0) | code;
    else       byte = (byte & 0x0F) | (code << 4);
    (void) sub_off;
}

// ===========================================================================
// throughput microbench: pure tensor-core issue rate
// ===========================================================================
template <int MODE>
__global__ void k_throughput(float * out, int iters) {
    const uint32_t lane = threadIdx.x & 31;
    uint32_t A[4] = { 0x22222222u + (lane & 3), 0x44444444u + (lane & 3), 0x11111111u + (lane & 3), 0x33333333u + (lane & 3) };
    uint32_t B[2] = { 0x22222222u + (lane & 3), 0x44444444u + (lane & 3) };
    uint32_t f16a[4] = { 0x3c003c00u + (lane & 1), 0x3c003c00u, 0x3c003c00u, 0x3c003c00u };
    uint32_t f16b[2] = { 0x3c003c00u, 0x3c003c00u };
    uint32_t sc = 0x38383838u + 0x40u * (lane & 1);   // 1.0 or 2.0 sub-scales, not const-foldable
    float D[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) D[i][j] = 0.f;

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if constexpr (MODE == 0) {
                mma_nvfp4(D[i], A, B, sc, sc);                       // fp4 m16n8k64
            } else if constexpr (MODE == 1) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(D[i][0]), "+f"(D[i][1]), "+f"(D[i][2]), "+f"(D[i][3])
                             : "r"(f16a[0]), "r"(f16a[1]), "r"(f16a[2]), "r"(f16a[3]), "r"(f16b[0]), "r"(f16b[1]));
            } else if constexpr (MODE == 2) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(D[i][0]), "+f"(D[i][1]), "+f"(D[i][2]), "+f"(D[i][3])
                             : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
            } else if constexpr (MODE == 3) {
                asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                             : "+f"(D[i][0]), "+f"(D[i][1]), "+f"(D[i][2]), "+f"(D[i][3])
                             : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(sc), "r"(sc));
            } else if constexpr (MODE == 4) {
                mma_nvfp4(D[i], A, B, sc, sc);
                mma_nvfp4(D[i], A, B, sc, sc);
            }
        }
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += D[i][0] + D[i][1] + D[i][2] + D[i][3];
    out[(threadIdx.x & 3) + (blockIdx.x & 7) * 4] = s;   // real store: no DCE
}

// ===========================================================================
// PathVEC: current fattn-vec nvfp4 QK^T semantics (q8_1 Q, integer dot, LUT codes)
//         warp = 32 threads, nthreads_KQ = 32, one K row per warp-iteration.
// ===========================================================================
__constant__ int c_kvalues_fp4[16];

// q8_1 Q in the SAME convention as fattn-common quantize_q8_1_to_shared:
// 32-element blocks (QK8_1=32): d = amax_block/127, q = round(x/d), ds = (d, sum_of_raw_x),
// one ds replicated for all 8 int-words of the block.
static void quantize_q8_1_fa(const float * x, int n, std::vector<int> & q8, std::vector<float2> & ds) {
    q8.resize(n / 4);
    ds.resize(n / 4);
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(x[b * 32 + i])); sum += x[b * 32 + i]; }
        const float d = amax / 127.0f;
        for (int w = 0; w < 8; ++w) {
            int32_t packed = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t q = (d != 0.f) ? (int8_t) lrintf(roundf(x[b * 32 + w * 4 + l] / d)) : 0;
                packed |= ((uint32_t)(uint8_t) q) << (8 * l);
            }
            q8[b * 8 + w] = (int) packed;
            ds[b * 8 + w] = make_float2(d, sum);
        }
    }
}

// PathVEC: replica of fattn-vec nvfp4 KQ semantics, now looping all nq Q rows per K token.
// Warp = one K token; each lane pre-loads its words' codes/scales, then per Q: integer dot + reduce.
__global__ void k_path_vec(const int * __restrict__ K_qs, const int * __restrict__ K_sc,
                           const int * __restrict__ Q_q8, const float2 * __restrict__ Q_ds,
                           float * __restrict__ out, int ntok, int nq, int D) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= ntok) return;

    const int nwords = D / 4;                       // 4-coord words per row (D=256 -> 64)
    const int wper   = nwords / 32;                 // words per thread (D=256 -> 2)

    const int * kq = K_qs + (size_t) warp * nwords;
    const uint8_t * kd = (const uint8_t *) K_sc;    // D/16 UE4M3 scale bytes per row
    const size_t qrow_stride = (size_t) nq * nwords;

    // preload per-token constants
    int8_t codes[16];
    float  dsub[4], sumc[4];
    for (int wi = 0; wi < wper; ++wi) {
        const int i  = lane * wper + wi;            // word index in the row
        const int ib = i >> 4;                      // 64-element block (16 words)
        const int iw = i & 15;                      // word within the block
        const int sub = iw >> 2;                    // 4 words per 16-element sub-block

        // ggml_cuda_ue4m3_to_fp32 includes the *0.5 convention so that (0.5*UE4M3) * (2*E2M1) = true value
        dsub[wi] = 0.5f * ue4m3_dev(kd[(size_t) warp * (D / 16) + ib * NSUB + sub]);
        const uint32_t word = (uint32_t) kq[ib * 8 + (iw >> 2) * 2 + (iw & 1)];   // 4 bytes at qs[ sub*8 + (iw&1)*4 ]; 8 words per 32-byte block
        float sc = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint8_t byte = (uint8_t) (word >> (8 * t));
            const uint8_t nib  = (byte >> ((iw & 2) ? 4 : 0)) & 0xF;
            codes[wi * 4 + t] = (int8_t) c_kvalues_fp4[nib];
            sc += codes[wi * 4 + t];
        }
        sumc[wi] = sc;
    }

    for (int q = 0; q < nq; ++q) {
        float sum = 0.0f;
        for (int wi = 0; wi < wper; ++wi) {
            const int gi = (size_t) q * nwords + lane * wper + wi;
            const int    u  = Q_q8[gi];
            const int8_t * q8 = (const int8_t *) &u;
            const float2 qds = Q_ds[gi];
            const float doti = q8[0]*codes[wi*4+0] + q8[1]*codes[wi*4+1] + q8[2]*codes[wi*4+2] + q8[3]*codes[wi*4+3];
            sum += dsub[wi] * (qds.x * doti + (qds.y / 32.0f) * sumc[wi]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, off, 32);
        if (lane == 0) out[(size_t) warp * nq + q] = sum;
    }
    (void) qrow_stride;
}

// ===========================================================================
// PathMMA: hardware block-scaled FP4 QK^T tile.
//   K tile: 64 tokens x D coords, copied verbatim from block_nvfp4 rows into shared
//   Q: nq rows (16/32/... of a 16-row multiple) pre-quantized to nvfp4 in global, copied verbatim
//   Each warp does 16 Q rows x 8 K tokens per mma, loops over k (D/64 fragments).
// ===========================================================================
// REPACK=false: K_qs / Q_qs already fragment-native. REPACK=true: K_qs / Q_qs are UNMODIFIED
// GGML block_nvfp4 words; nibble permutation happens in the staging loop.
template <int D, bool REPACK>
__global__ void k_path_mma(const int * __restrict__ K_qs, const int * __restrict__ K_sc,
                           const int * __restrict__ Q_qs, const int * __restrict__ Q_sc,
                           float * __restrict__ out, int ntok, int nq) {
    constexpr int KW  = D / 4;                  // code words per token row (D=256 -> 64)
    constexpr int KSc = KW + 1;                 // pad to break bank alignment
    constexpr int SW  = D / 64;                 // scale words per row (D=256 -> 4)

    extern __shared__ int shm[];
    int * Ksh  = shm;                                    // 64 rows x KSc
    int * Ksch = Ksh + 64 * KSc;                         // 64 rows x SW
    int * Qsh  = Ksch + 64 * SW;                         // nq rows x KSc
    int * Qsch = Qsh + nq * KSc;                         // nq rows x SW
    const int nwarps = blockDim.x / 32;

    {
        if constexpr (!REPACK) {
            const int total = 64 * KW;
            for (int i = threadIdx.x; i < total; i += blockDim.x) {
                const int r = i / KW, c = i % KW;
                Ksh[r * KSc + c] = K_qs[(size_t) (blockIdx.x * 64 + r) * KW + c];
            }
        } else {
            const int total = 64 * (KW / 2);                 // one thread per ggml word-pair
            for (int i = threadIdx.x; i < total; i += blockDim.x) {
                const int r = i / (KW / 2), p = i % (KW / 2);
                const int W0 = K_qs[(size_t) (blockIdx.x * 64 + r) * KW + 2 * p];
                const int W1 = K_qs[(size_t) (blockIdx.x * 64 + r) * KW + 2 * p + 1];
                uint32_t N0, N1;
                ggml_pair_to_native((uint32_t) W0, (uint32_t) W1, N0, N1);
                Ksh[r * KSc + 2 * p]     = (int) N0;
                Ksh[r * KSc + 2 * p + 1] = (int) N1;
            }
        }
        const int tots = 64 * SW;
        for (int i = threadIdx.x; i < tots; i += blockDim.x) {
            const int r = i / SW, c = i % SW;
            Ksch[r * SW + c] = K_sc[(size_t) (blockIdx.x * 64 + r) * SW + c];
        }
        if constexpr (!REPACK) {
            const int tq = nq * KW;
            for (int i = threadIdx.x; i < tq; i += blockDim.x) {
                const int r = i / KW, c = i % KW;
                Qsh[r * KSc + c] = Q_qs[(size_t) r * KW + c];
            }
        } else {
            const int tq = nq * (KW / 2);
            for (int i = threadIdx.x; i < tq; i += blockDim.x) {
                const int r = i / (KW / 2), p = i % (KW / 2);
                const int W0 = Q_qs[(size_t) r * KW + 2 * p];
                const int W1 = Q_qs[(size_t) r * KW + 2 * p + 1];
                uint32_t N0, N1;
                ggml_pair_to_native((uint32_t) W0, (uint32_t) W1, N0, N1);
                Qsh[r * KSc + 2 * p]     = (int) N0;
                Qsh[r * KSc + 2 * p + 1] = (int) N1;
            }
        }
        const int tsq = nq * SW;
        for (int i = threadIdx.x; i < tsq; i += blockDim.x) {
            const int r = i / SW, c = i % SW;
            Qsch[r * SW + c] = Q_sc[(size_t) r * SW + c];
        }
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int qslabs = nq / 16;
    const int ntiles = qslabs * 8;              // 8 token-octets per 64-token tile

    for (int t = warp; t < ntiles; t += nwarps) {
        const int q0 = (t / 8) * 16;
        const int k0 = (t % 8) * 8;

        float Dm[4] = { 0.f, 0.f, 0.f, 0.f };
        const int g = lane >> 2, tg = lane & 3;
        for (int f = 0; f < D / 64; ++f) {
            uint32_t A[4], B[2];
            A[0] = (uint32_t) Qsh[(q0 + g + 0) * KSc + f * 8 + tg + 0];
            A[1] = (uint32_t) Qsh[(q0 + g + 8) * KSc + f * 8 + tg + 0];
            A[2] = (uint32_t) Qsh[(q0 + g + 0) * KSc + f * 8 + tg + 4];
            A[3] = (uint32_t) Qsh[(q0 + g + 8) * KSc + f * 8 + tg + 4];
            B[0] = (uint32_t) Ksh[(k0 + g) * KSc + f * 8 + tg + 0];
            B[1] = (uint32_t) Ksh[(k0 + g) * KSc + f * 8 + tg + 4];

            const uint32_t a_scale = (uint32_t) Qsch[(q0 + g + 8 * (tg & 1)) * SW + f];
            const uint32_t b_scale = (uint32_t) Ksch[(k0 + g) * SW + f];

            mma_nvfp4(Dm, A, B, a_scale, b_scale);
        }

        // C fragment (m16n8 f32): m = lane/4 (+8), n = (lane%4)*2 + l%2.  Output layout: [token][qrow]
        const int cr = lane / 4;
        const int cc = (lane % 4) * 2;
        const int tok = blockIdx.x * 64 + k0;
        out[(size_t) (tok + cc + 0) * nq + (q0 + cr + 0)] = Dm[0];
        out[(size_t) (tok + cc + 1) * nq + (q0 + cr + 0)] = Dm[1];
        out[(size_t) (tok + cc + 0) * nq + (q0 + cr + 8)] = Dm[2];
        out[(size_t) (tok + cc + 1) * nq + (q0 + cr + 8)] = Dm[3];
    }
}

// ===========================================================================
// FUSED path: NVFP4 K decoded in-registers to f16 fragment words in smem (never any f16 KV
// in global memory), Q stays plain f16 (f32->f16 conversion only, no quantization), dense
// m16n8k16 f32.f16.f16.f32 mma computes QK^T (f32 accum).  Same tile shape / output layout
// as k_path_mma: block = 4 warps x 32 tokens (128 tokens), A side = 16 Q rows, B side = 8
// K-token columns, out[token*nq + q].  Input buffers identical to PathMMA-repacked
// (UNMODIFIED ggml words: KqW code + KsW scale words, fp32 Q rows).
//   K smem layout : Kh2[warp*32 + r][word j] = half2(dequant coords 2j,2j+1), stride HSTR.
//   Q smem layout : Qh2[q][word j] = half2(f16 coords 2j,2j+1).
//   frag loads    : A.x[0]=Qh2[(q0+g)*HSTR + 8f+t]  A.x[1]=row+8  A.x[2]=+4  A.x[3]=both
//                   B.x[0]=Kh2[(k0+g)*HSTR + 8f+t]  B.x[1]=+4          (PTX m16n8k16 f16 map)
// MODE 1 = decode only (no mma): measures pure NVFP4->f16 decode cost.
// ===========================================================================
#define FUSED_HSTR  (256 / 2 + 1)      // half2 words per row (+1 bank pad), D=256 only
// PRMT-based E2M1->f16 nibble-pair decode (avoids constant-cache serialization on divergent
// indices): magnitude high bytes {00,38,3C,3E,40,42,44,46} live in two source registers;
// f16 low bytes are always 0, so bits(T[n]) = T[n]<<8.  prmt selector nibbles pick table bytes
// directly (indices 0-3 = tl, 4-7 = th); d0=d2=0 via index 0 whose table byte is 0x00.
__device__ __forceinline__ uint32_t e2m1_pair_bits(uint32_t n0, uint32_t n1) {
    const uint32_t tl = 0x3E3C3800u;   // T0..T3  (0.0, 0.5, 1.0, 1.5 high bytes)
    const uint32_t th = 0x46444240u;   // T4..T7  (2, 3, 4, 6)
    const uint32_t sel = ((n0 & 7u) << 4) | ((n1 & 7u) << 12);
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(tl), "r"(th), "r"(sel));
    return r | ((n0 & 8u) << 12) | ((n1 & 8u) << 28);
}

__device__ __forceinline__ float ue4m3_hw_dev(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;
    if (exp == 0) return (float) mant * (1.0f / 512.0f);
    return (1.0f + (float) mant / 8.0f) * exp2f((float) (exp - 7));
}

__device__ __forceinline__ uint32_t pack_half2_dev(float a, float b) {
    half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t *>(&h);
}

template <int MODE, int TPR, bool QGMEM>
__global__ void k_path_fused(const int * __restrict__ K_qs, const int * __restrict__ K_sc,
                             const float * __restrict__ Q_f32, const uint32_t * __restrict__ Q16,
                             float * __restrict__ out, int ntok, int nq) {
    constexpr int KW   = 256 / 4;               // code words per K row (stride in K_qs)
    constexpr int SW   = 256 / 64;              // scale words per K row
    constexpr int NW   = 4;                     // warps
    constexpr int HSTR = FUSED_HSTR;
    constexpr int QSTR = QGMEM ? (256 / 2) : HSTR;

    extern __shared__ int shm[];
    int * Kh = shm;                                   // NW*TPR rows x HSTR half2 words
    int * Qh = Kh + NW * TPR * HSTR;                  // nq rows x HSTR (unused if QGMEM)

    const int lane = threadIdx.x & 31;

    // ---- Q stage: f32 -> f16 pairs, no quantization loss beyond f16 rounding.
    // With QGMEM the caller pre-staged f16 in gmem (query matrix only, NOT the KV cache).
    if constexpr (!QGMEM) {
        for (int i0 = 0; i0 < nq * (256 / 2); i0 += NW * 32) {
            const int i = i0 + threadIdx.x;
            const int j = i / (256 / 2), w = i % (256 / 2);
            const float a = Q_f32[(size_t) j * 256 + 2 * w + 0];
            const float b = Q_f32[(size_t) j * 256 + 2 * w + 1];
            Qh[j * HSTR + w] = (int) pack_half2_dev(a, b);
        }
    }

    // ---- K stage: warp-private decode of its TPR token rows, ggml bytes -> f16 half2 words
    const int wy   = threadIdx.x / 32;
    const int tokw = blockIdx.x * (NW * TPR) + wy * TPR;
    const int wbase = wy * TPR * HSTR;
    for (int i0 = 0; i0 < TPR * (256 / 16); i0 += 32) {
        const int i  = i0 + lane;
        const int r  = i / (256 / 16);
        const int pr = i % (256 / 16);
        const int grow = tokw + r;

        const uint32_t w0 = (uint32_t) K_qs[grow * KW + 2 * pr + 0];
        const uint32_t w1 = (uint32_t) K_qs[grow * KW + 2 * pr + 1];
        const uint32_t db = ((uint32_t) K_sc[grow * SW + (pr >> 2)] >> (8 * (pr & 3))) & 0xFFu;
        const half2 sh = __half2half2(__float2half_rn(ue4m3_hw_dev((uint8_t) db)));

        int * dst = Kh + wbase + r * HSTR + 8 * pr;
        // ggml nvfp4 subblock packing: byte b holds coords (b, b+8) of the 16-coord subblock.
        // half2 word w must hold consecutive coords (2w, 2w+1): low nibbles of bytes 2w,2w+1 for
        // w<4, high nibbles of bytes 2(w-4),2(w-4)+1 for w>=4.
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int sel = w & 3;
            const int j0 = 2 * sel, j1 = 2 * sel + 1;
            const uint32_t by0 = (j0 < 4) ? ((w0 >> (8 * j0)) & 0xFFu) : ((w1 >> (8 * (j0 - 4))) & 0xFFu);
            const uint32_t by1 = (j1 < 4) ? ((w0 >> (8 * j1)) & 0xFFu) : ((w1 >> (8 * (j1 - 4))) & 0xFFu);
            const uint32_t n0 = (w < 4) ? (by0 & 0xFu) : (by0 >> 4);
            const uint32_t n1 = (w < 4) ? (by1 & 0xFu) : (by1 >> 4);
            const uint32_t bits = e2m1_pair_bits(n0, n1);
            half2 hv = *reinterpret_cast<const half2 *>(&bits);
            hv = __hmul2(hv, sh);
            dst[w] = (int) *reinterpret_cast<uint32_t *>(&hv);
        }
    }
    __syncthreads();

    if constexpr (MODE == 1) {
        // decode-only: emit per-token sum of decoded halves (keeps decode + smem traffic live)
        const int r0 = lane / 4, sub = lane % 4;
        for (int r = r0; r < TPR; r += 8) {
            float s = 0.f;
            for (int w = sub * (256 / 2 / 4); w < (sub + 1) * (256 / 2 / 4); ++w) {
                const half2 hv = *reinterpret_cast<const half2 *>(&Kh[wbase + r * HSTR + w]);
                s += __low2float(hv) + __high2float(hv);
            }
#pragma unroll
            for (int off = 1; off < 4; off <<= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off, 32);
            if (sub == 0 && tokw + r < ntok) out[tokw + r] = s;
        }
        return;
    }

    // ---- QK^T: A = 16 Q rows, B = 8 token columns, C map (verified in k_path_mma):
    // row = lane/4 (+8) = q, col = (lane%4)*2 + e = token offset
    const int g = lane >> 2, tg = lane & 3;
    const int * Khw = Kh + wbase;
    const int * Qsrc = QGMEM ? reinterpret_cast<const int *>(Q16) : Qh;

    for (int q0 = 0; q0 < nq; q0 += 16) {
        for (int k0 = 0; k0 < TPR; k0 += 8) {
            float Dm[4] = { 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int f = 0; f < 256 / 16; ++f) {
                uint32_t A[4], B[2];
                A[0] = (uint32_t) Qsrc[(q0 + g + 0) * QSTR + f * 8 + tg + 0];
                A[1] = (uint32_t) Qsrc[(q0 + g + 8) * QSTR + f * 8 + tg + 0];
                A[2] = (uint32_t) Qsrc[(q0 + g + 0) * QSTR + f * 8 + tg + 4];
                A[3] = (uint32_t) Qsrc[(q0 + g + 8) * QSTR + f * 8 + tg + 4];
                B[0] = (uint32_t) Khw[(k0 + g) * HSTR + f * 8 + tg + 0];
                B[1] = (uint32_t) Khw[(k0 + g) * HSTR + f * 8 + tg + 4];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(Dm[0]), "+f"(Dm[1]), "+f"(Dm[2]), "+f"(Dm[3])
                             : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
            }
            const int tok = tokw + k0 + tg * 2;
            const int qq  = q0 + g;
            if (tok + 0 < ntok) out[(size_t) (tok + 0) * nq + qq + 0] = Dm[0];
            if (tok + 1 < ntok) out[(size_t) (tok + 1) * nq + qq + 0] = Dm[1];
            if (tok + 0 < ntok) out[(size_t) (tok + 0) * nq + qq + 8] = Dm[2];
            if (tok + 1 < ntok) out[(size_t) (tok + 1) * nq + qq + 8] = Dm[3];
        }
    }
}

// ===========================================================================
// Device Q prep: fp32 rows -> UNMODIFIED ggml block_nvfp4 words (codes + scale words).
// Bit-exact mirror of host nvfp4_quant_row above: same amax chain order (j=0..15), same
// ggml_fp32_to_ue4m3 port, and nearest-E2M1 as exact midpoint comparisons.  All E2M1
// thresholds times a 4-bit-mantissa dd are exactly representable in fp32, so comparing
// a < mid*dd reproduces the host's strict-< tie-break (ties go to the smaller code, and
// -0.1 -> code 0, never code 8) bit-for-bit.  Layout: one 16-coord subblock per lane,
// (c, c+8) byte order, words 2l / 2l+1 of the row, one LE scale word per 4 subblocks.
// ===========================================================================
__device__ __forceinline__ uint8_t ue4m3_from_f32_dev(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    const uint32_t bits = __float_as_uint(x);
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

__device__ __forceinline__ uint32_t e2m1_code_dev(float v, float dd) {
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

template <int D>
__global__ void k_quant_nvfp4(const float * __restrict__ x, int * __restrict__ qw,
                              int * __restrict__ qsw, int nrows) {
    constexpr int S = D / QK_NVFP4_SUB;   // subblocks per row (D=256 -> 16)
    constexpr int R = 32 / S;             // rows per warp
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const bool act = wid * R + lane / S < nrows;
    const int row  = act ? wid * R + lane / S : nrows - 1;
    const int l    = lane % S;

    const float * xb = x + (size_t) row * D + l * QK_NVFP4_SUB;
    float v[QK_NVFP4_SUB];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_NVFP4_SUB; ++j) { v[j] = xb[j]; amax = fmaxf(amax, fabsf(v[j])); }

    const uint8_t ue = ue4m3_from_f32_dev(amax / 6.0f);
    const float   dd = ue4m3_dev(ue);

    uint32_t W0 = 0, W1 = 0;
    #pragma unroll
    for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
        const uint32_t byte = e2m1_code_dev(v[j], dd) | (e2m1_code_dev(v[QK_NVFP4_SUB / 2 + j], dd) << 4);
        if (j < 4) W0 |= byte << (8 * j); else W1 |= byte << (8 * (j - 4));
    }
    if (act) {
        qw[(size_t) row * (D / 4) + 2 * l]     = (int) W0;
        qw[(size_t) row * (D / 4) + 2 * l + 1] = (int) W1;
    }

    // one scale word per 4 consecutive subblocks, LE: byte s = UE4M3 of subblock s
    const uint32_t uv = ue;
    const uint32_t u = uv
        | (__shfl_sync(0xffffffffu, uv, lane + 1, 32) << 8)
        | (__shfl_sync(0xffffffffu, uv, lane + 2, 32) << 16)
        | (__shfl_sync(0xffffffffu, uv, lane + 3, 32) << 24);
    if (act && (lane & 3) == 0) qsw[(size_t) row * (D / 64) + (l >> 2)] = (int) u;
}

// debug: same as k_path_vec but writes per-lane pre-reduce partials for one (tok,q)
__global__ void k_path_vec_dbg(const int * __restrict__ K_qs, const int * __restrict__ K_sc,
                               const int * __restrict__ Q_q8, const float2 * __restrict__ Q_ds,
                               float * __restrict__ part, int tok, int q, int D) {
    const int lane = threadIdx.x % 32;
    const int nwords = D / 4;
    const int wper   = nwords / 32;
    const int * kq = K_qs + (size_t) tok * nwords;
    const uint8_t * kd = (const uint8_t *) K_sc;
    float sum = 0.0f;
    for (int wi = 0; wi < wper; ++wi) {
        const int i  = lane * wper + wi;
        const int ib = i >> 4;
        const int iw = i & 15;
        const int sub = iw >> 2;
        const float dsub = 0.5f * ue4m3_dev(kd[(size_t) tok * (D / 16) + ib * NSUB + sub]);
        const uint32_t word = (uint32_t) kq[ib * 8 + (iw >> 2) * 2 + (iw & 1)];
        const int u = Q_q8[(size_t) q * nwords + i];
        const int8_t * q8 = (const int8_t *) &u;
        const float2 qds = Q_ds[(size_t) q * nwords + i];
        int kv[4]; float sc = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint8_t byte = (uint8_t) (word >> (8 * t));
            const uint8_t nib  = (byte >> ((iw & 2) ? 4 : 0)) & 0xF;
            kv[t] = c_kvalues_fp4[nib];
            sc += kv[t];
        }
        const float doti = q8[0]*kv[0] + q8[1]*kv[1] + q8[2]*kv[2] + q8[3]*kv[3];
        sum += dsub * (qds.x * doti + (qds.y / 32.0f) * sc);
    }
    part[lane] = sum;
}

// ---------------------------------------------------------------------------
// kernelcheck: literal replicas of the fattn-vec.cuh hi+lo Q prep and of its
// K-staging + two-chain block-scaled mma loop, compared against the host
// reference. Isolates "codec ported wrong" from "fragments/scales wired wrong".
// ---------------------------------------------------------------------------

// E2M1 dequant mirror of fattn-vec.cuh ggml_cuda_e2m1_value.
__device__ __forceinline__ float e2m1_value_dev(const uint32_t code) {
    const uint32_t m = code & 7u;
    const float mag = (m < 4u) ? ldexpf((float) m, -1)
                               : (m == 4u ? 2.0f : (m == 5u ? 3.0f : (m == 6u ? 4.0f : 6.0f)));
    return (code & 8u) ? -mag : mag;
}

// Replica of the KQ_mma Q-prep loop (8 rows, one 16-subblock per lane, fragment-native
// consecutive-pair words + LE scale dwords + residual-split lo chain). blockDim.x = 128.
__global__ void k_fa_prep_replica(const float * __restrict__ Q, int * __restrict__ Qnc, int * __restrict__ Qsc,
                                  int * __restrict__ Qnc2, int * __restrict__ Qsc2, int D) {
    const int F4_STR  = D / 8 + 1;
    const int F4_NSUB = D / QK_NVFP4_SUB;
    const int F4_RPP  = 32 / F4_NSUB;
    const int nwarps  = blockDim.x / 32;
    const int l  = threadIdx.x % F4_NSUB;
    const int rr = threadIdx.x / F4_NSUB;
    for (int j0 = 0; j0 < 8; j0 += nwarps * F4_RPP) {
        const int j = j0 + (threadIdx.x / 32) * F4_RPP + rr;
        if (j >= 8) break;
        const float * Q_f = Q + (size_t) j * D;

        float v[QK_NVFP4_SUB];
        float amax = 0.0f;
#pragma unroll
        for (int c = 0; c < QK_NVFP4_SUB; ++c) {
            v[c] = Q_f[l*QK_NVFP4_SUB + c];
            amax = fmaxf(amax, fabsf(v[c]));
        }
        const uint8_t ue = ue4m3_from_f32_dev(amax / 6.0f);
        const float   dd = ue4m3_dev(ue);

        uint32_t cd[QK_NVFP4_SUB];
        uint32_t N0 = 0, N1 = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            cd[2*b + 0] = e2m1_code_dev(v[2*b + 0], dd);
            cd[2*b + 1] = e2m1_code_dev(v[2*b + 1], dd);
            cd[2*b + 8] = e2m1_code_dev(v[2*b + 8], dd);
            cd[2*b + 9] = e2m1_code_dev(v[2*b + 9], dd);
            N0 |= (cd[2*b + 0] | (cd[2*b + 1] << 4)) << (8*b);
            N1 |= (cd[2*b + 8] | (cd[2*b + 9] << 4)) << (8*b);
        }
        Qnc[j*F4_STR + 2*l]     = (int) N0;
        Qnc[j*F4_STR + 2*l + 1] = (int) N1;

        uint32_t uu = ue;
        uu |= __shfl_sync(0xFFFFFFFFu, uu, threadIdx.x + 1, 32) << 8;
        uu |= __shfl_sync(0xFFFFFFFFu, uu, threadIdx.x + 2, 32) << 16;
        uu |= __shfl_sync(0xFFFFFFFFu, uu, threadIdx.x + 3, 32) << 24;
        if ((l & 3) == 0) Qsc[j*(D/64) + (l >> 2)] = (int) uu;

        float rv[QK_NVFP4_SUB];
        float amax_r = 0.0f;
#pragma unroll
        for (int c = 0; c < QK_NVFP4_SUB; ++c) {
            rv[c] = v[c] - dd * e2m1_value_dev(cd[c]);
            amax_r = fmaxf(amax_r, fabsf(rv[c]));
        }
        const uint8_t ue2 = ue4m3_from_f32_dev(amax_r / 6.0f);
        const float   dd2 = ue4m3_dev(ue2);

        uint32_t M0 = 0, M1 = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            M0 |= (e2m1_code_dev(rv[2*b + 0], dd2) | (e2m1_code_dev(rv[2*b + 1], dd2) << 4)) << (8*b);
            M1 |= (e2m1_code_dev(rv[2*b + 8], dd2) | (e2m1_code_dev(rv[2*b + 9], dd2) << 4)) << (8*b);
        }
        Qnc2[j*F4_STR + 2*l]     = (int) M0;
        Qnc2[j*F4_STR + 2*l + 1] = (int) M1;

        uint32_t u2 = ue2;
        u2 |= __shfl_sync(0xFFFFFFFFu, u2, threadIdx.x + 1, 32) << 8;
        u2 |= __shfl_sync(0xFFFFFFFFu, u2, threadIdx.x + 2, 32) << 16;
        u2 |= __shfl_sync(0xFFFFFFFFu, u2, threadIdx.x + 3, 32) << 24;
        if ((l & 3) == 0) Qsc2[j*(D/64) + (l >> 2)] = (int) u2;
    }
}

// Replica of the nvfp4 K-staging + mt/f mma loop (1 warp = 32 K tokens). Dt2 accumulates
// hi+lo, Dh1 hi only. out[((mt*32 + lane)*8) + e]: e0..3 Dt2 regs, e4..7 Dh1 regs.
__global__ void k_fa_mma_replica(const uint8_t * __restrict__ Kbytes, const int * __restrict__ Qnc, const int * __restrict__ Qsc,
                                 const int * __restrict__ Qnc2, const int * __restrict__ Qsc2,
                                 float * __restrict__ out, int D) {
    const int  F4_STR = D / 8 + 1;
    const int  nblk   = D / 64, nsb = D / 16;
    constexpr int SMAX = 33;
    __shared__ int sKnc[32*SMAX], sKsc[32*4], sQnc[8*SMAX], sQsc[8*4], sQnc2[8*SMAX], sQsc2[8*4];

    for (int i0 = 0; i0 < 32*nsb; i0 += 32) {
        const int i  = i0 + threadIdx.x;
        const int r  = i / nsb;
        const int pr = i % nsb;
        const int * words = (const int *) (Kbytes + (size_t) r * nblk * 36 + (pr >> 2) * 36);
        uint32_t N0, N1;
        ggml_pair_to_native((uint32_t) words[1 + 2*(pr & 3)], (uint32_t) words[2 + 2*(pr & 3)], N0, N1);
        sKnc[r*F4_STR + 2*pr]     = (int) N0;
        sKnc[r*F4_STR + 2*pr + 1] = (int) N1;
    }
    for (int i0 = 0; i0 < 32*nblk; i0 += 32) {
        const int i = i0 + threadIdx.x;
        const int r = i / nblk;
        const int f = i % nblk;
        sKsc[r*nblk + f] = *(const int *) (Kbytes + (size_t) r * nblk * 36 + (size_t) f * 36);
    }
    for (int i = threadIdx.x; i < 8*F4_STR; i += 32) { sQnc[i] = Qnc[i]; sQnc2[i] = Qnc2[i]; }
    for (int i = threadIdx.x; i < 8*nblk;  i += 32) { sQsc[i] = Qsc[i]; sQsc2[i] = Qsc2[i]; }
    __syncwarp();

    const int g = threadIdx.x >> 2;
    const int t = threadIdx.x &  3;
    for (int mt = 0; mt < 2; ++mt) {
        float Dt2[4] = {0.f, 0.f, 0.f, 0.f};
        float Dh1[4] = {0.f, 0.f, 0.f, 0.f};
        for (int f = 0; f < nblk; ++f) {
            uint32_t At[4], Bt[2], Bt2[2];
            At[0] = sKnc[(16*mt + g + 0)*F4_STR + f*8 + t + 0];
            At[1] = sKnc[(16*mt + g + 8)*F4_STR + f*8 + t + 0];
            At[2] = sKnc[(16*mt + g + 0)*F4_STR + f*8 + t + 4];
            At[3] = sKnc[(16*mt + g + 8)*F4_STR + f*8 + t + 4];
            Bt[0] = sQnc[g*F4_STR + f*8 + t + 0];
            Bt[1] = sQnc[g*F4_STR + f*8 + t + 4];
            const uint32_t sa  = (uint32_t) sKsc[(16*mt + g + 8*(threadIdx.x & 1))*nblk + f];
            const uint32_t sb  = (uint32_t) sQsc[g*nblk + f];
            const uint32_t sb2 = (uint32_t) sQsc2[g*nblk + f];
            mma_nvfp4(Dt2, At, Bt,  sa, sb);
            mma_nvfp4(Dh1, At, Bt,  sa, sb);
            mma_nvfp4(Dt2, At, Bt2, sa, sb2);
        }
        for (int e = 0; e < 4; ++e) {
            out[((mt*32 + threadIdx.x)*8) + e]     = Dt2[e];
            out[((mt*32 + threadIdx.x)*8) + 4 + e] = Dh1[e];
        }
    }
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static uint32_t pack4(const uint8_t * b) { return (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24); }

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static float frand(uint32_t & st) {
    st = st * 1664525u + 1013904223u;
    const float u = (float) (st >> 8) / 16777216.0f;
    st = st * 1664525u + 1013904223u;
    const float v = (float) (st >> 8) / 16777216.0f + 1e-9f;
    return sqrtf(-2.0f * logf(v)) * cosf(6.28318530718f * u);
}

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string mode = "all";
    int ntok = 4096, D = 256, nq = 16, iters = 2000, reps = 20;
    uint32_t seed = 12345;
    std::string qfile = "";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--D") && i + 1 < argc) D = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) nq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qfile") && i + 1 < argc) qfile = argv[++i];
    }

    int dev = 0; cudaGetDevice(&dev);
    cudaMemcpyToSymbol(c_kvalues_fp4, kvalues_fp4, sizeof(kvalues_fp4));
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, dev), "getDeviceProperties");
    int clk = 0, mclk = 0, busw = 0;
    cudaDeviceGetAttribute(&clk,  cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, dev);
    printf("# device: %s  cc=%d.%d  SMs=%d  clock=%.0f MHz  memclock=%d kHz  buswidth=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           clk / 1000.0, mclk, busw);

    // ------------------------------------------------------------------- verify
    if (mode == "verify" || mode == "all") {
        printf("\n=== VERIFY: block_nvfp4 bytes -> kind::mxf4nvf4 block_scale MMA ===\n");
        for (int test = 1; test <= 3; ++test) {
            // host tiles
            std::vector<int> qsA(16 * VSTRIDE, 0), qsB(8 * VSTRIDE, 0);
            std::vector<int> scA(16 * VSTRIDE, 0), scB(8 * VSTRIDE, 0);
            std::vector<uint8_t> dA(16 * NSUB), qA(16 * (QK_NVFP4 / 2));
            std::vector<uint8_t> dB(8 * NSUB),  qB(8 * (QK_NVFP4 / 2));
            std::vector<float>  fA(16 * QK_NVFP4), fB(8 * QK_NVFP4);

            uint32_t st = seed + test;
            for (int i = 0; i < 16 * QK_NVFP4; ++i) fA[i] = frand(st) * 2.0f;
            for (int i = 0; i < 8 * QK_NVFP4; ++i)  fB[i] = frand(st) * 2.0f;
            nvfp4_quant_row(fA.data(), 16, dA.data(), qA.data());   // 16 independent 64-coord rows
            nvfp4_quant_row(fB.data(), 8,  dB.data(), qB.data());   // 8 independent rows

            if (test == 1) {
                for (int i = 0; i < 16 * NSUB; ++i) dA[i] = ue4m3_from_f32(1.0f);
                for (int i = 0; i < 8 * NSUB; ++i)  dB[i] = ue4m3_from_f32(1.0f);
            } else if (test == 2) {
                for (int r = 0; r < 16; ++r) for (int s = 0; s < NSUB; ++s) dA[r * NSUB + s] = ue4m3_from_f32(1.0f * (r % 4 + 1));
                for (int r = 0; r < 8; ++r)  for (int s = 0; s < NSUB; ++s) dB[r * NSUB + s] = ue4m3_from_f32(1.0f);
            } else {
                for (int r = 0; r < 16; ++r) for (int s = 0; s < NSUB; ++s) dA[r * NSUB + s] = ue4m3_from_f32(ldexpf(1.0f, s));
                for (int r = 0; r < 8; ++r)  for (int s = 0; s < NSUB; ++s) dB[r * NSUB + s] = ue4m3_from_f32(1.0f);
            }
            // recompute dequantized values with the (possibly overwritten) scales
            nvfp4_dequant_row(dA.data(), qA.data(), 16, fA.data());
            nvfp4_dequant_row(dB.data(), qB.data(), 8,  fB.data());

            // scatter into the shared-style word layout the kernel expects (fragment-native order)
            for (int r = 0; r < 16; ++r) {
                ggml_to_native_words(qA.data(), r, 1, qsA.data(), VSTRIDE);
                scA[r * VSTRIDE] = (int) pack4(&dA[r * NSUB]);
            }
            for (int r = 0; r < 8; ++r) {
                ggml_to_native_words(qB.data(), r, 1, qsB.data(), VSTRIDE);
                scB[r * VSTRIDE] = (int) pack4(&dB[r * NSUB]);
            }

            int * d_qsA, * d_scA, * d_qsB, * d_scB; float * d_out;
            ck(cudaMalloc(&d_qsA, qsA.size() * 4), "malloc A");
            ck(cudaMalloc(&d_scA, scA.size() * 4), "malloc sA");
            ck(cudaMalloc(&d_qsB, qsB.size() * 4), "malloc B");
            ck(cudaMalloc(&d_scB, scB.size() * 4), "malloc sB");
            ck(cudaMalloc(&d_out, 16 * 8 * 4), "malloc out");
            ck(cudaMemcpy(d_qsA, qsA.data(), qsA.size() * 4, cudaMemcpyHostToDevice), "cpy A");
            ck(cudaMemcpy(d_scA, scA.data(), scA.size() * 4, cudaMemcpyHostToDevice), "cpy sA");
            ck(cudaMemcpy(d_qsB, qsB.data(), qsB.size() * 4, cudaMemcpyHostToDevice), "cpy B");
            ck(cudaMemcpy(d_scB, scB.size() ? scB.data() : nullptr, scB.size() * 4, cudaMemcpyHostToDevice), "cpy sB");

            k_verify<<<1, 32>>>(d_qsA, d_scA, d_qsB, d_scB, d_out, 1);
            ck(cudaGetLastError(), "launch verify");
            ck(cudaDeviceSynchronize(), "sync verify");

            std::vector<float> got(16 * 8);
            ck(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "cpy out");

            double maxdiff = 0.0, maxabs = 0.0, dot = 0.0, nrm = 0.0;
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 8; ++j) {
                    double ref = 0.0;
                    for (int c = 0; c < QK_NVFP4; ++c) ref += (double) fA[i * QK_NVFP4 + c] * (double) fB[j * QK_NVFP4 + c];
                    const double g = got[i * 8 + j];
                    maxdiff = fmax(maxdiff, fabs(g - ref));
                    maxabs  = fmax(maxabs, fabs(ref));
                    dot += g * ref; nrm += ref * ref;
                }
            }
            printf("T%d: max|gpu-ref| = %.6g   max|ref| = %.6g   cos = %.6f\n",
                   test, maxdiff, maxabs, nrm > 0 ? dot / sqrt(nrm * nrm) : 0.0);
            cudaFree(d_qsA); cudaFree(d_scA); cudaFree(d_qsB); cudaFree(d_scB); cudaFree(d_out);
        }
    }

    // ------------------------------------------------------------------- map
    if (mode == "map" || mode == "all") {
        printf("\n=== MAP v2: fragment-native one-hot sweep (expect diagonal only) ===\n");
        const size_t n = (size_t) 64 * 64 * 8 * 4;
        float * d_map = nullptr;
        ck(cudaMalloc(&d_map, n * 4), "malloc map");
        k_map<<<1, 32>>>(d_map);
        ck(cudaGetLastError(), "launch map");
        ck(cudaDeviceSynchronize(), "sync map");
        std::vector<float> mp(n);
        ck(cudaMemcpy(mp.data(), d_map, n * 4, cudaMemcpyDeviceToHost), "cpy map");
        int nok = 0, nbad = 0;
        for (size_t i = 0; i < n; i += 4) {
            if (mp[i] == 1.f) { ++nok; continue; }
            if (mp[i] != 0.f) {
                ++nbad;
                if (nbad <= 20) {
                    const int idx = (int) (i / 4);
                    printf("BAD c1=%d c2=%d r2=%d\n", idx / 512, (idx / 8) % 64, idx % 8);
                }
            }
        }
        printf("map groups: ok=%d wrong=%d (total %d)\n", nok, nbad, (int) (n / 4));
        cudaFree(d_map);

        printf("\n=== SCALE tests (constant 1.0 operands) ===\n");
        float * d_sc = nullptr;
        ck(cudaMalloc(&d_sc, 128 * 4), "malloc scale");
        for (int t = 1; t <= 3; ++t) {
            k_scale<<<1, 32>>>(t, d_sc);
            ck(cudaGetLastError(), "launch scale");
            ck(cudaDeviceSynchronize(), "sync scale");
            std::vector<float> g(128);
            ck(cudaMemcpy(g.data(), d_sc, 128 * 4, cudaMemcpyDeviceToHost), "cpy scale");
            double mn = 1e30, mx = -1e30; bool same = true; float first = g[0];
            for (float v : g) { mn = fmin(mn, (double) v); mx = fmax(mx, (double) v); if (v != first) same = false; }
            printf("t=%d: %s value=%g\n", t, same ? "uniform" : "NON-UNIFORM", first);
            if (!same) printf("   min=%g max=%g\n", mn, mx);
        }
        for (int t = 4; t <= 5; ++t) {
            k_scalerow<<<1, 32>>>(t, d_sc);
            ck(cudaGetLastError(), "launch scalerow");
            ck(cudaDeviceSynchronize(), "sync scalerow");
            float mat[16][8];
            for (auto & r : mat) for (auto & v : r) v = 0.f;
            std::vector<float> g(128);
            ck(cudaMemcpy(g.data(), d_sc, 128 * 4, cudaMemcpyDeviceToHost), "cpy scalerow");
            for (int L = 0; L < 32; ++L)
                for (int m = 0; m < 4; ++m)
                    mat[L / 4 + (m >= 2 ? 8 : 0)][(L % 4) * 2 + (m % 2)] = g[L * 4 + m];
            printf("t=%d grid (rows x cols), doubled rows/cols reveal lane->scale binding:\n", t);
            for (int r = 0; r < 16; ++r) {
                printf("  r%2d:", r);
                for (int c = 0; c < 8; ++c) printf(" %3.0f", mat[r][c]);
                printf("\n");
            }
        }
        cudaFree(d_sc);
    }

    // --------------------------------------------------------------- numerics
    if (mode == "numerics" || mode == "all") {
        printf("\n=== NUMERICS: score error by Q/K codec pair (synthetic gaussian, D=%d) ===\n", D);
        const int NQ = 64, NT = min(ntok, 4096), nb = D / QK_NVFP4;
        std::vector<float> Qf(NQ * D), Kf((size_t) NT * D);
        uint32_t st = seed;
        for (auto & v : Qf) v = frand(st);
        for (auto & v : Kf) v = frand(st) * 1.5f;

        // K codecs
        std::vector<uint8_t> Kd((size_t) NT * (D / 16)), Kq((size_t) NT * (D / 2)), Kdq8((size_t) NT * (D / 16));
        std::vector<uint8_t> Kq4((size_t) NT * (D / 2));           // mxfp4: ue8m0 per 32 + E2M1
        std::vector<float> Kdq((size_t) NT * D), Km((size_t) NT * D);
        for (int r = 0; r < NT; ++r) {
            nvfp4_quant_row(&Kf[(size_t) r * D], nb, &Kd[(size_t) r * (D / 16)], &Kq[(size_t) r * (D / 2)]);
            nvfp4_dequant_row(&Kd[(size_t) r * (D / 16)], &Kq[(size_t) r * (D / 2)], nb, &Kdq[(size_t) r * D]);
            // mxfp4 (OCP MX): ue8m0 power-of-two scale per 32 coords, E2M1 values max 6
            for (int blk = 0; blk < D / 32; ++blk) {
                const float * xb = &Kf[(size_t) r * D + blk * 32];
                float amax = 0.f;
                for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
                int e = 0;
                while (6.0f * ldexpf(1.0f, e) < amax) ++e;
                while (e > -127 && 3.0f * ldexpf(1.0f, e) >= amax) --e;
                const float sc = ldexpf(1.0f, e);
                Kdq8[(size_t) r * (D / 16) / 2 + blk] = (uint8_t) (e + 127);
                for (int j = 0; j < 32; ++j) {
                    const float y = xb[j] / sc;
                    static const float tab[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
                    float best = 1e30f; int bc = 0;
                    for (int c = 0; c < 8; ++c) { const float er = fabsf(y - tab[c]); if (er < best) { best = er; bc = c; } }
                    if (y < 0) bc |= 8;
                    float val = tab[bc & 7] * (float)((bc >> 3) ? -1 : 1) * sc;
                    Km[(size_t) r * D + blk * 32 + j] = val;
                }
            }
        }
        // Q codecs
        std::vector<float> Qn(D * NQ), Q8(D * NQ);
        std::vector<uint8_t> Qd(NQ * (D / 16)), Qqc(NQ * (D / 2));
        for (int r = 0; r < NQ; ++r) {
            nvfp4_quant_row(&Qf[(size_t) r * D], nb, &Qd[r * (D / 16)], &Qqc[r * (D / 2)]);
            nvfp4_dequant_row(&Qd[r * (D / 16)], &Qqc[r * (D / 2)], nb, &Qn[r * D]);
            std::vector<int> q8; std::vector<float2> ds;
            quantize_q8_1_fa(&Qf[(size_t) r * D], D, q8, ds);       // exact dequant of q8_1: d*q
            for (int c = 0; c < D; ++c) {
                const int w = c / 4, l = c % 4;
                const int8_t * q8b = (const int8_t *) &q8[w];
                Q8[r * D + c] = ds[w].x * (float) q8b[l];
            }
        }
        // e4m3 Q with ue8m0 per-32 (mxf8f6f4 semantics), paired with mxfp4 K
        std::vector<float> Qe(NQ * D);
        for (int r = 0; r < NQ; ++r) {
            for (int blk = 0; blk < D / 32; ++blk) {
                const float * xb = &Qf[(size_t) r * D + blk * 32];
                float amax = 0.f;
                for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
                int e = 0;
                while (448.0f * ldexpf(1.0f, e) < amax) ++e;
                while (e > -127 && 224.0f * ldexpf(1.0f, e) >= amax) --e;
                const float sc = ldexpf(1.0f, e);
                for (int j = 0; j < 32; ++j) {
                    const float y = xb[j] / sc;                      // in [-448, 448]
                    const float a = fabsf(y);
                    float q;
                    if (a >= 0.015625f) {                             // normal e4m3: 3-bit mantissa
                        int be; frexpf(a, &be);                       // a = m*2^be, m in [0.5,1)
                        int E = be - 1; if (E < -6) E = -6;
                        const float step = ldexpf(1.0f, E - 3);
                        q = rintf(a / step) * step;
                    } else q = rintf(a * 512.0f) / 512.0f;            // subnormal grid
                    if (q > 448.0f) q = 448.0f;
                    Qe[(size_t) r * D + blk * 32 + j] = q * (y < 0 ? -1.f : 1.f) * sc;
                }
            }
        }
        printf("  %-38s\n", "Score error vs TRUE fp32 QK^T (double accumulation):");
        { struct { const char * n; const float * q; const float * k; } rows[] = {
            { "Q fp32   x K nvfp4 (K-only floor)", Qf.data(), Kdq.data() },
            { "Q q8_1   x K nvfp4 (exact product)", Q8.data(), Kdq.data() },
            { "Q nvfp4  x K nvfp4 (= PathMMA)",     Qn.data(), Kdq.data() },
            { "Q e4m3   x K mxfp4 (mxf8f6f4 path)", Qe.data(), Km.data() },
        }; for (auto & rw : rows) { const float * Q = rw.q; const float * K = rw.k;
            double e2 = 0, mx = 0, ref = 0, cnt2 = 0;
            for (int t = 0; t < std::min(NT, 1024); ++t) for (int r = 0; r < NQ; ++r) {
                double tru = 0, got = 0;
                for (int c = 0; c < D; ++c) { tru += (double) Qf[r * D + c] * (double) Kf[(size_t) t * D + c]; got += (double) Q[r * D + c] * (double) K[(size_t) t * D + c]; }
                const double d = got - tru; e2 += d * d; mx = fmax(mx, fabs(d)); ref += fabs(tru); cnt2++;
            }
            printf("  %-38s rms = %-8.4g max = %-8.4g  (mean|score| = %.2f)\n", rw.n, sqrt(e2 / cnt2), mx, ref / cnt2);
        } }
    }

    // ------------------------------------------------- Q residual-split numerics
    if (mode == "residual" || mode == "all") {
        printf("\n=== RESIDUAL-SPLIT Q: Q_hi = nvfp4(Q), Q_lo = nvfp4(Q - Q_hi), K fixed nvfp4 (D=%d) ===\n", D);
        printf("  K cancels: reference = fp32 Q x dequant(nvfp4 K); errors below are pure Q-side.\n");
        const int NT = std::min(ntok, 1024), nb = D / QK_NVFP4;
        std::vector<float> Kf((size_t) NT * D), Kdq((size_t) NT * D);
        std::vector<uint8_t> Kd((size_t) NT * (D / 16)), Kq((size_t) NT * (D / 2));
        uint32_t kst = seed;
        for (auto & v : Kf) v = frand(kst) * 1.5f;
        for (int r = 0; r < NT; ++r) {
            nvfp4_quant_row(&Kf[(size_t) r * D], nb, &Kd[(size_t) r * (D / 16)], &Kq[(size_t) r * (D / 2)]);
            nvfp4_dequant_row(&Kd[(size_t) r * (D / 16)], &Kq[(size_t) r * (D / 2)], nb, &Kdq[(size_t) r * D]);
        }
        auto run_case = [&](const char * label, const std::vector<float> & Q, int NQ) {
            if (NQ <= 0) return;
            std::vector<float> Qh((size_t) NQ * D), Qhl((size_t) NQ * D), R((size_t) NQ * D), lo(D);
            std::vector<uint8_t> d(nb * NSUB), c(D / 2), d2(nb * NSUB), c2(D / 2);
            double sQ = 0, e1 = 0, e2 = 0;
            long nmax = 0, nall = 0;
            for (int r = 0; r < NQ; ++r) {
                const float * q = &Q[(size_t) r * D];
                float * h = &Qh[(size_t) r * D];
                nvfp4_quant_row(q, nb, d.data(), c.data());
                nvfp4_dequant_row(d.data(), c.data(), nb, h);
                for (int i = 0; i < nb * NSUB; ++i) {               // coords rounding to the E2M1 max bin (6)
                    const float dd = ue4m3_hw(d[i]);
                    if (dd == 0.f) continue;
                    for (int j = 0; j < QK_NVFP4_SUB; ++j) {
                        if (fabsf(q[i * QK_NVFP4_SUB + j]) / dd >= 5.0f) ++nmax;
                    }
                }
                nall += D;
                float * hl = &Qhl[(size_t) r * D];
                float * rr = &R[(size_t) r * D];
                for (int x = 0; x < D; ++x) rr[x] = q[x] - h[x];
                nvfp4_quant_row(rr, nb, d2.data(), c2.data());
                nvfp4_dequant_row(d2.data(), c2.data(), nb, lo.data());
                for (int x = 0; x < D; ++x) {
                    const float v1 = q[x] - h[x];
                    const float v2 = q[x] - h[x] - lo[x];
                    hl[x] = h[x] + lo[x];
                    sQ += (double) q[x] * q[x]; e1 += (double) v1 * v1; e2 += (double) v2 * v2;
                }
            }
            std::vector<double> d1v, d2v;
            d1v.reserve((size_t) NQ * NT); d2v.reserve((size_t) NQ * NT);
            double refabs = 0;
            for (int t = 0; t < NT; ++t) for (int r = 0; r < NQ; ++r) {
                double tru = 0, g1 = 0, g2 = 0;
                for (int x = 0; x < D; ++x) {
                    const double k = Kdq[(size_t) t * D + x];
                    tru += (double) Q[(size_t) r * D + x] * k;
                    g1  += (double) Qh[(size_t) r * D + x] * k;
                    g2  += (double) Qhl[(size_t) r * D + x] * k;
                }
                d1v.push_back(fabs(g1 - tru)); d2v.push_back(fabs(g2 - tru));
                refabs += fabs(tru);
            }
            auto stat = [](const std::vector<double> & v, double & rms, double & p99, double & mx) {
                double s = 0; mx = 0;
                for (double x : v) { s += x * x; if (x > mx) mx = x; }
                rms = sqrt(s / (double) v.size());
                std::vector<double> sv = v;
                std::sort(sv.begin(), sv.end());
                p99 = sv[(size_t) (0.99 * (double) (sv.size() - 1))];
            };
            double r1, p1, m1, r2, p2, m2;
            stat(d1v, r1, p1, m1); stat(d2v, r2, p2, m2);
            printf("  [%s] rows = %d, coords coded to E2M1 max bin (|x|/scale >= 5): %.2f%%\n",
                   label, NQ, 100.0 * (double) nmax / (double) nall);
            printf("      Q err rms:  pass1 = %.4g (%.2f%% of rms|Q|)   pass1+2 = %.4g (%.3f%%)   improvement = %.1fx\n",
                   sqrt(e1 / ((double) NQ * D)), 100.0 * sqrt(e1 / sQ),
                   sqrt(e2 / ((double) NQ * D)), 100.0 * sqrt(e2 / sQ), sqrt(e1 / std::max(e2, 1e-30)));
            printf("      score err:  pass1 rms %.4g p99 %.4g max %.4g  |  pass1+2 rms %.4g p99 %.4g max %.4g  (mean|score| %.2f)\n",
                   r1, p1, m1, r2, p2, m2, refabs / (double) d1v.size());
        };
        if (!qfile.empty()) {
            FILE * f = fopen(qfile.c_str(), "rb");
            if (!f) printf("  --qfile: cannot open %s\n", qfile.c_str());
            else {
                fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
                int NQ = (int) (sz / (D * 4));
                if (NQ > 4096) NQ = 4096;
                std::vector<float> Q((size_t) NQ * D);
                if (fread(Q.data(), 4, (size_t) NQ * D, f) == (size_t) NQ * D) run_case("real Q activations", Q, NQ);
                else printf("  --qfile: short read (expected rows of %d fp32)\n", D);
                fclose(f);
            }
        } else {
            std::vector<float> Q((size_t) 256 * D);
            uint32_t st = seed;
            for (auto & v : Q) v = frand(st);
            run_case("gaussian Q", Q, 256);
            st = seed;
            for (auto & v : Q) { v = frand(st); if (((st >> 12u) % 10u) < 1u) v *= 6.0f; }
            run_case("outlier-mix Q (10% x6)", Q, 256);
            st = seed + 7u;
            for (auto & v : Q) { v = frand(st); if (((st >> 12u) % 20u) < 1u) v *= 20.0f; }
            run_case("heavy-mix Q (5% x20)", Q, 256);
        }
        printf("  note: pass2 uses the SAME K bytes and K scales; only the Q operand doubles (hi+lo mma chains).\n");
    }

    // ----------------------------------------------------------- kernelcheck
    if (mode == "kernelcheck") {
        printf("\n=== KERNELCHECK: fattn-vec.cuh hi+lo prep + two-chain mma replicas vs exact host reference (D=%d) ===\n", D);
        fflush(stdout);
        const int NB  = D / QK_NVFP4;      // 64-coord blocks per row
        const int NSB = D / QK_NVFP4_SUB;  // 16-coord sub-blocks per row
        const int FSTR = D / 8 + 1;        // FA smem row stride (fragment words)

        // Q rows: dumped real activations if --qfile, else one synthetic outlier-heavy row set.
        std::vector<float> Qf;
        int nq_use = 0;
        if (!qfile.empty()) {
            FILE * f = fopen(qfile.c_str(), "rb");
            if (!f) { printf("  --qfile: cannot open %s\n", qfile.c_str()); return 2; }
            fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
            int NQ = (int) (sz / (D * 4)); if (NQ > 8) NQ = 8;
            Qf.resize((size_t) NQ * D); nq_use = NQ;
            if (!NQ || fread(Qf.data(), 4, (size_t) NQ * D, f) != (size_t) NQ * D) {
                printf("  --qfile: short read (expected rows of %d fp32)\n", D); fclose(f); return 2;
            }
            fclose(f);
        } else {
            Qf.resize(8 * D); nq_use = 8;
            uint32_t st = seed;
            for (auto & v : Qf) { v = frand(st); if (((st >> 13u) % 8u) < 1u) v *= 7.0f; }
        }
        printf("  Q rows fed to the prep replica: %d%s\n", nq_use, qfile.empty() ? " (synthetic)" : " (from --qfile)");
        fflush(stdout);

        // Exact replica of the FA closed-form prep on the host (integer words -> direct compare).
        // NOTE: deliberately NOT nvfp4_quant_row's table search: this must match the kernel codec;
        // any difference there is itself a finding. amax computed in fp64 to flag fp32 scale ties.
        std::vector<int> Qnc((size_t) 8 * FSTR, 0), Qsc(8 * NB, 0), Qnc2((size_t) 8 * FSTR, 0), Qsc2(8 * NB, 0);
        std::vector<float> QhiD(8 * D), QloD(8 * D);
        int n_uelo = 0;
        for (int j = 0; j < 8; ++j) {
            for (int s = 0; s < NSB; ++s) {
                const int f = s >> 2, b = s & 3;
                float v[QK_NVFP4_SUB];
                float amax = 0.0f; double amax64 = 0.0;
                for (int c = 0; c < QK_NVFP4_SUB; ++c) {
                    v[c] = (j < nq_use) ? Qf[(size_t) j * D + s * QK_NVFP4_SUB + c] : 0.0f;
                    amax = fmaxf(amax, fabsf(v[c]));
                    amax64 = (amax64 > fabs((double) v[c])) ? amax64 : fabs((double) v[c]);
                }
                const uint8_t ue = ue4m3_from_f32(amax / 6.0f);
                if (fabs(ue4m3_hw(ue) - ue4m3_exact(ue)) > 1e-6 * ue4m3_exact(ue)) ++n_uelo;
                const float dd = 2.0f * ue4m3_hw(ue);
                uint32_t cd[QK_NVFP4_SUB]; uint32_t N0 = 0, N1 = 0;
                for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                    cd[k] = e2m1_code_host(v[k], dd);
                    QhiD[(size_t) j * D + s * QK_NVFP4_SUB + k] = (float) e2m1_value((int) cd[k]) * dd;
                }
                for (int bb = 0; bb < 4; ++bb) {
                    N0 |= (cd[2*bb] | (cd[2*bb + 1] << 4)) << (8*bb);
                    N1 |= (cd[2*bb + 8] | (cd[2*bb + 9] << 4)) << (8*bb);
                }
                Qnc[(size_t) j * FSTR + 2*s]     = (int) N0;
                Qnc[(size_t) j * FSTR + 2*s + 1] = (int) N1;
                Qsc[(size_t) j * NB + f] |= (int) ue << (8*b);

                float rv[QK_NVFP4_SUB]; float am_r = 0.0f;
                for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                    rv[k] = v[k] - dd * e2m1_value((int) cd[k]);
                    am_r = fmaxf(am_r, fabsf(rv[k]));
                }
                const uint8_t ue2 = ue4m3_from_f32(am_r / 6.0f);
                const float dd2 = 2.0f * ue4m3_hw(ue2);
                uint32_t M0 = 0, M1 = 0;
                for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                    const uint32_t cl = e2m1_code_host(rv[k], dd2);
                    QloD[(size_t) j * D + s * QK_NVFP4_SUB + k] = (float) e2m1_value((int) cl) * dd2;
                }
                for (int bb = 0; bb < 4; ++bb) {
                    M0 |= (e2m1_code_host(rv[2*bb], dd2) | (e2m1_code_host(rv[2*bb + 1], dd2) << 4)) << (8*bb);
                    M1 |= (e2m1_code_host(rv[2*bb + 8], dd2) | (e2m1_code_host(rv[2*bb + 9], dd2) << 4)) << (8*bb);
                }
                Qnc2[(size_t) j * FSTR + 2*s]     = (int) M0;
                Qnc2[(size_t) j * FSTR + 2*s + 1] = (int) M1;
                Qsc2[(size_t) j * NB + f] |= (int) ue2 << (8*b);
            }
        }
        if (n_uelo) printf("  NOTE: %d hi scales differ >1e-6 rel between exact-fp64 and fp32 amax paths (borderline encodings)\n", n_uelo);

        auto cmp_words = [&](const char * name, const int * hbuf, const int *dbuf, int n) {
            size_t mism = 0; int shown = 0;
            for (int i = 0; i < n; ++i) {
                if (hbuf[i] != dbuf[i]) {
                    ++mism;
                    if (shown < 4) {
                        printf("    %-7s idx %4d: host %08X  replica %08X\n", name, i,
                               (unsigned) hbuf[i], (unsigned) dbuf[i]);
                        ++shown;
                    }
                }
            }
            printf("  %-9s word mismatches: %zu / %d %s\n", name, mism, n, mism ? "** MISMATCH **" : "OK (bit-exact)");
            fflush(stdout);
            return mism;
        };

        int * dqnc, * dqsc, * dqnc2, * dqsc2;
        float * d_Q;
        ck(cudaMalloc(&d_Q,   8 * D * 4),  "q");
        ck(cudaMemcpy(d_Q, Qf.data(), 8 * D * 4, cudaMemcpyHostToDevice), "q copy");
        ck(cudaMalloc(&dqnc,  8 * FSTR * 4), "qnc");
        ck(cudaMalloc(&dqsc,  8 * NB * 4),   "qsc");
        ck(cudaMalloc(&dqnc2, 8 * FSTR * 4), "qnc2");
        ck(cudaMalloc(&dqsc2, 8 * NB * 4),   "qsc2");
        k_fa_prep_replica<<<1, 128>>>(d_Q, dqnc, dqsc, dqnc2, dqsc2, D);
        ck(cudaGetLastError(), "prep launch");
        ck(cudaDeviceSynchronize(), "prep sync");
        std::vector<int> gQnc(8*FSTR), gQsc(8*NB), gQnc2(8*FSTR), gQsc2(8*NB);
        ck(cudaMemcpy(gQnc.data(),  dqnc,  8*FSTR*4, cudaMemcpyDeviceToHost), "dqnc");
        ck(cudaMemcpy(gQsc.data(),  dqsc,  8*NB*4,   cudaMemcpyDeviceToHost), "dqsc");
        ck(cudaMemcpy(gQnc2.data(), dqnc2, 8*FSTR*4, cudaMemcpyDeviceToHost), "dqnc2");
        ck(cudaMemcpy(gQsc2.data(), dqsc2, 8*NB*4,   cudaMemcpyDeviceToHost), "dqsc2");
        const size_t mm = cmp_words("Qnc",  Qnc.data(),  gQnc.data(),  8*FSTR)
                        + cmp_words("Qsc",  Qsc.data(),  gQsc.data(),  8*NB)
                        + cmp_words("Qnc2", Qnc2.data(), gQnc2.data(), 8*FSTR)
                        + cmp_words("Qsc2", Qsc2.data(), gQsc2.data(), 8*NB);

        // K: 32 tokens as ggml block_nvfp4 bytes (d[4] LE + qs[32]) - same generator as residual mode.
        std::vector<float> Kf(32 * D), Kdq(32 * D);
        std::vector<uint8_t> Kdb(32 * NB * 4), Kqb(32 * NB * 32);
        uint32_t kst = seed;
        for (auto & v : Kf) v = frand(kst) * 1.5f;
        for (int r = 0; r < 32; ++r) {
            nvfp4_quant_row(&Kf[(size_t) r * D], NB, &Kdb[(size_t) r * NB * 4], &Kqb[(size_t) r * NB * 32]);
            nvfp4_dequant_row(&Kdb[(size_t) r * NB * 4], &Kqb[(size_t) r * NB * 32], NB, &Kdq[(size_t) r * D]);
        }
        std::vector<uint8_t> Kb(32 * NB * 36);
        for (int r = 0; r < 32; ++r)
            for (int f = 0; f < NB; ++f) {
                memcpy(&Kb[(size_t) r * NB * 36 + (size_t) f * 36],     &Kdb[(size_t) r * NB * 4 + 4*f], 4);
                memcpy(&Kb[(size_t) r * NB * 36 + (size_t) f * 36 + 4], &Kqb[(size_t) r * NB * 32 + 32*f], 32);
            }

        uint8_t * d_Kb;
        ck(cudaMalloc(&d_Kb, Kb.size()), "kb");
        ck(cudaMemcpy(d_Kb, Kb.data(), Kb.size(), cudaMemcpyHostToDevice), "kb copy");
        float * d_out; ck(cudaMalloc(&d_out, 2 * 32 * 8 * 4), "mma out");
        k_fa_mma_replica<<<1, 32>>>(d_Kb, dqnc, dqsc, dqnc2, dqsc2, d_out, D);
        ck(cudaGetLastError(), "mma launch");
        ck(cudaDeviceSynchronize(), "mma sync");
        std::vector<float> out(2 * 32 * 8);
        ck(cudaMemcpy(out.data(), d_out, 2*32*8*4, cudaMemcpyDeviceToHost), "mma out back");

        // Exact fp64 references on the SAME codes+scale bytes, mapped through the verified
        // D-cell map (row r: lane 4*(r&7)+(j&1)... register 2*((r&15)>>3)+(j&1)).
        std::vector<double> ex1(32 * 8), ex2(32 * 8), tru(32 * 8);
        for (int r = 0; r < 32; ++r) for (int j = 0; j < 8; ++j) {
            double s1 = 0, s2 = 0, st = 0;
            for (int x = 0; x < D; ++x) {
                const double k = Kdq[(size_t) r * D + x];
                s1 += (double) QhiD[(size_t) j * D + x] * k;
                s2 += ((double) QhiD[(size_t) j * D + x] + (double) QloD[(size_t) j * D + x]) * k;
                st += (double) ((j < nq_use) ? Qf[(size_t) j * D + x] : 0.0f) * k;
            }
            const int lane = 4 * (r & 7) + (j >> 1), ri = 2 * ((r & 15) >> 3) + (j & 1);
            for (int mt = 0; mt < 2; ++mt) {
                if ((r >> 4) != mt) continue;
                ex1[(size_t) mt * 256 + lane * 8 + ri]     = s1;
                ex2[(size_t) mt * 256 + lane * 8 + ri]     = s2;
                tru[(size_t) mt * 256 + lane * 8 + ri]     = st;
            }
        }
        auto err2 = [&](const std::vector<double> & ex, int off, double & rms, double & mx) {
            double s = 0; mx = 0;
            for (size_t i = 0; i < ex.size(); ++i) {
                const double d = fabs((double) out[off + i] - ex[i]);
                s += d * d; if (d > mx) mx = d;
            }
            rms = sqrt(s / (double) ex.size());
        };
        double rms_hw1, mx_hw1, rms_hw2, mx_hw2, rms_q1, mx_q1, rms_q2, mx_q2;
        err2(ex1, 0, rms_hw1, mx_hw1);   // replica 2-chain vs same-codes exact (hi+lo)
        err2(ex2, 0, rms_hw2, mx_hw2);
        err2(ex1, 4, rms_q1, mx_q1);     // replica hi-only vs same-codes exact (hi)
        err2(ex2, 4, rms_q2, mx_q2);
        printf("  mma fragment wiring vs exact-same-codes:  2-chain cell err rms %.3g max %.3g | hi-only rms %.3g max %.3g\n",
               rms_hw1, mx_hw1, rms_q1, mx_q1);
        printf("  (tensor-core fp32 accumulate order -> rms ~1e-5 is bit-clean for scores O(10); rms ~0.1+ = wiring bug)\n");

        // Numerics of the replica path against the true fp32 Q (the FA-context-independent codec error).
        std::vector<double> dq1(32 * 8), dq2(32 * 8);
        double refabs = 0;
        for (size_t i = 0; i < dq1.size(); ++i) {
            dq1[i] = fabs(ex1[i] - tru[i]);
            dq2[i] = fabs(ex2[i] - tru[i]);
            refabs += fabs(tru[i]);
        }
        auto stat2 = [](const std::vector<double> & v, double & rms, double & mx) {
            double s = 0; mx = 0;
            for (double x : v) { s += x * x; if (x > mx) mx = x; }
            rms = sqrt(s / (double) v.size());
        };
        double rq1, mq1, rq2, mq2;
        stat2(dq1, rq1, mq1); stat2(dq2, rq2, mq2);
        printf("  replica scores vs fp32-Q x nvfp4-K:  hi-only rms %.4g max %.4g | hi+lo rms %.4g max %.4g (mean|score| %.2f)\n",
               rq1, mq1, rq2, mq2, refabs / (double) dq1.size());
        printf("  expected from residual study: hi-only ~1.14 rms, hi+lo ~0.11 rms\n");
        printf("  verdict: %s\n", (mm == 0 && rms_hw1 < 1e-3 && rms_q1 < 1e-3)
               ? "kernel logic replicates bit-exactly -> defect must be in the live FA kernel context"
               : "replica itself diverges -> see the mismatch lines above (prep words vs scores)");
        fflush(stdout);

        cudaFree(d_Q); cudaFree(d_Kb);
        cudaFree(dqnc); cudaFree(dqsc); cudaFree(dqnc2); cudaFree(dqsc2); cudaFree(d_out);
    }

    // ------------------------------------------------------------- throughput
    if (mode == "bench" || mode == "all") {
        printf("\n=== TENSOR CORE ISSUE RATE (independent accumulators, no memory) ===\n");
        const int blocks = prop.multiProcessorCount * 4;
        const int thr = 256;
        float * d_out; cudaMalloc(&d_out, 128 * 4);
        struct { const char * name; int mode; double flop_per_mma; } tm[] = {
            { "fp4  nvf4 4X m16n8k64",   0, 16.0 * 8 * 64 * 2 },
            { "fp16 m16n8k16",           1, 16.0 * 8 * 16 * 2 },
            { "fp8  dense m16n8k32",     2, 16.0 * 8 * 32 * 2 },
            { "fp8xfp4 mxf8f6f4 k32",    3, 16.0 * 8 * 32 * 2 },
        };
        for (auto & t : tm) {
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            auto run = [&](int m) {
                if (m == 0) k_throughput<0><<<blocks, thr>>>(d_out, iters);
                else if (m == 1) k_throughput<1><<<blocks, thr>>>(d_out, iters);
                else if (m == 2) k_throughput<2><<<blocks, thr>>>(d_out, iters);
                else k_throughput<3><<<blocks, thr>>>(d_out, iters);
            };
            run(t.mode); cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) run(t.mode);
            cudaEventRecord(e1);
            ck(cudaGetLastError(), "launch throughput");
            ck(cudaDeviceSynchronize(), "sync throughput");
            float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
            const double mmas = (double) blocks * (thr / 32.0) * 8.0 * iters;
            printf("%-24s %8.3f ms  %10.2f TFLOP/s  (%.2f G-mma/s)\n",
                   t.name, ms, mmas * t.flop_per_mma / (ms * 1e-3) / 1e12, mmas / (ms * 1e-3) / 1e9);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        cudaFree(d_out);
    }

    // --------------------------------------------------------------- QK bench
    if (mode == "bench" || mode == "all") {
        printf("\n=== QK^T TILE BENCH  D=%d  ntok=%d  nq=%d ===\n", D, ntok, nq);
        const int nb = D / QK_NVFP4;              // blocks per row
        std::vector<float> Qf(nq * D), Kf((size_t) ntok * D);
        uint32_t st = seed;
        for (auto & v : Qf) v = frand(st);
        for (auto & v : Kf) v = frand(st) * 1.5f;

        std::vector<uint8_t> Qd(nq * (D / 16)), Qq(nq * (D / 2)), Kd((size_t) ntok * (D / 16)), Kq((size_t) ntok * (D / 2));
        for (int r = 0; r < nq; ++r) nvfp4_quant_row(&Qf[(size_t) r * D], nb, &Qd[(size_t) r * (D / 16)], &Qq[(size_t) r * (D / 2)]);
        for (int r = 0; r < ntok; ++r) nvfp4_quant_row(&Kf[(size_t) r * D], nb, &Kd[(size_t) r * (D / 16)], &Kq[(size_t) r * (D / 2)]);

        // word/scale views for the kernels (4 scale words = 16 bytes per row of D coords)
        // row stride D/4 words; real code words are the first D/8 (D/2 bytes), the rest is padding
        std::vector<int> KqW((size_t) ntok * (D / 4)), KsW((size_t) ntok * (D / 64));
        for (size_t r = 0; r < (size_t) ntok; ++r)
            for (int w = 0; w < D / 4; ++w)
                KqW[r * (D / 4) + w] = w < D / 8 ? (int) pack4(&Kq[r * (D / 2) + w * 4]) : 0;
        for (size_t r = 0; r < (size_t) ntok; ++r)
            for (int s = 0; s < D / 64; ++s) KsW[r * (D / 64) + s] = (int) pack4(&Kd[r * (D / 16) + s * 4]);

        std::vector<int> QqW(nq * (D / 4)), QsW(nq * (D / 64));
        for (int r = 0; r < nq; ++r) {
            for (int w = 0; w < D / 4; ++w)
                QqW[r * (D / 4) + w] = w < D / 8 ? (int) pack4(&Qq[r * (D / 2) + w * 4]) : 0;
            for (int s = 0; s < D / 64; ++s) QsW[r * (D / 64) + s] = (int) pack4(&Qd[r * (D / 16) + s * 4]);
        }

        // fragment-native (consecutive-pair) word order for the MMA path
        std::vector<int> KqWN((size_t) ntok * (D / 4)), QqWN(nq * (D / 4));
        for (size_t r = 0; r < (size_t) ntok; ++r) ggml_to_native_words(Kq.data(), (int) r, nb, KqWN.data(), D / 4);
        for (int r = 0; r < nq; ++r) ggml_to_native_words(Qq.data(), r, nb, QqWN.data(), D / 4);

        // q8_1 quantized Q in the FA kernel's own convention (32-elem blocks, ds = (d, sum_x))
        std::vector<int> Qq8;
        std::vector<float2> Qds;
        Qq8.resize(nq * (D / 4));
        Qds.resize(nq * (D / 4));
        for (int r = 0; r < nq; ++r) {
            std::vector<int> q8; std::vector<float2> ds;
            quantize_q8_1_fa(&Qf[(size_t) r * D], D, q8, ds);
            const int nw8 = (int) q8.size();
            for (int w = 0; w < D / 4; ++w) {
                Qq8[r * (D / 4) + w] = w < nw8 ? q8[w] : 0;
                Qds[r * (D / 4) + w] = w < nw8 ? ds[w] : make_float2(0.f, 0.f);
            }
        }

        int * d_KqW, * d_KsW, * d_QqW, * d_QsW, * d_Qq8, * d_KqWN, * d_QqWN; float2 * d_Qds; float * d_out, * d_out2;
        ck(cudaMalloc(&d_KqW, KqW.size() * 4), "m KqW");
        ck(cudaMalloc(&d_KsW, KsW.size() * 4), "m KsW");
        ck(cudaMalloc(&d_QqW, QqW.size() * 4), "m QqW");
        ck(cudaMalloc(&d_QsW, QsW.size() * 4), "m QsW");
        ck(cudaMalloc(&d_KqWN, KqWN.size() * 4), "m KqWN");
        ck(cudaMalloc(&d_QqWN, QqWN.size() * 4), "m QqWN");
        ck(cudaMalloc(&d_Qq8, Qq8.size() * 4), "m Qq8");
        ck(cudaMalloc(&d_Qds, Qds.size() * 8), "m Qds");
        ck(cudaMalloc(&d_out, (size_t) ntok * nq * 4), "m out");
        ck(cudaMalloc(&d_out2, (size_t) ntok * nq * 4), "m out2");
        cudaMemcpy(d_KqW, KqW.data(), KqW.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_KsW, KsW.data(), KsW.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_QqW, QqW.data(), QqW.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_QsW, QsW.data(), QsW.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_KqWN, KqWN.data(), KqWN.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_QqWN, QqWN.data(), QqWN.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Qq8, Qq8.data(), Qq8.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Qds, Qds.data(), Qds.size() * 8, cudaMemcpyHostToDevice);

        const double flop = 2.0 * (double) nq * (double) ntok * (double) D;
        const double kvbytes = (double) ntok * (D / 2 + D / 16);

        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        // PathVEC
        auto run_vec = [&]() {
            int nblk = ntok;                       // one warp (32 thr) per block, one K token per warp
            k_path_vec<<<nblk, 32>>>(d_KqW, d_KsW /* scales as words */, d_Qq8, d_Qds, d_out, ntok, nq, D);
        };
        run_vec(); ck(cudaGetLastError(), "launch vec"); cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) run_vec();
        cudaEventRecord(e1); ck(cudaDeviceSynchronize(), "sync vec");
        float ms_vec; cudaEventElapsedTime(&ms_vec, e0, e1); ms_vec /= reps;

        // PathMMA
        if (nq % 16 != 0) {
            printf("PathMMA skipped: nq must be a multiple of 16 (got %d)\n", nq);
        } else {
            const size_t shm_bytes = (size_t) (64 * (D / 4 + 1) + 64 * (D / 64) + nq * (D / 4 + 1) + nq * (D / 64)) * 4;
            if (D == 256) {
                ck(cudaFuncSetAttribute(k_path_mma<256, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm_bytes), "smem attr");
            }
            const int nwarps = 8;
            auto run_mma = [&]() {
                const int nblk = ntok / 64;
                if (D == 256) {
                    k_path_mma<256, false><<<nblk, nwarps * 32, shm_bytes>>>(d_KqWN, d_KsW, d_QqWN, d_QsW, d_out2, ntok, nq);
                } else {
                    fprintf(stderr, "mma path only supports D=256 in this build\n");
                }
            };
            run_mma(); ck(cudaGetLastError(), "launch mma"); cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) run_mma();
            cudaEventRecord(e1); ck(cudaDeviceSynchronize(), "sync mma");
            float ms_mma; cudaEventElapsedTime(&ms_mma, e0, e1); ms_mma /= reps;

            // PathMMA-repacked: UNMODIFIED ggml block_nvfp4 words, nibble permute in-kernel
            ck(cudaFuncSetAttribute(k_path_mma<256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm_bytes), "smem attr rp");
            float * d_out3; ck(cudaMalloc(&d_out3, (size_t) ntok * nq * 4), "m out3");
            auto run_rp = [&]() {
                const int nblk = ntok / 64;
                k_path_mma<256, true><<<nblk, nwarps * 32, shm_bytes>>>(d_KqW, d_KsW, d_QqW, d_QsW, d_out3, ntok, nq);
            };
            run_rp(); ck(cudaGetLastError(), "launch repack"); cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) run_rp();
            cudaEventRecord(e1); ck(cudaDeviceSynchronize(), "sync repack");
            float ms_rp; cudaEventElapsedTime(&ms_rp, e0, e1); ms_rp /= reps;

            std::vector<float> h2((size_t) ntok * nq), h3((size_t) ntok * nq);
            cudaMemcpy(h2.data(), d_out2, h2.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h3.data(), d_out3, h3.size() * 4, cudaMemcpyDeviceToHost);
            size_t mism = 0;
            for (size_t i = 0; i < h2.size(); ++i) if (!(h2[i] == h3[i])) ++mism;

            printf("PathVEC (int dot, q8_1 Q): %8.3f ms  %8.2f GFLOP/s  %7.1f GB/s KV\n",
                   ms_vec, flop / (ms_vec * 1e-3) / 1e9, kvbytes / (ms_vec * 1e-3) / 1e9);
            printf("PathMMA (host-repacked):   %8.3f ms  %8.2f GFLOP/s  %7.1f GB/s KV\n",
                   ms_mma, flop / (ms_mma * 1e-3) / 1e9, kvbytes / (ms_mma * 1e-3) / 1e9);
            printf("PathMMA (ggml bytes+repack in-kernel): %8.3f ms  %8.2f GFLOP/s  %7.1f GB/s KV   bit-exact vs host-repacked: %s\n",
                   ms_rp, flop / (ms_rp * 1e-3) / 1e9, kvbytes / (ms_rp * 1e-3) / 1e9,
                   mism == 0 ? "YES (0 mismatch)" : "NO");
            if (mism) printf("  !! repack mismatches: %zu / %zu\n", mism, h2.size());
            cudaFree(d_out3);

            // device Q prep: fp32 -> ggml nvfp4 words on GPU, must match host quantizer bit-for-bit
            float * d_Qf; int *d_QqW2, *d_QsW2; float * d_out4;
            ck(cudaMalloc(&d_Qf, (size_t) nq * D * 4), "m Qf");
            ck(cudaMalloc(&d_QqW2, QqW.size() * 4), "m QqW2");
            ck(cudaMalloc(&d_QsW2, QsW.size() * 4), "m QsW2");
            cudaMemset(d_QqW2, 0, QqW.size() * 4);          // padding words must read as zeros
            ck(cudaMalloc(&d_out4, (size_t) ntok * nq * 4), "m out4");
            cudaMemcpy(d_Qf, Qf.data(), (size_t) nq * D * 4, cudaMemcpyHostToDevice);
            const int Rq = 32 / (D / QK_NVFP4_SUB);
            const int qthr = ((nq + Rq - 1) / Rq) * 32;
            auto run_qprep = [&]() { k_quant_nvfp4<256><<<(qthr + 255) / 256, 256>>>(d_Qf, d_QqW2, d_QsW2, nq); };
            run_qprep(); ck(cudaGetLastError(), "launch qprep"); cudaDeviceSynchronize();
            std::vector<int> hqW(QqW.size()), hqS(QsW.size());
            cudaMemcpy(hqW.data(), d_QqW2, hqW.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hqS.data(), d_QsW2, hqS.size() * 4, cudaMemcpyDeviceToHost);
            size_t mw = 0, ms = 0;
            for (size_t i = 0; i < hqW.size(); ++i) if (hqW[i] != QqW[i]) ++mw;
            for (size_t i = 0; i < hqS.size(); ++i) if (hqS[i] != QsW[i]) ++ms;
            printf("Device Q prep vs host nvfp4_quant_row: %s  (code-word mism %zu/%zu, scale-word mism %zu/%zu)\n",
                   (mw + ms) == 0 ? "BIT-EXACT" : "MISMATCH", mw, hqW.size(), ms, hqS.size());
            if (getenv("FP4DEBUG") && mw) {
                size_t shown = 0;
                for (size_t w = 0; w < hqW.size() && shown < 4; ++w) if (hqW[w] != QqW[w]) {
                    const int row = (int) (w / (D / 4)), pr = (int) (w % (D / 4)), l = pr / 2, j0 = 4 * (pr & 1);
                    printf("  word %zu row %d sub %d byte-window %d..%d\n", w, row, l, j0, j0 + 3);
                    for (int j = j0; j < j0 + 4; ++j)
                        printf("    x[%2d]=%12.6f  host %02X  dev %02X\n", j,
                               Qf[(size_t) row * D + l * QK_NVFP4_SUB + j],
                               (QqW[w] >> (8 * (j - j0))) & 0xFF, (hqW[w] >> (8 * (j - j0))) & 0xFF);
                    ++shown;
                }
            }

            cudaEventRecord(e0);
            for (int r = 0; r < reps * 20; ++r) run_qprep();
            cudaEventRecord(e1); ck(cudaDeviceSynchronize(), "sync qprep");
            float ms_qp; cudaEventElapsedTime(&ms_qp, e0, e1); ms_qp /= reps * 20;
            printf("Device Q prep cost: %8.4f ms (%d rows x %d coords)\n", ms_qp, nq, D);

            k_path_mma<256, true><<<ntok / 64, nwarps * 32, shm_bytes>>>(d_KqW, d_KsW, d_QqW2, d_QsW2, d_out4, ntok, nq);
            ck(cudaGetLastError(), "launch mma qp"); ck(cudaDeviceSynchronize(), "sync mma qp");
            std::vector<float> h4((size_t) ntok * nq);
            cudaMemcpy(h4.data(), d_out4, h4.size() * 4, cudaMemcpyDeviceToHost);
            size_t mism2 = 0;
            for (size_t i = 0; i < h4.size(); ++i) if (!(h2[i] == h4[i])) ++mism2;
            printf("E2E fp32 Q -> device prep -> repack MMA == host reference: %s\n",
                   mism2 == 0 ? "BIT-EXACT" : "MISMATCH");

            // PathFUSED: f16 Q (lossless fp32->f16 only) x NVFP4 K decoded to f16 fragments in
            // smem, dense m16n8k16 f32-accum mma.  No f16 KV in global memory.
            {
                const int FNW = 4;
                int dev_id = 0, smem_max = 0;
                cudaGetDevice(&dev_id);
                cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);

                // host f16 copy of the QUERY matrix only (not KV): for the QGMEM=true variant
                std::vector<uint32_t> Q16w((size_t) nq * (D / 2));
                for (int rr = 0; rr < nq; ++rr) for (int w = 0; w < D / 2; ++w) {
                    half2 h = __floats2half2_rn(Qf[(size_t) rr * D + 2 * w], Qf[(size_t) rr * D + 2 * w + 1]);
                    Q16w[rr * (D / 2) + w] = *reinterpret_cast<uint32_t *>(&h);
                }
                uint32_t * d_Q16 = nullptr;
                ck(cudaMalloc(&d_Q16, Q16w.size() * 4), "m Q16");
                ck(cudaMemcpy(d_Q16, Q16w.data(), Q16w.size() * 4, cudaMemcpyHostToDevice), "cpy Q16");

                float * d_outF = nullptr, * d_outD = nullptr;
                ck(cudaMalloc(&d_outF, (size_t) ntok * nq * 4), "m outF");
                ck(cudaMalloc(&d_outD, (size_t) ntok * 4), "m outD");

                const size_t shm_a = (size_t) (FNW * 32 + nq) * FUSED_HSTR * 4;   // TPR=32, Q smem
                const size_t shm_b = (size_t) (FNW * 16 + nq) * FUSED_HSTR * 4;   // TPR=16, Q smem
                const size_t shm_c = (size_t) (FNW * 16) * FUSED_HSTR * 4;        // TPR=16, Q gmem
                struct { int tpr; bool qg; size_t shm; const char * name; } vars[] = {
                    { 32, false, shm_a, "TPR=32 Qsmem" },
                    { 16, false, shm_b, "TPR=16 Qsmem" },
                    { 16, true,  shm_c, "TPR=16 Qgmem" },
                };
                for (auto & v : vars) {
                    if (ntok % (FNW * v.tpr) != 0 || (int) v.shm > smem_max) {
                        printf("  %s skipped: ntok %d, smem %zu B vs optin %d B\n", v.name, ntok, v.shm, smem_max);
                        continue;
                    }
                    if (v.tpr == 32) {
                        ck(cudaFuncSetAttribute(k_path_fused<0, 32, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) v.shm), "sa0");
                        ck(cudaFuncSetAttribute(k_path_fused<1, 32, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) v.shm), "sa1");
                    } else if (v.qg) {
                        ck(cudaFuncSetAttribute(k_path_fused<0, 16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) v.shm), "sc0");
                        ck(cudaFuncSetAttribute(k_path_fused<1, 16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) v.shm), "sc1");
                    } else {
                        ck(cudaFuncSetAttribute(k_path_fused<0, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) v.shm), "sb0");
                        ck(cudaFuncSetAttribute(k_path_fused<1, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) v.shm), "sb1");
                    }
                    auto launch = [&](int mv) {
                        const int grid = ntok / (FNW * v.tpr);
                        float * o = (mv == 1) ? d_outD : d_outF;
                        if (v.tpr == 32 && !v.qg) {
                            if (mv == 0) k_path_fused<0, 32, false><<<grid, FNW * 32, v.shm>>>(d_KqW, d_KsW, d_Qf, d_Q16, o, ntok, nq);
                            else         k_path_fused<1, 32, false><<<grid, FNW * 32, v.shm>>>(d_KqW, d_KsW, d_Qf, d_Q16, o, ntok, nq);
                        } else if (!v.qg) {
                            if (mv == 0) k_path_fused<0, 16, false><<<grid, FNW * 32, v.shm>>>(d_KqW, d_KsW, d_Qf, d_Q16, o, ntok, nq);
                            else         k_path_fused<1, 16, false><<<grid, FNW * 32, v.shm>>>(d_KqW, d_KsW, d_Qf, d_Q16, o, ntok, nq);
                        } else {
                            if (mv == 0) k_path_fused<0, 16, true><<<grid, FNW * 32, v.shm>>>(d_KqW, d_KsW, d_Qf, d_Q16, o, ntok, nq);
                            else         k_path_fused<1, 16, true><<<grid, FNW * 32, v.shm>>>(d_KqW, d_KsW, d_Qf, d_Q16, o, ntok, nq);
                        }
                    };
                    auto time_it = [&](int mv, float * ms) {
                        launch(mv); ck(cudaGetLastError(), "launch fused"); ck(cudaDeviceSynchronize(), "sync fused");
                        cudaEventRecord(e0);
                        for (int rr = 0; rr < reps; ++rr) launch(mv);
                        cudaEventRecord(e1); ck(cudaDeviceSynchronize(), "sync fused t");
                        cudaEventElapsedTime(ms, e0, e1); *ms /= reps;
                    };
                    float ms_f = 0, ms_d = 0;
                    time_it(0, &ms_f);
                    time_it(1, &ms_d);
                    printf("  FUSED %s: full %8.3f ms %8.2f GF/s | decode-only %8.3f ms | mma+frag %8.3f ms (%.0f%% full, %.0f%% of decode+mma)\n",
                           v.name, ms_f, flop / (ms_f * 1e-3) / 1e9, ms_d, ms_f - ms_d,
                           100.0 * (double) (ms_f - ms_d) / (double) ms_f,
                           100.0 * (double) (ms_f - ms_d) / (double) (ms_f + ms_d));
                    if (&v == &vars[0]) {   // numerics on the first variant
                        std::vector<float> hF((size_t) ntok * nq);
                        cudaMemcpy(hF.data(), d_outF, hF.size() * 4, cudaMemcpyDeviceToHost);
                        std::vector<float> KdqF((size_t) ntok * D);
                        for (int rr = 0; rr < ntok; ++rr)
                            nvfp4_dequant_row(&Kd[(size_t) rr * (D / 16)], &Kq[(size_t) rr * (D / 2)], nb, &KdqF[(size_t) rr * D]);
                        std::vector<double> Q16((size_t) nq * D);
                        for (size_t i = 0; i < Q16.size(); ++i)
                            Q16[i] = (double) __half2float(__float2half_rn(Qf[i]));
                        double ef = 0, mxf = 0, refabsF = 0; size_t cntF = 0;
                        double e16 = 0, mx16 = 0;           // f16 Q rounding only (exact K)
                        double et = 0, mxt = 0, refabsT = 0; // fused vs fully unquantized fp32 QK
                        double ek = 0;                       // K nvfp4 quantization contribution only
                        const int nchkF = std::min(ntok, 512);
                        for (int t = 0; t < nchkF; ++t) for (int rr = 0; rr < nq; ++rr) {
                            double ref64 = 0, refu = 0, refk = 0;
                            for (int c = 0; c < D; ++c) {
                                ref64 += Q16[(size_t) rr * D + c] * (double) KdqF[(size_t) t * D + c];
                                refu  += (double) Qf[(size_t) rr * D + c] * (double) Kf[(size_t) t * D + c];
                                refk  += (double) Qf[(size_t) rr * D + c] * (double) KdqF[(size_t) t * D + c];
                            }
                            const double got = hF[(size_t) t * nq + rr];
                            ef += (got - ref64) * (got - ref64);
                            mxf = fmax(mxf, fabs(got - ref64));
                            e16 += (ref64 - refk) * (ref64 - refk);   // f16 Q rounding only (same quantized K)
                            mx16 = fmax(mx16, fabs(ref64 - refk));
                            et += (got - refu) * (got - refu);
                            mxt = fmax(mxt, fabs(got - refu));
                            ek += (refk - refu) * (refk - refu);      // K nvfp4 quantization only
                            refabsF += fabs(ref64); refabsT += fabs(refu); ++cntF;
                        }
                        printf("  fused vs fp64 ref (f16-rounded Q x exact-decoded NVFP4 K), n=%d mean|score|=%.2f:  rms = %.4g  max = %.4g\n",
                               (int) cntF, refabsF / cntF, sqrt(ef / cntF), mxf);
                        printf("  fused vs UNQUANTIZED fp32 Q x raw K (mean|score|=%.2f):  rms = %.4g  max = %.4g\n",
                               refabsT / cntF, sqrt(et / cntF), mxt);
                        printf("    decomposition: K nvfp4 alone rms=%.4g   Q f16-rounding alone rms=%.4g  max=%.4g  (mma path adds nothing)\n",
                               sqrt(ek / cntF), sqrt(e16 / cntF), mx16);
                    }
                }
                cudaFree(d_outF); cudaFree(d_outD); cudaFree(d_Q16);
            }
            cudaFree(d_Qf); cudaFree(d_QqW2); cudaFree(d_QsW2); cudaFree(d_out4);
        }

        // lane-level debug: GPU partials vs host for (tok 4, q 0)
        if (getenv("FP4DEBUG")) {
            float * d_part; cudaMalloc(&d_part, 32 * 4);
            k_path_vec_dbg<<<1, 32>>>(d_KqW, d_KsW, d_Qq8, d_Qds, d_part, 4, 0, D);
            ck(cudaGetLastError(), "dbg launch"); ck(cudaDeviceSynchronize(), "dbg sync");
            float dpart[32];
            cudaMemcpy(dpart, d_part, 128, cudaMemcpyDeviceToHost);
            int nlanebad = 0;
            for (int L = 0; L < 32; ++L) {
                double s = 0.0;
                for (int wi = 0; wi < 2; ++wi) {
                    const int i = L * 2 + wi, ib = i / 16, iw = i % 16, sub = iw / 4;
                    const double dsub = 0.5 * (double) ue4m3_hw(Kd[4 * (D / 16) + ib * NSUB + sub]);
                    const uint8_t * qsb = &Kq[4 * (size_t) 4 * (D / 2) + ib * 32 + sub * 8 + (iw & 1) * 4];
                    const int u = Qq8[i];
                    const int8_t * q8 = (const int8_t *) &u;
                    const float2 qds = Qds[i];
                    double doti = 0.0, sumc = 0.0;
                    for (int t = 0; t < 4; ++t) {
                        const uint8_t byte = qsb[t];
                        const uint8_t nib = (iw & 2) ? (byte >> 4) : (byte & 0xF);
                        const int kv = kvalues_fp4[nib];
                        doti += (double) q8[t] * kv;
                        sumc += kv;
                    }
                    s += dsub * ((double) qds.x * doti + ((double) qds.y / 32.0) * sumc);
                }
                if (fabs(s - (double) dpart[L]) > 0.01 && nlanebad < 8) {
                    printf("  lane %d: gpu %.4f host %.4f  DIFF\n", L, dpart[L], s);
                    ++nlanebad;
                }
            }
            if (!nlanebad) printf("  dbg: all lane partials match for tok4 q0 (total %.4f)\n", dpart[0]);
            cudaFree(d_part);
        }

        // host double-precision simulation of the EXACT PathVEC formula (same q8_1 arrays, same
        auto sim_score = [&](int tok, int q) -> double {
            double s = 0.0;
            const uint8_t * kq8 = &Kq[(size_t) tok * (D / 2)];
            const uint8_t * kd8 = &Kd[(size_t) tok * (D / 16)];
            const int     * qq8 = &Qq8[(size_t) q * (D / 4)];
            const float2  * qds = &Qds[(size_t) q * (D / 4)];
            for (int kw = 0; kw < D / 4; ++kw) {
                const int ib = kw / 16, iw = kw % 16, sub = iw >> 2;
                const double dsub = 0.5 * (double) ue4m3_hw(kd8[ib * NSUB + sub]);
                const uint8_t * qsb = kq8 + ib * 32 + sub * 8 + (iw & 1) * 4;
                const int8_t * q8 = (const int8_t *) &qq8[kw];
                double doti = 0.0, sumc = 0.0;
                for (int t = 0; t < 4; ++t) {
                    const uint8_t byte = qsb[t];
                    const uint8_t nib = (iw & 2) ? (byte >> 4) : (byte & 0xF);
                    const int kv = kvalues_fp4[nib];
                    doti += (double) q8[t] * kv;
                    sumc += kv;
                }
                s += dsub * ((double) qds[kw].x * doti + ((double) qds[kw].y / 32.0) * sumc);
            }
            return s;
        };

        // numerical comparison of the two paths against an exact reference
        std::vector<float> gvec((size_t) ntok * nq), gmma((size_t) ntok * nq);
        cudaMemcpy(gvec.data(), d_out, gvec.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gmma.data(), d_out2, gmma.size() * 4, cudaMemcpyDeviceToHost);

        std::vector<float> Kdq((size_t) ntok * D), Qdq((size_t) nq * D);
        for (int r = 0; r < ntok; ++r) nvfp4_dequant_row(&Kd[(size_t) r * (D / 16)], &Kq[(size_t) r * (D / 2)], nb, &Kdq[(size_t) r * D]);
        for (int r = 0; r < nq; ++r) nvfp4_dequant_row(&Qd[r * (D / 16)], &Qq[r * (D / 2)], nb, &Qdq[r * D]);

        double e_vec = 0, e_mma = 0, e_mma_ref4 = 0, mx_vec = 0, mx_mma = 0, cnt = 0, sumabs = 0;
        const int nchk = nq * std::min(ntok, 512);
        for (int t = 0; t < std::min(ntok, 512); ++t) {
            for (int r = 0; r < nq; ++r) {
                double ref4 = 0;   // exact over the nvfp4-representable values
                for (int c = 0; c < D; ++c) ref4 += (double) Qdq[r * D + c] * (double) Kdq[(size_t) t * D + c];
                const double gv = gvec[(size_t) t * nq + r];
                const double gm = gmma[(size_t) t * nq + r];
                e_vec += (gv - ref4) * (gv - ref4);
                e_mma += (gm - ref4) * (gm - ref4);
                mx_vec = fmax(mx_vec, fabs(gv - ref4));
                mx_mma = fmax(mx_mma, fabs(gm - ref4));
                sumabs += fabs(ref4); cnt += 1;
            }
        }
        printf("vs exact-over-codec reference (n=%d, mean|score|=%.2f):\n", (int) cnt, sumabs / cnt);
        printf("  PathVEC rms err = %.4g   max err = %.4g\n", sqrt(e_vec / cnt), mx_vec);
        printf("  PathMMA rms err = %.4g   max err = %.4g\n", sqrt(e_mma / cnt), mx_mma);
        {
            double e_sim = 0, mx_sim = 0;
            for (int t = 0; t < std::min(ntok, 512); ++t) {
                for (int r = 0; r < nq; ++r) {
                    const double sm = sim_score(t, r);
                    const double gv = gvec[(size_t) t * nq + r];
                    e_sim += (gv - sm) * (gv - sm);
                    mx_sim = fmax(mx_sim, fabs(gv - sm));
                }
            }
            printf("  PathVEC vs host-sim of its own formula: rms = %.4g  max = %.4g\n", sqrt(e_sim / cnt), mx_sim);
            {
                int nexact = 0, nwrong = 0, wrongtok = 0, wrongq_first = -1, wrongtok_first = -1;
                bool tokbad[2048] = { false };
                for (int t = 0; t < std::min(ntok, 512); ++t) {
                    bool bad = false;
                    for (int r = 0; r < nq; ++r) {
                        const double d = fabs(gvec[(size_t) t * nq + r] - (float) sim_score(t, r));
                        if (d < 0.01) ++nexact; else { ++nwrong; bad = true; if (wrongq_first < 0) wrongq_first = r; }
                    }
                    if (bad) { tokbad[t] = true; ++wrongtok; if (wrongtok_first < 0) wrongtok_first = t; }
                }
                printf("  exact %d wrong %d; tokens with any wrong: %d (first tok %d, first q %d)\n",
                       nexact, nwrong, wrongtok, wrongtok_first, wrongq_first);
                for (int t = 0; t + 8 <= wrongtok_first + 8 && t < std::min(ntok, 512); ++t) {
                    printf("   tok%-3d %s", t, tokbad[t] ? "BAD" : "ok");
                    for (int r = 0; r < std::min(nq, 4); ++r)
                        printf("  [g %8.3f s %8.3f]", gvec[(size_t) t * nq + r], (float) sim_score(t, r));
                    printf("\n");
                }
            }
            if (getenv("FP4DEBUG")) {
                double worst[6] = {0,0,0,0,0,0};
                int widx[6] = {0,0,0,0,0,0};
                for (int t = 0; t < std::min(ntok, 512); ++t) {
                    for (int r = 0; r < nq; ++r) {
                        const double d = fabs(gvec[(size_t) t * nq + r] - (float) sim_score(t, r));
                        for (int s = 0; s < 6; ++s) {
                            if (d > worst[s]) {
                                for (int u = 5; u > s; --u) { worst[u] = worst[u-1]; widx[u] = widx[u-1]; }
                                worst[s] = d; widx[s] = t * 1000 + r;
                                break;
                            }
                        }
                    }
                }
                for (int s = 0; s < 6; ++s)
                    if (worst[s] > 0) printf("  worst: tok%d q%d diff %.4g (gpu %.4g)\n", widx[s] / 1000, widx[s] % 1000, worst[s], gvec[(size_t) widx[s] / 1000 * nq + widx[s] % 1000]);
            }
        }

        // error against the *unquantized* rows (the number that matters for model quality)
        double ev2 = 0, em2 = 0, mx2v = 0, mx2m = 0, refabs = 0;
        for (int t = 0; t < std::min(ntok, 512); ++t) {
            for (int r = 0; r < nq; ++r) {
                double ref = 0;
                for (int c = 0; c < D; ++c) ref += (double) Qf[r * D + c] * (double) Kf[(size_t) t * D + c];
                const double gv = gvec[(size_t) t * nq + r];
                const double gm = gmma[(size_t) t * nq + r];
                ev2 += (gv - ref) * (gv - ref); em2 += (gm - ref) * (gm - ref);
                mx2v = fmax(mx2v, fabs(gv - ref)); mx2m = fmax(mx2m, fabs(gm - ref));
                refabs += fabs(ref);
            }
        }
        printf("vs unquantized fp32 (mean|score|=%.2f):\n", refabs / cnt);
        printf("  PathVEC (q8_1 Q) rms err = %.4g   max err = %.4g\n", sqrt(ev2 / cnt), mx2v);
        printf("  PathMMA (fp4  Q) rms err = %.4g   max err = %.4g\n", sqrt(em2 / cnt), mx2m);

        cudaFree(d_KqW); cudaFree(d_KsW); cudaFree(d_QqW); cudaFree(d_QsW);
        cudaFree(d_KqWN); cudaFree(d_QqWN);
        cudaFree(d_Qq8); cudaFree(d_Qds); cudaFree(d_out); cudaFree(d_out2);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    return 0;
}
