// fa_nvfp4_devdiff.cu - DEVICE-side bitwise isolation of the nvfp4-dot PPL anomaly (9.3682 vs
// 5.5052 at Chunks-4, fa_comp_20261102_015717). Three checks on the CURRENT headers:
//  (A) old kvalues_fp4-LUT vec_dot KQ (verbatim pre-rewrite body) vs current PRMT+dp4a version
//      - per-thread partial sums, bitwise float compare, D=256 (model head), nthreads_KQ=32.
//  (B) current dequantize_V_nvfp4<half,2> vs three references for every (row, even coord i0):
//      B1 consecutive-coord LUT reference   (contract stated in the header comment)
//      B2 (coord c, coord c+8) LUT reference (the packing the current cvt line actually encodes)
//      B3 historical 10f whole-byte PRMT path (probe-validated at the time, fa_vnvfp4_decode_diff)
//  (C) sanity: current V == B3 expected if 10g packing preserved semantics.
// Reports PASS/FAIL + first failing examples. Run standalone (no ggml runtime needed).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#include "ggml.h"
#include "ggml-cuda/fattn-common.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

static const int NROW = 1024;

// ---------------- host reference pieces ----------------
static const int8_t kv_tab[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };

static float ue4m3_f32_host(uint8_t b) {          // same NaN->0 flush as ggml_cuda_ue4m3_to_fp32
    if ((b & 0x7F) == 0x7F) return 0.0f;
    const int s = (b & 0x80) ? -1 : 1;
    const int e = (b >> 3) & 0xF, m = b & 7;
    float v = (e == 0) ? (m / 8.0f) * (1.0f/16.0f) : (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s * v;
}
static int8_t code_at(const block_nvfp4 & blk, int c) {   // nibble code of coord c in a 64-elem block
    const int sub = c / 16, o = c % 16;
    const uint8_t byte = blk.qs[sub*8 + (o & 7)];
    return (int8_t) ((o < 8 ? byte & 0xF : byte >> 4) & 0xF);
}

// ---------------- (A) old dot kernel (verbatim pre-rewrite decode) ----------------
// kvalues_fp4 comes from ggml-common.h (same __constant__ array the pre-§10h dot used).

template <int D, int nthreads>
static __device__ float vec_dot_KQ_nvfp4_OLD(
        const char * __restrict__ K_c, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_nvfp4 * K_nvfp4 = (const block_nvfp4 *) K_c;
    float sum = 0.0f;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int ib  = k_KQ / (QK_NVFP4/4);
        const int iw  = k_KQ % (QK_NVFP4/4);
        const int sub = iw >> 2;
        const float d_sub = ggml_cuda_ue4m3_to_fp32(K_nvfp4[ib].d[sub]);
        uint32_t b;
        ggml_cuda_memcpy_1<sizeof(b)>(&b, K_nvfp4[ib].qs + sub*(QK_NVFP4_SUB/2) + (iw & 1)*4);
        const int    u    = Q_q8[k_KQ_0/nthreads];
        const int8_t * q8 = (const int8_t *) &u;
        const float2 Q_ds  = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        int kv[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint8_t byte = (uint8_t) (b >> (8*t));
            const uint8_t nib  = (byte >> ((iw & 2) ? 4 : 0)) & 0xF;
            kv[t] = kvalues_fp4[nib];
        }
        const float doti = q8[0]*kv[0] + q8[1]*kv[1] + q8[2]*kv[2] + q8[3]*kv[3];
        const float sumc = kv[0] + kv[1] + kv[2] + kv[3];
        sum += d_sub*(Q_ds.x*doti + (Q_ds.y/QK8_1)*sumc);
    }
    return sum;
}

template <int D, int NT, bool NEWVER>
__global__ void k_dot(const char * K, const int * Qq, const float2 * Qds, float * out) {
    const int row = blockIdx.x;
    const float r = NEWVER
        ? vec_dot_fattn_vec_KQ_nvfp4<D, NT>(K + (size_t) row * (D/8 + (D/64)*4), nullptr,
                                            Qq + row*(D/4), Qds + row*(D/4), 0)
        : vec_dot_KQ_nvfp4_OLD<D, NT>(K + (size_t) row * (D/8 + (D/64)*4), Qq + row*(D/4), Qds + row*(D/4));
    if (threadIdx.x < NT) out[(size_t) row * NT + threadIdx.x] = r;
}

// ---------------- (B) V dequant kernels ----------------
template <int NE, bool HIST>
__global__ void k_v_new(const char * Vrow, uint32_t * out) {
    const int row = blockIdx.x;
    const int j = threadIdx.x;                 // i0 = 2*j, D=256 rows -> 128 lanes
    half2 tmp;
    if constexpr (HIST) {
        // historical 10f path: whole-byte operands to e2m1_pair_f16, consecutive coords
        const block_nvfp4 * x = (const block_nvfp4 *) (Vrow + (size_t) row * 144);
        const int64_t i0 = 2*j;
        const int ib = i0 / QK_NVFP4, c0 = i0 % QK_NVFP4;
        const int sub = c0 / QK_NVFP4_SUB, o = c0 % QK_NVFP4_SUB;
        uint16_t w;
        ggml_cuda_memcpy_1<sizeof(w)>(&w, x[ib].qs + sub*(QK_NVFP4_SUB/2) + (o & 7));
        if (o >= 8) w >>= 4;
        const half2 sh = __half2half2(__float2half(2.0f * ggml_cuda_ue4m3_to_fp32(x[ib].d[sub])));
        const uint32_t bits0 = ggml_cuda_e2m1_pair_f16(w & 0xFF, (w >> 8) & 0xFF);
        tmp = __hmul2(*reinterpret_cast<const half2 *>(&bits0), sh);
    } else {
        dequantize_V_nvfp4<half, NE>(Vrow + (size_t) row * 144, &tmp, 2*j, 0);
    }
    out[(size_t) row * 128 + j] = (uint32_t) *reinterpret_cast<const uint32_t *>(&tmp);
}

static uint32_t ref_half2(const block_nvfp4 & blk, int c0, int e0, int e1) {
    const int sub = c0 / 16;
    const float d2 = 2.0f * ue4m3_f32_host(blk.d[sub]);   // raw-scale trick: 2*d_sub, exact in f16
    half2 h = __floats2half2_rn((float) kv_tab[e0] * ue4m3_f32_host(blk.d[sub]),
                                (float) kv_tab[e1] * ue4m3_f32_host(blk.d[sub]));
    (void) d2;
    return (uint32_t) *reinterpret_cast<const uint32_t *>(&h);
}

template <int D>
static void run_dot(const char * Kc, const int * Qq, const float2 * Qds) {
    std::vector<float> oldo((size_t)NROW*32), newo((size_t)NROW*32);
    k_dot<D,32,false><<<NROW,32>>>(Kc, Qq, Qds, oldo.data());
    k_dot<D,32,true ><<<NROW,32>>>(Kc, Qq, Qds, newo.data());
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    long diff = 0;
    for (size_t i = 0; i < oldo.size() && diff < 3; ++i) {
        uint32_t a, b2; float fa = oldo[i], fb = newo[i];
        memcpy(&a, &fa, 4); memcpy(&b2, &fb, 4);
        if (a != b2) { printf("  DOT D=%d diff #%zu: old %.9g new %.9g\n", D, i, fa, fb); ++diff; }
    }
    printf("(A) dot old-vs-new D=%d: %s (%ld bitwise diffs of %zu)\n", D,
        diff ? "FAIL" : "PASS", diff, oldo.size());
}

int main() {
    printf("GPU probe: nvfp4 KQ-dot old/new + V-dequant reference matrix\n");
    srand(20261102);
    const int D = 256;                       // model head size; row = 144 B = 4 blocks*36B
    std::vector<uint8_t> Kc((size_t) NROW * (D/8 + (D/64)*4));
    for (auto & x : Kc) x = (uint8_t) rand();
    for (int r = 0; r < NROW; ++r) {
        for (int ib = 0; ib < D/64; ++ib) {
            // exercise all scale classes incl. 0, NaN(0x7F), subnormal
            Kc[(size_t) r*144 + ib*36 + rand()%4] = (uint8_t) (rand() & 0xFF);
        }
    }
    std::vector<int> Qq((size_t) NROW * (D/4));   // each int = 4 q8_1 bytes in [-127,127]
    for (auto & x : Qq) {
        uint32_t r = 0;
        for (int t = 0; t < 4; ++t) r |= (uint32_t) ((rand() % 255) - 127) & 0xFFu << (8*t);
        x = (int) r;
    }
    std::vector<float2> Qds((size_t) NROW * (D/4));
    for (auto & x : Qds) { x.x = (rand()%2000-1000)/1000.0f; x.y = (rand()%2000-1000)/1000.0f; }

    char * dK; int * dQ; float2 * dDs;
    CK(cudaMalloc(&dK, Kc.size())); CK(cudaMemcpy(dK, Kc.data(), Kc.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dQ, Qq.size()*4)); CK(cudaMemcpy(dQ, Qq.data(), Qq.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dDs, Qds.size()*8)); CK(cudaMemcpy(dDs, Qds.data(), Qds.size()*8, cudaMemcpyHostToDevice));
    run_dot<256>(dK, dQ, dDs);   // model head size; one row = 144 B = 4 blocks of 36 B

    // V: 144B rows, 256 coords, 128 half2 slots
    std::vector<uint32_t> vnew((size_t)NROW*128), vhist((size_t)NROW*128);
    k_v_new<2,false><<<NROW,128>>>((const char*)dK, vnew.data());
    k_v_new<2,true ><<<NROW,128>>>((const char*)dK, vhist.data());
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

    long new_vs_hist = 0, new_vs_consec = 0, new_vs_c8 = 0;
    int shown = 0;
    for (int r = 0; r < NROW; ++r) {
        const block_nvfp4 * blk = (const block_nvfp4 *) &Kc[(size_t) r*144];
        for (int j = 0; j < 128; ++j) {
            const int c0 = 2*j;
            const uint32_t got  = vnew[(size_t) r*128 + j];
            const uint32_t hist = vhist[(size_t) r*128 + j];
            const int e0 = code_at(blk[c0/64], c0 % 64);
            const int e1 = code_at(blk[c0/64], c0 % 64 == 15 ? c0 % 64 : (c0 % 64) + 1);
            const int e8 = code_at(blk[c0/64], (c0 % 64) + 8);
            const uint32_t rconsec = ref_half2(blk[c0/64], c0, e0, e1);
            const uint32_t rc8     = ref_half2(blk[c0/64], c0, e0, e8);
            auto eq0 = [](uint32_t x, uint32_t y) {   // treat +0/-0 as equal
                const half2 a = *(const half2*)&x, b = *(const half2*)&y;
                return (a.x == b.x || (__half2float(a.x) == 0.0f && __half2float(b.x) == 0.0f))
                    && (a.y == b.y || (__half2float(a.y) == 0.0f && __half2float(b.y) == 0.0f));
            };
            if (!eq0(got, hist))   ++new_vs_hist;
            if (!eq0(got, rconsec)) ++new_vs_consec;
            if (!eq0(got, rc8))     ++new_vs_c8;
            if ((!eq0(got, rconsec) || !eq0(got, hist)) && shown < 4) {
                printf("  V row %d slot %d (coord %d): prod %08X hist %08X consec-ref %08X c8-ref %08X\n",
                       r, j, c0, got, hist, rconsec, rc8);
                ++shown;
            }
        }
    }
    printf("(B) V current vs historical-PRMT : %s (%ld mismatches / %d)\n",
        new_vs_hist ? "FAIL" : "PASS", new_vs_hist, NROW*128);
    printf("(B) V current vs consecutive-ref : %s (%ld mismatches)\n",
        new_vs_consec ? "FAIL (packing is NOT consecutive-coord!)" : "PASS", new_vs_consec);
    printf("(B) V current vs (c,c+8)-ref     : %s (%ld mismatches)\n",
        new_vs_c8 ? "no match" : "PASS -> current packing encodes (c, c+8) pairs", new_vs_c8);
    return 0;
}
