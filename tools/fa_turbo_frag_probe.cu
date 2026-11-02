// fa_turbo_frag_probe.cu - S2 fragment-mapping unit probe (NVFP4_FA_RESEARCH.md SS12).
// Runs ggml_cuda_fattn_turbo_kq_afrag<type_K>() for every (f, tg) and every fragment slot on a
// synthetic K block pair, and compares each f16 in the fragment against the element reference
// (turboN_dequant_element / _dithered with the intended coord j = 16*(f&7) + 8*e + 2*tg).
// On a bitwise mismatch it scans all 128 coords of the block and reports which coord's value
// the fragment word actually contains - pinpoints any coord<->byte mispairing class (§10j).
// Standalone tiny GPU run (~1 MB), compile:
//   nvcc -O3 -std=c++17 -DGGML_TURBO_DITHER --generate-code=arch=compute_120a,code=[sm_120a]
//        -I ggml\include -I ggml\src tools\fa_turbo_frag_probe.cu -o build...\bin\fa_turbo_frag_probe.exe
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>

#include "ggml.h"
#include "ggml-cuda/fattn-vec.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

// per-type reference value of coord j in block b of a row (block pointer is row-relative;
// seed must be the linear block index used by the kernel for that same block).
static __device__ float ref_val_turbo4(const block_turbo4_0 * row, int b, int j, uint32_t seed) {
    const float norm = __half2float(row[b].norm);
    const float u = turbo_dither(seed, j, TURBO_DITHER_STEP_4BIT);
    return turbo4_dequant_element_u(&row[b], j, norm, u);
}
static __device__ float ref_val_turbo5(const block_turbo5_0 * row, int b, int j, uint32_t seed) {
    GGML_UNUSED(seed);
    const float norm = __half2float(row[b].norm);
    return turbo5_dequant_element(&row[b], j, norm);
}
static __device__ float ref_val_turbo4q(const block_turbo4q_0 * row, int b, int j, uint32_t seed) {
    GGML_UNUSED(seed);
    const float norm      = __half2float(row[b].norm);
    const float qjl_scale = TURBO_QJL_ALPHA * __half2float(row[b].rnorm);
    return turbo4q_dequant_element(&row[b], j, norm, qjl_scale);
}

static __device__ __forceinline__ uint16_t hbits_of(float v) {   // same memcpy pattern as fattn-vec.cuh
    const __half h = __float2half_rn(v);
    uint16_t b; memcpy(&b, &h, 2); return b;
}

// value->coord locator for the mismatch diagnostic (same seed => same dither u as the fragment)
static __device__ int loc_turbo4(const block_turbo4_0 * row, int b, uint16_t hbits, uint32_t seed) {
    for (int j = 0; j < QK_TURBO4; ++j) {
        if (hbits_of(ref_val_turbo4(row, b, j, seed)) == hbits) return j;
    }
    return -1;
}
static __device__ int loc_turbo5(const block_turbo5_0 * row, int b, uint16_t hbits) {
    for (int j = 0; j < QK_TURBO5; ++j) {
        if (hbits_of(ref_val_turbo5(row, b, j, 0)) == hbits) return j;
    }
    return -1;
}
static __device__ int loc_turbo4q(const block_turbo4q_0 * row, int b, uint16_t hbits) {
    for (int j = 0; j < QK_TURBO4Q; ++j) {
        if (hbits_of(ref_val_turbo4q(row, b, j, 0)) == hbits) return j;
    }
    return -1;
}

template <typename T>
static void fill_norms(T * blk, int nblk, float v, float vr) {   // host-side init only
    for (int i = 0; i < nblk; ++i) {
        blk[i].norm = __float2half(v);
        if constexpr (sizeof(T) == 52 || sizeof(T) == 68) blk[i].rnorm = __float2half(vr);
        GGML_UNUSED(vr);
    }
}

template <ggml_type TK, typename TB>
__global__ void frag_k(TB * row0, TB * row1, uint32_t seedbase, const int64_t blk_per_row, int * bad) {
    constexpr int BSIZE = (int) sizeof(TB);
    for (int f = 0; f < 8; ++f) {           // one full 128-coord block per row (D=128 view)
        for (int tg = 0; tg < 4; ++tg) {
            const uint32_t s0 = seedbase + (f >> 3);
            const uint32_t s1 = seedbase + (uint32_t) blk_per_row + (f >> 3);
            uint32_t A[4];
            ggml_cuda_fattn_turbo_kq_afrag<TK>((const char *) row0 + (f >> 3)*BSIZE,
                                               (const char *) row1 + (f >> 3)*BSIZE,
                                               f, tg, s0, s1, A);
            const int cc = f & 7;
            // slot order: A0=r0e0 A1=r1e0 A2=r0e1 A3=r1e1
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int row = slot & 1;
                const int e   = (slot >> 1);
                TB * rb = row ? row1 : row0;
                const uint32_t seed = row ? s1 : s0;
                const int b = f >> 3;   // slot words come from block b (afrag picked block kb=f>>3)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 16*cc + 8*e + 2*tg + h;   // INTENDED coord
                    const uint16_t got = (uint16_t) ((A[slot] >> (16*h)) & 0xFFFF);
                    float rv;
                    if constexpr (TK == GGML_TYPE_TURBO4_0)  rv = ref_val_turbo4 (rb, b, j, seed);
                    if constexpr (TK == GGML_TYPE_TURBO5_0)  rv = ref_val_turbo5 (rb, b, j, seed);
                    if constexpr (TK == GGML_TYPE_TURBO4Q_0) rv = ref_val_turbo4q(rb, b, j, seed);
                    const uint16_t exp = hbits_of(rv);
                    if (got != exp) {
                        int gj;
                        if constexpr (TK == GGML_TYPE_TURBO4_0)  gj = loc_turbo4 (rb, b, got, seed);
                        if constexpr (TK == GGML_TYPE_TURBO5_0)  gj = loc_turbo5 (rb, b, got);
                        if constexpr (TK == GGML_TYPE_TURBO4Q_0) gj = loc_turbo4q(rb, b, got);
                        printf("MISMATCH TK=%d f=%d tg=%d slot=%d half=%d row=%d: expected j=%d (%04x), got value of coord j'=%d (%04x)\n",
                               (int) TK, f, tg, slot, h, row, j, exp, gj, got);
                        ++*bad;
                    }
                }
            }
        }
    }
}

int main() {
    const int64_t blk_per_row = 2;   // any value matching a D=256 row is fine (seeds consistent)

    // synthetic benign blocks (norms moderate, random index bytes)
    block_turbo4_0  h4[2], hd4[2];
    block_turbo5_0  h5[2], hd5[2];
    block_turbo4q_0 hq[2], hdq[2];
    unsigned s = 7;
    auto rnd = [&](uint8_t * p, size_t n) { for (size_t i = 0; i < n; ++i) { s = s*1664525u + 1013904223u; p[i] = (uint8_t)(s >> 16); } };
    rnd(((uint8_t*)h4)+2,  sizeof(h4)-2);  rnd(((uint8_t*)hd4)+2,  sizeof(hd4)-2);
    rnd(((uint8_t*)h5)+2,  sizeof(h5)-2);  rnd(((uint8_t*)hd5)+2,  sizeof(hd5)-2);
    rnd(((uint8_t*)hq)+4,  sizeof(hq)-4);  rnd(((uint8_t*)hdq)+4,  sizeof(hdq)-4);
    fill_norms(h4, 2, 0.7f, 0);  fill_norms(hd4, 2, 0.3f, 0);
    fill_norms(h5, 2, 0.9f, 0);  fill_norms(hd5, 2, 0.4f, 0);
    fill_norms(hq, 2, 1.1f, 0.6f); fill_norms(hdq, 2, 0.2f, 0.9f);

    block_turbo4_0  *d4,  *dd4;  block_turbo5_0  *d5,  *dd5;  block_turbo4q_0 *dq, *ddq;
    CK(cudaMalloc(&d4,  sizeof(h4)));  CK(cudaMemcpy(d4,  h4,  sizeof(h4),  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dd4, sizeof(hd4))); CK(cudaMemcpy(dd4, hd4, sizeof(hd4), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d5,  sizeof(h5)));  CK(cudaMemcpy(d5,  h5,  sizeof(h5),  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dd5, sizeof(hd5))); CK(cudaMemcpy(dd5, hd5, sizeof(hd5), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dq,  sizeof(hq)));  CK(cudaMemcpy(dq,  hq,  sizeof(hq),  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ddq, sizeof(hdq))); CK(cudaMemcpy(ddq, hdq, sizeof(hdq), cudaMemcpyHostToDevice));

    int * bad; CK(cudaMalloc(&bad, 4*sizeof(int))); CK(cudaMemset(bad, 0, 4*sizeof(int)));
    // row1 = second block set of the SAME allocation (two independent blocks per row array)
    frag_k<GGML_TYPE_TURBO4_0, block_turbo4_0> <<<1,1>>>(d4,  d4+1, 12345u, blk_per_row, bad);
    frag_k<GGML_TYPE_TURBO4_0, block_turbo4_0> <<<1,1>>>(dd4, dd4+1, 999u,  blk_per_row, bad+1);
    frag_k<GGML_TYPE_TURBO5_0, block_turbo5_0> <<<1,1>>>(d5,  d5+1, 222u,  blk_per_row, bad+2);
    frag_k<GGML_TYPE_TURBO4Q_0, block_turbo4q_0><<<1,1>>>(dq, dq+1, 333u,  blk_per_row, bad+3);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

    int hb[4]; CK(cudaMemcpy(hb, bad, sizeof(hb), cudaMemcpyDeviceToHost));
    printf("frag mismatch counts: turbo4(a)=%d turbo4(b-norms)=%d turbo5=%d turbo4q=%d (0=OK, expect 0/192 slots each)\n",
           hb[0], hb[1], hb[2], hb[3]);
    return (hb[0] || hb[1] || hb[2] || hb[3]) ? 1 : 0;
}
