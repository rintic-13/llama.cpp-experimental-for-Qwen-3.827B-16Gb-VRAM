// fa_nvfp4_cvt_probe.cu - (1) bitwise A/B: HW cvt.rn.f16x2.e2m1x2 (CUDA __nv_cvt_fp4x2_to_halfraw2)
// vs PRMT software decoder vs old LUT scalar decode for nvfp4 e2m1 nibble pairs (all 256 packings);
// (2) micro-bench: per-pair decode throughput prmt vs hw-cvt vs lut on sm_120a.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_fp4.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

static __device__ __forceinline__ uint32_t prmt_pair(const uint32_t n0, const uint32_t n1) {
    const uint32_t tl = 0x3E3C3800u, th = 0x46444240u;
    const uint32_t sel = ((n0 & 7u) << 4) | ((n1 & 7u) << 12);
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(tl), "r"(th), "r"(sel));
    return r | ((n0 & 8u) << 12) | ((n1 & 8u) << 28);
}

// HW: one byte = two packed e2m1 nibbles (low nibble = elem0, high nibble = elem1) -> packed half2 bits.
// __nv_cvt_fp4x2_to_halfraw2 returns __half2_raw = the packed 32-bit half2 already.
static __device__ __forceinline__ uint32_t cvt_pair(const uint32_t packed_byte) {
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(packed_byte & 0xFFu), __NV_E2M1);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// LUT reference (production-old semantics): e2m1 value as f16 (doubled table * 0.5f is exact).
__constant__ const int8_t kvalues_fp4[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };
static __device__ __forceinline__ uint32_t lut_pair(const uint32_t n0, const uint32_t n1) {
    const half h0 = __float2half((float)kvalues_fp4[n0 & 0xF] * 0.5f);
    const half h1 = __float2half((float)kvalues_fp4[n1 & 0xF] * 0.5f);
    const half2 h2 = make_half2(h0, h1);
    return *reinterpret_cast<const uint32_t *>(&h2);
}

// per-lane-pair classification: 0 exact, 1 signed-zero (+0 vs -0), 2 real difference
static __device__ __forceinline__ int cmp_u32(const uint32_t a, const uint32_t b) {
    if (a == b) return 0;
    int sz = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t ah = (a >> (16*h)) & 0xFFFFu, bh = (b >> (16*h)) & 0xFFFFu;
        if (ah == bh) continue;
        if ((ah & 0x7FFFu) == 0 && (bh & 0x7FFFu) == 0) { sz = 1; continue; }
        return 2;
    }
    return sz;
}

__global__ void test_codes(int * bad) {
    // bad layout: [0..2] cvt-vs-prmt classes, [3..5] cvt-vs-lut classes
    const int byte = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte >= 256) return;
    const uint32_t lo = byte & 0xF, hi = (byte >> 4) & 0xF;
    const uint32_t p = prmt_pair(lo, hi);
    const uint32_t c = cvt_pair((uint32_t)byte);
    const uint32_t l = lut_pair(lo, hi);
    atomicAdd(&bad[cmp_u32(c, p)], 1);
    atomicAdd(&bad[3 + cmp_u32(c, l)], 1);
}

// bench: decode 4 nibble-bytes (8 e2m1 elems) per random word, chained half2 FMAs keep decode live.
template <int MODE> // 0 prmt, 1 cvt, 2 lut
__global__ void bench(const uint32_t * buf, const int nw, half2 * sink, long iters) {
    half2 acc = __float2half2_rn(1.0009765625f);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = (unsigned)tid * 7919u;
    for (long it = 0; it < iters; ++it) {
        const uint32_t w = buf[idx % (unsigned)nw];
        idx = idx * 1664525u + 1013904223u;
        if (MODE == 0) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t byte = (w >> (8*b)) & 0xFFu;
                const uint32_t bits = prmt_pair(byte & 0xF, (byte >> 4) & 0xF);
                acc += __hmul2(*reinterpret_cast<const half2 *>(&bits), acc);
            }
        } else if (MODE == 1) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t bits = cvt_pair((w >> (8*b)) & 0xFFu);
                acc += __hmul2(*reinterpret_cast<const half2 *>(&bits), acc);
            }
        } else {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t byte = (w >> (8*b)) & 0xFFu;
                const uint32_t bits = lut_pair(byte & 0xF, (byte >> 4) & 0xF);
                acc += __hmul2(*reinterpret_cast<const half2 *>(&bits), acc);
            }
        }
    }
    sink[tid] = acc;
}

template <int MODE>
static float run_bench(const uint32_t * buf, int nw, half2 * sink, long iters) {
    bench<MODE><<<512, 256>>>(buf, nw, sink, iters);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    bench<MODE><<<512, 256>>>(buf, nw, sink, iters);
    cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

int main() {
    srand(1234);
    int * bad; CK(cudaMalloc(&bad, 6*sizeof(int))); CK(cudaMemset(bad, 0, 6*sizeof(int)));
    test_codes<<<1, 256>>>(bad);
    CK(cudaDeviceSynchronize());
    int hb[6]; CK(cudaMemcpy(hb, bad, sizeof(hb), cudaMemcpyDeviceToHost));
    printf("cvt-vs-prmt : exact %d  sign-zero %d  REAL %d\n", hb[0], hb[1], hb[2]);
    printf("cvt-vs-lut  : exact %d  sign-zero %d  REAL %d\n", hb[3], hb[4], hb[5]);

    const int nw = 1 << 20;
    std::vector<uint32_t> hw(nw);
    for (auto & x : hw) x = ((uint32_t)rand()<<17) ^ ((uint32_t)rand()<<3) ^ (uint32_t)rand();
    uint32_t * buf; CK(cudaMalloc(&buf, sizeof(uint32_t)*nw));
    CK(cudaMemcpy(buf, hw.data(), sizeof(uint32_t)*nw, cudaMemcpyHostToDevice));
    half2 * sink; CK(cudaMalloc(&sink, 512*256*sizeof(half2)));
    const long iters = 20000;
    // one warmup pass
    run_bench<0>(buf, nw, sink, 100); run_bench<1>(buf, nw, sink, 100); run_bench<2>(buf, nw, sink, 100);
    const float t0 = run_bench<0>(buf, nw, sink, iters);
    const float t1 = run_bench<1>(buf, nw, sink, iters);
    const float t2 = run_bench<2>(buf, nw, sink, iters);
    const double elems = 512.0*256*iters*8;
    printf("bench elems/s: prmt %.3e (%.2f ms)  cvt %.3e (%.2f ms)  lut %.3e (%.2f ms)\n",
        elems/t0, t0, elems/t1, t1, elems/t2, t2);
    printf("cvt/prmt speed: %.2fx   prmt is baseline\n", t0/t1);
    const bool pass = (hb[2] | hb[5]) == 0;
    printf("%s\n", pass ? "PASS: HW cvt is value-identical to PRMT and to LUT (only +0/-0 may differ)"
                        : "FAIL: real value differences between decoders");
    return pass ? 0 : 1;
}
