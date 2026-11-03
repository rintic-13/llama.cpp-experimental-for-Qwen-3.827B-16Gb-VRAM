// fa_vnvfp4_decode_diff.cu - bitwise A/B of the two nvfp4 V-side dequantizers used by the FA
// vector kernel: OLD scalar kvalues_fp4-LUT path vs NEW PRMT register path (dequantize_V_nvfp4,
// fattn-common.cuh).  Criterion: the fused/V path must not add ANY error vs the production path.
// The only expected raw-bit differences are signed zero: the LUT maps nibble 8 (E2M1 -0) to +0,
// PRMT keeps the sign -> -0.  +0/-0 are value-equal everywhere in the VKQ half2 accumulation
// chain (verified reasoning; counted separately here).
//
// Build:  nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] tools\fa_vnvfp4_decode_diff.cu -o build...\bin\fa_vnvfp4_decode_diff.exe
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

#define QK_NVFP4      64
#define QK_NVFP4_SUB  16
struct blk { uint8_t d[4]; uint8_t qs[32]; };   // == block_nvfp4

static __constant__ const int8_t kvalues_fp4[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };

static __device__ __forceinline__ float ue4m3_to_fp32_(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF);
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}

// ---------------- OLD (production before this change) ----------------
static __device__ __forceinline__ float kv_fp4_val(const blk * x, const int64_t ib, const int sub, const float d_sub, const int c) {
    const uint8_t byte = x[ib].qs[sub*(QK_NVFP4_SUB/2) + (c & 7)];
    const uint8_t nib  = (byte >> ((c & 8) ? 4 : 0)) & 0xF;
    return (float) kvalues_fp4[nib] * d_sub;
}
template <int ne>
static __device__ __forceinline__ void deq_old_h(const blk * x, const int64_t i0, uint32_t * dst) {
    const int64_t ib = i0 / QK_NVFP4;
    const int     c0 = i0 % QK_NVFP4;
    const int sub = c0 / QK_NVFP4_SUB;
    const float d_sub = ue4m3_to_fp32_(x[ib].d[sub]);
#pragma unroll
    for (int l0 = 0; l0 < ne; l0 += 2) {
        const float v0 = kv_fp4_val(x, ib, sub, d_sub, c0 + l0 + 0);
        const float v1 = kv_fp4_val(x, ib, sub, d_sub, c0 + l0 + 1);
        half2 hv = make_half2(__float2half(v0), __float2half(v1));
        dst[l0/2] = *(uint32_t *) &hv;
    }
}

// ---------------- NEW (exact copy of the edited production code) ----------------
static __device__ __forceinline__ uint32_t e2m1_pair_f16(const uint32_t n0, const uint32_t n1) {
    const uint32_t tl = 0x3E3C3800u, th = 0x46444240u;
    const uint32_t sel = ((n0 & 7u) << 4) | ((n1 & 7u) << 12);
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(tl), "r"(th), "r"(sel));
    return r | ((n0 & 8u) << 12) | ((n1 & 8u) << 28);
}
template <int ne>
static __device__ __forceinline__ void deq_new_h(const blk * x, const int64_t i0, uint32_t * dst) {
    const int64_t ib = i0 / QK_NVFP4;
    const int     c0 = i0 % QK_NVFP4;
    const int sub = c0 / QK_NVFP4_SUB;
    const int   o = c0 % QK_NVFP4_SUB;
    typedef typename std::conditional<(ne >= 4), uint32_t, uint16_t>::type word_t;
    word_t w; memcpy(&w, x[ib].qs + sub*(QK_NVFP4_SUB/2) + (o & 7), sizeof(w));
    if (o >= 8) w >>= 4;
    const half2 sh = __half2half2(__float2half_rn(2.0f * ue4m3_to_fp32_(x[ib].d[sub])));
#pragma unroll
    for (int l0 = 0; l0 < ne; l0 += 2) {
        const uint32_t bits = e2m1_pair_f16((w >> (8*l0)) & 0xFFu, (w >> (8*(l0 + 1))) & 0xFFu);
        half2 hv = __hmul2(*reinterpret_cast<const half2 *>(&bits), sh);
        dst[l0/2] = *(uint32_t *) &hv;
    }
}

// float branch (non-half build path): old returns f32 LUT product, new f16-exact then widened.
template <int ne>
static __device__ __forceinline__ void deq_old_f(const blk * x, const int64_t i0, float * dst) {
    const int64_t ib = i0 / QK_NVFP4; const int c0 = i0 % QK_NVFP4;
    const int sub = c0 / QK_NVFP4_SUB; const float d_sub = ue4m3_to_fp32_(x[ib].d[sub]);
#pragma unroll
    for (int l = 0; l < ne; ++l) dst[l] = kv_fp4_val(x, ib, sub, d_sub, c0 + l);
}
template <int ne>
static __device__ __forceinline__ void deq_new_f(const blk * x, const int64_t i0, float * dst) {
    const int64_t ib = i0 / QK_NVFP4; const int c0 = i0 % QK_NVFP4;
    const int sub = c0 / QK_NVFP4_SUB; const int o = c0 % QK_NVFP4_SUB;
    typedef typename std::conditional<(ne >= 4), uint32_t, uint16_t>::type word_t;
    word_t w; memcpy(&w, x[ib].qs + sub*(QK_NVFP4_SUB/2) + (o & 7), sizeof(w));
    if (o >= 8) w >>= 4;
    const half2 sh = __half2half2(__float2half_rn(2.0f * ue4m3_to_fp32_(x[ib].d[sub])));
#pragma unroll
    for (int l0 = 0; l0 < ne; l0 += 2) {
        const uint32_t bits = e2m1_pair_f16((w >> (8*l0)) & 0xFFu, (w >> (8*(l0 + 1))) & 0xFFu);
        float2 fv = __half22float2(__hmul2(*reinterpret_cast<const half2 *>(&bits), sh));
        ((float2 *) dst)[l0/2] = fv;
    }
}

// classification: 0 = bitwise equal, 1 = signed-zero pair (+0 vs -0), 2 = real mismatch
// The 32-bit word packs two independent f16 lanes; compare each lane separately.
static __device__ __forceinline__ int cmp_u32(const uint32_t a, const uint32_t b) {
    if (a == b) return 0;
    int sz = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t ah = (a >> (16*h)) & 0xFFFFu;
        const uint32_t bh = (b >> (16*h)) & 0xFFFFu;
        if (ah == bh) continue;
        if ((ah & 0x7FFFu) == 0 && (bh & 0x7FFFu) == 0) { sz = 1; continue; } // +0 vs -0
        return 2;
    }
    return sz;
}

__device__ int real_mm = 0;

__global__ void test_rows(const blk * K, const int nrows, int * bad /*[3: exact,signedzero,real]*/) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const blk * x = K + (size_t) row * (256/QK_NVFP4);
    for (int i0 = threadIdx.x; i0 < 256; i0 += 32) {
        if (i0 & 3) continue;                    // kernel visits 4-aligned starts only
        uint32_t o4[2], n4[2];  deq_old_h<4>(x, i0, o4); deq_new_h<4>(x, i0, n4);
        float of[4], nf[4];     deq_old_f<4>(x, i0, of); deq_new_f<4>(x, i0, nf);
        uint32_t o2[1], n2[1];  deq_old_h<2>(x, i0, o2); deq_new_h<2>(x, i0, n2);
        for (int p = 0; p < 2; ++p) {
            const int c = cmp_u32(o4[p], n4[p]);
            if (c == 2) {
                const int r = atomicAdd(&real_mm, 1);
                if (r < 12) {
                    const int ib = i0 / QK_NVFP4, c0 = i0 % QK_NVFP4, sub = c0 / QK_NVFP4_SUB, o = c0 % QK_NVFP4_SUB;
                    half2 ho = *reinterpret_cast<const half2 *>(&o4[p]);
                    half2 hn = *reinterpret_cast<const half2 *>(&n4[p]);
                    printf("MM i0=%d sub=%d o=%d d=%02X qs=%02X %02X | old=%08X(%.9g,%.9g) new=%08X(%.9g,%.9g)\n",
                        i0, sub, o, x[ib].d[sub],
                        x[ib].qs[sub*8 + (o&7)], x[ib].qs[sub*8 + (o&7) + 1],
                        o4[p], __half2float(ho.x), __half2float(ho.y),
                        n4[p], __half2float(hn.x), __half2float(hn.y));
                }
            }
            atomicAdd(&bad[c], 1);
            uint32_t ua, ub; memcpy(&ua, &of[2*p], 4); memcpy(&ub, &nf[2*p], 4);
            const int cf = (ua == ub) ? 0 : (((ua ^ ub) == 0x80000000u && (ua & 0x7FFFFFFFu) == 0) ? 1 : 2);
            atomicAdd(&bad[3 + cf], 1);
        }
        { const int c = cmp_u32(o2[0], n2[0]); atomicAdd(&bad[6 + c], 1); }
    }
}

// exhaustive: every scale byte x every nibble-pair code (byte value) at coord 0 and coord 8
__global__ void test_exhaustive(int * bad /*3 classes*/) {
    const int sc = blockIdx.x;                    // 0..255 scale byte
    blk x; memset(&x, 0, sizeof(x));
    x.d[0] = (uint8_t) sc; x.d[1] = x.d[2] = x.d[3] = (uint8_t) sc;
    for (int code = threadIdx.x; code < 256; code += 32) {   // one byte: low nibble n0, high nibble n1
        x.qs[0] = (uint8_t) code;
        x.qs[8] = (uint8_t) code;                             // same pair at sub 0 coord 8..15 side
        uint32_t o[2], n[2];
        deq_old_h<4>(&x, 0, o);  deq_new_h<4>(&x, 0, n);      // low nibbles of bytes 0..3 (rest 0)
        atomicAdd(&bad[cmp_u32(o[0], n[0])], 1);
        deq_old_h<4>(&x, 8, o);  deq_new_h<4>(&x, 8, n);      // high nibbles of bytes 0..3
        atomicAdd(&bad[cmp_u32(o[0], n[0])], 1);
    }
}

int main() {
    srand(4242);
    const int nrows = 4096;
    std::vector<blk> h((size_t) nrows * (256/QK_NVFP4));
    for (auto & b : h) { for (int i = 0; i < 32; ++i) b.qs[i] = rand() & 0xFF; for (int i = 0; i < 4; ++i) b.d[i] = rand() & 0xFF; }
    blk * Kd; CK(cudaMalloc(&Kd, sizeof(blk) * h.size()));
    CK(cudaMemcpy(Kd, h.data(), sizeof(blk) * h.size(), cudaMemcpyHostToDevice));
    int * bad; CK(cudaMalloc(&bad, 9 * sizeof(int))); CK(cudaMemset(bad, 0, 9 * sizeof(int)));

    test_rows<<<nrows, 32>>>(Kd, nrows, bad);
    test_exhaustive<<<256, 32>>>(bad);
    CK(cudaDeviceSynchronize());
    int hb[9]; CK(cudaMemcpy(hb, bad, sizeof(hb), cudaMemcpyDeviceToHost));
    printf("half ne=4 : exact %d  signed-zero-equal %d  MISMATCH %d\n", hb[0], hb[1], hb[2]);
    printf("float ne=4: exact %d  signed-zero-equal %d  MISMATCH %d\n", hb[3], hb[4], hb[5]);
    printf("half ne=2 : exact %d  signed-zero-equal %d  MISMATCH %d\n", hb[6], hb[7], hb[8]);
    const bool pass = (hb[2] | hb[5] | hb[8]) == 0 && (hb[1] | hb[4]) > 0;
    printf("%s\n", pass ? "PASS: new decode is bitwise identical (only +0/-0 encoding differs, value-equal)"
                       : ((hb[2]|hb[5]|hb[8]) ? "FAIL: real value mismatches found" : "WARNING: no signed-zero cases exercised"));
    return (hb[2]|hb[5]|hb[8]) ? 1 : 0;
}
