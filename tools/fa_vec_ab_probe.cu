// fa_vec_ab_probe.cu - times the PRODUCTION flash_attn_ext_vec kernel directly (no ggml graph)
// for the nq=1 / D=256 decode shape (bench tg @ d32768 equivalent), comparing:
//   V1 q4_1 K + q4_1 V, dot          (SPEED ETALON)
//   V2 nvfp4 K + nvfp4 V, dot        (production nvfp4 vec)
//   V3 nvfp4 K + nvfp4 V, f16dq      (nvfp4-fused, current build)
//   V4 nvfp4 K + q4_1  V, f16dq      (cross: K side fused nvfp4, V side q4_1)
//   V5 q4_1  K + nvfp4 V, dot        (cross: K side q4_1, V side nvfp4)
// (V3-V4) and (V1-V5) attribute the gap to the V side; (V4-V1) / (V5-V2) to the K side.
// Launch config replicated exactly from launch_fattn(): occupancy query + parallel_blocks loop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#include "ggml.h"
#include "ggml-cuda/fattn-vec.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

static const int D = 256;
static const int NQ = 1;
static const int NHEAD = 24;      // q heads
static const int NKVH  = 4;       // kv heads
static const int NKV   = 32768;   // cache length (d32768)
static const int NB1   = 144;     // row bytes for both q4_1 and nvfp4 at D=256

template <typename T>
static T * dalloc_random(size_t n, int seed) {
    std::vector<uint8_t> h(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; ++i) { s = s*1664525u + 1013904223u; h[i] = (uint8_t)(s >> 16); }
    T * d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
    return d;
}

// benign q4_1 cache: d=0.25, m=0, random qs nibbles (avoid inf/nan in scales)
static std::vector<uint8_t> make_q41_cache() {
    std::vector<uint8_t> h((size_t)NKVH*NKV*NB1);
    unsigned s = 7;
    for (int blk = 0; blk < NKVH*NKV*(D/32); ++blk) {
        uint8_t * b = h.data() + (size_t)blk*18;
        const __half d = __float2half(0.25f), m = __float2half(0.0f);
        uint16_t dh, mh; memcpy(&dh, &d, 2); memcpy(&mh, &m, 2);
        b[0] = dh & 0xFF; b[1] = dh >> 8; b[2] = mh & 0xFF; b[3] = mh >> 8;
        for (int i = 0; i < 16; ++i) { s = s*1664525u + 1013904223u; b[4+i] = (uint8_t)(s >> 13) & 0x77; }
    }
    return h;
}
// benign nvfp4 cache: d bytes random ue4m3 finite small (<=0x70), qs random nibbles
static std::vector<uint8_t> make_nvfp4_cache() {
    std::vector<uint8_t> h((size_t)NKVH*NKV*NB1);
    unsigned s = 11;
    for (size_t i = 0; i < h.size(); ++i) { s = s*1664525u + 1013904223u; h[i] = (uint8_t)(s >> 13); }
    for (int blk = 0; blk < NKVH*NKV*(D/64); ++blk) {
        uint8_t * b = h.data() + (size_t)blk*36;
        for (int j = 0; j < 4; ++j) b[j] = (uint8_t)((b[j] & 0x77) % 0x60);   // keep ue4m3 finite, moderate
    }
    return h;
}

// local copy of the host-side init_fastdiv_values body (common.cuh version is static and
// not emitted for the host pass of a standalone TU)
static const uint3 probe_fastdiv(uint64_t d_64) {
    uint32_t d = (uint32_t) d_64;
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }
    uint32_t mp = (uint32_t) ((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

template <bool KQmma, bool KQf16dq, ggml_type TK, ggml_type TV>
static void variant(const char * name, const char * Kc, const char * Vc,
                    const float * Q, float * dst, float2 * meta, int nsm, int reps) {
    typedef decltype(&flash_attn_ext_vec<D, 1, TK, TV, false, KQmma, KQf16dq>) kern_t;
    kern_t kern = &flash_attn_ext_vec<D, 1, TK, TV, false, KQmma, KQf16dq>;

    cudaFuncAttributes fa{}; CK(cudaFuncGetAttributes(&fa, (const void *)kern));

    int max_blocks_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_sm, (const void *)kern, 128, 0));

    // replicate launch_fattn parallel_blocks selection (non-stream_k path)
    const int ntiles_x = 1, ncols1 = 1, ncols2 = 1;
    const int gqa_ratio = NHEAD / NKVH;
    const int ntiles_z_gqa = gqa_ratio;               // ncols2 == 1
    const int ntiles_dst = ntiles_x * ntiles_z_gqa * NKVH;   // * ne03(1)
    const int ntiles_KV = (NKV + D - 1) / D;          // nbatch_fa = D
    int parallel_blocks = std::max(1, max_blocks_sm);
    parallel_blocks = std::min(parallel_blocks, ntiles_KV);
    const int blocks_per_wave = nsm * std::max(1, max_blocks_sm);
    int nwaves_best = 0, eff_best = 0;
    for (int pb = parallel_blocks; pb <= ntiles_KV; ++pb) {
        const int nblocks_total = ntiles_dst * pb;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int eff = 100 * nblocks_total / (nwaves * blocks_per_wave);
        if (eff_best >= 95 && nwaves > nwaves_best) break;
        if (eff > eff_best) { nwaves_best = nwaves; eff_best = eff; parallel_blocks = pb; }
    }

    const dim3 grid(ntiles_x, parallel_blocks, ntiles_z_gqa * NKVH);
    const dim3 block(32, 4, 1);

    const uint3 ne01 = probe_fastdiv(NQ);

    auto launch = [&]() {
        kern<<<grid, block>>>(
            (const char *)Q, Kc, Vc, nullptr, nullptr, nullptr, dst, meta,
            1.0f / 16.0f, 0.0f, 1.0f, 1.0f, 16u, 0.0f,
            D, ne01, NHEAD, 1, D * 4, D * 4, NHEAD * D * 4,
            D, NKV, NKVH, 1, NB1, NKV * NB1, (int64_t)NKVH * NKV * NB1,
            NB1, NKV * NB1, (int64_t)NKVH * NKV * NB1,
            0, 0, 0, 0, 0, 0);
    };

    launch(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); launch(); cudaEventRecord(e1);
        CK(cudaDeviceSynchronize());
        float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    std::sort(ms.begin(), ms.end());
    const float med = ms[ms.size()/2];
    const double tok_per_s = 1000.0 / med;   // one launch = attention of one decode token at d32768
    printf("%-26s occ=%d pb=%d regs=%d smem=%zuB | med %7.3f ms  min %7.3f max %7.3f | %6.2f t/s-equiv\n",
        name, max_blocks_sm, parallel_blocks, fa.numRegs, fa.sharedSizeBytes,
        med, ms.front(), ms.back(), tok_per_s);
}

int main() {
    int dev = 0; cudaDeviceProp dp{}; CK(cudaGetDeviceProperties(&dp, dev));
    printf("GPU: %s SMs=%d cc=%d.%d  (probe: flash_attn_ext_vec D=256 nq=1 nkv=%d heads=%d/%d)\n",
        dp.name, dp.multiProcessorCount, dp.major, dp.minor, NKV, NHEAD, NKVH);

    // Q random small
    std::vector<float> hq(NHEAD * NQ * D);
    unsigned s = 3;
    for (auto & x : hq) { s = s*1664525u + 1013904223u; x = ((int)(s >> 9) % 4000 - 2000) / 1000.0f; }
    float * Q; CK(cudaMalloc(&Q, hq.size()*4)); CK(cudaMemcpy(Q, hq.data(), hq.size()*4, cudaMemcpyHostToDevice));

    const auto q41 = make_q41_cache(); const auto nv4 = make_nvfp4_cache();
    char *Kq41, *Kv4, *Vq41, *Vnv4;
    CK(cudaMalloc(&Kq41, q41.size())); CK(cudaMemcpy(Kq41, q41.data(), q41.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Vq41, q41.size())); CK(cudaMemcpy(Vq41, q41.data(), q41.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Kv4,  nv4.size())); CK(cudaMemcpy(Kv4,  nv4.data(),  nv4.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Vnv4, nv4.size())); CK(cudaMemcpy(Vnv4, nv4.data(),  nv4.size(), cudaMemcpyHostToDevice));

    const int maxpb = (NKV + D - 1) / D;
    float * dst;  CK(cudaMalloc(&dst,  (size_t)maxpb * D * NQ * NHEAD * 4));  CK(cudaMemset(dst, 0, (size_t)maxpb * D * NQ * NHEAD * 4));
    float2 * meta; CK(cudaMalloc(&meta, (size_t)maxpb * NQ * NHEAD * sizeof(float2))); CK(cudaMemset(meta, 0, (size_t)maxpb * NQ * NHEAD * sizeof(float2)));

    const int reps = 11;
    variant<false,false,GGML_TYPE_Q4_1, GGML_TYPE_Q4_1 >("V1 q4_1/q4_1 dot",   Kq41, Vq41, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<false,false,GGML_TYPE_NVFP4,GGML_TYPE_NVFP4>("V2 nvfp4/nvfp4 dot", Kv4, Vnv4, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<false,true, GGML_TYPE_NVFP4,GGML_TYPE_NVFP4>("V3 nvfp4/nvfp4 f16dq",Kv4, Vnv4, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<false,true, GGML_TYPE_NVFP4,GGML_TYPE_Q4_1 >("V4 nvfp4K+q4_1V fus", Kv4, Vq41, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<false,false,GGML_TYPE_Q4_1, GGML_TYPE_NVFP4>("V5 q4_1K+nvfp4V dot", Kq41, Vnv4, Q, dst, meta, dp.multiProcessorCount, reps);

    return 0;
}
