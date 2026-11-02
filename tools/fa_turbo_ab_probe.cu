// fa_turbo_ab_probe.cu - times the PRODUCTION flash_attn_ext_vec kernel for the nq=1 / D=256
// decode shape (bench tg @ d32768 equivalent) for the turbo KV combos, and dumps each variant's
// exact output for bitwise old-vs-new comparison (S1 mux decode vs LUT decode).
//
// Build the SAME source against the pre-S1 and post-S1 headers (header swap, see
// rebuild_turbo_ab_both.cmd), run both exes with a dump dir, then:
//   fc /b dump_old\T1_turbo4_0-turbo3_0.bin dump_new\T1_turbo4_0-turbo3_0.bin   (must be identical)
// and compare the med timings.
//
//   T1 turbo4_0 K + turbo3_0 V, dot  (economical production combo)
//   T2 q5_1    K + turbo3_0 V, dot
//   T3 turbo4_0 K + turbo4_0 V, dot
//   T4 turbo5_0 K + turbo3_0 V, dot
//   T5 turbo4q_0 K + turbo4_0 V, dot
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

// row bytes at D=256 (2 quant blocks per row)
static const int NB_TURBO4  = 2*(int) sizeof(block_turbo4_0);   // 132
static const int NB_TURBO3  = 2*(int) sizeof(block_turbo3_0);   // 100
static const int NB_TURBO5  = 2*(int) sizeof(block_turbo5_0);   // 164
static const int NB_TURBO4Q = 2*(int) sizeof(block_turbo4q_0);  // 136
static const int NB_Q51     = (D/QK5_1)*(int) sizeof(block_q5_1);

// benign cache: random index bytes, half-norms patched to 0.25 (and rnorm for Q_prod) so all
// decoded values are finite and moderate. norm (and rnorm) are the first fields of every block.
static std::vector<uint8_t> make_turbo_cache(size_t row_bytes, int blk_bytes, int nhalves, unsigned seed) {
    std::vector<uint8_t> h((size_t) NKVH * NKV * row_bytes);
    unsigned s = seed;
    for (size_t i = 0; i < h.size(); ++i) { s = s*1664525u + 1013904223u; h[i] = (uint8_t)(s >> 16); }
    const __half d = __float2half(0.25f);
    uint16_t dh; memcpy(&dh, &d, 2);
    for (size_t blk = 0; blk < h.size()/(size_t) blk_bytes; ++blk) {
        uint8_t * b = h.data() + blk*(size_t) blk_bytes;
        for (int j = 0; j < nhalves; ++j) { b[2*j] = dh & 0xFF; b[2*j + 1] = dh >> 8; }
    }
    return h;
}

static std::vector<uint8_t> make_q51_cache() {
    std::vector<uint8_t> h((size_t) NKVH * NKV * NB_Q51);
    unsigned s = 5;
    for (int blk = 0; blk < NKVH*NKV*(D/QK5_1); ++blk) {
        uint8_t * b = h.data() + (size_t) blk*sizeof(block_q5_1);
        const __half d = __float2half(0.25f), m = __float2half(0.0f);
        uint16_t dh, mh; memcpy(&dh, &d, 2); memcpy(&mh, &m, 2);
        b[0] = dh & 0xFF; b[1] = dh >> 8; b[2] = mh & 0xFF; b[3] = mh >> 8;
        for (int i = 4; i < (int) sizeof(block_q5_1); ++i) { s = s*1664525u + 1013904223u; b[i] = (uint8_t)(s >> 13); }
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

static char g_dump_dir[512] = "";

template <ggml_type TK, ggml_type TV>
static void variant(const char * name, const char * Kc, const char * Vc, int nb11, int nb21,
                    const float * Q, float * dst, float2 * meta, int nsm, int reps) {
    typedef decltype(&flash_attn_ext_vec<D, 1, TK, TV, false, false, false>) kern_t;
    kern_t kern = &flash_attn_ext_vec<D, 1, TK, TV, false, false, false>;

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
            D, NKV, NKVH, 1, nb11, NKV * nb11, (int64_t)NKVH * NKV * nb11,
            nb21, NKV * nb21, (int64_t)NKVH * NKV * nb21,
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

    // dump the final output (the last launch left the correct result in dst/meta) for bitwise diff
    uint64_t fnv = 1469598103934665603ull;
    if (g_dump_dir[0]) {
        const size_t ndst = (size_t) ntiles_KV * D * NQ * NHEAD;   // worst-case tile count
        std::vector<float> hd(ndst);
        CK(cudaMemcpy(hd.data(), dst, ndst*4, cudaMemcpyDeviceToHost));
        std::vector<float2> hm((size_t) ntiles_KV * NQ * NHEAD);
        CK(cudaMemcpy(hm.data(), meta, hm.size()*sizeof(float2), cudaMemcpyDeviceToHost));
        char path[700]; snprintf(path, sizeof(path), "%s\\%s.bin", g_dump_dir, name);
        FILE * f = fopen(path, "wb");
        if (!f) { printf("dump open fail %s\n", path); exit(1); }
        fwrite(hd.data(), 4, ndst, f); fwrite(hm.data(), sizeof(float2), hm.size(), f);
        fclose(f);
        const uint8_t * p = (const uint8_t *) hd.data();
        for (size_t i = 0; i < ndst*4; ++i) { fnv = (fnv ^ p[i]) * 1099511628211ull; }
        const uint8_t * q = (const uint8_t *) hm.data();
        for (size_t i = 0; i < hm.size()*sizeof(float2); ++i) { fnv = (fnv ^ q[i]) * 1099511628211ull; }
    }

    printf("%-26s occ=%d pb=%d regs=%d smem=%zuB | med %7.3f ms  min %7.3f max %7.3f | %6.2f t/s-equiv | fnv %016llx\n",
        name, max_blocks_sm, parallel_blocks, fa.numRegs, fa.sharedSizeBytes,
        med, ms.front(), ms.back(), tok_per_s, (unsigned long long) fnv);
}

int main(int argc, char ** argv) {
    if (argc > 1) { snprintf(g_dump_dir, sizeof(g_dump_dir), "%s", argv[1]); }
    int dev = 0; cudaDeviceProp dp{}; CK(cudaGetDeviceProperties(&dp, dev));
    printf("GPU: %s SMs=%d cc=%d.%d  (probe: flash_attn_ext_vec D=256 nq=1 nkv=%d heads=%d/%d, turbo A/B)\n",
        dp.name, dp.multiProcessorCount, dp.major, dp.minor, NKV, NHEAD, NKVH);

    // Q random small
    std::vector<float> hq(NHEAD * NQ * D);
    unsigned s = 3;
    for (auto & x : hq) { s = s*1664525u + 1013904223u; x = ((int)(s >> 9) % 4000 - 2000) / 1000.0f; }
    float * Q; CK(cudaMalloc(&Q, hq.size()*4)); CK(cudaMemcpy(Q, hq.data(), hq.size()*4, cudaMemcpyHostToDevice));

    const auto t4  = make_turbo_cache(NB_TURBO4,  sizeof(block_turbo4_0),  1, 11);
    const auto t3  = make_turbo_cache(NB_TURBO3,  sizeof(block_turbo3_0),  1, 13);
    const auto t5  = make_turbo_cache(NB_TURBO5,  sizeof(block_turbo5_0),  1, 17);
    const auto t4q = make_turbo_cache(NB_TURBO4Q, sizeof(block_turbo4q_0), 2, 19);
    const auto q51 = make_q51_cache();

    char *Kt4, *Kt5, *Kt4q, *Kq51, *Vt3, *Vt4;
    CK(cudaMalloc(&Kt4,  t4.size()));  CK(cudaMemcpy(Kt4,  t4.data(),  t4.size(),  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Kt5,  t5.size()));  CK(cudaMemcpy(Kt5,  t5.data(),  t5.size(),  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Kt4q, t4q.size())); CK(cudaMemcpy(Kt4q, t4q.data(), t4q.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Kq51, q51.size())); CK(cudaMemcpy(Kq51, q51.data(), q51.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Vt3,  t3.size()));  CK(cudaMemcpy(Vt3,  t3.data(),  t3.size(),  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Vt4,  t4.size()));  CK(cudaMemcpy(Vt4,  t4.data(),  t4.size(),  cudaMemcpyHostToDevice));

    const int maxpb = (NKV + D - 1) / D;
    float * dst;  CK(cudaMalloc(&dst,  (size_t)maxpb * D * NQ * NHEAD * 4));  CK(cudaMemset(dst, 0, (size_t)maxpb * D * NQ * NHEAD * 4));
    float2 * meta; CK(cudaMalloc(&meta, (size_t)maxpb * NQ * NHEAD * sizeof(float2))); CK(cudaMemset(meta, 0, (size_t)maxpb * NQ * NHEAD * sizeof(float2)));

    const int reps = 11;
    variant<GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0>("T1_turbo4-t3",         Kt4,  Vt3, NB_TURBO4, NB_TURBO3, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<GGML_TYPE_Q5_1,     GGML_TYPE_TURBO3_0>("T2_q5_1-t3",           Kq51, Vt3, NB_Q51,    NB_TURBO3, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0>("T3_turbo4-t4",         Kt4,  Vt4, NB_TURBO4, NB_TURBO4, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<GGML_TYPE_TURBO5_0, GGML_TYPE_TURBO3_0>("T4_turbo5-t3",         Kt5,  Vt3, NB_TURBO5, NB_TURBO3, Q, dst, meta, dp.multiProcessorCount, reps);
    variant<GGML_TYPE_TURBO4Q_0,GGML_TYPE_TURBO4_0>("T5_turbo4q-t4",        Kt4q, Vt4, NB_TURBO4Q,NB_TURBO4, Q, dst, meta, dp.multiProcessorCount, reps);

    return 0;
}
