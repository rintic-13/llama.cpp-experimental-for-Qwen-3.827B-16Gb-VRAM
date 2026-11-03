/*
 * fa_mma_mixtest.cu — SM120 mixed-precision MMA capability matrix for the fused-decode plan.
 *
 * Question (NVFP4_FA_RESEARCH.md §9 next direction): can any mma.sync variant take FP16 Q
 * together with FP4/FP8 K natively, and which block_scale kinds exist for non-e2m1 A/B pairs?
 * ptxas operand-type validation IS the capability answer (a variant that doesn't compile does
 * not exist on sm_120a); the one-hot map + random-dot tests verify surviving variants on-device.
 *
 * Variants (the f16-mixed probes build only with -DMIXF16 so a default build always succeeds):
 *   V1  dense m16n8k16     f16 x f16                 (control, known good)
 *   V2  kind::mxf8f6f4 1X k32  e2m1 x e2m1  ue8m0    (fp4 pair under the fp8 kind)
 *   V3  kind::mxf8f6f4 1X k32  e4m3 x e2m1  ue8m0    (control: already used in probe bench)
 *   V4  kind::mxf8f6f4 1X k32  e4m3 x e5m2  ue8m0
 *   V5  kind::mxf8f6f4 1X k32  e5m2 x e2m1  ue8m0
 *   V6  kind::mxf8f6f4 1X k32  f16  x e2m1  ue8m0    <- the "ideal candidate" probe
 *   V7  kind::mxf8f6f4 1X k32  f16  x e4m3  ue8m0
 *   V8  kind::mxf8f6f4 1X k32  e2m1 x f16   ue8m0
 *
 * Fragment maps used (PTX ISA; cross-checked against the k_verify/k_map results for V3):
 *   k16 f16: a0=(row g,    k 4t..4t+1) a1=(row g+8, 4t..4t+1) a2=(row g, 8+4t..) a3=(row g+8, 8+4t..)
 *            b0=(col g, k 4t..4t+1)    b1=(col g, 8+4t..8+4t+1)          [half2, 2 coords/word]
 *   k32 fp8/fp4: a_r=(row g or g+8, byte i = coords 8r+2i, 8r+2i+1); b0=(col g, coords 0..15),
 *            b1=(col g, 16..31), byte i = coords 2i,2i+1                 [1 coord/byte]
 *   C: c0/c1=(row g/g+8, col 2t/2t+1); SF_A: lanes 4g+0/4g+1 (rows g/g+8), SF_B: lane 4g.
 *
 * nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] -DMIXF16 tools\fa_mma_mixtest.cu -o build...\bin\fa_mma_mixtest.exe
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <type_traits>

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

static float frand01(uint32_t & st) {   // xorshift, [0,1)
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (float) (st & 0xFFFFFF) / 16777216.0f;
}

// ---------------------------------------------------------------------------
// one instruction per variant; operand type strings are what ptxas does/does not accept
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_v1(float * D, const uint32_t * A, const uint32_t * B) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}
#define BS32(opA, opB) \
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32." opA "." opB ".f32.ue8m0 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};" \
                 : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]) \
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(sa), "r"(sb))
__device__ __forceinline__ void mma_v2(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("e2m1", "e2m1"); }
__device__ __forceinline__ void mma_v3(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("e4m3", "e2m1"); }
__device__ __forceinline__ void mma_v4(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("e4m3", "e5m2"); }
__device__ __forceinline__ void mma_v5(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("e5m2", "e2m1"); }
#ifdef MIXF16
__device__ __forceinline__ void mma_v6(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("f16", "e2m1"); }
__device__ __forceinline__ void mma_v7(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("f16", "e4m3"); }
__device__ __forceinline__ void mma_v8(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) { BS32("e2m1", "f16"); }
#endif

template <int V>
__device__ __forceinline__ void runmma(float * D, const uint32_t * A, const uint32_t * B, uint32_t sa, uint32_t sb) {
    if constexpr (V == 1)      { (void) sa; (void) sb; mma_v1(D, A, B); }
    else if constexpr (V == 2) mma_v2(D, A, B, sa, sb);
    else if constexpr (V == 3) mma_v3(D, A, B, sa, sb);
    else if constexpr (V == 4) mma_v4(D, A, B, sa, sb);
    else if constexpr (V == 5) mma_v5(D, A, B, sa, sb);
#ifdef MIXF16
    else if constexpr (V == 6) mma_v6(D, A, B, sa, sb);
    else if constexpr (V == 7) mma_v7(D, A, B, sa, sb);
    else if constexpr (V == 8) mma_v8(D, A, B, sa, sb);
#endif
}

// ---------------------------------------------------------------------------
// one-hot map: value 1.0 placed at exactly one A coord and one B coord per mma,
// uniform scales 1.0 -> C must light up iff Acoord == Bcoord (one diagonal cell).
// ---------------------------------------------------------------------------
template <int V>
__device__ __forceinline__ uint32_t one_val(int coord) {   // one coord encoded as value 1.0
    if constexpr (V == 1)      return (uint32_t) 0x3C00u;              // f16 1.0 (16 bits)
    else if constexpr (V == 4 || V == 7) { (void) coord; return 0x38u; } // e4m3 A-side 1.0
    else if constexpr (V == 5) { (void) coord; return 0x3Cu; }         // e5m2 1.0
    else if constexpr (V == 6) { (void) coord; return 0x3C00u; }       // V6 A is f16 (if it existed)
    else                       return 0x22u;                           // e2m1 code 2 (=1.0)
}
template <int V>
__device__ __forceinline__ uint32_t one_val_b(int coord) {
    if constexpr (V == 1)      return 0x3C00u;
    else if constexpr (V == 4 || V == 7) { (void) coord; return 0x38u; } // e4m3 B-side 1.0
    else if constexpr (V == 8) { (void) coord; return 0x3C00u; }       // f16 B-side
    else                       return 0x22u;                           // e2m1 for all fp4 B-sides
}

template <int V>
__global__ void k_mixmap(float * __restrict__ out, float * __restrict__ dbg,
                         int c1t, int c2t, int r2t, uint32_t sfb) {
    constexpr int NK = (V == 1) ? 16 : 32;
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    for (int c1 = 0; c1 < NK; ++c1) for (int c2 = 0; c2 < NK; ++c2) for (int r2 = 0; r2 < 8; ++r2) {
        if (dbg && c1t >= 0 && (c1 != c1t || c2 != c2t || r2 != r2t)) continue;
        uint32_t A[4] = { 0,0,0,0 }, B[2] = { 0,0 };
        if constexpr (V == 1) {
            if (g == 0) {
                const int slot = (c1 & 7) - 4 * t;               // this lane's t-window covers k 4t..4t+3 per half
                if (slot >= 0 && slot < 4) A[(c1 >= 8) ? 2 : 0] = 0x3C00u << (16 * ((slot >= 2) ? 1 : 0));
            }
            if (g == r2) {
                const int slot = ((c2 >= 8) ? (c2 - 8) : c2) - 4 * t;
                if (slot >= 0 && slot < 4) B[(c2 >= 8) ? 1 : 0] = 0x3C00u << (16 * ((slot >= 2) ? 1 : 0));
            }
        } else {
            const uint32_t vA = one_val<V>(c1), vB = one_val_b<V>(c2);
            const int ar = c1 >> 3;                                // 0..3 register quarter
            const int sh = 4 * (c1 & 7);
            if (g == 0) A[ar] |= vA << sh;
            const int bh = (c2 >= 16) ? 1 : 0;
            const int sh2 = 4 * (c2 & 7);
            if (g == r2) B[bh] |= vB << sh2;
        }
        float D[4] = { 0.f, 0.f, 0.f, 0.f };
        runmma<V>(D, A, B, sfb, sfb);
        if (dbg) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (D[e] != 0.f) {
                    const int slot = atomicAdd((int *) (dbg + 1024), 1);
                    if (slot < 250) {
                        float4 * rec = (float4 *) (dbg + 1040);
                        const float cell = (float) (((lane >> 2) + ((e >= 2) ? 8 : 0)) * 1000 + (2 * (lane & 3) + (e & 1)));
                        rec[slot] = make_float4((float) c1, (float) c2, cell, D[e]);
                    }
                }
            }
        }
        float s = fabsf(D[0]) + fabsf(D[1]) + fabsf(D[2]) + fabsf(D[3]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off, 32);
        if (lane != 0) continue;
        const float expv = (c1 == c2) ? 1.f : 0.f;
        out[(size_t) ((c1 * NK + c2) * 8 + r2) * 4] = (fabsf(s - expv) < 0.5f) ? 1.f : 2.f;
    }
}

// Minimal ground-truth probe: for one (c1,c2,r2) case, dump all four D regs of all 32 lanes.
// out layout: [case][lane][4]
template <int V>
__global__ void k_mixdump(float * __restrict__ out, int c1, int c2, int r2) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    uint32_t A[4] = { 0,0,0,0 }, B[2] = { 0,0 };
    if constexpr (V == 1) {
        if (g == 0) {
            const int slot = (c1 & 7) - 4 * t;
            if (slot >= 0 && slot < 4) A[(c1 >= 8) ? 2 : 0] = 0x3C00u << (16 * ((slot >= 2) ? 1 : 0));
        }
        if (g == r2) {
            const int slot = ((c2 >= 8) ? (c2 - 8) : c2) - 4 * t;
            if (slot >= 0 && slot < 4) B[(c2 >= 8) ? 1 : 0] = 0x3C00u << (16 * ((slot >= 2) ? 1 : 0));
        }
    } else {
        const uint32_t vA = one_val<V>(c1), vB = one_val_b<V>(c2);
        if (g == 0) A[c1 >> 3] |= vA << (4 * (c1 & 7));
        if (g == r2) B[(c2 >= 16) ? 1 : 0] |= vB << (4 * (c2 & 7));
    }
    float D[4] = { 0.f, 0.f, 0.f, 0.f };
    runmma<V>(D, A, B, 0x81818181u, 0x81818181u);
#pragma unroll
    for (int e = 0; e < 4; ++e) out[lane * 4 + e] = D[e];
}

// Scale binding test for the k32 block_scale variants: all coords 1.0 (32 k-slots),
// all scale bytes = bA/bB.
// t=0: uniform byte b (arg) over all lanes -> C = 32 * fA * fB * sa(b) * sb(b)
// t=1: A uniform 0x78, B bytes vary by sub-chunk: byte ch = arg + ch -> C = sum_ch 8*sa*sb(ch)
//      (fp8 side: 8 coords per 8-byte... careful: ue8m0 scale_vec::1X chunk = 32 coords? k32 has one byte per operand;
//       for k32 there is exactly ONE scale byte per operand per instruction operand half? test both bindings)
template <int V>
__global__ void k_mixscale(float * __restrict__ out, int bA, int bB, int mode, int fillbA, int fillbB) {
    const int lane = threadIdx.x;
    uint32_t sa, sb;
    if (mode == 0) { sa = (uint32_t) bA * 0x01010101u; sb = (uint32_t) bB * 0x01010101u; }
    else { sa = 0x80808080u; sb = (uint32_t) ((bB & 0xFF) | ((bB + 1) << 8) | ((bB + 2) << 16) | ((bB + 3) << 24)); }
    uint32_t A[4], B[2];
    const uint32_t dfltA = (V == 3 || V == 4) ? 0x38383838u : (V == 5) ? 0x3C3C3C3Cu : 0x22222222u;
    const uint32_t dfltB = (V == 2 || V == 3 || V == 5) ? 0x22222222u : 0x3C3C3C3Cu;   // V4 B = e5m2 1.0 = 0x3C
    const uint32_t fillA  = (fillbA >= 0) ? ((uint32_t) (fillbA & 0xFF) * 0x01010101u) : dfltA;
    const uint32_t fillBx = (fillbB >= 0) ? ((uint32_t) (fillbB & 0xFF) * 0x01010101u) : dfltB;
#pragma unroll
    for (int q = 0; q < 4; ++q) A[q] = fillA;
#pragma unroll
    for (int q = 0; q < 2; ++q) B[q] = fillBx;
    float D[4] = { 0.f, 0.f, 0.f, 0.f };
    runmma<V>(D, A, B, sa, sb);
    const int g = lane >> 2, t = lane & 3;
    out[g * 8 + 2 * t + 0] = D[0];
    out[g * 8 + 2 * t + 1] = D[1];
    out[(g + 8) * 8 + 2 * t + 0] = D[2];
    out[(g + 8) * 8 + 2 * t + 1] = D[3];
}

// ---------------------------------------------------------------------------
// numeric host-side format helpers
// ---------------------------------------------------------------------------
static float e4m3_host(uint8_t x) {
    if ((x & 0x7F) == 0x7F) return NAN;
    const float s = (x >> 7) ? -1.0f : 1.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    if (e == 0) return s * (float) m / 8.0f * ldexpf(1.0f, -6);
    return s * (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7);
}
static float e5m2_host(uint8_t x) {
    const float s = (x >> 7) ? -1.0f : 1.0f;
    const int e = (x >> 2) & 0x1F, m = x & 3;
    if (e == 0) return s * (float) m / 4.0f * ldexpf(1.0f, -14);
    if (e == 0x1F) return NAN;
    return s * (1.0f + m / 4.0f) * ldexpf(1.0f, e - 15);
}
static float e2m1_host(int code) {
    static const float tab[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
    const float v = tab[code & 7];
    return (code & 8) ? -v : v;
}
static float half_bits_host(uint16_t h) {
    __half_raw r; r.x = h;
    __half hh = *reinterpret_cast<__half *>(&r);
    return __half2float(hh);
}

// ---------------------------------------------------------------------------
// random values + per-16-coord-chunk ue8m0 scales; exact fp64 reference.
// ---------------------------------------------------------------------------
template <int V>
__global__ void k_mixdot(const uint32_t * __restrict__ Afr, const uint32_t * __restrict__ Bfr,
                         const uint32_t * __restrict__ sA, const uint32_t * __restrict__ sB,
                         float * __restrict__ out) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    float D[4] = { 0.f, 0.f, 0.f, 0.f };
    uint32_t A[4], B[2];
#pragma unroll
    for (int q = 0; q < 4; ++q) A[q] = Afr[lane * 4 + q];
#pragma unroll
    for (int q = 0; q < 2; ++q) B[q] = Bfr[lane * 2 + q];
    runmma<V>(D, A, B, sA[lane], sB[lane]);
    (void) g; (void) t;
    out[(g + 0) * 8 + 2 * t + 0] = D[0];
    out[(g + 0) * 8 + 2 * t + 1] = D[1];
    out[(g + 8) * 8 + 2 * t + 0] = D[2];
    out[(g + 8) * 8 + 2 * t + 1] = D[3];
}

int main() {
    int dev = 0; cudaDeviceProp prop;
    ck(cudaGetDevice(&dev), "getdev");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    printf("# device: %s  cc=%d.%d  SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fflush(stdout);

    // Ground-truth single-case dumps (lane0 D-registers), before any sweep logic.
    {
        float * d_o; ck(cudaMalloc(&d_o, 128 * 4), "dump");
        struct { int V; const char * n; int c1, c2, r2; float want; } cases[] = {
            { 3, "V3 e4m3(1.0) x e2m1(1.0) same coord 0 ", 0,  0,  0, 1.f },
            { 3, "V3 e4m3(1.0) x e2m1(1.0) same coord 20", 20, 20, 5, 1.f },
            { 2, "V2 e2m1(1.0) x e2m1(1.0) same coord 0 ", 0,  0,  0, 1.f },
            { 1, "V1 f16(1.0)  x f16(1.0)   same coord 0", 0,  0,  0, 1.f },
        };
        for (auto & cs : cases) {
            float * p = nullptr;
            if (cs.V == 3) { k_mixdump<3><<<1, 32>>>(d_o, cs.c1, cs.c2, cs.r2); p = nullptr; }
            else if (cs.V == 2) k_mixdump<2><<<1, 32>>>(d_o, cs.c1, cs.c2, cs.r2);
            else k_mixdump<1><<<1, 32>>>(d_o, cs.c1, cs.c2, cs.r2);
            (void) p;
            cudaError_t le = cudaGetLastError(), se = cudaDeviceSynchronize();
            if (le || se) { printf("  dump %s: FAIL %s/%s\n", cs.n, cudaGetErrorString(le), cudaGetErrorString(se)); cudaGetLastError(); continue; }
            std::vector<float> o(128);
            ck(cudaMemcpy(o.data(), d_o, 128 * 4, cudaMemcpyDeviceToHost), "dump cpy");
            int hits = 0; double sum = 0;
            char l0buf[96];
            snprintf(l0buf, sizeof(l0buf), "[%g %g %g %g]", o[0], o[1], o[2], o[3]);
            for (float x : o) { if (x != 0.f) ++hits; sum += fabs((double) x); }
            printf("  dump %s: expect %g -> total|C| = %.6g  lane0 D %s  nonzero=%d\n", cs.n, cs.want, sum, l0buf, hits);
        }
        cudaFree(d_o);
    }
    fflush(stdout);

    // ue8m0 scale sweep for the k32 block_scale kind (all coords 1.0):
    {
        float * d_o; ck(cudaMalloc(&d_o, 128 * 4), "sweep");
        std::vector<float> o(128);
        auto s1 = [&](int V, int bA, int bB) -> double {
            if (V == 2) k_mixscale<2><<<1, 32>>>(d_o, bA, bB, 0, -1, -1);
            else if (V == 3) k_mixscale<3><<<1, 32>>>(d_o, bA, bB, 0, -1, -1);
            else k_mixscale<4><<<1, 32>>>(d_o, bA, bB, 0, -1, -1);
            if (cudaGetLastError() || cudaDeviceSynchronize()) { cudaGetLastError(); return -1.0; }
            ck(cudaMemcpy(o.data(), d_o, 128 * 4, cudaMemcpyDeviceToHost), "sweep cpy");
            double s = 0; for (float x : o) if (x != 0.f) { s = x; break; }
            return s;
        };
        const double fillv[2][2] = { { 1.0, 1.0 }, { 1.0, 1.0 } };   // V2: e2m1*e2m1=1 ; V3: e4m3*e2m1=1
        for (int b = 0x70; b <= 0x84; ++b) {
            const double c2 = s1(2, b, b), c3 = s1(3, b, b);
            printf("  scale byte 0x%02X: V2 C0=%12.6g (sa*sb=%10.6g)   V3 C0=%12.6g (sa*sb=%10.6g)\n",
                   b, c2, c2 / 32.0, c3, c3 / 32.0);
        }
        printf("  reference ue8m0(b)=2^(b-127): 0x7E=0.5 0x7F=1.0 0x80=2.0\n");
        // V3 operand-mixing matrix: A fill x B fill x scale byte -> C0 (expect 32*va*vb*sa*sb)
        {
            float * d_c; ck(cudaMalloc(&d_c, 128 * 4), "mixmat");
            std::vector<float> o(128);
            struct Row { uint32_t fA, fB; int b; };
            const Row rows[] = {
                { 0x38383838u, 0x22222222u, 0x81 }, { 0x3C3C3C3Cu, 0x22222222u, 0x81 },
                { 0x22222222u, 0x38383838u, 0x81 }, { 0x22222222u, 0x3C3C3C3Cu, 0x81 },
                { 0x38383838u, 0x38383838u, 0x81 }, { 0x3C3C3C3Cu, 0x3C3C3C3Cu, 0x81 },
                { 0x38383838u, 0x22222222u, 0x7F }, { 0x38383838u, 0x22222222u, 0x83 },
            };
            for (auto & r : rows) {
                k_mixscale<3><<<1, 32>>>(d_c, 0x81, r.b, 0, (int) (r.fA >> 24), (int) (r.fB >> 24));
                ck(cudaGetLastError(), "mm launch"); ck(cudaDeviceSynchronize(), "mm sync");
                ck(cudaMemcpy(o.data(), d_c, 128 * 4, cudaMemcpyDeviceToHost), "mm cpy");
                double s = 0; for (float x : o) s += fabs((double) x);
                printf("  V3 matrix: fA=0x%02X fB=0x%02X b=0x%02X -> sum|C|=%.6g  C[lane0]=[%.6g %.6g %.6g %.6g]\n",
                       (int) (r.fA >> 24), (int) (r.fB >> 24), r.b, s, o[0], o[1], o[2], o[3]);
            }
            cudaFree(d_c);
        }
        // per-operand scale byte sweep to isolate the V3 -4 factor
        {
            float * d_c2; ck(cudaMalloc(&d_c2, 128 * 4), "sw2");
            std::vector<float> o2(128);
            for (int bB : {0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82}) {
                k_mixscale<3><<<1, 32>>>(d_c2, 0x81, bB, 0, -1, -1);
                ck(cudaGetLastError(), "sw2a"); ck(cudaDeviceSynchronize(), "sw2as");
                ck(cudaMemcpy(o2.data(), d_c2, 128 * 4, cudaMemcpyDeviceToHost), "sw2c");
                printf("  V3 bA=0x81 bB=0x%02X -> C0=%g\n", bB, o2[0]);
            }
            for (int bA : {0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82}) {
                k_mixscale<3><<<1, 32>>>(d_c2, bA, 0x7D, 0, -1, -1);
                ck(cudaGetLastError(), "sw2b"); ck(cudaDeviceSynchronize(), "sw2bs");
                ck(cudaMemcpy(o2.data(), d_c2, 128 * 4, cudaMemcpyDeviceToHost), "sw2d");
                printf("  V3 bA=0x%02X bB=0x7D -> C0=%g\n", bA, o2[0]);
            }
            cudaFree(d_c2);
        }
        cudaFree(d_o);
    }

    auto test_onehot = [&](auto tag, const char * name, int V_unused) {
        constexpr int V = decltype(tag)::value;
        constexpr int NK = (V == 1) ? 16 : 32;
        const size_t n = (size_t) NK * NK * 8;
        float * d_map; ck(cudaMalloc(&d_map, n * 4), "map alloc");
        k_mixmap<V><<<1, 32>>>(d_map, nullptr, -1, -1, -1, 0x81818181u);
        cudaError_t le = cudaGetLastError(), se = cudaDeviceSynchronize();
        if (le != cudaSuccess || se != cudaSuccess) {
            printf("  %s MAP: FAIL %s / %s\n", name, cudaGetErrorString(le), cudaGetErrorString(se));
            cudaGetLastError(); cudaFree(d_map); return;
        }
        std::vector<float> mp(n);
        ck(cudaMemcpy(mp.data(), d_map, n * 4, cudaMemcpyDeviceToHost), "map cpy");
        size_t bad = 0, firstbad = (size_t) -1;
        for (size_t x = 0; x < n; ++x) if (mp[x] > 1.5f) { ++bad; if (firstbad == (size_t) -1) firstbad = x; }
        printf("  %s MAP one-hot: %s (%zu wrong / %zu)\n", name, bad ? "**WRONG**" : "ok", bad, n);
        if (bad) {
            const int r2i = (int) (firstbad % 8), c2i = (int) ((firstbad / 8) % NK), c1i = (int) ((firstbad / 8) / NK);
            float * d_dbg; ck(cudaMalloc(&d_dbg, (1040 + 250 * 4) * 4), "dbg alloc");
            ck(cudaMemset(d_dbg, 0, (1040 + 250 * 4) * 4), "dbg memset");
            k_mixmap<V><<<1, 32>>>(d_map, d_dbg, c1i, c2i, r2i, 0x81818181u);
            ck(cudaGetLastError(), "dbg launch"); ck(cudaDeviceSynchronize(), "dbg sync");
            std::vector<float> dbg(1040 + 250 * 4);
            ck(cudaMemcpy(dbg.data(), d_dbg, dbg.size() * 4, cudaMemcpyDeviceToHost), "dbg cpy");
            const int nrec = (int) dbg[1024];
            printf("      debug case A coord=%d B coord=%d col=%d -> %d nonzero C cells; lane0 D=[%g %g %g %g] A0=%08X A1=%08X B0=%08X B1=%08X\n",
                   c1i, c2i, r2i, nrec > 16 ? 16 : nrec,
                   dbg[1028], dbg[1029], dbg[1030], dbg[1031],
                   *(int *) &dbg[1032], *(int *) &dbg[1033], *(int *) &dbg[1036], *(int *) &dbg[1037]);
            for (int i = 0; i < nrec && i < 16; ++i) {
                const float * rec = &dbg[1040 + i * 4];
                printf("        A=%2d B=%2d row=%3.0f col=%3.0f val=%g\n",
                       (int) rec[0], (int) rec[1], rec[2] / 1000.f, rec[2] - 1000.f * floorf(rec[2] / 1000.f), rec[3]);
            }
            cudaFree(d_dbg);
        }
        cudaFree(d_map);
    };

    auto test_dot = [&](auto tag, const char * name) {
        constexpr int V = decltype(tag)::value;
        constexpr int NK = (V == 1) ? 16 : 32;
        constexpr bool AF16  = (V == 1 || V == 6);        // A-side half
        constexpr bool AE4M3 = (V == 4 || V == 7);        // A-side e4m3
        constexpr bool AE5M2 = (V == 5);
        constexpr bool BF16  = (V == 8);                  // B-side half
        constexpr bool BE4M3 = (V == 4);                  // B-side e4m3
        constexpr bool BF16B = false;
        (void) BF16B;
        uint32_t st = 0x9E3779B9u + (uint32_t) V;
        std::vector<std::vector<float>> av(16, std::vector<float>(NK)), bv(8, std::vector<float>(NK));
        std::vector<std::vector<uint32_t>> ea(16, std::vector<uint32_t>(NK)), eb(8, std::vector<uint32_t>(NK));
        for (int i = 0; i < 16; ++i) for (int c = 0; c < NK; ++c) {
            if (AF16)           { ea[i][c] = (uint32_t) (uint16_t) (0x3C00u + (uint32_t)(frand01(st) * 2000.0f)); av[i][c] = half_bits_host((uint16_t) ea[i][c]); }
            else if (AE4M3)     { ea[i][c] = (uint8_t) ((uint32_t)(frand01(st) * 40.0f) | ((frand01(st) < 0.5f) ? 0x80u : 0u)); av[i][c] = e4m3_host((uint8_t) ea[i][c]); }
            else if (AE5M2)     { ea[i][c] = (uint8_t) ((uint32_t)(frand01(st) * 16.0f) | ((frand01(st) < 0.5f) ? 0x80u : 0u)); av[i][c] = e5m2_host((uint8_t) ea[i][c]); }
            else                { ea[i][c] = (uint32_t) ((int)(frand01(st) * 16.0f) & 15); av[i][c] = e2m1_host((int) ea[i][c]); }
        }
        for (int j = 0; j < 8; ++j) for (int c = 0; c < NK; ++c) {
            if (BF16)           { eb[j][c] = (uint32_t) (uint16_t) (0x3C00u + (uint32_t)(frand01(st) * 2000.0f)); bv[j][c] = half_bits_host((uint16_t) eb[j][c]); }
            else if (BE4M3)     { eb[j][c] = (uint8_t) ((uint32_t)(frand01(st) * 40.0f) | ((frand01(st) < 0.5f) ? 0x80u : 0u)); bv[j][c] = e4m3_host((uint8_t) eb[j][c]); }
            else                { eb[j][c] = (uint32_t) ((int)(frand01(st) * 16.0f) & 15); bv[j][c] = e2m1_host((int) eb[j][c]); }
        }
        // empirical neutral byte for scale_vec::1X ue8m0 on this chip is 0x81 (=1.0, see sweep)
        std::vector<std::vector<uint8_t>> sAe(16, std::vector<uint8_t>(4, 129)), sBe(8, std::vector<uint8_t>(4, 129));
        if constexpr (V != 1) {
            for (int i = 0; i < 16; ++i) for (int ch = 0; ch < 4; ++ch) sAe[i][ch] = (uint8_t) (120 + (int)(frand01(st) * 15.0f));
            for (int j = 0; j < 8; ++j)  for (int ch = 0; ch < 4; ++ch) sBe[j][ch] = (uint8_t) (120 + (int)(frand01(st) * 15.0f));
        }

        std::vector<uint32_t> Afr(32 * 4, 0), Bfr(32 * 2, 0), sA(32, 0), sB(32, 0);
        for (int lane = 0; lane < 32; ++lane) {
            const int g = lane >> 2, t = lane & 3;
            for (int q = 0; q < 4; ++q) {
                const int row = (q & 1) ? g + 8 : g;
                const int khalf = q >= 2;
                uint32_t w = 0;
                if constexpr (NK == 32) {
                    const int koff = 8 * q;
                    if constexpr (AF16) { /* f16 A-side is k16-only (V1); V6 has NK=32 with fp4-sized A? no: V6 A is f16 at k32 = 4 bytes? see below */ }
                    if constexpr (AE4M3 || AE5M2) {                       // fp8: 1 coord per byte
                        for (int i = 0; i < 8; ++i) w |= (ea[row][koff + i] & 0xFFu) << (8 * i);
                    } else if constexpr (AF16) {                          // V6: f16 coords in an fp8-kind k32? treat 2B/coord
                        for (int i = 0; i < 4; ++i) w |= (ea[row][koff + 2*i] & 0xFFFFu) << (16 * i);
                    } else {                                              // e2m1: 1 nibble/coord, byte j = coords 2j,2j+1
                        for (int i = 0; i < 4; ++i) w |= ((ea[row][koff + 2*i] & 0xFu) | ((ea[row][koff + 2*i + 1] & 0xFu) << 4)) << (8 * i);
                    }
                } else {
                    const int koff = 8 * khalf + 4 * t;
                    for (int i = 0; i < 2; ++i) w |= (ea[row][koff + i] & 0xFFFFu) << (16 * i);
                }
                Afr[lane * 4 + q] = w;
            }
            for (int q = 0; q < 2; ++q) {
                uint32_t w = 0;
                if constexpr (NK == 32) {
                    const int koff = 16 * q;
                    if constexpr (BF16 || BE4M3) {
                        for (int i = 0; i < 8; ++i) w |= (eb[g][koff + i] & 0xFFu) << (8 * i);
                    } else {
                        for (int i = 0; i < 4; ++i) w |= ((eb[g][koff + 2*i] & 0xFu) | ((eb[g][koff + 2*i + 1] & 0xFu) << 4)) << (8 * i);
                    }
                } else {
                    const int koff = 8 * q + 4 * t;
                    for (int i = 0; i < 2; ++i) w |= (eb[g][koff + i] & 0xFFFFu) << (16 * i);
                }
                Bfr[lane * 2 + q] = w;
            }
            uint32_t wa = 0, wb = 0;
            if constexpr (V != 1) {
                if (lane % 4 == 0) { for (int ch = 0; ch < 4; ++ch) wa |= (uint32_t) sAe[g][ch]     << (8 * ch); }
                if (lane % 4 == 1) { for (int ch = 0; ch < 4; ++ch) wa |= (uint32_t) sAe[g + 8][ch] << (8 * ch); }
                if (lane % 4 == 0) { for (int ch = 0; ch < 4; ++ch) wb |= (uint32_t) sBe[g][ch]     << (8 * ch); }
            }
            sA[lane] = wa; sB[lane] = wb;
        }

        uint32_t * dA, * dB, * dSA, * dSB; float * d_out;
        ck(cudaMalloc(&dA, Afr.size() * 4), "dA"); ck(cudaMalloc(&dB, Bfr.size() * 4), "dB");
        ck(cudaMalloc(&dSA, 128), "dSA"); ck(cudaMalloc(&dSB, 128), "dSB"); ck(cudaMalloc(&d_out, 128 * 4), "dout");
        ck(cudaMemcpy(dA, Afr.data(), Afr.size() * 4, cudaMemcpyHostToDevice), "cpyA");
        ck(cudaMemcpy(dB, Bfr.data(), Bfr.size() * 4, cudaMemcpyHostToDevice), "cpyB");
        ck(cudaMemcpy(dSA, sA.data(), 128, cudaMemcpyHostToDevice), "cpySA");
        ck(cudaMemcpy(dSB, sB.data(), 128, cudaMemcpyHostToDevice), "cpySB");
        k_mixdot<V><<<1, 32>>>(dA, dB, dSA, dSB, d_out);
        cudaError_t le = cudaGetLastError(), se = cudaDeviceSynchronize();
        if (le != cudaSuccess || se != cudaSuccess) {
            printf("  %s DOT: FAIL %s / %s\n", name, cudaGetErrorString(le), cudaGetErrorString(se));
            cudaGetLastError();
        } else {
            std::vector<float> got(128);
            ck(cudaMemcpy(got.data(), d_out, 128 * 4, cudaMemcpyDeviceToHost), "cpyout");
            double maxd = 0, maxref = 0;
            for (int i = 0; i < 16; ++i) for (int j = 0; j < 8; ++j) {
                double ref = 0;
                for (int ch = 0; ch < NK / 16; ++ch) {
                    const double sa = (V == 1) ? 1.0 : pow(2.0, (int) sAe[i][ch] - 128);
                    const double sb = (V == 1) ? 1.0 : pow(2.0, (int) sBe[j][ch] - 128);
                    double part = 0;
                    for (int c = 16 * ch; c < 16 * ch + 16; ++c) part += (double) av[i][c] * (double) bv[j][c];
                    ref += sa * sb * part;
                }
                maxd = fmax(maxd, fabs((double) got[i * 8 + j] - ref));
                maxref = fmax(maxref, fabs(ref));
            }
            printf("  %s DOT random+scales: max|gpu-ref| = %.4g  max|ref| = %.6g  -> %s\n",
                   name, maxd, maxref, (maxd <= 1e-3 * maxref + 1e-9) ? "NUMERIC OK" : "MISMATCH");
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(d_out);
    };

    test_onehot(std::integral_constant<int,1>{}, "V1 dense  k16 f16xf16         ", 1);
    test_dot   (std::integral_constant<int,1>{}, "V1 dense  k16 f16xf16         ");
    test_onehot(std::integral_constant<int,2>{}, "V2 mxf8f6f4 e2m1 x e2m1      ", 2);
    test_dot   (std::integral_constant<int,2>{}, "V2 mxf8f6f4 e2m1 x e2m1      ");
    test_onehot(std::integral_constant<int,3>{}, "V3 mxf8f6f4 e4m3 x e2m1      ", 3);
    test_dot   (std::integral_constant<int,3>{}, "V3 mxf8f6f4 e4m3 x e2m1      ");
    test_onehot(std::integral_constant<int,4>{}, "V4 mxf8f6f4 e4m3 x e5m2      ", 4);
    test_dot   (std::integral_constant<int,4>{}, "V4 mxf8f6f4 e4m3 x e5m2      ");
    test_onehot(std::integral_constant<int,5>{}, "V5 mxf8f6f4 e5m2 x e2m1      ", 5);
    test_dot   (std::integral_constant<int,5>{}, "V5 mxf8f6f4 e5m2 x e2m1      ");
#ifdef MIXF16
    test_onehot(std::integral_constant<int,6>{}, "V6 mxf8f6f4 f16  x e2m1      ", 6);
    test_dot   (std::integral_constant<int,6>{}, "V6 mxf8f6f4 f16  x e2m1      ");
    test_onehot(std::integral_constant<int,7>{}, "V7 mxf8f6f4 f16  x e4m3      ", 7);
    test_dot   (std::integral_constant<int,7>{}, "V7 mxf8f6f4 f16  x e4m3      ");
    test_onehot(std::integral_constant<int,8>{}, "V8 mxf8f6f4 e2m1 x f16       ", 8);
    test_dot   (std::integral_constant<int,8>{}, "V8 mxf8f6f4 e2m1 x f16       ");
#endif
    return 0;
}
