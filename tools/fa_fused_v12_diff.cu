// fa_fused_v12_diff.cu - bitwise A/B of the two fused-KQ decode maps (production v1 smem-staging
// vs v2 register/byte_perm) + fp64 reference closeness.  Motivation: PPL of the fused path moved
// 7.1417 (v1) -> 7.1607 (v2) although the intended math is identical; PPL is deterministic, so the
// f16 operands fed to the MMA must actually differ somewhere.  The probe-validated map (§10) was
// the smem style (v1); the v2 byte_perm extraction was never checked standalone.
//
// Build:  nvcc -O3 --generate-code=arch=compute_120a,code=[sm_120a] tools\fa_fused_v12_diff.cu -o build...\bin\fa_fused_v12_diff.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

// ggml nvfp4 block: d[4] ue4m3 bytes + qs[32], 16-coord subblock s: byte b = coords (b, b+8)
struct blk { uint8_t d[4]; uint8_t qs[32]; };

static __device__ __forceinline__ float ue4m3_to_fp32_(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF);
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2.0f;
}
static __device__ __forceinline__ uint32_t e2m1_pair_f16_(const uint32_t n0, const uint32_t n1) {
    const uint32_t tl = 0x3E3C3800u, th = 0x46444240u;
    const uint32_t sel = ((n0 & 7u) << 4) | ((n1 & 7u) << 12);
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(tl), "r"(th), "r"(sel));
    return r | ((n0 & 8u) << 12) | ((n1 & 8u) << 28);
}
static __device__ __forceinline__ half2 dec_scale2(const blk * K_b, int sub) {
    return __half2half2(__float2half_rn(2.0f * ue4m3_to_fp32_(K_b[sub >> 2].d[sub & 3])));
}

// ---- decode array dumps: write exact f16 values the two paths feed into the MMA, per (row, coord)
// v1 map: word w (coords 2w,2w+1) from low nibbles of bytes 2w,2w+1 (w<4) / high nibbles of bytes 2(w&3).. (w>=4)
__global__ void dump_v1(const blk * K, float * out /*[16][256]*/) {
    __shared__ int Kh[16*(256/2+1)];
    const int pr = threadIdx.x & 15, r = threadIdx.x >> 4; // 8 rows per pass, 2 passes
    for (int rr = r; rr < 16; rr += 2) {
        const blk * K_b = K + rr*4;
        const uint32_t w0 = *(const uint32_t *) &K_b[pr >> 2].qs[8*(pr & 3) + 0];
        const uint32_t w1 = *(const uint32_t *) &K_b[pr >> 2].qs[8*(pr & 3) + 4];
        const half2 sh = dec_scale2(K_b, pr);
        int * dst = Kh + rr*(256/2+1) + 8*pr;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int sel = w & 3;
            const uint32_t by0 = (sel < 2) ? ((w0 >> (16*sel)) & 0xFFu) : ((w1 >> (16*(sel - 2))) & 0xFFu);
            const uint32_t by1 = (sel < 2) ? ((w0 >> (16*sel + 8)) & 0xFFu) : ((w1 >> (16*(sel - 2) + 8)) & 0xFFu);
            const uint32_t n0 = (w < 4) ? (by0 & 0xFu) : (by0 >> 4);
            const uint32_t n1 = (w < 4) ? (by1 & 0xFu) : (by1 >> 4);
            const uint32_t bits = e2m1_pair_f16_(n0, n1);
            half2 hv = *reinterpret_cast<const half2 *>(&bits);
            hv = __hmul2(hv, sh);
            dst[w] = (int) *reinterpret_cast<uint32_t *>(&hv);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16*256; i += blockDim.x) {
        const int rr = i / 256, c = i % 256;
        const int word = Kh[rr*(256/2+1) + (c >> 1)];
        half2 hv = *reinterpret_cast<const half2 *>(&word);
        out[rr*256 + c] = __half2float((c & 1) ? hv.y : hv.x);
    }
}
// v2 map: lane-owned nibble pairs from __byte_perm-selected bytes 2tg,2tg+1 of the subblock
__global__ void dump_v2(const blk * K, float * out) {
    // emulate 32-lane fragment ownership with 32 threads
    for (int rr = 0; rr < 16; ++rr) {
        for (int sub = 0; sub < 16; ++sub) {
            const blk * K_b = K + rr*4;
            const uint32_t W0 = *(const uint32_t *) &K_b[sub >> 2].qs[8*(sub & 3) + 0];
            const uint32_t W1 = *(const uint32_t *) &K_b[sub >> 2].qs[8*(sub & 3) + 4];
            const half2 sh = dec_scale2(K_b, sub);
            const uint32_t s  = __byte_perm(W0, W1, (2*threadIdx.x & 7) | (((2*threadIdx.x & 7) + 1) << 4)); // tx 0..3 -> tg
            if (threadIdx.x < 4) {
                const uint32_t a = e2m1_pair_f16_( s        & 0xFu, (s >> 8) & 0xFu); // coords 2tg,2tg+1
                const uint32_t b = e2m1_pair_f16_((s >> 4)  & 0xFu, (s >> 12) & 0xFu); // coords 8+2tg,9+2tg
                half2 ha = __hmul2(*reinterpret_cast<const half2 *>(&a), sh);
                half2 hb = __hmul2(*reinterpret_cast<const half2 *>(&b), sh);
                const int c0 = 16*sub + 2*(int)threadIdx.x;
                out[rr*256 + c0 + 0] = __half2float(ha.x); out[rr*256 + c0 + 1] = __half2float(ha.y);
                out[rr*256 + c0 + 8] = __half2float(hb.x); out[rr*256 + c0 + 9] = __half2float(hb.y);
            }
        }
    }
}

// ---- score paths: one warp, one m16 tile, 8 query cols
__global__ void score_v1(const blk * K, const float * Q, float * out /*[8][16]*/) {
    __shared__ int Qh[8*(256/2+1)], Kh[16*(256/2+1)];
    for (int j = 0; j < 8; ++j) {
        for (int w = threadIdx.x; w < 128; w += 32)
            { half2 h = __floats2half2_rn(Q[j*256 + 2*w], Q[j*256 + 2*w + 1]); Qh[j*(256/2+1) + w] = (int) *(uint32_t *) &h; }
    }
    const int pr = threadIdx.x & 15;
    for (int rr = threadIdx.x >> 4; rr < 16; rr += 2) {
        const blk * K_b = K + rr*4;
        const uint32_t w0 = *(const uint32_t *) &K_b[pr >> 2].qs[8*(pr & 3) + 0];
        const uint32_t w1 = *(const uint32_t *) &K_b[pr >> 2].qs[8*(pr & 3) + 4];
        const half2 sh = dec_scale2(K_b, pr);
        int * dst = Kh + rr*(256/2+1) + 8*pr;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int sel = w & 3;
            const uint32_t by0 = (sel < 2) ? ((w0 >> (16*sel)) & 0xFFu) : ((w1 >> (16*(sel - 2))) & 0xFFu);
            const uint32_t by1 = (sel < 2) ? ((w0 >> (16*sel + 8)) & 0xFFu) : ((w1 >> (16*(sel - 2) + 8)) & 0xFFu);
            const uint32_t n0 = (w < 4) ? (by0 & 0xFu) : (by0 >> 4);
            const uint32_t n1 = (w < 4) ? (by1 & 0xFu) : (by1 >> 4);
            const uint32_t bits = e2m1_pair_f16_(n0, n1);
            half2 hv = *reinterpret_cast<const half2 *>(&bits);
            hv = __hmul2(hv, sh);
            dst[w] = (int) *reinterpret_cast<uint32_t *>(&hv);
        }
    }
    __syncwarp();
    const int g = threadIdx.x >> 2, tg = threadIdx.x & 3;
    float Dt[4] = {0,0,0,0};
#pragma unroll
    for (int f = 0; f < 16; ++f) {
        uint32_t A[4], Bfr[2];
        A[0] = (uint32_t) Kh[(g + 0)*(256/2+1) + f*8 + tg + 0];
        A[1] = (uint32_t) Kh[(g + 8)*(256/2+1) + f*8 + tg + 0];
        A[2] = (uint32_t) Kh[(g + 0)*(256/2+1) + f*8 + tg + 4];
        A[3] = (uint32_t) Kh[(g + 8)*(256/2+1) + f*8 + tg + 4];
        Bfr[0] = (uint32_t) Qh[g*(256/2+1) + f*8 + tg + 0];
        Bfr[1] = (uint32_t) Qh[g*(256/2+1) + f*8 + tg + 4];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(Dt[0]), "+f"(Dt[1]), "+f"(Dt[2]), "+f"(Dt[3])
                     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(Bfr[0]), "r"(Bfr[1]));
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int j = 2*tg + e;
        out[j*16 + g + 0] = Dt[0 + e];
        out[j*16 + g + 8] = Dt[2 + e];
    }
}
__global__ void score_v2(const blk * K, const float * Q, float * out) {
    const int g = threadIdx.x >> 2, tg = threadIdx.x & 3;
    uint32_t Bfr[16][2];
#pragma unroll
    for (int f = 0; f < 16; ++f)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int c = 16*f + 8*e + 2*tg;
            half2 h = __floats2half2_rn(Q[g*256 + c], Q[g*256 + c + 1]);
            Bfr[f][e] = (uint32_t) *(const uint32_t *) &h;
        }
    const uint32_t sq = (2*tg) | ((2*tg + 1) << 4);
    float Dt[4] = {0,0,0,0};
#pragma unroll
    for (int f = 0; f < 16; ++f) {
        const blk * K_r0 = K + (g + 0)*4 + (f >> 2);
        const blk * K_r1 = K + (g + 8)*4 + (f >> 2);
        const int sb = 8*(f & 3);
        const uint32_t s0 = __byte_perm(*(const uint32_t *) &K_r0->qs[sb + 0], *(const uint32_t *) &K_r0->qs[sb + 4], sq);
        const uint32_t s1 = __byte_perm(*(const uint32_t *) &K_r1->qs[sb + 0], *(const uint32_t *) &K_r1->qs[sb + 4], sq);
        const half2 sh0 = dec_scale2(K + (g + 0)*4, f);
        const half2 sh1 = dec_scale2(K + (g + 8)*4, f);
        const uint32_t c0 = e2m1_pair_f16_( s0        & 0xFu, (s0 >> 8) & 0xFu);
        const uint32_t c2 = e2m1_pair_f16_((s0 >> 4)  & 0xFu, (s0 >> 12) & 0xFu);
        const uint32_t c1 = e2m1_pair_f16_( s1        & 0xFu, (s1 >> 8) & 0xFu);
        const uint32_t c3 = e2m1_pair_f16_((s1 >> 4)  & 0xFu, (s1 >> 12) & 0xFu);
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
        out[j*16 + g + 0] = Dt[0 + e];
        out[j*16 + g + 8] = Dt[2 + e];
    }
}

// scalar reference: per (row,coord) dequant + direct dot, no fragment maps involved
__global__ void score_scalar(const blk * K, const float * Q, float * out) {
    if (threadIdx.x >= 16) return;
    for (int j = 0; j < 8; ++j) {
        float acc = 0.f;
        for (int c = 0; c < 256; ++c) {
            const blk & B = K[threadIdx.x*4 + c/64];
            const int sub = (c / 16) & 3, cc = c % 16;
            const uint8_t code = (cc < 8) ? (B.qs[8*sub + cc] & 0xF) : (B.qs[8*sub + (cc - 8)] >> 4);
            static const float m[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
            const float ev = (code & 8) ? -m[code & 7] : m[code & 7];
            const float sc = 2.0f * ue4m3_to_fp32_(B.d[sub]);
            const float kh = __half2float(__float2half_rn(ev * sc));
            const float qh = __half2float(__float2half_rn(Q[j*256 + c]));
            acc += qh * kh;
        }
        out[j*16 + threadIdx.x] = acc;
    }
}

static double e2m1_host(uint8_t c) {
    static const double m[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return (c & 8) ? -m[c & 7] : m[c & 7];
}

int main() {
    srand(12345);
    std::vector<blk> Kh(16);
    std::vector<float> Qh(8*256);
    for (auto & b : Kh) for (int i = 0; i < 32; ++i) b.qs[i] = rand() & 0xFF;
    for (auto & b : Kh) for (int i = 0; i < 4;  ++i) b.d[i]  = rand() & 0x7E;   // avoid NaN encodings
    for (auto & q : Qh) q = (rand() / (double) RAND_MAX - 0.5) * 8.0;

    blk * Kd; float * Qd, * outd; CK(cudaMalloc(&Kd, sizeof(blk)*16)); CK(cudaMalloc(&Qd, 8*256*4));
    CK(cudaMemcpy(Kd, Kh.data(), sizeof(blk)*16, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Qd, Qh.data(), 8*256*4, cudaMemcpyHostToDevice));
    float *d1v, *d2v, *s1v, *s2v, *s3v;
    CK(cudaMalloc(&d1v, 16*256*4)); CK(cudaMalloc(&d2v, 16*256*4)); CK(cudaMalloc(&s1v, 8*16*4)); CK(cudaMalloc(&s2v, 8*16*4)); CK(cudaMalloc(&s3v, 8*16*4));

    dump_v1<<<1, 32>>>(Kd, d1v); dump_v2<<<1, 32>>>(Kd, d2v);
    score_v1<<<1, 32>>>(Kd, Qd, s1v); score_v2<<<1, 32>>>(Kd, Qd, s2v); score_scalar<<<1, 32>>>(Kd, Qd, s3v);
    CK(cudaDeviceSynchronize());

    std::vector<float> a(16*256), b(16*256);
    CK(cudaMemcpy(a.data(), d1v, 16*256*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), d2v, 16*256*4, cudaMemcpyDeviceToHost));
    int ndiff = 0; printf("first decode-array mismatches (row, coord): ");
    for (int i = 0; i < 16*256; ++i) {
        uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua != ub) { if (ndiff < 8) printf("(%d,%d) v1=%.6g v2=%.6g  ", i/256, i%256, a[i], b[i]); ++ndiff; }
    }
    printf("\ndecode array: %d / %d bitwise mismatches\n", ndiff, 16*256);

    // reference: fp64 exact sum of f16(Q) x (e2m1 x e4m3fn-scale)
    std::vector<float> sa(8*16), sb(8*16), sc3(8*16);
    CK(cudaMemcpy(sa.data(), s1v, 8*16*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(sb.data(), s2v, 8*16*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sc3.data(), s3v, 8*16*4, cudaMemcpyDeviceToHost));
    double e_sc_v1 = 0, e_sc_v2 = 0;
    for (int i = 0; i < 8*16; ++i) {
        double a = sa[i], b = sb[i], r = sc3[i];
        e_sc_v1 += (a-r)*(a-r); e_sc_v2 += (b-r)*(b-r);
    }
    printf("score rms vs SCALAR ref: v1=%.4g v2=%.4g\n", sqrt(e_sc_v1/(8*16)), sqrt(e_sc_v2/(8*16)));
    int sdiff = 0; printf("first score mismatches (j, tok): ");
    for (int i = 0; i < 8*16; ++i) {
        if (memcmp(&sa[i], &sb[i], 4) != 0) { if (sdiff < 8) printf("(%d,%d) v1=%.6g v2=%.6g ", i/16, i%16, sa[i], sb[i]); ++sdiff; }
    }
    printf("\nscores: %d / %d bitwise mismatches\n", sdiff, 8*16);

    double err1 = 0, err2 = 0, maxabs = 0;
    for (int j = 0; j < 8; ++j) for (int t = 0; t < 16; ++t) {
        double ref = 0;
        for (int c = 0; c < 256; ++c) {
            half qh = __float2half_rn(Qh[j*256 + c]);
            const blk & B = Kh[t*4 + c / 64]; const int sub = (c / 16) & 3, cc = c % 16;
            const uint8_t code = (cc < 8) ? (B.qs[8*sub + cc] & 0xF) : (B.qs[8*sub + (cc - 8)] >> 4);
            // NOTE: host ref follows ggml coord order (byte cc = coords cc, cc+8 within the 16-subblock)
            __nv_fp8_e4m3 sfd; uint8_t db = B.d[sub]; uint32_t bits = db * (db != 0x7F && db != 0xFF);
            memcpy(&sfd, &bits, 1);
            double scale = (double)(float) sfd; // ue4m3 = e4m3fn * 0.5; cache uses 2*ue4m3 = e4m3fn raw
            double kv = e2m1_host(code) * 2.0 * ((double)(float) sfd / 2.0);
            ref += (double) qh * kv; (void) scale;
        }
        double d1 = fabs(sa[j*16 + t] - ref), d2 = fabs(sb[j*16 + t] - ref);
        err1 += d1*d1; err2 += d2*d2; if (fabs(ref) > maxabs) maxabs = fabs(ref);
    }
    printf("score rms vs fp64-ref: v1=%.6g  v2=%.6g  (max|ref|=%.4g)\n",
           sqrt(err1/(8*16)), sqrt(err2/(8*16)), maxabs);
    printf("%s\n", (ndiff == 0 && sdiff == 0) ? "IDENTICAL: v1 and v2 feed bitwise equal fragments and scores"
                                              : "MISMATCH FOUND: v1 and v2 differ - one map is wrong");
    return 0;
}
