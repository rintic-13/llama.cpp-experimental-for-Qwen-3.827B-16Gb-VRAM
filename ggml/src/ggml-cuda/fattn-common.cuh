#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"
#include "turbo-quant.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
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
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

// k_ibase is the linear quantization-block index of the K row inside the cache tensor, i.e.
// (row byte offset) / block size - the same number k_set_rows_turbo3/4 hashed for the subtractive
// dither (see the dither contract in ggml-common.h). Only the dithered turbo types need it.
typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds,
    const int64_t k_ibase);

// v_ibase is the linear quantization-block index of the V row inside the cache tensor, same
// contract as k_ibase above. Only dequantize_V_turbo3_0 / _turbo4_0 consume it.
typedef void (*dequantize_V_t)(const void *, void *, const int64_t, const int64_t v_ibase);

// Bytes per quantization block of a FA vector kernel KV type (F16 / BF16 have no blocks: the
// seed value computed from 1-byte "blocks" is never consumed by those kernels).
template <ggml_type type>
static constexpr __device__ int64_t ggml_cuda_fattn_vec_block_bytes() {
    if constexpr (type == GGML_TYPE_Q4_0) {
        return sizeof(block_q4_0);
    } else if constexpr (type == GGML_TYPE_Q4_1) {
        return sizeof(block_q4_1);
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        return sizeof(block_q5_0);
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        return sizeof(block_q5_1);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        return sizeof(block_q8_0);
    } else if constexpr (type == GGML_TYPE_TURBO3_0) {
        return sizeof(block_turbo3_0);
    } else if constexpr (type == GGML_TYPE_TURBO4_0) {
        return sizeof(block_turbo4_0);
    } else if constexpr (type == GGML_TYPE_TURBO5_0) {
        return sizeof(block_turbo5_0);
    } else if constexpr (type == GGML_TYPE_TURBO3Q_0) {
        return sizeof(block_turbo3q_0);
    } else if constexpr (type == GGML_TYPE_TURBO4Q_0) {
        return sizeof(block_turbo4q_0);
    } else if constexpr (type == GGML_TYPE_NVFP4) {
        return sizeof(block_nvfp4);
    } else {
        return 1;
    }
}

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

// ---- TurboQuant K dot for the FA vector kernel (values stay in the WHT-rotated domain) ----
// K blocks hold 4-bit PolarQuant indices into the 16 Lloyd-Max centroids scaled by a per-block fp16 norm,
// i.e. the same values the f16 fallback dequantizes. Q is q8_1-quantized by the kernel:
// q_i = Q_ds.x * qint_i + Q_ds.y/QK8_1 (group mean). Each thread applies the group mean to its own 4 K
// values, so the QI8_1 words of a q8_1 group sum to mean * sum_group(K) exactly.
// With GGML_TURBO_DITHER the blocks hold (centroid - u) reconstructions, u = turbo_dither(k_ibase + ib, j):
// subtract u from the centroid before both the dot and the group-mean sum, exactly like convert.cu does.
// Requires D % QK_TURBO4 == 0: a 128-element block does not fit in a 64-element row.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t k_ibase) {
    GGML_UNUSED(Q_v);
    GGML_UNUSED(k_ibase);

    const block_turbo4_0 * K_turbo4_0 = (const block_turbo4_0 *) K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / (QK_TURBO4/4);
        const int iqs4 = k_KQ % (QK_TURBO4/4);

        const float norm = __half2float(K_turbo4_0[ib].norm);

        uint16_t nib4;
        ggml_cuda_memcpy_1<sizeof(nib4)>(&nib4, K_turbo4_0[ib].qs + sizeof(nib4)*iqs4);

        const float c0 = TURBO_CENTROIDS_4BIT[(nib4 >>  0) & 0xF];
        const float c1 = TURBO_CENTROIDS_4BIT[(nib4 >>  4) & 0xF];
        const float c2 = TURBO_CENTROIDS_4BIT[(nib4 >>  8) & 0xF];
        const float c3 = TURBO_CENTROIDS_4BIT[(nib4 >> 12) & 0xF];

        const int    u    = Q_q8[k_KQ_0/nthreads];
        const int8_t * q8 = (const int8_t *) &u;
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

#ifdef GGML_TURBO_DITHER
        const int64_t blk = k_ibase + ib;
        // One group hash for the 4-coord word (j = 4*iqs4 + k, so w = j/4 = iqs4 exactly):
        // 4x fewer hashes than the scalar stream - this is the FA decode hot path.
        float du[4];
        turbo_dither4((uint32_t) blk, iqs4, TURBO_DITHER_STEP_4BIT, du);
        const float k0 = c0 - du[0];
        const float k1 = c1 - du[1];
        const float k2 = c2 - du[2];
        const float k3 = c3 - du[3];

        const float doti = q8[0]*k0 + q8[1]*k1 + q8[2]*k2 + q8[3]*k3;
        const float sumc = k0 + k1 + k2 + k3;
#else
        const float doti = q8[0]*c0 + q8[1]*c1 + q8[2]*c2 + q8[3]*c3;
        const float sumc = c0 + c1 + c2 + c3;
#endif

        sum += norm * (Q_ds.x*doti + (Q_ds.y/QK8_1)*sumc);
    }

    return sum;
}

// ---- TurboQuant KQ dot for 5-bit (K in turbo5_0, Q in q8_0 / f16) ----
// 5-bit index: low 4 bits in qs[] (nibble packed), high bit in signs[]
// FIX 2026-10-31: the old loop advanced k_KQ as an ELEMENT index and used only q8[0] of each
// q8_1 word, pairing Q coord 4*k_KQ with K coord k_KQ and covering only D/4 elements (garbage
// scores in the decode/vec path; PPL never hit it because prefill uses the MMA_F16 kernel).
// Now mirrors vec_dot_fattn_vec_KQ_turbo4_0 exactly: one 4-coord q8_1 word per thread/iteration.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {
    GGML_UNUSED(Q_v);

    const block_turbo5_0 * K_turbo5_0 = (const block_turbo5_0 *) K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / (QK_TURBO5/4);
        const int iqs4 = k_KQ % (QK_TURBO5/4);

        const float norm = __half2float(K_turbo5_0[ib].norm);

        float c[4], sumc = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int j = 4*iqs4 + l;
            const uint8_t lo  = (K_turbo5_0[ib].qs   [j / 2] >> ((j % 2) * 4)) & 0xF;
            const uint8_t hi  = (K_turbo5_0[ib].signs[j / 8] >>  (j % 8))      & 0x1;
            c[l] = TURBO_CENTROIDS_5BIT[lo | (hi << 4)] * norm;
            sumc += c[l];
        }

        const int    u    = Q_q8[k_KQ_0/nthreads];
        const int8_t * q8 = (const int8_t *) &u;
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        const float doti = q8[0]*c[0] + q8[1]*c[1] + q8[2]*c[2] + q8[3]*c[3];

        sum += Q_ds.x*doti + (Q_ds.y/QK8_1)*sumc;
    }

    return sum;
}

// ---- TurboQuant KQ dot for the 4-bit Q_prod type (K in turbo4q_0: 3-bit PolarQuant + QJL) ----
// Reconstruction per element: norm * (centroid3[idx] + sign * TURBO_QJL_ALPHA * rnorm).
// FIX 2026-10-31: same k_KQ/q8[0] indexing bug as turbo5_0 above, fixed the same way
// (4 coords per q8_1 word, per vec_dot_fattn_vec_KQ_turbo4_0).
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4q_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {
    GGML_UNUSED(Q_v);

    const block_turbo4q_0 * K_turbo4q = (const block_turbo4q_0 *) K_c;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib   = k_KQ / (QK_TURBO4Q/4);
        const int iqs4 = k_KQ % (QK_TURBO4Q/4);

        const float norm       = __half2float(K_turbo4q[ib].norm);
        const float qjl_scale  = TURBO_QJL_ALPHA * __half2float(K_turbo4q[ib].rnorm);

        float c[4], sumc = 0.0f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int iq = 4*iqs4 + l;
            const uint8_t lo  = (K_turbo4q[ib].qs   [iq / 4] >> ((iq % 4) * 2)) & 0x3;
            const uint8_t hi  = (K_turbo4q[ib].hi1  [iq / 8] >>  (iq % 8))      & 0x1;
            const uint8_t sgn = (K_turbo4q[ib].signs[iq / 8] >>  (iq % 8))      & 0x1;
            c[l] = norm * (TURBO_CENTROIDS_3BIT[lo | (hi << 2)] + (sgn ? qjl_scale : -qjl_scale));
            sumc += c[l];
        }

        const int    u    = Q_q8[k_KQ_0/nthreads];
        const int8_t * q8 = (const int8_t *) &u;
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        const float doti = q8[0]*c[0] + q8[1]*c[1] + q8[2]*c[2] + q8[3]*c[3];

        sum += Q_ds.x*doti + (Q_ds.y/QK8_1)*sumc;
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
#if defined(GGML_USE_HIP)
    // Keep this VMEM read close to its packed-byte dequantization. Hoisting it too far
    // increases VGPR pressure substantially in some FlashAttention vector kernels.
    __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
#endif // defined(GGML_USE_HIP)
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsub4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsub4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

// ---- TurboQuant V dequant for the FA vector kernel (values stay in the WHT-rotated domain) ----

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t v_ibase) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;

    const int64_t ib = i0 / QK_TURBO4;
    const int     j0 = i0 % QK_TURBO4;

    static_assert(ne == 2 || ne == 4, "bad ne");
    const float norm = __half2float(x[ib].norm);

#ifdef GGML_TURBO_DITHER
    // ib is row-relative; the dither was hashed over the LINEAR block index in the cache tensor.
    // Batched group hashes (1 per 4 coords) - FA decode hot path, see turbo_dither_range.
    const uint32_t seed = (uint32_t) (v_ibase + ib);
    float du[4];
    turbo_dither_range(seed, j0, ne, TURBO_DITHER_STEP_4BIT, du);
#endif

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
#ifdef GGML_TURBO_DITHER
            ((half2 *) dst)[l0/2] = make_half2(turbo4_dequant_element_u(x + ib, j0 + l0 + 0, norm, du[l0 + 0]),
                                                turbo4_dequant_element_u(x + ib, j0 + l0 + 1, norm, du[l0 + 1]));
#else
            ((half2 *) dst)[l0/2] = make_half2(turbo4_dequant_element(x + ib, j0 + l0 + 0, norm),
                                                turbo4_dequant_element(x + ib, j0 + l0 + 1, norm));
#endif
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
#ifdef GGML_TURBO_DITHER
            ((float *) dst)[l] = turbo4_dequant_element_u(x + ib, j0 + l, norm, du[l]);
#else
            ((float *) dst)[l] = turbo4_dequant_element(x + ib, j0 + l, norm);
#endif
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_turbo5_0 * x = (const block_turbo5_0 *) vx;

    const int64_t ib = i0 / QK_TURBO5;
    const int     j0 = i0 % QK_TURBO5;

    static_assert(ne == 2 || ne == 4, "bad ne");
    const float norm = __half2float(x[ib].norm);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = make_half2(turbo5_dequant_element(x + ib, j0 + l0 + 0, norm),
                                                turbo5_dequant_element(x + ib, j0 + l0 + 1, norm));
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = turbo5_dequant_element(x + ib, j0 + l, norm);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t v_ibase) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;

    const int64_t ib = i0 / QK_TURBO3;
    const int     j0 = i0 % QK_TURBO3;

    static_assert(ne == 2 || ne == 4, "bad ne");
    const float norm = __half2float(x[ib].norm);

#ifdef GGML_TURBO_DITHER
    // ib is row-relative; the dither was hashed over the LINEAR block index in the cache tensor.
    // Batched group hashes (1 per 4 coords) - FA decode hot path, see turbo_dither_range.
    const uint32_t seed = (uint32_t) (v_ibase + ib);
    float du[4];
    turbo_dither_range(seed, j0, ne, TURBO_DITHER_STEP_3BIT, du);
#endif

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
#ifdef GGML_TURBO_DITHER
            ((half2 *) dst)[l0/2] = make_half2(turbo3_dequant_element_u(x + ib, j0 + l0 + 0, norm, du[l0 + 0]),
                                               turbo3_dequant_element_u(x + ib, j0 + l0 + 1, norm, du[l0 + 1]));
#else
            ((half2 *) dst)[l0/2] = make_half2(turbo3_dequant_element(x + ib, j0 + l0 + 0, norm),
                                                turbo3_dequant_element(x + ib, j0 + l0 + 1, norm));
#endif
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
#ifdef GGML_TURBO_DITHER
            ((float *) dst)[l] = turbo3_dequant_element_u(x + ib, j0 + l, norm, du[l]);
#else
            ((float *) dst)[l] = turbo3_dequant_element(x + ib, j0 + l, norm);
#endif
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

// ---- V dequant for the Q_prod types (see ggml-common.h; 3-bit or 2-bit codebook + QJL sign) ----

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3q_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_turbo3q_0 * x = (const block_turbo3q_0 *) vx;

    const int64_t ib = i0 / QK_TURBO3Q;
    const int     j0 = i0 % QK_TURBO3Q;

    static_assert(ne == 2 || ne == 4, "bad ne");
    const float norm = __half2float(x[ib].norm);
    const float qjl_scale = TURBO_QJL_ALPHA * __half2float(x[ib].rnorm);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = make_half2(turbo3q_dequant_element(x + ib, j0 + l0 + 0, norm, qjl_scale),
                                                turbo3q_dequant_element(x + ib, j0 + l0 + 1, norm, qjl_scale));
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = turbo3q_dequant_element(x + ib, j0 + l, norm, qjl_scale);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4q_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_turbo4q_0 * x = (const block_turbo4q_0 *) vx;

    const int64_t ib = i0 / QK_TURBO4Q;
    const int     j0 = i0 % QK_TURBO4Q;

    static_assert(ne == 2 || ne == 4, "bad ne");
    const float norm = __half2float(x[ib].norm);
    const float qjl_scale = TURBO_QJL_ALPHA * __half2float(x[ib].rnorm);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = make_half2(turbo4q_dequant_element(x + ib, j0 + l0 + 0, norm, qjl_scale),
                                                turbo4q_dequant_element(x + ib, j0 + l0 + 1, norm, qjl_scale));
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = turbo4q_dequant_element(x + ib, j0 + l, norm, qjl_scale);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

// ---- NVFP4-like KV cache (Stage A, single-level: per-16 UE4M3 scale + packed E2M1 nibbles) ----
// No dither, no rotation - the encoder writes plain (unrotated) rows, max/6 scale policy.
// block_nvfp4 (64 elements): d[4] UE4M3 sub-block scales, qs[32] packed E2M1. Nibble code n:
// value = kvalues_fp4[n] * d_sub  (kvalues_fp4 are doubled E2M1, d_sub is UE4M3 * 0.5).
// K side of the vec kernel uses this scalar form; the fused KQ MMA path and dequantize_V_nvfp4
// below use the PRMT register decoder (ggml_cuda_e2m1_pair_f16) instead.

// E2M1 nibble pair -> packed half2 bits, table-free (magnitude high bytes 00,38,3C,3E,40,42,44,46
// for values 0,.5,1,1.5,2,3,4,6 with always-zero f16 low byte; sign from nibble bit 3).  PRMT
// avoids constant-cache serialization on divergent nibble indices (probe: 2-5x decode speedup).
// Operands may be whole BYTES: only bits 0..3 (n0) / 0..3 (n1) are consumed, bit 3 = sign.
// Bit-exact vs kvalues_fp4 LUT magnitudes: E2M1 x UE4M3 products have <= 6 significant mantissa
// bits, exact in f16, so f16-domain results equal the f32 LUT path rounded to f16.
static __device__ __forceinline__ uint32_t ggml_cuda_e2m1_pair_f16(const uint32_t n0, const uint32_t n1) {
    const uint32_t tl = 0x3E3C3800u;   // T0..T3
    const uint32_t th = 0x46444240u;   // T4..T7
    const uint32_t sel = ((n0 & 7u) << 4) | ((n1 & 7u) << 12);
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(tl), "r"(th), "r"(sel));
    return r | ((n0 & 8u) << 12) | ((n1 & 8u) << 28);
}

// NB: must NOT be guarded on GGML_USE_CUDA - that macro is only defined for the HOST side
// (ggml-backend-reg.cpp), never inside the ggml-cuda CUDA-file compiles. __CUDACC__ is what
// actually identifies nvcc kernel compilation here. (SASS forensics 2026-11-02: with the old
// guard the DLL contained 0x E2M1 SASS opcodes; the "HW cvt" path had never been active.)
#if defined(__CUDACC__) && !defined(GGML_USE_HIP) && defined(FP8_AVAILABLE) && CUDA_VERSION >= 12080 \
    && !defined(GGML_CUDA_FORCE_E2M1_PRMT)
#include <cuda_fp4.h>
#define GGML_CUDA_E2M1_HW_CVT
#endif

// Packed e2m1 byte (low nibble = elem0, high nibble = elem1) -> packed half2 bits via the HW
// cvt.rn.f16x2.e2m1x2 instruction (SM100+/SM120; the CUDA header emits it verbatim, PTX checked).
// Bitwise-identical to ggml_cuda_e2m1_pair_f16 for every input (probe fa_nvfp4_cvt_probe:
// 256/256 exact, incl. -0 for E2M1 code 8) but one instruction instead of ~5.
static __device__ __forceinline__ uint32_t ggml_cuda_e2m1x2_f16(const uint32_t packed_byte) {
#ifdef GGML_CUDA_E2M1_HW_CVT
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t) (packed_byte & 0xFFu), __NV_E2M1);
    return *reinterpret_cast<const uint32_t *>(&h);
#else
    return ggml_cuda_e2m1_pair_f16(packed_byte & 0xFu, (packed_byte >> 4) & 0xFu);
#endif
}

// UE4M3 scale byte -> broadcast half2 holding the raw e4m3 value as f16.
// The e4m3->f16 conversion is EXACT (all e4m3 values including subnormals are representable in
// f16 without rounding), so this is bitwise-identical to the f32 path
// __half2half2(__float2half(2.0f * ggml_cuda_ue4m3_to_fp32(b))) - note 2*ue4m3_to_fp32(b) == the
// raw e4m3 value. The NaN->0 flush of ggml_cuda_ue4m3_to_fp32 is preserved explicitly.
// dword arg: the 4 scale bytes of a block loaded as one u32, b = byte index 0..3.
#if !defined(GGML_USE_HIP) && defined(FP8_AVAILABLE)
#include <cuda_fp8.h>
#endif
static __device__ __forceinline__ half2 ggml_cuda_ue4m3_dword_h2(const uint32_t dwords, const int b) {
#if !defined(GGML_USE_HIP) && defined(FP8_AVAILABLE)
    uint32_t byte = (dwords >> (8*b)) & 0xFFu;
    byte = ((byte & 0x7Fu) == 0x7Fu) ? 0u : byte;   // NaN -> 0, same as ue4m3_to_fp32
    const uint32_t packed = byte | (byte << 8);     // e4m3x2 storage: both lanes the same byte
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) packed, __NV_E4M3);
    return *reinterpret_cast<const half2 *>(&h);
#else
    return __half2half2(__float2half(2.0f * ggml_cuda_ue4m3_to_fp32((uint8_t) ((dwords >> (8*b)) & 0xFFu))));
#endif
}

// Sub-block s of a 16-element group packs 8 bytes: qs[s*8 + c]  holds coord c (low nibble) and
// coord c+8 (high nibble) of the group.
// K dot layout: one 4-coord q8_1 word per thread/iteration (k_KQ), exactly like q5_1/q4_0:
//   ib  = k_KQ / 16  (QK_NVFP4/4 words per 64-element block)
//   iw  = k_KQ % 16  (word within block)
//   sub = iw / 4     (4 words per sub-block)
// The 4 coords of a word never cross a sub-block boundary (4-aligned within a 16-group), so one
// scale is used per word. Bytes needed for the word: window qs[base + (iw&1)*4 .. +3], with
// base = sub*8; low nibbles when (iw&2)==0, high nibbles otherwise.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_nvfp4(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v,
    const int64_t /* k_ibase */) {
    GGML_UNUSED(Q_v);

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

        // code values only (kvalues_fp4 are doubled E2M1); the single UE4M3 scale of this
        // sub-block factors out of both the dot and the group-mean term (turbo4 pattern):
        // sum += d_sub * (Q_ds.x * sum(q8*code) + (Q_ds.y/QK8_1) * sum(code))
        // Register decode instead of the kvalues_fp4 constant-LUT gather (divergent constant
        // loads serialized the loop - probe fa_vec_ab_probe: nvfp4 KQ was ~1.6x q4_1 KQ):
        //   u4 = table{0,1,2,3,4,6,8,12}[code&7] via one PRMT, sign applied with SIMD
        //   mask/negate. All integers, so doti/sumc equal the old int8 expressions EXACTLY
        //   (small exact sums), keeping this dot bitwise-identical (kvalues_fp4[8] = +0 maps
        //   to -0 -> 0 under two's-complement negate, same value).
        const int      sh  = (iw & 2) ? 4 : 0;
        const uint32_t bs  = b >> sh;                                  // this word's nibbles at bits 0..3
        const uint32_t sel = bs & 0x07070707u;                         // magnitude index per byte
        const uint32_t sn4 = bs & 0x08080808u;                         // sign bit per byte
        const uint32_t u4  = __byte_perm(0x03020100u, 0x0C080604u, sel); // magnitudes 0,1,2,3 | 4,6,8,12
        const uint32_t m4  = __vcmpgts4((int) sn4, 0);                   // 0xFF where negative
        const uint32_t v4  = __vsub4(u4 ^ m4, m4);                     // signed byte lanes = kv values

        const uint32_t sc2 = __vadd4(v4, v4 >> 16);
        const uint32_t sc4 = __vadd4(sc2, sc2 >> 8);
        const int      sumc = (int) (int8_t) (sc4 & 0xFFu);

        const float doti = (float) __dp4a(u, (int) v4, 0);

        sum += d_sub*(Q_ds.x*doti + (Q_ds.y/QK8_1)*(float)sumc);
    }

    return sum;
}

// NVFP4 V dequant for the FA vector kernel. Each thread handles ne consecutive coordinates of a
// V row; ne = 2 or 4 and the starting coordinate is 4-aligned, so a thread never crosses a
// sub-block boundary.
// (The old scalar kv_fp4_val LUT helper was replaced by the PRMT register decoder below -
//  kvalues_fp4 divergent constant-cache loads were the nvfp4 V-side bottleneck.)
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_nvfp4(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0, const int64_t /* v_ibase */) {
    const block_nvfp4 * x = (const block_nvfp4 *) vx;

    static_assert(ne == 2 || ne == 4, "bad ne");

    const int64_t ib = i0 / QK_NVFP4;
    const int     c0 = i0 % QK_NVFP4;

    const int sub = c0 / QK_NVFP4_SUB;
    const int   o = c0 % QK_NVFP4_SUB;   // 0,4 for ne=4; any 4-aligned for ne=2

    // The ne coords of this call live in the low (o<8) or high (o>=8) nibbles of ne consecutive
    // bytes.  One register load replaces ne byte loads; for the high nibbles a single word shift
    // moves them to bits 0..3 of each byte (the garbage entering the upper nibbles is never read:
    // ggml_cuda_e2m1_pair_f16 consumes bits 0..3 only, bit 3 = sign).
    // Load width = ne bytes: ne=4 implies o%8 in {0,4}, ne=2 implies o%8 in {0,2,4,6}, so the
    // window never crosses the 32-byte qs[] end of the block.
    typedef std::conditional_t<(ne >= 4), uint32_t, uint16_t> nvfp4_word_t;
    nvfp4_word_t w;
    ggml_cuda_memcpy_1<sizeof(w)>(&w, x[ib].qs + sub*(QK_NVFP4_SUB/2) + (o & 7));
    if (o >= 8) {
        w >>= 4;
    }

    // 2.0f * UE4M3 = the raw scale byte as fp32; e2m1 x e4m3 products have <= 6 significant
    // mantissa bits -> exact in f16, making this bit-identical to the old scalar LUT path
    // ((float) kvalues_fp4[nib] * d_sub rounded to half), see the comment on e2m1_pair_f16.
    const half2 sh = ggml_cuda_ue4m3_dword_h2(*(const uint32_t *) &x[ib].d[0], sub);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            const uint32_t s = w >> (8*l0);
            const uint32_t bits = ggml_cuda_e2m1x2_f16((s & 0x0Fu) | ((s >> 4) & 0xF0u));
            ((half2 *) dst)[l0/2] = __hmul2(*reinterpret_cast<const half2 *>(&bits), sh);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            const uint32_t s = w >> (8*l0);
            const uint32_t bits = ggml_cuda_e2m1x2_f16((s & 0x0Fu) | ((s >> 4) & 0xF0u));
            ((float2 *) dst)[l0/2] = __half22float2(__hmul2(*reinterpret_cast<const half2 *>(&bits), sh));
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO4_0) {
        return vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO5_0) {
        return vec_dot_fattn_vec_KQ_turbo5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO4Q_0) {
        return vec_dot_fattn_vec_KQ_turbo4q_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_NVFP4) {
        return vec_dot_fattn_vec_KQ_nvfp4<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {
        return dequantize_V_turbo4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO5_0) {
        return dequantize_V_turbo5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3_0) {
        return dequantize_V_turbo3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3Q_0) {
        return dequantize_V_turbo3q_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO4Q_0) {
        return dequantize_V_turbo4q_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_NVFP4) {
        return dequantize_V_nvfp4<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream);

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const bool use_sparse,
    const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V);

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(f16_extra.V != 0);
            half * V_f16 = (half *) f16_extra.V;
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // sparse: a query tile of ncols1 queries shares one index list, the union of the queries' visible columns
    int32_t n_kv_max = 0;
    if (use_sparse) {
        GGML_ASSERT(mask != nullptr);
        const int32_t n_kv_max_query = ggml_get_op_params_i32(KQV, 4);
        GGML_ASSERT(n_kv_max_query > 0);
        n_kv_max = std::min<int64_t>(K->ne[1], int64_t(ncols1)*n_kv_max_query);

        const size_t n_lists = size_t(ntiles_x) * mask->ne[3];

        KV_max.alloc(size_t(n_kv_max)*n_lists + n_lists);
        ggml_cuda_flash_attn_ext_compact_mask(mask, KV_max.ptr, KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], ncols1, n_kv_max, main_stream);
    }

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (!use_sparse && mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int64_t n_kv = use_sparse ? n_kv_max : K->ne[1];
    const int ntiles_KV = (n_kv + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        auto should_use_stream_k = [](const int cc, const int ntiles_dst, const int max_blocks, const int DKQ) {
            const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
            const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

            if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return true;
            }
            if (amd_wmma_available(cc) && DKQ == 64) {
                return true; // TODO better configuration
            }
            return tiles_efficiency_percent < 75;
        };

        const int  max_blocks   = max_blocks_per_sm*nsm;
        const bool use_stream_k = should_use_stream_k(cc, ntiles_dst, max_blocks, Q->ne[0]);

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], n_kv, K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
