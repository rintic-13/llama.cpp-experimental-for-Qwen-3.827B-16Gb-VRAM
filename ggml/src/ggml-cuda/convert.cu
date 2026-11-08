#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>
#include <type_traits>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d * (q[l] & 0xF) + dm);
        y[l+16] = ggml_cuda_cast<dst_t>(d * (q[l] >>  4) + dm);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = ggml_cuda_cast<dst_t>(d.x * (q[l] >>  4) + d.y);
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q2_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q3_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q5_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q6_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_m(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;

    const int64_t ib = 8*i + threadIdx.x%8;
    if (ib >= nb32) {
        return;
    }

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_mxfp4(vx, i, yy + i*QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

// ---- turbo5 dequant (2026-11-08 per-block sign masks) ----
// Each warp owns whole 128-coord blocks (lane l covers the block-relative coordinates
// j = 4*l + a, i.e. qs bytes 2*l, 2*l+1 and signs byte l/2) and applies the block's inverse
// FHT G_b^-1 with the seed stored in the block, producing RAW-domain output.  The FHT runs as
// a warp shuffle tree exactly like dequantize_V_turbo5_0 in fattn-common.cuh (see the pairing
// proof there).
// This kernel replaces dequantize_block_cont_cuda<..., dequantize_turbo5_0> in all three
// to_fp conversions and in the rowwise (from_to) case; the mmfa/vec FA f16 scratch staging and
// ggml_get_rows fallbacks consume raw K/V, matching the FA kernels and the graph (no rotation
// chain for turbo5).
template <typename dst_t>
static __global__ void dequantize_block_turbo5_cont(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t nb) {
    static_assert(QK_TURBO5 == 128, "turbo5 dequant kernel assumes QK_TURBO5 == 128");
    const int64_t ib  = blockIdx.x*(int64_t)(blockDim.x/WARP_SIZE) + (threadIdx.x/WARP_SIZE);
    const int     lane = threadIdx.x % WARP_SIZE;

    if (ib >= nb) {
        return;
    }

    const block_turbo5_0 * blk = (const block_turbo5_0 *) vx + ib;
    const float norm = __half2float(blk->norm);
    const uint16_t seed = blk->seed;

    // lane owns the block coords j = 4*lane + a (a=0..3): qs bytes 2*lane / 2*lane+1 hold them,
    // signs byte lane/2 carries their high bits at position 4*(lane&1)+a.  Every qs byte is read
    // once per warp, every signs byte by two lanes - no shuffle needed for the unpack.
    const uint8_t q0 = blk->qs[2*lane + 0];
    const uint8_t q1 = blk->qs[2*lane + 1];
    const uint8_t sg = blk->signs[lane >> 1] >> (4*(lane & 1));

    float v[4];
    v[0] = turbo5_centroid((uint8_t) ((q0      & 0xF) | ((sg      & 1) << 4))) * norm;
    v[1] = turbo5_centroid((uint8_t) ((q0      >>  4) | (((sg >> 1) & 1) << 4))) * norm;
    v[2] = turbo5_centroid((uint8_t) ((q1      & 0xF) | (((sg >> 2) & 1) << 4))) * norm;
    v[3] = turbo5_centroid((uint8_t) ((q1      >>  4) | (((sg >> 3) & 1) << 4))) * norm;

    // inverse FHT: h=1,2 in-lane (j = 4*lane + a), h=4,8,16,32,64 via lane ^ m (m = h/4)
    for (int l = 0; l < 4; ++l) {
        v[l] *= ggml_turbo5_sign2(seed, 4*lane + l);
    }
    {   // h=1: pairs (4l, 4l+1), (4l+2, 4l+3)
        const float a0 = v[0], b0 = v[1], a2 = v[2], b2 = v[3];
        v[0] = a0 + b0; v[1] = a0 - b0;
        v[2] = a2 + b2; v[3] = a2 - b2;
    }
    {   // h=2: pairs (4l, 4l+2), (4l+1, 4l+3)
        const float a0 = v[0], a1 = v[1], b0 = v[2], b1 = v[3];
        v[0] = a0 + b0; v[2] = a0 - b0;
        v[1] = a1 + b1; v[3] = a1 - b1;
    }
#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {   // h = 4*m
        const float o0 = __shfl_xor_sync(0xffffffff, v[0], m, WARP_SIZE);
        const float o1 = __shfl_xor_sync(0xffffffff, v[1], m, WARP_SIZE);
        const float o2 = __shfl_xor_sync(0xffffffff, v[2], m, WARP_SIZE);
        const float o3 = __shfl_xor_sync(0xffffffff, v[3], m, WARP_SIZE);
        if (lane & m) {
            v[0] = o0 - v[0]; v[1] = o1 - v[1]; v[2] = o2 - v[2]; v[3] = o3 - v[3];
        } else {
            v[0] += o0; v[1] += o1; v[2] += o2; v[3] += o3;
        }
    }

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    for (int l = 0; l < 4; ++l) {
        v[l] *= inv_sqrt_128 * ggml_turbo5_sign1(seed, 4*lane + l);
    }

    dst_t * out = y + ib*QK_TURBO5;
    if constexpr (std::is_same_v<dst_t, float>) {
        float2 * out2 = (float2 *) out;
        out2[2*lane + 0] = make_float2(v[0], v[1]);   // coords 4*lane .. 4*lane+1
        out2[2*lane + 1] = make_float2(v[2], v[3]);   // coords 4*lane+2 .. 4*lane+3
    } else {
        out[4*lane + 0] = ggml_cuda_cast<dst_t>(v[0]);
        out[4*lane + 1] = ggml_cuda_cast<dst_t>(v[1]);
        out[4*lane + 2] = ggml_cuda_cast<dst_t>(v[2]);
        out[4*lane + 3] = ggml_cuda_cast<dst_t>(v[3]);
    }
}

template <typename dst_t>
static void dequantize_row_turbo5_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int64_t nb = k / QK_TURBO5;
    dequantize_block_turbo5_cont<dst_t><<<(int) ((nb + WARP_SIZE - 1)/WARP_SIZE), WARP_SIZE, 0, stream>>>(vx, y, nb);
}

// rowwise variant (from_to / ggml_get_rows style strided rows): one warp per cache block
template <typename dst_t>
static __global__ void dequantize_block_turbo5_rowwise(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne0203, const uint3 ne02_fdv,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    static_assert(QK_TURBO5 == 128, "turbo5 dequant kernel assumes QK_TURBO5 == 128");
    const int64_t ib     = blockIdx.x;               // block-within-row index (grid.x == ne00/QK_TURBO5)
    const int     lane   = threadIdx.x;              // blockDim == WARP_SIZE
    const int64_t nb_per_row = ne00 / QK_TURBO5;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            if (ib >= nb_per_row) {
                continue;
            }

            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02_fdv);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;   // strides are in block units
            const block_turbo5_0 * blk = (const block_turbo5_0 *) vx + ibx0 + ib;

            const float norm = __half2float(blk->norm);
            const uint16_t seed = blk->seed;

            const uint8_t q0 = blk->qs[2*lane + 0];
            const uint8_t q1 = blk->qs[2*lane + 1];
            const uint8_t sg = blk->signs[lane >> 1] >> (4*(lane & 1));

            float v[4];
            v[0] = turbo5_centroid((uint8_t) ((q0      & 0xF) | ((sg      & 1) << 4))) * norm;
            v[1] = turbo5_centroid((uint8_t) ((q0      >>  4) | (((sg >> 1) & 1) << 4))) * norm;
            v[2] = turbo5_centroid((uint8_t) ((q1      & 0xF) | (((sg >> 2) & 1) << 4))) * norm;
            v[3] = turbo5_centroid((uint8_t) ((q1      >>  4) | (((sg >> 3) & 1) << 4))) * norm;

            for (int l = 0; l < 4; ++l) {
                v[l] *= ggml_turbo5_sign2(seed, 4*lane + l);
            }
            {
                const float a0 = v[0], b0 = v[1], a2 = v[2], b2 = v[3];
                v[0] = a0 + b0; v[1] = a0 - b0;
                v[2] = a2 + b2; v[3] = a2 - b2;
            }
            {
                const float a0 = v[0], a1 = v[1], b0 = v[2], b1 = v[3];
                v[0] = a0 + b0; v[2] = a0 - b0;
                v[1] = a1 + b1; v[3] = a1 - b1;
            }
#pragma unroll
            for (int m = 1; m <= 16; m <<= 1) {
                const float o0 = __shfl_xor_sync(0xffffffff, v[0], m, WARP_SIZE);
                const float o1 = __shfl_xor_sync(0xffffffff, v[1], m, WARP_SIZE);
                const float o2 = __shfl_xor_sync(0xffffffff, v[2], m, WARP_SIZE);
                const float o3 = __shfl_xor_sync(0xffffffff, v[3], m, WARP_SIZE);
                if (lane & m) {
                    v[0] = o0 - v[0]; v[1] = o1 - v[1]; v[2] = o2 - v[2]; v[3] = o3 - v[3];
                } else {
                    v[0] += o0; v[1] += o1; v[2] += o2; v[3] += o3;
                }
            }

            constexpr float inv_sqrt_128 = 0.08838834764831845f;
            for (int l = 0; l < 4; ++l) {
                v[l] *= inv_sqrt_128 * ggml_turbo5_sign1(seed, 4*lane + l);
            }

            dst_t * yrow = y + (i0203*ne01 + i01)*ne00 + ib*QK_TURBO5;
            yrow[4*lane + 0]        = ggml_cuda_cast<dst_t>(v[0]);
            yrow[4*lane + 1]        = ggml_cuda_cast<dst_t>(v[1]);
            yrow[4*lane + 2]        = ggml_cuda_cast<dst_t>(v[2]);
            yrow[4*lane + 3]        = ggml_cuda_cast<dst_t>(v[3]);
        }
    }
}

template <typename dst_t>
static void dequantize_row_turbo5_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const int64_t nb_per_row = ne00 / QK_TURBO5;
    const dim3 num_blocks((int) std::min(nb_per_row, (int64_t) 65535), (int) std::min(ne01, (int64_t) 65535), (int) std::min(ne0203, (int64_t) 65535));
    dequantize_block_turbo5_rowwise<dst_t><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / QK4_NL;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename T> struct alignas(sizeof(T)*4) cvt_vec4 { T v[4]; };

// four elements per thread, so a warp moves 512B (RDNA) / 1k (CDNA) per load
template <typename src_t, typename dst_t>
static __global__ void convert_unary_cont_vec4(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k4) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k4) {
        return;
    }

    const cvt_vec4<src_t> xv = ((const cvt_vec4<src_t> *) vx)[i];

    cvt_vec4<dst_t> yv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yv.v[j] = ggml_cuda_cast<dst_t>(xv.v[j]);
    }

    ((cvt_vec4<dst_t> *) y)[i] = yv;
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    if (k % 4 == 0 &&
        (uintptr_t) vx % alignof(cvt_vec4<src_t>) == 0 &&
        (uintptr_t) y  % alignof(cvt_vec4<dst_t>) == 0) {
        const int64_t k4 = k/4;
        const int64_t num_blocks = (k4 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
        convert_unary_cont_vec4<src_t, dst_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k4);
        return;
    }

    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_TURBO3_0:
            return dequantize_block_cont_cuda<QK_TURBO3, QR_TURBO3, dequantize_turbo3_0>;
        case GGML_TYPE_TURBO4_0:
            return dequantize_block_cont_cuda<QK_TURBO4, QR_TURBO4, dequantize_turbo4_0>;
        case GGML_TYPE_TURBO5_0:
            return dequantize_row_turbo5_cont_cuda;  // per-block inverse FHT (see the kernel above)
        case GGML_TYPE_TURBO3Q_0:
            return dequantize_block_cont_cuda<QK_TURBO3Q, QR_TURBO3Q, dequantize_turbo3q_0>;
        case GGML_TYPE_TURBO4Q_0:
            return dequantize_block_cont_cuda<QK_TURBO4Q, QR_TURBO4Q, dequantize_turbo4q_0>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_TURBO3_0:
            return dequantize_block_cont_cuda<QK_TURBO3, QR_TURBO3, dequantize_turbo3_0>;
        case GGML_TYPE_TURBO4_0:
            return dequantize_block_cont_cuda<QK_TURBO4, QR_TURBO4, dequantize_turbo4_0>;
        case GGML_TYPE_TURBO5_0:
            return dequantize_row_turbo5_cont_cuda;  // per-block inverse FHT (see the kernel above)
        case GGML_TYPE_TURBO3Q_0:
            return dequantize_block_cont_cuda<QK_TURBO3Q, QR_TURBO3Q, dequantize_turbo3q_0>;
        case GGML_TYPE_TURBO4Q_0:
            return dequantize_block_cont_cuda<QK_TURBO4Q, QR_TURBO4Q, dequantize_turbo4q_0>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_TURBO3_0:
            return dequantize_block_cont_cuda<QK_TURBO3, QR_TURBO3, dequantize_turbo3_0>;
        case GGML_TYPE_TURBO4_0:
            return dequantize_block_cont_cuda<QK_TURBO4, QR_TURBO4, dequantize_turbo4_0>;
        case GGML_TYPE_TURBO5_0:
            return dequantize_row_turbo5_cont_cuda;  // per-block inverse FHT (see the kernel above)
        case GGML_TYPE_TURBO3Q_0:
            return dequantize_block_cont_cuda<QK_TURBO3Q, QR_TURBO3Q, dequantize_turbo3q_0>;
        case GGML_TYPE_TURBO4Q_0:
            return dequantize_block_cont_cuda<QK_TURBO4Q, QR_TURBO4Q, dequantize_turbo4q_0>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_TURBO3_0:
            return dequantize_block_cuda<QK_TURBO3, QR_TURBO3, dequantize_turbo3_0>;
        case GGML_TYPE_TURBO4_0:
            return dequantize_block_cuda<QK_TURBO4, QR_TURBO4, dequantize_turbo4_0>;
        case GGML_TYPE_TURBO5_0:
            return dequantize_row_turbo5_cuda;  // per-block inverse FHT, rowwise strides
        case GGML_TYPE_TURBO3Q_0:
            return dequantize_block_cuda<QK_TURBO3Q, QR_TURBO3Q, dequantize_turbo3q_0>;
        case GGML_TYPE_TURBO4Q_0:
            return dequantize_block_cuda<QK_TURBO4Q, QR_TURBO4Q, dequantize_turbo4q_0>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
