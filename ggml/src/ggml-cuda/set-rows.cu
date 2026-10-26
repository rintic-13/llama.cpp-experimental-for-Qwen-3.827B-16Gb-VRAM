#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

// ---- TurboQuant4 set_rows: 128-element groups with WHT rotation + 4-bit quantization ----
//
// turbo4 block size IS the WHT group size (QK_TURBO4 == 128), so 1 CUDA block = 1 turbo4
// block and thread j handles element j. WHT: signs1 -> warp-shuffle butterflies (h = 1..16)
// -> shared butterflies (h = 32, 64) -> *1/sqrt(128) * signs2. This computes the exact same
// pairings and operand order as turbo_cpu_fwht() in ggml-turbo-quant.c (bit-identical).
// Packing: qs[j/2] |= (idx & 0xF) << ((j%2)*4), warp-cooperative via shuffle, no atomics.

template <typename idx_t, typename src_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turbo4(
        const src_t * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    // blockIdx.x = flat block index; threadIdx.x = element within block (0..127)
    const int j = threadIdx.x;

    // Decode blockIdx.x -> (i_blk, i01, i02, i03), same mapping as k_set_rows
    const int64_t n_blocks_per_row = ne00 / QK_TURBO4;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t dst_row   = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const src_t * src_row   = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 * dst_row_ptr = (block_turbo4_0 *) ((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo4_0 * blk = dst_row_ptr + i_blk;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBO4 + j];
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = 128 / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v  = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 -> butterfly -> signs2, normalized) ----
    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

    const int lane = j & 31;
    float val = x[j];

#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }

    x[j] = val;
    __syncthreads();

    if (j % 64 < 32) {
        float a = x[j], b = x[j + 32];
        x[j]      = a + b;
        x[j + 32] = a - b;
    }
    __syncthreads();

    if (j % 128 < 64) {
        float a = x[j], b = x[j + 64];
        x[j]      = a + b;
        x[j + 64] = a - b;
    }
    __syncthreads();

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Step 5: Quantize element j to 4-bit centroid ----
    const uint8_t idx = turbo_nearest_centroid_4bit(x[j]);

    // ---- Step 6: Pack qs (nibble packed, warp-cooperative) ----
    // Thread pair (j, j+1) shares one qs byte.
    const uint8_t my_nibble = idx & 0xF;
    const uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    // ---- Step 7: Reconstruction norm (parallel) ----
    const float c = TURBO_CENTROIDS_4BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm (one thread per block) ----
    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda_turbo4_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const src_t * src0_d = (const src_t *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_TURBO4 == 0);  // block-aligned (128), head_dim 256 % 128 == 0

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBO4;
    const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;

    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    if (ne_total > 0) {
        k_set_rows_turbo4<idx_t, src_t><<<(int) ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turbo4_0 *) dst->data,
            ne00, ne01, ne02, ne11, ne12,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32) {
        set_rows_cuda_turbo4_impl<float, idx_t>(ctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16) {
        set_rows_cuda_turbo4_impl<half, idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}

// ---- TurboQuant3 set_rows: same structure, 3-bit codebook ----
//
// 3-bit index split: low 2 bits -> qs[] (4 codes/byte), high bit -> signs[] (1 bit/element).
// Packing uses width-32 __shfl_sync gathers per 4-/8-lane group (NOT __ballot_sync: a 64-bit
// ballot mask would truncate on wave64 devices). GROUP == QK_TURBO3 == 128: 1 CUDA block per
// turbo3 block, norm written by thread j == 0.

template <typename idx_t, typename src_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turbo3(
        const src_t * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    const int j = threadIdx.x;

    const int64_t n_groups_per_row = ne00 / QK_TURBO3;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t dst_row   = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const src_t * src_row   = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *) ((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo3_0 * blk = dst_row_ptr + i_grp;

    __shared__ float x[128];
    x[j] = src_row[i_grp * QK_TURBO3 + j];
    __syncthreads();

    constexpr int n_warps = 128 / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v  = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    x[j] *= inv_norm;
    __syncthreads();

    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

    const int lane = j & 31;
    float val = x[j];

#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }

    x[j] = val;
    __syncthreads();

    if (j % 64 < 32) {
        float a = x[j], b = x[j + 32];
        x[j]      = a + b;
        x[j + 32] = a - b;
    }
    __syncthreads();

    if (j % 128 < 64) {
        float a = x[j], b = x[j + 64];
        x[j]      = a + b;
        x[j + 64] = a - b;
    }
    __syncthreads();

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Quantize element j to 3-bit centroid ----
    const uint8_t idx = turbo_nearest_centroid_3bit(x[j]);

    // ---- Pack qs: 4 elements per byte via a 4-lane shuffle gather ----
    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        qs_byte |= (uint8_t) (__shfl_sync(0xffffffff, my_low2, (lane & ~3) + k, WARP_SIZE) << (k * 2));
    }
    if (lane % 4 == 0) {
        blk->qs[j / 4] = qs_byte;
    }

    // ---- Pack signs: 8 elements per byte via an 8-lane shuffle gather ----
    const uint8_t my_sign = (idx >> 2) & 1;
    uint8_t signs_byte = 0;
#pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        signs_byte |= (uint8_t) (__shfl_sync(0xffffffff, my_sign, (lane & ~7) + sb, WARP_SIZE) << sb);
    }
    if (lane % 8 == 0) {
        blk->signs[j / 8] = signs_byte;
    }

    // ---- Reconstruction norm (parallel, same pattern as the group norm) ----
    const float c = TURBO_CENTROIDS_3BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda_turbo3_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const src_t * src0_d = (const src_t *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_TURBO3 == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBO3;
    const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;

    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    if (ne_total > 0) {
        k_set_rows_turbo3<idx_t, src_t><<<(int) ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turbo3_0 *) dst->data,
            ne00, ne01, ne02, ne11, ne12,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32) {
        set_rows_cuda_turbo3_impl<float, idx_t>(ctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16) {
        set_rows_cuda_turbo3_impl<half, idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<int32_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<int32_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<int64_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<int64_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(
        src0->type == GGML_TYPE_F32 ||
        (src0->type == GGML_TYPE_F16 && (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0)));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
