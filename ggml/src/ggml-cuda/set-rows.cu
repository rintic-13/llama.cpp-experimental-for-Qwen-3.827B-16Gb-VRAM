#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// ============================================================================
// KV probe dump (NVFP4 feasibility study): env-gated dump of RAW (pre-rotation,
// pre-quantization) K/V rows at the exact point where the KV encoder consumes
// them. Analysis (scale policies, overflow census, MSE vs turbo codecs) runs
// offline in tools/probe_nvfp4.cpp.
//
//   GGML_KV_DUMP           - output file path (binary, appended); unset = OFF
//   GGML_KV_DUMP_ROWS      - max rows dumped per set_rows call (default 128)
//   GGML_KV_DUMP_TYPES     - optional substring filter on ggml_type_name(dst),
//                            e.g. "turbo4" captures only the K cache of
//                            -ctk turbo4 runs
//   GGML_KV_DUMP_MAX_CALLS - hard cap of dumped calls per process (default 400),
//                            safety net against runaway files on long runs
//
// K and V of one run are told apart by dst->type (use different cache types,
// e.g. -ctk turbo4 -ctv turbo5). With GGML_KV_DUMP unset the hook costs one
// cached getenv check, touches no numerics and no devices - bit-identical
// baselines are unaffected.
// ============================================================================

#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static void ggml_cuda_kv_dump_rows(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * src0, const ggml_tensor * dst) {
    static const char * dump_path = getenv("GGML_KV_DUMP");
    if (dump_path == nullptr) {
        return; // common case: nothing to do, cached check
    }
    // optional companion dump (GPU-encoder bit-exactness check, see "KVG1" below);
    // nullptr disables it, set to nullptr permanently on file errors
    static const char * gpu_dump_path_c = getenv("GGML_KV_GPU_DUMP");

    static const int64_t max_rows = getenv("GGML_KV_DUMP_ROWS") ? atoll(getenv("GGML_KV_DUMP_ROWS")) : 128;
    static uint64_t calls_left = getenv("GGML_KV_DUMP_MAX_CALLS") ? (uint64_t) atoll(getenv("GGML_KV_DUMP_MAX_CALLS")) : 400;
    if (calls_left == 0) {
        return;
    }
    // quantized KV-encoder dst types routed through set_rows only
    switch (dst->type) {
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO5_0:
        case GGML_TYPE_TURBO3Q_0:
        case GGML_TYPE_TURBO4Q_0:
        case GGML_TYPE_NVFP4:
            break;
        default:
            return;
    }
    if (getenv("GGML_KV_DUMP_TYPES")) {
        const char * filter = getenv("GGML_KV_DUMP_TYPES");
        if (strstr(ggml_type_name(dst->type), filter) == nullptr) {
            return;
        }
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    if (!ggml_is_contiguous(src0)) {
        fprintf(stderr, "GGML_KV_DUMP: skipped call - src0 not contiguous\n");
        return;
    }

    const int64_t ne0   = src0->ne[0];
    const int64_t rows  = src0->ne[1] < max_rows ? src0->ne[1] : max_rows;
    if (rows <= 0) {
        return;
    }
    const size_t elsize = ggml_type_size(src0->type);
    const size_t bytes  = (size_t) (rows * ne0) * elsize;

    std::vector<char> host;
    host.resize(bytes);

    CUDA_CHECK(cudaMemcpyAsync(host.data(), src0->data, bytes, cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    struct kv_dump_hdr {
        char     magic[4];   // "KVD1"
        uint32_t dst_type;   // ggml enum of the KV cache type (identifies K vs V)
        uint32_t src_type;   // enum of the dumped value tensor (f32/f16)
        uint64_t ne0;        // elements per row (n_embd_kv_gqa)
        uint32_t rows;       // rows actually dumped
        uint32_t elsize;     // bytes per element
    };
    static FILE * f = nullptr;
    if (f == nullptr) {
        f = fopen(dump_path, "wb");
        if (f == nullptr) {
            fprintf(stderr, "GGML_KV_DUMP: cannot open '%s' - dump disabled\n", dump_path);
            calls_left = 0;
            return;
        }
    }

    kv_dump_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic[0] = 'K'; hdr.magic[1] = 'V'; hdr.magic[2] = 'D'; hdr.magic[3] = '1';
    hdr.dst_type = (uint32_t) dst->type;
    hdr.src_type = (uint32_t) src0->type;
    hdr.ne0      = (uint64_t) ne0;
    hdr.rows     = (uint32_t) rows;
    hdr.elsize   = (uint32_t) elsize;

    fwrite(&hdr, sizeof(hdr), 1, f);
    if (fwrite(host.data(), 1, bytes, f) != bytes) {
        fprintf(stderr, "GGML_KV_DUMP: short write - dump disabled\n");
        calls_left = 0;
    } else {
        fflush(f);
        calls_left--;
    }

    // ---- GPU-encoder bit-exactness companion dump ("KVG1") ----
    // Env GGML_KV_GPU_DUMP: for every float-dump call also write the row indices
    // (src1) and the GPU-encoded dst bytes of the SAME first `rows` src0 rows, so
    // the offline verifier can re-quantize the raw rows with quantize_row_nvfp4_ref
    // on the CPU and memcmp byte-for-byte.
    // Alignment note: valid while rows <= ne01 (every dumped row has i02 = i03 = 0) -
    // true for the KV path where an ubatch contributes >= 512 rows and max_rows <= 512.
    if (gpu_dump_path_c != nullptr) {
        static FILE * fg = nullptr; // opened once, same lifecycle as the float dump
        if (fg == nullptr) {
            fg = fopen(gpu_dump_path_c, "wb");
            if (fg == nullptr) {
                fprintf(stderr, "GGML_KV_GPU_DUMP: cannot open '%s' - gpu dump disabled\n", gpu_dump_path_c);
                gpu_dump_path_c = nullptr;
                return;
            }
        }

        ggml_tensor * idx_t = dst->src[1];
        GGML_ASSERT(idx_t != nullptr && (idx_t->type == GGML_TYPE_I32 || idx_t->type == GGML_TYPE_I64));
        GGML_ASSERT(ggml_is_contiguous(idx_t)); // rows <= src0 ne[1] (see note) => idx elements 0..rows-1 are the row indices
        const size_t elsize_idx = idx_t->type == GGML_TYPE_I64 ? 8 : 4;

        // dst row stride in bytes; encoded row payload = (ne0 / blck) * block_size
        const size_t row_bytes = (size_t) dst->nb[1];

        std::vector<char> idxh;
        idxh.resize(rows * elsize_idx);
        CUDA_CHECK(cudaMemcpyAsync(idxh.data(), idx_t->data, idxh.size(), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream())); // host needs the indices before gathering dst rows

        // dst is 4D; dumped rows sit in the (i02, i03) = (0, 0) plane (see alignment note)
        std::vector<char> gpu_rows;
        gpu_rows.resize(rows * row_bytes);
        for (int r = 0; r < rows; ++r) {
            int64_t idx_v = elsize_idx == 8 ? ((const int64_t *) idxh.data())[r] : (int64_t) ((const int32_t *) idxh.data())[r];
            CUDA_CHECK(cudaMemcpyAsync(gpu_rows.data() + (size_t) r*row_bytes, (const char *) dst->data + idx_v*dst->nb[1],
                row_bytes, cudaMemcpyDeviceToHost, ctx.stream()));
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

        struct gpu_dump_hdr {
            char     magic[4];   // "KVG1"
            uint32_t idx_type;   // ggml enum of the row-index tensor (I32/I64)
            uint32_t row_bytes;  // bytes per encoded row (= dst row stride)
            uint64_t ne0;        // elements per row, must match the float-dump record
            uint32_t rows;       // rows dumped, must match the float-dump record
            uint32_t pad;
        };
        static_assert(sizeof(gpu_dump_hdr) == 32, "kvg hdr size");

        gpu_dump_hdr ghdr;
        memset(&ghdr, 0, sizeof(ghdr));
        ghdr.magic[0] = 'K'; ghdr.magic[1] = 'V'; ghdr.magic[2] = 'G'; ghdr.magic[3] = '1';
        ghdr.idx_type = (uint32_t) idx_t->type;
        ghdr.row_bytes = (uint32_t) row_bytes;
        ghdr.ne0 = (uint64_t) ne0;
        ghdr.rows = (uint32_t) rows;

        if (fwrite(&ghdr, sizeof(ghdr), 1, fg) != 1 ||
            fwrite(idxh.data(), 1, idxh.size(), fg) != idxh.size() ||
            fwrite(gpu_rows.data(), 1, gpu_rows.size(), fg) != gpu_rows.size()) {
            fprintf(stderr, "GGML_KV_GPU_DUMP: short write - gpu dump disabled\n");
            fclose(fg);
            fg = nullptr;
            gpu_dump_path_c = nullptr;
        } else {
            fflush(fg);
        }
    }
}

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

// ---- NVFP4-like KV cache encoder (Stage A) ----
// Single-level NVFP4: per-16 UE4M3 scale (amax/6, max policy) + packed E2M1 nibbles. Unrotated,
// no dither - written rows are exactly what the probe analyzed (relMSE 0.87%/0.87% on K/V).
// Byte-exact replica of quantize_row_nvfp4_ref (ggml-quants.c): manual UE4M3 encode (CPU round
// semantics, NOT the Blackwell hw fp8 path) and the 16-entry kvalues_fp4 argmin.
static __device__ __forceinline__ uint8_t nvfp4_fp32_to_ue4m3_ref(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

static __device__ __forceinline__ uint8_t nvfp4_nearest_kvalue(float x, float d) {
    // exact CPU semantics: strict-less argmin over kvalues_fp4 in table order
    int   best_index = 0;
    float best_err   = fabsf((float) kvalues_fp4[0]*d - x);
#pragma unroll
    for (int i = 1; i < 16; i++) {
        const float err = fabsf((float) kvalues_fp4[i]*d - x);
        if (err < best_err) {
            best_index = i;
            best_err   = err;
        }
    }
    return (uint8_t) best_index;
}

// One thread quantizes one 16-element sub-block: i indexes sub-blocks across the whole tensor.
template <typename src_t, typename idx_t>
static __global__ void k_set_rows_nvfp4(
        const src_t * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_nvfp4 * __restrict__ dst,
        const int64_t ne_total,
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
        const int64_t s3,
        const uint3   ne00,
        const uint3   ne01,
        const uint3   ne02,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {
    const int64_t isub = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (isub >= ne_total) {
        return;
    }

    const int64_t i_base = isub * QK_NVFP4_SUB;
    uint32_t      tmp    = (uint32_t) i_base;

    uint2 div_mod;

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

    const src_t * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_nvfp4 * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_nvfp4);

    const int    blkidx = (int) (i00 / QK_NVFP4);
    const int    c0     = (int) (i00 % QK_NVFP4);
    const int    sub    = c0 / QK_NVFP4_SUB;

    float v[QK_NVFP4_SUB];
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) {
        v[k] = static_cast<float>(src_row[i00 + k]);
    }

    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) {
        const float a = fabsf(v[k]);
        if (amax < a) {
            amax = a;
        }
    }

    const uint8_t ue = nvfp4_fp32_to_ue4m3_ref(amax / 6.0f);
    const float   d  = ggml_cuda_ue4m3_to_fp32(ue);

    block_nvfp4 * blk = dst_row_ptr + blkidx;
    blk->d[sub] = ue;

    uint8_t * qs = blk->qs + sub*(QK_NVFP4_SUB/2);
#pragma unroll
    for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
        const uint8_t x0 = nvfp4_nearest_kvalue(v[0        + j], d);
        const uint8_t x1 = nvfp4_nearest_kvalue(v[8 + j        ], d);
        qs[j] = x0 | (x1 << 4);
    }

    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
}

template <typename idx_t>
static void set_rows_cuda_nvfp4(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ne00 % QK_NVFP4_SUB == 0);

    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / QK_NVFP4_SUB;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01 / (src0->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(half));
    const int64_t s02 = nb02 / (src0->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(half));
    const int64_t s03 = nb03 / (src0->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(half));
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        if (src0->type == GGML_TYPE_F32) {
            k_set_rows_nvfp4<float, idx_t><<<grid_size, block_size, 0, stream>>>(
                (const float *) src0->data, (const idx_t *) src1->data, (block_nvfp4 *) dst->data,
                ne_total, ne11, ne12, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        } else {
            k_set_rows_nvfp4<half, idx_t><<<grid_size, block_size, 0, stream>>>(
                (const half *) src0->data, (const idx_t *) src1->data, (block_nvfp4 *) dst->data,
                ne_total, ne11, ne12, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
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

#ifdef GGML_TURBO_DITHER
    // dither seed: linear 128-value block index inside the cache tensor; convert.cu's
    // dequantize kernels pass their `ib` argument, which is the very same linear index.
    const int64_t lblk = (dst_row*s1 + i02*s2 + i03*s3) / (int64_t) sizeof(block_turbo4_0) + i_blk;
    const float dith_u = turbo_dither((uint32_t) lblk, j, TURBO_DITHER_STEP_4BIT);
#else
    const float dith_u = 0.0f;
#endif

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

    // ---- Step 5: quantize element j to 4-bit centroid (subtractive dither when enabled) ----
#ifdef GGML_TURBO_SCALE_SEARCH
    // [llama-tq encoder upgrade] mirror of the CPU reference codec (ggml-turbo-quant.c):
    //   32 trial scales, per-trial warp-tree sums, thread 0 scans k ascending with strict >
    //   (the exact nominal multiplier k=16 wins ties), the winner is broadcast, element j is
    //   re-quantized at it, and the stored scalar is the unbiased scale grp_norm / inner
    //   instead of the norm correction grp_norm / ||rec||.
    // The warp tree + 4-warp sum differs from the CPU serial sum at last-ulp (exactly like
    // the group norm does today): near-tie winners may differ per engine, and every written
    // cache stays self-describing (idx + norm) and self-consistent for its readers.
    const float xv = x[j];

    __shared__ float warp_in [TURBO_N_TRIAL_SCALES][n_warps];
    __shared__ float warp_rsq[TURBO_N_TRIAL_SCALES][n_warps];

#pragma unroll
    for (int k = 0; k < TURBO_N_TRIAL_SCALES; k++) {
        const uint8_t tidx = turbo_nearest_centroid_4bit(xv * TURBO_TRIAL_SCALE_INV[k] + dith_u);
        const float   trec = TURBO_CENTROIDS_4BIT[tidx] - dith_u;

        float w_in  = xv * trec;
        float w_rsq = trec * trec;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            w_in  += __shfl_xor_sync(0xffffffff, w_in,  offset, WARP_SIZE);
            w_rsq += __shfl_xor_sync(0xffffffff, w_rsq, offset, WARP_SIZE);
        }

        if (j % WARP_SIZE == 0) {
            warp_in[k][j / WARP_SIZE]  = w_in;
            warp_rsq[k][j / WARP_SIZE] = w_rsq;
        }
    }
    __syncthreads();

    __shared__ float s_best_in;
    __shared__ int   s_best_k;
    if (j == 0) {
        float best_in  = 0.0f;
        float best_rsq = 0.0f;
        int   best_k   = 16;

        for (int k = 0; k < TURBO_N_TRIAL_SCALES; k++) {
            float inner = 0.0f, rsq = 0.0f;

            for (int w = 0; w < n_warps; w++) {
                inner += warp_in[k][w];
                rsq   += warp_rsq[k][w];
            }

            if (inner <= 1e-10f) {
                continue; // degenerate trial (zero block): never selected
            }

            if (best_in <= 0.0f || inner * inner * best_rsq > best_in * best_in * rsq) {
                best_in  = inner;
                best_rsq = rsq;
                best_k   = k;
            }
        }

        s_best_in = best_in;
        s_best_k  = best_k;
    }
    __syncthreads();

    // quantize element j at the winning scale (same operand order as the trial loop)
    const uint8_t idx = turbo_nearest_centroid_4bit(xv * TURBO_TRIAL_SCALE_INV[s_best_k] + dith_u);

    // ---- Step 6: Pack qs (nibble packed, warp-cooperative) ----
    // Thread pair (j, j+1) shares one qs byte.
    const uint8_t my_nibble = idx & 0xF;
    const uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    // ---- Step 8: write the unbiased scale (one thread per block) ----
    const float corrected_norm = (s_best_in > 1e-10f) ? grp_norm / s_best_in : grp_norm;
    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
#else
    const uint8_t idx = turbo_nearest_centroid_4bit(x[j] + dith_u);

    // ---- Step 6: Pack qs (nibble packed, warp-cooperative) ----
    // Thread pair (j, j+1) shares one qs byte.
    const uint8_t my_nibble = idx & 0xF;
    const uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    // ---- Step 7: Reconstruction norm (parallel) ----
    // with dither enabled the stored reconstruction value is (centroid - u), matching the
    // convert.cu dequantize path; dith_u is 0.0f when the flag is off (bit-identical to before)
    const float c = TURBO_CENTROIDS_4BIT[idx] - dith_u;
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
#endif
}

// ---- TurboQuant5 set_rows: same structure, 5-bit codebook ----
//
// 5-bit index split: low 4 bits -> qs[] (nibble packed, 2 elements/byte),
// high bit -> signs[] (1 bit/element, 8 elements/byte).
// Packing: qs uses warp-cooperative nibble pair write (even thread writes full byte);
// signs use 8-lane shuffle gather per byte (like turbo3).
//
// 2026-11-08 per-block sign masks: the WHT frame of each block is its own signed Hadamard
// G_b = D(s2_b)*H/sqrt(128)*D(s1_b) derived from a 16-bit seed stored in the block. The seed is
// ggml_turbo5_seed_bits over the RAW fp32 bits of the block (shared helpers in ggml-common.h):
// a commutative u64 sum, so this warp-shuffle reduction and the CPU reference codec derive the
// identical seed from identical input bytes (CPU<->GPU byte mirror gate stays bit-exact).
// turbo3/turbo4 keep the shared TURBO_WHT_SIGNS1/2 tables unchanged.

template <typename idx_t, typename src_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turbo5(
        const src_t * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo5_0 * __restrict__ dst,
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

    const int64_t n_blocks_per_row = ne00 / QK_TURBO5;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;   // FIX: was tmp % ne02 (wrong for ne02*ne03 > ne02)

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t dst_row   = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const src_t * src_row   = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo5_0 * dst_row_ptr = (block_turbo5_0 *) ((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turbo5_0 * blk = dst_row_ptr + i_blk;

    // No dither here: subtractive dither is applied to turbo3_0/turbo4_0 only (see ggml-common.h).
    const float dith_u = 0.0f;

    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBO5 + j];
    __syncthreads();

    // ---- per-block seed from the RAW bits of x[] (before normalization) ----
    // commutative u64 sum: warp-shuffle reduction + warp accumulators give every thread the
    // same seed the CPU reference codec derives from the same 128 input floats.
    constexpr int n_warps = 128 / WARP_SIZE;
    __shared__ uint64_t warp_h[n_warps];
    {
        const uint32_t raw_bits = *(const uint32_t *) &x[j];
        uint64_t h = (uint64_t) raw_bits * (ggml_turbo5_sm64((uint64_t) j + 0x12345678ULL) | 1ULL);
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            h += __shfl_xor_sync(0xffffffff, h, offset, WARP_SIZE);
        if (j % WARP_SIZE == 0)
            warp_h[j / WARP_SIZE] = h;
    }
    __syncthreads();

    __shared__ uint16_t s_seed16;
    if (j == 0) {
        uint64_t total = 0;
        for (int w = 0; w < n_warps; w++) total += warp_h[w];
        s_seed16 = (uint16_t) (ggml_turbo5_sm64(total) >> 32);
    }
    __syncthreads();
    const uint16_t seed16 = s_seed16;

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

    // per-block mask: D(s1_b) from the stored seed (was the shared TURBO_WHT_SIGNS1 table)
    x[j] *= ggml_turbo5_sign1(seed16, j);
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
    // per-block mask: D(s2_b) from the stored seed (was the shared TURBO_WHT_SIGNS2 table)
    x[j] = x[j] * inv_sqrt_128 * ggml_turbo5_sign2(seed16, j);
    __syncthreads();

    // ---- Quantize element j to 5-bit centroid (subtractive dither when enabled) ----
#ifdef GGML_TURBO_SCALE_SEARCH
    // [llama-tq encoder upgrade] mirror of k_set_rows_turbo4 above and of the CPU reference
    //   codec (ggml-turbo-quant.c): 32 trial scales, per-trial warp-tree sums, thread 0 scans
    //   k ascending with strict > (the exact nominal multiplier k=16 wins ties), the winner is
    //   broadcast, element j is re-quantized at it, and the stored scalar is the unbiased scale
    //   grp_norm / inner. Dither-free: dith_u is constant 0.0f for turbo5, kept in the operands
    //   for structural symmetry with the 3/4-bit mirrors.
    //   The sign-mask packing below is UNCHANGED: qs and signs both derive from the SAME
    //   post-search idx each call (verified separately CPU<->GPU, incl. non-nominal winners).
    const float xv = x[j];

    __shared__ float warp_in [TURBO_N_TRIAL_SCALES][n_warps];
    __shared__ float warp_rsq[TURBO_N_TRIAL_SCALES][n_warps];

#pragma unroll
    for (int k = 0; k < TURBO_N_TRIAL_SCALES; k++) {
        const uint8_t tidx = turbo_nearest_centroid_5bit(xv * TURBO_TRIAL_SCALE_INV[k] + dith_u);
        const float   trec = TURBO_CENTROIDS_5BIT[tidx] - dith_u;

        float w_in  = xv * trec;
        float w_rsq = trec * trec;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            w_in  += __shfl_xor_sync(0xffffffff, w_in,  offset, WARP_SIZE);
            w_rsq += __shfl_xor_sync(0xffffffff, w_rsq, offset, WARP_SIZE);
        }

        if (j % WARP_SIZE == 0) {
            warp_in[k][j / WARP_SIZE]  = w_in;
            warp_rsq[k][j / WARP_SIZE] = w_rsq;
        }
    }
    __syncthreads();

    __shared__ float s_best_in;
    __shared__ int   s_best_k;
    if (j == 0) {
        float best_in  = 0.0f;
        float best_rsq = 0.0f;
        int   best_k   = 16;

        for (int k = 0; k < TURBO_N_TRIAL_SCALES; k++) {
            float inner = 0.0f, rsq = 0.0f;

            for (int w = 0; w < n_warps; w++) {
                inner += warp_in[k][w];
                rsq   += warp_rsq[k][w];
            }

            if (inner <= 1e-10f) {
                continue; // degenerate trial (zero block): never selected
            }

            if (best_in <= 0.0f || inner * inner * best_rsq > best_in * best_in * rsq) {
                best_in  = inner;
                best_rsq = rsq;
                best_k   = k;
            }
        }

        s_best_in = best_in;
        s_best_k  = best_k;
    }
    __syncthreads();

    // quantize element j at the winning scale (same operand order as the trial loop)
    const uint8_t idx = turbo_nearest_centroid_5bit(xv * TURBO_TRIAL_SCALE_INV[s_best_k] + dith_u);

    // ---- Pack qs: nibble packed, 2 elements per byte, even thread writes ----
    const uint8_t my_nibble = idx & 0xF;
    const uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    // ---- Pack signs: 1 bit per element, 8 elements per byte ----
    const uint8_t my_sign = (idx >> 4) & 1;
    uint8_t signs_byte = 0;
#pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        signs_byte |= (uint8_t) (__shfl_sync(0xffffffff, my_sign, (lane & ~7) + sb, WARP_SIZE) << sb);
    }
    if (lane % 8 == 0) {
        blk->signs[j / 8] = signs_byte;
    }

    // ---- write the unbiased scale (one thread per block) ----
    const float corrected_norm = (s_best_in > 1e-10f) ? grp_norm / s_best_in : grp_norm;
    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
        blk->seed = seed16;   // per-block mask seed: readers rebuild G_b from this
    }
#else
    // ---- Quantize element j to 5-bit centroid (subtractive dither when enabled) ----
    const uint8_t idx = turbo_nearest_centroid_5bit(x[j] + dith_u);

    // ---- Pack qs: nibble packed, 2 elements per byte, even thread writes ----
    const uint8_t my_nibble = idx & 0xF;
    const uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    // ---- Pack signs: 1 bit per element, 8 elements per byte ----
    const uint8_t my_sign = (idx >> 4) & 1;
    uint8_t signs_byte = 0;
#pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        signs_byte |= (uint8_t) (__shfl_sync(0xffffffff, my_sign, (lane & ~7) + sb, WARP_SIZE) << sb);
    }
    if (lane % 8 == 0) {
        blk->signs[j / 8] = signs_byte;
    }

    // ---- Reconstruction norm (parallel); recon value is centroid - dith_u (0.0f when flag off) ----
    const float c = TURBO_CENTROIDS_5BIT[idx] - dith_u;
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
        blk->seed = seed16;   // per-block mask seed (legacy branch: same seed as the search branch)
    }
#endif
}

template<typename src_t, typename idx_t>
static void set_rows_cuda_turbo5_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const src_t * src0_d = (const src_t *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_TURBO5 == 0);  // block-aligned (128), head_dim 256 % 128 == 0

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBO5;
    const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;

    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    if (ne_total > 0) {
        k_set_rows_turbo5<idx_t, src_t><<<(int) ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turbo5_0 *) dst->data,
            ne00, ne01, ne02, ne11, ne12,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo5(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32) {
        set_rows_cuda_turbo5_impl<float, idx_t>(ctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16) {
        set_rows_cuda_turbo5_impl<half, idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
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

#ifdef GGML_TURBO_DITHER
    // dither seed: linear 128-value block index inside the cache tensor; convert.cu's
    // dequantize kernels pass their `ib` argument, which is the very same linear index.
    const int64_t lblk = (dst_row*s1 + i02*s2 + i03*s3) / (int64_t) sizeof(block_turbo3_0) + i_grp;
    const float dith_u = turbo_dither((uint32_t) lblk, j, TURBO_DITHER_STEP_3BIT);
#else
    const float dith_u = 0.0f;
#endif

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
#ifdef GGML_TURBO_SCALE_SEARCH
    // [llama-tq encoder upgrade] mirror of k_set_rows_turbo4 above and of the CPU reference
    // codec: 32 trial scale multipliers, warp-tree sums per trial, thread 0 picks the max
    // angle cosine (k ascending, strict >, nominal k=16 wins ties), unbiased stored scale.
    const float xv = x[j];

    __shared__ float warp_in [TURBO_N_TRIAL_SCALES][n_warps];
    __shared__ float warp_rsq[TURBO_N_TRIAL_SCALES][n_warps];

#pragma unroll
    for (int k = 0; k < TURBO_N_TRIAL_SCALES; k++) {
        const uint8_t tidx = turbo_nearest_centroid_3bit(xv * TURBO_TRIAL_SCALE_INV[k] + dith_u);
        const float   trec = TURBO_CENTROIDS_3BIT[tidx] - dith_u;

        float w_in  = xv * trec;
        float w_rsq = trec * trec;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            w_in  += __shfl_xor_sync(0xffffffff, w_in,  offset, WARP_SIZE);
            w_rsq += __shfl_xor_sync(0xffffffff, w_rsq, offset, WARP_SIZE);
        }

        if (j % WARP_SIZE == 0) {
            warp_in[k][j / WARP_SIZE]  = w_in;
            warp_rsq[k][j / WARP_SIZE] = w_rsq;
        }
    }
    __syncthreads();

    __shared__ float s_best_in;
    __shared__ int   s_best_k;
    if (j == 0) {
        float best_in  = 0.0f;
        float best_rsq = 0.0f;
        int   best_k   = 16;

        for (int k = 0; k < TURBO_N_TRIAL_SCALES; k++) {
            float inner = 0.0f, rsq = 0.0f;

            for (int w = 0; w < n_warps; w++) {
                inner += warp_in[k][w];
                rsq   += warp_rsq[k][w];
            }

            if (inner <= 1e-10f) {
                continue; // degenerate trial (zero block): never selected
            }

            if (best_in <= 0.0f || inner * inner * best_rsq > best_in * best_in * rsq) {
                best_in  = inner;
                best_rsq = rsq;
                best_k   = k;
            }
        }

        s_best_in = best_in;
        s_best_k  = best_k;
    }
    __syncthreads();

    const uint8_t idx = turbo_nearest_centroid_3bit(xv * TURBO_TRIAL_SCALE_INV[s_best_k] + dith_u);

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

    // ---- Unbiased stored scale: grp_norm / <x_hat_rot, rec> (see the turbo4 kernel above) ----
    const float corrected_norm = (s_best_in > 1e-10f) ? grp_norm / s_best_in : grp_norm;

    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
#else
    const uint8_t idx = turbo_nearest_centroid_3bit(x[j] + dith_u);

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
    const float c = TURBO_CENTROIDS_3BIT[idx] - dith_u;
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
#endif
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

// ---- TurboQuant Q_prod set_rows: (b-1)-bit PolarQuant + 1-bit QJL residual sign ----
//
// codebook_bits==2 -> block_turbo3q_0 (qs = 2-bit indices, signs = QJL residual sign)
// codebook_bits==3 -> block_turbo4q_0 (qs/hi1 like k_set_rows_turbo3, signs = QJL residual sign)
// WHT pipeline is bit-identical to k_set_rows_turbo3. Unlike the MSE-only turbo types the norm
// stored here is the RAW group norm (no /recon_norm correction): the QJL term makes the inner
// product estimate unbiased (arXiv 2504.19874 Alg. 2), so rescaling would bias it again.
// rnorm = L2 norm of the residual in the unit-normalized WHT-rotated domain (fp16).

template <typename idx_t, typename src_t, typename block_t, int codebook_bits>
__launch_bounds__(128)
static __global__ void k_set_rows_turboq(
        const src_t * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_t * __restrict__ dst,
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

    const int64_t n_groups_per_row = ne00 / 128;   // QK_TURBO3Q == QK_TURBO4Q == 128
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
    block_t * dst_row_ptr = (block_t *) ((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_t * blk = dst_row_ptr + i_grp;

    __shared__ float x[128];
    x[j] = src_row[i_grp * 128 + j];
    __syncthreads();

    // ---- Group L2 norm (warp partials + shared reduction) ----
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

    // ---- Forward WHT (signs1 -> shuffle butterflies -> shared butterflies -> signs2) ----
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

    // ---- Quantize to the (b-1)-bit centroid; QJL sign = sign of the residual ----
    const uint8_t idx = (codebook_bits == 3) ? turbo_nearest_centroid_3bit(x[j])
                                             : turbo_nearest_centroid_2bit(x[j]);
    const float   c   = (codebook_bits == 3) ? TURBO_CENTROIDS_3BIT[idx] : TURBO_CENTROIDS_2BIT[idx];
    const float   e   = x[j] - c;
    const uint8_t qjl_sign = (e >= 0.0f) ? 1 : 0;

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

    // ---- 3-bit index high bit (turbo4q only): 8 per byte, 8-lane gather ----
    if constexpr (codebook_bits == 3) {
        const uint8_t my_hi1 = (idx >> 2) & 1;
        uint8_t hi1_byte = 0;
#pragma unroll
        for (int sb = 0; sb < 8; sb++) {
            hi1_byte |= (uint8_t) (__shfl_sync(0xffffffff, my_hi1, (lane & ~7) + sb, WARP_SIZE) << sb);
        }
        if (lane % 8 == 0) {
            blk->hi1[j / 8] = hi1_byte;
        }
    }

    // ---- Pack QJL signs: 8 elements per byte via an 8-lane shuffle gather ----
    uint8_t signs_byte = 0;
#pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        signs_byte |= (uint8_t) (__shfl_sync(0xffffffff, qjl_sign, (lane & ~7) + sb, WARP_SIZE) << sb);
    }
    if (lane % 8 == 0) {
        blk->signs[j / 8] = signs_byte;
    }

    // ---- Residual norm ||e|| (parallel reduction, same pattern as the group norm) ----
    float esq = e * e;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        esq += __shfl_xor_sync(0xffffffff, esq, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = esq;
    __syncthreads();

    __shared__ float s_res_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_res_sq = total;
    }
    __syncthreads();

    if (j == 0) {
        blk->norm  = __float2half(grp_norm);            // raw group norm, no recon correction
        blk->rnorm = __float2half(sqrtf(s_res_sq));
    }
}

template<typename src_t, typename idx_t, typename block_t, int codebook_bits>
static void set_rows_cuda_turboq_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const src_t * src0_d = (const src_t *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % 128 == 0);  // QK_TURBO3Q == QK_TURBO4Q == 128

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / 128;
    const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;

    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    if (ne_total > 0) {
        k_set_rows_turboq<idx_t, src_t, block_t, codebook_bits><<<(int) ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_t *) dst->data,
            ne00, ne01, ne02, ne11, ne12,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo3q(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32) {
        set_rows_cuda_turboq_impl<float, idx_t, block_turbo3q_0, 2>(ctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16) {
        set_rows_cuda_turboq_impl<half, idx_t, block_turbo3q_0, 2>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}

template<typename idx_t>
static void set_rows_cuda_turbo4q(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32) {
        set_rows_cuda_turboq_impl<float, idx_t, block_turbo4q_0, 3>(ctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16) {
        set_rows_cuda_turboq_impl<half, idx_t, block_turbo4q_0, 3>(ctx, src0, src1, dst);
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
    } else if (dst->type == GGML_TYPE_NVFP4) {
        set_rows_cuda_nvfp4<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO5_0) {
        set_rows_cuda_turbo5<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO3Q_0) {
        set_rows_cuda_turbo3q<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4Q_0) {
        set_rows_cuda_turbo4q<idx_t>(ctx, src0, src1, dst);
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
    } else if (dst->type == GGML_TYPE_NVFP4) {
        set_rows_cuda_nvfp4<int32_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<int32_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<int32_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO5_0) {
        set_rows_cuda_turbo5<int32_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO3Q_0) {
        set_rows_cuda_turbo3q<int32_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4Q_0) {
        set_rows_cuda_turbo4q<int32_t>(ctx, src0, src1, dst);
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
    } else if (dst->type == GGML_TYPE_NVFP4) {
        set_rows_cuda_nvfp4<int64_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        set_rows_cuda_turbo3<int64_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        set_rows_cuda_turbo4<int64_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO5_0) {
        set_rows_cuda_turbo5<int64_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO3Q_0) {
        set_rows_cuda_turbo3q<int64_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBO4Q_0) {
        set_rows_cuda_turbo4q<int64_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(
        src0->type == GGML_TYPE_F32 ||
        (src0->type == GGML_TYPE_F16 && (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO5_0 ||
                                         dst->type == GGML_TYPE_TURBO3Q_0 || dst->type == GGML_TYPE_TURBO4Q_0 || dst->type == GGML_TYPE_NVFP4)));
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

    // DUMP hook runs AFTER the kernel: dst bytes captured post-write (previously this ran
    // before the dispatch, so the KVG1 companion recorded the pre-write state - zeros for
    // fresh cache rows - and every offline mirror byte-compare failed). src0 rows are read
    // only, identical before and after; src1 (idx) is untouched by the kernels too.
    ggml_cuda_kv_dump_rows(ctx, src0, dst);
}
