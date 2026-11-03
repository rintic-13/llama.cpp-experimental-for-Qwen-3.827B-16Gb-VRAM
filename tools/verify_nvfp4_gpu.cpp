/*
 * verify_nvfp4_gpu.cpp - GPU-encoder bit-exactness verifier for the NVFP4-like KV cache (Stage A)
 *
 * Inputs (both produced by the GGML_KV_DUMP hook in ggml/src/ggml-cuda/set-rows.cu):
 *   <floats_dump>  KVD1 records: raw pre-quantization rows (f32 or f16) + dst/src types + ne0 + rows
 *   <gpu_dump>     KVG1 records (GGML_KV_GPU_DUMP): row indices + the GPU-encoded dst bytes
 *
 * The two files contain one record per set_rows call in identical order. For every record
 * this tool re-quantizes each raw row on the CPU with the production reference
 * quantize_row_nvfp4_ref and memcmp's it against the GPU bytes - byte-for-byte.
 *
 * Exit 0 = every dumped row matched; exit 1 = at least one mismatch (details printed).
 *
 * Usage:
 *   verify_nvfp4_gpu <floats_dump> <gpu_dump> [--print N]   (default N = 8 mismatching rows shown)
 *
 * Build (see build_verify.ps1):
 *   cl /O2 /EHsc verify_nvfp4_gpu.cpp /I ggml\include /I ggml\src /link ggml\src\ggml-base.lib
 */

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// exact mirrors of the hook's on-disk headers (same compiler/alignment rules!)
struct kv_dump_hdr {
    char     magic[4];   // "KVD1"
    uint32_t dst_type;
    uint32_t src_type;
    uint64_t ne0;
    uint32_t rows;
    uint32_t elsize;
};
struct gpu_dump_hdr {
    char     magic[4];   // "KVG1"
    uint32_t idx_type;
    uint32_t row_bytes;
    uint64_t ne0;
    uint32_t rows;
    uint32_t pad;
};
static_assert(sizeof(kv_dump_hdr) == 32, "kvd hdr layout");
static_assert(sizeof(gpu_dump_hdr) == 32, "kvg hdr layout");

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <floats_dump(KVD1)> <gpu_dump(KVG1)> [--print N]\n", argv[0]);
        return 2;
    }
    const char * fpath = argv[1];
    const char * gpath = argv[2];
    int print_max = 8;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--print") && i + 1 < argc) {
            print_max = atoi(argv[++i]);
        }
    }

    FILE * ff = fopen(fpath, "rb");
    if (!ff) { fprintf(stderr, "cannot open %s\n", fpath); return 2; }
    FILE * fg = fopen(gpath, "rb");
    if (!fg) { fprintf(stderr, "cannot open %s\n", gpath); fclose(ff); return 2; }

    long long rec = 0, rows_total = 0, rows_bad = 0, bytes_bad = 0, printed = 0;

    for (;; ++rec) {
        kv_dump_hdr fhdr;
        if (fread(&fhdr, sizeof(fhdr), 1, ff) != 1) {
            break; // EOF of float dump
        }
        if (memcmp(fhdr.magic, "KVD1", 4) != 0) {
            fprintf(stderr, "record %lld: bad float-dump magic\n", rec);
            return 2;
        }
        gpu_dump_hdr ghdr;
        if (fread(&ghdr, sizeof(ghdr), 1, fg) != 1) {
            fprintf(stderr, "record %lld: gpu dump ended before float dump\n", rec);
            return 2;
        }
        if (memcmp(ghdr.magic, "KVG1", 4) != 0) {
            fprintf(stderr, "record %lld: bad gpu-dump magic\n", rec);
            return 2;
        }
        if (fhdr.rows != ghdr.rows || fhdr.ne0 != ghdr.ne0) {
            fprintf(stderr, "record %lld: record mismatch f(%u x %llu) vs g(%u x %llu)\n",
                    rec, fhdr.rows, (unsigned long long) fhdr.ne0, ghdr.rows, (unsigned long long) ghdr.ne0);
            return 2;
        }

        const uint32_t rows = fhdr.rows;
        const uint64_t ne0  = fhdr.ne0;

        std::vector<float> raw((size_t) rows * (size_t) ne0);
        if (fhdr.elsize == 4) {
            if (fread(raw.data(), 4, raw.size(), ff) != raw.size()) { fprintf(stderr, "record %lld: short float payload\n", rec); return 2; }
        } else {
            std::vector<uint16_t> h16(raw.size());
            if (fread(h16.data(), 2, h16.size(), ff) != h16.size()) { fprintf(stderr, "record %lld: short f16 payload\n", rec); return 2; }
            for (size_t i = 0; i < h16.size(); ++i) {
                raw[i] = ggml_fp16_to_fp32((ggml_fp16_t) h16[i]);
            }
        }

        const size_t idx_el = (ghdr.idx_type == (uint32_t) GGML_TYPE_I64) ? 8 : 4;
        std::vector<char> idxh((size_t) rows * idx_el);
        if (fread(idxh.data(), 1, idxh.size(), fg) != idxh.size()) { fprintf(stderr, "record %lld: short idx payload\n", rec); return 2; }
        std::vector<uint8_t> gpu((size_t) rows * ghdr.row_bytes);
        if (fread(gpu.data(), 1, gpu.size(), fg) != gpu.size()) { fprintf(stderr, "record %lld: short gpu payload\n", rec); return 2; }

        const size_t ref_row_bytes = (size_t) (ne0 / QK_NVFP4) * sizeof(block_nvfp4);
        if (ref_row_bytes != ghdr.row_bytes) {
            fprintf(stderr, "record %lld: gpu row_bytes %u != expected %zu (ne0=%llu)\n",
                    rec, ghdr.row_bytes, ref_row_bytes, (unsigned long long) ne0);
            return 2;
        }

        std::vector<block_nvfp4> ref((size_t) (ne0 / QK_NVFP4));
        for (uint32_t r = 0; r < rows; ++r) {
            quantize_row_nvfp4_ref(&raw[(size_t) r * (size_t) ne0], ref.data(), (int64_t) ne0);
            const uint8_t * g = gpu.data() + (size_t) r * ghdr.row_bytes;
            const uint8_t * c = (const uint8_t *) ref.data();

            int64_t idx_v = idx_el == 8 ? ((const int64_t *) idxh.data())[r] : (int64_t) ((const int32_t *) idxh.data())[r];
            if (memcmp(g, c, ghdr.row_bytes) != 0) {
                ++rows_bad;
                size_t bad = 0;
                for (size_t b = 0; b < ghdr.row_bytes; ++b) {
                    if (g[b] != c[b]) {
                        int at = (int) b;
                        fprintf(stderr, "MISMATCH rec %lld row %u (kv_row %lld) byte %d: gpu=0x%02X cpu=0x%02X\n",
                                rec, r, (long long) idx_v, at, g[b], c[b]);
                        if (++printed >= print_max) {
                            fprintf(stderr, "(further mismatches suppressed, use --print N)\n");
                        }
                        break;
                    }
                }
                for (size_t b = 0; b < ghdr.row_bytes; ++b) {
                    if (g[b] != c[b]) ++bad;
                }
                bytes_bad += (long long) bad;
            }
        }

        rows_total += rows;
        printf("rec %3lld | %-9s | ne0 %3llu rows %4u | running rows_ok %lld rows_bad %lld\n",
               rec, ggml_type_name((enum ggml_type) fhdr.dst_type),
               (unsigned long long) ne0, rows, rows_total - rows_bad, rows_bad);
        fflush(stdout);
    }

    fclose(ff);
    fclose(fg);

    printf("\nrecords %lld | rows verified %lld | rows MISMATCHED %lld | bad bytes %lld\n",
           rec, rows_total, rows_bad, bytes_bad);
    if (rows_total == 0) {
        printf("NOTHING VERIFIED - empty dumps?\n");
        return 2;
    }
    if (rows_bad == 0) {
        printf("GPU ENCODER BIT-EXACT vs quantize_row_nvfp4_ref\n");
        return 0;
    }
    printf("GPU ENCODER DIVERGES FROM CPU REFERENCE\n");
    return 1;
}
