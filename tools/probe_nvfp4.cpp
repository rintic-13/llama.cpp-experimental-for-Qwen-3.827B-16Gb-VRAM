/*
 * probe_nvfp4.cpp - NVFP4-like KV cache feasibility probe (offline, CPU-only)
 *
 * Consumes the dump file produced by the GGML_KV_DUMP hook in
 * ggml/src/ggml-cuda/set-rows.cu and evaluates, on REAL post-rope K rows and
 * raw V rows of the actual model:
 *
 *   "NVFP4-like" (single-level codec: per-16-element UE4M3 scale + E2M1 codes,
 *                 no global FP32 scale - the dual-level standard NVFP4 naming
 *                 does NOT apply):
 *     max6  - production semantics, byte-for-byte quantize_row_nvfp4_ref
 *             (scale = amax/6 from ggml_fp32_to_ue4m3)
 *     mse   - per-subblock scale chosen by exhaustive UE4M3 SSE argmin
 *             (upper bound of what local scale search can buy)
 *     2nd   - outlier-robust scale from 2nd-largest |v| in the subblock
 *
 *   References (production codecs, CPU reference quantizers, no dither):
 *     turbo3 / turbo4 (/ turbo5) - quantized on the raw rows; MSE measured in
 *     the WHT-rotated domain via an exact replica of turbo_cpu_fwht.
 *
 * NVFP4-like rows go through the same exact WHT ("Krot") to mirror the planned
 * set_rows-NVFP4 pipeline (rotate K with the turbo sign pairing so the graph's
 * pre-rotated Q and the codec-rotated K stay consistent), and through the raw
 * rows as well ("Kraw") to price the no-rotation option. V is evaluated raw
 * (the planned design leaves V unrotated) and rotated for reference.
 *
 * Usage:
 *   probe_nvfp4 <dumpfile> [--csv out.csv] [--k turbo4_0] [--v turbo5_0]
 *
 * Build (see build_probe.ps1):
 *   cl /O2 /EHsc probe_nvfp4.cpp /I ggml\include /I ggml\src ^
 *      /link ggml\src\ggml-base.lib
 */

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

/* -------- helpers -------- */

static const int8_t PROBE_KV_FP4[16] = { // same as kvalues_fp4 (doubled E2M1)
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static float probe_f16_to_f32(uint16_t h) {
    const uint32_t sgn = (uint32_t) (h >> 15) & 1u;
    const uint32_t exp = (uint32_t) (h >> 10) & 0x1Fu;
    const uint32_t man = (uint32_t) h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            return sgn ? -0.0f : 0.0f;
        }
        int    e = -1;
        uint32_t m = man;
        while (!(m & 0x400u)) { m <<= 1; e--; }
        m &= 0x3FFu;
        bits = (sgn << 31) | ((uint32_t) (127 - 15 + e + 1) << 23) | (m << 13);
    } else if (exp == 0x1Fu) {
        bits = (sgn << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sgn << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct Agg {
    double sum_x2 = 0.0, sum_e2 = 0.0;
    long long elems = 0, flush = 0, zscale = 0, sscale = 0, rows = 0;
};

static std::map<std::string, Agg> g_agg;

static void accum(Agg & a, const float * x, const float * yq, int64_t n, long long zscale, long long sscale) {
    for (int64_t i = 0; i < n; i++) {
        const double e = (double) x[i] - (double) yq[i];
        a.sum_e2 += e * e;
        a.sum_x2 += (double) x[i] * (double) x[i];
        if (x[i] != 0.0f && yq[i] == 0.0f) {
            a.flush++;
        }
    }
    a.elems += n;
    a.zscale += zscale;
    a.sscale += sscale;
    a.rows++;
}

/* exact replica of turbo_cpu_fwht (ggml-turbo-quant.c:164-185);
 * signs come from the library (ggml_turbo_wht_signs) so they can never drift */
static void probe_fwht_forward_128(float * x, const float * s1, const float * s2) {
    const float inv_sqrt = 0.08838834764831845f;

    for (int i = 0; i < 128; i++) x[i] *= s1[i];

    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt * s2[i];
}

/* nearest doubled-kvalue index for |x|/scale, identical tie behaviour to
 * best_index_mxfp4 (exact argmin, ties keep the lowest index) */
static inline int probe_nearest_doubled(float v_over_d) {
    const float v = fabsf(v_over_d);
    int best = 0;
    float best_err = fabsf((float) PROBE_KV_FP4[0] - v);
    for (int i = 1; i < 16; i++) {
        const float err = fabsf((float) PROBE_KV_FP4[i] - v);
        if (err < best_err) {
            best    = i;
            best_err = err;
        }
    }
    return best;
}

static inline int probe_kvalue_index(float v, float d) {
    const int i = probe_nearest_doubled(v / d);
    return (v < 0.0f) ? (i == 0 ? 0 : i + 8) : i; // negatives live in nibles 8..15
}

static inline float probe_kvalue_decode(int idx, float d) {
    return (float) PROBE_KV_FP4[idx] * d;
}

/* -------- NVFP4-like policies -------- */

/* policy: 0 = production max/6 (via library), 1 = exhaustive MSE scale,
 * 2 = robust scale from 2nd-largest |v| */

static void encode_nvfp4_subblock(const float * v, block_nvfp4 * b, int sub, uint8_t ue) {
    const float d = ggml_ue4m3_to_fp32(ue);
    b->d[sub] = ue;
    for (int j = 0; j < QK_NVFP4_SUB / 2; j++) {
        const uint8_t i0 = (uint8_t) probe_kvalue_index(v[0 + j],            d);
        const uint8_t i1 = (uint8_t) probe_kvalue_index(v[QK_NVFP4_SUB / 2 + j], d);
        b->qs[sub * (QK_NVFP4_SUB / 2) + j] = (uint8_t) (i0 | (i1 << 4));
    }
}

static uint8_t search_nvfp4_scale_mse(const float * v) {
    double best_sse = -1.0;
    uint8_t best_ue = 0;
    for (int c = 1; c < 0x7F; c++) {
        const float d = ggml_ue4m3_to_fp32((uint8_t) c);
        if (!(d > 0.0f)) {
            continue;
        }
        double sse = 0.0;
        for (int i = 0; i < QK_NVFP4_SUB; i++) {
            const int    idx = probe_kvalue_index(v[i], d);
            const double e   = (double) probe_kvalue_decode(idx, d) - (double) v[i];
            sse += e * e;
        }
        if (best_sse < 0.0 || sse < best_sse) {
            best_sse = sse;
            best_ue  = (uint8_t) c;
        }
    }
    return best_ue;
}

/* -------- evaluation of one row -------- */

static void eval_nvfp4_policy(std::map<std::string, Agg> & agg, const std::string & tag,
                              const float * xrow, int64_t ne0, int policy, block_nvfp4 * blocks) {
    std::vector<float> yq(ne0);
    long long zscale = 0, sscale = 0;

    if (policy == 0) {
        quantize_row_nvfp4_ref(xrow, blocks, ne0);          // production-exact
        dequantize_row_nvfp4(blocks, yq.data(), ne0);
        // scale-byte census from the produced blocks (overflow/underflow of UE4M3)
        for (int64_t ib = 0; ib < ne0 / QK_NVFP4; ib++) {
            for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; s++) {
                const uint8_t ue = blocks[ib].d[s];
                if (ue == 0 || ue == 0x7F) {
                    zscale++;
                } else if (ue == 0x7E) {
                    sscale++;
                }
            }
        }
    } else {
        for (int64_t ib = 0; ib < ne0 / QK_NVFP4; ib++) {
            block_nvfp4 & b = blocks[ib];
            for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; s++) {
                const float * v = xrow + ib * QK_NVFP4 + s * QK_NVFP4_SUB;
                uint8_t ue = 0;
                if (policy == 2) {
                    float m1 = 0.0f, m2 = 0.0f;
                    for (int i = 0; i < QK_NVFP4_SUB; i++) {
                        const float a = fabsf(v[i]);
                        if (a > m1) { m2 = m1; m1 = a; } else if (a > m2) { m2 = a; }
                    }
                    ue = ggml_fp32_to_ue4m3(m2 / 6.0f);
                } else {
                    ue = search_nvfp4_scale_mse(v);
                }
                if (ue == 0 || ue == 0x7F) {
                    zscale++;
                } else if (ue == 0x7E) {
                    sscale++;
                }
                encode_nvfp4_subblock(v, &b, s, ue);
            }
        }
        dequantize_row_nvfp4(blocks, yq.data(), ne0);
    }

    Agg & a = agg[tag];
    accum(a, xrow, yq.data(), ne0, zscale, sscale);
}

static void eval_turbo(std::map<std::string, Agg> & agg, const std::string & tag,
                       const float * xrow, int64_t ne0, int bits,
                       const float * s1, const float * s2,
                       block_turbo3_0 * b3, block_turbo4_0 * b4) {
    std::vector<float> rot(xrow, xrow + ne0); // copy for the rotated reference domain
    std::vector<float> yq(ne0);

    for (int64_t g = 0; g < ne0 / 128; g++) {
        probe_fwht_forward_128(rot.data() + g * 128, s1, s2);
    }

    if (bits == 3) {
        quantize_row_turbo3_0_ref(xrow, b3, ne0);
        dequantize_row_turbo3_0(b3, yq.data(), ne0);
    } else {
        quantize_row_turbo4_0_ref(xrow, b4, ne0);
        dequantize_row_turbo4_0(b4, yq.data(), ne0);
    }

    Agg & a = agg[tag];
    accum(a, rot.data(), yq.data(), ne0, 0, 0); // sum_x2 uses the rotated domain (identical norm)
}

/* -------- dump reading -------- */

struct kv_dump_hdr {
    char     magic[4];  // "KVD1"
    uint32_t dst_type;
    uint32_t src_type;
    uint64_t ne0;
    uint32_t rows;
    uint32_t elsize;
};

int main(int argc, char ** argv) {
    const char * dump_path = nullptr;
    const char * csv_path  = "probe_nvfp4_results.csv";
    const char * k_name    = nullptr;
    const char * v_name    = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (!strcmp(argv[i], "--k") && i + 1 < argc) {
            k_name = argv[++i];
        } else if (!strcmp(argv[i], "--v") && i + 1 < argc) {
            v_name = argv[++i];
        } else if (argv[i][0] != '-') {
            dump_path = argv[i];
        }
    }
    if (!dump_path) {
        fprintf(stderr, "usage: probe_nvfp4 <dumpfile> [--csv out.csv] [--k turbo4_0] [--v turbo5_0]\n");
        return 1;
    }

    FILE * f = fopen(dump_path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open dump '%s'\n", dump_path);
        return 1;
    }

    const float * s1 = nullptr;
    const float * s2 = nullptr;
    ggml_turbo_wht_signs(&s1, &s2);

    printf("probe_nvfp4: dump='%s'\n", dump_path);

    struct kv_dump_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));

    long long recs = 0;

    while (fread(&hdr, sizeof(hdr), 1, f) == 1) {
        if (hdr.magic[0] != 'K' || hdr.magic[1] != 'V' || hdr.magic[2] != 'D' || hdr.magic[3] != '1') {
            fprintf(stderr, "bad dump record magic - file corrupted\n");
            break;
        }

        const int64_t ne0  = (int64_t) hdr.ne0;
        const int64_t rows = (int64_t) hdr.rows;
        const size_t  bytes = (size_t) rows * (size_t) ne0 * hdr.elsize;

        if (ne0 % 128 != 0) {
            fprintf(stderr, "record ne0=%lld not 128-aligned - skipping\n", (long long) ne0);
            continue;
        }

        std::vector<char> raw(bytes);
        if (fread(raw.data(), 1, bytes, f) != bytes) {
            fprintf(stderr, "short read\n");
            break;
        }
        recs++;

        std::vector<float> x(rows * ne0);
        if (hdr.src_type == GGML_TYPE_F32) {
            memcpy(x.data(), raw.data(), bytes);
        } else if (hdr.src_type == GGML_TYPE_F16) {
            const uint16_t * h = (const uint16_t *) raw.data();
            for (int64_t i = 0; i < rows * ne0; i++) {
                x[i] = probe_f16_to_f32(h[i]);
            }
        } else {
            fprintf(stderr, "unknown src type %u\n", hdr.src_type);
            continue;
        }

        const char * tname = ggml_type_name((ggml_type) hdr.dst_type);
        if (tname == nullptr) {
            tname = "?";
        }

        const bool is_k = (k_name && !strcmp(tname, k_name)) || (!k_name && !v_name);
        const bool is_v = (v_name && !strcmp(tname, v_name));

        std::vector<block_nvfp4>   blk_rows(rows * (ne0 / QK_NVFP4));
        std::vector<block_turbo3_0> b3(rows * (ne0 / 128));
        std::vector<block_turbo4_0> b4(rows * (ne0 / 128));
        std::vector<float> rot(x.data(), x.data() + rows * ne0); // rotate-in-place copy (must start from x!)

        for (int64_t r = 0; r < rows; r++) {
            const float * xrow = x.data() + r * ne0;
            for (int64_t g = 0; g < ne0 / 128; g++) {
                probe_fwht_forward_128(&rot[r * ne0 + g * 128], s1, s2);
            }

            if (is_k) {
                const std::string p = std::string(tname) + "(K)rot";
                eval_nvfp4_policy(g_agg, p + "|max6", rot.data() + r * ne0, ne0, 0, blk_rows.data() + r * (ne0 / QK_NVFP4));
                eval_nvfp4_policy(g_agg, p + "|mse",  rot.data() + r * ne0, ne0, 1, blk_rows.data() + r * (ne0 / QK_NVFP4));
                eval_nvfp4_policy(g_agg, p + "|2nd",  rot.data() + r * ne0, ne0, 2, blk_rows.data() + r * (ne0 / QK_NVFP4));
                const std::string p2 = std::string(tname) + "(K)raw";
                eval_nvfp4_policy(g_agg, p2 + "|max6", xrow, ne0, 0, blk_rows.data() + r * (ne0 / QK_NVFP4));
            }
            if (is_v) {
                const std::string p = std::string(tname) + "(V)raw";
                eval_nvfp4_policy(g_agg, p + "|max6", xrow, ne0, 0, blk_rows.data() + r * (ne0 / QK_NVFP4));
                eval_nvfp4_policy(g_agg, p + "|mse",  xrow, ne0, 1, blk_rows.data() + r * (ne0 / QK_NVFP4));
                eval_nvfp4_policy(g_agg, p + "|2nd",  xrow, ne0, 2, blk_rows.data() + r * (ne0 / QK_NVFP4));
                const std::string p2 = std::string(tname) + "(V)rot";
                eval_nvfp4_policy(g_agg, p2 + "|max6", rot.data() + r * ne0, ne0, 0, blk_rows.data() + r * (ne0 / QK_NVFP4));
            }
            if (!is_k && !is_v) {
                const std::string p = std::string(tname);
                eval_nvfp4_policy(g_agg, p + "|raw|max6", xrow, ne0, 0, blk_rows.data() + r * (ne0 / QK_NVFP4));
            }

            eval_turbo(g_agg, std::string(tname) + "|turbo3-ref", xrow, ne0, 3, s1, s2,
                       b3.data() + r * (ne0 / 128), b4.data() + r * (ne0 / 128));
            eval_turbo(g_agg, std::string(tname) + "|turbo4-ref", xrow, ne0, 4, s1, s2,
                       b3.data() + r * (ne0 / 128), b4.data() + r * (ne0 / 128));
        }
    }
    fclose(f);

    if (recs == 0 || g_agg.empty()) {
        fprintf(stderr, "no usable dump records found (got %lld)\n", recs);
        return 1;
    }

    FILE * csv = fopen(csv_path, "w");
    if (csv) {
        fprintf(csv, "bucket,rows,elems,rel_mse_pct,flush_ppm,zero_scale_subblocks,sat_scale_subblocks\n");
    }
    printf("\n%-40s %9s %10s %8s %12s %12s\n", "bucket", "rows", "relMSE%", "flush", "zeroScale", "satScale");
    printf("%s\n", std::string(120, '-').c_str());

    for (const auto & kv : g_agg) {
        const Agg & a  = kv.second;
        const double mse = (a.sum_x2 > 0.0) ? 100.0 * a.sum_e2 / a.sum_x2 : 0.0;
        const double fl  = (a.elems > 0) ? 1e6 * (double) a.flush / (double) a.elems : 0.0;
        char line[512];
        snprintf(line, sizeof(line), "%-40s %9lld %10.4f %8.1f %12lld %12lld",
                 kv.first.c_str(), a.rows, mse, fl, a.zscale, a.sscale);
        printf("%s\n", line);
        if (csv) {
            fprintf(csv, "%s,%lld,%lld,%.4f,%.1f,%lld,%lld\n", kv.first.c_str(), a.rows, a.elems, mse, fl, a.zscale, a.sscale);
        }
    }

    if (csv) {
        fclose(csv);
        printf("\nCSV: %s\n", csv_path);
    }
    printf("\nrecords: %lld, distinct buckets: %zu\n", recs, g_agg.size());
    printf("\nnotes: turbo refs use the CPU reference quantizers WITHOUT dither (GGML_TURBO_DITHER only\n"
           "       affects CUDA/KV production path); scale census counts subblocks with UE4M3 code 0x7E\n"
           "       (= the 448 clamp, 'satScale') or decoded-to-zero scale ('zeroScale'); NVFP4-like here is\n"
           "       single-level (per-16 UE4M3 only) - NOT dual-level standard NVFP4.\n");
    return 0;
}
