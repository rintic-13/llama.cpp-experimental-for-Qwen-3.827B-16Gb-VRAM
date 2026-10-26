/*
 * Round-trip test for the ported TurboQuant CPU codec (stage 1).
 * Standalone: compiles ggml-turbo-quant.c directly, no ggml library linkage.
 *
 * Checks:
 *  1. WHT sign arrays bit-identical to the CUDA reference packed words
 *     (TURBO_WHT_SIGNBITS1/2 from ggml-cuda/turbo-quant.cuh).
 *  2. WHT orthonormality (norm preservation) and codec fwht_inverse == analytic inverse.
 *  3. quantize -> dequant round trip IN THE ROTATED DOMAIN: dequant values are meant to be
 *     compared against WHT(x), not x (upstream design: Q is pre-rotated, <Q_rot,K_rot> is
 *     the correct score). Per-coordinate relative RMSE targets from the Lloyd-Max tables:
 *     turbo4 ~0.098 sigma_rel, turbo3 ~0.17, turbo2 ~0.30.
 *  4. Corrected-norm property: dequantized group L2 norm == input group norm.
 *  5. Determinism, all-zero safety, multi-group rows (256 = head_dim 256).
 *  6. Block layout sizes: turbo2=34 B, turbo3=50 B, turbo4=66 B per 128 values.
 */

#ifdef GGML_SHARED
#undef GGML_SHARED   /* static test build: GGML_API = extern */
#endif

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* reference packed sign words from ggml/src/ggml-cuda/turbo-quant.cuh (upstream, unchanged) */
static const unsigned REF_SIGNBITS1[4] = { 0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u };
static const unsigned REF_SIGNBITS2[4] = { 0x16ACEE90u, 0x3F628FDCu, 0xB7A5357Au, 0xBEA56562u };

/* reference float sign arrays from the same file (must match the CPU codec's internal ones) */
static const float REF_S1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
     1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};
static const float REF_S2[128] = {
     1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
     1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

/* local replica of the CPU codec's WHT (structure copy; used as the mathematical reference) */
static void ref_fwht(float * x, int n) {
    for (int i = 0; i < n; i++) x[i] *= REF_S1[i];
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv = (n == 128) ? 0.08838834764831845f : 0.125f;
    for (int i = 0; i < n; i++) x[i] *= inv * REF_S2[i];
}

static void ref_fwht_inv(float * x, int n) {
    for (int i = 0; i < n; i++) x[i] *= REF_S2[i];
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv = (n == 128) ? 0.08838834764831845f : 0.125f;
    for (int i = 0; i < n; i++) x[i] *= inv * REF_S1[i];
}

/* CPU codec entry point (defined by the linked ggml-turbo-quant.c object) */
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

/* xorshift64* PRNG, deterministic */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static double rng_uniform(void) {
    rng_state ^= rng_state >> 30; rng_state *= 0xBF58476D1CE4E5B9ULL;
    rng_state ^= rng_state >> 27; rng_state *= 0x94D049BB133111EBULL;
    rng_state ^= rng_state >> 31;
    return (double)(rng_state >> 11) / (double)(1ULL << 53);
}
static double rng_normal(void) {
    double u1 = rng_uniform(); if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * rng_uniform());
}

static int failures = 0;
static void check(int ok, const char * name) {
    printf("%-52s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) failures++;
}

/* shared measurement: dq is in the ROTATED domain, compare against ref_fwht(x) */
typedef struct { double rmse_rot, rmse_back, norm_err, max_coord_err, max_x_coord; } measure_t;

static measure_t measure(const float * x, const float * dq, int n) {
    measure_t m = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    const int groups = n / 128;
    for (int g = 0; g < groups; g++) {
        const float * xg = x + g * 128;
        const float * dg = dq + g * 128;
        float xr[128];
        memcpy(xr, xg, sizeof(xr));
        ref_fwht(xr, 128);                      // reference forward rotation

        float back[128];
        memcpy(back, dg, sizeof(back));
        turbo_cpu_fwht_inverse(back, 128);      // codec inverse of the dequantized block

        float e1 = 0, e2 = 0, nx = 0, nd = 0;
        for (int i = 0; i < 128; i++) {
            const float d1 = dg[i] - xr[i];
            const float d2 = back[i] - xg[i];
            e1 += d1 * d1;
            e2 += d2 * d2;
            nx += xg[i] * xg[i];
            nd += dg[i] * dg[i];
            if (fabsf(d1) > m.max_coord_err) m.max_coord_err = fabsf(d1);
            if (fabsf(xr[i]) > m.max_x_coord) m.max_x_coord = fabsf(xr[i]);
        }
        m.rmse_rot  += sqrtf(e1 / nx);
        m.rmse_back += sqrtf(e2 / nx);
        const float nrel = fabsf(sqrtf(nd) - sqrtf(nx)) / sqrtf(nx);
        if (nrel > m.norm_err) m.norm_err = nrel;
    }
    m.rmse_rot /= groups;
    m.rmse_back /= groups;
    return m;
}

int main(void) {
    /* ---- 1. sign arrays vs packed CUDA reference ---- */
    int signs_ok = 1;
    for (int w = 0; w < 4; w++) {
        for (int b = 0; b < 32; b++) {
            const int e1 = w * 32 + b;
            const float want1 = (REF_SIGNBITS1[w] & (1u << b)) ? -1.0f : 1.0f;
            const float want2 = (REF_SIGNBITS2[w] & (1u << b)) ? -1.0f : 1.0f;
            if (REF_S1[e1] != want1) signs_ok = 0;
            if (REF_S2[e1] != want2) signs_ok = 0;
        }
    }
    check(signs_ok, "1a. ref sign arrays match packed CUDA signbits");

    {
        float a[128], b[128];
        memset(a, 0, sizeof(a)); a[0] = 1.0f;
        memcpy(b, a, sizeof(b));
        turbo_cpu_fwht_inverse(a, 128);
        ref_fwht_inv(b, 128);
        float max_diff = 0.0f;
        for (int i = 0; i < 128; i++) {
            float d = fabsf(a[i] - b[i]);
            if (d > max_diff) max_diff = d;
        }
        check(max_diff < 1e-6f, "1b. CPU fwht_inverse == CUDA-order reference (e0 probe)");
    }

    /* ---- 2. orthonormality + inverse correctness ---- */
    {
        int ort_ok = 1, inv_ok = 1;
        for (int t = 0; t < 2000; t++) {
            float x[128], y[128];
            for (int i = 0; i < 128; i++) { x[i] = (float)rng_normal(); y[i] = x[i]; }
            ref_fwht(y, 128);
            float nx = 0, ny = 0;
            for (int i = 0; i < 128; i++) { nx += x[i]*x[i]; ny += y[i]*y[i]; }
            if (fabsf(ny - nx) / nx > 2e-5f) ort_ok = 0;
            ref_fwht_inv(y, 128);
            for (int i = 0; i < 128; i++) {
                float d = fabsf(y[i] - x[i]) / (1.0f + fabsf(x[i]));
                if (d > 1e-5f) inv_ok = 0;
            }
        }
        check(ort_ok, "2a. WHT preserves vector norm (orthonormal)");
        check(inv_ok, "2b. reference inverse == forward^-1");

        int codec_inv_ok = 1;
        for (int t = 0; t < 200; t++) {
            float x[128], a[128], b[128];
            for (int i = 0; i < 128; i++) x[i] = (float)rng_normal();
            memcpy(a, x, sizeof(a)); memcpy(b, x, sizeof(b));
            turbo_cpu_fwht_inverse(a, 128);
            ref_fwht_inv(b, 128);
            for (int i = 0; i < 128; i++) {
                if (fabsf(a[i] - b[i]) > 1e-4f * (1.0f + fabsf(b[i]))) codec_inv_ok = 0;
            }
        }
        check(codec_inv_ok, "2c. codec turbo_cpu_fwht_inverse == reference");
    }

    /* ---- turbo4 ---- */
    {
        const int nrow = 4000;
        measure_t tot = { 0, 0, 0, 0, 0 };
        for (int row = 0; row < nrow; row++) {
            const int n = (row % 2 == 0) ? 128 : 256;   // single- and dual-group rows (head_dim 256)
            float x[256], dq[256];
            block_turbo4_0 q[2];
            for (int i = 0; i < n; i++) x[i] = (float)rng_normal();
            quantize_turbo4_0(x, q, 1, n, NULL);
            dequantize_row_turbo4_0(q, dq, n);
            measure_t m = measure(x, dq, n);
            tot.rmse_rot += m.rmse_rot; tot.rmse_back += m.rmse_back;
            if (m.norm_err > tot.norm_err) tot.norm_err = m.norm_err;
            if (m.max_coord_err > tot.max_coord_err) tot.max_coord_err = m.max_coord_err;
            if (m.max_x_coord > tot.max_x_coord) tot.max_x_coord = m.max_x_coord;
        }
        tot.rmse_rot /= nrow; tot.rmse_back /= nrow;
        printf("  turbo4: rel RMSE rot=%.4f back=%.4f (Lloyd-Max 4-bit ~0.098), max norm err=%.4f, max |coord err|/max|coord x|=%.3f\n",
               tot.rmse_rot, tot.rmse_back, tot.norm_err, tot.max_coord_err / tot.max_x_coord);
        check(tot.rmse_rot < 0.115 && tot.rmse_back < 0.115, "3a. turbo4 RMSE within bounds");
        check(tot.norm_err < 0.01, "4a. turbo4 corrected norm preserved (<1%)");
        check(tot.max_coord_err < 0.75f * tot.max_x_coord, "5a. turbo4 coord outliers bounded (<0.75 of max |x|)");

        float x[128]; block_turbo4_0 a[1], b[1];
        for (int i = 0; i < 128; i++) x[i] = (float)rng_normal();
        quantize_row_turbo4_0_ref(x, a, 128);
        quantize_row_turbo4_0_ref(x, b, 128);
        check(memcmp(a, b, sizeof(a)) == 0, "5b. turbo4 quantize deterministic");

        memset(x, 0, sizeof(x));
        quantize_row_turbo4_0_ref(x, a, 128);
        float y[128];
        dequantize_row_turbo4_0(a, y, 128);
        int allzero = 1;
        for (int i = 0; i < 128; i++) if (y[i] != 0.0f) allzero = 0;
        check(allzero, "5c. turbo4 all-zero row -> all-zero dequant");
    }

    /* ---- turbo3 ---- */
    {
        const int nrow = 4000;
        measure_t tot = { 0, 0, 0, 0 };
        for (int row = 0; row < nrow; row++) {
            const int n = (row % 2 == 0) ? 128 : 256;
            float x[256], dq[256];
            block_turbo3_0 q[2];
            for (int i = 0; i < n; i++) x[i] = (float)rng_normal();
            quantize_turbo3_0(x, q, 1, n, NULL);
            dequantize_row_turbo3_0(q, dq, n);
            measure_t m = measure(x, dq, n);
            tot.rmse_rot += m.rmse_rot; tot.rmse_back += m.rmse_back;
            if (m.norm_err > tot.norm_err) tot.norm_err = m.norm_err;
            if (m.max_coord_err > tot.max_coord_err) tot.max_coord_err = m.max_coord_err;
        }
        tot.rmse_rot /= nrow; tot.rmse_back /= nrow;
        printf("  turbo3: rel RMSE rot=%.4f back=%.4f (Lloyd-Max 3-bit ~0.17), max norm err=%.4f, max coord err=%.3f\n",
               tot.rmse_rot, tot.rmse_back, tot.norm_err, tot.max_coord_err);
        check(tot.rmse_rot < 0.19 && tot.rmse_back < 0.19, "3b. turbo3 RMSE within bounds");
        check(tot.norm_err < 0.01, "4b. turbo3 corrected norm preserved (<1%)");
    }

    /* ---- turbo2 ---- */
    {
        const int nrow = 4000;
        measure_t tot = { 0, 0, 0, 0 };
        for (int row = 0; row < nrow; row++) {
            const int n = (row % 2 == 0) ? 128 : 256;
            float x[256], dq[256];
            block_turbo2_0 q[2];
            for (int i = 0; i < n; i++) x[i] = (float)rng_normal();
            quantize_turbo2_0(x, q, 1, n, NULL);
            dequantize_row_turbo2_0(q, dq, n);
            measure_t m = measure(x, dq, n);
            tot.rmse_rot += m.rmse_rot; tot.rmse_back += m.rmse_back;
            if (m.norm_err > tot.norm_err) tot.norm_err = m.norm_err;
            if (m.max_coord_err > tot.max_coord_err) tot.max_coord_err = m.max_coord_err;
        }
        tot.rmse_rot /= nrow; tot.rmse_back /= nrow;
        printf("  turbo2: rel RMSE rot=%.4f back=%.4f (Lloyd-Max 2-bit ~0.30), max norm err=%.4f, max coord err=%.3f\n",
               tot.rmse_rot, tot.rmse_back, tot.norm_err, tot.max_coord_err);
        check(tot.rmse_rot < 0.36 && tot.rmse_back < 0.36, "3c. turbo2 RMSE within bounds");
        check(tot.norm_err < 0.01, "4c. turbo2 corrected norm preserved (<1%)");
    }

    /* ---- 6. block layout ---- */
    {
        check(sizeof(block_turbo2_0) == 34, "6a. sizeof(block_turbo2_0)==34 (2.125 bpw)");
        check(sizeof(block_turbo3_0) == 50, "6b. sizeof(block_turbo3_0)==50 (3.125 bpw)");
        check(sizeof(block_turbo4_0) == 66, "6c. sizeof(block_turbo4_0)==66 (4.125 bpw)");
        check(QK_TURBO2 == 128 && QK_TURBO3 == 128 && QK_TURBO4 == 128, "6d. QK_TURBO* == 128");
        check(GGML_TYPE_TURBO2_0 == 44 && GGML_TYPE_TURBO3_0 == 45 && GGML_TYPE_TURBO4_0 == 46,
              "6e. enum ids 44/45/46");
    }

    printf("\n%s (%d failures)\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
