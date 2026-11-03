// fa_nvfp4_dot_intdiff.cu - exhaustive+random host check that the new SIMD integer decode in
// vec_dot_fattn_vec_KQ_nvfp4 yields EXACTLY the old kvalues_fp4 int8 kv[] values, the same
// integer doti (dp4a) and the same integer sumc for every input. Same ints + unchanged float
// tail => bitwise-identical dot output (PPL unchanged).
#include <cstdio>
#include <cstdint>
#include <cstdlib>

static const int8_t kvalues_fp4[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };

static inline uint32_t host_prmt8(uint32_t a, uint32_t b, uint32_t s) { // sel nibbles assumed < 8
    uint32_t bytes[8] = { a&0xFF,(a>>8)&0xFF,(a>>16)&0xFF,(a>>24)&0xFF,
                          b&0xFF,(b>>8)&0xFF,(b>>16)&0xFF,(b>>24)&0xFF };
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= bytes[(s >> (8*i)) & 7u] << (8*i);
    return r;
}

static int check_word(uint32_t b, int hi, uint32_t qword) {
    int kv[4];
    for (int t = 0; t < 4; ++t) {
        const uint8_t byte = (uint8_t)((b >> (8*t)) & 0xFFu);
        const uint8_t nib  = (uint8_t)((byte >> (hi ? 4 : 0)) & 0xF);
        kv[t] = kvalues_fp4[nib];
    }
    const int      sh  = hi ? 4 : 0;
    const uint32_t bs  = b >> sh;
    const uint32_t sel = bs & 0x07070707u;
    const uint32_t sn4 = bs & 0x08080808u;
    const uint32_t u4  = host_prmt8(0x03020100u, 0x0C080604u, sel);
    uint32_t m4 = 0;
    for (int t = 0; t < 4; ++t) m4 |= (((sn4 >> (8*t)) & 0xFFu) ? 0xFFu : 0u) << (8*t);
    uint32_t v4 = 0;
    for (int t = 0; t < 4; ++t) {
        const uint32_t mm = (m4 >> (8*t)) & 0xFFu;
        v4 |= ((((u4 >> (8*t)) ^ mm) - mm) & 0xFFu) << (8*t);        // mod-256 two's complement
    }
    for (int t = 0; t < 4; ++t) {
        const int vn = (int)(int8_t)((v4 >> (8*t)) & 0xFFu);
        if (vn != kv[t]) { printf("FAIL decode b=%08X hi=%d t=%d old=%d new=%d\n", b, hi, t, kv[t], vn); return 1; }
    }
    // dp4a(u, v4) == old scalar integer sum (both exact ints)
    int doti_old = 0, doti_new = 0, sumc_old = 0, sumc_new = 0;
    for (int t = 0; t < 4; ++t) {
        const int8_t q = (int8_t)((qword >> (8*t)) & 0xFFu);
        doti_old += q * kv[t];
        doti_new += q * (int)(int8_t)((v4 >> (8*t)) & 0xFFu);
        sumc_old += kv[t];
    }
    // emulate __vadd4(v4, v4>>16) then __vadd4(s2, s2>>8) byte-lane wraps:
    uint32_t a2 = 0, a4 = 0;
    for (int t = 0; t < 4; ++t) {
        const uint32_t l = (v4 >> (8*t)) & 0xFFu;
        const uint32_t r = t + 2 < 4 ? (v4 >> (8*(t+2))) & 0xFFu : 0u;
        a2 |= ((l + r) & 0xFFu) << (8*t);
    }
    for (int t = 0; t < 4; ++t) {
        const uint32_t l = (a2 >> (8*t)) & 0xFFu;
        const uint32_t r = t + 1 < 4 ? (a2 >> (8*(t+1))) & 0xFFu : 0u;
        a4 |= ((l + r) & 0xFFu) << (8*t);
    }
    sumc_new = (int)(int8_t)(a4 & 0xFFu);
    if (doti_new != doti_old) { printf("FAIL dot b=%08X hi=%d q=%08X old=%d new=%d\n", b, hi, qword, doti_old, doti_new); return 1; }
    if (sumc_new != sumc_old) { printf("FAIL sumc b=%08X hi=%d old=%d new=%d\n", b, hi, sumc_old, sumc_new); return 1; }
    return 0;
}

int main() {
    srand(99);
    // 1) every single nibble value in every lane position (replicated byte, both halves)
    for (uint32_t nib = 0; nib < 16; ++nib) {
        for (int hi = 0; hi < 2; ++hi) {
            const uint32_t byteval = nib << (hi ? 4 : 0);
            const uint32_t b = byteval * 0x01010101u;
            for (int qq = 0; qq < 64; ++qq) {
                const uint32_t qword = ((uint32_t)rand()<<17) ^ ((uint32_t)rand()<<3) ^ (uint32_t)rand();
                if (check_word(b, hi, qword)) return 1;
            }
        }
    }
    // 2) random full words
    for (int it = 0; it < 2000000; ++it) {
        const uint32_t b = ((uint32_t)rand()<<17) ^ ((uint32_t)rand()<<3) ^ (uint32_t)rand();
        const uint32_t qword = ((uint32_t)rand()<<17) ^ ((uint32_t)rand()<<3) ^ (uint32_t)rand();
        if (check_word(b, rand() & 1, qword)) return 1;
    }
    printf("PASS: SIMD integer decode == kvalues_fp4 LUT (values, doti, sumc) on all tested inputs\n");
    return 0;
}
