# TURBO5: отказ от shared sign-mask, независимые маски по блокам (2026-11-08)

## Решение пользователя
> «Убрать Shared sign-mask из основного Turbo5. Оставить независимые маски по блокам.»

turbo3/turbo4 остаются на общих `turbo_cpu_s1/s2` (`TURBO_WHT_SIGNS1/2`) — их поведение не менялось.

## Мотивировка (замер, frozen в TURBO5_SCALE_SEARCH_20261107.md, Gate 3)
Анализатор `kv_mask_corr.exe` на 200 реальных K-записях (204800 128-блоков × 8 head-slices,
реальные Q из `ppl_eval\Q_real.f32`):

| вариант | relL2 | channel bias ‖μ‖ (ratio к indep) | score coherence mean/max |
|---|---|---|---|
| A shared-mask t5 (старый production) | 0.04403 | 1.26x | 1.14 / 4.00 |
| B per-block full masks | 0.04405 | 1.02x | 0.68 / 2.77 |
| C sign-sandwich flipfold | 0.04520 | 1.98x | 1.67 / 5.86 |

Общий MSE идентичен, но general-mask выравнивает ошибки блоков: систематическое
смещение каналов (1.26x) и когерентное искажение attention-скоров (до 4x).
Per-block маски дают статистически независимые локальные ошибки.

**Вариант C отвергнут**: сворачивание per-block flip'a в 5-bit индекс (`flip5(idx)=32-idx`)
в shared-фрейме ХУЖЕ обоих (1.98x bias) — 5-bit решётка центров асимметрична
(уровень −0.262203 не имеет отрицания в таблице → клиппинг), flips не компенсируют
асимметрию. Никакой index folding в финальном дизайне не используется.

## Финальный дизайн (production)
Каждый 128-блок квантуется в СВОЁМ signed-Hadamard фрейме

    G_b = D(s2_b) * H/sqrt(128) * D(s1_b)

- **seed**: 16 бит, хранится в блоке. Выводится из RAW fp32-бит блока commutative u64
  sum + splitmix64 (`ggml_turbo5_seed_bits`), хелперы `ggml_turbo5_sm64/sign1/sign2` —
  в `ggml/src/ggml-common.h` как `GGML_TURBO5_INLINE` (C: static inline, CUDA:
  __device__ __forceinline__) → CPU- и CUDA-энкодеры из одинаковых входных байт
  получают одинаковый seed (byte mirror gate остаётся bit-exact); блок сам-описывающий
  (defrag/cache-shift безопасны — seed едет в байтах).
- **block_turbo5_0**: norm(f16) + **seed(u16)** + qs(64) + signs(16) = **84 B**
  (было 82 B; 5.25 bpw вместо 5.125 — сравнимо с q5_1). static_assert обновлён.
- **Reader-путь выводит RAW-домен**: все де-кванты делают G_b^-1 (v*=s2; FHT;
  *=inv√128·s1). Нормаль остаётся z-frame (`grp_norm/inner`) — корректно, т.к. inner
  inвариантен до знаковой симметрии.
- **Граф больше НЕ вращает ничего для turbo5**: attn_rot_k/v для turbo5 = false
  (`llama_cache_type_is_turbo_shared` в `llama-impl.h` ограничивает shared-типы),
  signed-матрицы attn_rot_wht_fwd/inv строятся только для shared-типов, self_k_rot /
  self_v_rot для turbo5 не создаются → Q идёт raw, FA-скор = <q_raw, G_b^-1 k_store>,
  attention output уже raw. Norm-зависимая FA math не затронута.

### Реализованные точки
- `ggml/src/ggml-common.h`: seed-поле + u64-хелперы (единые для прод-кодека и gate-тулз).
- `ggml/src/ggml-turbo-quant.c`: `turbo_cpu_fwht_masked` / `turbo_cpu_fwht_inverse_masked`;
  `quantize_row_turbo5_0_ref` (seed + per-block поиск в своём фрейме; обе ветки — search и
  legacy), `dequantize_row_turbo5_0` → RAW.
- `set-rows.cu k_set_rows_turbo5`: seed редукцией warp-shuffle из shared `x[]` (raw bits),
  маски через `ggml_turbo5_sign1/2(seed16, j)`, `blk->seed` пишется в обеих ветках.
- `fattn-common.cuh`: `vec_dot_fattn_vec_KQ_turbo5_0` — warp-FHT де-ротация K (lane владеет
  4 элементами j=4·lane+a блока: интред-бабочки h=1,2 + shuffle-стадии m=1,2,4,8,16=h=4..64),
  `dequantize_V_turbo5_0` — та же схема, float staging, raw output; assert nthreads==WARP_SIZE
  для turbo5 (D=64 turbo-инстансов не существует).
- `fattn-vec.cuh`: turbo5 убран из `kq_f16_build` (fused raw-afrag staging не может работать
  с per-block фреймами) → decode идёт fp32-dotом `vec_dot_fattn_vec_KQ_turbo5_0` с in-warp FHT;
  cp.async и prefetch turbo5 и так не касались; статические assert V-lane-layout
  (nthreads_V == WARP_SIZE && nthreads_V*V_rows_per_thread == QK_TURBO5).
- `fattn.cu` + `template-instances/fattn-vec-instance-q5_1-turbo5_0.cu`: turbo5 V-кейсы только
  D=128/256 (D=64 даёт nthreads_V=16 и ломает warp-FHT layout; llama и так ABORTит turbo при
  head_dim%128≠0). turbo5-K комбо (t5/t3,t5/t4,t5/t5) были 128/256 уже раньше.
- `convert.cu`: новые ядра `dequantize_block_turbo5_cont` / `_rowwise` (warp на блок,
  warp-FHT) заменили все 4 turbo5-dispatch случая (to_fp16/to_fp32/cont/rowwise);
  mmfa/prefill f16-staging теперь raw.
- `llama-impl.h` / `llama-kv-cache.cpp`: `llama_cache_type_is_turbo_shared` (t3q/t3/t4/t4q),
  attn_rot_k/v и signed-матрицы gated по нему; turbo5 исключён из generic quantized
  rotation-правил (как nvfp4).

## Гейты (все закрыты, 2026-11-08)

1. **CPU parity** — тестовых кейсов turbo5 в `tests/test-turbo-quant.c` форк не имеет
   (turbo5 туда никогда не добавлялся); CPU-референс покрыт гейтом 2 (тот же
   `quantize_row_turbo5_0_ref` побитово сверяется с GPU-энкодером) и гейтом 3
   (variant D = production-кодек, round-trip relL2 0.04405). — PASS (coverage note)
2. **KVG1 byte mirror** — lease-прогон `-ctk turbo5 -ctv turbo5 -c 1024 -b 512 --chunks 2`
   (`ppl_results\t5perblock_gate_20261108\{kv_dump,gpu_dump}.bin`, 6 вызовов set_rows,
   24576 блоков) + `verify_t5mirror.exe`, два прогона, идентичный вывод:
   **seed mismatches = 0** (CPU/GPU выбирают identical пер-блокные маски из identical байтов),
   qs mismatch 5/24576, signs mismatch 1/24576 (1 sign-bit), norm mismatch 16389/24576
   (fp16 last-ulp от разного порядка суммирования inner/rsq — победитель поиска тот же,
   idx overlap 128/128). real gap = 0, max rel gap = 2.0e-8. **Verdict: PASS**
   (near-tie scale drift, документированный ещё в эпоху scale-search).
   Логи: `ppl_results\t5perblock_gate_20261108\A1_dump_run.log`.
3. **kv_mask_corr variant D** (production seed-формулы прямо из ggml-common.h, реальные
   K-блоки 204800 шт): relL2 0.04405 (≈B 0.04405), channel bias **1.01x** (A: 1.26x),
   score coherence mean **0.84** / max 3.82 (A: 1.14/4.00, B: 0.68/2.77). Максимум чуть выше B
   (16-бит seed вместо 64-бит хэша маски) — в пределах нормы. **PASS** — систематика
   shared-маски устранена, MSE не изменился. Лог: `Temp\opencode\maskcorr_1108.log`.
4. **8K PPL** (lease-батч B, frozen protocol `-c 8192 --chunks 1 -b 1 -ub 1 -s 42`,
   файл `wiki.test.quick.txt`; логи `ppl_results\t5perblock_gate_20261108\B{1,2}_ppl_*.log`):

   | ряд | PPL | сравнение |
   |---|---|---|
   | **t5/t5 (new)** | **7.2381 ± 0.275** | лучше старого t5/t3 (7.2863) на 0.048; лучше t4/t4 (7.2465) на 0.008; ниже gold (7.2415) |
   | **t5/t3 (new)** | **7.2610 ± 0.276** | лучше старого t5/t3 (7.2863) на 0.025; чуть хуже t4/t3 (7.2483) |

   t5/t5 впервые обгоняет t4-rows того же дня — independent masks дали реальный,
   а не только статистический выигрыш (пер-блокные ошибки не складываются в скорах).
5. **Decode cost per-block FHT** (lease-батч A, `llama-bench -d 8192 -p 512 -n 128 -r 3`,
   лог `A2_bench.log`): q5_1/turbo5 tg128 = **23.29 ± 0.04** против q5_1/turbo4
   tg128 = **29.09 ± 0.06** (тот же protocol, тот же билд) → пер-блокная inverse-FHT
   де-ротация V в FA vec стоит ≈ **-20% decode t/s**; pp512 не изменился
   (890.6 vs 889.0) — АННУЛИРОВАНО в тот же день: mmfa+raw turbo5 = NaN (E1); честные pp/tg
   turbo5-рядов — в разделе «FA routing fix» ниже. Старый shared-mask turbo5 (20261031,
   другой protocol d1024/tg8) = 30.05 t/s. Цена принята ради независимости масок;
   путь оптимизации — signed-диагональ, свёрнутая в centroid-lookup (не в этой сессии).

## Память
turbo5-блок 82 → 84 B (5.25 bpw, parity с q5_1). Измерено в A1-прогоне против того же
прогона 20261107 (shared-mask билд, та же команда c=1024): KV buffer 20.50 → **21.00 MiB**
(+2.44% ровно по 84/82; на 8K контекст ≈ +4 MiB на всю модель). принято как цена
независимых масок.

## FA routing fix (2026-11-08, батчи E–H): turbo5 → векторное ядро

**Симптом (пользовательский):** `llama-server -ctk turbo5 -ctv turbo4 -b 2048 -ub 512
--spec-type draft-mtp` выдавал литеральные `//////////`. Воспроизведено в E2 (`llama-cli`
t5/t4 ub512 → весь вывод слэши, `E2_gen_t5_t4_ub512.log`).

**Корень (E1, `E1_ppl_t5_t5_stageparams.log`):** per-block-mask редизайн сделал turbo5-читатели
RAW-домена; сырые K/V-координаты Qwen-класса достигают 1e4..1e5. Ядра mmfa/tile (путь ub>1)
не читают квантованный KV нативно: `launch_fattn` стейджит их через f16-скретч —
`max=65504.0` (f16-потолок), `nan=274/32768`, `inf=1`; второй FA-вызов уже весь NaN
(`nan=32768`). Старый shared-mask turbo5 хранился повёрнутым (малые значения) — mmfa работал
до редизайна. Vec-ядро turbo5 не стейджит (fp32-dot `vec_dot_fattn_vec_KQ_turbo5_0`,
fp32 `dequantize_V_turbo5_0`) — поэтому ub=1 PPL-гейты ряда B проходили чисто и ломался
только batch-путь.

**Фикс 1 (`fattn.cu`, `ggml_cuda_get_best_fattn_kernel`):** если K или V = turbo5_0 — vec-ядро
на любом n_q; если инстанса нет — NONE (FA unsupported); mmfa/tile-стейджинг turbo5 запрещён.

**Находка F2/F3 (батч F, lease `20261108-040850-ppl-7668`):** combo t5/t4 после фикса 1 —
speed-gate vec-таблицы (`ggml_cuda_get_fattn_vec_case`: turbo4 на любой стороне → nullptr в
dither-сборке) блокировал vec → NONE → llama разбила FA-узлы на CPU0 (`sched: splits = 34`,
CUDA_Host буфер 245 MiB), а CPU FA вызвал `ggml_type_traits_cpu(turbo).vec_dot == NULL`
→ access violation по адресу 0 (0xC0000005, модуль «unknown»; для сравнения GGML_ABORT E3 —
0xC0000409/ucrtbase).

**Фикс 2 (`fattn.cu`, gate):** combos с turbo5 на любой стороне исключены из turbo4/FALLBACK
speed-gate (для чистых turbo3/turbo4-рядов гейт сохранён).

**Фикс 3 (инстансы + CMake):** добавлены недостающие vec-инстансации turbo5-комбо —
`fattn-vec-instance-turbo4_0-turbo5_0.cu` и `fattn-vec-instance-turbo5_0-q5_1.cu`
(+ `FATTN_VEC_CASE(128/256,…)` в fattn.cu); `GGML_CUDA_FA_QUANTS` расширен в кэше сборки
(`reconfigure_H.log`). Комбо без инстанса (turbo5+nvfp4, turbo5+q8_0, …) — официально
unsupported: **фикс 4** — guard-assert в `ggml-cpu/ops.cpp`
(`fattn: CPU FA has no vec_dot for this K-type`) вместо segfault по NULL.

**Проверка (батч H, lease `20261108-043150-ppl-2824`, `GGML_FATTN_DEBUG=1`,
c1024 b512 ub512 chunks 1 no-warmup):** 6/6 строк exit 0, все combo `turbo5 routing … ->
VEC(native)`, `splits = 2` (CPU-сплитов нет):

| ряд | PPL (первый чанк) | роутинг |
|---|---|---|
| H1 t5/t4 (пользовательский combo) | **3.8610 ± 0.396** | VEC(native) |
| H3 t4/t5 (новая инстанса) | 3.8563 ± 0.398 | VEC(native) |
| H4 t5/q5_1 (новая инстанса) | 3.8598 ± 0.398 | VEC(native) |
| H5 q5_1/t5 | 3.8689 ± 0.399 | VEC(native) |
| H6 t5/t5 (= F1, регрессия) | 3.8812 ± 0.401 | VEC(native) |

H2: `llama-cli --temp 0 -n 48 -ub 512` t5/t4 генерирует связный текст
(«The capital of France is Paris.») вместо слэшей.

**Честная скорость (батч G, lease `20261108-043526-bench-5732`,
`llama-bench -d 8192 -p 512 -n 128 -r 3 -ub 512`):**

| combo | pp512 t/s | tg128 t/s |
|---|---|---|
| t5/t5 | 63.5 ± 0.01 | 18.75 ± 0.04 |
| t5/t4 | 96.8 ± 0.07 | 22.03 ± 0.04 |
| t5/t3 | 103.3 ± 0.19 | 22.38 ± 0.03 |
| q5_1/q5_1 (mmfa, контроль) | **893.0 ± 10.3** | 31.20 ± 0.22 |

→ turbo5-комбо платят за корректность: prefill в ~9–14× медленнее mmfa-пути (vec-ядро —
скалярный FHT-dot; turbo5 исключён из kq_f16_build 2026-11-08, т.к. 16-координатный layout
не делает warp-wide inverse FHT). Числа pp512 гейта 5 (≈890 t/s) аннулированы — измерены на
NaN-mmfa пути. Направления возврата скорости (отдельная задача): fused KQ^T MMA по
per-block-фрагментам turbo5 либо f32-скретч-вариант mmfa.
ЗАМЕНЕНО ниже: «Staging fix» (батчи J–L) — скорость mmfa восстановлена честно.

## Staging fix (2026-11-08, батчи J–L): корень NaN — cont-стейджинг, скорость mmfa возвращена

**Гипотеза, проверенная по коду:** «RAW-доменные величины turbo5 не влезают в f16» — неверна
(реальные активации ≤10.8, см. L1 ниже). Разбор показал:
1. `ggml_is_contiguously_allocated(K)` вернула true в E1 **законно**: strides вью unified-KV
   вложены плотно (nb2=168=ne0·ts/bs, nb1=672=ne2·nb2) — K-срезы лежат подряд, дыр нет.
   Но вью при этом **permuted** (nb1 > nb2: порядок хранения [token][head]), а ветка
   `launch_fattn` выбирает плоский конвертер только по span-проверке, не глядя на порядок.
2. Плоский cont-конвертер turbo5 в сборке那一刻 E1 выдал мусор: отпечаток `max=65504,
   nan=274/32768, inf=1` = случайные байты как поле `norm` (f16) — например чтение блоков с
   шагом старого 82-байтного layout вместо новых 84 B сдвигает `norm` на байты `qs` начиная со
   второго блока. Первый блок (vals[0..3]) при этом совпадает с корректным — и он совпадает
   в E1 и L (`0.31152 0.47974 -1.0684 -0.6084`). Нативный же путь vec читает блоки по
   записанным адресам — поэтому ub=1 гейты были чисты.
3. NC-strided лаунчер (`to_fp16_nc_cuda`, proven путь всех bs=32-квантов) стейджит чисто —
   это и есть то, что должно нести turbo5.

**Фикс (`fattn-common.cuh`, K и V staging):** cont-плоская ветка только при
`ggml_is_contiguously_allocated(X) && !ggml_is_permuted(X)` — permuted-вью unified-кэша
(nb2 < nb1) уходят в nc-strided лаунчер. Это закрывает и латентный случай для других bs=128
и bf16 unified-KV. Flat-конвертер turbo5 (bs=128) остаётся непроверенным на KV-вью — guard
держит turbo5 на доказанном nc-пути; если захотим вернуть cont, сперва unit-тест
GPU-dequant против CPU `dequantize_row_turbo5_0`.

**Routing (`fattn.cu`) смягчён:** спец-блок turbo5 теперь только для decode (n_q ≤ 2):
vec-инстанс (с защитой NONE для combos без инстанса); prefill (n_q > 2) падает в обычный
MMA_F16 с исправленным nc-стейджингом.

**Доказательства (lease `20261108-142815-custom-10048` = J, `20261108-143511-custom-5004` = K,
`20261108-145147-custom-13992` = L):**

L1 (PPL t5/t5 c1024 ub512, `GGML_FATTN_DEBUG=1`, лог `L1_ppl_t5_t5_debug_stagefix.log`):
```
FATTN_DEBUG: kernel=mma_f16 K=turbo5 V=turbo5 D=256 nq=512 gqa=6 vec_compiled=1
FATTN_DEBUG: nc-stage K: type=turbo5 ne={256,512,4,1} nb={84,672,168,688128} bs=128 ts=84 s={8,2,8192} cont=1
FATTN_DEBUG: staged K_f16 src=turbo5 … nan=0 inf=0 max=6.6641   (E1: nan=274 inf=0 max=65504)
FATTN_DEBUG: staged V_f16 src=turbo5 … nan=0 inf=0 max=8.0156   (E1: nan=257 inf=1 max=65440)
Final estimate: PPL = 3.8571 ± 0.39711
```
Реальные激活 turbo5 лежат в единицах (max ≤ 10.8 на всех сэмплированных чанках) — f16-скретч
безопасен с запасом. `cont=1` в логе подтверждает: span-проверка по-прежнему обманывается,
guard по permuted и уводит в nc.

**Скорость (J3, `llama-bench -d 8192 -p 512 -n 128 -r 2 -ub 512`, лог
`J3_bench_t5_ub512_stagefix.log`):**

| combo | pp512 t/s | tg128 t/s | против G |
|---|---|---|---|
| t5/t5 | **875.33 ± 6.66** | 18.76 ± 0.02 | 63.5 → ×13.8 |
| t5/t4 | **877.96 ± 4.99** | 22.04 ± 0.04 | 96.8 → ×9.1 |

Контроль тех же дней: t4/t4 819.27, t4/q5_1 823.42, q5_1/q5_1 893.0. Staged-f16 mmfa для
turbo5 вышел даже быстрее fused-turbo4 mmfa — цена стейджинга ub512 компенсируется чистым
f16-тензорным ядром. tg128 не изменился (decode остаётся на vec-инстансах turbo5 —
спец-блок n_q≤2; для decode стейджинг всего KV на токен был бы катастрофой).

**PPL mmfa-пути (J1/J4, c1024 ub512):** t5/t5 **3.8571**, t5/t4 **3.8709** — против vec-рядов
H (3.8812 / 3.8610): различия ≤0.024 в обе стороны — разница численных путей
(f16-staged attention vs fp32 vec-dot), обе корректны.

**Полный 8K гейт на ub512 (батч M, lease `20261108-145801-custom-3148`, протокол B но
`-b 512 -ub 512`, файл `wiki.test.quick.txt`):**

| ряд | PPL (8K ub512) | PPL (8K ub=1, B) | gold |
|---|---|---|---|
| M1 t5/t4 (пользовательский combo) | **7.2607 ± 0.277** | — | 7.2415 |
| M2 t5/t5 | **7.2425 ± 0.276** | 7.2381 | 7.2415 |

mmfa-путь воспроизводит качество vec-пути (Δ t5/t5 = 0.0044 — численный шум путей
f16-staged vs fp32-vec). Пользовательский combo чуть выше gold (+0.019) — как и все
t5-rows с turbo4-V; t5/t5 практически на gold.

**Методическая ловушка (K):** новый llama-cli (TUI) давит ggml-предупреждения —
`GGML_FATTN_DEBUG` с ним молчит (в логе нет ни одной `0.xxx` строки). Для routing/stage-статист
ики использовать llama-perplexity / llama-bench.

**Итог для сервера пользователя:** `-ctk turbo5 -ctv turbo4 -ub 512` теперь и корректен, и
быстрый (pp ≈ 878); после подмены `ggml-cuda.dll` (08.11 14:28) — перезапустить llama-server.
