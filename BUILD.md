# BUILD.md — сборка нашей ветки llama.cpp (fork) на Windows / RTX 5060 Ti

Проверенный рецепт сборки ветки **точно в том виде, в котором она работает сейчас** (состояние
исходников на 26.10.2026, сборка `build-x64-windows-msvc-release`, `ggml-cuda.dll` от 26.10.2026 15:25,
54 935 552 байт). Все параметры ниже сверены с рабочим `CMakeCache.txt`, а наличие кернелов
подтверждено разборомobj/DLL (см. §«Проверка»).

---

## 0. Что нужно скопировать

Дерево **не является git-репозиторием** (`llama.cpp version: 0.5.0-dev`, `ggml 0.25.3`, git-описания нет).
В нём живёт множество правок форка, помеченных маркерами `[TAG_...]`, без них ветка не воспроизводится
из upstream:

- `[TAG_FFN_CPU_SPLIT]` — механизм `--ffn-cpu-count` (host-resident доноры dense FFN + CPU worker):
  `common/arg.cpp`, `common/common.h`, `src/llama*.cpp|h`, `ggml/src/ggml-backend.cpp`,
  `ggml/src/ggml-cpu/*` (fused multi-column q4_K/IQ3_S dots для верификации MTP).
  Механизм runtime-разреза FFN (`--ffn-host-rows` / `--ffn-host-type` / `--ffn-cpu-rows`) и тип
  `IQ3_S_FLAT3` **вырезаны** (фаза 2B, 26.10; протокол: `tmp_sched_test\ffn_cpu_split_notes.md` §13.73);
- кванты **turbo3_0 / turbo4_0** (типы `GGML_TYPE_TURBO3_0`/`GGML_TYPE_TURBO4_0`) + их FA-ядра
  (`ggml/src/ggml-cuda/fattn-vec.cuh`, `fattn.cu`, `template-instances/fattn-vec-instance-*`);
- `[TAG_ALLOC_SIZE_EXPAND]` — `ggml/src/ggml-backend-impl.h`, `ggml-backend.cpp`;
- правка MMQ **[IQ3S_SIGN_SWAR]** в `ggml/src/ggml-cuda/mmq-load-tiles.cuh` (~строка 1406, деquant
  IQ3_S: `__vsub4(g^m,m)` заменён на сложение с маской — −3 SASS-инструкции на слово grid,
  baseline pp 832.6–838 t/s);
- `mmq-config-ampere.cuh` — **чистая оригинальная CASE-таблица** (эксперименты откатаны).

**Поэтому:** для сборки чужим человеком дерево передаётся архивом целиком (zip/7z всей папки
`llama.cpp-master`, можно исключить `build-x64-windows-msvc-release`, `build-x64-vnni`, `tmp_sched_test`).
Пересобирать «поверх upstream» нельзя — правки закоммичены nowhere.

## 1. Инструменты (те же версии, что в рабочем билде)

| компонент | версия / путь |
|---|---|
| OS | Windows x64 |
| Visual Studio | VS 2022 Community, MSVC **14.44.35207** (`vcvars64.bat`) |
| CUDA Toolkit | **v13.2** (`nvcc.exe`) |
| CMake | **4.3.0-rc1** (`C:\Program Files\CMake\bin\cmake.exe`) |
| Ninja | из VS 2022: `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\COMMON~1\MICROS~1\CMake\Ninja\ninja.exe` (его нашёл CMake сам) |
| GPU | RTX 5060 Ti, **sm_120 (cc 12.0)**; кернелы компилируются под **`120a-real`** (CMake сам заменяет `120-real` → `120a-real` — нужно для Blackwell FP4 MMA) |

Не найдены (это норма, только предупреждения): NCCL (много-GPU будет медленнее), OpenSSL (HTTPS выключен),
ccache (медленнее повторные сборки).

## 2. Конфигурирование

Открыть «Developer Command Prompt for VS 2022» (x64) или выполнить из обычной консоли:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d <корень дерева llama.cpp-master>

cmake -B build-x64-windows-msvc-release ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DGGML_CUDA=ON ^
  -DGGML_NATIVE=ON ^
  -DGGML_OPENMP=ON ^
  -DGGML_LLAMAFILE=ON ^
  -DGGML_CUDA_FA=ON ^
  -DGGML_CUDA_GRAPHS=ON ^
  -DGGML_CUDA_FORCE_MMQ=OFF ^
  -DGGML_CUDA_FORCE_CUBLAS=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;f16-f16;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0"
```

Важные детали (ровно как в рабочем кэше):

- **`RelWithDebInfo`, а не `Release`** — в кэше `CMAKE_BUILD_TYPE:STRING=RelWithDebInfo`
  (`/O2 /Ob1 /DNDEBUG -Zi`).
- **MMQ собирается по умолчанию**: `-DGGML_CUDA=ON` + `FORCE_CUBLAS=OFF` (в `ggml/CMakeLists.txt` опции
  `GGML_CUDA_FORCE_MMQ`/`GGML_CUDA_FORCE_CUBLAS` по умолчанию OFF; MMQ-кернелы, включая
  `mmq-instance-iq3_s.cu`, компилируются всегда и выбираются в ран-тайме). **Не включать cuBLAS-путь**
  (`FORCE_CUBLAS=ON` — запрещённая конфигурация: +7 % pp ценой +2 GB VRAM, см. протокол).
  `FORCE_MMQ=ON` не нужен (это форсированный обход cuBLAS, не «включатель MMQ»).
- **FA_QUANTS**: разделители `;` (или `,`); в CMake-списке типы турбо пишутся **с суффиксом `_0`**:
  `q5_1-turbo4_0`, `q5_1-turbo3_0`, `turbo4_0-turbo3_0` (в CLI сервера — `turbo4`/`turbo3`).
  Строка `q5_1-q4_0` добавлена в этот список (её не было в дефолтном) — без неё FA с `-ctk q5_1 -ctv q4_0`
  не соберётся.
- `GGML_NATIVE=ON` включает авто-детект CPU: на рабочем Xeon/Zen4 вышел
  `/arch:AVX512` + `GGML_AVX512_VBMI/VNNI` (в кэше `GGML_AVX=OFF, GGML_AVX2=OFF, GGML_AVX512=ON`).
  На более старом CPU можно оставить ON — детект сам выберет набор.
- Архитектура CUDA: CMake сам поднимает `120a-real` (лог: `Using CMAKE_CUDA_ARCHITECTURES=120a-real`).
  Для другой Blackwell-карты можно явно добавить `-DCMAKE_CUDA_ARCHITECTURES=120a` (или `100a` и т. п.).

Проверка после configure (`tmp_sched_test\reconfigure.log` в рабочем дереве):

```text
-- FlashAttention K-V type combinations: f16-f16;q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0
-- Using CMAKE_CUDA_ARCHITECTURES=120a-real
```

И в `build-x64-windows-msvc-release\CMakeCache.txt`:

```text
GGML_CUDA_FA_QUANTS:STRING=q4_0-q4_0;q8_0-q8_0;q8_0-q4_0;q5_1-q5_1;q5_1-q4_1;q5_1-q4_0;q4_1-q4_1;f16-f16;q5_1-turbo4_0;q5_1-turbo3_0;turbo4_0-turbo3_0
```

## 3. Сборка

```bat
cmake --build build-x64-windows-msvc-release --target llama-server
```

Готовые артефакты: `build-x64-windows-msvc-release\bin\llama-server.exe`, `bin\ggml-cuda.dll` и т. д.
Полная сборка CUDA-объектов — ~15–20 мин; инкрементальная правка одного TU — ~3.5 мин.

**Ловушка инкрементальной сборки** (уже вшита в `rebuild.cmd` в корне дерева): CMake генерирует
`unscanned` ninja-правила — **заголовки не отслеживаются**. После правки `.cuh` нужно «тронуть» TU,
которые их включают:

```bat
rebuild.cmd cuda   :: трогает ggml-cuda.cu
rebuild.cmd fa     :: трогает fattn.cu
rebuild.cmd all    :: оба
```

Смена `GGML_CUDA_FA_QUANTS` меняет `-D`-дефайны **каждой** CUDA-компиляции → после reconfigure
будет полная пересборка ggml-cuda (это ожидаемо).

## 4. Проверка, что нужные ядра реально собраны

1. Статика (быстрая):
   - `build.ninja` содержит `-DGGML_CUDA_FA_Q5_1_Q4_0=1`, `-DGGML_CUDA_FA_Q5_1_TURBO4_0=1`,
     `-DGGML_CUDA_FA_Q5_1_TURBO3_0=1`, `-DGGML_CUDA_FA_TURBO4_0_TURBO3_0=1`;
   - после сборки существуют объекты
     `ggml\src\ggml-cuda\CMakeFiles\ggml-cuda.dir\template-instances\fattn-vec-instance-q5_1-q4_0.cu.obj`,
     `...q5_1-turbo4_0.cu.obj`, `...q5_1-turbo3_0.cu.obj`, `...turbo4_0-turbo3_0.cu.obj`;
   - проверено на нашей сборке (26.10): device-символы этих четырёх комбо присутствуют внутри
     `ggml-cuda.dll` (`_Z18flash_attn_ext_vecI...EL9ggml_type7ELS0_2E...` = q5_1→q4_0,
     `...EL9ggml_type7ELS0_45E...` = q5_1→turbo3_0, `...EL9ggml_type46ELS0_45E...` = turbo4_0→turbo3_0,
     `...EL9ggml_type7ELS0_46E...` = q5_1→turbo4_0; типы 7/2/45/46 = Q5_1/Q4_0/TURBO3_0/TURBO4_0).
2. Ран-тайм:

```bat
build-x64-windows-msvc-release\bin\llama-server.exe -m <модель> -ngl 99 -ctk q5_1 -ctv q4_0 ...
```

В логе строка `system_info: ... | CUDA : ARCHS = 1200 | FA_QUANTS = q4_0-q4_0,q8_0-q8_0,...,q5_1-turbo4_0,q5_1-turbo3_0,turbo4_0-turbo3_0 | ...`
и ** отсутствие** предупреждения
`no FlashAttention vector kernel compiled for K/V types ... Add "<K>-<V>" to GGML_CUDA_FA_QUANTS`
= все нужные ядра на месте.

## 5. Эталонная строка запуска (с которой измерялся baseline)

```bat
build-x64-windows-msvc-release\bin\llama-server.exe -m "G:\Models\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" ^
  -ngl 99 -c 50000 -b 2048 -ub 512 -np 1 -t 7 -ctk q5_1 -ctv turbo4 -lm mmap -ctxcp 0 -cram 0 -lv 3 ^
  --host 127.0.0.1 --port 8080 ^
  --reasoning-budget 12200 --reasoning-budget-message "Лимит рассуждений исчерпан, дай ответ" ^
  --spec-type draft-mtp --spec-draft-n-max 3 ^
  --ffn-cpu-count 32
```

Baseline dense-префилла MMQ при `-ub 512`: **832.6–838 t/s** (полоса; кернел `mul_mat_q<IQ3_S, J=112,
fb=false>`, 5 токен-тайлов, smem 96 704 B — J=128 не проходит smem-guard 101 376 B на 5060 Ti).
Для контрольных замеров брать `-c`, при котором нет warning `failed to fit params to free device memory`
(`-c 150000` с q5_1-K уже не влезает в 16 GB).

Доступные режимы доноров (`--ffn-cpu-op`): `down` (default) | `up` | `gate` | `largest`
(`--ffn-cpu-count` = число донорских тензоров, раскладка равномерная; `--ffn-cpu-gap` — stride).
Режим `gate_up` (пара gate+up одного слоя) был реализован и **отменён/откачен** (26.10): на 27B
`gate_up 26` дал tg 47,08 против 58,06 у `down 26` при меньшем DMA — пара ломает окно подкачки
(копии пары попадают в соседние сплиты, ring gap_min=2 против 7, head-of-line в copy-стриме).
Протокол: `tmp_sched_test\ffn_cpu_split_notes.md` §13.69-13.70; контроль откатa —
`tmp_sched_test\smoke_revert.ps1`.

Для diagnostics-строк этого форка (`select_cpu_donors: chose ...`, `[ENV]`, `[PREFETCH_WINDOW]`,
числа проекции fit) нужен **`-lv 4` и выше**: INFO отображается в TRACE (`common\log.cpp:519-529`),
при `-lv 3` видны только warning'и. `GGML_SCHED_LOG_VERBOSE=1` - summaries кольца; =2/3 включают
per-node/per-split строки (на 27B они дороги: `-lv 5` + подробный лог меняет t/s, см. §13.59).

## 6. Если что-то не так

- `GGML_CUDA_FA_QUANTS: unknown type "turbo4_0"` → скопирована не наша ветка (в upstream turbo-типов нет).
- Предупреждение про FA-ядро при `-ctk q5_1 -ctv turbo4` → FA_QUANTS не применён: reconfigure + полная
  пересборка (`rebuild.cmd all` или `cmake --build` заново).
- Сервер «не видит» MMQ-ускорения IQ3_S → проверить, что `GGML_CUDA_FORCE_CUBLAS=OFF`.
