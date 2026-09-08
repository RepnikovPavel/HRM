# RRN vs HRM на Sudoku-Extreme (1k) — сравнение

Данные у всех прогонов одинаковые: `sudoku-extreme-1k-aug-1000`
(1 001 000 train / 422 786 test), дата-бюджет 20M просмотренных примеров.
Замеры скорости: CUDA events, среднее 10 прогонов после 3 warmup;
HRM — `scripts/bench_fwdbwd.py`, RRN — `rrn/bench.py`.

## Модели

| | HRM | RRN-96 | RRN-27M |
|---|---|---|---|
| параметры | 27 275 266 | 191 114 | 27 362 026 |
| скрытая размерность | 512 | 96 | 1168 |
| итераций рассуждения на инференс | 16 ACT-сегментов × 24 layer-pass | 32 шага | 13 шагов |
| inference FLOPs/пример, GFLOP | 214.9 | 3.6 | 213.4 |

RRN — Recurrent Relational Network (Palm et al. 2018). RRN-27M — тот же
`SudokuRRN` из `rrn/`, отмасштабированный в паритет с HRM по параметрам
(+0.32%) и inference FLOPs (−0.7%); вывод d=1168, T=13:
`rrn27m/docs/rrn27m.md`. FLOPs измерены torch.profiler with_flops.

## Скорость

Сервер 2× RTX 5060 Ti, рабочий батч обучения каждого прогона (указан в
примечаниях). Сравнение по µs/элемент и samples/s.

| метрика | HRM | RRN-96 | RRN-27M |
|---|---|---|---|
| train fwd, ms/батч | 932.21 | 1287.85 | 3189.26 |
| train bwd, ms/батч | 315.26 | 4584.08 | 10127.17 |
| train fwd, µs/элемент | 856.81 | 209.61 | 6229.03 |
| train bwd, µs/элемент | 289.76 | 746.11 | 19779.63 |
| train fwd+bwd, µs/элемент | 1146.57 | 955.72 | 26008.66 |
| test fwd, ms/батч | 7460.44 | 1131.27 | 3054.04 |
| test fwd, µs/элемент | 6857.02 | 184.13 | 5964.92 |
| train samples/s, 1 GPU | 872 | 1046.3 | 38.4 |
| train samples/s, 2 GPU | 1745 | 1979.4 | 69.2 |

Примечания:

- батчи (global / на GPU): HRM 2176/1088; RRN-96 12288/6144 (fused-ядро
  rrnfast, `rrnfast/docs/rrnfast.md`); RRN-27M 1024/512 (torch.compile);
- train fwd у HRM включает 2 inner-прохода (основной + target-Q);
  test fwd — полный ACT-инференс, 16 сегментов × 24 layer-pass;
- строки «2 GPU» — из реальных обучающих прогонов, остальное — бенчи на
  1 GPU.

## Стоимость обучения

FLOPs train-шага на элемент — torch.profiler with_flops
(`scripts/bench_train_flops.py`, HRM без hrmfast: кастомные ядра
профилеру не видны; RRN eager + activation checkpointing).

| метрика | HRM | RRN-27M |
|---|---|---|
| train fwd, GFLOP/элемент | 26.9 | 213.4 |
| train bwd, GFLOP/элемент | 8.9 | 639.2 |
| train fwd+bwd, GFLOP/элемент | 35.77 | 852.6 |
| просмотрено примеров, M | 56.65 | 5.53 |
| FLOPs прогона, EFLOP | 2.03 | 4.72 |
| часы, 2× 5060 Ti | 8.2 | 21.8 |

Примечания:

- структура train fwd HRM (по коду, `models/hrm/hrm_act_v1.py:203-218,293`):
  48 layer-pass = 16 no-grad + 8 с grad (основной сегмент) + 24 no-grad
  (target-Q); bwd только по 8 layer-pass с grad (one-step gradient);
- RRN-27M: deep supervision по всем 13 шагам + checkpoint-перевычисление,
  итого 4× fwd;
- прогон RRN-27M остановлен досрочно (5.53M из 20M примеров); полный
  бюджет — 17.1 EFLOP, ~79 ч;
- «часы» HRM — между первой и последней записью metrics.jsonl.

## Пик BF16 dense, TFLOPS

| метод | 5060 Ti | 4070 Ti |
|---|---|---|
| спецификация | 47.4 | 80.2 |
| cuBLAS 8192³ | 51.1 | 76.4 |
| mma.sync микробенч, `hrmfast/tests/bench_mma_peak.cu` | 52.6 | — |
| gpubench WMMA, после фикса | 46.4 | 61.8 |
| gpubench WMMA, до фикса | 182.9 | 248.6 |

Примечания:

- cuBLAS 8192³ на 5060 Ti выше спеки (108%) — boost-частота выше
  номинальных 2.57 ГГц;
- в gpubench была ошибка: аккумуляторы ILP-линий u>0 не сохранялись
  (`u == 0 && threadIdx.x == 0`), компилятор удалял 3/4 MMA-цепочек как
  dead code; исправлено в upstream, commit b3aa28b; до фикса завышение
  3.1-3.9x;
- референс пика для эффективности — спецификация.

## Вычислительная эффективность обучения

Эффективные TFLOPS = профильные FLOPs/элемент × измеренные samples/s;
доля — от пика по спецификации (5060 Ti: 47.4, 4070 Ti: 80.2 TFLOPS).

| сетап | HRM samples/s | HRM TFLOPS | HRM, % пика | RRN-27M samples/s | RRN-27M TFLOPS | RRN-27M, % пика |
|---|---|---|---|---|---|---|
| сервер 1× 5060 Ti | 936.4 | 33.5 | 70.7 | 38.4 | 32.7 | 69.0 |
| сервер 2× 5060 Ti, суммарно | 1745 | 62.4 | 65.8 | 69.2 | 59.0 | 62.2 |
| dev 1× 4070 Ti | 1195.5 | 42.8 | 53.4 | 55.9 | 47.7 | 59.5 |

Примечания:

- батчи: сервер HRM 1536/GPU, RRN-27M 512/GPU; dev HRM 384, RRN-27M 256;
- строка «2× 5060 Ti» — из реальных обучающих прогонов, остальное —
  бенчи на 1 GPU;
- обе модели на одном железе дают почти одинаковые эффективные TFLOPS
  (5060 Ti: 32.7-33.5; 4070 Ti: 42.8-47.7) — разница в samples/s
  объясняется FLOPs/элемент (23.8x);
- отношение 4070 Ti / 5060 Ti по samples/s: RRN 1.46, HRM 1.28;
  отношение cuBLAS-потолков: 1.50.

## Качество

Test exact_accuracy, полный test set (422 786 примера):

| модель | прогон | шагов | примеров, M | exact_accuracy, % | cell accuracy, % |
|---|---|---|---|---|---|
| HRM | sudoku-1k-server-v2 | 18 440 | 56.65 | 64.4 | — |
| RRN-96 | rrn-sudoku-1k-v4 | 1 627 | 20.0 | 3.11 | 67.7 |
| RRN-27M | rrn27m-sudoku-1k | 5 400 | 5.53 | 16.39 | 73.4 |

Примечания:

- HRM: `reports/hrm_reproduction.html`, 2026-09-06; цель из статьи 55.0%
  — совпала;
- RRN-96: `rrn/evaluate.py`, step_1627, 2026-09-07;
- RRN-27M: лучший чекпоинт step_3000, 2026-09-08; прогон остановлен
  досрочно — результат underfit (loss ещё снижался);
- сырые метрики: `/mnt/hdd2/hrm/checkpoints/{rrn-sudoku-1k-v4,rrn27m-sudoku-1k}/metrics.jsonl`.

## Выводы

- Качество при полном бюджете: HRM 64.4% против 3.11% у RRN-96 (в 143
  раза меньше параметров) и 16.39% у RRN-27M (паритет параметров и
  inference FLOPs, но 5.53M из 20M примеров — underfit).
- Инференс при равных FLOPs: RRN-27M 5965 против 6857 µs/элемент у HRM.
- Обучение при равных inference FLOPs: RRN-27M стоит 23.8x FLOPs на
  пример (852.6 против 35.77 GFLOP) — HRM платит one-step gradient только
  за 8 из 48 layer-pass сегмента, а 16 инференсных ACT-сегментов в
  обучение не входят; RRN платит за все 13 шагов рассуждения и в
  обучении.
