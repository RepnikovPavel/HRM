# RRN vs HRM на Sudoku-Extreme (1k) — сравнение

Одинаковые данные: train/test из `sudoku-extreme-1k-aug-1000`
(1 001 000 train / 422 786 test). Одинаковый дата-бюджет обучения:
20M просмотренных примеров. Замеры скорости — сервер, 1x RTX 5060 Ti,
CUDA events, среднее 10 прогонов; HRM — `scripts/bench_fwdbwd.py`,
RRN — `rrn/bench.py`.

## Модели

| | HRM | RRN (Palm et al. 2018) |
|---|---|---|
| параметры | 27 275 266 | 191 114 |
| вычисление | 2 модуля × 4 трансформерных слоя, 24 исполнения блока/сегмент, до 16 сегментов ACT | 32 шага message passing по графу 81 узел × 20 соседей, LSTM-обновление |
| лосс | stablemax CE + ACT Q-loss | CE по всем 32 шагам (deep supervision) |
| оптимизатор | AdamATan2 + SignSGD (puzzle emb) | Adam (lr 2e-4, wd 1e-4) |

## Скорость (замерено, сервер)

Батчи разные по размеру (у каждой модели свой рабочий gbs), поэтому
сравнивать корректно по µs/элемент и samples/s.

| метрика | HRM (gbs 1088/GPU) | RRN (gbs 6144/GPU, rrnfast) |
|---|---|---|
| train fwd, ms/батч | 932.21 | 1287.85 |
| train bwd, ms/батч | 315.26 | 4584.08 |
| train fwd, µs/элемент | 856.81 | 209.61 |
| train bwd, µs/элемент | 289.76 | 746.11 |
| train fwd+bwd, µs/элемент | 1146.57 | 955.72 |
| test fwd, ms/батч | 7460.44 | 1131.27 |
| test fwd, µs/элемент | 6857.02 | 184.13 |
| throughput train, samples/s/GPU | ~872 | 1046.3 |
| throughput train, samples/s (2 GPU) | ~1750 (gbs 2176) | 1979.4 (gbs 12288, из реального обучения v4) |

Замечания:

- train fwd у HRM включает два inner-прохода (main + target-Q, 48
  исполнений блока); bwd — только grad-часть (16 блок-эквивалентов).
- test fwd у HRM — полный инференс до остановки: 16 сегментов × 24 блока
  = 384 исполнения блока, поэтому на порядок дороже train forward на
  элемент. У RRN test fwd = те же 32 шага, но без dropout и без хранения
  графа.
- RRN bwd дорогой относительно fwd (3.6x): backward edge-ядра
  перевычисляет e0/e1/e2 из сохранённого m и считает dW через mma по
  транспонированным фрагментам; зато не хранит промежуточные активации
  32 шагов (только x/h/c/m на шаг), что и позволяет gbs 6144/GPU в
  15.5 ГиБ (peak 13.7 ГиБ при обучении).
- RRN-замеры — с fused CUDA-ядром rrnfast (rrnfast/docs/rrnfast.md):
  x2.06 fwd / x1.92 train против torch.compile(default) и x4.05/x2.97
  против eager на сервере. Эталонный PyTorch `_step` остаётся fallback
  и эталоном parity (tests/test_parity.py: fwd и все градиенты на уровне
  bf16-округления, dropout детерминирован по (seed, step)).

## Качество (test exact_accuracy, полный test set 422 786)

| модель | exact_accuracy | источник |
|---|---|---|
| HRM (run sudoku-1k-server-v2, 18440 steps @ gbs 3072) | **64.4%** | reports/hrm_reproduction.html (полный test, 2026-09-06); paper target 55.0% — MATCH |
| RRN (run rrn-sudoku-1k-v4, 1627 steps @ gbs 12288, 2 GPU, rrnfast) | **3.11%** (cell accuracy 67.7%) | полный test 422 786, `rrn/evaluate.py` на step_1627, 2026-09-07 |

Сырые метрики обучения RRN: `/mnt/hdd2/hrm/checkpoints/rrn-sudoku-1k-v4/metrics.jsonl`.
Скоростные колонки RRN — `rrn/bench.py` на сервере при gbs 6144/GPU
(рабочий батч обучения v4), CUDA events, среднее 10 прогонов после 3
warmup; peak 13.13 ГиБ/GPU.

Вывод по качеству: при равном дата-бюджете (20M примеров) RRN — модель,
которую сама статья HRM приводит как предшественника, — решает лишь 3.1%
досок целиком против 64.4% у HRM, хотя на элемент у RRN train fwd в 4.1x
быстрее и test fwd в 37x быстрее (184 µs vs 6857 µs — цена ACT-инференса
HRM из 16 сегментов). Сравнение с baseline-ами из статьи (0% у
GPT/DeepSeek-подобных) здесь заменено честным: RRN метрики ненулевые,
обучение сходится (loss 2.3 → 0.72), но до качества HRM не дотягивает на
порядок.
