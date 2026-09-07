# RRN baseline для Sudoku-Extreme

Базовая модель для честного сравнения с HRM: **Recurrent Relational Network**
(RRN, Palm et al., NeurIPS 2018, arXiv:1711.08028) — общепринятая нейросетевая
модель для судоку; именно на неё ссылается сама статья HRM (ref 62) как на
канонический решатель судоку. Реализация — векторизованный PyTorch без DGL,
по эталону [dmlc/dgl examples/pytorch/rrn](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rrn),
который воспроизводит оригинальный код
[rasmusbergpalm/recurrent-relational-networks](https://github.com/rasmusbergpalm/recurrent-relational-networks).

Почему не «чистый RL»: существующие RL-реализации судоку на GitHub
(DQN/PPO-игрушки уровня kaggle-датасета) не решают сложные судоку и не имеют
воспроизводимых метрик — сравнение с ними было бы тем же «нулевым бейзлайном»,
за который мы критикуем авторов. RRN — самая сильная общепринятая нейросетевая
модель на этой задаче, поэтому она выбрана как baseline.

## Архитектура (в точности по DGL-эталону)

- Граф: 81 клетка = узел; направленные рёбра между клетками одной строки,
  столбца или блока 3x3 — у каждого узла 20 соседей (8 строка + 8 столбец +
  4 блока), сообщения идут в обе стороны.
- Признаки узла: embedding цифры (10→16) + embedding строки (9→16) +
  embedding столбца (9→16), concat → MLP 48→96→96→96→96 (ReLU).
- 32 шага message passing; шаг: сообщение ребра
  `e_ij = MLP([h_i; h_j])` (MLP 192→96→96→96→96, ReLU), elementwise dropout
  0.4 по сообщениям (train), `m_j = sum_i e_ij`, обновление узла —
  LSTMCell([x_j; m_j] → 96, без bias). В коде LSTMCell развёрнут вручную
  (`lstm_ih`/`lstm_hh` + pointwise gates) — nn.LSTMCell(bias=False) ломает
  мета-регистрацию torch.compile; математика идентична.
- Выход: Linear(96→10) на каждом шаге; лосс при обучении — CE по всем 32
  шагам (deep supervision, как в эталоне), на тесте — argmax последнего
  шага.
- Параметров: 191 114 (в 143 раза меньше HRM — это часть сравнения).

## Данные

Те же файлы, что у HRM: `/data/built/sudoku-extreme-1k-aug-1000/{train,test}`
(1 001 000 train / 422 786 test примеров, `all__inputs.npy`,
`all__labels.npy`). Перекодировка: значения датасета сдвинуты на +1
(1=blank, 2..10=цифры); RRN вычитает 1 обратно (0=blank, 1..9=цифры,
10 классов CE). Тестовый набор идентичен HRM — exact_accuracy сравнимы
напрямую.

## Обучение

`train.py`: Adam(lr 2e-4, wd 1e-4) — гиперпараметры эталона; обучение
распределённое (DDP, torchrun, `NGPU` в `scripts/train_rrn.sh`). Быстрый
путь — fused CUDA-ядро rrnfast (`rrnfast/`, собирается
`scripts/build_rrnfast.sh` в `$DATA_ROOT/python-packages`): весь шаг
message passing (edge MLP + dropout + сумма по узлам + LSTM) одним
проходом ядер, без activation checkpointing — backward перевычисляет
промежуточные e0/e1/e2 из сохранённого m. Parity с эталонным `_step`:
rrnfast/docs/rrnfast.md. Если расширение не собрано — fallback на
эталонный PyTorch `_step` с checkpointing (bf16 autocast, TF32).
Дата-бюджет тот же, что у HRM sudoku-1k: 20M просмотренных примеров
(20000 эпох × 1000 групп).

Замерено на сервере (2x 5060 Ti, run v4): 1990.7 samples/s суммарно при
gbs 12288 (6144/GPU), peak 13.7 ГиБ/GPU из 15.5, обе карты под 100%.
gbs 8192 (4096/GPU) не влезает по памяти с fused-ядром (хранение
x/h/c/m на каждый из 32 шагов); 6144 — максимум без OOM.

```bash
# dev
DATA_ROOT=/mnt/nvme/hrm bash scripts/train_rrn.sh --gbs 1024
# сервер (2 GPU)
ssh user@192.168.0.1 'cd ~/HRM && DATA_ROOT=/mnt/hdd2/hrm NGPU=2 \
  RUN_NAME=rrn-sudoku-1k-v4 bash scripts/train_rrn.sh \
  --gbs 12288 --eval-interval 100'
```

Метрики: `<out>/metrics.jsonl` (loss, steps/s, samples/s каждые 100
шагов; test accuracy / exact_accuracy каждые 100 шагов на 8192 примерах),
чекпоинты `step_<N>`.

## Замеры производительности

`bench.py`: fwd/bwd (train) и fwd (test) на батч и на элемент, CUDA events,
среднее 10 прогонов после 3 warmup:

```bash
docker run --rm --gpus all ... hrm-rrn-train:cuda12 \
  python -u bench.py --data /data/built/sudoku-extreme-1k-aug-1000 --gbs 1024
```

Итоговая сводка сравнения с HRM — в `rrn/docs/comparison.md` (заполняется
по факту обучения).
