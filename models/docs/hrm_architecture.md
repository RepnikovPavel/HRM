# HRM (Hierarchical Reasoning Model) — конспект архитектуры

Строго по коду: `models/hrm/hrm_act_v1.py`, `models/layers.py`,
`models/losses.py`, `models/sparse_embedding.py`. Размерности — под наш
рабочий конфиг (Sudoku-Extreme, `config/arch/hrm_v1.yaml`).

## Обозначения

$$
\begin{gather*}
N\text{: batch (на GPU 1088..1728)} \qquad
S = 81 + 1 = 82\text{: токены (доска + puzzle-токен)}\\
D = 512\text{: hidden} \qquad H = 8\text{: головы} \qquad d_h = D/H = 64
\qquad I = 1536\text{: inter SwiGLU}\\
V = 11\text{: vocab (цифры 1..9, blank, pad)} \qquad
M_{max} = 16\text{: макс. число ACT-сегментов}
\end{gather*}
$$

Параметров всего: **27 275 266** (замерено при старте обучения).

## Компоненты (определения)

- **Puzzle embedding** (`CastedSparseEmbedding`): обучаемый вектор на
  puzzle_id, $E_{pz} \in \mathbb{R}^{N_{pz} \times 512}$, инициализация
  нулём (init_std=0). Это «маркер идентичности задачи» — модель по нему
  различает, какой пазл решает. Обновляется отдельным оптимизатором
  (sparse SignSGD, свой lr). Встает первым токеном последовательности.
- **RMSNorm** (`rms_norm` в layers.py): без обучаемого gain, считается в
  fp32: $\text{rms}(x) = x / \sqrt{\frac{1}{D}\sum x_i^2 + \varepsilon}$.
- **Attn** (класс `Attention`): некаузальный мультиголовый attention;
  возвращает **спроецированный** выход $W_O \cdot \text{concat}(\text{heads})$.
- **SwiGLU**: $\text{down}\big(\text{silu}(g) \odot u\big)$, где
  $g, u = \text{split}(W_{gu}\, x)$.
- **RoPE**: поворот q,k на углы $\theta_i = 10000^{-2i/d_h}$;
  $\cos,\sin \in \mathbb{R}^{82 \times 64}$ предвычислены.

## Эмбеддинг входа

$$
e = \sqrt{D} \cdot \text{Cat}\big[\, E_{pz}[pz],\ E[x]\, \big]
\in \mathbb{R}^{N \times 82 \times 512}
$$

## Блок трансформера (один слой; post-norm по коду)

$$
\begin{gather*}
h \leftarrow \text{RMSNorm}\big(h + \text{Attn}(h)\big)
\qquad \text{где } \text{Attn}(h) = W_O \cdot \text{softmax}\Big(\frac{\text{RoPE}(Q)\,\text{RoPE}(K)^T}{\sqrt{64}}\Big) V\\
h \leftarrow \text{RMSNorm}\big(h + \text{SwiGLU}(h)\big)
\end{gather*}
$$

| операция | in | вес | out | FLOPs/token |
|---|---|---|---|---|
| qkv_proj | $N{\times}S{\times}512$ | $1536{\times}512$ | $N{\times}S{\times}1536$ | 1.57M |
| RoPE(q,k) | $N{\times}S{\times}8{\times}64$ ×2 | — | — | ~0.25M |
| softmax($QK^T/\sqrt{d_h}$)$V$ | $N{\times}8{\times}82{\times}64$ | — | $N{\times}S{\times}512$ | 0.17M |
| o_proj | $N{\times}S{\times}512$ | $512{\times}512$ | $N{\times}S{\times}512$ | 0.52M |
| gate_up_proj | $N{\times}S{\times}512$ | $3072{\times}512$ | $N{\times}S{\times}3072$ | 3.15M |
| down_proj | $N{\times}S{\times}1536$ | $512{\times}1536$ | $N{\times}S{\times}512$ | 1.57M |
| **блок итого** | | | | **6.98M** |

## Два уровня и двойной вложенный цикл (ядро модели)

Модули $L$ (быстрый, детали) и $H$ (медленный, план), по 4 блока каждый.
Состояние $z_L, z_H \in \mathbb{R}^{N \times 82 \times 512}$.

Один inner forward (`_Inner.forward`), точно по коду:

```python
# no-grad часть (градиент не течёт):
for i_H in range(H_cycles):            # 2 итерации
    for i_L in range(L_cycles):        # 2 итерации
        if not (i_H == H_cycles-1 and i_L == L_cycles-1):
            z_L = L(z_L, z_H + e)      # вход e только сюда
    if i_H != H_cycles-1:
        z_H = H(z_H, z_L)

# grad-часть (1-step grad, без BPTT):
z_L = L(z_L, z_H + e)
z_H = H(z_H, z_L)
```

Итого за inner forward: $L$-проходов $3+1$, $H$-проходов $1+1$ →
24 блока = **167.6 MFLOP/token** + головы (lm_head, q_head).

Вход $e$ подаётся только в $L$ (input injection $z_H + e$); $H$ видит
только $z_L$. Градиент течёт только через последнюю пару проходов —
это и есть one-step gradient approximation из статьи.

## ACT-обёртка (halting, Q-learning)

Один optimizer step = один сегмент на пример. Для обучения:

```python
# main (с градиентом):
z', y, q_halt, q_cont = inner(z, e)

# target-Q (no-grad, второй вызов inner):
q'_halt, q'_cont = inner(z', e).q
q_target = sigmoid(q'_halt if last_step else max(q'_halt, q'_cont))

halt = (steps >= M_max) or ((q_halt > q_cont) and (steps >= rand_min_halt))
```

При halt: $z \leftarrow (H_{init}, L_{init})$, пример в батче подменяется
новым пазлом (через `torch.where` по флагу).

## Loss (stablemax из статьи, НЕ softmax)

$$
s(x) = \begin{cases} x + 1, & x \ge 0\\ \dfrac{1}{1 - x + \varepsilon}, & x < 0 \end{cases}
\qquad
\log p_i = \log \frac{s(x_i)}{\sum_j s(x_j)}
$$

$\mathcal{L} = \text{CE}_{stablemax}(y, \hat y) + 0.5\big(
\text{BCE}(q_{halt}, \mathbb{1}[\text{пазл решён}]) +
\text{BCE}(q_{cont}, \tilde q)\big)$

Stablemax — замена softmax без $\exp$: не взрывается при больших логитах
(статья, §стабильность; в коде считается в float64). Для sudoku-full в
README используется обычный softmax CE (`arch.loss.loss_type=softmax_cross_entropy`).

## Inference pipeline

```python
z = (H_init, L_init)
for m in range(M_max):          # eval: всегда до all-halt или M_max
    z, y_hat, q_halt, q_cont = inner(z, e)
    if all(q_halt > q_cont):
        break
answer = argmax(y_hat)          # logits последнего сегмента
```

По коду eval идёт до `all_finish` (все halt или $M_{max}$); финальный
ответ — logits последнего сегмента, argmax по vocab.

## Связь с трансформером

Блок HRM — это трансформерный слой (attention + MLP), но: post-norm
RMSNorm вместо pre-norm LayerNorm, SwiGLU, RoPE, без bias и dropout,
bf16. HRM отличается от трансформера тем, что **глубина получается
рекуррентностью**: два одинаковых по устройству модуля L/H гоняются по
циклам и ACT-сегментам, т.е. 4+4 слоя исполняются десятки раз на один
вход (effective depth ≫ 8 слоёв), без роста числа параметров.

## Backward FLOPs (1-step grad: градиент только через последнюю пару L,H)

Backward идёт только по grad-части inner forward (8 блоков). Для каждого
линейного слоя backward = dx (2MNK) + dW (2MNK) ≈ 2× forward:

| операция (на блок) | fwd | bwd dx | bwd dW | итого |
|---|---|---|---|---|
| qkv_proj | 1.57M | 1.57M | 1.57M | 4.71M |
| attention (softmax·V) | 0.17M | ~0.34M | — | ~0.51M |
| o_proj | 0.52M | 0.52M | 0.52M | 1.56M |
| gate_up_proj | 3.15M | 3.15M | 3.15M | 9.45M |
| down_proj | 1.57M | 1.57M | 1.57M | 4.71M |
| **блок** | 6.98M | **13.96M** (≈2×fwd) | | |

За optimizer step (N=1088, S=82, на GPU):

| фаза | блоков | TFLOP |
|---|---|---|
| fwd main (no-grad 16 + grad 8) | 24 | 14.9 |
| fwd target-Q (весь inner, no-grad) | 24 | 14.9 |
| bwd (8 блоков × 2×fwd) | 16 | 10.0 |
| **итого** | **64** | **39.9** |

## FLOPs на optimizer step (на GPU, N=1088, S=82, замеренный конфиг)

$$
\underbrace{(24+24)}_{\text{fwd main + target-Q}} + \underbrace{2 \cdot 8}_{\text{bwd}}
= 64 \text{ блока}
\;\Rightarrow\; 64 \times 6.98\text{M} \times 1088 \times 82 \approx \mathbf{39.9\ TFLOP/step/GPU}
$$

Замеры (2× RTX 5060 Ti; потолок mma.sync 52.6 TFLOPS/GPU по
`hrmfast/tests/bench_mma_peak.cu`):

| | время шага | эффективность |
|---|---|---|
| author (cuBLAS/SDPA) | 1.28 с @ gbs 2176 | 64% |
| hrmfast (наши mma-ядра) | 1.25 с @ gbs 2176 | 65% |

Остаток — не-GEMM работа (RMSNorm, softmax, RoPE, elementwise,
оптимизатор), на tensor pipes не ложится; GEMM-часть сама по себе на
~92% от потолка.
