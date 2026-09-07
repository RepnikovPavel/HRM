# rrnfast: fused CUDA kernel for one RRN message-passing step

One training step of `SudokuRRN._step` (models the 81x20 edge tensor
`e = relu(hw1[nb] + hw2[j] + b1)`, three more 96x96 relu layers, dropout,
per-node sum, LSTM cell) is fused into three kernels:

- `edge_fwd.cu` / `edge_bwd.cu`: e0 built straight into shared memory from the
  two factorized GEMM outputs `hw12 = h @ W1^T` ([B*81, 192]), then three
  96x96 layers as m16n8k16 bf16 mma over an [80, 104]-strided smem tile
  (4 nodes x 20 edges per block), philox dropout on (edge, feature/4, step),
  warp-reduced rowsum into `m` [B, 81, 96]. Backward recomputes e0/e1/e2
  from `hw12` instead of storing them, so only `m` is saved per step.
- `lstm.cu`: fused gate pre-activation GEMMs stay in `gemm_family.cuh`
  (shared with hrmfast); elementwise i/f/g/o + c/h update is one kernel.
- `rrn_step.cu`: orchestrates one step, returns (h1, c1, m); the autograd
  wrapper in `rrnfast/__init__.py` saves x/h/c/m per step (no checkpoint
  recompute needed, backward is a single pass).

## Parity vs reference `_step` (tests/test_parity.py)

bf16 tensors vs fp64 ground truth, 32 steps chained: forward h/c and all
weight/input grads match the PyTorch reference at bf16 rounding level
(ours~fp64 <= ref~fp64 on every tensor). Dropout: deterministic per
(seed, step), seed-sensitive, unbiased (max|mean diff| 0.005 vs mean|m| 1.25).
Numeric gradient check rel err ~0.3 (expected for bf16 finite differences).

## Measured speed (32 steps, bf16, dropout 0.4)

Dev RTX 4070 Ti, B=1024/GPU:

| path            | fwd samples/s | fwd+bwd samples/s |
|-----------------|---------------|-------------------|
| eager           | 1510          | 405               |
| torch.compile   | 3032          | 640               |
| rrnfast         | 8554          | 1607              |

speedup vs compile: fwd x2.82, train x2.51 (vs eager: x5.66 / x3.97).

Server 2x RTX 5060 Ti, B=1024/GPU: fwd 5492, fwd+bwd 1070 samples/s per GPU
(x2.06 fwd / x1.92 train vs compile).

## Build

    scripts/build_rrnfast.sh   # pip install --no-build-isolation --target=$DATA_ROOT/python-packages

setup.py compiles for sm_89, sm_90, sm_120. pip build isolation hides the
host torch; always install with --no-build-isolation (see pyproject.toml).

`rrn/rrn/model.py` imports rrnfast opportunistically and falls back to the
reference PyTorch `_step` (with activation checkpointing) when the extension
is absent; dropout seed advances once per optimizer step so DDP ranks draw
identical masks, matching nn.Dropout RNG semantics.
