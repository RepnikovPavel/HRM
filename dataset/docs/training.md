# Training stage (task 2)

Reproduction of the paper's "launch experiments" for Sudoku-Extreme (1k),
Maze-Hard 30x30 (1k) and Sudoku-Extreme-Full, on two machines:

- server: 2x RTX 5060 Ti (16 GB, sm_120), data at `/mnt/hdd2/hrm`
- dev:    1x RTX 4070 Ti (12 GB, sm_89), data at `/mnt/nvme/hrm`

## Glossary

- **gbs** = `global_batch_size` — batch size per optimizer step summed over
  ALL GPUs (server gbs 3072 = 1536 per GPU). In bench/train logs and docs
  the short form `gbs` is used; per-GPU batch = gbs / num_gpus.
- **steps/s** — optimizer steps per second (global, synchronized across GPUs).
- **samples/s** — throughput = gbs x steps/s (examples per second).
- **compute efficiency** — model FLOPs per step / (measured step time x
  measured hardware ceiling in FLOPS); ceiling measured by
  hrmfast/tests/bench_mma_peak.cu (52.6 TFLOPS mma.sync on 5060 Ti).

## Changes vs upstream (all cosmetic / infra, algorithm untouched)

The original authors' code stays untouched in git HEAD; everything below
is an uncommitted optimization/infra delta documented here (no external
proprietary code is referenced or included anywhere in the repo).

- wandb removed (task requirement); metrics go to
  `<checkpoint>/metrics.jsonl` with wall-clock timestamps, progress lines
  are printed for `docker logs`.
- `models/layers.py`: SDPA fallback when flash-attn is absent
  (flash-attn has no sm_120 wheels; SDPA flash backend is used instead).
- `pretrain.py`: grads all-reduce coalesced into one flat buffer per dtype
  instead of one NCCL call per parameter; `COMPILE_MODE` env selects the
  torch.compile mode; `eval_test_examples` limits the *in-training*
  monitoring eval (final `evaluate.py` always uses the full test set).
- `dataset/build_*_dataset.py`: read local CSVs (`--source-dir`) from the
  task-1 data layer instead of `hf_hub_download`; `seed` config added so
  subsample/augmentation is reproducible.
- `config/cfg_pretrain.yaml`: `checkpoint_path: null` (hydra struct),
  hydra run dir moved to /tmp.

## Images

- `docker/DockerfileTrain` (CUDA 12.8, torch 2.9.1) — primary
- `docker/DockerfileTrainCuda13` (CUDA 13.0, torch 2.12.1) — second
  architecture per AGENTS.md
- adam-atan2 0.0.3 is built from sdist with sm_120 added to
  NVIDIA_SUPPORTED_ARCHS (upstream wheels and even its own sdist ship
  only sm_80/86/89/90).
- The image is built on the server and transferred to dev with
  `docker save | ssh | docker load` (dev internet is limited).

## Measured performance (server, 2x 5060 Ti, sudoku-1k)

| config | steps/s | samples/s | peak VRAM | power |
|---|---|---|---|---|
| gbs 768 (paper), compile | - | - | 6.2 GB | - |
| gbs 1536, no compile | 0.53 | 810 | 14.4 GB | 141 W |
| gbs 1536, compile default | 1.10 | 1690 | 11.5 GB | 149 W |
| gbs 2176, compile default | 0.78 | 1700 | 15.7 GB | 154 W |
| gbs 2176, max-autotune-no-cudagraphs | 0.79 | 1720 | 15.7 GB | 148 W |
| gbs 2176, reduce-overhead (cudagraphs) | 0.78 | 1700 | 15.8 GB | 151 W |

torch.compile alone: 2.1x. GPU is compute-bound from gbs 1536 up
(samples/s flat); bigger batches only fill VRAM, wall time unchanged.
Usable VRAM reported by CUDA is 15.48/16.0 GiB per card (driver
reservation) — gbs 2176 fills 15.7 GB = ~100% of usable. gbs 2304 OOMs
during ACT halt-step spikes even with expandable_segments.

Dev (4070 Ti): sudoku gbs 704 = 11.7 GB peak (100% of usable 11.59 GiB),
1.77-2.0 steps/s; gbs 736 OOMs. Maze (seq 900): server gbs 192 = 15.5 GB,
0.68 steps/s; gbs 256 OOMs.

## Hardware ceiling analysis (2026-09-05, ncu + microbenches)

ncu 2025.3 (from the CUDA 13 image; the 12.8 one fails with
LibraryNotLoaded / CUPTI_INVALID_DEVICE on driver 580) profiling plus
cuBLAS microbenchmarks on the model's GEMM shapes:

- cuBLAS bf16 on (M=90304, N=1536..4096, K=512): 34-49 TFLOPS
- mma.sync bf16 ceiling measured by a register-only microbench
  (hrmfast/tests/bench_mma_peak.cu): 52.6 TFLOPS — this is the practical
  tensor-core peak of the 5060 Ti (no tcgen05 on consumer Blackwell)
- measured train step at gbs 1088/GPU: 1.19 s vs 1.10 s predicted from
  summing the cuBLAS kernel times — the code runs at ~95% of the GEMM
  ceiling; it is NOT launch- or fusion-bound (cudagraphs confirmed: 0%)

## hrmfast custom kernels (separate package, author code untouched)

`hrmfast/` — CUDA extension (setup.py, sm_89/90/120), fused SwiGLU:

- forward `swiglu_forward`: y = silu(x·Wg)·(x·Wu) in ONE kernel —
  mma.sync.m16n8k16 bf16 inline PTX, ldmatrix fragments, cp.async 16B
  double-buffered smem pipeline, XOR swizzle, fused SiLU epilogue,
  fp32 accumulators
- backward: dx GEMM (virtual K=2N, no concat), dW with split-M
  gridDim.z=8 + fp32 atomicAdd workspace, g/u recomputed in registers
  (never saved) — 462 MB less activation memory per batch

Verified by parity tests vs author SwiGLU (fwd rel 0.4%, all grads
rel < 1%) and measured on server:

| | author | hrmfast |
|---|---|---|
| MLP fwd | 12.70 ms | 8.90 ms (kernel 5.89 ms at 48.3 TFLOPS) |
| MLP fwd+bwd | 39.84 ms | 33.76 ms |
| train step gbs 2176 | 0.78 steps/s, 15.7 GB | 0.82 steps/s, 11.4 GB |
| train step gbs 3072 | OOM | 0.59 steps/s, 15.8 GB (97%) |

At gbs 3072 the card hits its power wall (178/180 W). Note the tradeoff:
bigger batch fills memory but costs wall time at fixed optimizer-step
count; paper-faithful runs keep 26041 steps at gbs 2176.

Forward/backward breakdown (server, 1x 5060 Ti, per-GPU batch 1088 = the
per-GPU load of gbs 2176 on 2 GPUs, compile default; CUDA events, mean of
10 steps after 3 warmup; `scripts/bench_fwdbwd.py`):

| impl | fwd ms/batch | bwd ms/batch | fwd us/sample | bwd us/sample |
|---|---|---|---|---|
| author  | 890.66 | 301.63 | 818.62 | 277.23 |
| hrmfast | 821.90 | 343.39 | 755.42 | 315.61 |

fwd = the full model call, i.e. BOTH inner passes of the train step (main
+ no-grad target-Q, 48 block executions); bwd = `loss.backward()` only
(grad part, 16 block-equivalents). Optimizer step and grad all-reduce are
excluded. Sanity check: fwd+bwd 1192 ms author matches the measured 1.19 s
train step at this batch size. hrmfast wins 69 ms on fwd (fused SwiGLU +
LinearResidRmsNorm) but pays back 42 ms on bwd (custom dW path vs cuBLAS);
net step gain here is ~27 ms — the big hrmfast win is memory (462 MB less
activations), which is what unlocks gbs 3072.

Consequence for wall time: sudoku-1k (26041 optimizer steps, the paper's
schedule) takes ~8.8 h on the server. A 1 h run would need ~9x the FLOPs
of 2x 5060 Ti at the algorithm's fixed compute per sample — not reachable
by kernel work alone (we are already at ~95% of the measured GEMM
ceiling); it would require changing the training recipe itself. The
paper's "~10 min" figure is for an 8x datacenter-GPU rig; per-sample-step
cost here matches the authors' code on RTX 5090 within 20% (issue #12).

## Runs

Paper data budget: 20000 epochs over 1000 groups = 20M samples seen.
We keep the data budget and let the batch size float to fill the GPU
(authors used gbs 768 on 8 GPUs; their own README laptop variant uses
384 — batch size is a free knob in the recipe).

- server sudoku-1k: gbs 3456 (99.3% of usable VRAM), epochs 20000,
  eval_interval 2000, 5787 optimizer steps, ~3.1 h measured pace
- server maze-1k: gbs 192
- sudoku-full: README config (epochs 100, lr 3e-4, lr_min_ratio 0.1,
  softmax CE, L_cycles 8, halt_max_steps 8, learned pos), gbs by VRAM

Training-loop metrics: logged to metrics.jsonl every 100 steps
(deferred GPU->CPU sync; METRICS_FLUSH env overrides); eval during
training on 20k-example prefix (eval_test_examples), final evaluate.py
always full test set.

Commands:

```bash
# server
ssh user@192.168.0.1  # via eno1
cd ~/HRM && DATA_ROOT=/mnt/hdd2/hrm NGPU=2 RUN_NAME=<run> \
  bash scripts/train.sh data_path=/data/built/<dataset> ...

# dev
cd ~/HRM && NGPU=1 RUN_NAME=<run> bash scripts/train.sh ...

# final full-test evaluation
bash scripts/evaluate.sh checkpoint=/data/checkpoints/<run>/step_<N>

# HTML report (rsyncs server checkpoints, renders reports/hrm_reproduction.html)
bash scripts/make_report.sh
```
