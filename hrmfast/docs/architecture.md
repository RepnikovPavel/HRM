# hrmfast conventions

Install: `pip install --no-build-isolation ./hrmfast/` (build needs the host
torch; pip build isolation hides it — `ModuleNotFoundError: torch`), or
`scripts/build_hrmfast.sh` (docker, installs into $DATA_ROOT/python-packages).


Custom CUDA/PTX kernels for HRM. No cuBLAS/cuDNN/CUTLASS in the fast path:
tensor cores are driven via inline PTX (`mma.sync`), global memory only via
aligned vectorized packed accesses (`uint4` / `cp.async.cg` 16B), smem
layouts hand-swizzled.

Kernels are specialized for a concrete (GPU arch, task shape) pair and the
naming/ folder structure must reflect that:

- `csrc/arch_sm120/` — RTX 5060 Ti (Blackwell consumer, 2x16 GB setup)
- `csrc/arch_sm89/`  — RTX 4070 Ti (Ada)
- shared helpers in `csrc/common/`
- kernel entry points named `<op>_<layout>_sm<cc>`, e.g.
  `swiglu_gemm_kernel` specialized/instanced per arch; the host dispatch
  table picks the instance from `cudaDeviceProp.major/minor` once.

A kernel tuned for one arch must never silently run on another: dispatch
table only, re-tune tiling per arch (SM count, smem, power envelope).

Parity rule: every kernel ships with a test comparing fwd AND bwd against
the author implementation in `models/` (rel err < 2e-2, bf16), including
edge shapes (M not multiple of tile).

Efficiency rule: report compute efficiency = model_FLOPs /
(measured_time x measured_hw_ceiling). Use scripts/model_flops.py for
model FLOPs and hrmfast/tests/bench_mma_peak.cu for the ceiling. Below 0.9
is unfinished work.
