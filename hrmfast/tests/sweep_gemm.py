import sys
import time

import torch

import hrmfast_backend as be
from hrmfast import swiglu_forward

M = 90304
NCFG = 5

SHAPES = [
    ("qkv", 1536, 512),
    ("o", 512, 512),
    ("down", 512, 1536),
]


def bench(fn, n=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


torch.manual_seed(0)
print(f"M={M}")
results = {}

for name, N, K in SHAPES:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref_y = x @ w.t()
    ref_dx = dy @ w
    ref_dw = dy.t() @ x
    fl_tn = 2 * M * N * K
    fl_nt = 2 * M * N * K
    fl_nn = 2 * M * N * K

    cu_tn = bench(lambda: x @ w.t())
    cu_nn = bench(lambda: dy @ w)
    cu_nt = bench(lambda: dy.t() @ x)
    print(f"[{name}] cuBLAS tn {cu_tn:.3f} ms ({fl_tn/cu_tn/1e9:.1f} TF) | "
          f"nn {cu_nn:.3f} ms ({fl_nn/cu_nn/1e9:.1f} TF) | nt {cu_nt:.3f} ms ({fl_nt/cu_nt/1e9:.1f} TF)")

    for op in ("tn", "nn", "nt"):
        best = None
        for cfg in range(NCFG):
            if op == "tn":
                fn = lambda: be.linear_fwd(x, w, cfg)
                out = fn()
                r = rel(out, ref_y)
            elif op == "nn":
                fn = lambda: be.dbg_nn(dy, w, cfg)
                out = fn()
                r = rel(out, ref_dx)
            else:
                fn = lambda: be.dbg_nt(dy, x, cfg)
                out = fn()
                r = rel(out, ref_dw)
            if r > 2e-2:
                print(f"  {op} cfg{cfg}: PARITY FAIL rel={r:.4f}")
                continue
            ms = bench(fn)
            tf = fl_tn / ms / 1e9
            print(f"  {op} cfg{cfg}: {ms:.3f} ms ({tf:.1f} TF) rel={r:.4f}")
            if best is None or ms < best[1]:
                best = (cfg, ms)
        print(f"  {op} {N}x{K} best: cfg{best[0]} {best[1]:.3f} ms")
        results[(op, N, K)] = best[0]

xg = torch.randn(M, 512, device="cuda", dtype=torch.bfloat16)
wgm = torch.randn(1536, 512, device="cuda", dtype=torch.bfloat16) * 0.05
wum = torch.randn(1536, 512, device="cuda", dtype=torch.bfloat16) * 0.05
ref_sg = torch.nn.functional.silu(xg @ wgm.t()) * (xg @ wum.t())
for cfg in range(NCFG):
    fn = lambda: be.swiglu_forward(xg, wgm, wum, cfg)
    r = rel(fn(), ref_sg)
    ms = bench(fn)
    print(f"  swiglu cfg{cfg}: {ms:.3f} ms ({2*2*M*1536*512/ms/1e9:.1f} TF) rel={r:.4f}")

print()
print("table = {")
for k, v in sorted(results.items()):
    print(f'    ("{k[0]}", {k[1]}, {k[2]}): {v},')
print("}")
