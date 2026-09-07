import os
import sys
import time
import torch
from torch.utils.cpp_extension import load

mod = load(name="probe_perf", sources=["tests/probe_perf.cu"], extra_cuda_cflags=["-O3"],
           build_directory="/tmp/probe_build7", verbose=False)

torch.manual_seed(0)
dev = "cuda"
B = 1024
hw12 = torch.randn(B * 81, 192, device=dev).bfloat16()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))
from rrn.model import sudoku_edge_index
nb = sudoku_edge_index().cuda().int()
b1 = torch.randn(96, device=dev)
w = [torch.randn(96, 96, device=dev).bfloat16() * 0.1 for _ in range(3)]
bs = [torch.randn(96, device=dev).bfloat16() * 0.1 for _ in range(3)]
m = torch.empty(B, 81, 96, device=dev).bfloat16()

for level in (0, 1, 2, 3, 13):
    fn = lambda: mod.run_probe(hw12, nb, b1, w[0], bs[0], w[1], bs[1], w[2], bs[2], m, B, level)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10 * 1e3
    print(f"level {level}: {dt:.3f} ms")
