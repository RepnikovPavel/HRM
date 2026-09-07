import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, "/tmp/rrnfast-install")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import rrnfast
import rrnfast_backend as _be
from rrn.model import SudokuRRN

torch.manual_seed(0)
dev = "cuda"
model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
rrnfast.prepare(model)

B = 1024
x = torch.randn(B, 81, 96, device=dev).bfloat16()
h = torch.randn(B, 81, 96, device=dev).bfloat16()
c = torch.zeros(B, 81, 96, device=dev).bfloat16()
gh = torch.randn(B, 81, 96, device=dev).bfloat16()
gc = torch.randn(B, 81, 96, device=dev).bfloat16()

w = rrnfast.model_weights(model)
nb32 = model.neighbors.int()
pos = model._rrnfast_pos

for _ in range(3):
    h1, c1, m = _be.rrn_step_fwd(x, h, c, nb32, *w, True, 0.4, 0, 0)
    out = _be.rrn_step_bwd(x, h, c, m, gh, gc, nb32, pos, *w, True, 0.4, 0, 0)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        h1, c1, m = _be.rrn_step_fwd(x, h, c, nb32, *w, True, 0.4, 0, 0)
    torch.cuda.synchronize()
print("== FWD step (x3) ==")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        out = _be.rrn_step_bwd(x, h, c, m, gh, gc, nb32, pos, *w, True, 0.4, 0, 0)
    torch.cuda.synchronize()
print("== BWD step (x3) ==")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
