import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))
from rrn.model import sudoku_edge_index

mod = load(name="probe_bwdnt", sources=["tests/probe_bwd.cu"], extra_cuda_cflags=["-O3"],
           build_directory="/tmp/probe_build5", verbose=False)

torch.manual_seed(5)
dev = "cuda"
nb = sudoku_edge_index().cuda()

h = torch.randn(81, 96, device=dev).bfloat16()
w0 = torch.randn(96, 192, device=dev) * 0.1
b1 = torch.randn(96, device=dev) * 0.1
w2 = torch.randn(96, 96, device=dev) * 0.1
b2 = torch.randn(96, device=dev) * 0.1
w3 = torch.randn(96, 96, device=dev) * 0.1
b3 = torch.randn(96, device=dev) * 0.1
w4 = torch.randn(96, 96, device=dev) * 0.1
b4 = torch.randn(96, device=dev) * 0.1
dm_full = torch.randn(81, 192, device=dev).bfloat16()

w0b = w0.bfloat16()
hw1 = F.linear(h, w0b[:, :96])
hw2 = F.linear(h, w0b[:, 96:])
hw12 = torch.cat([hw1, hw2], -1).contiguous()

e0 = torch.relu(hw1.float()[nb] + hw2.float().unsqueeze(1) + b1).bfloat16()
e1 = torch.relu(F.linear(e0, w2.bfloat16(), b2.bfloat16()))
e2 = torch.relu(F.linear(e1, w3.bfloat16(), b3.bfloat16()))
de3 = dm_full[:, 96:].contiguous().view(81, 1, 96).expand(81, 20, 96).contiguous()
de2 = F.linear(de3, w4.bfloat16())
da1 = de2 * (e2 > 0)

dw4_ref = de3.float().reshape(-1, 96).t() @ e2.float().reshape(-1, 96)
dw3_ref = da1.float().reshape(-1, 96).t() @ e1.float().reshape(-1, 96)

ws4, ws3 = mod.run_probe(hw12, nb.int(), b1.float(), w2.bfloat16(), b2.bfloat16(),
                         w3.bfloat16(), b3.bfloat16(), w4.bfloat16(), b4.bfloat16(),
                         dm_full[:, 96:])

for nm, got, rf in [("dW4", ws4, dw4_ref), ("dW3", ws3, dw3_ref)]:
    e = (got - rf).norm() / (rf.norm() + 1e-12)
    print(f"{nm}: rel err {e.item():.4e}")
print("dW4[0,:6] got", ws4[0, :6].tolist())
print("dW4[0,:6] ref", dw4_ref[0, :6].tolist())
print("dW3[0,:6] got", ws3[0, :6].tolist())
print("dW3[0,:6] ref", dw3_ref[0, :6].tolist())
