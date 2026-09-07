import os
import sys
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))
from rrn.model import sudoku_edge_index

mod = load(name="probe_e0", sources=["tests/probe_e0.cu"], extra_cuda_cflags=["-O3"],
           build_directory="/tmp/probe_build4", verbose=False)

torch.manual_seed(3)
B = 4
h = torch.randn(B, 81, 96, device="cuda").bfloat16()
w0 = torch.randn(96, 192, device="cuda") * 0.1
b1 = torch.randn(96, device="cuda") * 0.1
nb = sudoku_edge_index().cuda()

w0b = w0.bfloat16()
hw1 = torch.nn.functional.linear(h, w0b[:, :96])
hw2 = torch.nn.functional.linear(h, w0b[:, 96:])
hw12 = torch.cat([hw1, hw2], -1).view(B * 81, 192).contiguous()
e0_ref = torch.relu(hw1.float()[:, nb] + hw2.float().unsqueeze(2) + b1).bfloat16()

e0 = mod.run_probe(hw12, nb.int(), b1.float(), B)
d = (e0.float() - e0_ref.float())
print("e0 rel err:", (d.norm() / e0_ref.float().norm()).item())
print("err by col mod 8:", [round(d[..., i::8].norm().item(), 4) for i in range(8)])
