import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import torch.nn.functional as F
import rrnfast_backend as _be
from rrn.model import SudokuRRN

torch.manual_seed(0)
dev = "cuda"
model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
gen = torch.Generator(device=dev).manual_seed(1234)
B = 4
h = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
c = torch.zeros(B, 81, 96, device=dev).bfloat16()
x = torch.zeros(B, 81, 96, device=dev).bfloat16()

ml = model.msg_layer
w0, b1 = ml[0].weight, ml[0].bias
w2, b2 = ml[2].weight, ml[2].bias
w3, b3 = ml[4].weight, ml[4].bias
w4, b4 = ml[6].weight, ml[6].bias
wih0 = torch.zeros(384, 192, device=dev)
whh0 = torch.zeros(384, 96, device=dev)
nb = model.neighbors

def re(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)

with torch.autocast("cuda", dtype=torch.bfloat16):
    hw1 = F.linear(h, w0[:, :96])
    hw2 = F.linear(h, w0[:, 96:])
    e = hw1[:, nb] + hw2.unsqueeze(2) + b1
    e0 = e  # fp32
    e1 = ml[2](ml[1](e))
    e2 = ml[4](ml[3](e1))
    e3 = ml[6](ml[5](e2))

m = _be.rrn_step_fwd(x, h, c, nb.int(), w0, b1, w2, b2, w3, b3, w4, b4,
                     wih0, whh0, False, 0.0, 0, 0)[2]

for name, ee in [("e0", e0), ("e1", e1), ("e2", e2), ("e3", e3)]:
    print(name, "sum rowsum-norm:", ee.float().sum(2).norm().item())
print("m rel err vs e3.sum:", re(m, e3.sum(2)))
print("m rel err vs e2.sum:", re(m, e2.sum(2)))
print("m rel err vs e1.sum:", re(m, e1.sum(2)))
print("m rel err vs e0.sum:", re(m, e0.sum(2)))
d = (m.float() - e3.float().sum(2))
print("err by col mod 8:", [round(d[..., i::8].norm().item(), 4) for i in range(8)])
print("err by node[:8]:", [round(v, 2) for v in d.norm(dim=-1)[0, :8].tolist()])
print("err by node[20:28]:", [round(v, 2) for v in d.norm(dim=-1)[0, 20:28].tolist()])
print("err by b:", d.norm(dim=(1, 2)).tolist())
