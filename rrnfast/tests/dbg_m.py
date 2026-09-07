import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, "/tmp/rrnfast-install")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import torch.nn.functional as F

import rrnfast
from rrn.model import SudokuRRN

torch.manual_seed(0)
dev = "cuda"
model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
pos = rrnfast.neighbor_pos(model.neighbors)
gen = torch.Generator(device=dev).manual_seed(1234)

B = 4
x = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
h = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
c = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()

# reference m
with torch.autocast("cuda", dtype=torch.bfloat16):
    w = model.msg_layer[0].weight
    hw1 = F.linear(h, w[:, :96])
    hw2 = F.linear(h, w[:, 96:])
    e = hw1[:, model.neighbors] + hw2.unsqueeze(2) + model.msg_layer[0].bias
    for layer in model.msg_layer[1:]:
        e = layer(e)
    m_ref = e.sum(2)

# reference gates/h1/c1
with torch.autocast("cuda", dtype=torch.bfloat16):
    flat = B * 81
    xm = torch.cat([x, m_ref], -1).view(flat, 192)
    gates = model.lstm_ih(xm) + model.lstm_hh(h.view(flat, 96))
    i, f, g, o = gates.chunk(4, -1)
    i = torch.sigmoid(i); f = torch.sigmoid(f); g = torch.tanh(g); o = torch.sigmoid(o)
    c_ref = (f * c.view(flat, 96) + i * g)
    h_ref = (o * torch.tanh(c_ref))

import rrnfast_backend as _be
h1f, c1f, mf = _be.rrn_step_fwd(x, h, c, model.neighbors.int(),
                                *rrnfast.model_weights(model), False, 0.0, 0, 0)

def re(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)

print("m  rel err:", re(mf.view(B, 81, 96), m_ref))
print("h1 rel err:", re(h1f, h_ref.view(B, 81, 96)))
print("c1 rel err:", re(c1f, c_ref.view(B, 81, 96)))

# per-node breakdown of m error
d = (mf.view(B, 81, 96).float() - m_ref.float()).norm(dim=-1)
print("worst nodes:", d.max(dim=0)[0].topk(5))
print("node0 err:", d[:, 0], "node1 err:", d[:, 1], "node80 err:", d[:, 80])
