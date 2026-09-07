import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import torch.nn.functional as F
import rrnfast
import rrnfast_backend as _be
from rrn.model import SudokuRRN


def rel_err(a, b):
    return (a.double() - b.double()).norm().item() / (b.double().norm().item() + 1e-12)


torch.manual_seed(0)
dev = "cuda"
model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
pos = rrnfast.neighbor_pos(model.neighbors)
nb32 = model.neighbors.int()
gen = torch.Generator(device=dev).manual_seed(1234)

B = 64
x = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
h = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
c = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
gh = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
gc = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()

# bf16 reference (autocast, as in training)
xr, hr, cr = (t.clone().requires_grad_() for t in (x, h, c))
with torch.autocast("cuda", dtype=torch.bfloat16):
    h1r, c1r = model._step(xr, hr, cr, False)
((h1r * gh).sum() + (c1r * gc).sum()).backward()

# fp64 truth: same math, fp64 weights/inputs
model64 = SudokuRRN(num_steps=32, edge_drop=0.4).double().cuda()
model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
with torch.no_grad():
    pass
x64 = x.double().requires_grad_()
h64 = h.double().requires_grad_()
c64 = c.double().requires_grad_()
w = model64.msg_layer[0].weight
hw1 = F.linear(h64, w[:, :96])
hw2 = F.linear(h64, w[:, 96:])
e = hw1[:, model64.neighbors] + hw2.unsqueeze(2) + model64.msg_layer[0].bias
for layer in model64.msg_layer[1:]:
    e = layer(e)
m64 = e.sum(2)
xm = torch.cat([x64, m64], -1).view(B * 81, 192)
gates = model64.lstm_ih(xm) + model64.lstm_hh(h64.view(B * 81, 96))
i, f, g, o = gates.chunk(4, -1)
c1_64 = torch.sigmoid(f) * c64.view(B * 81, 96) + torch.sigmoid(i) * torch.tanh(g)
h1_64 = torch.sigmoid(o) * torch.tanh(c1_64)
((h1_64.view(B, 81, 96) * gh.double()).sum() + (c1_64.view(B, 81, 96) * gc.double()).sum()).backward()

# ours
wfast = rrnfast.model_weights(model)
h1f, c1f, mf = _be.rrn_step_fwd(x, h, c, nb32, *wfast, False, 0.0, 0, 0)
out = _be.rrn_step_bwd(x, h, c, mf, gh, gc, nb32, pos, *wfast, False, 0.0, 0, 0)
dx, dhp, dcp, dw0, db1, dw2, db2, dw3, db3, dw4, db4, dwih, dwhh = out

pairs = [
    ("dh", dhp, hr.grad, h64.grad),
    ("w0", dw0, model.msg_layer[0].weight.grad, model64.msg_layer[0].weight.grad),
    ("b1", db1, model.msg_layer[0].bias.grad, model64.msg_layer[0].bias.grad),
    ("w2", dw2, model.msg_layer[2].weight.grad, model64.msg_layer[2].weight.grad),
    ("b2", db2, model.msg_layer[2].bias.grad, model64.msg_layer[2].bias.grad),
    ("w3", dw3, model.msg_layer[4].weight.grad, model64.msg_layer[4].weight.grad),
    ("b3", db3, model.msg_layer[4].bias.grad, model64.msg_layer[4].bias.grad),
    ("w4", dw4, model.msg_layer[6].weight.grad, model64.msg_layer[6].weight.grad),
]
print(f"{'name':4s} {'ours vs 64':>12s} {'ref vs 64':>12s} {'ours vs ref':>12s}")
for nm, ours, ref, gt in pairs:
    print(f"{nm:4s} {rel_err(ours, gt):12.3e} {rel_err(ref, gt):12.3e} {rel_err(ours, ref):12.3e}")
