import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import rrnfast
import rrnfast_backend as _be
from rrn.model import SudokuRRN


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


torch.manual_seed(0)
dev = "cuda"
model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
pos = rrnfast.neighbor_pos(model.neighbors)
nb32 = model.neighbors.int()
gen = torch.Generator(device=dev).manual_seed(1234)

B = 4
x = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
h = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
c = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
gh = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
gc = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()

# reference
xr = x.clone().requires_grad_()
hr = h.clone().requires_grad_()
cr = c.clone().requires_grad_()
for p_ in model.parameters():
    p_.grad = None
with torch.autocast("cuda", dtype=torch.bfloat16):
    h1r, c1r = model._step(xr, hr, cr, False)
((h1r * gh).sum() + (c1r * gc).sum()).backward()

# ours, direct C++ calls (no autograd wrapper)
w = rrnfast.model_weights(model)
h1f, c1f, mf = _be.rrn_step_fwd(x, h, c, nb32, *w, False, 0.0, 0, 0)
out = _be.rrn_step_bwd(x, h, c, mf, gh, gc, nb32, pos, *w, False, 0.0, 0, 0)
dx, dhp, dcp, dw0, db1, dw2, db2, dw3, db3, dw4, db4, dwih, dwhh = out

print("h1 ", rel_err(h1f, h1r))
print("c1 ", rel_err(c1f, c1r))
print("dx ", rel_err(dx, xr.grad))
print("dh ", rel_err(dhp, hr.grad))
print("dc ", rel_err(dcp, cr.grad))
print("w0 ", rel_err(dw0, model.msg_layer[0].weight.grad))
print("b1 ", rel_err(db1, model.msg_layer[0].bias.grad))
print("w2 ", rel_err(dw2, model.msg_layer[2].weight.grad))
print("b2 ", rel_err(db2, model.msg_layer[2].bias.grad))
print("w3 ", rel_err(dw3, model.msg_layer[4].weight.grad))
print("b3 ", rel_err(db3, model.msg_layer[4].bias.grad))
print("w4 ", rel_err(dw4, model.msg_layer[6].weight.grad))
print("b4 ", rel_err(db4, model.msg_layer[6].bias.grad))
print("wih", rel_err(dwih, model.lstm_ih.weight.grad))
print("whh", rel_err(dwhh, model.lstm_hh.weight.grad))

# structure of w3 error
d = (dw3.float() - model.msg_layer[4].weight.grad.float())
print("w3 err by row mod 8:", [round(d[i::8].norm().item(), 4) for i in range(8)])
print("w3 err by col mod 8:", [round(d[:, i::8].norm().item(), 4) for i in range(8)])
print("w3 got[0,:6] ", dw3[0, :6].tolist())
print("w3 ref[0,:6] ", model.msg_layer[4].weight.grad[0, :6].tolist())
print("w3 got[0,48:54]", dw3[0, 48:54].tolist())
print("w3 ref[0,48:54]", model.msg_layer[4].weight.grad[0, 48:54].tolist())

# cross-correlation: maybe outputs are swapped/misassigned
refs = {"w0": model.msg_layer[0].weight.grad, "b1": model.msg_layer[0].bias.grad,
        "w2": model.msg_layer[2].weight.grad, "b2": model.msg_layer[2].bias.grad,
        "w3": model.msg_layer[4].weight.grad, "b3": model.msg_layer[4].bias.grad,
        "w4": model.msg_layer[6].weight.grad, "b4": model.msg_layer[6].bias.grad}
gots = {"w0": dw0, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3, "w4": dw4, "b4": db4}
for gn, g in gots.items():
    cands = {rn: rv for rn, rv in refs.items() if rv.shape == g.shape}
    best = min(cands, key=lambda rn: rel_err(g, cands[rn]))
    print(f"ours[{gn}] best-matches ref[{best}] rel={rel_err(g, cands[best]):.3f}")
