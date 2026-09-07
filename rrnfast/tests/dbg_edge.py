import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import torch.nn.functional as F
import rrnfast_backend as _be

dev = "cuda"
B = 2
gen = torch.Generator(device=dev).manual_seed(7)

nb = torch.zeros(81, 20, dtype=torch.long)
for i in range(81):
    r, c = divmod(i, 9)
    nbs = [j for j in range(81)
           if j != i and (j // 9 == r or j % 9 == c or (j // 27 == r // 3 and (j % 9) // 3 == c // 3))]
    nb[i] = torch.tensor(sorted(nbs))
nb = nb.cuda()

h = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
c = torch.zeros(B, 81, 96, device=dev).bfloat16()
x = torch.zeros(B, 81, 96, device=dev).bfloat16()

w0 = torch.randn(96, 192, generator=gen, device=dev) * 0.1
b1 = torch.randn(96, generator=gen, device=dev) * 0.1
wih0 = torch.zeros(384, 192, device=dev)
whh0 = torch.zeros(384, 96, device=dev)
I96 = torch.eye(96, device=dev)
zb = torch.zeros(96, device=dev)

def run(w2, b2, w3, b3, w4, b4):
    return _be.rrn_step_fwd(x, h, c, nb.int(), w0, b1, w2, b2, w3, b3, w4, b4,
                            wih0, whh0, False, 0.0, 0, 0)[2]

def re(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)

hw1 = F.linear(h, w0[:, :96].bfloat16())
hw2 = F.linear(h, w0[:, 96:].bfloat16())
e0 = torch.relu(hw1.float()[:, nb] + hw2.float().unsqueeze(2) + b1).bfloat16()

w2r = torch.randn(96, 96, generator=gen, device=dev) * 0.15
b2r = torch.randn(96, generator=gen, device=dev) * 0.1
w3r = torch.randn(96, 96, generator=gen, device=dev) * 0.15
b3r = torch.randn(96, generator=gen, device=dev) * 0.1
w4r = torch.randn(96, 96, generator=gen, device=dev) * 0.15
b4r = torch.randn(96, generator=gen, device=dev) * 0.1

# L2 only
m = run(w2r, b2r, I96 * 0, zb, I96 * 0, zb)
e1 = torch.relu(F.linear(e0, w2r.bfloat16(), b2r.bfloat16()))
print("L2 only:", re(m, e1.sum(2)))

# L2+L3
m = run(w2r, b2r, w3r, b3r, I96 * 0, zb)
e2 = torch.relu(F.linear(e1, w3r.bfloat16(), b3r.bfloat16()))
print("L2+L3:", re(m, e2.sum(2)))

# all
m = run(w2r, b2r, w3r, b3r, w4r, b4r)
e3 = F.linear(e2, w4r.bfloat16(), b4r.bfloat16())
print("L2+L3+L4:", re(m, e3.sum(2)))

d = (m.float() - e3.sum(2).float())
print("err by col mod 8:", [round(d[..., i::8].norm().item(), 4) for i in range(8)])
print("err by col mod 16:", [round(d[..., i::16].norm().item(), 4) for i in range(16)])

print("max abs diff:", d.abs().max().item())
print("m[0,0,:6] ", m[0,0,:6].float().tolist())
print("ref[0,0,:6]", e3.sum(2)[0,0,:6].float().tolist())
print("m zero frac:", (m == 0).float().mean().item(), "ref zero frac:", (e3.sum(2) == 0).float().mean().item())
