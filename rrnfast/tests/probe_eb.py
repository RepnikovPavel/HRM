import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))
from rrn.model import sudoku_edge_index

mod = load(name="probe_eb", sources=["tests/probe_eb.cu"], extra_cuda_cflags=["-O3"],
           build_directory="/tmp/probe_build6", verbose=False)

torch.manual_seed(9)
dev = "cuda"
B = 4
nb = sudoku_edge_index().cuda()

h = torch.randn(B, 81, 96, device=dev).bfloat16()
w0 = torch.randn(96, 192, device=dev) * 0.1
b1 = torch.randn(96, device=dev) * 0.1
w2 = torch.randn(96, 96, device=dev) * 0.1
b2 = torch.randn(96, device=dev) * 0.1
w3 = torch.randn(96, 96, device=dev) * 0.1
b3 = torch.randn(96, device=dev) * 0.1
w4 = torch.randn(96, 96, device=dev) * 0.1
b4 = torch.randn(96, device=dev) * 0.1
dm = torch.randn(B, 81, 96, device=dev).bfloat16()

w0b = w0.bfloat16()
hw1 = F.linear(h, w0b[:, :96])
hw2 = F.linear(h, w0b[:, 96:])
hw12 = torch.cat([hw1, hw2], -1).view(B * 81, 192).contiguous()

e0 = torch.relu(hw1.float()[:, nb] + hw2.float().unsqueeze(2) + b1).bfloat16()
e1 = torch.relu(F.linear(e0, w2.bfloat16(), b2.bfloat16()))
e2 = torch.relu(F.linear(e1, w3.bfloat16(), b3.bfloat16()))
de3 = dm.view(B, 81, 1, 96).expand(B, 81, 20, 96).contiguous()
de2 = torch.matmul(de3, w4.bfloat16())
da1 = de2 * (e2 > 0)
de1 = torch.matmul(da1, w3.bfloat16())
da0 = de1 * (e1 > 0)
de0 = torch.matmul(da0, w2.bfloat16())
dz0 = de0 * (e0 > 0)

dw4_ref = de3.float().reshape(-1, 96).t() @ e2.float().reshape(-1, 96)
dw3_ref = da1.float().reshape(-1, 96).t() @ e1.float().reshape(-1, 96)
dw2_ref = da0.float().reshape(-1, 96).t() @ e0.float().reshape(-1, 96)
db4_ref = de3.float().sum((0, 1, 2))
db3_ref = da1.float().sum((0, 1, 2))
db2_ref = da0.float().sum((0, 1, 2))
db1_ref = dz0.float().sum((0, 1, 2))
dhw2_ref = dz0.float().sum(2)
dhw1_ref = torch.zeros(B, 81, 96, device=dev)
pos = torch.zeros(81, 20, dtype=torch.long)
for s in range(81):
    for k in range(20):
        j = int(nb[s, k])
        pos[s, k] = int((nb[j] == s).nonzero()[0, 0])
        dhw1_ref[:, s] += dz0.float()[:, j, pos[s, k]]
pos = pos.cuda()

dm192 = torch.zeros(B * 81, 192, device=dev).bfloat16()
dm192[:, 96:] = dm.view(B * 81, 96)

ws2, ws3, ws4, db1, db2, db3, db4, dz0k, dhw12 = mod.run_probe(
    hw12, nb.int(), pos.int(), b1.float(),
    w2.bfloat16(), b2.bfloat16(), w3.bfloat16(), b3.bfloat16(),
    w4.bfloat16(), b4.bfloat16(), dm192, B)

def re(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()

print("dW4:", re(ws4, dw4_ref))
print("dW3:", re(ws3, dw3_ref))
print("dW2:", re(ws2, dw2_ref))
print("db4:", re(db4, db4_ref))
print("db3:", re(db3, db3_ref))
print("db2:", re(db2, db2_ref))
print("db1:", re(db1, db1_ref))
print("dz0:", re(dz0k.view(B, 81, 20, 96), dz0))
print("dhw1:", re(dhw12.view(B, 81, 192)[:, :, :96], dhw1_ref))
print("dhw2:", re(dhw12.view(B, 81, 192)[:, :, 96:], dhw2_ref))

# autograd cross-check of the manual backward chain
w2a = w2.clone().requires_grad_()
w3a = w3.clone().requires_grad_()
w4a = w4.clone().requires_grad_()
b2a = b2.clone().requires_grad_()
b3a = b3.clone().requires_grad_()
b4a = b4.clone().requires_grad_()
e0l = e0.detach().clone().requires_grad_()
z1 = F.linear(e0l, w2a.bfloat16(), b2a.bfloat16())
e1l = torch.relu(z1)
z2 = F.linear(e1l, w3a.bfloat16(), b3a.bfloat16())
e2l = torch.relu(z2)
e3l = F.linear(e2l, w4a.bfloat16(), b4a.bfloat16())
(e3l * de3).sum().backward()
print("autograd dW3 vs manual:", re(w3a.grad.float(), dw3_ref))
print("autograd dW4 vs manual:", re(w4a.grad.float(), dw4_ref))
print("autograd dW2 vs manual:", re(w2a.grad.float(), dw2_ref))
print("autograd db3 vs manual:", re(b3a.grad.float(), db3_ref))
print("autograd dz0 vs manual:", re(e0l.grad.float(), dz0.float() * 0 + e0l.grad.float()))

# direct intermediate check
w3b = w3.clone().requires_grad_()
w4b = w4.clone().requires_grad_()
e1l2 = e1.detach().clone()
z2b = F.linear(e1l2, w3b.bfloat16(), b3.bfloat16())
e2b = torch.relu(z2b)
z2b.retain_grad()
e3b = F.linear(e2b, w4b.bfloat16(), b4.bfloat16())
(e3b * de3).sum().backward()
print("z2.grad vs da1:", re(z2b.grad.float(), da1.float()))
de2_chk = torch.matmul(de3, w4.bfloat16())
print("de2 grad-at-e2 check:", re(e2b.grad.float() if e2b.grad is not None else z2b.grad.float(), de2_chk.float()))
# also check dW3 via de3 with torch matmul using autograd's z2.grad
dw3_auto_manual = z2b.grad.float().reshape(-1,96).t() @ e1.float().reshape(-1,96)
print("dW3 from z2.grad vs autograd w3.grad:", re(dw3_auto_manual, w3b.grad.float()))
print("w3.grad sample:", w3b.grad[0,:4].tolist())
print("da1^T e1 sample:", (da1.float().reshape(-1,96).t() @ e1.float().reshape(-1,96))[0,:4].tolist())

print("da1   [0,0,0,:8]:", da1[0,0,0,:8].float().tolist())
print("z2.gr [0,0,0,:8]:", z2b.grad[0,0,0,:8].float().tolist())
print("de2   [0,0,0,:8]:", de2[0,0,0,:8].float().tolist())
u = torch.matmul(de3, w4b.bfloat16())   # unmasked dL/de2 in autograd dtype path
print("unmask[0,0,0,:8]:", u[0,0,0,:8].float().tolist())
print("mask e2[0,0,0,:8]:", (e2[0,0,0,:8] > 0).float().tolist())
print("mask z2b[0,0,0,:8]:", (z2b[0,0,0,:8] > 0).float().tolist())
print("de3 [0,0,0,:8]:", de3[0,0,0,:8].float().tolist())

# fp64 ground truth for dL/de2[0,0,0,0]
gt = (de3[0,0,0].double() @ w4.double())[0].item()
print("fp64 dL/de2[0,0,0,0]:", gt)
print("F.linear            :", u[0,0,0,0].item())
# full e2b.grad via retain_grad
w4c = w4.clone().requires_grad_()
e2c = e2.detach().clone().requires_grad_()
e3c = F.linear(e2c, w4c.bfloat16(), b4.bfloat16())
(e3c * de3).sum().backward()
print("e2c.grad[0,0,0,:8]  :", e2c.grad[0,0,0,:8].float().tolist())
print("u        [0,0,0,:8] :", u[0,0,0,:8].float().tolist())
print("de3 shape/strides   :", de3.shape, de3.stride(), de3.dtype)
print("e3c shape           :", e3c.shape)
