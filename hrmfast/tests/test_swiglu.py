import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")

from models.layers import SwiGLU
import models.layers as L
from hrmfast import swiglu_forward, swiglu_fused

L.SwiGLUFused = None  # ref module must exercise the author path; hrmfast is called directly

M, H = 1088 * 83, 512

torch.manual_seed(0)
ref = SwiGLU(hidden_size=H, expansion=4.0).cuda().to(torch.bfloat16)
INTER = ref.gate_up_proj.weight.shape[0] // 2
print("inter:", INTER)

x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

wg = ref.gate_up_proj.weight[:INTER].contiguous()
wu = ref.gate_up_proj.weight[INTER:].contiguous()
wd = ref.down_proj.weight.contiguous()

with torch.no_grad():
    y_ref = ref.down_proj(F.silu(ref.gate_up_proj(x)[:, :INTER]) * ref.gate_up_proj(x)[:, INTER:])
    y_fast = swiglu_forward(x, wg, wu) @ wd.t()

diff = (y_ref.float() - y_fast.float()).abs()
rel = diff.max().item() / y_ref.float().abs().max().item()
print(f"parity fwd: max abs diff {diff.max().item():.5f}, rel {rel:.5f}")
assert rel < 2e-2, "parity FAILED"

dout = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)


def check_grads(x_in, tag):
    xa = x_in.clone().requires_grad_(True)
    wga = wg.detach().clone().requires_grad_(True)
    wua = wu.detach().clone().requires_grad_(True)
    wda = wd.detach().clone().requires_grad_(True)
    ya = (F.silu(xa @ wga.t()) * (xa @ wua.t())) @ wda.t()
    ya.backward(dout)

    xb = x_in.clone().requires_grad_(True)
    wgb = wg.detach().clone().requires_grad_(True)
    wub = wu.detach().clone().requires_grad_(True)
    wdb = wd.detach().clone().requires_grad_(True)
    yb = swiglu_fused(xb, wgb, wub) @ wdb.t()
    yb.backward(dout)

    for name, a, b in (("dx", xa.grad, xb.grad), ("dWg", wga.grad, wgb.grad),
                       ("dWu", wua.grad, wub.grad), ("dWd", wda.grad, wdb.grad)):
        d = (a.float() - b.float()).abs().max().item()
        r = d / a.float().abs().max().item()
        print(f"parity bwd {tag} {name}: max abs diff {d:.5f}, rel {r:.5f}")
        assert r < 2e-2, f"grad parity FAILED: {name}"


check_grads(x, "full")

xs = torch.randn(58240, H, device="cuda", dtype=torch.bfloat16)


def check_grads_fused_only(x_in, tag):
    dout_f = torch.randn(x_in.shape[0], INTER, device="cuda", dtype=torch.bfloat16)
    xa = x_in.clone().requires_grad_(True)
    wga = wg.detach().clone().requires_grad_(True)
    wua = wu.detach().clone().requires_grad_(True)
    (F.silu(xa @ wga.t()) * (xa @ wua.t())).backward(dout_f)
    xb = x_in.clone().requires_grad_(True)
    wgb = wg.detach().clone().requires_grad_(True)
    wub = wu.detach().clone().requires_grad_(True)
    swiglu_fused(xb, wgb, wub).backward(dout_f)
    for name, a, b in (("dx", xa.grad, xb.grad), ("dWg", wga.grad, wgb.grad), ("dWu", wua.grad, wub.grad)):
        d = (a.float() - b.float()).abs().max().item()
        r = d / a.float().abs().max().item()
        print(f"parity bwd {tag} {name}: max abs diff {d:.5f}, rel {r:.5f}")
        assert r < 2e-2, f"grad parity FAILED: {name}"


check_grads_fused_only(torch.randn(1000, H, device="cuda", dtype=torch.bfloat16), "M=1000")
check_grads_fused_only(xs, "M=58240")

for name, fn in (("author", lambda: ref(x)), ("hrmfast", lambda: swiglu_forward(x, wg, wu) @ wd.t())):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 50
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f"{name} fwd: {(time.time() - t0) / n * 1e3:.3f} ms")

dout_full = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

params_ref = [p for p in ref.parameters()]
x_grad = x.clone().requires_grad_(True)
xg_fused = x.clone().requires_grad_(True)
wgb = wg.detach().clone().requires_grad_(True)
wub = wu.detach().clone().requires_grad_(True)
wdb = wd.detach().clone().requires_grad_(True)


def step_author():
    y = ref(x_grad)
    torch.autograd.backward([y], [dout_full])


def step_fused():
    y = swiglu_fused(xg_fused, wgb, wub) @ wdb.t()
    torch.autograd.backward([y], [dout_full])


for name, fn in (("author", step_author), ("hrmfast", step_fused)):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 20
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f"{name} fwd+bwd: {(time.time() - t0) / n * 1e3:.3f} ms")

print("OK")
