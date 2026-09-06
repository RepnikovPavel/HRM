import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")

import models.layers as L
import models.hrm.hrm_act_v1 as H
from hrmfast import LinearFused, linear_resid_rmsnorm

torch.manual_seed(0)
EPS = 1e-5


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-9)).item()


def check_linear(M, N, K):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    dout = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    (F.linear(xa, wa)).backward(dout)
    xb = x.clone().requires_grad_(True)
    wb = w.clone().requires_grad_(True)
    LinearFused.apply(xb, wb).backward(dout)
    ry = rel(LinearFused.apply(x, w), F.linear(x, w))
    print(f"linear M={M} N={N} K={K}: y rel={ry:.5f} dx rel={rel(xb.grad, xa.grad):.5f} "
          f"dW rel={rel(wb.grad, wa.grad):.5f}")
    assert ry < 2e-2 and rel(xb.grad, xa.grad) < 2e-2 and rel(wb.grad, wa.grad) < 2e-2


def check_resid_rmsnorm(M, N, K):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    resid = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    ra = resid.clone().requires_grad_(True)
    L.rms_norm(ra + F.linear(xa, wa), EPS).backward(dout)

    xb = x.clone().requires_grad_(True)
    wb = w.clone().requires_grad_(True)
    rb = resid.clone().requires_grad_(True)
    yb = linear_resid_rmsnorm(xb, wb, rb, EPS)
    yb.backward(dout)

    ya = L.rms_norm(resid + F.linear(x, w), EPS)
    print(f"resid_rmsnorm M={M} N={N} K={K}: y rel={rel(yb, ya):.5f} dx rel={rel(xb.grad, xa.grad):.5f} "
          f"dW rel={rel(wb.grad, wa.grad):.5f} dresid rel={rel(rb.grad, ra.grad):.5f}")
    assert rel(yb, ya) < 2e-2 and rel(xb.grad, xa.grad) < 2e-2 and rel(wb.grad, wa.grad) < 2e-2 \
        and rel(rb.grad, ra.grad) < 2e-2


def set_fused(on):
    L.SwiGLUFused = L._SwiGLUFused if on else None
    L.LinearFused = L._LinearFused if on else None
    L.LinearResidRmsNorm = L._LinearResidRmsNorm if on else None
    L.AttentionFused = L._AttentionFused if on else None
    H.LinearResidRmsNorm = L._LinearResidRmsNorm if on else None


L._SwiGLUFused = L.SwiGLUFused
L._LinearFused = L.LinearFused
L._LinearResidRmsNorm = L.LinearResidRmsNorm
L._AttentionFused = L.AttentionFused

check_linear(90304, 1536, 512)
check_linear(90304, 512, 512)
check_linear(90304, 512, 1536)
check_linear(1000, 512, 1536)
check_resid_rmsnorm(90304, 512, 512)
check_resid_rmsnorm(90304, 512, 1536)
check_resid_rmsnorm(1000, 512, 1536)

cfg = H.HierarchicalReasoningModel_ACTV1Config(
    batch_size=4, seq_len=83, puzzle_emb_ndim=0, num_puzzle_identifiers=1, vocab_size=16,
    H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=512, expansion=4.0,
    num_heads=8, pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0)
torch.manual_seed(1)
block_a = H.HierarchicalReasoningModel_ACTV1Block(cfg).cuda().to(torch.bfloat16)
block_b = H.HierarchicalReasoningModel_ACTV1Block(cfg).cuda().to(torch.bfloat16)
block_b.load_state_dict(block_a.state_dict())
rot = L.RotaryEmbedding(dim=64, max_position_embeddings=83, base=10000.0).cuda()
cos_sin = rot()
cos_sin = (cos_sin[0].cuda(), cos_sin[1].cuda())

x = torch.randn(4, 83, 512, device="cuda", dtype=torch.bfloat16)
dout = torch.randn(4, 83, 512, device="cuda", dtype=torch.bfloat16)

set_fused(False)
xa = x.clone().requires_grad_(True)
ya = block_a(cos_sin, xa)
ya.backward(dout)

set_fused(True)
xb = x.clone().requires_grad_(True)
yb = block_b(cos_sin, xb)
yb.backward(dout)

print(f"block: y rel={rel(yb, ya):.5f} dx rel={rel(xb.grad, xa.grad):.5f}")
assert rel(yb, ya) < 2e-2 and rel(xb.grad, xa.grad) < 2e-2
for (na, pa), (nb, pb) in zip(block_a.named_parameters(), block_b.named_parameters()):
    r = rel(pb.grad, pa.grad)
    print(f"block: d[{na}] rel={r:.5f}")
    assert r < 2e-2, na

M, H_ = 1088 * 83, 512
xs = torch.randn(M, H_, device="cuda", dtype=torch.bfloat16, requires_grad=True)
wqkv = torch.randn(1536, H_, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn(M, 1536, device="cuda", dtype=torch.bfloat16)


def bench(fn, n=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


w = wqkv.detach()
resid512 = torch.randn(M, H_, device="cuda", dtype=torch.bfloat16)
w512 = w[:H_].contiguous()
print(f"qkv fwd: cublas {bench(lambda: xs.detach() @ w.t()):.3f} ms, "
      f"hrmfast {bench(lambda: LinearFused.apply(xs.detach(), w)):.3f} ms")
print(f"linear+resid+rmsnorm: author {bench(lambda: L.rms_norm(resid512 + (xs.detach() @ w512.t()), EPS)):.3f} ms, "
      f"hrmfast {bench(lambda: linear_resid_rmsnorm(xs.detach(), w512, resid512, EPS)):.3f} ms")

print("OK")
