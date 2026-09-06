import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")

from hrmfast import attention_fused


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-9)).item()


def author_attn(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False).transpose(1, 2)


def check_attn(B, S, H=8):
    torch.manual_seed(B * 1000 + S)
    q = torch.randn(B, S, H, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, 64, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(B, S, H, 64, device="cuda", dtype=torch.bfloat16)

    qa = q.clone().requires_grad_(True)
    ka = k.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    ya = author_attn(qa, ka, va)
    ya.backward(dout)

    qb = q.clone().requires_grad_(True)
    kb = k.clone().requires_grad_(True)
    vb = v.clone().requires_grad_(True)
    yb = attention_fused(qb, kb, vb, 64 ** -0.5)
    yb.backward(dout)

    # at S=1 the true dq/dk are ~0 (dS = p*(dP-D) cancels); bf16 author and fused
    # round o differently, so compare against the fp32 ground truth absolutely
    if S == 1:
        qf, kf, vf, df = [t.float().transpose(1, 2) for t in (q, k, v, dout)]
        p = torch.softmax(qf @ kf.transpose(-1, -2) / 8, dim=-1)
        o = p @ vf
        ds = p * (df @ vf.transpose(-1, -2) - (df * o).sum(-1, keepdim=True))
        dq_ref = (ds @ kf / 8).transpose(1, 2)
        dk_ref = (ds.transpose(-1, -2) @ qf / 8).transpose(1, 2)
        print(f"attn B={B} S={S}: y rel={rel(yb, ya):.5f} "
              f"dq abs={(qb.grad.float() - dq_ref).abs().max().item():.5f} "
              f"dk abs={(kb.grad.float() - dk_ref).abs().max().item():.5f}")
        assert rel(yb, ya) < 2e-2
        assert (qb.grad.float() - dq_ref).abs().max().item() < 2e-2
        assert (kb.grad.float() - dk_ref).abs().max().item() < 2e-2
        return

    ry = rel(yb, ya)
    rdq, rdk, rdv = rel(qb.grad, qa.grad), rel(kb.grad, ka.grad), rel(vb.grad, va.grad)
    print(f"attn B={B} S={S}: y rel={ry:.5f} dq rel={rdq:.5f} dk rel={rdk:.5f} dv rel={rdv:.5f}")
    assert max(ry, rdq, rdk, rdv) < 2e-2, f"attn parity FAILED B={B} S={S}"


def check_attn_strided(B, S, H=8):
    torch.manual_seed(7)
    qkv = torch.randn(B, S, 3 * H, 64, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv[:, :, :H], qkv[:, :, H:2 * H], qkv[:, :, 2 * H:]
    dout = torch.randn(B, S, H, 64, device="cuda", dtype=torch.bfloat16)
    qa = q.clone().requires_grad_(True)
    ka = k.clone().requires_grad_(True)
    va = v.clone().requires_grad_(True)
    author_attn(qa, ka, va).backward(dout)
    qb, kb, vb = q.requires_grad_(True), k.requires_grad_(True), v.requires_grad_(True)
    attention_fused(qb, kb, vb, 64 ** -0.5).backward(dout)
    rdq, rdk, rdv = rel(qb.grad, qa.grad), rel(kb.grad, ka.grad), rel(vb.grad, va.grad)
    print(f"attn strided-view B={B} S={S}: dq rel={rdq:.5f} dk rel={rdk:.5f} dv rel={rdv:.5f}")
    assert max(rdq, rdk, rdv) < 2e-2


check_attn(1088, 82)
check_attn(1, 82)
check_attn(3, 1)
check_attn(17, 96)
check_attn(2, 128)
check_attn(5, 33)
check_attn_strided(64, 82)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


B, S, H, D = 1088, 82, 8, 64
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

print(f"attn fwd: author {bench(lambda: author_attn(q, k, v)):.3f} ms, "
      f"hrmfast {bench(lambda: attention_fused(q, k, v, D ** -0.5)):.3f} ms")
ya = author_attn(q, k, v)
yb = attention_fused(q, k, v, D ** -0.5)
print(f"attn fwd+bwd: author {bench(lambda: ya.backward(dout, retain_graph=True)):.3f} ms (bwd only), "
      f"hrmfast {bench(lambda: yb.backward(dout, retain_graph=True)):.3f} ms (bwd only)")

print("OK")
