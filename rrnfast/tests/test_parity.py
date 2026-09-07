import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, "/tmp/rrnfast-install")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
import torch.nn.functional as F

import rrnfast
import rrnfast_backend
from rrn.model import SudokuRRN


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def rel_err64(a, b):
    return (a.double() - b.double()).norm().item() / (b.double().norm().item() + 1e-12)


def make_inputs(B, dev, gen):
    x = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
    h = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
    c = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
    return x, h, c


def ref_step(model, x, h, c, gh, gc):
    x = x.clone().requires_grad_()
    h = h.clone().requires_grad_()
    c = c.clone().requires_grad_()
    for p_ in model.parameters():
        p_.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h1, c1 = model._step(x, h, c, False)
    ((h1 * gh).sum() + (c1 * gc).sum()).backward()
    g = {
        "x": x.grad, "h": h.grad, "c": c.grad,
        "w0": model.msg_layer[0].weight.grad, "b1": model.msg_layer[0].bias.grad,
        "w2": model.msg_layer[2].weight.grad, "b2": model.msg_layer[2].bias.grad,
        "w3": model.msg_layer[4].weight.grad, "b3": model.msg_layer[4].bias.grad,
        "w4": model.msg_layer[6].weight.grad, "b4": model.msg_layer[6].bias.grad,
        "wih": model.lstm_ih.weight.grad, "whh": model.lstm_hh.weight.grad,
    }
    return h1, c1, g


def ref_step_fp64(model, x, h, c, gh, gc):
    m64 = SudokuRRN(num_steps=32, edge_drop=0.4).double().cuda()
    m64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    B = x.shape[0]
    x64 = x.double().requires_grad_()
    h64 = h.double().requires_grad_()
    c64 = c.double().requires_grad_()
    w = m64.msg_layer[0].weight
    hw1 = F.linear(h64, w[:, :96])
    hw2 = F.linear(h64, w[:, 96:])
    e = hw1[:, m64.neighbors] + hw2.unsqueeze(2) + m64.msg_layer[0].bias
    for layer in m64.msg_layer[1:]:
        e = layer(e)
    m = e.sum(2)
    xm = torch.cat([x64, m], -1).view(B * 81, 192)
    gates = m64.lstm_ih(xm) + m64.lstm_hh(h64.view(B * 81, 96))
    i, f, g, o = gates.chunk(4, -1)
    c1 = torch.sigmoid(f) * c64.view(B * 81, 96) + torch.sigmoid(i) * torch.tanh(g)
    h1 = torch.sigmoid(o) * torch.tanh(c1)
    ((h1.view(B, 81, 96) * gh.double()).sum() + (c1.view(B, 81, 96) * gc.double()).sum()).backward()
    g = {
        "x": x64.grad, "h": h64.grad, "c": c64.grad,
        "w0": m64.msg_layer[0].weight.grad, "b1": m64.msg_layer[0].bias.grad,
        "w2": m64.msg_layer[2].weight.grad, "b2": m64.msg_layer[2].bias.grad,
        "w3": m64.msg_layer[4].weight.grad, "b3": m64.msg_layer[4].bias.grad,
        "w4": m64.msg_layer[6].weight.grad, "b4": m64.msg_layer[6].bias.grad,
        "wih": m64.lstm_ih.weight.grad, "whh": m64.lstm_hh.weight.grad,
    }
    return h1.view(B, 81, 96), c1.view(B, 81, 96), g


def fast_step(model, x, h, c, gh, gc, pos, train=False, p=0.0, seed=0, step=0):
    x = x.clone().requires_grad_()
    h = h.clone().requires_grad_()
    c = c.clone().requires_grad_()
    w = [p_.detach().clone().requires_grad_() for p_ in rrnfast.model_weights(model)]
    h1, c1 = rrnfast.RRNStep.apply(x, h, c, model.neighbors.int(), pos, *w, train, p, seed, step)
    ((h1 * gh).sum() + (c1 * gc).sum()).backward()
    keys = ["w0", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "wih", "whh"]
    g = {"x": x.grad, "h": h.grad, "c": c.grad}
    g.update({k: w_.grad for k, w_ in zip(keys, w)})
    return h1, c1, g


def numeric_grad_rel(model, pos, x, h, c, p, seed, step, gen, eps=0.25, n=128):
    B = x.shape[0]
    args = (model.neighbors.int(), pos, *rrnfast.model_weights(model))
    hf = h.clone().requires_grad_()
    h1, c1 = rrnfast.RRNStep.apply(x, hf, c, *args, True, p, seed, step)
    (h1.float().sum() + c1.float().sum()).backward()
    ga = hf.grad.float().view(-1)
    idxs = torch.randint(0, B * 81 * 96, (n,), generator=gen, device="cuda")
    gn = torch.zeros(n, device="cuda")
    for j, i in enumerate(idxs.tolist()):
        for s in (1, -1):
            hp = h.clone().view(-1)
            hp[i] = (hp[i].float() + s * eps).bfloat16()
            with torch.no_grad():
                a, b_ = rrnfast.RRNStep.apply(x, hp.view(B, 81, 96), c, *args, True, p, seed, step)
            gn[j] += s * (a.float().sum() + b_.float().sum()).item()
    gn /= 2 * eps
    return ((ga[idxs] - gn).norm() / (gn.norm() + 1e-9)).item()


def main():
    torch.manual_seed(0)
    dev = "cuda"
    model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
    pos = rrnfast.neighbor_pos(model.neighbors)
    gen = torch.Generator(device=dev).manual_seed(1234)

    B = 4
    x, h, c = make_inputs(B, dev, gen)
    gh = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()
    gc = torch.randn(B, 81, 96, generator=gen, device=dev).bfloat16()

    h1r, c1r, gr = ref_step(model, x, h, c, gh, gc)
    h1_64, c1_64, g64 = ref_step_fp64(model, x, h, c, gh, gc)
    g64 = dict(g64, h1=h1_64, c1=c1_64)
    h1f, c1f, gf = fast_step(model, x, h, c, gh, gc, pos)

    ok = True
    print(f"{'tensor':6s} {'vs ref':>10s} {'ours~fp64':>10s} {'ref~fp64':>10s}  verdict")
    for name, a, b in [("h1", h1f, h1r), ("c1", c1f, c1r)] + [(k, gf[k], gr[k]) for k in gr]:
        e = rel_err(a, b)
        e_ours = rel_err64(a, g64[name])
        e_ref = rel_err64(b, g64[name])
        # strict <1% where bf16 noise is shallow; deep grads: must not be
        # further from fp64 truth than the reference itself (x1.3 margin)
        strict = e < 1e-2
        fair = e_ours <= e_ref * 1.3 + 1e-3
        good = strict or (fair and e < 5e-2)
        ok &= good
        print(f"{name:6s} {e:10.2e} {e_ours:10.2e} {e_ref:10.2e}  {'OK' if good else 'FAIL'}"
              f"{'' if strict else ' (deep: vs-fp64 criterion)'}")

    # dropout: determinism, seed sensitivity, unbiasedness of m
    B2 = 256
    x2, h2, c2 = make_inputs(B2, dev, gen)
    mk = rrnfast.model_weights(model)
    nb32 = model.neighbors.int()
    ma = rrnfast_backend.rrn_step_fwd(x2, h2, c2, nb32, *mk, True, 0.4, 7, 0)
    mb = rrnfast_backend.rrn_step_fwd(x2, h2, c2, nb32, *mk, True, 0.4, 7, 0)
    mc = rrnfast_backend.rrn_step_fwd(x2, h2, c2, nb32, *mk, True, 0.4, 8, 0)
    m0 = rrnfast_backend.rrn_step_fwd(x2, h2, c2, nb32, *mk, False, 0.0, 7, 0)
    same = all(torch.equal(a, b) for a, b in zip(ma, mb))
    diff = not torch.equal(ma[0], mc[0])
    print(f"dropout determinism: {'OK' if same else 'FAIL'}, seed sensitivity: {'OK' if diff else 'FAIL'}")
    ok &= same and diff
    mr = ma[2].float().mean(dim=(0, 1))
    m0r = m0[2].float().mean(dim=(0, 1))
    bias = (mr - m0r).abs().max().item()
    spread = m0r.abs().mean().item()
    good = bias < 0.05 * spread + 1e-3
    print(f"dropout m unbiased: max|mean diff| {bias:.4f} vs mean|m| {spread:.4f} "
          f"{'OK' if good else 'FAIL'}")
    ok &= good

    # train-mode fwd/bwd mask consistency: numeric-grad noise floor with p=0.4
    # must match the p=0 floor (same seed on both sides of the perturbation)
    B3 = 2
    x3, h3, c3 = make_inputs(B3, dev, gen)
    x3, h3, c3 = x3 * 0.5, h3 * 0.5, c3 * 0.5
    e0n = numeric_grad_rel(model, pos, x3, h3, c3, 0.0, 11, 3, gen)
    e4n = numeric_grad_rel(model, pos, x3, h3, c3, 0.4, 11, 3, gen)
    good = e4n < e0n * 1.5 + 0.02
    print(f"numeric grad rel err: p=0 {e0n:.3e}, p=0.4 {e4n:.3e} {'OK' if good else 'FAIL'}")
    ok &= good

    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
