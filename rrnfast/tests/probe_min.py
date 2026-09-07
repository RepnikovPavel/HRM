import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"

B = 4
for dt in (torch.float32, torch.bfloat16):
    e2 = torch.randn(B, 81, 20, 96, device=dev).to(dt)
    w4 = torch.randn(96, 96, device=dev) * 0.1
    de3 = torch.randn(B, 81, 20, 96, device=dev).to(dt)
    e2l = e2.clone().requires_grad_()
    w4l = w4.clone().requires_grad_()
    e3 = F.linear(e2l, w4l.to(dt))
    (e3 * de3).sum().backward()
    u = F.linear(de3, w4.to(dt))
    print(dt, "dL/de2 rel err:", ((e2l.grad.float() - u.float()).norm() / u.float().norm()).item())
    # fp64 gt
    gt = de3.double() @ w4.double().t()
    print("  u vs fp64:", ((u.double() - gt).norm() / gt.norm()).item())
    print("  autograd vs fp64:", ((e2l.grad.double() - gt).norm() / gt.norm()).item())
