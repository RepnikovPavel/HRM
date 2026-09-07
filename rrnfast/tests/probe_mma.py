import torch
from torch.utils.cpp_extension import load

mod = load(name="probe_mma", sources=["tests/probe_mma.cu"], extra_cuda_cflags=["-O2"],
           build_directory="/tmp/probe_build", verbose=False)
W = (torch.arange(16, device="cuda").view(-1, 1) * 1000 +
     torch.arange(16, device="cuda").view(1, -1)).bfloat16().contiguous()
for m0, k0 in [(0, 0), (0, 4), (0, 5), (0, 8), (0, 12), (3, 4), (9, 4), (8, 9)]:
    out = mod.run_probe(W, m0, k0)
    got = out[m0]
    exp = (torch.arange(8) * 1000 + k0).float()
    print(f"M0={m0} K0={k0}: max dev {(got - exp.cuda()).abs().max().item():.0f}", got.tolist())
