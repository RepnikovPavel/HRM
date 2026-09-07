import torch
from torch.utils.cpp_extension import load

mod = load(name="probe_gemm", sources=["tests/probe_gemm.cu"], extra_cuda_cflags=["-O3"],
           build_directory="/tmp/probe_build3", verbose=False)
torch.manual_seed(0)
M, N, K = 324, 192, 96
a = torch.randn(M, K, device="cuda").bfloat16()
b = torch.randn(N, K, device="cuda").bfloat16()
out = mod.tn(a, b)
ref = (a.float() @ b.float().t()).bfloat16()
err = (out.float() - ref.float()).norm() / ref.float().norm()
print("tn rel err:", err.item())
d = (out.float() - ref.float())
print("err by col half:", d[:, :96].norm().item(), d[:, 96:].norm().item())
print("err by col mod 8:", [round(d[:, i::8].norm().item(), 4) for i in range(8)])
print("err by row mod 5:", [round(d[i::5].norm().item(), 4) for i in range(5)])
