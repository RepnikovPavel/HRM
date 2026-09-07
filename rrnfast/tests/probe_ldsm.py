import torch
from torch.utils.cpp_extension import load

mod = load(name="probe_ldsm", sources=["tests/probe_ldsm.cu"], extra_cuda_cflags=["-O2"],
           build_directory="/tmp/probe_build2", verbose=False)

for m0, k0 in [(0, 0), (0, 16)]:
    out = mod.run_probe(m0, k0).cpu()
    # expected: thread t regs {a0a1, a2a3, a4a5, a6a7} =
    # (r, k), (r+8, k), (r, k+8), (r+8, k+8) with r = m0 + t//4, k = k0 + (t%4)*2
    ok = True
    for t in range(32):
        r, k = m0 + t // 4, k0 + (t % 4) * 2
        exp = [100 * r + k, 100 * r + k + 1, 100 * (r + 8) + k, 100 * (r + 8) + k + 1,
               100 * r + k + 8, 100 * r + k + 9, 100 * (r + 8) + k + 8, 100 * (r + 8) + k + 9]
        exp = torch.tensor(exp, dtype=torch.float32).bfloat16().float().tolist()
        got = out[t].tolist()
        if got != exp:
            ok = False
            print(f"t={t}: got {got} exp {exp}")
    print(f"m0={m0} k0={k0}:", "OK" if ok else "FAIL")
