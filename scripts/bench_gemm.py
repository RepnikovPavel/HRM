import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)

BATCH = 1088
SEQ = 83
H = 512
INTER = 2048

shapes = [
    ("qkv  (M,N,K)=(bs*seq,3H,H)", BATCH * SEQ, 3 * H, H),
    ("o    (M,N,K)=(bs*seq,H,H)", BATCH * SEQ, H, H),
    ("gate (M,N,K)=(bs*seq,2I,H)", BATCH * SEQ, 2 * INTER, H),
    ("down (M,N,K)=(bs*seq,H,I)", BATCH * SEQ, H, INTER),
]

for name, M, N, K in shapes:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    n = 20
    for _ in range(n):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"{name}: {dt * 1e3:7.3f} ms, {2 * M * N * K / dt / 1e12:6.1f} TFLOPS")

q = torch.randn(BATCH, 8, SEQ, 64, device="cuda", dtype=torch.bfloat16)
for _ in range(5):
    F.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
t0 = time.time()
n = 50
for _ in range(n):
    F.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize()
dt = (time.time() - t0) / n
print(f"sdpa heads8 seq{SEQ}: {dt * 1e3:7.3f} ms, {4 * BATCH * 8 * SEQ * SEQ * 64 / dt / 1e12:6.1f} TFLOPS")

x = torch.randn(BATCH * SEQ, H, device="cuda", dtype=torch.bfloat16)
for _ in range(5):
    y = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n):
    y = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
torch.cuda.synchronize()
dt = (time.time() - t0) / n
print(f"rmsnorm fp32 cast: {dt * 1e3:7.3f} ms, {x.numel() * 2 * 3 / dt / 1e9:6.1f} GB/s")
