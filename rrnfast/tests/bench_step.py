import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".install")))
sys.path.insert(0, "/tmp/rrnfast-install")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rrn"))

import torch
from torch.utils.checkpoint import checkpoint

import rrnfast
from rrn.model import SudokuRRN


def bench(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.manual_seed(0)
    dev = "cuda"
    model = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
    rrnfast.prepare(model)
    model_c = SudokuRRN(num_steps=32, edge_drop=0.4).cuda()
    model_c.load_state_dict(model.state_dict())
    step_c = torch.compile(model_c._step)

    for B in (1024, 4096):
        x = torch.randn(B, 81, 96, device=dev).bfloat16()
        h0 = torch.randn(B, 81, 96, device=dev).bfloat16()
        c0 = torch.zeros(B, 81, 96, device=dev).bfloat16()
        res = {}

        def ref_fwd(step):
            h, c = h0, c0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(32):
                    h, c = step(x, h, c, False)
            return h

        def ref_train(step):
            h, c = h0.clone().requires_grad_(), c0.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(32):
                    h, c = checkpoint(step, x, h, c, True, use_reentrant=False)
            (h.float().sum() + c.float().sum()).backward()

        def fast_fwd():
            h, c = h0, c0
            with torch.no_grad():
                for t in range(32):
                    h, c = rrnfast.rrn_step(x, h, c, model, False, seed=0, step=t)
            return h

        def fast_train():
            h, c = h0.clone().requires_grad_(), c0.clone().requires_grad_()
            for t in range(32):
                h, c = rrnfast.rrn_step(x, h, c, model, True, seed=0, step=t)
            (h.float().sum() + c.float().sum()).backward()

        res["eager fwd"] = bench(lambda: ref_fwd(model._step))
        res["compile fwd"] = bench(lambda: ref_fwd(step_c))
        res["ours fwd"] = bench(fast_fwd)

        if B <= 1024:
            res["eager fwd+bwd"] = bench(lambda: ref_train(model._step), warmup=2, iters=3)
            res["compile fwd+bwd"] = bench(lambda: ref_train(step_c), warmup=2, iters=3)
            res["ours fwd+bwd"] = bench(fast_train, warmup=2, iters=3)

        print(f"== B={B} (32 steps) ==")
        for k, v in res.items():
            print(f"  {k:16s} {v:8.1f} ms   {B * 1000.0 / v:8.0f} samples/s (step-throughput)")
        if "ours fwd+bwd" in res:
            print(f"  speedup vs eager: fwd x{res['eager fwd'] / res['ours fwd']:.2f}, "
                  f"train x{res['eager fwd+bwd'] / res['ours fwd+bwd']:.2f}")
            print(f"  speedup vs compile: fwd x{res['compile fwd'] / res['ours fwd']:.2f}, "
                  f"train x{res['compile fwd+bwd'] / res['ours fwd+bwd']:.2f}")
        else:
            print(f"  speedup vs eager: fwd x{res['eager fwd'] / res['ours fwd']:.2f}, "
                  f"vs compile fwd x{res['compile fwd'] / res['ours fwd']:.2f}")
        del x, h0, c0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
