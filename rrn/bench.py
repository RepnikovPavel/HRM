import argparse

import torch

from rrn.data import SudokuSet
from rrn.model import SudokuRRN


def timed(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--gbs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", type=str, default="", choices=["", "default", "reduce-overhead", "max-autotune-no-cudagraphs"])
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_data = SudokuSet(f"{args.data}/train")
    test_data = SudokuSet(f"{args.data}/test")

    model = SudokuRRN(num_steps=args.steps, hidden_dim=args.hidden_dim, amp=not args.no_amp).cuda()
    if args.compile:
        model._step = torch.compile(model._step, dynamic=False,
                                    mode=None if args.compile == "default" else args.compile)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    q, a = train_data.batch(range(args.gbs))
    tq, ta = test_data.batch(range(args.gbs))

    def fwd_train():
        return model.loss(model(q, train=True), a)

    def fwd_bwd():
        loss = fwd_train()
        opt.zero_grad(set_to_none=True)
        loss.backward()

    def fwd_test():
        with torch.no_grad():
            model(q, train=False)

    fwd_ms = timed(fwd_train)
    fwdbwd_ms = timed(fwd_bwd)
    test_ms = timed(fwd_test)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    gbs = args.gbs
    print(f"gbs={gbs} steps={args.steps} peak_mem_gib={peak_gib:.2f}")
    print(f"train_fwd_batch_ms={fwd_ms:.2f} train_bwd_batch_ms={fwdbwd_ms - fwd_ms:.2f} train_fwdbwd_batch_ms={fwdbwd_ms:.2f}")
    print(f"train_fwd_sample_us={1000 * fwd_ms / gbs:.2f} train_bwd_sample_us={1000 * (fwdbwd_ms - fwd_ms) / gbs:.2f}")
    print(f"test_fwd_batch_ms={test_ms:.2f} test_fwd_sample_us={1000 * test_ms / gbs:.2f}")
    print(f"steps_per_s={1000 / fwdbwd_ms:.2f} samples_per_s={1000 * gbs / fwdbwd_ms:.1f}")


if __name__ == "__main__":
    main()
