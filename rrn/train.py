import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from rrn.data import SudokuSet, train_batches
from rrn.model import SudokuRRN


def evaluate(model, data, batch_size, limit=None):
    model.eval()
    n = min(limit, len(data)) if limit else len(data)
    correct_cells = 0
    correct_boards = 0
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            q, a = data.batch(range(i, min(i + batch_size, n)))
            logits = model(q, train=False)
            total_loss += model.loss(logits, a).item() * q.shape[0]
            pred = logits.argmax(-1)
            ok = pred == a
            correct_cells += ok.sum().item()
            correct_boards += (ok.sum(1) == 81).sum().item()
    model.train()
    return {
        "loss": total_loss / n,
        "accuracy": correct_cells / (n * 81),
        "exact_accuracy": correct_boards / n,
        "examples": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gbs", type=int, default=1024)
    p.add_argument("--total-samples", type=int, default=20_000_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--edge-drop", type=float, default=0.4)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-examples", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", type=str, default="", choices=["", "default", "max-autotune-no-cudagraphs"])
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_data = SudokuSet(os.path.join(args.data, "train"))
    test_data = SudokuSet(os.path.join(args.data, "test"))

    model = SudokuRRN(num_steps=args.steps, edge_drop=args.edge_drop).cuda()
    if args.compile:
        model._step = torch.compile(model._step, dynamic=False,
                                    mode=None if args.compile == "default" else args.compile)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    raw_model = model.module if world_size > 1 else model
    n_params = sum(t.numel() for t in raw_model.parameters())
    if rank == 0:
        print(f"params: {n_params}, world_size: {world_size}, gbs: {args.gbs}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out, exist_ok=True)
    metrics_file = open(os.path.join(args.out, "metrics.jsonl"), "a", buffering=1) if rank == 0 else None

    total_steps = args.total_samples // args.gbs
    batches = train_batches(train_data, args.gbs, seed=args.seed, rank=rank, world_size=world_size)
    t_start = time.time()
    for step in range(1, total_steps + 1):
        q, a = next(batches)
        logits = model(q, train=True)
        loss = raw_model.loss(logits, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if rank == 0 and step % 100 == 0:
            elapsed = time.time() - t_start
            row = {"step": step, "time": time.time(), "lm_loss": loss.item(),
                   "steps_per_s": step / elapsed, "samples_per_s": step * args.gbs / elapsed,
                   "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps(row), flush=True)
            metrics_file.write(json.dumps(row) + "\n")

        if step % args.eval_interval == 0 or step == total_steps:
            if rank == 0:
                m = evaluate(raw_model, test_data, args.gbs // world_size, args.eval_examples)
                row = {"step": step, "time": time.time(),
                       **{f"test_{k}": v for k, v in m.items()}}
                print(json.dumps(row), flush=True)
                metrics_file.write(json.dumps(row) + "\n")
                torch.save(raw_model.state_dict(), os.path.join(args.out, f"step_{step}"))
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
