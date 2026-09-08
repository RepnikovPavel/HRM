import os
import sys

import torch
from omegaconf import OmegaConf
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/work")

from pretrain import PretrainConfig, init_train_state, create_dataloader

base = OmegaConf.load("/work/config/cfg_pretrain.yaml")
base.pop("defaults")
base.pop("hydra")
base["arch"] = OmegaConf.load("/work/config/arch/hrm_v1.yaml")
merged = OmegaConf.merge(base, OmegaConf.from_cli())
OmegaConf.resolve(merged)
cfg = PretrainConfig(**OmegaConf.to_container(merged))

torch.random.manual_seed(cfg.seed)

train_loader, train_metadata = create_dataloader(
    cfg, "train", test_set_mode=False, epochs_per_iter=20,
    global_batch_size=cfg.global_batch_size, rank=0, world_size=1)
train_state = init_train_state(cfg, train_metadata, world_size=1)

it = iter(train_loader)


def next_batch():
    _, batch, gbs = next(it)
    return {k: v.cuda() for k, v in batch.items()}


def fwd(batch):
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)
    carry, loss, _, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    train_state.carry = carry
    return loss


for _ in range(3):
    loss = fwd(next_batch())
    (loss / cfg.global_batch_size).backward()
    train_state.model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             with_flops=True) as prof:
    loss = fwd(next_batch())
    (loss / cfg.global_batch_size).backward()
    train_state.model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
gbs = cfg.global_batch_size
per_sample = flops / gbs
print(f"train step FLOPs total: {flops/1e12:.2f} TFLOP ({flops:.4e})")
print(f"per sample: {per_sample/1e9:.2f} GFLOP")
print(f"gbs: {gbs}")
for e in sorted(prof.key_averages(), key=lambda e: -e.flops)[:6]:
    if e.flops > 0:
        print(f"  {e.key[:60]:60s} {e.flops/1e12:.3f} TFLOP")
