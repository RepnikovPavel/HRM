import os
import sys
import time

import torch
from omegaconf import OmegaConf

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

n = 10
fwd_ms = 0.0
bwd_ms = 0.0
for _ in range(n):
    batch = next_batch()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e0.record()
    loss = fwd(batch)
    e1.record()
    (loss / cfg.global_batch_size).backward()
    e2.record()
    torch.cuda.synchronize()
    fwd_ms += e0.elapsed_time(e1)
    bwd_ms += e1.elapsed_time(e2)
    train_state.model.zero_grad(set_to_none=True)

gbs = cfg.global_batch_size
fwd_ms /= n
bwd_ms /= n
print(f"gbs_per_gpu={gbs} steps_measured={n}")
print(f"fwd_batch_ms={fwd_ms:.2f} bwd_batch_ms={bwd_ms:.2f} total_ms={fwd_ms + bwd_ms:.2f}")
print(f"fwd_sample_us={1000 * fwd_ms / gbs:.2f} bwd_sample_us={1000 * bwd_ms / gbs:.2f}")

train_state.model.eval()
train_state.carry = None
batch = next_batch()
with torch.inference_mode():
    with torch.device("cuda"):
        carry = train_state.model.initial_carry(batch)
    while True:
        carry, _, _, _, all_finish = train_state.model(carry=carry, batch=batch, return_keys=[])
        if all_finish:
            break
    torch.cuda.synchronize()
    m = 3
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(m):
        with torch.device("cuda"):
            carry = train_state.model.initial_carry(batch)
        while True:
            carry, _, _, _, all_finish = train_state.model(carry=carry, batch=batch, return_keys=[])
            if all_finish:
                break
    e1.record()
    torch.cuda.synchronize()
eval_ms = e0.elapsed_time(e1) / m
print(f"eval_fwd_batch_ms={eval_ms:.2f} eval_fwd_sample_us={1000 * eval_ms / gbs:.2f}")
