import os
import sys
import time

import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/work")

from pretrain import PretrainConfig, init_train_state, create_dataloader, train_batch

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
for _ in range(2):
    _, batch, gbs = next(it)
    train_batch(cfg, train_state, batch, gbs, rank=0, world_size=1)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(8):
    _, batch, gbs = next(it)
    train_batch(cfg, train_state, batch, gbs, rank=0, world_size=1)
torch.cuda.synchronize()
print(f"steps/s: {8 / (time.time() - t0):.2f}")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        _, batch, gbs = next(it)
        train_batch(cfg, train_state, batch, gbs, rank=0, world_size=1)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=80))
prof.export_chrome_trace("/data/checkpoints/trace.json.gz")
