from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import json
import math
import time
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    eval_test_examples: Optional[int] = None


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int

    metric_accum: Any = None
    metric_keys: Any = None
    metric_batches: int = 0


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False, mode=os.environ.get("COMPILE_MODE", "default"))  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    # puzzle_emb exists only when arch.puzzle_emb_ndim > 0 (disabled for
    # sudoku: the dataset has a single shared puzzle id, the token carries
    # no information)
    optimizers = []
    optimizer_lrs = []
    if hasattr(model.model, "puzzle_emb"):
        optimizers.append(CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore

            lr=0,  # Needs to be set by scheduler
            weight_decay=config.puzzle_emb_weight_decay,

            world_size=world_size
        ))
        optimizer_lrs.append(config.puzzle_emb_lr)
    optimizers.append(AdamATan2(
        model.parameters(),

        lr=0,  # Needs to be set by scheduler
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    ))
    optimizer_lrs.append(config.lr)

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def clone_carry(carry):
    inner = carry.inner_carry
    return type(carry)(
        inner_carry=type(inner)(z_H=inner.z_H.clone(), z_L=inner.z_L.clone()),
        steps=carry.steps.clone(),
        halted=carry.halted.clone(),
        current_data={k: v.clone() for k, v in carry.current_data.items()},
    )


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    # cudagraph trees: carry is a previous graph output, clone before reuse
    if os.environ.get("COMPILE_MODE") == "reduce-overhead":
        torch.compiler.cudagraph_mark_step_begin()
        train_state.carry = clone_carry(train_state.carry)
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce: coalesce grads into one flat buffer per dtype; hundreds of
    # tiny all_reduce calls (one per parameter) are latency-bound on PCIe
    if world_size > 1:
        grads_by_dtype = {}
        for param in train_state.model.parameters():
            if param.grad is not None:
                grads_by_dtype.setdefault(param.grad.dtype, []).append(param.grad)
        for grads in grads_by_dtype.values():
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            offset = 0
            for g in grads:
                g.copy_(flat[offset:offset + g.numel()].view_as(g))
                offset += g.numel()
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics: stay on GPU between steps; the per-step .cpu() sync
    # stalls the pipeline (GPU idles through optimizer/python). Flush to CPU
    # every 100 steps — logging-only data, exact per-step values are not
    # needed for training.
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if train_state.metric_accum is None:
            train_state.metric_accum = metric_values
            train_state.metric_keys = metric_keys
        else:
            train_state.metric_accum += metric_values
        train_state.metric_batches += 1

        flush_every = int(os.environ.get("METRICS_FLUSH", "100"))
        if train_state.step % flush_every != 0 and train_state.step < train_state.total_steps:
            return None

        accum, n_batches = train_state.metric_accum, train_state.metric_batches
        train_state.metric_accum = None
        train_state.metric_batches = 0
        if world_size > 1:
            dist.reduce(accum, dst=0)

        if rank == 0:
            vals = accum.cpu().numpy()
            summed = {k: vals[i] for i, k in enumerate(metric_keys)}

            count = max(summed["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size * n_batches if k.endswith("loss") else count)
                               for k, v in summed.items() if k != "count"}
            reduced_metrics["train/count"] = summed["count"] / n_batches

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            while True:
                if os.environ.get("COMPILE_MODE") == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                    carry = clone_carry(carry)
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                
                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)


def log_metrics(config: PretrainConfig, record: dict, step: int):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    with open(os.path.join(config.checkpoint_path, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, "time": time.time(), **record}, default=lambda o: float(o)) + "\n")


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, test_examples_limit=config.eval_test_examples, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress and metric logging
    t_start = time.time()
    if RANK == 0:
        print(f"num_params {sum(x.numel() for x in train_state.model.parameters())}", flush=True)
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                log_metrics(config, metrics, train_state.step)
                if train_state.step % 500 == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / train_state.step * (train_state.total_steps - train_state.step)
                    loss = metrics.get("train/lm_loss", float("nan"))
                    print(f"step {train_state.step}/{train_state.total_steps} lm_loss={loss:.4f} "
                          f"elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m", flush=True)

        ############ Evaluation
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and metrics is not None:
            log_metrics(config, {"eval": metrics}, train_state.step)
            eval_acc = {s: round(float(m.get("exact_accuracy", 0)), 4) for s, m in metrics.items()}
            print(f"step {train_state.step} eval exact_accuracy {eval_acc}", flush=True)
            
        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    launch()
