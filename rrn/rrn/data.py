import numpy as np
import torch


class SudokuSet:
    def __init__(self, path):
        self.inputs = np.load(f"{path}/all__inputs.npy").astype(np.int64) - 1
        self.labels = np.load(f"{path}/all__labels.npy").astype(np.int64) - 1
        assert self.inputs.shape == self.labels.shape
        assert self.inputs.shape[1] == 81
        assert self.inputs.min() >= 0 and self.inputs.max() <= 9
        assert self.labels.min() >= 1 and self.labels.max() <= 9

    def __len__(self):
        return self.inputs.shape[0]

    def batch(self, idx):
        return (torch.from_numpy(self.inputs[idx]).cuda(non_blocking=True),
                torch.from_numpy(self.labels[idx]).cuda(non_blocking=True))


def train_batches(data, batch_size, seed=0, rank=0, world_size=1):
    rng = np.random.default_rng(seed)
    n = len(data)
    per_rank = batch_size // world_size
    while True:
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            yield data.batch(perm[i + rank * per_rank: i + (rank + 1) * per_rank])
