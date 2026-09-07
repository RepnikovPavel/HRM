import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    import rrnfast
except ImportError:
    rrnfast = None


def sudoku_edge_index():
    neighbors = torch.zeros(81, 20, dtype=torch.long)
    for i in range(81):
        r, c = divmod(i, 9)
        nb = [j for j in range(81)
              if j != i and (j // 9 == r or j % 9 == c
                             or (j // 27 == r // 3 and (j % 9) // 3 == c // 3))]
        assert len(nb) == 20
        neighbors[i] = torch.tensor(sorted(nb))
    return neighbors


class SudokuRRN(nn.Module):
    def __init__(self, num_steps=32, embed_size=16, hidden_dim=96, edge_drop=0.4, amp=True):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.edge_drop = edge_drop
        self.amp = amp

        self.digit_embed = nn.Embedding(10, embed_size)
        self.row_embed = nn.Embedding(9, embed_size)
        self.col_embed = nn.Embedding(9, embed_size)

        self.input_layer = nn.Sequential(
            nn.Linear(3 * embed_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.msg_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.lstm_ih = nn.Linear(2 * hidden_dim, 4 * hidden_dim, bias=False)
        self.lstm_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, 10)

        self.register_buffer("neighbors", sudoku_edge_index(), persistent=False)
        self.register_buffer("rows", torch.arange(81) // 9, persistent=False)
        self.register_buffer("cols", torch.arange(81) % 9, persistent=False)

    def _step(self, x, h, c, train):
        batch_size = x.shape[0]
        d = self.hidden_dim
        w = self.msg_layer[0].weight
        hw1 = F.linear(h, w[:, :d])
        hw2 = F.linear(h, w[:, d:])
        e = hw1[:, self.neighbors] + hw2.unsqueeze(2) + self.msg_layer[0].bias
        for layer in self.msg_layer[1:]:
            e = layer(e)
        e = F.dropout(e, self.edge_drop, train)
        m = e.sum(2)
        flat = batch_size * 81
        xm = torch.cat([x, m], -1).view(flat, 2 * self.hidden_dim)
        gates = self.lstm_ih(xm) + self.lstm_hh(h.view(flat, self.hidden_dim))
        i, f, g, o = gates.chunk(4, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c.view(flat, self.hidden_dim) + i * g
        h = o * torch.tanh(c)
        return h.view(batch_size, 81, self.hidden_dim), c.view(batch_size, 81, self.hidden_dim)

    def forward(self, inputs, train=True):
        batch_size = inputs.shape[0]
        x = torch.cat([
            self.digit_embed(inputs),
            self.row_embed(self.rows).unsqueeze(0).expand(batch_size, -1, -1),
            self.col_embed(self.cols).unsqueeze(0).expand(batch_size, -1, -1),
        ], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            x = self.input_layer(x)

            h = x
            c = torch.zeros_like(x)
            use_fast = (rrnfast is not None and x.is_cuda
                        and x.dtype == torch.bfloat16 and self.hidden_dim == 96)
            if use_fast and not hasattr(self, "_rrnfast_pos"):
                rrnfast.prepare(self)
                self._rrn_seed = int(torch.initial_seed())
            outputs = []
            for t in range(self.num_steps):
                if use_fast:
                    if train:
                        self._rrn_seed += 1
                    h, c = rrnfast.rrn_step(x, h, c, self, train,
                                            seed=self._rrn_seed, step=t)
                elif train:
                    h, c = checkpoint(self._step, x, h, c, train, use_reentrant=False)
                else:
                    h, c = self._step(x, h, c, train)
                outputs.append(self.output_layer(h) if train else None)

            if train:
                logits = torch.stack(outputs, 0)
            else:
                logits = self.output_layer(h)
        return logits.float()

    def loss(self, logits, labels):
        if logits.dim() == 4:
            labels = labels.unsqueeze(0).expand_as(logits[..., 0])
        return F.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
