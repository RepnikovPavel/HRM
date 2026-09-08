import argparse

import torch

from rrn.data import SudokuSet
from rrn.model import SudokuRRN
from train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--gbs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    args = p.parse_args()

    model = SudokuRRN(num_steps=args.steps, hidden_dim=args.hidden_dim).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    model.eval()

    test_data = SudokuSet(args.data)
    model.train()
    m = evaluate(model, test_data, args.gbs)
    for k, v in m.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
