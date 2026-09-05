import argparse
import csv
import json
import os
import random
import time
from collections import Counter

csv.field_size_limit(1 << 30)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def count_csv_rows(path):
    rows = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            rows += chunk.count(b"\n")
    return max(rows - 1, 0)


def sample_csv_rows(path, n):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def grid_sizes_arc(examples):
    sizes = Counter()
    for ex in examples:
        grid = ex["input"]
        sizes[f"{len(grid)}x{len(grid[0])}"] += 1
    return sizes


def export_arc_like(name, root, split_dirs, n_samples, seed=42):
    splits = {}
    for split, sub in split_dirs.items():
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        files = []
        for r, _, fs in os.walk(base):
            for f in sorted(fs):
                if f.endswith(".json"):
                    files.append(os.path.join(r, f))
        puzzles = []
        grid_sizes = Counter()
        n_pairs = 0
        for path in files:
            with open(path) as f:
                obj = json.load(f)
            pid = os.path.splitext(os.path.basename(path))[0]
            group = os.path.relpath(path, base).split(os.sep)
            group = group[0] if len(group) > 1 else ""
            puzzles.append({"id": pid, "group": group, "train": obj["train"], "test": obj["test"]})
            for ex in obj["train"] + obj["test"]:
                n_pairs += 1
                g = ex["input"]
                grid_sizes[f"{len(g)}x{len(g[0])}"] += 1
        splits[split] = {"puzzles": puzzles, "files": len(files),
                         "input_output_pairs": n_pairs, "grid_sizes": grid_sizes}
    if not splits:
        return None
    rng = random.Random(seed)
    all_puzzles = [(s, p) for s, d in splits.items() for p in d["puzzles"]]
    samples = [{"split": s, **p} for s, p in rng.sample(all_puzzles, min(n_samples, len(all_puzzles)))]
    stats = {
        "type": "arc",
        "splits": {s: {"puzzles": len(d["puzzles"]), "files": d["files"],
                       "input_output_pairs": d["input_output_pairs"],
                       "top_grid_sizes": d["grid_sizes"].most_common(5)}
                   for s, d in splits.items()},
        "colors": list(range(10)),
    }
    return stats, samples


def export_grid_csv(name, root, kind, n_samples):
    splits = {}
    samples = []
    for split in ("train", "test"):
        path = os.path.join(root, f"{split}.csv")
        if not os.path.isfile(path):
            continue
        t = time.time()
        n_rows = count_csv_rows(path)
        log(f"{name}/{split}.csv: {n_rows} rows counted in {time.time() - t:.0f}s")
        rows = sample_csv_rows(path, n_samples)
        size_bytes = os.path.getsize(path)
        side = int(len(rows[0][1]) ** 0.5) if rows else 0
        splits[split] = {"examples": n_rows, "grid": f"{side}x{side}", "file_bytes": size_bytes}
        for source, q, a, rating in rows:
            if kind == "sudoku":
                q = q.replace(".", "0")
            samples.append({"split": split, "source": source, "rating": rating,
                            "input": q, "output": a, "side": side})
    if not splits:
        return None
    return {"type": kind, "splits": splits}, samples


SUMMARY_HEAD = "# {name}\n\nType: {type}\n\nDisk size: {size}\n\n"

TASKS = {
    "arc": """# Task definition (RL notation)

Abstraction and Reasoning Corpus: few-shot program induction on grids.

- Observation s: k demonstration pairs (input grid -> output grid) plus one
  test input grid. Grids are H x W with 1 <= H, W <= 30, cell values are
  colors 0..9.
- Action a: predict the complete test output grid (a color 0..9 for every
  cell). Single-step episodic task: the whole grid is emitted at once.
- Reward r: 1 iff the predicted grid exactly matches the hidden reference
  output, 0 otherwise. No partial credit.
- Rules of the data: one latent transformation explains every demonstration
  pair of a puzzle and the same transformation maps the test input to the
  test output; the output grid size may differ from the input size.
- Valid states: any grid up to 30x30 with colors 0..9. There are no
  hard-invalid predictions, but any grid that violates the latent rule
  scores 0.
""",
    "sudoku": """# Task definition (RL notation)

9x9 Sudoku (extreme difficulty split; rating column = difficulty score,
higher is harder).

- Observation s: 9x9 board; digits 1..9 are givens, 0 (source '.') marks
  blank cells.
- Action a: assign a digit 1..9 to every blank cell (emitted as a full
  9x9 board).
- Allowed states: boards where every row, every column and every 3x3 box
  contains each digit 1..9 at most once, and all givens are unchanged.
- Prohibited states: any board that duplicates a digit in a row/column/
  box, or overwrites a given.
- Reward r: 1 iff the final board satisfies all constraints (the puzzle
  has a unique solution, so this equals matching the reference), else 0.
- Episodic: one episode = one board, solved in a single emission.
""",
    "maze": """# Task definition (RL notation)

30x30 gridworld pathfinding (hard split; rating column = difficulty).

- Observation s: 30x30 grid; '#' = wall, ' ' = open cell, 'S' = start,
  'G' = goal. The reference output marks the solution path with 'o'.
- Action a: a path from S to G; equivalently a sequence of moves
  {up, down, left, right} executed deterministically from S.
- Allowed states: open cells (' '), S and G.
- Prohibited states: walls '#' and anything outside the grid; a move
  into a prohibited cell is an illegal transition.
- Reward r: 1 iff the path reaches G using only legal moves (the emitted
  output must mark exactly the cells of an S->G path), else 0.
- Episodic MDP: deterministic transitions, terminal state G.
""",
}


def write_outputs(root, name, stats, samples, size_bytes):
    vis_dir = os.path.join(root, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    with open(os.path.join(vis_dir, "TASK.md"), "w") as f:
        f.write(TASKS[stats["type"]])
    with open(os.path.join(vis_dir, "samples.json"), "w") as f:
        json.dump({"dataset": name, "type": stats["type"], "samples": samples}, f)
    stats["disk_bytes"] = size_bytes
    with open(os.path.join(vis_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    lines = [SUMMARY_HEAD.format(name=name, type=stats["type"], size=human(size_bytes))]
    for split, s in stats["splits"].items():
        lines.append(f"## {split}\n")
        for k, v in s.items():
            v = human(v) if k == "file_bytes" else v
            lines.append(f"- {k}: {v}")
        lines.append("")
    with open(os.path.join(vis_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/raw-data")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    t0 = time.time()
    manifest = {"datasets": []}

    arc_layouts = {
        "ARC-AGI": {"training": "data/training", "evaluation": "data/evaluation"},
        "ARC-AGI-2": {"training": "data/training", "evaluation": "data/evaluation"},
        "ConceptARC": {"corpus": "corpus"},
    }
    for name, split_dirs in arc_layouts.items():
        root = os.path.join(args.data_root, name)
        if not os.path.isdir(root):
            continue
        log(f"export {name}")
        result = export_arc_like(name, root, split_dirs, args.samples)
        if result is None:
            log(f"skip {name}: no json puzzles found")
            continue
        stats, samples = result
        size_bytes = dir_size_bytes(root)
        write_outputs(root, name, stats, samples, size_bytes)
        manifest["datasets"].append({"name": name, "type": "arc", "vis": f"{name}/vis"})

    for name, kind in (("sudoku-extreme", "sudoku"), ("maze-30x30-hard-1k", "maze")):
        root = os.path.join(args.data_root, name)
        if not os.path.isdir(root):
            continue
        log(f"export {name}")
        result = export_grid_csv(name, root, kind, args.samples)
        if result is None:
            log(f"skip {name}: no csv files found")
            continue
        stats, samples = result
        size_bytes = dir_size_bytes(root)
        write_outputs(root, name, stats, samples, size_bytes)
        manifest["datasets"].append({"name": name, "type": kind, "vis": f"{name}/vis"})

    manifest_dir = os.path.join(args.data_root, "vis")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "index.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"total elapsed {time.time() - t0:.0f}s, datasets: {len(manifest['datasets'])}")


if __name__ == "__main__":
    main()
