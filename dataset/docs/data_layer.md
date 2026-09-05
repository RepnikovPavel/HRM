# Data layer

Raw dataset storage lives outside the repo (AGENTS.md rule 14):

- dev machine: `/mnt/nvme/hrm/raw-data`
- server:      `/mnt/hdd2/hrm/raw-data`

Layout:

```
raw-data/
  ARC-AGI/              git clone fchollet/ARC-AGI        (data/training, data/evaluation)
  ARC-AGI-2/            git clone arcprize/ARC-AGI-2      (data/training, data/evaluation)
  ConceptARC/           git clone victorvikram/ConceptARC (corpus/<concept>/*.json)
  sudoku-extreme/       sapientinc/sudoku-extreme CSVs    (train.csv, test.csv)
  maze-30x30-hard-1k/   sapientinc/maze-30x30-hard-1k CSVs(train.csv, test.csv)
  vis/index.json        viewer manifest
  <dataset>/vis/        samples.json, stats.json, SUMMARY.md (per dataset)
```

## Stages

One docker image per stage, wrappers in `scripts/` build and run it with
`--rm --init --user $(id -u):$(id -g)`, mounting only what the stage needs.
Both stages are CPU-only, so no CUDA 12/13 variants are needed yet; that
rule applies to the training stage.

### Download — `scripts/download_data.sh`, `scripts/download_on_server.sh`

`download_on_server.sh` is the end-to-end entry point: it rsyncs the
download stage's code to the server, runs `download_data.sh` there with
`DATA_ROOT=/mnt/hdd2/hrm`, then pulls the result back over LAN via
`rsync_from_server.sh`. Verified idempotent (second run skips
everything, rsync is a no-op).

`download_data.sh` builds `hrm-download:local` from
`docker/DockerfileDownload`, runs `dataset/download_raw.py` inside.
Downloads 3 git repos + 2 HF CSV datasets into `$DATA_ROOT/raw-data`
(default `/mnt/nvme/hrm`). `HF_TOKEN` is read from the environment,
never stored. Idempotent: existing non-empty datasets are skipped.

Network fallbacks (provider/regulator restrictions, AGENTS.md spec):

- GitHub: `git clone --depth 1` from `github.com`, then `gitclone.com`
  mirror, then `archive/refs/heads/{main,master}.tar.gz` tarballs from
  each host.
- HuggingFace: direct `resolve/main` URL per file; falls back from
  `huggingface.co` to `hf-mirror.com`. `curl` retries with resume (`-C -`).

### Sync — `scripts/rsync_from_server.sh`

Dev machine internet is limited, the server's is not: download on the
server, then pull over LAN. Prefers the wire path
(`ssh -o BindInterface=eno1 user@192.168.0.1`), falls back to WiFi
(`192.168.1.68`). Override `SERVER_DATA_ROOT` / `DATA_ROOT` via env.

### Visualization — `scripts/export_vis.sh`, `scripts/serve_vis.sh`

`export_vis.sh` builds `hrm-visualize:local` from
`docker/DockerfileVisualize` and runs `dataset/visualize/export_vis.py`:
for every dataset found under `$DATA_ROOT/raw-data` it writes
`vis/{samples.json,stats.json,SUMMARY.md,TASK.md}` next to the data plus a
top-level `vis/index.json` manifest. `TASK.md` holds the formal task
definition in RL notation (observation, action, reward, allowed and
prohibited states) and the viewer renders it in a collapsible "Task
definition" panel above the samples.

`serve_vis.sh` serves the micro-frontend viewer
(`dataset/visualize/viewer/`: `index.html` + per-dataset ES-module
renderers under `js/renderers/`) and the data dir; it auto-picks the
first free port starting from `$PORT` (default 8971) and prints the URL.
The viewer never touches the repo copy of the data — it reads only
`vis/*.json`. Rendering is verified with headless-Chrome screenshots of
all 5 datasets (no console errors, grids/sudoku/maze paths drawn).

## Measured run (2026-09-05)

- download on server: 102s total; ARC-AGI 7 MB, ARC-AGI-2 9 MB,
  ConceptARC 2 MB, maze 4 MB, sudoku-extreme 762 MB (784 MB total)
- rsync server -> dev over eno1: 813 MB at ~110 MB/s (~8s)
- vis export on dev: 1s for all 5 datasets (incl. counting 3 831 994
  sudoku train rows and 422 786 test rows)

Dataset facts (from generated SUMMARY.md files):

- ARC-AGI: 400 train + 400 eval puzzles, 1718/1782 input-output pairs
- sudoku-extreme: 3 831 994 train + 422 786 test boards, 9x9
- maze-30x30-hard-1k: 1000 + 1000 mazes, 30x30
