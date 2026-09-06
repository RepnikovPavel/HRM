# Gotchas

- `.gitmodules` uses `git@github.com:` SSH URLs; the download stage
  deliberately uses HTTPS hosts instead, because the server has no SSH
  keys for GitHub and SSH (port 22) is exactly what providers throttle.
- `huggingface_hub.hf_hub_download` pins `HF_ENDPOINT` at import time,
  which makes per-request endpoint fallback impossible; the download
  stage therefore fetches plain `resolve/main` URLs with `curl`
  (resume via `-C -` included).
- `git clone` exits 0 even when the remote hands back an empty
  repository (observed with the gitclone.com mirror: "You appear to have
  cloned an empty repository"). The downloader treats a clone containing
  only `.git` as failure and moves on to the next mirror / the tarball
  fallback.
- Mirror stalls: hf-mirror.com once served headers then 0 bytes/s for
  minutes. `curl` alone does not treat that as an error, so downloads
  pass `--speed-limit 10240 --speed-time 30` to abort slow transfers and
  let `--retry` kick in.
- Host/port overrides: `GIT_HOSTS` and `HF_ENDPOINTS` env vars replace
  the default mirror lists (comma-separated), e.g.
  `HF_ENDPOINTS=https://hf-mirror.com bash scripts/download_data.sh`.
- Port conflicts: the viewer script `scripts/serve_vis.sh` auto-picks
  the next free port starting from `$PORT` (default 8971); verified:
  with 8971 busy a second instance bound 8972. Never leave an old
  container running and expect the same port.
- Measured on 2026-09-05: gitclone.com mirror cloned ARC-AGI (7 MB) in
  23 s; hf-mirror.com served maze train.csv+test.csv (3.6 MB) in ~2 s.
- The build scripts (`dataset/build_*.py`) still reference
  `dataset/raw-data/...` inside the repo; pointing them at
  `$DATA_ROOT/raw-data` is task 3 of the specification.

## Training stage (task 2)

- adam-atan2 0.0.3 (wheel AND sdist) hardcodes
  `NVIDIA_SUPPORTED_ARCHS = {"80","86","89","90"}` — no sm_120 kernel for
  RTX 5060 Ti, fails with `cudaErrorNoKernelImageForDevice`. The training
  Dockerfiles patch the sdist at build time.
- torch 2.9 `cache_dir_utils.default_cache_dir()` calls
  `getpass.getuser()` before reading `TORCHINDUCTOR_CACHE_DIR`, so
  `docker run --user <uid>` without a passwd entry crashes
  `torch.compile` at import; the images create uid 1000 (`useradd hrm`).
- Hydra struct mode: `checkpoint_path` must exist in
  `config/cfg_pretrain.yaml` to be overridable from CLI.
- Memory allocator: torch 2.9 warns `PYTORCH_CUDA_ALLOC_CONF` is
  deprecated, but the replacement `PYTORCH_ALLOC_CONF` is silently IGNORED
  in 2.9.1 — a full training run OOMed with it at gbs 2176 while the old
  variable works. We ship `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  despite the deprecation warning; expandable segments let bigger batches
  survive ACT halt-step memory spikes (2176 OK, 2304 still OOM).
- torch.compile mode `reduce-overhead` (CUDA graphs) initially crashed
  with "accessing tensor output of CUDAGraphs that has been overwritten";
  fixed by cloning the ACT carry between steps in `pretrain.py`. Result
  measured: 0% speedup — the workload is GEMM-bound, not launch-bound.
- `eval_interval` must divide `epochs` (assert in pretrain.py).
- In-training eval on the full 422k-example sudoku test set costs more
  than the training it monitors; `eval_test_examples=20000` keeps it
  cheap (final `evaluate.py` always runs the full set).
