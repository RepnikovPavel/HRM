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
