import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

GIT_REPOS = [
    ("ARC-AGI", "fchollet", "ARC-AGI"),
    ("ARC-AGI-2", "arcprize", "ARC-AGI-2"),
    ("ConceptARC", "victorvikram", "ConceptARC"),
]

HF_DATASETS = [
    ("sudoku-extreme", "sapientinc/sudoku-extreme", ["train.csv", "test.csv"]),
    ("maze-30x30-hard-1k", "sapientinc/maze-30x30-hard-1k", ["train.csv", "test.csv"]),
]

GIT_HOSTS = os.environ.get(
    "GIT_HOSTS", "https://github.com,https://gitclone.com/github.com").split(",")

HF_ENDPOINTS = os.environ.get(
    "HF_ENDPOINTS", "https://huggingface.co,https://hf-mirror.com").split(",")

RETRIES = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout=7200):
    log("+ " + " ".join(cmd))
    try:
        return subprocess.run(cmd, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        log("timeout expired")
        return False


def curl_fetch(url, dest, token=None):
    cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
           "--speed-limit", "10240", "--speed-time", "30",
           "-C", "-", "--connect-timeout", "20", "-o", dest, url]
    if token:
        cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
    return run(cmd)


def clone_has_content(dest):
    if not os.path.isdir(dest):
        return False
    return any(entry != ".git" for entry in os.listdir(dest))


def fetch_git_repo(name, org, repo, out_root):
    dest = os.path.join(out_root, name)
    if clone_has_content(dest):
        log(f"skip {name}: already present")
        return True
    for host in GIT_HOSTS:
        for attempt in range(1, RETRIES + 1):
            log(f"{name}: clone {host}/{org}/{repo} attempt {attempt}")
            shutil.rmtree(dest, ignore_errors=True)
            if run(["git", "clone", "--depth", "1", f"{host}/{org}/{repo}.git", dest]) \
                    and clone_has_content(dest):
                return True
            log(f"{name}: clone failed or empty")
        for branch in ("main", "master"):
            url = f"{host}/{org}/{repo}/archive/refs/heads/{branch}.tar.gz"
            log(f"{name}: tarball {url}")
            with tempfile.TemporaryDirectory() as tmp:
                archive = os.path.join(tmp, "repo.tar.gz")
                if not curl_fetch(url, archive):
                    continue
                try:
                    with tarfile.open(archive) as tar:
                        tar.extractall(tmp, filter="data")
                except (tarfile.TarError, EOFError) as exc:
                    log(f"{name}: bad tarball: {exc}")
                    continue
                extracted = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
                if not extracted:
                    continue
                shutil.rmtree(dest, ignore_errors=True)
                shutil.move(os.path.join(tmp, extracted[0]), dest)
                return True
    return False


def fetch_hf_dataset(name, repo, files, out_root, token):
    dest = os.path.join(out_root, name)
    os.makedirs(dest, exist_ok=True)
    ok = True
    for fname in files:
        target = os.path.join(dest, fname)
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            log(f"skip {name}/{fname}: already present")
            continue
        done = False
        for endpoint in HF_ENDPOINTS:
            url = f"{endpoint}/datasets/{repo}/resolve/main/{fname}"
            log(f"{name}: fetch {url}")
            if curl_fetch(url, target + ".part", token):
                os.replace(target + ".part", target)
                size_mb = os.path.getsize(target) / 1e6
                log(f"{name}/{fname}: {size_mb:.1f} MB")
                done = True
                break
        if not done:
            log(f"FAIL {name}/{fname}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/raw-data")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    failures = []

    for name, org, repo in GIT_REPOS:
        if args.only and name not in args.only:
            continue
        t = time.time()
        if not fetch_git_repo(name, org, repo, args.output_dir):
            failures.append(name)
        log(f"{name}: done in {time.time() - t:.0f}s")

    for name, repo, files in HF_DATASETS:
        if args.only and name not in args.only:
            continue
        t = time.time()
        if not fetch_hf_dataset(name, repo, files, args.output_dir, token):
            failures.append(name)
        log(f"{name}: done in {time.time() - t:.0f}s")

    log(f"total elapsed {time.time() - t0:.0f}s")
    for entry in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, entry)
        if run(["du", "-sm", path]):
            pass
    if failures:
        log("FAILED: " + ", ".join(failures))
        sys.exit(1)
    log("ALL OK")


if __name__ == "__main__":
    main()
