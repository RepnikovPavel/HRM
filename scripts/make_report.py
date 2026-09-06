import glob
import json
import os
import sys
import time

import yaml

PAPER = {
    "sudoku-extreme-1k-aug-1000": {"paper": 55.0, "ref": "Fig.1 (Sudoku-Extreme 9x9, 1000 examples)"},
    "maze-30x30-hard-1k": {"paper": 74.5, "ref": "Fig.1 (Maze-Hard 30x30, 1000 examples)"},
    "sudoku-extreme-full": {"paper": 99.5, "ref": "Fig.2 (Sudoku-Extreme-Full, near-perfect)"},
}

OPT = [
    ("SwiGLU MLP forward", "12.70 ms", "8.90 ms", "1.43x"),
    ("SwiGLU MLP fwd+bwd", "39.84 ms", "34.31 ms", "1.16x"),
    ("Attention fwd (seq 82)", "0.959 ms", "0.915 ms", "DRAM roofline 95%"),
    ("Attention bwd (kernels)", "8.20 ms", "2.96 ms", "2.77x"),
    ("linear+residual+rmsnorm chain", "5.49 ms", "2.12 ms", "2.59x"),
    ("qkv GEMM [M,512,1536]", "3.09 ms (46.0 TF)", "2.92 ms (48.7 TF)", "1.06x"),
]

E2E = [
    ("throughput, samples/s (gbs 2176)", "1697", "1741"),
    ("train step time (gbs 2176)", "1.282 s", "1.250 s"),
    ("compute efficiency (of 52.6 TFLOPS mma ceiling)", "73.8%", "76.4%"),
    ("peak VRAM at gbs 2176", "15689 MiB", "10253 MiB"),
    ("max batch without OOM", "2176", "3456 (99.3% usable VRAM)"),
    ("throughput at max batch", "OOM", "1762 samples/s @ gbs 3456"),
]

CSS = """
body { font-family: sans-serif; margin: 24px; }
table { border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #bbb; padding: 4px 10px; text-align: right; }
th { background: #eee; }
td.l, th.l { text-align: left; }
.ok { color: #0a0; font-weight: bold; }
.bad { color: #c00; font-weight: bold; }
pre { background: #f6f6f6; padding: 8px; overflow-x: auto; }
h2 { margin-top: 1.5em; }
"""


def all_evals(metrics_file):
    out = []
    with open(metrics_file) as f:
        for line in f:
            row = json.loads(line)
            if "eval" in row:
                out.append(row)
    return out


def last_eval(metrics_file):
    evs = all_evals(metrics_file)
    return evs[-1] if evs else None


def fmt_pct(x):
    return f"{100 * x:.1f}" if x is not None else "-"


def main(ckpt_roots, out_html):
    runs = []
    for root in ckpt_roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*/"))):
            metrics_file = os.path.join(run_dir, "metrics.jsonl")
            config_file = os.path.join(run_dir, "all_config.yaml")
            if not os.path.isfile(metrics_file):
                continue
            run_name = os.path.basename(run_dir.rstrip("/"))
            if run_name.startswith(("bench-", "smoke", "ncu", "prof")):
                continue
            last = last_eval(metrics_file)
            cfg = {}
            if os.path.isfile(config_file):
                with open(config_file) as f:
                    cfg = yaml.safe_load(f)
            dataset = os.path.basename(cfg.get("data_path", "").rstrip("/"))
            full_path = os.path.join(run_dir, "eval_full.json")
            full = None
            if os.path.isfile(full_path):
                with open(full_path) as f:
                    full = json.load(f)
            runs.append({
                "run": run_name,
                "host": os.path.basename(root.rstrip("/")),
                "dataset": dataset,
                "steps": cfg.get("epochs"), "gbs": cfg.get("global_batch_size"),
                "last": last,
                "trajectory": [(e["step"], e["eval"]["all"].get("exact_accuracy")) for e in all_evals(metrics_file)],
                "full": full,
                "mtime": os.path.getmtime(metrics_file),
            })

    rows = []
    seen = {}
    for r in sorted(runs, key=lambda r: (r["full"] is not None, r["mtime"])):
        seen[r["run"]] = r   # same run synced to several hosts: keep the fullest/newest
    for r in seen.values():
        if r["full"]:
            acc = r["full"]["exact_accuracy"]
            acc_note = " (full test)"
        else:
            acc = (r["last"] or {}).get("eval", {}).get("all", {}).get("exact_accuracy")
            acc_note = " (20k subset)" if acc is not None else ""
        paper = PAPER.get(r["dataset"], {})
        pacc = paper.get("paper")
        verdict = ""
        if acc is not None and pacc is not None:
            ok = 100 * acc >= pacc - 5.0
            verdict = f'<span class="{"ok" if ok else "bad"}">{"MATCH" if ok else "BELOW"}</span>'
        rows.append(f"<tr><td class='l'>{r['dataset']}</td><td class='l'>{r['host']}</td>"
                    f"<td class='l'>{r['run']}</td><td>{r['gbs']}</td>"
                    f"<td>{(r['last'] or {}).get('step', '-')}</td>"
                    f"<td>{fmt_pct(acc)}%{acc_note}</td><td>{pacc}%</td><td>{verdict}</td>"
                    f"<td>{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['mtime']))}</td></tr>")

    traj = []
    for r in runs:
        if not r["trajectory"]:
            continue
        pts = " ".join(f"{s}:{fmt_pct(a)}" for s, a in r["trajectory"])
        traj.append(f"<tr><td class='l'>{r['run']}</td><td class='l' style='font-family:monospace'>{pts}</td></tr>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HRM reproduction report</title>
<style>{CSS}</style></head><body>
<h1>HRM reproduction report</h1>
<p>Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/make_report.py.
Metric: exact_accuracy on the full test set (single inference pass).</p>
<h2>Authors' targets (paper/document.md)</h2>
<table><tr><th class='l'>Dataset</th><th>Paper acc %</th><th class='l'>Source</th></tr>
{"".join(f"<tr><td class='l'>{k}</td><td>{v['paper']}</td><td class='l'>{v['ref']}</td></tr>" for k, v in PAPER.items())}
</table>
<h2>Optimization results (hrmfast, measured 2026-09-05/06, server 2x RTX 5060 Ti)</h2>
<p>author = original upstream code path (cuBLAS/SDPA); hrmfast = custom mma.sync PTX kernels.
Isolated ops at M=90304 (batch 1088 x seq 82, bf16).</p>
<table><tr><th class='l'>Op</th><th>author (before)</th><th>hrmfast (after)</th><th>ratio</th></tr>
{"".join(f"<tr><td class='l'>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in OPT)}
</table>
<table><tr><th class='l'>Training metric</th><th>author (before)</th><th>hrmfast (after)</th></tr>
{"".join(f"<tr><td class='l'>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in E2E)}
</table>
<h2>Our runs</h2>
<table><tr><th class='l'>Dataset</th><th class='l'>Machine</th><th class='l'>Run</th>
<th>Batch</th><th>Step</th><th>Our acc</th><th>Paper</th><th>Verdict</th><th>Updated</th></tr>
{"".join(rows)}
</table>
<h2>Training trajectories (in-training eval, 20k subset; step:exact_accuracy%)</h2>
<table>{"".join(traj)}</table>
<p>Verdict MATCH = within 5 pp of the paper target (paper std on Sudoku-Extreme-1k is
~2 pp per issue #12; evaluate.py is 1-shot, no majority voting).
"full test" = final evaluate.py over all 422786 test examples.</p>
</body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html} with {len(runs)} runs")


if __name__ == "__main__":
    roots = sys.argv[1:-1]
    main(roots, sys.argv[-1])
