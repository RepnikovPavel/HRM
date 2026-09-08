import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CKPT = os.environ.get("CKPT_DIR", "/data/ckpt")
PORT = int(os.environ.get("PORT", "8377"))
FRONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "front")
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
        ".css": "text/css", ".json": "application/json", ".png": "image/png"}

RUNS = [
    {"name": "HRM 27M (sudoku-1k-server-v2, train exact on train batches)",
     "path": "sudoku-1k-server-v2/metrics.jsonl", "key": "train/exact_accuracy",
     "color": "#1f77b4"},
    {"name": "RRN 27M (rrn27m-sudoku-1k, test exact on 8192 test)",
     "path": "rrn27m-sudoku-1k/metrics.jsonl", "key": "test_exact_accuracy",
     "color": "#d62728"},
]


def load_series(run):
    pts = []
    try:
        with open(os.path.join(CKPT, run["path"])) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run["key"] in row and "time" in row:
                    pts.append((row["time"], row[run["key"]]))
    except OSError:
        return run["name"], run["color"], []
    if not pts:
        return run["name"], run["color"], []
    t0 = pts[0][0]
    series = [[round((t - t0) / 3600.0, 4), round(v * 100.0, 3)] for t, v in pts]
    return run["name"], run["color"], series


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            path = "/front/index.html"
        if path == "/api/series":
            out = {"generated": time.time(),
                   "runs": [dict(name=n, color=c, series=s)
                            for n, c, s in (load_series(r) for r in RUNS)]}
            self._send(200, "application/json", json.dumps(out).encode())
            return
        rel = os.path.normpath(path).lstrip("/")
        full = os.path.join(os.path.dirname(FRONT), rel)
        if full.startswith(os.path.dirname(FRONT)) and os.path.isfile(full):
            ext = os.path.splitext(full)[1]
            with open(full, "rb") as f:
                self._send(200, MIME.get(ext, "application/octet-stream"), f.read())
            return
        self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    print(f"serving on :{PORT}, ckpt dir {CKPT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
