import argparse
import http.server
import os


class Handler(http.server.SimpleHTTPRequestHandler):
    viewer_root = "/viewer"
    data_root = "/data"

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path == "/":
            path = "/viewer/"
        if path.startswith("/viewer"):
            rel = path[len("/viewer"):]
            root = self.viewer_root
        elif path.startswith("/data"):
            rel = path[len("/data"):]
            root = self.data_root
        else:
            rel = path
            root = self.data_root
        rel = os.path.normpath(rel).lstrip("/")
        if rel.startswith(".."):
            return os.path.join(root, "nonexistent")
        return os.path.join(root, rel)

    def log_message(self, fmt, *args):
        print(f"[http] {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
