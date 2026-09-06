import json
import os
import sys
import time

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2, summary_pb2


def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}/{k}", v, out)
    else:
        out[prefix] = float(obj)


def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    w = EventFileWriter(out_dir)
    pos = 0
    while True:
        try:
            with open(src) as f:
                f.seek(pos)
                lines = f.readlines()
                pos = f.tell()
        except FileNotFoundError:
            lines = []
        for line in lines:
            row = json.loads(line)
            step = row.pop("step", 0)
            row.pop("time", None)
            flat = {}
            flatten("", row, flat)
            for k, v in flat.items():
                s = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=k.strip("/"), simple_value=v)])
                ev = event_pb2.Event(step=step, summary=s, wall_time=time.time())
                w.add_event(ev)
        w.flush()
        time.sleep(10)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
