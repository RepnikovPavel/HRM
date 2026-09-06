import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[0]


def col(sub):
    return next(i for i, h in enumerate(hdr) if sub in h)


i_name = col("Kernel Name")
i_dur = col("gpu__time_duration.sum")
i_sm = col("sm__throughput.avg.pct")
i_mem = col("gpu__compute_memory_throughput.avg.pct")

agg = {}
for r in rows[2:]:
    if len(r) <= i_mem or not r[i_dur]:
        continue
    a = agg.setdefault(r[i_name], [0, 0.0, 0.0, 0.0])
    a[0] += 1
    a[1] += float(r[i_dur])
    a[2] += float(r[i_sm] or 0)
    a[3] += float(r[i_mem] or 0)

tot = sum(v[1] for v in agg.values())
print(f"total kernel time {tot / 1e6:.1f} ms, {sum(v[0] for v in agg.values())} launches, {len(agg)} unique kernels")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1])[:18]:
    print(f"{v[1] / 1e3:9.1f}us | {v[0]:4d}x | SM {v[2] / v[0]:5.1f}% | MEM {v[3] / v[0]:5.1f}% | {k[:75]}")
