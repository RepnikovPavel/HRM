# Task 1 (data layer) — cloud cost accounting

Goal: reproduce the "download raw data" stage in a cloud VM and estimate
its cost. Formula:

```
cost = egress_GB * egress_price + cpu_hours * vcpu_price
```

Inputs measured on 2026-09-05 (see dataset/docs/data_layer.md):

- payload downloaded: 0.784 GB (ARC repos 18 MB + sudoku 762 MB + maze 4 MB)
- wall time: 102 s on 48-core server; the stage is network-bound and
  single-stream, so 1 vCPU suffices in the cloud
- rsync server->dev (813 MB over LAN) is free in-cloud if VM and storage
  are colocated; storing 0.784 GB on a cloud disk adds
  `0.784 * disk_GB_month_price` per month

Estimate with AWS eu-central-1 on-demand prices:

```
egress (internet -> VM): free (ingress)
compute: 102 s * $0.048/vCPU-h (t3.large ~2 vCPU $0.096/h) = $0.0027
storage: 0.784 GB * $0.08/GB-month (gp3)                    = $0.063/month
```

Total per full re-download: **≈ $0.003** compute + **≈ $0.06/GB·month**
storage. Data transfer out to a dev machine over the internet would add
`0.784 GB * $0.09/GB ≈ $0.07`; avoided here by LAN rsync.
