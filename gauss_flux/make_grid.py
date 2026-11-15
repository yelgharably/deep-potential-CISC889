#!/usr/bin/env python
"""Write grid.tsv (one row per task). Duplicate configs across groups are dropped (first group wins),
so e.g. robustness s=1.0 and gamma=1.0 reuse the main-ablation runs. Analysis filters by config, not group.

  python make_grid.py --steps 3000 --out grid.tsv
"""
import argparse, itertools
from run_experiment import make_tag, FIELDS

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--out", default="grid.tsv")
ap.add_argument("--seeds-main", type=int, default=10)
ap.add_argument("--seeds-sweep", type=int, default=5)
ap.add_argument("--seeds-small", type=int, default=3)
a = ap.parse_args()

BASE = dict(s=1.0, gamma=1.0, f_b=0.3, delta=0.2, noise=0.0, n_data="all", steps=a.steps)
rows = []


def add(group, arm, seeds, **kw):
    for sd in range(seeds):
        c = {**BASE, **kw, "group": group, "arm": arm, "seed": sd}
        rows.append(c)


ARMS4 = ["baseline", "geom", "lower", "lohi"]
# A: main ablation
for arm in ARMS4: add("A_ablation", arm, a.seeds_main)
# B: mis-specified external mass (lower / lohi bracket scaled by s)
for arm, s in itertools.product(["lower", "lohi"], [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]):
    add("B_robust", arm, a.seeds_sweep, s=s)
# C: weight gamma of the flux loss
for arm, g in itertools.product(["geom", "lohi"], [0.1, 0.3, 1.0, 3.0, 10.0]):
    add("C_gamma", arm, a.seeds_sweep, gamma=g)
# D: bracket shape: lower-bound fraction f_b, upper-bound slack delta
for fb in [0.1, 0.3, 0.6, 0.9]: add("D_fb", "lower", a.seeds_small, f_b=fb)
for d in [0.05, 0.2, 0.5, 1.0]: add("D_delta", "lohi", a.seeds_small, delta=d)
# E: training length (is the baseline just under-trained?)
for arm, st in itertools.product(["baseline", "geom", "lohi"], [1000, a.steps * 2]):
    add("E_steps", arm, a.seeds_small, steps=st)
# F: noise on df/deta (repo notebook shows noise biases the potential shallow)
for arm, nz in itertools.product(["baseline", "geom", "lohi"], [0.001, 0.004, 0.016]):
    add("F_noise", arm, a.seeds_small, noise=nz)
# G: data availability
for arm, nd in itertools.product(["baseline", "geom", "lohi"], [2048, 8192, 16384]):
    add("G_ndata", arm, a.seeds_small, n_data=nd)

seen, uniq = set(), []
for c in rows:
    t = make_tag(c)
    if t not in seen:
        seen.add(t); uniq.append(c)

with open(a.out, "w", newline="\n") as fh:
    fh.write("\t".join(FIELDS) + "\n")
    for c in uniq:
        fh.write("\t".join(str(c[k]) for k in FIELDS) + "\n")

from collections import Counter
print(f"{len(uniq)} tasks ({len(rows) - len(uniq)} duplicates dropped) -> {a.out}")
for g, n in Counter(c["group"] for c in uniq).items(): print(f"  {g:10s} {n}")
