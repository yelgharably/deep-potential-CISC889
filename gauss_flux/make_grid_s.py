#!/usr/bin/env python
"""Stabilised-protocol grid (run_experiment_s.py -> results_s/). Separates three candidate fixes for the
optimizer collapse seen with the repo default lr=5e-2: lower lr, gradient clipping, best-checkpoint selection."""
import itertools
import run_experiment_s as S

BASE = dict(s=1.0, gamma=1.0, f_b=0.3, delta=0.2, noise=0.0, n_data="all", steps=3000, n_dirs=64, lr=5e-2, clip=0.0)
ARMS = ["baseline", "geom", "lower", "lohi"]
rows = []
def add(group, arms, seeds, **kw):
    for arm, sd in itertools.product(arms, range(seeds)):
        rows.append({**BASE, **kw, "group": group, "arm": arm, "seed": sd})

for lr in [5e-4, 1e-3, 3e-3, 1e-2, 2e-2]: add("S1_lr", ARMS, 10, lr=lr)
for lr in [1e-2, 5e-2]:                    add("S2_clip", ARMS, 5, lr=lr, clip=1.0)
for lr in [3e-3, 1e-2]:                    add("S3_long", ["baseline", "lohi"], 5, lr=lr, steps=12000)
add("S4_orig_lr_bestckpt", ARMS, 10, lr=5e-2)

seen, uniq = set(), []
for c in rows:
    t = S.make_tag(c)
    if t not in seen: seen.add(t); uniq.append(c)
with open("grid_s.tsv", "w", newline="\n") as fh:
    fh.write("\t".join(S.FIELDS) + "\n")
    for c in uniq: fh.write("\t".join(str(c[k]) for k in S.FIELDS) + "\n")
from collections import Counter
print(f"grid_s.tsv: {len(uniq)} tasks ({len(rows)-len(uniq)} duplicates)"); print(dict(Counter(c['group'] for c in uniq)))
