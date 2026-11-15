#!/usr/bin/env python
"""Noise / data-scarcity grid under the stable protocol (run_experiment_s.py -> results_n/)."""
import itertools
import run_experiment_s as S
BASE = dict(s=1.0, gamma=1.0, f_b=0.3, delta=0.2, noise=0.0, n_data="all", steps=3000, n_dirs=64, lr=3e-3, clip=0.0)
ARMS = ["baseline", "geom", "lower", "lohi"]
rows = []
for lr in [3e-3, 1e-2]:
    for arm, sd in itertools.product(ARMS, range(10)):
        for nz in [0.004, 0.016, 0.05]: rows.append({**BASE, "lr": lr, "noise": nz, "group": "N1_noise", "arm": arm, "seed": sd})
        for nd in [1024, 2048, 4096, 8192]: rows.append({**BASE, "lr": lr, "n_data": nd, "group": "N2_ndata", "arm": arm, "seed": sd})
seen, uniq = set(), []
for c in rows:
    t = S.make_tag(c)
    if t not in seen: seen.add(t); uniq.append(c)
with open("grid_n.tsv", "w", newline="\n") as fh:
    fh.write("\t".join(S.FIELDS) + "\n")
    for c in uniq: fh.write("\t".join(str(c[k]) for k in S.FIELDS) + "\n")
print(len(uniq))
