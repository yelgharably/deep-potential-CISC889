#!/usr/bin/env python
"""Second-wave grids (separate files, so the running main array is never disturbed).

grid_b.tsv  (run_experiment.py, results/)    : 20 extra seeds for the main ablation + 12000-step check
grid_x.tsv  (run_experiment_x.py, results_x/): term-level ablation (mono-only, pos-only) + surface density n_dirs
"""
import itertools
from run_experiment import make_tag, FIELDS
import run_experiment_x as X

BASE = dict(s=1.0, gamma=1.0, f_b=0.3, delta=0.2, noise=0.0, n_data="all", steps=3000)


def write(path, rows, fields, tagfn):
    seen, uniq = set(), []
    for c in rows:
        t = tagfn(c)
        if t not in seen: seen.add(t); uniq.append(c)
    with open(path, "w", newline="\n") as fh:
        fh.write("\t".join(fields) + "\n")
        for c in uniq: fh.write("\t".join(str(c[k]) for k in fields) + "\n")
    print(f"{path}: {len(uniq)} tasks")


# ---- grid_b: seeds 10..29 for the 4 main arms (pairs with seeds 0-9 of the main grid), and 12000 steps
b = []
for arm in ["baseline", "geom", "lower", "lohi"]:
    for sd in range(10, 30): b.append({**BASE, "group": "B_moreseeds", "arm": arm, "seed": sd})
for arm in ["baseline", "geom", "lohi"]:
    for sd in range(5): b.append({**BASE, "group": "B_steps12000", "arm": arm, "seed": sd, "steps": 12000})
write("grid_b.tsv", b, FIELDS, make_tag)

# ---- grid_x: which term matters, and does denser surface sampling matter?
x = []
for arm in ["mono", "pos"]:
    for sd in range(10): x.append({**BASE, "group": "X_terms", "arm": arm, "seed": sd, "n_dirs": 64})
for arm, nd in itertools.product(["geom", "lohi"], [16, 256]):
    for sd in range(5): x.append({**BASE, "group": "X_ndirs", "arm": arm, "seed": sd, "n_dirs": nd})
write("grid_x.tsv", x, X.FIELDS, X.make_tag)
