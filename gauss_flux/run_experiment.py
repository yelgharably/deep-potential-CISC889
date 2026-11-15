#!/usr/bin/env python
"""
One experiment task = one trained potential. Designed for Slurm arrays (one task per grid row).

  python run_experiment.py --grid grid.tsv --task-id 17 --outdir results
  python run_experiment.py --smoke --outdir /tmp/smoke          # tiny plumbing check

Outputs (never overwritten; an existing <tag>.json means the task is done and is skipped):
  <outdir>/<tag>.pt     state_dict of the trained PhiNN (written first)
  <outdir>/<tag>.json   config + metrics + enclosed-mass curve + downsampled loss history (written last)
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import ablation as A

ARM_CODES = {"baseline": "baseline", "geom": "flux: mono+pos",
             "lower": "flux: +lower", "lohi": "flux: +lower+upper"}
FIELDS = ["group", "arm", "s", "gamma", "f_b", "delta", "noise", "n_data", "steps", "seed"]


def make_tag(c):
    nd = "all" if c["n_data"] in (None, "", "all") else int(c["n_data"])
    return (f"{c['arm']}_s{float(c['s']):g}_g{float(c['gamma']):g}_fb{float(c['f_b']):g}"
            f"_d{float(c['delta']):g}_n{float(c['noise']):g}_N{nd}_st{int(c['steps'])}_seed{int(c['seed'])}")


def load_cfg(a):
    if a.smoke:
        return dict(group="smoke", arm="lohi", s=1.0, gamma=1.0, f_b=0.3, delta=0.2,
                    noise=0.0, n_data="all", steps=a.smoke_steps, seed=0)
    if a.grid is None or a.task_id is None:
        sys.exit("FATAL: need --grid and --task-id (or --smoke)")
    with open(a.grid, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not (0 <= a.task_id < len(rows)):
        sys.exit(f"FATAL: task-id {a.task_id} outside grid of {len(rows)} rows")
    return rows[a.task_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid"); ap.add_argument("--task-id", type=int)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--smoke-steps", type=int, default=30)
    a = ap.parse_args()

    c = load_cfg(a)
    if c["arm"] not in ARM_CODES:
        sys.exit(f"FATAL: unknown arm {c['arm']!r}")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tag = make_tag(c)
    jpath, wpath = out / f"{tag}.json", out / f"{tag}.pt"
    if jpath.exists():
        print(f"SKIP (already done): {tag}"); return
    if not A.CACHE.exists():
        sys.exit(f"FATAL: data cache {A.CACHE} missing (not regenerating: tasks would race)")

    print(f"task {a.task_id} tag={tag} device={A.DEV} threads={torch.get_num_threads()}", flush=True)
    data = A.build_dataset()
    n_data = None if c["n_data"] in (None, "", "all") else int(c["n_data"])
    t0 = time.time()
    phi, hist = A.train_arm(data, ARM_CODES[c["arm"]], n_steps=int(c["steps"]), seed=int(c["seed"]),
                            gamma=float(c["gamma"]), s=float(c["s"]), f_b=float(c["f_b"]),
                            delta=float(c["delta"]), noise=float(c["noise"]), n_data=n_data,
                            log_every=max(1, int(c["steps"]) // 6))
    wall = time.time() - t0
    ev = A.evaluate(phi, data)

    stat = np.array(hist["stat"]); flux = np.array(hist["flux"])
    ok = bool(np.isfinite(stat).all() and np.isfinite(flux).all()
              and all(math.isfinite(ev[k]) for k in ("M_err", "phi_rms")))
    blk = 10; nb = len(stat) // blk
    res = dict(config={k: c[k] for k in FIELDS}, tag=tag, status="ok" if ok else "NONFINITE",
               M_err=ev["M_err"], M_err_in=ev["M_err_in"], M_err_out=ev["M_err_out"], phi_rms=ev["phi_rms"],
               r=ev["r"].tolist(), M=ev["M"].tolist(), M_true=ev["M_true"].tolist(),
               stat_loss_blk10=stat[:nb * blk].reshape(nb, blk).mean(1).tolist(),
               flux_loss_blk10=flux[:nb * blk].reshape(nb, blk).mean(1).tolist(),
               wall_s=wall, device=str(A.DEV), threads=torch.get_num_threads(), torch=torch.__version__)
    torch.save(phi.state_dict(), str(wpath) + ".tmp"); os.replace(str(wpath) + ".tmp", wpath)
    with open(str(jpath) + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(str(jpath) + ".tmp", jpath)                  # json last = completion marker
    print(f"DONE {tag} status={res['status']} M_err={ev['M_err']:.3f} (in {ev['M_err_in']:.3f} / "
          f"out {ev['M_err_out']:.3f}) phi_rms={ev['phi_rms']:.4f} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
