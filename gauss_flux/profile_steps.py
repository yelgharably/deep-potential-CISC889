#!/usr/bin/env python
"""Where does the time go? Times the first steps individually (warmup) and steady state per component."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import ablation as A
from gauss_flux import enclosed_mass, pick_spheres

print("torch", torch.__version__, "threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
print("cpu count seen:", os.cpu_count(), "affinity:", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a")
print(torch.__config__.parallel_info().strip())

d = A.build_dataset()
phi = A.potential_torch.PhiNN(3, 3, 128).to(A.DEV)
N = d["q"].shape[0]


def timed(fn, n=1):
    t = time.time()
    for _ in range(n): fn()
    return (time.time() - t) / n


def stat():
    idx = torch.randint(N, (1024,))
    l = A.stationarity_loss(phi, d["q"][idx], d["p"][idx], d["dq"][idx], d["dp"][idx]); l.backward()


def flux():
    c, r = pick_spheres(d["q"], 16, 0.2, 1.0, n_radii=4, max_knn_dist=0.5)
    gl, _, _ = A.gauss_flux_loss(phi, c, r, n_dirs=64); gl.backward()


print("-- first 6 steps of the stationarity loss, individually (s):", [round(timed(stat), 3) for _ in range(6)])
print("-- first 3 flux-loss steps, individually (s):", [round(timed(flux), 3) for _ in range(3)])
print("-- steady state (mean of 10):")
print(f"   stationarity step      {timed(stat, 10):.3f} s")
print(f"   flux step (all-in)     {timed(flux, 10):.3f} s")
cfg = dict(centers=torch.zeros(8, 3), radii=torch.linspace(0.3, 5, 8)[None].expand(8, -1).contiguous())
print(f"   enclosed_mass fwd+bwd  {timed(lambda: enclosed_mass(phi, **cfg, n_dirs=64).sum().backward(), 10):.3f} s")
print(f"   pick_spheres           {timed(lambda: pick_spheres(d['q'], 16, 0.2, 1.0, n_radii=4, max_knn_dist=0.5), 10):.3f} s")
for nt in (1, 2, 4):
    torch.set_num_threads(nt)
    print(f"   threads={nt}: stationarity step {timed(stat, 5):.3f} s, flux step {timed(flux, 5):.3f} s")
print("profile done")
