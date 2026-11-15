#!/usr/bin/env python
"""Train one normalizing flow on the scaled Gaia 6D sample, then (same task) evaluate f and df/d(q,p) on a fixed
set of training stars and save them. One flow per task (array); gaia_phi.py averages the ensemble.
Scaling: q' = (q - c)/L, p' = p/V with L = 1 kpc, V = 100 km/s (p is NOT recentred: the Boltzmann equation needs
the inertial-frame velocity); the flow additionally standardises its input internally (a constant rescaling of f,
harmless because the stationarity equation is linear in f)."""
import argparse, sys, time, numpy as np, torch
from pathlib import Path
REPO = next(p for p in [Path(__file__).resolve().parents[2] / "deep-potential-master" / "deep-potential-master", Path(__file__).resolve().parents[1] / "deep-potential-master" / "deep-potential-master",
                        Path(__file__).resolve().parents[1] / "deep-potential-master"] if (p / "scripts").exists())
sys.path.insert(0, str(REPO / "scripts"))
import flow_torch, potential_torch
L, V = 1.0, 100.0
C = np.array([-8.122, 0.0, 0.0208])
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="gaia_6d.npz"); ap.add_argument("--flow-id", type=int, default=0)
ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--units", type=int, default=16)
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--n-eval", type=int, default=100000)
ap.add_argument("--max-n", type=int, default=0); ap.add_argument("--outdir", default="flows")
a = ap.parse_args()
Path(a.outdir).mkdir(exist_ok=True); out = Path(a.outdir) / f"flow_{a.flow_id:02d}.npz"
if out.exists(): raise SystemExit(f"{out} exists; not overwriting")
d = np.load(a.data); perm = np.random.default_rng(0).permutation(len(d["q"]))  # fixed order, same for every flow
q = (d["q"][perm] - C) / L; p = d["p"][perm] / V
eta = np.concatenate([q, p], 1).astype("f4")
if a.max_n: eta = eta[:a.max_n]
mu, sd = eta.mean(0), eta.std(0)
torch.manual_seed(a.flow_id); np.random.seed(a.flow_id)
torch.set_num_threads(1)
flow = flow_torch.NormalizingFlow(6, a.units)
z = torch.tensor((eta - mu) / sd)
t0 = time.time()
hist = flow_torch.train_flow(flow, z, n_epochs=a.epochs, batch_size=a.batch, callback=lambda *x: None)
print(f"trained {a.epochs} epochs on {len(z)} stars in {time.time()-t0:.0f}s; loss {np.mean(hist[:20]):.3f} -> {np.mean(hist[-20:]):.3f}", flush=True)
# f and gradients at the first n_eval stars (the sample is in random order), in scaled (q', p') coordinates
sel = torch.tensor(eta[:a.n_eval]); qg = sel[:, :3].clone().requires_grad_(True); pg = sel[:, 3:].clone().requires_grad_(True)
mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
def f_star(q_, p_):
    zz = (torch.cat([q_, p_], 1) - mu_t) / sd_t
    _, lp, ld = flow(zz)
    return torch.exp(lp + ld)
f, dq, dp = [], [], []
for i in range(0, len(sel), 4096):
    qq = qg[i:i+4096].detach().clone().requires_grad_(True); pp = pg[i:i+4096].detach().clone().requires_grad_(True)
    r = potential_torch.calc_df_deta(f_star, qq, pp); f.append(r[0].detach()); dq.append(r[1].detach()); dp.append(r[2].detach())
f, dq, dp = torch.cat(f), torch.cat(dq), torch.cat(dp)
print("finite frac", float(torch.isfinite(dq).all(1).float().mean()), "| f range", float(f.min()), float(f.max()), flush=True)
np.savez(out, q=sel[:, :3].numpy(), p=sel[:, 3:].numpy(), f=f.numpy(), dq=dq.numpy(), dp=dp.numpy(), loss=np.array(hist), epochs=a.epochs)
