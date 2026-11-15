#!/usr/bin/env python
"""Train Phi on Gaia df/d(q,p) from the flow ensemble (flows/*.npz) with/without the Gauss-law flux loss, using the
stable protocol found on the Plummer toy (lr 3e-3, best checkpoint by the unsupervised stationarity loss).
Arms: baseline | lower (M_enc >= scale*M_bar, one-sided) | monopos (mono+pos only, no baryon model).
No truth exists, so evaluation = physical checks (see evaluate()): rho_tot(Sun) via Poisson, Sigma(<z) at R0 vs
literature, M_enc(r) vs the baryon-only mass (a physical Phi must give M_enc >= M_bar)."""
import argparse, glob, json, sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "gauss_flux"))
REPO = next(p for p in [HERE.parents[1] / "deep-potential-master" / "deep-potential-master", HERE.parent / "deep-potential-master" / "deep-potential-master", HERE.parent / "deep-potential-master"] if (p / "scripts").exists())
sys.path.insert(0, str(REPO / "scripts"))
import flow_torch, potential_torch
from gauss_flux import gauss_flux_loss, enclosed_mass, asinh
import baryon_model
L, V, GK = 1.0, 100.0, 4.30091e-6            # kpc, km/s, G in kpc (km/s)^2 / Msun
Gs = GK / (L * V ** 2)                       # G in scaled units: M_enc comes out in Msun
C = np.array([-8.122, 0.0, 0.0208])
RADII = [0.1, 0.2, 0.3, 0.4]

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="baseline", choices=["baseline", "lower", "monopos", "lohi"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--hi-scale", type=float, default=1.3, help="lohi arm: M_upper = hi_scale * M_bar (a PRIOR: local dark matter is a small fraction)")
ap.add_argument("--pool-rmax", type=float, default=0.5, help="sphere centres are stars within this distance of the Sun [kpc]")
ap.add_argument("--w", type=float, default=1.0, help="flux weight"); ap.add_argument("--bar-scale", type=float, default=1.0)
ap.add_argument("--flows", default="flows"); ap.add_argument("--outdir", default="phi_results"); ap.add_argument("--n-pool", type=int, default=300)
a = ap.parse_args()
Path(a.outdir).mkdir(exist_ok=True); tag = f"{a.arm}{('_hi%g' % a.hi_scale) if a.arm == 'lohi' else ''}_bs{a.bar_scale:g}_w{a.w:g}_lr{a.lr:g}_st{a.steps}{('_pr%g' % a.pool_rmax) if a.pool_rmax != 0.5 else ''}_seed{a.seed}"
if (Path(a.outdir) / f"{tag}.json").exists(): raise SystemExit("exists")
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1)

fs = [np.load(f) for f in sorted(glob.glob(f"{a.flows}/flow_*.npz"))]; print(len(fs), "flows")
q = torch.tensor(fs[0]["q"]); p = torch.tensor(fs[0]["p"])
dq = torch.tensor(np.mean([f["dq"] for f in fs], 0)); dp = torch.tensor(np.mean([f["dp"] for f in fs], 0))
ok = torch.isfinite(dq).all(1) & torch.isfinite(dp).all(1)
ok &= (dq.abs().max(1).values < dq[ok].abs().max(1).values.quantile(0.999)) & (dp.abs().max(1).values < dp[ok].abs().max(1).values.quantile(0.999))
q, p, dq, dp = q[ok], p[ok], dq[ok], dp[ok]; sc = dq.abs().mean(); dq, dp = dq / sc, dp / sc  # f scale is arbitrary
print("stars kept", len(q), flush=True)

phi = potential_torch.PhiNN(3, 3, 128)
params = list(phi.parameters())
opt = flow_torch.RAdam(params, lr=a.lr)
# fixed sphere pool near the Sun (spheres must lie inside the d<1 kpc sample): centres = random stars with |q'|<0.5
near = torch.where(q.norm(dim=1) < a.pool_rmax)[0]; pick = near[torch.randperm(len(near))[:a.n_pool]]
cen = q[pick]; rad = torch.tensor(RADII, dtype=torch.float32)[None].expand(len(pick), -1).contiguous()
Mb = torch.tensor(baryon_model.M_bar(cen.numpy() + C, rad.numpy(), a.bar_scale), dtype=torch.float32)
print("pool", len(pick), "median M_bar per radius", Mb.median(0).values.tolist(), flush=True)

def stat(qb, pb, dqb, dpb, lam=2.0):
    qb = qb.detach().clone().requires_grad_(True)
    g, lap = potential_torch.calc_phi_derivatives(phi, qb)
    return torch.mean(asinh((torch.sum(dpb * g - dqb * pb, 1)).abs()) + lam * asinh(torch.clamp(-lap, 0.0)))

hist, best, bs = [], 1e9, None; t0 = time.time()
for i in range(a.steps):
    idx = torch.randint(len(q), (1024,)); opt.zero_grad()
    ls = stat(q[idx], p[idx], dq[idx], dp[idx]); lf = torch.zeros(())
    if a.arm != "baseline":
        j = torch.randperm(len(pick))[:16]
        kw = dict(n_dirs=64, G=Gs, m_scale=1e7)
        if a.arm == "lower": lf, _, _ = gauss_flux_loss(phi, cen[j], rad[j], M_lower=Mb[j], w_mono=0.0, w_pos=0.0, **kw)
        elif a.arm == "lohi": lf, _, _ = gauss_flux_loss(phi, cen[j], rad[j], M_lower=Mb[j], M_upper=a.hi_scale * Mb[j], w_mono=0.0, w_pos=0.0, **kw)
        else: lf, _, _ = gauss_flux_loss(phi, cen[j], rad[j], **kw)
    (ls + a.w * lf).backward(); opt.step(); hist.append(float(ls))
    if (i + 1) % 10 == 0:
        for g in opt.param_groups: g["lr"] = a.lr * 0.9992 ** ((i + 1) // 10)
    if (i + 1) % 50 == 0:
        m = float(np.mean(hist[-50:]))
        if np.isfinite(m) and m < best: best, bs = m, {k: v.clone() for k, v in phi.state_dict().items()}
    if (i + 1) % 500 == 0: print(i + 1, f"stat {np.mean(hist[-50:]):.4f} flux {float(lf):.4f} {time.time()-t0:.0f}s", flush=True)

def evaluate(state):
    phi.load_state_dict(state)
    sun = torch.tensor((np.array([-8.122, 0, 0.0208]) - C)[None], dtype=torch.float32)       # = (0,0,0) in scaled coords
    x = sun.repeat(2, 1).requires_grad_(True); _, lap = potential_torch.calc_phi_derivatives(phi, x)
    rho = float(lap.mean()) * V ** 2 / L ** 2 / (4 * np.pi * GK) / 1e9                               # Msun/pc^3 (Poisson)
    out = dict(rho_sun=rho)
    rr = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.6]]); Mt = enclosed_mass(phi, sun, rr, n_dirs=512, G=Gs)[0].detach().numpy()
    Mbar = baryon_model.M_bar(sun.numpy() + C, rr.numpy())[0]
    out.update(r=rr[0].tolist(), M_enc=Mt.tolist(), M_bar=Mbar.tolist(), M_ratio=(Mt / Mbar).tolist())
    ks = []
    for z in (0.5, 1.0):
        zp = torch.tensor([[0, 0, z], [0, 0, -z]], dtype=torch.float32, requires_grad=True)
        gz = torch.autograd.grad(phi(zp).sum(), zp)[0][:, 2]
        ks.append(float((gz[0] - gz[1]) / 2) * V ** 2 / L / (2 * np.pi * GK) / 1e6)       # Sigma(<z) [Msun/pc^2] from Kz=2 pi G Sigma
    out.update(Sigma_z0p5=ks[0], Sigma_z1p0=ks[1])
    return out
final = {k: v.clone() for k, v in phi.state_dict().items()}
res = dict(tag=tag, cfg=vars(a), best_stat=best, final=evaluate(final), best=evaluate(bs) if bs else None, wall_s=time.time() - t0)
torch.save(dict(best=bs, final=final), Path(a.outdir) / f"{tag}.pt")
json.dump(res, open(Path(a.outdir) / f"{tag}.json", "w"), indent=1)
print(json.dumps(res["best"] if bs else res["final"], indent=1))
