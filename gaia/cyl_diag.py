#!/usr/bin/env python
"""Cylinder Gauss test, free of the thin-slab assumption. For a cylinder (radius Rc, half-height H) on the Sun's
vertical axis: M_enc = (1/4 pi G) * closed-surface integral of grad(Phi).n dA, by quadrature of the two caps and the side.
Compared with the baryon mass in the same cylinder (numerical integral of the McMillan density). Phi comes from a saved
checkpoint (phi_w/*.pt)."""
import glob, json, sys, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaia_phi_defs as D, baryon_model as B
CYL = [(0.3, 0.5), (0.3, 1.0), (0.5, 0.5), (0.5, 1.0)]

def grad(phi, x):
    x = x.clone().requires_grad_(True); return torch.autograd.grad(phi(x).sum(), x)[0]

def enclosed(phi, Rc, H, nr=40, nphi=48, nz=60):
    r = (np.arange(nr) + .5) / nr * Rc; ph = (np.arange(nphi) + .5) / nphi * 2 * np.pi; z = (np.arange(nz) + .5) / nz * 2 * H - H
    # caps: normal +-z, dA = r dr dphi
    R, P = np.meshgrid(r, ph, indexing="ij"); pts = lambda zz: torch.tensor(np.stack([R * np.cos(P), R * np.sin(P), np.full_like(R, zz)], -1).reshape(-1, 3), dtype=torch.float32)
    w = (R * (Rc / nr) * (2 * np.pi / nphi)).reshape(-1)
    top = (grad(phi, pts(H))[:, 2].detach().numpy() * w).sum(); bot = (-grad(phi, pts(-H))[:, 2].detach().numpy() * w).sum()
    Z, P2 = np.meshgrid(z, ph, indexing="ij"); xs = np.stack([Rc * np.cos(P2), Rc * np.sin(P2), Z], -1).reshape(-1, 3)
    g = grad(phi, torch.tensor(xs, dtype=torch.float32)).detach().numpy(); n = np.stack([np.cos(P2), np.sin(P2), 0 * P2], -1).reshape(-1, 3)
    side = ((g * n).sum(1) * (Rc * (2 * np.pi / nphi)) * (2 * H / nz)).sum()
    return (top + bot + side) * D.V ** 2 * D.L / (4 * np.pi * D.GK)   # Msun (flux in scaled units -> physical)

def bar_mass(Rc, H, n=60):
    r = (np.arange(n) + .5) / n * Rc; z = (np.arange(2 * n) + .5) / (2 * n) * 2 * H - H; ph = (np.arange(32) + .5) / 32 * 2 * np.pi
    R, Z, P = np.meshgrid(r, z, ph, indexing="ij")
    rho = B.rho_bar(D.C[0] + R * np.cos(P), R * np.sin(P), D.C[2] + Z)
    return (rho * R).sum() * (Rc / n) * (2 * H / (2 * n)) * (2 * np.pi / 32)

if __name__ == "__main__":
    from collections import defaultdict
    out = defaultdict(list)
    for f in sorted(glob.glob("phi_w/*.pt")):
        meta = json.load(open(f.replace(".pt", ".json")))["cfg"]; key = meta["arm"] + (f" ≤{meta['hi_scale']:g}" if meta["arm"] == "lohi" else (f" w={meta['w']:g}" if meta["arm"] == "lower" else ""))
        phi = D.make_phi(); phi.load_state_dict(torch.load(f)["best"])
        out[key].append([enclosed(phi, Rc, H) for Rc, H in CYL])
    Mb = [bar_mass(Rc, H) for Rc, H in CYL]
    print("cylinders (Rc,H) [kpc]:", CYL); print("baryon mass [1e7 Msun]:", np.round(np.array(Mb) / 1e7, 2))
    res = {}
    for k, v in out.items():
        v = np.array(v) / np.array(Mb); res[k] = v.tolist()
        print(f"{k:12s} n={len(v)}  M_cyl / M_bar_cyl mean:", v.mean(0).round(2), " sd:", v.std(0).round(2))
    json.dump(dict(cyl=CYL, M_bar=Mb, ratios=res), open("cyl_diag.json", "w"), indent=1)
