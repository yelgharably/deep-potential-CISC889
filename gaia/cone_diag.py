#!/usr/bin/env python
"""Cone-sector Gauss test (apex at the Sun, axis u, half-angle theta, out to radius r2). Closed surface = spherical cap
at r2 (normal radial) + conical side (normal cos(th)*e_perp - sin(th)*u). M_enc = flux/(4 pi G). Compared with the baryon
mass in the same cone (rho_bar integrated in spherical coordinates) and with an 'expected' value that adds a uniform local
dark-matter density rho_DM = 0.01 Msun/pc^3 (the expected M/M_bar for a physical potential). Same checkpoints as cyl_diag."""
import glob, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaia_phi_defs as D, baryon_model as B
from cyl_diag import grad
RHO_DM = 0.01e9  # Msun/kpc^3
AXES = {"up(+z)": [0, 0, 1], "down(-z)": [0, 0, -1], "GC(+x)": [1, 0, 0], "anti-GC(-x)": [-1, 0, 0], "rot(+y)": [0, 1, 0], "anti-rot(-y)": [0, -1, 0]}
TH, RS = np.radians(30), [0.5, 0.9]

def frame(u):
    u = np.array(u, float); a = np.array([1., 0, 0]) if abs(u[0]) < .9 else np.array([0, 1., 0]); e1 = np.cross(u, a); e1 /= np.linalg.norm(e1); return u, e1, np.cross(u, e1)

def cone_flux_mass(phi, u, r2, th=TH, na=24, nphi=48, ns=40):
    u, e1, e2 = frame(u); ph = (np.arange(nphi) + .5) / nphi * 2 * np.pi; al = (np.arange(na) + .5) / na * th
    A, P = np.meshgrid(al, ph, indexing="ij"); ehat = np.cos(P)[..., None] * e1 + np.sin(P)[..., None] * e2
    nrm = np.cos(A)[..., None] * u + np.sin(A)[..., None] * ehat                      # unit radial vectors on the cap
    dA = r2 ** 2 * np.sin(A) * (th / na) * (2 * np.pi / nphi)
    g = grad(phi, torch.tensor((r2 * nrm).reshape(-1, 3), dtype=torch.float32)).detach().numpy()
    cap = ((g * nrm.reshape(-1, 3)).sum(1) * dA.reshape(-1)).sum()
    s = (np.arange(ns) + .5) / ns * r2; S, P2 = np.meshgrid(s, ph, indexing="ij"); e2h = np.cos(P2)[..., None] * e1 + np.sin(P2)[..., None] * e2
    pts = S[..., None] * (np.cos(th) * u + np.sin(th) * e2h); n2 = np.cos(th) * e2h - np.sin(th) * u
    g2 = grad(phi, torch.tensor(pts.reshape(-1, 3), dtype=torch.float32)).detach().numpy()
    side = ((g2 * n2.reshape(-1, 3)).sum(1) * (S * np.sin(th) * (r2 / ns) * (2 * np.pi / nphi)).reshape(-1)).sum()
    return (cap + side) * D.V ** 2 * D.L / (4 * np.pi * D.GK)

def bar_and_dm(u, r2, th=TH, ns=60, na=24, nphi=32):
    u, e1, e2 = frame(u); s = (np.arange(ns) + .5) / ns * r2; al = (np.arange(na) + .5) / na * th; ph = (np.arange(nphi) + .5) / nphi * 2 * np.pi
    S, A, P = np.meshgrid(s, al, ph, indexing="ij")
    d = np.cos(A)[..., None] * u + np.sin(A)[..., None] * (np.cos(P)[..., None] * e1 + np.sin(P)[..., None] * e2)
    x = D.C + S[..., None] * d; dV = S ** 2 * np.sin(A) * (r2 / ns) * (th / na) * (2 * np.pi / nphi)
    return (B.rho_bar(x[..., 0], x[..., 1], x[..., 2]) * dV).sum(), RHO_DM * dV.sum()

if __name__ == "__main__":
    from collections import defaultdict
    names = [(k, r) for k in AXES for r in RS]; base = {(k, r): bar_and_dm(AXES[k], r) for k, r in names}
    out = defaultdict(list)
    for f in sorted(glob.glob("phi_w/*.pt")):
        meta = json.load(open(f.replace(".pt", ".json")))["cfg"]; key = meta["arm"] + (f" <={meta['hi_scale']:g}" if meta["arm"] == "lohi" else (f" w={meta['w']:g}" if meta["arm"] == "lower" else ""))
        phi = D.make_phi(); phi.load_state_dict(torch.load(f)["best"])
        out[key].append([cone_flux_mass(phi, AXES[k], r) / base[(k, r)][0] for k, r in names])
    print("cone: half-angle 30deg, apex at Sun;  columns = (axis, r2[kpc])")
    print("%-14s %5s %8s %9s" % ("axis", "r2", "M_bar", "expected(M_bar+DM)/M_bar"))
    for (k, r) in names: b, dm = base[(k, r)]; print("%-14s %5.1f %8.2e %9.2f" % (k, r, b, (b + dm) / b))
    for key, v in out.items():
        v = np.array(v); print(f"\n{key} (n={len(v)}) M_cone/M_bar_cone, mean over seeds:")
        for i, (k, r) in enumerate(names): print("   %-14s r2=%.1f  %.2f ± %.2f" % (k, r, v[:, i].mean(), v[:, i].std()))
    json.dump({kk: np.array(v).tolist() for kk, v in out.items()} | {"names": [[k, r] for k, r in names], "base": [list(base[n]) for n in names]}, open("cone_diag.json", "w"), indent=1)
