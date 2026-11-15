"""
Plummer-sphere ablation + robustness experiments for the Gauss-law flux loss.
Helpers imported by gauss_flux_ablation.ipynb.  Re-uses the pretrained normalizing-flow
ensemble in deep-potential-master/plummer_flow and the repo's PhiNN / RAdam.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _find_repo():
    """Locate deep-potential-master/ (works for the local layout and the Expanse layout)."""
    cands = [os.environ.get("CISC889_REPO", ""),
             HERE.parent / "deep-potential-master" / "deep-potential-master",
             HERE.parent.parent / "deep-potential-master" / "deep-potential-master",
             HERE.parent / "deep-potential-master"]
    for c in cands:
        if c and (Path(c) / "scripts" / "potential_torch.py").exists():
            return Path(c)
    raise FileNotFoundError(f"deep-potential repo not found; tried {cands}")


REPO = _find_repo()
sys.path[:0] = [str(REPO / "scripts"), str(HERE)]
if os.environ.get("SLURM_CPUS_PER_TASK"):                   # never leave torch at its default on a shared node
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))

import scipy.integrate                                              # noqa: E402
if not hasattr(scipy.integrate, "cumtrapz"):                         # removed in SciPy >= 1.14
    scipy.integrate.cumtrapz = scipy.integrate.cumulative_trapezoid
import potential_torch, flow_torch, toy_systems                     # noqa: E402
from gauss_flux import (gauss_flux_loss, enclosed_mass, pick_spheres,  # noqa: E402
                        plummer_phi, plummer_enclosed_mass, plummer_bounds, asinh)

DEV = torch.device(os.environ.get("CISC889_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
CACHE = HERE / "plummer_data_cache.pt"

ARMS = {                                    # which flux terms are switched on
    "baseline":      dict(flux=False),
    "flux: mono+pos": dict(flux=True, lo=False, hi=False),
    "flux: +lower":  dict(flux=True, lo=True,  hi=False),
    "flux: +lower+upper": dict(flux=True, lo=True, hi=True),
}


# ------------------------------------------------------------------ data
def build_dataset(n_points=32768, n_flows=16, r_clip=8.0, seed=0, use_cache=True):
    """Sample Plummer phase-space points and estimate df/dq, df/dp as the mean over an
    ensemble of pretrained normalizing flows (as in the repo notebook)."""
    if use_cache and CACHE.exists():
        return {k: v.to(DEV) for k, v in torch.load(CACHE).items()}
    np.random.seed(seed); torch.manual_seed(seed)
    ps = toy_systems.PlummerSphere()
    q, p = ps.sample_df(int(n_points * 1.3))
    keep = np.sum(q ** 2, axis=1) < r_clip ** 2
    q, p = q[keep][:n_points], p[keep][:n_points]
    # flow_torch hardcodes device="cpu", so estimate the flow gradients on CPU (one-off, cached)
    q = torch.tensor(q.astype("f4"), requires_grad=True)
    p = torch.tensor(p.astype("f4"), requires_grad=True)
    dq = torch.zeros_like(q); dp = torch.zeros_like(p)
    for i in range(n_flows):
        flow = flow_torch.NormalizingFlow(6, 4)
        flow.load_state_dict(torch.load(REPO / "plummer_flow" / f"plummer_flow_{i:02d}.pth",
                                        map_location="cpu"))
        def f_star(q_, p_):
            _, lp, ld = flow(torch.cat([q_, p_], dim=1))
            return torch.exp(lp + ld)
        _, a, b = potential_torch.calc_df_deta(f_star, q, p)
        dq += a.detach() / n_flows; dp += b.detach() / n_flows
    out = dict(q=q.detach(), p=p.detach(), dq=dq, dp=dp)
    torch.save(out, CACHE)
    return {k: v.to(DEV) for k, v in out.items()}


# ------------------------------------------------------------------ losses
def stationarity_loss(phi, q, p, dq, dp, lam=2.0):
    """Same loss as potential_torch.get_phi_loss_gradients with mu=0, without the
    unused extra parameter-gradient call."""
    q = q.detach().clone().requires_grad_(True)
    dphi_dq, lap = potential_torch.calc_phi_derivatives(phi, q)
    df_dt = torch.sum(dp * dphi_dq - dq * p, dim=1)
    return torch.mean(asinh(df_dt.abs()) + lam * asinh(torch.clamp(-lap, 0.0)))


_POOL = {}


def _local_pool(stars, n_chunks=8, chunk=128):
    """Coverage-filtered sphere centres, computed ONCE per dataset (stars never change within a task).
    Doing the filter every step cost ~70% of the flux step on Expanse (profile job 55690649)."""
    key = (stars.data_ptr(), tuple(stars.shape))
    if key not in _POOL:
        _POOL[key] = torch.cat([pick_spheres(stars, chunk, 0.2, 1.0, n_radii=4,
                                             max_knn_dist=0.5, n_ref=5000)[0] for _ in range(n_chunks)])
    return _POOL[key]


def _flux_terms(phi, stars, cfg, r_max, s, f_b, delta, gamma, n_dirs):
    """Origin shells (mono, pos, lo, hi) + off-centre local spheres (mono, pos)."""
    z = torch.zeros(8, 3, device=DEV)
    shells = torch.linspace(0.3, r_max, 8, device=DEV)[None].expand(8, -1).contiguous()
    lo, hi = plummer_bounds(f_b, delta)
    lo_s = (lambda c, r: s * lo(c, r)) if cfg.get("lo") else None
    hi_s = (lambda c, r: s * hi(c, r)) if cfg.get("hi") else None
    l1, t1, _ = gauss_flux_loss(phi, z, shells, n_dirs=n_dirs, M_lower=lo_s, M_upper=hi_s,
                                w_lo=1.0, w_hi=1.0)
    pool = _local_pool(stars)
    c = pool[torch.randint(pool.shape[0], (16,), device=pool.device)]
    r = torch.linspace(0.2, 1.0, 4, device=pool.device)[None].expand(16, -1).contiguous()
    l2, t2, _ = gauss_flux_loss(phi, c, r, n_dirs=n_dirs)
    return gamma * (l1 + l2), {**{k: v.item() for k, v in t1.items()}, "local": l2.item()}


# ------------------------------------------------------------------ training
def train_arm(data, arm, n_steps=1500, batch=1024, seed=0, gamma=1.0, s=1.0,
              f_b=0.3, delta=0.2, lam=2.0, n_dirs=64, log_every=250, verbose=True,
              noise=0.0, n_data=None):
    """noise: sigma of fixed Gaussian noise added to df/dq, df/dp (as in the repo notebook, cell 21).
    n_data: train on a random subset of this many phase-space points (data-availability test)."""
    cfg = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    if n_data is not None or noise > 0:
        data = dict(data)
        if n_data is not None:
            sel = torch.randperm(data["q"].shape[0], device=data["q"].device)[:n_data]
            data = {k: v[sel] for k, v in data.items()}
        if noise > 0:
            data["dq"] = data["dq"] + noise * torch.randn_like(data["dq"])
            data["dp"] = data["dp"] + noise * torch.randn_like(data["dp"])
    phi = potential_torch.PhiNN(n_dim=3, n_hidden=3, n_features=128).to(DEV)
    params = [p for p in phi.parameters()]
    opt = flow_torch.RAdam(params, lr=5e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9992)
    N = data["q"].shape[0]
    r_max = float(torch.quantile(data["q"].norm(dim=1), 0.95))   # data-driven, not hand-set
    hist = dict(stat=[], flux=[])
    import time as _t; _t0 = _t.time()
    for i in range(n_steps):
        idx = torch.randint(N, (batch,), device=DEV)
        l_stat = stationarity_loss(phi, data["q"][idx], data["p"][idx],
                                   data["dq"][idx], data["dp"][idx], lam)
        loss = l_stat
        l_flux = torch.zeros(())
        if cfg["flux"]:
            l_flux, _ = _flux_terms(phi, data["q"], cfg, r_max, s, f_b, delta, gamma, n_dirs)
            loss = loss + l_flux
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 10 == 0: sched.step()
        hist["stat"].append(l_stat.item()); hist["flux"].append(float(l_flux))
        if verbose and (i % log_every == 0 or i == n_steps - 1):
            print(f"[{arm} s={s}] step {i:5d} stat={np.mean(hist['stat'][-50:]):.4f} "
                  f"flux={np.mean(hist['flux'][-50:]):.4f} t={_t.time()-_t0:.0f}s")
    return phi, hist


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def _phi_np(phi, q):
    return phi(q).squeeze(-1)


def evaluate(phi, data, r_eval=5.0, n_dirs=256):
    """M_enc(r) relative error on r in [0.3, r_eval] and RMS Phi error (offset-fitted)
    over stars with r < r_eval."""
    r = torch.linspace(0.3, r_eval, 30, device=DEV)[None]
    M = enclosed_mass(phi, torch.zeros(1, 3, device=DEV), r, n_dirs=n_dirs)[0].detach()
    M_true = plummer_enclosed_mass(r[0])
    rel = (M - M_true).abs() / M_true
    m_err = rel.median().item()
    inner = r[0] <= 3.0
    m_err_in, m_err_out = rel[inner].median().item(), rel[~inner].median().item()
    q = data["q"]; q = q[q.norm(dim=1) < r_eval]
    dphi = _phi_np(phi, q) - plummer_phi(q)
    dphi = dphi - dphi.median()
    return dict(M_err=m_err, M_err_in=m_err_in, M_err_out=m_err_out,
                phi_rms=dphi.pow(2).mean().sqrt().item(),
                r=r[0].cpu().numpy(), M=M.cpu().numpy(), M_true=M_true.cpu().numpy())


def run_ablation(data, seeds=(0, 1), **kw):
    res = {}
    for arm in ARMS:
        res[arm] = []
        for sd in seeds:
            phi, h = train_arm(data, arm, seed=sd, **kw)
            res[arm].append(dict(hist=h, **evaluate(phi, data)))
    return res


def run_robustness(data, scales=(0.5, 0.8, 1.0, 1.25, 2.0, 3.0, 4.0), seeds=(0,),
                   arms=("flux: +lower", "flux: +lower+upper"), **kw):
    """Mis-specify the external mass by factor s: M_lower -> s*M_lower, M_upper -> s*M_upper.
    s = 1 is the correctly specified bracket [f_b*M, (1+delta)*M]."""
    res = {a: {s: [] for s in scales} for a in arms}
    for a in arms:
        for s in scales:
            for sd in seeds:
                phi, _ = train_arm(data, a, seed=sd, s=s, verbose=False, **kw)
                e = evaluate(phi, data); e.pop("r"); e.pop("M"); e.pop("M_true")
                res[a][s].append(e)
            print(f"{a} s={s}: M_err={np.mean([e['M_err'] for e in res[a][s]]):.3f} "
                  f"phi_rms={np.mean([e['phi_rms'] for e in res[a][s]]):.3f}")
    return res


# ------------------------------------------------------------------ figures
def make_figures(abl, rob, f_b=0.3, delta=0.2):
    import matplotlib.pyplot as plt
    cols = dict(zip(ARMS, ["k", "tab:blue", "tab:orange", "tab:green"]))

    # Fig 1: stationarity loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, runs in abl.items():
        h = np.mean([r["hist"]["stat"] for r in runs], axis=0)
        k = 50; h = np.convolve(h, np.ones(k) / k, mode="valid")
        ax.semilogy(h, c=cols[arm], label=arm)
    ax.set_xlabel("training step"); ax.set_ylabel("stationarity loss (smoothed)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig1_loss.png", dpi=150)

    # Fig 2: enclosed mass profiles
    fig, ax = plt.subplots(figsize=(6, 4))
    r0 = next(iter(abl.values()))[0]
    ax.plot(r0["r"], r0["M_true"], "k--", lw=2, label="truth")
    ax.fill_between(r0["r"], f_b * r0["M_true"], (1 + delta) * r0["M_true"], color="gray", alpha=0.15,
                    label="assumed bracket")
    for arm, runs in abl.items():
        ax.plot(r0["r"], np.mean([r["M"] for r in runs], axis=0), c=cols[arm], label=arm)
    ax.set_xlabel("r"); ax.set_ylabel(r"$M_{enc}(<r)$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig2_enclosed_mass.png", dpi=150)

    # Fig 3: robustness to mis-specified external mass
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    base = {k: np.mean([r[k] for r in abl["baseline"]]) for k in ("M_err", "phi_rms")}
    for ax, key, lab in zip(axs, ("M_err", "phi_rms"),
                            ("median rel. error of $M_{enc}$", r"RMS $\Phi$ error (offset-fit)")):
        ax.axhline(base[key], c="k", ls="--", label="baseline (no flux loss)")
        for arm, d in rob.items():
            s = sorted(d)
            ax.plot(s, [np.mean([e[key] for e in d[x]]) for x in s], "o-", c=cols[arm], label=arm)
        ax.axvline(1.0, c="gray", ls=":"); ax.set_xscale("log")
        ax.set_xlabel("mass mis-specification factor $s$"); ax.set_ylabel(lab)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_robustness.png", dpi=150)

    # Table
    rows = ["| arm | M_enc rel. err | Phi RMS err |", "|---|---|---|"]
    for arm, runs in abl.items():
        m = [r["M_err"] for r in runs]; p = [r["phi_rms"] for r in runs]
        rows.append(f"| {arm} | {np.mean(m):.3f} ± {np.std(m):.3f} | {np.mean(p):.3f} ± {np.std(p):.3f} |")
    (FIG / "table1_ablation.md").write_text("\n".join(rows), encoding="utf-8")
    return "\n".join(rows)
