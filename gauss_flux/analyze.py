#!/usr/bin/env python
"""
Aggregate pulled Expanse results (results/*.json) into tables + figures.

  python analyze.py --results results --out figures_expanse

Statistics: arms share seeds (same init seed), so we report PAIRED differences vs baseline with a 95% bootstrap CI,
alongside mean +- std. A claim is only "supported" when the CI excludes 0. Lower is better for every metric.
Selection is by config (not group), because duplicate configs were dropped from the grid.
"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["baseline", "geom", "lower", "lohi"]
LAB = {"baseline": "baseline", "geom": "mono+pos", "lower": "+lower", "lohi": "+lower+upper"}
COL = {"baseline": "k", "geom": "tab:blue", "lower": "tab:orange", "lohi": "tab:green"}
METRICS = [("M_err_in", "M_enc err (r<=3)"), ("M_err_out", "M_enc err (r>3)"), ("phi_rms", "Phi RMS err")]
DEFAULT = dict(s=1.0, gamma=1.0, f_b=0.3, delta=0.2, noise=0.0, n_data="all")


def load(d):
    runs = []
    for f in sorted(glob.glob(str(Path(d) / "*.json"))):
        r = json.load(open(f))
        c = r["config"]
        runs.append(dict(arm=c["arm"], s=float(c["s"]), gamma=float(c["gamma"]), f_b=float(c["f_b"]),
                         delta=float(c["delta"]), noise=float(c["noise"]), n_data=str(c["n_data"]),
                         steps=int(c["steps"]), seed=int(c["seed"]), status=r["status"], res=r))
    return runs


def sel(runs, steps, **kw):
    cfg = {**DEFAULT, **kw}
    cfg["n_data"] = str(cfg["n_data"])
    return [r for r in runs if r["steps"] == steps and r["status"] == "ok"
            and all(r[k] == (float(v) if k not in ("arm", "n_data") else v) for k, v in cfg.items())]


def vals(rs, key):
    return {r["seed"]: r["res"][key] for r in rs}


def paired(rs, base, key, nboot=4000, rng=np.random.default_rng(0)):
    a, b = vals(rs, key), vals(base, key)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3: return None
    d = np.array([a[s] - b[s] for s in seeds])
    bs = rng.choice(d, (nboot, len(d))).mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)


def ms(rs, key):
    v = np.array(list(vals(rs, key).values()))
    return (v.mean(), v.std(), len(v)) if len(v) else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--out", default="figures_expanse")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    runs = load(a.results)
    bad = [r["res"]["tag"] for r in runs if r["status"] != "ok"]
    print(f"{len(runs)} runs loaded, {len(bad)} non-finite: {bad[:5]}")
    S = a.steps
    lines = []

    # ---- Table 1: main ablation with paired differences
    base = sel(runs, S, arm="baseline")
    lines += ["| arm | n | " + " | ".join(l for _, l in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for arm in ARMS:
        rs = sel(runs, S, arm=arm); cells = []
        for k, _ in METRICS:
            m, sd, n = ms(rs, k); cell = f"{m:.3f} +- {sd:.3f}"
            if arm != "baseline":
                p = paired(rs, base, k)
                if p: cell += f" (d={p[0]:+.3f} [{p[1]:+.3f},{p[2]:+.3f}])"
            cells.append(cell)
        lines.append(f"| {LAB[arm]} | {len(rs)} | " + " | ".join(cells) + " |")
    (out / "table1_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- Fig A: boxplots per arm
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k, lab) in zip(axs, METRICS):
        data = [list(vals(sel(runs, S, arm=ar), k).values()) for ar in ARMS]
        bp = ax.boxplot([d if d else [np.nan] for d in data], labels=[LAB[x] for x in ARMS], showmeans=True)
        ax.set_ylabel(lab); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "figA_ablation.png", dpi=150)

    # ---- Fig B: robustness to mis-specified mass s
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k, lab) in zip(axs, METRICS):
        m, sd, _ = ms(base, k)
        if _: ax.axhspan(m - sd, m + sd, color="k", alpha=0.12); ax.axhline(m, c="k", ls="--", label="baseline")
        for arm in ("lower", "lohi"):
            ss = sorted({r["s"] for r in runs if r["arm"] == arm and r["steps"] == S})
            mm = [ms(sel(runs, S, arm=arm, s=s), k) for s in ss]
            ax.errorbar(ss, [x[0] for x in mm], [x[1] / np.sqrt(max(x[2], 1)) for x in mm], c=COL[arm],
                        marker="o", capsize=2, label=LAB[arm])
        ax.set_xscale("log"); ax.set_xlabel("mass mis-specification s"); ax.set_ylabel(lab); ax.axvline(1, c="gray", ls=":")
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figB_robustness.png", dpi=150)

    # ---- Fig C: gamma and bracket shape
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k, lab) in zip(axs, METRICS):
        for arm in ("geom", "lohi"):
            gs = sorted({r["gamma"] for r in runs if r["arm"] == arm and r["steps"] == S})
            mm = [ms(sel(runs, S, arm=arm, gamma=g), k) for g in gs]
            ax.errorbar(gs, [x[0] for x in mm], [x[1] / np.sqrt(max(x[2], 1)) for x in mm], c=COL[arm], marker="o", label=LAB[arm])
        m, _, n = ms(base, k)
        if n: ax.axhline(m, c="k", ls="--", label="baseline")
        ax.set_xscale("log"); ax.set_xlabel(r"flux-loss weight $\gamma$"); ax.set_ylabel(lab)
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figC_gamma.png", dpi=150)

    # ---- Fig D: noise / data / training length  (baseline vs mono+pos vs lohi)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    spec = [("noise", "sigma of noise on df/deta", {"steps": S}), ("n_data", "# training points", {"steps": S}),
            ("steps", "training steps", {})]
    for ax, (key, xl, _) in zip(axs, spec):
        for arm in ("baseline", "geom", "lohi"):
            pts = []
            if key == "noise":
                xs = sorted({r["noise"] for r in runs if r["arm"] == arm and r["steps"] == S})
                pts = [(x, ms(sel(runs, S, arm=arm, noise=x), "phi_rms")) for x in xs]
            elif key == "n_data":
                xs = sorted({r["n_data"] for r in runs if r["arm"] == arm and r["steps"] == S}, key=lambda v: 1e9 if v == "all" else int(v))
                pts = [(32768 if x == "all" else int(x), ms(sel(runs, S, arm=arm, n_data=x), "phi_rms")) for x in xs]
            else:
                xs = sorted({r["steps"] for r in runs if r["arm"] == arm and r["s"] == 1.0 and r["gamma"] == 1.0 and r["noise"] == 0 and r["n_data"] == "all"})
                pts = [(x, ms(sel(runs, x, arm=arm), "phi_rms")) for x in xs]
            pts = [p for p in pts if p[1][2]]
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], [p[1][1] / np.sqrt(p[1][2]) for p in pts],
                            c=COL[arm], marker="o", label=LAB[arm])
        ax.set_xlabel(xl); ax.set_ylabel("Phi RMS err")
        if key != "noise": ax.set_xscale("log")
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figD_noise_data_steps.png", dpi=150)

    # ---- Fig E: enclosed-mass curves (mean +- std over seeds)
    fig, ax = plt.subplots(figsize=(6, 4)); r0 = None
    for arm in ARMS:
        rs = sel(runs, S, arm=arm)
        if not rs: continue
        M = np.array([x["res"]["M"] for x in rs]); r = np.array(rs[0]["res"]["r"]); r0 = (r, rs[0]["res"]["M_true"])
        ax.plot(r, M.mean(0), c=COL[arm], label=LAB[arm]); ax.fill_between(r, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=COL[arm], alpha=0.15)
    if r0: ax.plot(*r0, "k--", lw=2, label="truth")
    ax.set_xlabel("r"); ax.set_ylabel(r"$M_{enc}(<r)$"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figE_enclosed_mass.png", dpi=150)
    print("figures ->", out)


if __name__ == "__main__":
    main()
