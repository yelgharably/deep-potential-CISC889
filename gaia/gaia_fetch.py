#!/usr/bin/env python
"""Fetch a clean 6D Gaia DR3 sample within --dmax kpc of the Sun (TAP over HTTP, no astropy needed) and convert to
Galactocentric Cartesian (kpc, km/s) with the astropy-default solar frame (R0=8.122 kpc, z_sun=20.8 pc,
v_sun=(12.9,245.6,7.78) km/s). Output gaia_6d.npz: q (N,3), p (N,3), G.
Selection (this IS the selection function): RVS radial velocity present, RUWE<1.4, parallax_over_error>10, d<dmax."""
import argparse, io, numpy as np, pandas as pd, requests
ap = argparse.ArgumentParser(); ap.add_argument("--dmax", type=float, default=1.0); ap.add_argument("--limit", type=int, default=400000)
ap.add_argument("--rmax", type=int, default=1800000000); ap.add_argument("--csv", default=None, help="skip the download and convert a CSV exported from the Gaia archive"); ap.add_argument("--out", default="gaia_6d.npz"); a = ap.parse_args()
if a.csv:
    t = pd.read_csv(a.csv); print("rows:", len(t))
else:
    import time
    from concurrent.futures import ThreadPoolExecutor
    LEV, PER = 3, a.limit // 768 + 1  # HEALPix level-3 tiles (768), PER stars per tile, so the sky is covered evenly
    WIDTH = 34359738368 * 4 ** (12 - LEV)
    import os
    os.makedirs('tiles', exist_ok=True)
    def tile(i):
        cf = f'tiles/t{i:03d}.csv'
        if os.path.exists(cf): return pd.read_csv(cf)
        adql = f"""SELECT TOP {PER} ra, dec, parallax, pmra, pmdec, radial_velocity, phot_g_mean_mag FROM gaiadr3.gaia_source
    WHERE source_id >= {i*WIDTH} AND source_id < {(i+1)*WIDTH} AND random_index < {a.rmax} AND radial_velocity IS NOT NULL
    AND ruwe < 1.4 AND parallax_over_error > 10 AND parallax > {1.0/a.dmax}"""
        for attempt in range(4):
            try:
                r = requests.post("https://gea.esac.esa.int/tap-server/tap/async", data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", PHASE="RUN", MAXREC="100000", QUERY=adql), allow_redirects=False, timeout=120)
                ju = r.headers["Location"]
                while True:
                    ph = requests.get(ju + "/phase", timeout=120).text.strip()
                    if ph in ("COMPLETED", "ERROR", "ABORTED"): break
                    time.sleep(3)
                if ph == "COMPLETED":
                    df_ = pd.read_csv(io.StringIO(requests.get(ju + "/results/result", timeout=600).text)); df_.to_csv(cf, index=False); print('tile', i, len(df_), flush=True); return df_
                print('tile', i, ph, flush=True)
            except Exception as e: print('tile',i,'attempt',attempt,type(e).__name__,str(e)[:80],flush=True)
            time.sleep(20)
        print("tile failed", i); return None
    with ThreadPoolExecutor(3) as ex: parts = list(ex.map(tile, range(768)))
    t = pd.concat([x for x in parts if x is not None], ignore_index=True); print("rows:", len(t), "tiles failed:", sum(x is None for x in parts))
    
ra, dec = np.radians(t.ra.values), np.radians(t.dec.values); k = 4.740470446  # km/s per (mas/yr * kpc)
d = 1.0 / t.parallax.values  # kpc
# ICRS -> Galactic rotation (Hipparcos/astropy matrix)
A = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
              [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
              [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])
cr, sr, cd, sd = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec)
xyz = np.stack([cd * cr, cd * sr, sd], 1) * d[:, None]
e_ra = np.stack([-sr, cr, 0 * sr], 1); e_dec = np.stack([-sd * cr, -sd * sr, cd], 1); e_r = np.stack([cd * cr, cd * sr, sd], 1)
vel = e_ra * (k * d * t.pmra.values)[:, None] + e_dec * (k * d * t.pmdec.values)[:, None] + e_r * t.radial_velocity.values[:, None]
xg, vg = xyz @ A.T, vel @ A.T  # heliocentric Galactic: x to GC, y rotation, z NGP
R0, zsun, vsun = 8.122, 0.0208, np.array([12.9, 245.6, 7.78])
th = np.arcsin(zsun / R0)  # astropy tilts the frame so the Sun sits at z_sun
Rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
q = xg @ Rot.T + np.array([-R0 * np.cos(th), 0.0, zsun])  # Sun at (-R0 cos th, 0, z_sun) in the Galactocentric frame
p = vg @ Rot.T + vsun
ok = np.isfinite(q).all(1) & np.isfinite(p).all(1)
np.savez(a.out, q=q[ok], p=p[ok], G=t.phot_g_mean_mag.values[ok])
print("saved", ok.sum(), "| q mean", q[ok].mean(0).round(3), "| p mean", p[ok].mean(0).round(1), "| p std", p[ok].std(0).round(1))
