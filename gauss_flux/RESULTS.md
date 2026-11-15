# Gauss-law flux loss for Deep Potential: results

Component: Gauss-law flux loss (Youssef El Gharably). Project: Deep Potential (Green et al.), CISC889.

## 1. What was built

For a sphere S_R(c) of radius R, Gauss's law gives M_enc(<R) = R² ⟨∂Φ/∂n⟩ / G, where the average is over the sphere's surface. This needs only first derivatives of Φ. Terms added to the stationarity loss (all one-sided, asinh-scaled hinges):

| Term | Constraint |
|---|---|
| mono | M_enc non-decreasing across concentric radii |
| pos | M_enc ≥ 0 for any local sphere |
| lower | M_enc ≥ M_lower (baryonic mass) |
| upper | M_enc ≤ M_upper (optional prior on total mass) |

Spheres are local (not Galactocentric shells). Directions are antithetic pairs, redrawn each step and shared across the concentric radii of one sphere. G and the frame are arguments; M_bar is a swappable callable. Code: `gauss_flux.py` with 8 unit tests (`test_gauss_flux.py`), which check the analytic Plummer mass, an off-centre sphere, zero loss for the true potential, and that wrong or heavy/light potentials are penalised.

Arms used below: baseline (stationarity only), `geom` (mono+pos), `lower`, `lohi` (lower + upper), `monopos` (Gaia: mono+pos).

## 2. Plummer toy (G = M = a = 1; bounds from the true mass, f_b = 0.3, δ = 0.2)

**2.1 The original baseline was unstable, which inflated the first flux result.** At the repository default lr = 5e-2, about 90% of baseline runs collapsed to the zero-mass potential (M_err = 1.000, loss back at its initial value). Any comparison against that baseline overstated the flux benefit, because a lower bound prevents collapse (zero mass violates it) while mono/pos do not.

**2.2 With a stable baseline the flux loss mostly does not help on clean data** (stable grid, 300 runs, 10 seeds per cell, paired differences with 95% bootstrap CIs; ΔΦ_rms = arm − baseline, negative = flux better; best checkpoint chosen by the unsupervised 50-step mean stationarity loss):

- Collapse is removed by any of: a lower learning rate (the baseline never collapsed at lr 1e-3 to 2e-2; at lr 5e-4 it is undertrained and 5 of 10 runs sit near zero mass), gradient clipping with best-checkpoint, or best-checkpoint alone at lr 5e-2.
- Healthy baseline: Φ_rms ≈ 0.011–0.013, M_err ≈ 0.07–0.12.
- `lower` and `geom` are slightly worse than the baseline at every stable setting (ΔΦ_rms ≈ +0.003 to +0.008, CIs exclude 0).
- `lohi` helps at one setting only: lr = 3e-3, ΔΦ_rms = −0.006 [−0.009, −0.003] (M_err 0.044 vs 0.118). At lr = 1e-2 and 2e-2 it is neutral or slightly worse.
- Training 12,000 steps closes the gap: baseline Φ_rms 0.008, M_err 0.04–0.05, with `lohi` statistically indistinguishable.
- The flux loss helps clearly only when the baseline is crippled: at lr = 5e-4 the baseline is undertrained (M_err 0.90) and `lohi` fixes it (ΔΦ_rms −0.034). At lr = 5e-2, apparent flux gains in the final checkpoint (−0.16) vanish once the baseline gets best-checkpoint selection.

**2.3 The flux loss helps under noise** (noise σ on the flow gradients; 560 runs, 10 seeds, lr = 3e-3, best checkpoint, paired differences vs the baseline):

| σ | baseline Φ_rms | `lohi` ΔΦ_rms | `lower` ΔΦ_rms | `geom` ΔΦ_rms |
|---|---|---|---|---|
| 0.004 | 0.018 | −0.007 [−0.010, −0.003] | +0.016 | ≈ 0 |
| 0.016 | 0.057 | −0.031 [−0.038, −0.025] | −0.030 [−0.037, −0.023] | −0.006 (CI includes 0) |
| 0.05 | 0.139 | −0.025 [−0.032, −0.019] | −0.040 [−0.051, −0.031] | −0.002 (CI includes 0) |

Mass error shows the same pattern: the baseline degrades to M_err 0.58–0.67 at σ = 0.05 (0.07–0.12 when clean); `lohi` removes 0.14–0.16 of that and `lower` up to 0.27. Noise biases the baseline's potential shallow, and a mass lower bound pulls it back.

Data scarcity alone (1,024–8,192 points, clean) does not hurt the baseline (Φ_rms 0.012–0.015), and the flux loss gives no meaningful gain there.

The `lohi` advantage depends on the learning rate: at lr = 1e-2 the noise gains shrink and the clean baseline is better than `lohi`.

Figures: `figures/stable_overview.png`, `figures/stable_paired.png`, `figures/noise_paired.png`, `figures/noise_ndata.png`.

## 3. Gaia DR3 (local sample)

**Data.** 549,189 stars within 1 kpc of the Sun: RVS radial velocity present, RUWE < 1.4, parallax/error > 10, random-index subsample, full-sky coverage (checked by declination and RA histograms). Converted to Galactocentric Cartesian coordinates with the astropy-default solar frame (R₀ = 8.122 kpc, z☉ = 20.8 pc, v☉ = (12.9, 245.6, 7.78) km/s). The selection function is not corrected for. An earlier export truncated by `TOP 400000` covered only the northern sky and was discarded.

**Flows.** 16 normalizing flows (30 epochs each) trained on the sample; f and ∂f/∂(q,p) evaluated at 100,000 stars. All converged to the same loss (≈ 7.94) with finite gradients, but they disagree strongly: the scatter between flows is about 60% of the mean gradient. That is the heavy-noise regime of section 2.3.

**Baryon model.** McMillan (2017)-style density (thin + thick stellar disks, HI and H₂ gas, bulge), parameters typed from memory and not checked against the paper. It gives ρ_bar(Sun) = 0.102 M☉/pc³, at the high end of the literature (~0.09–0.10).

**Training.** Same stable protocol as the toy (lr 3e-3, 3000 steps, best checkpoint). A fixed pool of local spheres (radii 0.1–0.4 kpc; centres within 0.5 kpc of the Sun, or 0.6 kpc for the wide-pool runs) with precomputed M_bar. `lohi` uses M_bar ≤ M_enc ≤ k·M_bar, with k = 1.3 or 1.6 as a prior (local dark matter is a small fraction of the mass).

**Results** (3 seeds per arm; no truth exists, so evaluation uses physical checks):

*Local density and sphere mass around the Sun:*

| Arm | ρ(Sun) [M☉/pc³] | M_enc/M_bar, 0.1 → 0.6 kpc |
|---|---|---|
| baseline | 0.029 ± 0.004 | 0.32 → 0.59 |
| mono+pos | 0.042 ± 0.006 | 0.45 → 0.82 |
| `lower`, w = 0.1 / 0.3 / 1 | 0.117 / 0.122 / 0.119 | 1.23–1.28 → 2.08–2.18 |
| `lohi` ≤ 1.3 (wide pool) | 0.095 ± 0.003 | 0.97 → 1.16 |
| `lohi` ≤ 1.6 (wide pool) | 0.101 ± 0.003 | 1.03 → 1.27 |

The literature value is ρ_tot ≈ 0.10 M☉/pc³. The baseline gives an enclosed mass well below the baryon mass, which is unphysical. `lower` fixes this but overshoots; the loss weight barely matters. Adding the upper bound brings the density and sphere mass into the physical range.

*Cylinder test* (Gauss flux through the caps and side of cylinders on the Sun's vertical axis, no thin-slab assumption; quadrature validated on a Plummer test to 4 digits). M_cyl/M_bar for Rc, H = 0.3/0.5, 0.3/1.0, 0.5/0.5, 0.5/1.0 kpc, with the physical expectation ≈ 1.2, 1.4, 1.2, 1.4:

| Arm | 0.3/0.5 | 0.3/1.0 | 0.5/0.5 | 0.5/1.0 |
|---|---|---|---|---|
| baseline | 0.63 | 1.04 | 0.62 | 1.03 |
| `lohi` ≤ 1.3 | 1.18 | 1.42 | 1.16 | 1.40 |
| `lohi` ≤ 1.6 | 1.30 | 1.50 | 1.28 | 1.47 |
| `lower` w = 0.3 | 2.39 | 3.31 | 2.35 | 3.26 |

*Cone test* (30° cones with apex at the Sun, radii 0.5 and 0.9 kpc, closed with a spherical cap plus the conical side; validated on centred and off-centre Plummer potentials). "Expected" adds a uniform ρ_DM = 0.01 M☉/pc³ to the baryon mass:

- In-plane cones (toward/away from the Galactic centre, with/against rotation): expected 1.11–1.19. Baseline 0.23–0.49; `lohi` ≤ 1.3 gives 1.02–1.09; `lohi` ≤ 1.6 gives 1.10–1.21; `lower` 1.38–1.75.
- Vertical cones (±z): expected 1.34–1.40 at 0.5 kpc and 1.74–1.85 at 0.9 kpc. Baseline 1.05–1.09 at 0.5 kpc and 2.12–2.30 at 0.9 kpc; `lohi` 1.21–1.48 at 0.5 kpc and 1.48–2.14 at 0.9 kpc (seed sd 0.6–1.2); `lower` 3.3–8.0.

Figure: `figures/gaia_cone_cylinder.png`; also `figures/gaia_physical.png`.

## 4. What the results support

1. On the toy, the flux loss does not improve a stable, well-tuned baseline on clean data. Its measurable benefit appears when the data gradients are noisy (σ ≥ 0.016) or when the baseline is undertrained or unstable.
2. The same behaviour shows up on Gaia, where the flow ensemble's gradients are noisy. The stationarity loss alone gives an unphysical potential: its enclosed mass is below the baryon mass in the plane.
3. A baryonic lower bound corrects the mass scale but overshoots by about 1.4–3.3× in cylinders and in-plane cones (up to 8× in vertical cones). The optional upper bound brings spheres, cylinders and in-plane cones close to the physical expectation.
4. The baseline's error is anisotropic: too little mass in the plane (0.23–0.49 of the baryons) but too much vertically at 0.9 kpc (2.1–2.3 against an expected 1.7–1.85). It is not simply a uniformly shallow potential.
5. A cylinder and a cone test, computed independently, tell the same story. They are two views of one trained potential, not independent measurements.

## 5. Caveats and corrections

- **Toy bounds come from the true mass**, so they are near-oracle. On Gaia the lower bound needs a real baryon model, and the upper bound is a prior. The upper bound at 1.3×M_bar sits close to the physical value, so part of the agreement in sections 3 is built in. The cylinders and cones are not directly constrained, which makes agreement there more informative, but it is still not an independent measurement.
- **The Σ(<z) estimate was misleading.** An early check used a thin-slab estimate, Σ(<1 kpc) = Kz/(2πG), from the vertical force on the Sun's axis. For `lohi` it gave 15–22 M☉/pc² against ~70 in the literature and 54 for the baryons alone. The direct cylinder test contradicts this, so the Σ(<z) numbers should not be reported as a failure of the potential. I did not establish exactly why the estimator disagrees.
- **A "coverage" hypothesis was tested and not supported.** Widening the sphere pool to reach z ≈ 1 kpc did not change Σ(<1 kpc).
- **Vertical cones are noisy.** The baryon mass there is only about 10⁶ M☉ and few stars lie far above the disk, so seed scatter is large.
- **Small sample:** 3 seeds per Gaia arm, one flow ensemble, one sample draw. There is no uncertainty from redrawing the stars, and the selection function and non-stationarity of the disk (spiral arms, bar) are not modelled.
- **Baryon model:** parameters typed from memory; ρ_bar(Sun) = 0.102 is at the high end. A lower baryon scale would change the `lower` overshoot.
- **Noise model:** the toy noise is Gaussian on the flow gradients; real flow errors are probably correlated and non-Gaussian.
- The Gaia numbers are a qualitative demonstration, not a Milky Way mass measurement.

## 6. Reproducing

Toy: `gauss_flux/` (`make_grid_s.py`, `make_grid_n.py`, `run_experiment_s.py`, `analyze_s.py`, `analyze_n.py`, `fig_s.py`, `fig_n.py`; Slurm scripts in `sbatch/`). Gaia: `gaia/` (`gaia_fetch.py --csv`, `gaia_flows.py`, `gaia_phi.py`, `baryon_model.py`, `cyl_diag.py`, `cone_diag.py`, `fig_gaia.py`, `fig_geom.py`). Cluster results: `expanse_pull/results_s/`, `expanse_pull/results_n/`; Gaia flows: `gaia/flows/`; trained Gaia potentials: `gaia/phi_w/`.
