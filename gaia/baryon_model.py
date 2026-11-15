"""Baryonic mass M_bar(centre, radii) for the Galactic baryons, as the swappable callable the flux loss expects.
Density model: McMillan (2017) best-fit baryons (thin+thick stellar disks, HI and H2 gas disks, bulge), parameters
typed from the paper's Table (verify against the paper before quoting numbers). Units: kpc, Msun.
M_bar(sphere) is integrated by fixed quasi-random points in the unit ball (deterministic -> a constant for training)."""
import numpy as np
S = 1e6  # Msun/pc^2 -> Msun/kpc^2
DISKS = dict(thin=(896 * S, 2.50, 0.300), thick=(183 * S, 3.02, 0.900))
GAS = dict(HI=(53.1 * S, 7.0, 4.0, 0.085), H2=(2180 * S, 1.5, 12.0, 0.045))
BULGE = dict(rho0=98.4e9, r0=0.075, rcut=2.1, alpha=1.8, q=0.5)

def rho_bar(x, y, z, scale=1.0):
    R = np.sqrt(x * x + y * y) + 1e-9; az = np.abs(z); rho = 0.0
    for S0, Rd, zd in DISKS.values(): rho = rho + S0 / (2 * zd) * np.exp(-az / zd - R / Rd)
    for S0, Rd, Rm, zd in GAS.values(): rho = rho + S0 / (4 * zd) * np.exp(-Rm / R - R / Rd) / np.cosh(z / (2 * zd)) ** 2
    rp = np.sqrt(R ** 2 + (z / BULGE["q"]) ** 2)
    rho = rho + BULGE["rho0"] / (1 + rp / BULGE["r0"]) ** BULGE["alpha"] * np.exp(-(rp / BULGE["rcut"]) ** 2)
    return scale * rho

def _ball(n=40000, seed=0):
    g = np.random.default_rng(seed); v = g.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * g.random((n, 1)) ** (1 / 3)
_B = _ball()

def M_bar(centers, radii, scale=1.0):
    """centers (S,3) kpc, radii (S,K) kpc (numpy) -> baryonic mass (S,K) in Msun inside each sphere."""
    centers = np.asarray(centers, float); radii = np.asarray(radii, float); out = np.zeros(radii.shape)
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pts = centers[i] + radii[i, j] * _B
            out[i, j] = rho_bar(pts[:, 0], pts[:, 1], pts[:, 2], scale).mean() * 4 / 3 * np.pi * radii[i, j] ** 3
    return out

if __name__ == "__main__":
    sun = np.array([[-8.122, 0, 0.0208]]); print("rho_bar at Sun [Msun/pc^3]:", rho_bar(*sun[0]) / 1e9)
    print("M_bar(r=0.1,0.3,0.5,1 kpc) around Sun [Msun]:", M_bar(sun, np.array([[0.1, 0.3, 0.5, 1.0]])))
