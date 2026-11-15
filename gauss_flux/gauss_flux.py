"""
Gauss-law flux loss for Deep Potential.

Integral form of Poisson's equation on a sphere S_R(c) of radius R about centre c:

    M_enc(<R) = (1/(4 pi G)) * closed_integral( grad(Phi) . n dA )
              = R^2 * <dPhi/dn>_{directions} / G

Only first derivatives of Phi are needed (no Laplacian).

Constraints built on M_enc (all one-sided hinges, scaled by asinh like the paper):
  mono : concentric spheres must have non-decreasing M_enc(R)       (rho >= 0)
  pos  : M_enc >= 0 for any local sphere                            (rho >= 0)
  lo   : M_enc >= M_lower(sphere)   e.g. baryonic mass              (dark matter >= 0)
  hi   : M_enc <= M_upper(sphere)   e.g. ceiling on total mass      (optional)

Conventions
  * centres are in the SAME frame/units as phi_func's input (e.g. Galactocentric kpc)
  * G is an argument (Plummer toy: 1.0; kpc, km/s, Msun: 4.30091e-6)
  * directions are shared across the K concentric radii of a sphere (common random
    numbers -> sampling noise cancels in M(R_j) - M(R_i)); they are antithetic
    (n, -n) and are redrawn every call.
"""
import torch


def asinh(x):
    return torch.log(x + (x ** 2 + 1) ** 0.5)


# ------------------------------------------------------------------ sampling
def sample_dirs(n_spheres, n_dirs, device=None, dtype=torch.float32):
    """Antithetic unit vectors, shape (S, D, 3). n_dirs must be even."""
    assert n_dirs % 2 == 0, "n_dirs must be even (antithetic pairs)"
    v = torch.randn(n_spheres, n_dirs // 2, 3, device=device, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True)
    return torch.cat([v, -v], dim=1)


# ------------------------------------------------------------------ core
def enclosed_mass(phi_func, centers, radii, n_dirs=128, G=1.0):
    """
    centers: (S,3)   radii: (S,K) concentric radii per sphere (increasing in K)
    returns M_enc: (S,K), differentiable w.r.t. phi_func's parameters.
    """
    S, K = radii.shape
    n_hat = sample_dirs(S, n_dirs, device=radii.device, dtype=radii.dtype)    # (S,D,3)
    q = centers[:, None, None, :] + radii[:, :, None, None] * n_hat[:, None]  # (S,K,D,3)
    q = q.reshape(-1, 3).requires_grad_(True)
    phi = phi_func(q)
    (grad,) = torch.autograd.grad(phi.sum(), q, create_graph=True)
    grad = grad.view(S, K, n_dirs, 3)
    dphi_dn = (grad * n_hat[:, None]).sum(-1)                                 # (S,K,D)
    return radii ** 2 * dphi_dn.mean(-1) / G                                  # (S,K)


# ------------------------------------------------------------------ loss
def gauss_flux_loss(phi_func, centers, radii, *, n_dirs=128, G=1.0, m_scale=1.0,
                    M_lower=None, M_upper=None,
                    w_mono=1.0, w_pos=1.0, w_lo=1.0, w_hi=1.0):
    """
    M_lower / M_upper: None, or callable(centers, radii) -> (S,K) tensor, or a (S,K) tensor.
    m_scale: typical mass in code units; normalises the mono / pos hinges.
    Returns (total, dict_of_terms, M_enc).
    """
    M = enclosed_mass(phi_func, centers, radii, n_dirs, G)
    terms = {}

    if w_mono and radii.shape[1] > 1:
        terms['mono'] = w_mono * asinh(torch.relu(M[:, :-1] - M[:, 1:]) / m_scale).mean()
    if w_pos:
        terms['pos'] = w_pos * asinh(torch.relu(-M) / m_scale).mean()

    def _get(b):
        return b(centers, radii) if callable(b) else b

    if M_lower is not None and w_lo:
        lo = _get(M_lower)
        terms['lo'] = w_lo * asinh(torch.relu(lo - M) / lo.abs().clamp_min(1e-12)).mean()
    if M_upper is not None and w_hi:
        hi = _get(M_upper)
        terms['hi'] = w_hi * asinh(torch.relu(M - hi) / hi.abs().clamp_min(1e-12)).mean()

    total = sum(terms.values()) if terms else M.sum() * 0.0
    return total, terms, M


# ------------------------------------------------------- choosing local spheres
@torch.no_grad()
def pick_spheres(stars, n_spheres, r_min, r_max, n_radii=6,
                 max_knn_dist=None, n_probe=64, n_ref=20000, max_tries=20):
    """
    Choose sphere centres at (resampled) star positions and keep only spheres whose
    OUTER surface lies near data: median nearest-star distance of probe points on the
    r_max surface must be < max_knn_dist. This bounds how much of the flux integral is
    NN extrapolation. Returns centers (S,3), radii (S,K) (linear grid r_min..r_max).
    """
    N = stars.shape[0]
    ref = stars[torch.randperm(N)[:min(n_ref, N)]]
    kept = []
    for _ in range(max_tries):
        c = stars[torch.randint(N, (n_spheres,))]
        if max_knn_dist is None:
            kept.append(c)
        else:
            probe = c[:, None, :] + r_max * sample_dirs(n_spheres, n_probe,
                                                         device=stars.device, dtype=stars.dtype)
            d = torch.cdist(probe.reshape(-1, 3), ref).min(dim=1).values.view(n_spheres, n_probe)
            kept.append(c[d.median(dim=1).values < max_knn_dist])
        if sum(k.shape[0] for k in kept) >= n_spheres:
            break
    centers = torch.cat(kept)[:n_spheres]
    radii = torch.linspace(r_min, r_max, n_radii, device=stars.device, dtype=stars.dtype)
    return centers, radii[None].expand(centers.shape[0], -1).contiguous()


# ------------------------------------------------------- Plummer toy helpers (G=1, M=1, a=1)
def plummer_phi(q):
    return -(1 + (q ** 2).sum(-1)) ** -0.5


def plummer_enclosed_mass(r):
    return r ** 3 / (1 + r ** 2) ** 1.5      # spherical shells about the ORIGIN only


def plummer_bounds(f_b=0.3, delta=0.2):
    """
    By-fiat 'baryonic' fraction for the toy: M_lower = f_b * M_true, M_upper = (1+delta) * M_true.
    Only valid for spheres centred on the origin (the closed form needs spherical symmetry);
    for off-centre spheres, precompute the true enclosed mass numerically and pass tensors.
    """
    def true_M(centers, radii):
        assert centers.abs().max() < 1e-6, "closed form only valid for origin-centred spheres"
        return plummer_enclosed_mass(radii)
    return (lambda c, r: f_b * true_M(c, r)), (lambda c, r: (1 + delta) * true_M(c, r))
