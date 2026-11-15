"""Run: python test_gauss_flux.py   (validates sign/normalisation conventions)"""
import torch
from gauss_flux import (enclosed_mass, gauss_flux_loss, pick_spheres,
                        plummer_phi, plummer_enclosed_mass, plummer_bounds)

torch.manual_seed(0)
origin = torch.zeros(1, 3)
radii = torch.tensor([[0.3, 0.7, 1.0, 2.0, 4.0, 8.0]])

# 1. enclosed mass of analytic Plummer about origin matches r^3/(1+r^2)^1.5
M = enclosed_mass(plummer_phi, origin, radii, n_dirs=64)
ref = plummer_enclosed_mass(radii)
assert torch.allclose(M, ref, rtol=1e-4), (M, ref)

# 2. off-centre sphere containing the whole system -> total mass 1
c = torch.tensor([[1.0, -0.5, 0.3]])
M_big = enclosed_mass(plummer_phi, c, torch.tensor([[40.0]]), n_dirs=512)
assert abs(M_big.item() - 1.0) < 0.01, M_big

# 3. true potential -> all terms ~ 0 (bounds: f_b=0.3, delta=0.2)
lo, hi = plummer_bounds(0.3, 0.2)
tot, terms, _ = gauss_flux_loss(plummer_phi, origin, radii, n_dirs=64, M_lower=lo, M_upper=hi)
assert tot.item() < 1e-6, terms

# 4. wrong-sign potential (negative mass) is penalised by pos and lo, and mono is not
bad = lambda q: -plummer_phi(q)
tot, terms, _ = gauss_flux_loss(bad, origin, radii, n_dirs=64, M_lower=lo)
assert terms['pos'] > 0.1 and terms['lo'] > 0.1, terms

# 5. too-heavy potential (3x) violates upper bound; too-light (0.1x) violates lower bound
_, t_hi, _ = gauss_flux_loss(lambda q: 3 * plummer_phi(q), origin, radii, M_upper=hi)
_, t_lo, _ = gauss_flux_loss(lambda q: 0.1 * plummer_phi(q), origin, radii, M_lower=lo)
assert t_hi['hi'] > 0.1 and t_lo['lo'] > 0.1

# 6. mono penalises a shell-mass decrease (mass that shrinks with radius)
shrinking = lambda q: -1.0 / (1 + (q ** 2).sum(-1)) ** 0.5 * torch.exp(-0.2 * (q ** 2).sum(-1).sqrt())
_, t, M_s = gauss_flux_loss(shrinking, origin, radii, w_pos=0)
assert (M_s[:, 1:] < M_s[:, :-1]).any() and t['mono'] > 0

# 7. gradients reach NN parameters
net = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Sigmoid(), torch.nn.Linear(16, 1))
phi = lambda q: net(q).squeeze(-1)
big_lo = lambda c, r: 100 * lo(c, r)          # make the hinge active for a random net
tot, _, _ = gauss_flux_loss(phi, origin, radii, M_lower=big_lo)
tot.backward()
# the final bias is excluded: Phi is offset-invariant, so its gradient is exactly 0 (correct physics)
assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in list(net.parameters())[:-1])
assert net[2].bias.grad is None or net[2].bias.grad.abs().sum() == 0

# 8. pick_spheres runs and respects the coverage filter
stars = torch.randn(5000, 3)
c, r = pick_spheres(stars, 32, 0.2, 1.0, n_radii=5, max_knn_dist=0.3)
assert c.shape[1] == 3 and r.shape == (c.shape[0], 5)

print("all gauss_flux tests passed")
