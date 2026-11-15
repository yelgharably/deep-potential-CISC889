"""Shared constants/model for the Gaia scripts (mirrors gaia_phi.py)."""
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = next(p for p in [HERE.parents[1] / "deep-potential-master" / "deep-potential-master", HERE.parent / "deep-potential-master" / "deep-potential-master", HERE.parent / "deep-potential-master"] if (p / "scripts").exists())
sys.path.insert(0, str(REPO / "scripts"))
import potential_torch
L, V, GK = 1.0, 100.0, 4.30091e-6
C = np.array([-8.122, 0.0, 0.0208])
def make_phi(): return potential_torch.PhiNN(3, 3, 128)
