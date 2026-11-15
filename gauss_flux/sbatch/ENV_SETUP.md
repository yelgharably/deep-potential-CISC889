# Python env for CISC889 on Expanse

Default: the existing conda env `chgnet` (miniforge at
`/expanse/lustre/projects/del126/yelgharably/LAMMPS/miniforge3`). Needs numpy, scipy, matplotlib, torch (CPU is enough).
`check_env.sbatch` reports which of these are missing. [UNCONFIRMED] whether `chgnet` already has all four
(torch is likely since CHGNet is a PyTorch package, but nobody has checked on the cluster).

**Never modify `chgnet`** (other projects use it). If anything is missing, make a separate env on a LOGIN node
(compute nodes may have no internet [UNCONFIRMED]):

```bash
source /expanse/lustre/projects/del126/yelgharably/LAMMPS/miniforge3/etc/profile.d/conda.sh
conda create -n cisc889 python=3.11 -y
conda activate cisc889
pip install numpy scipy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then submit with the env name passed in:

```bash
sbatch --export=ALL,CONDA_ENV=cisc889 /expanse/lustre/projects/del126/yelgharably/CISC889/sbatch/check_env.sbatch
sbatch --export=ALL,CONDA_ENV=cisc889 /expanse/lustre/projects/del126/yelgharably/CISC889/sbatch/run_grid.sbatch
```

Version notes: the repo's `toy_systems.py` uses `scipy.integrate.cumtrapz` (removed in SciPy >= 1.14); `ablation.py`
shims it. `flow_torch.py` uses `torch.lu`, but only when BUILDING flows, which the grid never does (it loads the
cached `plummer_data_cache.pt`).
