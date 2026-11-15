#!/bin/bash
#SBATCH --job-name=cisc889_check
#SBATCH --account=del126
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%j.out
#SBATCH --error=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%j.err
#SBATCH --mail-user=yelghara@udel.edu
#SBATCH --mail-type=ALL

# Run this FIRST. Checks the python env, runs the unit tests, and trains a 30-step smoke task.
#   sbatch /expanse/lustre/projects/del126/yelgharably/CISC889/sbatch/check_env.sbatch
# Writes only to results_smoke/ (never to results/). Re-running skips an existing smoke result.

set -euo pipefail
ROOT=/expanse/lustre/projects/del126/yelgharably/CISC889
CONDA_ENV="${CONDA_ENV:-chgnet}"
cd "$ROOT/gauss_flux"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CISC889_DEVICE=cpu MPLBACKEND=Agg PYTHONUNBUFFERED=1

ok=0
for try in 1 2 3 4; do
    if module purge && module load cpu/0.21.2a; then ok=1; break; fi
    echo "module load failed (try ${try}); retrying" >&2; sleep 10
done
[ "$ok" = 1 ] || echo "WARNING: module load never succeeded; continuing with conda env only" >&2

source /expanse/lustre/projects/del126/yelgharably/LAMMPS/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
echo "== python: $(which python3)  $(python3 --version)"
for m in numpy scipy matplotlib torch; do
    python3 -c "import $m; print('$m', $m.__version__)" || echo "MISSING: $m" >&2
done
python3 -c "import torch" || { echo "FATAL: torch missing in '$CONDA_ENV' -- see sbatch/ENV_SETUP.md" >&2; exit 1; }

echo "== unit tests"
python3 -u test_gauss_flux.py
echo "== smoke task (30 steps, one lohi run)"
python3 -u run_experiment.py --smoke --smoke-steps 30 --outdir "$ROOT/results_smoke"
echo "== data cache"
ls -l plummer_data_cache.pt
echo "check done $(date)"
