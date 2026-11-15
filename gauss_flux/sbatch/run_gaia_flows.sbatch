#!/bin/bash
#SBATCH --job-name=cisc889g
#SBATCH --account=del126
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%A_%a.out
#SBATCH --error=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%A_%a.err
#SBATCH --mail-user=yelghara@udel.edu
#SBATCH --mail-type=ALL

# One array task = one normalizing flow on the Gaia sample (also writes f, df/dq, df/dp at 100k stars).
#   sbatch /expanse/lustre/projects/del126/yelgharably/CISC889/sbatch/run_gaia_flows.sbatch
set -euo pipefail
ROOT=/expanse/lustre/projects/del126/yelgharably/CISC889
CONDA_ENV="${CONDA_ENV:-chgnet}"
cd "$ROOT/gaia"; mkdir -p flows
THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
       NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export CISC889_DEVICE=cpu MPLBACKEND=Agg PYTHONUNBUFFERED=1

# Lmod's cache fails intermittently on some nodes -> retry, then continue (the conda env does not need the modules)
ok=0
for try in 1 2 3 4; do
    if module purge && module load cpu/0.21.2a; then ok=1; break; fi
    echo "module load failed (try ${try}); retrying" >&2; sleep 10
done
[ "$ok" = 1 ] || echo "WARNING: module load never succeeded; continuing with conda env only" >&2

source /expanse/lustre/projects/del126/yelgharably/LAMMPS/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
python3 -c "import numpy, scipy, matplotlib, torch; print('env ok: torch', torch.__version__, 'numpy', numpy.__version__)" \
    || { echo "FATAL: env '$CONDA_ENV' lacks numpy/scipy/matplotlib/torch -- see sbatch/ENV_SETUP.md" >&2; exit 1; }

echo "flow ${SLURM_ARRAY_TASK_ID} host=$(hostname) $(date)"
python3 -u gaia_flows.py --flow-id "${SLURM_ARRAY_TASK_ID}" --epochs "${EPOCHS:-30}" --units "${UNITS:-16}" --outdir flows
echo "finished $(date)"
