#!/bin/bash
#SBATCH --job-name=cisc889x
#SBATCH --account=del126
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --output=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%A_%a.out
#SBATCH --error=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%A_%a.err
#SBATCH --mail-user=yelghara@udel.edu
#SBATCH --mail-type=ALL

# One array task = one trained potential = one row of grid.tsv (row index = SLURM_ARRAY_TASK_ID).
# Submit ONCE:   sbatch /expanse/lustre/projects/del126/yelgharably/CISC889/sbatch/run_grid_x.sbatch
# Rerun failures: sbatch --array=3,17,42 --export=ALL,ALLOW_PARTIAL=1 .../run_grid_x.sbatch
#   (finished tasks are skipped automatically: an existing results/<tag>.json is never overwritten)

set -euo pipefail
ROOT=/expanse/lustre/projects/del126/yelgharably/CISC889
GRID="${GRID:-$ROOT/gauss_flux/grid_x.tsv}"
OUTDIR="${OUTDIR:-$ROOT/results_x}"
CONDA_ENV="${CONDA_ENV:-chgnet}"
cd "$ROOT/gauss_flux"
mkdir -p "$OUTDIR"

# Array size must equal the number of grid rows, otherwise part of the grid is silently skipped.
NROWS=$(( $(wc -l < "$GRID") - 1 ))
if [ "${ALLOW_PARTIAL:-0}" != "1" ] && [ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne "${NROWS}" ]; then
    echo "FATAL: --array size ${SLURM_ARRAY_TASK_COUNT:-0} != grid rows ${NROWS} (set ALLOW_PARTIAL=1 for a rerun)." >&2
    exit 1
fi

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

echo "task ${SLURM_ARRAY_TASK_ID}/${NROWS} host=$(hostname) cpus=${THREADS} $(date)"
python3 -u run_experiment_x.py --grid "$GRID" --task-id "${SLURM_ARRAY_TASK_ID}" --outdir "$OUTDIR"
echo "finished $(date)"
