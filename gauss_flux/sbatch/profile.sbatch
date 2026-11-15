#!/bin/bash
#SBATCH --job-name=cisc889_prof
#SBATCH --account=del126
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%j.out
#SBATCH --error=/expanse/lustre/projects/del126/yelgharably/CISC889/logs/%x_%j.err
#SBATCH --mail-user=yelghara@udel.edu
#SBATCH --mail-type=ALL

# Diagnoses the ~4.4 s/step seen in check_env (expected ~0.1 s). Run BEFORE run_grid.sbatch.
#   sbatch /expanse/lustre/projects/del126/yelgharably/CISC889/sbatch/profile.sbatch
set -euo pipefail
ROOT=/expanse/lustre/projects/del126/yelgharably/CISC889
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
conda activate chgnet
echo "== node: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|MHz" || true
python3 -u profile_steps.py
echo "profile finished $(date)"
