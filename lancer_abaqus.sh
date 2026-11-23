#!/bin/bash
#SBATCH --job-name=lamb-abaqus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=abaqus-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load abaqus/2023
# If needed, load the Python module used when creating .venv-euler here.
PYTHON="${PYTHON:-$SLURM_SUBMIT_DIR/.venv-euler/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python introuvable : $PYTHON. Créer .venv-euler ou définir PYTHON." >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
RUN="$SLURM_SUBMIT_DIR/abaqus_runs/$SLURM_JOB_ID"
mkdir -p "$RUN"
if [[ -n "${DESIGN_PATH:-}" ]]; then
    "$PYTHON" abaqus_generate.py --output "$RUN" --design "$DESIGN_PATH"
else
    "$PYTHON" abaqus_generate.py --output "$RUN"
fi
cd "$RUN"
abaqus job=bare input=bare.inp cpus="$SLURM_CPUS_PER_TASK" mp_mode=threads interactive
abaqus python "$SLURM_SUBMIT_DIR/abaqus_extract.py" bare.odb manifest.json bare_history.json
if [[ -n "${DESIGN_PATH:-}" ]]; then
    abaqus job=motifs input=motifs.inp cpus="$SLURM_CPUS_PER_TASK" mp_mode=threads interactive
    abaqus python "$SLURM_SUBMIT_DIR/abaqus_extract.py" motifs.odb manifest.json motifs_history.json
    "$PYTHON" "$SLURM_SUBMIT_DIR/abaqus_compare.py" .
fi
echo "Résultats : $RUN"
