#!/bin/bash
# Materializa base/, grid_preprocess/ y grid_hyperparams/ a partir de data/raw/.
# Uso:
#   sbatch --partition=standard slurm/01_preprocess.slurm
# Opcional:
#   PROJECT_DIR=/ruta/ml-project CONDA_ENV=ml-unsw sbatch slurm/01_preprocess.slurm
#   LIMITE_POR_CATEGORIA=50000 sbatch slurm/01_preprocess.slurm

#SBATCH --job-name=unsw-preprocess
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# La cuenta de pregrado permite hasta 32 cores, 98 GB y 8 horas por job.
# Se solicita margen bajo el límite de memoria.
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j-preprocess.out
#SBATCH --error=slurm-%j-preprocess.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"
LIMITE_POR_CATEGORIA="${LIMITE_POR_CATEGORIA:-100000}"

cd "$PROJECT_DIR"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/results/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SLURM_JOB_NAME:-unsw-preprocess}-${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log unificado: $LOG_FILE"

# Lmod en Khipu referencia LD_PRELOAD durante su inicialización y este job
# usa `set -u`.
export LD_PRELOAD="${LD_PRELOAD-}"
module purge
module load miniconda/3.0 >/dev/null 2>&1 || true

if ! command -v conda >/dev/null 2>&1; then
    export PATH="${CONDA_INSTALL_DIR:-${HOME}/.local/miniforge3}/bin:$PATH"
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: no se encontró conda. Ejecute setup.sh primero." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [[ ! -d data/raw ]]; then
    echo "ERROR: falta data/raw/. Prepare el dataset antes de enviar el job." >&2
    exit 1
fi

echo "Iniciando preprocesamiento en $(hostname)"
echo "Límite de train por categoría: $LIMITE_POR_CATEGORIA"
srun --ntasks=1 python notebooks/pre-process.py \
    --limite-por-categoria "$LIMITE_POR_CATEGORIA"
echo "Preprocesamiento completado."
