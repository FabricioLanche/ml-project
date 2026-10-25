#!/bin/bash
# Materializa base/ y grid/ a partir de data/raw/.
# Uso:
#   sbatch --partition=big-mem slurm/01_preprocess.slurm
# Opcional:
#   PROJECT_DIR=/ruta/ml-project CONDA_ENV=ml-unsw sbatch slurm/01_preprocess.slurm

#SBATCH --job-name=unsw-preprocess
#SBATCH --partition=big-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j-preprocess.out
#SBATCH --error=slurm-%j-preprocess.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"

cd "$PROJECT_DIR"
# Lmod en Khipu referencia LD_PRELOAD durante su inicialización y este job
# usa `set -u`.
export LD_PRELOAD="${LD_PRELOAD-}"
module purge
module load miniconda/3.0 >/dev/null 2>&1 || true

if ! command -v conda >/dev/null 2>&1; then
    export PATH="${CONDA_INSTALL_DIR:-${HOME}/.local/miniforge3}/bin:$PATH"
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: no se encontró conda. Ejecute preparar_entorno.sh primero." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [[ ! -d data/raw ]]; then
    echo "ERROR: falta data/raw/. Prepare el dataset antes de enviar el job." >&2
    exit 1
fi

echo "Iniciando preprocesamiento en $(hostname)"
srun --ntasks=1 python notebooks/pre-process.py
echo "Preprocesamiento completado."
