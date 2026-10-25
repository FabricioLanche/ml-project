#!/bin/bash
# Ejecuta fase 1, fase 2 y fase 3 final del orquestador.
# Uso:
#   sbatch --partition=big-mem slurm/02_kfoldcv.slurm
# Opcional:
#   ITERACIONES=1 PROJECT_DIR=/ruta/ml-project CONDA_ENV=ml-unsw \
#     sbatch slurm/02_kfoldcv.slurm

#SBATCH --job-name=unsw-kfold
#SBATCH --partition=big-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j-kfold.out
#SBATCH --error=slurm-%j-kfold.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"
ITERACIONES="${ITERACIONES:-1}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-32}"
DRIVER_MEMORY="${DRIVER_MEMORY:-100g}"

cd "$PROJECT_DIR"
module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

for required in \
    data/preprocessed/base/metadatos.json \
    data/preprocessed/base/test.parquet \
    data/preprocessed/grid/manifiesto.csv
do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: falta $required. Ejecute primero 01_preprocess.slurm." >&2
        exit 1
    fi
done

echo "Iniciando k-fold CV en $(hostname)"
echo "Iteraciones: $ITERACIONES"
echo "Driver memory: $DRIVER_MEMORY"

srun --ntasks=1 python notebooks/k-foldCV.py \
    --iteraciones "$ITERACIONES" \
    --driver-memory "$DRIVER_MEMORY" \
    --shuffle-partitions "$SHUFFLE_PARTITIONS"

echo "k-fold CV y evaluación final completados."
