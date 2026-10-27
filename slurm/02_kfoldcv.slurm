#!/bin/bash
# Ejecuta fase 1, fase 2 y fase 3 final del orquestador.
# Uso:
#   sbatch --partition=standard slurm/02_kfoldcv.slurm
# Opcional:
#   ITERACIONES=1 PROJECT_DIR=/ruta/ml-project CONDA_ENV=ml-unsw \
#     sbatch slurm/02_kfoldcv.slurm
#   FASES="1" ITERACION=1 sbatch slurm/02_kfoldcv.slurm
#   FASES="2" ITERACION=1 REANUDAR=1 sbatch slurm/02_kfoldcv.slurm
#   FASES="1 2 3" ITERACION=1 FORZAR=1 sbatch slurm/02_kfoldcv.slurm

#SBATCH --job-name=unsw-kfold
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
# La cuenta de pregrado permite hasta 32 cores, 98 GB y 8 horas por job.
# Se solicitan 32 cores para que Spark local[*] use el nodo completo.
#SBATCH --cpus-per-task=32
# Se solicita margen bajo el límite de memoria.
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j-kfold.out
#SBATCH --error=slurm-%j-kfold.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"
ITERACIONES="${ITERACIONES:-1}"
ITERACION="${ITERACION:-}"
FASES="${FASES:-1 2 3}"
REANUDAR="${REANUDAR:-0}"
FORZAR="${FORZAR:-0}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-64}"
# Debe quedar por debajo de los 96G asignados por Slurm para dejar margen
# al proceso Python y al overhead de Spark.
DRIVER_MEMORY="${DRIVER_MEMORY:-72g}"

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
    echo "ERROR: no se encontró conda. Ejecute setup.sh primero." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

for required in \
    data/preprocessed/base/metadatos.json \
    data/preprocessed/base/test.parquet \
    data/preprocessed/grid_preprocess/manifiesto.csv \
    data/preprocessed/grid_hyperparams/hiperparametros.json
do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: falta $required. Ejecute primero 01_preprocess.slurm." >&2
        exit 1
    fi
done

echo "Iniciando k-fold CV en $(hostname)"
echo "Fases: $FASES"
echo "Iteraciones: ${ITERACION:-$ITERACIONES}"
echo "Reanudar: $REANUDAR · Forzar: $FORZAR"
echo "Driver memory: $DRIVER_MEMORY"

args=(--fase $FASES --driver-memory "$DRIVER_MEMORY"
      --shuffle-partitions "$SHUFFLE_PARTITIONS")
if [[ -n "$ITERACION" ]]; then
    args+=(--iteracion "$ITERACION")
else
    args+=(--iteraciones "$ITERACIONES")
fi
if [[ "$FORZAR" == "1" ]]; then
    args+=(--forzar)
elif [[ "$REANUDAR" == "1" ]]; then
    args+=(--reanudar)
fi

srun --ntasks=1 python -u notebooks/k-foldCV.py "${args[@]}"

echo "k-fold CV y evaluación final completados."
