#!/bin/bash
# Ejecuta fase 1, fase 2 y fase 3 final del orquestador.
# Uso:
#   sbatch --partition=standard slurm/02_kfoldcv.slurm
# Opcional:
#   ITERACIONES=1 PROJECT_DIR=/ruta/ml-project CONDA_ENV=ml-unsw \
#     sbatch slurm/02_kfoldcv.slurm
#   FASES="1" ITERACION=1 sbatch slurm/02_kfoldcv.slurm
#   FASES="2" ITERACION=1 REANUDAR=1 sbatch slurm/02_kfoldcv.slurm
#   FASES="1" ITERACION=1 CELDA=G03 sbatch slurm/02_kfoldcv.slurm
#   FASES="1" ITERACION=1 MODELOS="logreg rf" RONDAS="1 3" sbatch slurm/02_kfoldcv.slurm

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
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-64}"
# Debe quedar por debajo de los 96G asignados por Slurm para dejar margen
# al proceso Python y al overhead de Spark.
DRIVER_MEMORY="${DRIVER_MEMORY:-72g}"
CELDA="${CELDA:-}"
MODELOS="${MODELOS:-}"
RONDAS="${RONDAS:-}"

cd "$PROJECT_DIR"
echo "Proyecto: $PROJECT_DIR"
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

read -r -a fases_array <<< "$FASES"
required_files=(
    data/preprocessed/grid_preprocess/manifiesto.csv
    data/preprocessed/grid_hyperparams/hiperparametros.json
)
for fase in "${fases_array[@]}"; do
    case "$fase" in
        1|2)
            for celda in {00..03}; do
                for ronda in {1..5}; do
                    required_files+=(
                        "data/preprocessed/grid_preprocess/G$celda/r$ronda/train.parquet"
                        "data/preprocessed/grid_preprocess/G$celda/r$ronda/val.parquet"
                    )
                done
            done
            ;;
        3)
            required_files+=(
                data/preprocessed/base/metadatos.json
                data/preprocessed/base/test.parquet
            )
            for fold in {1..5}; do
                required_files+=("data/preprocessed/base/fold_$fold.parquet")
            done
            ;;
    esac
done
missing_files=()
for required in "${required_files[@]}"; do
    if [[ ! -s "$required" ]]; then
        missing_files+=("$required")
    fi
done
if (( ${#missing_files[@]} > 0 )); then
    echo "ERROR: faltan artefactos de preprocesamiento en '$PROJECT_DIR':" >&2
    printf '  - %s\n' "${missing_files[@]}" >&2
    echo "Artefactos requeridos para FASES='$FASES'. La fase 1 necesita la rejilla materializada; la fase 3 necesita base/." >&2
    echo "Revise los logs slurm-<JOB_PREPROCESS>-preprocess.out y .err, y el estado con: sacct -j <JOB_PREPROCESS> --format=JobID,State,ExitCode" >&2
    exit 1
fi

echo "Iniciando k-fold CV en $(hostname)"
echo "Fases: $FASES"
echo "Iteraciones: ${ITERACION:-$ITERACIONES}"
echo "Reanudar: $REANUDAR"
echo "Driver memory: $DRIVER_MEMORY"

args=(--fase "${fases_array[@]}" --driver-memory "$DRIVER_MEMORY"
      --shuffle-partitions "$SHUFFLE_PARTITIONS")
if [[ -n "$CELDA" ]]; then
    args+=(--celdas "$CELDA")
fi
if [[ -n "$MODELOS" ]]; then
    read -r -a modelos_array <<< "$MODELOS"
    args+=(--modelos "${modelos_array[@]}")
fi
if [[ -n "$RONDAS" ]]; then
    read -r -a rondas_array <<< "$RONDAS"
    args+=(--rondas "${rondas_array[@]}")
fi
if [[ -n "$ITERACION" ]]; then
    args+=(--iteracion "$ITERACION")
else
    args+=(--iteraciones "$ITERACIONES")
fi
if [[ "$REANUDAR" == "1" ]]; then
    args+=(--reanudar)
fi

srun --ntasks=1 python -u notebooks/k-foldCV.py "${args[@]}"

echo "k-fold CV y evaluación final completados."
