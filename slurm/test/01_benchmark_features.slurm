#!/bin/bash
# Benchmark corto: una celda, un fold y un limite de filas.
# Uso desde el nodo de acceso:
#   sbatch slurm/test/01_benchmark_features.slurm

#SBATCH --job-name=unsw-bench
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j-benchmark.out
#SBATCH --error=slurm-%j-benchmark.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"
LIMIT="${LIMIT:-5000}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-16}"
DRIVER_MEMORY="${DRIVER_MEMORY:-8g}"

cd "$PROJECT_DIR"
module purge
module load miniconda/3.0 >/dev/null 2>&1 || true
if ! command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="${CONDA_INSTALL_DIR:-$HOME/.local/miniforge3}"
    export PATH="$CONDA_ROOT/bin:$PATH"
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: no se encontro conda. Ejecute setup.sh primero." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

srun --ntasks=1 python -u slurm/test/benchmark_features.py \
    --limit "$LIMIT" \
    --master "local[$SLURM_CPUS_PER_TASK]" \
    --driver-memory "$DRIVER_MEMORY" \
    --shuffle-partitions "$SHUFFLE_PARTITIONS"
