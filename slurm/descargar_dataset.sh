#!/usr/bin/env bash
# Descarga el dataset desde Kaggle en el nodo de acceso.
# Uso:
#   bash slurm/descargar_dataset.sh
# Opcional:
#   PROJECT_DIR=/ruta/ml-project CONDA_ENV=ml-unsw \
#     bash slurm/descargar_dataset.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"
KAGGLE_DATASET="${KAGGLE_DATASET:-harshwardhanbhangale/unsw-complete-dataset}"
DATA_DIR="$PROJECT_DIR/data/raw"

cd "$PROJECT_DIR"
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
mkdir -p "$DATA_DIR"

required_files=()
for index in 1 2 3 4; do
    required_files+=("UNSW-NB15_${index}.csv")
done

missing=0
for file in "${required_files[@]}"; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        missing=1
        break
    fi
done

if [[ "$missing" -eq 0 ]]; then
    echo "Dataset ya presente en $DATA_DIR"
    exit 0
fi

echo "Descargando '$KAGGLE_DATASET' desde el nodo de acceso..."
download_dir="$(mktemp -d "${TMPDIR:-/tmp}/unsw-kaggle.XXXXXX")"
cleanup() {
    rm -rf "$download_dir"
}
trap cleanup EXIT

DOWNLOAD_DIR="$download_dir" DATA_DIR="$DATA_DIR" KAGGLE_DATASET="$KAGGLE_DATASET" \
python -c '
import os
import shutil
from pathlib import Path

import kagglehub

source_dir = Path(kagglehub.dataset_download(os.environ["KAGGLE_DATASET"]))
target_dir = Path(os.environ["DATA_DIR"])
for index in range(1, 5):
    name = f"UNSW-NB15_{index}.csv"
    source = source_dir / name
    if not source.is_file():
        raise FileNotFoundError(f"No se encontró {name} en {source_dir}")
    shutil.copy2(source, target_dir / name)
    print(f"  -> {target_dir / name}")
'

for file in "${required_files[@]}"; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "ERROR: la descarga no dejó el archivo esperado: $DATA_DIR/$file" >&2
        exit 1
    fi
done

echo "Dataset disponible en $DATA_DIR"
