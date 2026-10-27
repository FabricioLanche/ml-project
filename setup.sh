#!/bin/bash
# Prepara el entorno Conda y descarga el dataset si todavía no está disponible.
# Uso desde el nodo de acceso: bash setup.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$PROJECT_DIR/requirements.txt}"
CONDA_INSTALL_DIR="${CONDA_INSTALL_DIR:-${HOME}/.local/miniforge3}"
MINIFORGE_VERSION="${MINIFORGE_VERSION:-24.11.3-2}"

cd "$PROJECT_DIR"
# Lmod en Khipu referencia LD_PRELOAD durante su inicialización. Con
# `set -u`, la variable debe existir aunque no se necesite un preload.
export LD_PRELOAD="${LD_PRELOAD-}"
module purge
if ! module load miniconda/3.0 >/dev/null 2>&1; then
    echo "ADVERTENCIA: no se pudo cargar el módulo miniconda/3.0."
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda no está disponible. Instalando Miniforge en '$CONDA_INSTALL_DIR'..."

    case "$(uname -m)" in
        x86_64) miniforge_arch="x86_64" ;;
        aarch64|arm64) miniforge_arch="aarch64" ;;
        *)
            echo "ERROR: arquitectura no soportada para Miniforge: $(uname -m)" >&2
            exit 1
            ;;
    esac

    if [[ -e "$CONDA_INSTALL_DIR" && ! -x "$CONDA_INSTALL_DIR/bin/conda" ]]; then
        echo "ERROR: '$CONDA_INSTALL_DIR' existe, pero no contiene una instalación válida de Conda." >&2
        echo "Defina CONDA_INSTALL_DIR con una ruta vacía o válida e inténtelo de nuevo." >&2
        exit 1
    fi

    miniforge_file="Miniforge3-Linux-${miniforge_arch}.sh"
    miniforge_url="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${miniforge_file}}"
    installer_path="${TMPDIR:-/tmp}/${miniforge_file}.$$"

    if [[ ! -x "$CONDA_INSTALL_DIR/bin/conda" ]]; then
        if command -v curl >/dev/null 2>&1; then
            curl --fail --location --retry 3 --output "$installer_path" "$miniforge_url"
        elif command -v wget >/dev/null 2>&1; then
            wget --https-only --tries=3 --output-document="$installer_path" "$miniforge_url"
        else
            echo "ERROR: se requiere curl o wget para descargar Miniforge." >&2
            exit 1
        fi

        bash "$installer_path" -b -p "$CONDA_INSTALL_DIR"
        rm -f "$installer_path"
    fi

    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: no se pudo disponer de conda mediante el módulo ni Miniforge." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "ERROR: no existe el archivo de dependencias: $REQUIREMENTS_FILE" >&2
    exit 1
fi

if conda env list | awk -v env="$CONDA_ENV" '$1 == env { found = 1 } END { exit !found }'; then
    echo "El entorno Conda '$CONDA_ENV' ya existe."
else
    echo "Creando el entorno Conda '$CONDA_ENV'..."
    conda create --name "$CONDA_ENV" python=3.12 -y
fi

conda activate "$CONDA_ENV"
python -m pip install --upgrade pip
python -m pip install --requirement "$REQUIREMENTS_FILE"

java_major_version() {
    local version
    version="$(java -version 2>&1 | sed -nE 's/.*version "([0-9]+)(\.[0-9]+)?.*/\1/p' | head -n 1)"
    [[ "$version" =~ ^[0-9]+$ ]] && printf '%s\n' "$version" || return 1
}

java_is_compatible() {
    local major
    command -v java >/dev/null 2>&1 || return 1
    major="$(java_major_version)" || return 1
    (( major >= 17 ))
}

if java_is_compatible; then
    echo "Java compatible disponible:"
    java -version
else
    echo "No se encontró Java 17 o superior. Buscando módulos Java..."
    module_java_found=0
    for java_module in java/17 jdk/17 openjdk/17; do
        if module load "$java_module" >/dev/null 2>&1 && java_is_compatible; then
            echo "Java cargado desde el módulo '$java_module':"
            java -version
            module_java_found=1
            break
        fi
    done

    if (( module_java_found == 0 )); then
        echo "No se encontró un módulo Java compatible."
    else
        echo "El módulo solo está disponible en esta sesión; se instalará Java en Conda para los jobs."
    fi
    if (( module_java_found == 0 )) || ! [[ "${JAVA_HOME:-}" == "$CONDA_PREFIX"* ]]; then
        echo "Instalando OpenJDK 17 dentro del entorno Conda '$CONDA_ENV'..."
        conda install --channel conda-forge openjdk=17 -y
        hash -r
    fi
fi

if ! java_is_compatible; then
    echo "ERROR: no se pudo disponer de Java 17 o superior." >&2
    exit 1
fi

echo "Python: $(python --version)"
echo "Ejecutable: $(command -v python)"
echo "Java:"
java -version

python - <<'PY'
import numpy
import pandas
import pyarrow
import pyspark

print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("pyarrow:", pyarrow.__version__)
print("pyspark:", pyspark.__version__)
PY

DATA_DIR="$PROJECT_DIR/data/raw"
KAGGLE_DATASET="${KAGGLE_DATASET:-harshwardhanbhangale/unsw-complete-dataset}"
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
    echo "Dataset ya presente en '$DATA_DIR'."
else
    echo "Descargando '$KAGGLE_DATASET' desde el nodo de acceso..."
    download_dir="$(mktemp -d "${TMPDIR:-/tmp}/unsw-kaggle.XXXXXX")"
    cleanup() {
        rm -rf "$download_dir"
    }
    trap cleanup EXIT

    DATA_DIR="$DATA_DIR" KAGGLE_DATASET="$KAGGLE_DATASET" \
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
fi

for file in "${required_files[@]}"; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "ERROR: falta el archivo esperado tras la preparación: $DATA_DIR/$file" >&2
        exit 1
    fi
done

echo "Entorno '$CONDA_ENV' preparado correctamente."
echo "Dataset disponible en '$DATA_DIR'."
