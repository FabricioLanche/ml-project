#!/bin/bash
# Verifica el entorno antes de reservar recursos para una corrida completa.
# Uso: sbatch --partition=debug slurm/00_verificar_entorno.slurm

#SBATCH --job-name=unsw-check
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j-check.out
#SBATCH --error=slurm-%j-check.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-ml-unsw}"

cd "$PROJECT_DIR"
for required in \
    requirements.txt \
    notebooks/common.py \
    notebooks/train.py \
    notebooks/k-foldCV.py
do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: falta el archivo requerido: $PROJECT_DIR/$required" >&2
        exit 1
    fi
done

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

echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Python: $(python --version)"
echo "Executable: $(command -v python)"

python - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit("ERROR: se requiere Python 3.12 o superior.")
print("Python compatible.")
PY

if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: Java no está disponible en el entorno." >&2
    exit 1
fi

java_version="$(java -version 2>&1 | sed -nE 's/.*version "([0-9]+)(\.[0-9]+)?.*/\1/p' | head -n 1)"
if [[ ! "$java_version" =~ ^[0-9]+$ ]] || (( java_version < 17 )); then
    echo "ERROR: se requiere Java 17 o superior; versión detectada: ${java_version:-desconocida}." >&2
    exit 1
fi
echo "Java compatible:"
java -version

python - <<'PY'
import importlib
import pandas
import numpy
import pyarrow
import pyspark

print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("pyarrow:", pyarrow.__version__)
print("pyspark:", pyspark.__version__)

for module_name in ("pandas", "numpy", "pyarrow", "pyspark"):
    importlib.import_module(module_name)
print("Imports de dependencias: OK")

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("unsw-environment-check")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
assert spark.range(2).count() == 2
spark.stop()
print("Prueba Spark local: OK")
PY

python -m pip check
python -m py_compile notebooks/common.py notebooks/train.py notebooks/k-foldCV.py notebooks/pre-process.py
echo "Dependencias declaradas: consistentes."

if [[ -d data/raw ]]; then
    echo "Dataset raw: data/raw/ presente."
else
    echo "ADVERTENCIA: data/raw/ no existe; aún no se puede ejecutar 01_preprocess.slurm."
fi

if [[ -f data/preprocessed/base/metadatos.json &&
      -f data/preprocessed/base/test.parquet &&
      -f data/preprocessed/grid/manifiesto.csv ]]; then
    echo "Datos preprocesados: base/ y grid/ presentes."
else
    echo "ADVERTENCIA: faltan datos preprocesados; ejecute 01_preprocess.slurm antes de 02_kfoldcv.slurm."
fi

echo "Entorno Python, Java y Spark verificado correctamente."
