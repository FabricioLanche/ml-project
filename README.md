# Detección multiclase de ataques de red con UNSW-NB15

Proyecto del curso de Machine Learning (CS3061, UTEC) para analizar y clasificar flujos de red en diez categorías: tráfico benigno y nueve familias de ataques.

## Objetivo

Comparar modelos supervisados para la clasificación multiclase de UNSW-NB15, a partir de un análisis exploratorio, una búsqueda de variantes de preprocesamiento y el ajuste de hiperparámetros.

## Integrantes

- Fabricio Alonso Lanche Pacsi
- Jhogan Haldo Pachacutec Aguilar
- Anyeli Azumi Tamara Ureta
- Paulo Isael Miranda Barrientos
- Christofer Renato Perez Torres

## Estructura del repositorio

```text
ml-project/
├── README.md
├── requirements.txt
├── docs/
│   ├── P1.tex                    # Informe de análisis exploratorio
│   └── P2.tex                    # Informe de metodología y resultados
├── src/
│   ├── eda.py                    # Análisis exploratorio; genera tablas y figuras
│   ├── pre-process.py            # Limpieza, partición y materialización de datos
│   ├── pipeline-train.py         # Búsqueda y entrenamiento con Spark MLlib
│   ├── result-process.py         # Exportación de métricas y figuras finales
│   └── utilities/
│       ├── preprocessing-config.py # Variantes de preprocesamiento
│       ├── hyperparams-config.py   # Parámetros base y grillas
│       ├── training-utilities.py   # Pipelines, modelos y evaluación
│       └── training-functions.py   # Sesión Spark y operaciones de entrenamiento
├── data/
│   ├── raw/                      # CSV originales descargados de Kaggle
│   ├── eda/
│   │   ├── tablas/               # Tablas CSV del EDA
│   │   └── figs/                 # Figuras del EDA
│   └── preprocessed/             # Particiones y variantes generadas; ignoradas por Git
└── results/
    ├── iteracion_N/              # Resultados de fases 1 y 2 y checkpoints
    ├── fase_3/                   # Modelos y métricas sobre el conjunto de prueba
    └── figs/                     # Figuras y tablas exportadas para el informe
```

## Dependencias

Las versiones utilizadas están fijadas en `requirements.txt`.

| Paquete | Uso |
|---|---|
| `pandas`, `numpy` | Carga, limpieza y transformación de datos |
| `matplotlib` | Figuras del EDA y de resultados |
| `kagglehub` | Descarga de UNSW-NB15 desde Kaggle |
| `pyarrow` | Escritura y lectura de archivos Parquet |
| `pyspark` | Entrenamiento y evaluación con Spark MLlib |

## Requisitos

- Python 3.12 o superior, por la versión de NumPy fijada en `requirements.txt`.
- Java 17, 21 o 25 para la versión de Spark fijada. Compruebe la versión disponible con `java -version`.
- Espacio en disco y memoria suficientes para los CSV originales, los Parquet intermedios y el entrenamiento. La búsqueda completa puede tardar y usar bastantes recursos; por defecto Spark utiliza todos los procesadores disponibles (`local[*]`).

## Reproducción

Clone el repositorio y entre a su carpeta (si aún no lo hizo):

```bash
git clone https://github.com/FabricioLanche/ml-project.git
cd ml-project
```

Luego cree un entorno e instale las dependencias.

**Linux o macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

El análisis y el preprocesamiento buscan los cuatro archivos `UNSW-NB15_1.csv` a `UNSW-NB15_4.csv` en `data/raw/`. Si no están, intentan descargarlos desde Kaggle mediante `kagglehub`; también puede colocarlos allí manualmente. Luego ejecute los pasos en orden:

```bash
# 1. Análisis exploratorio: data/eda/tablas/ y data/eda/figs/
python src/eda.py

# 2. Split cronológico, folds y grillas: data/preprocessed/
python src/pre-process.py

# 3. Fases 1 y 2 de validación y fase 3 de prueba: results/
python src/pipeline-train.py

# 4. Tablas y figuras finales para el informe: results/figs/
python src/result-process.py
```

El preprocesamiento deduplica antes de la partición, reserva el 15 % más reciente para prueba y crea cinco folds temporales sobre el 85 % de entrenamiento. Por defecto, limita a 100 000 filas por categoría cada conjunto de entrenamiento de fold; la validación y la prueba se mantienen completas. Para cambiar ese límite, vuelva a generar los datos antes de entrenar:

```bash
python src/pre-process.py --limite-por-categoria 50000
```

Use `--limite-por-categoria 0` para desactivar el límite. Las opciones de entrenamiento, incluida la selección de fases, modelos, memoria y número de procesadores locales, se consultan con:

```bash
python src/pipeline-train.py --help
```

Por ejemplo, para limitar Spark a cuatro procesadores:

```bash
python src/pipeline-train.py --master 'local[4]' --shuffle-partitions 8
```

`src/eda.py` y `src/pre-process.py` regeneran sus salidas al ejecutarse. `src/pipeline-train.py` usa los artefactos de `data/preprocessed/`; ejecútelo después del preprocesamiento. `src/result-process.py` requiere que la fase 3 haya terminado.

## Dataset

Se utiliza la versión completa de UNSW-NB15 publicada en Kaggle, compuesta por cuatro CSV con 2 540 047 registros y 49 columnas. Los archivos originales se guardan en `data/raw/`, carpeta excluida del control de versiones.
