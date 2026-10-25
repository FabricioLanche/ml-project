# Detección de ataques de red multiclase sobre UNSW-NB15: análisis exploratorio y evaluación de modelos de Machine Learning

Proyecto de Machine Learning (CS3061, UTEC) para la detección de ataques de red multiclase sobre el dataset UNSW-NB15.

## Objetivo

Desarrollar un clasificador de tráfico de red que distinga el tráfico normal (`Benign`) de 10 familias de ataques, mediante análisis exploratorio de datos y la implementación de modelos de Machine Learning.

## Integrantes

- Fabricio Alonso Lanche Pacsi
- Jhogan Haldo Pachacutec Aguilar
- Anyeli Azumi Tamara Ureta
- Paulo Isael Miranda Barrientos
- Christofer Renato Perez Torres

## Estructura del repositorio

```
ml-project/
├── notebooks/
│   ├── eda.ipynb         # Análisis exploratorio de datos (principal)
│   ├── eda.py            # Versión en script (desarrollo y debugging)
│   ├── pre-process.ipynb # Preprocesamiento e ingeniería de características
│   └── pre-process.py    # Versión en script (desarrollo y debugging)
├── data/                 # Datos generados (no versionado)
│   ├── raw/              # CSVs originales de UNSW-NB15
│   ├── eda/              # Salidas del EDA (no del entrenamiento)
│   │   ├── tablas/       # Tablas *.csv
│   │   └── figuras/      # Figuras *.png
│   └── preprocessed/     # Entrada del flujo de entrenamiento
│       ├── base/         # Split 85/15 + folds, features sin transformar
│       └── grid/         # Variantes de preprocesamiento por celda
├── results/              # Salidas del entrenamiento (métricas, modelos)
│   ├── iteracion_N/
│   │   ├── fase_1/
│   │   └── fase_2/
│   └── fase_3/            # Evaluación final sobre test
├── slurm/                # Jobs para ejecutar el flujo en Khipu
│   ├── 00_verificar_entorno.slurm
│   ├── 01_preprocess.slurm
│   └── 02_kfoldcv.slurm
├── docs/                 # Documento del informe (LaTeX)
├── requirements.txt      # Dependencias del entorno
└── .agents/              # Documentos de trabajo internos
```

## Dependencias

| Paquete | Uso |
|---|---|
| `pandas`, `numpy` | Carga, limpieza y transformación |
| `matplotlib` | Figuras del EDA |
| `kagglehub` | Descarga del dataset desde Kaggle |
| `pyarrow` | Lectura y escritura de las particiones en parquet |
| `pyspark` | Etapa de modelado (MLlib), requiere Java 17 |

El entrenamiento con PySpark es la vía recomendada para la etapa de modelado: procesa las
particiones de forma particionada en disco, de modo que el conjunto de 2.5M filas no
necesita residir íntegro en memoria.

## Requisitos

- **Python 3.12 o superior**. Las versiones fijadas en `requirements.txt` (`pandas` 3.0.x, `numpy` 2.5.x) requieren Python ≥ 3.12: `numpy` 2.5.x no publica builds para 3.11. Verificar: `python3 --version` en Linux o `py --version` en Windows.
- **Java 17 o superior** para PySpark. Verificar con `java -version`.

## Reproducción

1. Instalar las dependencias (`pandas`, `numpy`, `matplotlib` y `kagglehub`). El uso de un entorno virtual es **opcional pero recomendado** para no afectar el intérprete global. Lo importante es instalar las dependencias con el mismo intérprete que ejecutará el script o el notebook.

   **Linux / WSL (Ubuntu):** aquí el comando es `python3` (no `python`).

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

   **Windows (PowerShell o CMD):**

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

2. Ejecutar el análisis exploratorio (genera las tablas en `data/eda/tablas/` y las figuras en `data/eda/figuras/`):

   - **Notebook**: abrir `notebooks/eda.ipynb` en Jupyter / VSCode y ejecutar todas las celdas. Asegurarse de que el kernel use el intérprete del entorno activado (en VSCode: `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir el del `.venv`). Si el kernel usa otro Python, fallará la importación de los paquetes.
   - **Script**: equivalente al notebook, usado durante el desarrollo y debugging.

     ```bash
     python3 notebooks/eda.py
     ```

3. Ejecutar el preprocesamiento. Genera, en `data/preprocessed/`, el split cronológico 85/15
   (`base/fold_1..5.parquet` y `base/test.parquet`) y las variantes de la rejilla
   (`grid/<celda>/r1..r5/`). Además ejecuta las verificaciones de integridad (presencia de
   las diez clases, alineación de los cortes temporales, ausencia de solapamiento de la
   ventana `ct_*` entre particiones y ausencia de filas idénticas repartidas entre
   conjuntos).

   ```bash
   python3 notebooks/pre-process.py
   ```

## Ejecución en Khipu mediante Slurm

Los jobs Slurm ejecutan PySpark en un único nodo con `local[*]`, usando los CPU reservados
por Slurm. Antes de enviarlos, copie o clone el repositorio en Khipu y coloque el dataset en
`data/raw/`. Las dependencias se instalan una sola vez en el nodo de acceso, que es el
nodo con salida a Internet; los jobs de cómputo únicamente activan el entorno.

```bash
cd /ruta/al/ml-project
bash slurm/preparar_entorno.sh
```

El script crea el entorno `ml-unsw` si todavía no existe, instala las versiones fijadas en
`requirements.txt` y verifica Python, Java, `pandas`, `numpy`, `pyarrow` y `pyspark`. Para
Java, primero intenta utilizar una versión 17 o superior ya disponible y luego módulos
habituales (`java/17`, `jdk/17` u `openjdk/17`). Si no encuentra una versión compatible,
instala `openjdk=17` desde `conda-forge` dentro del entorno del curso. Si se usa otro nombre
de entorno o una ruta distinta al proyecto:

```bash
CONDA_ENV=mi-entorno PROJECT_DIR=/ruta/al/ml-project \
  bash /ruta/al/ml-project/slurm/preparar_entorno.sh
```

El script deja Java 17 o superior disponible dentro del entorno. Falla explícitamente si no
puede encontrar o instalar una versión compatible, si falta el archivo de dependencias o si
alguno de los paquetes requeridos no puede importarse.

Primero compruebe el entorno en la partición de depuración:

```bash
cd /ruta/al/ml-project
CONDA_ENV=ml-unsw sbatch --partition=debug slurm/00_verificar_entorno.slurm
```

Después ejecute los jobs en orden. El segundo debe terminar antes de enviar el tercero:

```bash
PROJECT_DIR=/ruta/al/ml-project CONDA_ENV=ml-unsw \
  sbatch slurm/01_preprocess.slurm

PROJECT_DIR=/ruta/al/ml-project CONDA_ENV=ml-unsw ITERACIONES=1 \
  sbatch slurm/02_kfoldcv.slurm
```

`01_preprocess.slurm` crea `data/preprocessed/base/` y
`data/preprocessed/grid/`. `02_kfoldcv.slurm` ejecuta las fases 1 y 2 por modelo y la
evaluación final de la fase 3. Para consultar el estado y los logs:

```bash
squeue -u "$USER"
cat slurm-<jobid>-preprocess.out
cat slurm-<jobid>-kfold.out
```

Los valores de partición, memoria, tiempo y CPU están declarados en los encabezados
`#SBATCH` y pueden sobrescribirse al enviar el job según la cuenta y las políticas vigentes
de Khipu. Los scripts fallan explícitamente si falta el entorno, el dataset o la salida del
preprocesamiento anterior.

## Dataset

UNSW-NB15 (Moustafa y Slay, 2015): tráfico de red capturado en el Cyber Range Lab de la ACCS. Consta de 4 archivos CSV con 2,540,047 registros y 49 columnas.

La primera ejecución del análisis exploratorio descarga el dataset desde Kaggle (`harshwardhanbhangale/unsw-complete-dataset`) a la carpeta `data/raw/`. Esta carpeta está excluida del repositorio (`.gitignore`).