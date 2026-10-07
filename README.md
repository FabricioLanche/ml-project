# Detección de ataques de red multiclase sobre UNSW-NB15: análisis exploratorio y evaluación de modelos de Machine Learning

Proyecto de Machine Learning (CS3061, UTEC) para la detección de ataques de red multiclase sobre el dataset UNSW-NB15.

## Objetivo

Desarrollar un clasificador de tráfico de red que distinga el tráfico normal (`Benign`) de 10 familias de ataques, mediante análisis exploratorio de datos y la implementación de modelos de Machine Learning.

## Integrantes

- Fabricio Alonso Lanche Pacsi
- Jhogan Haldo Pachacutec Aguilar
- Anyeli Azumi Tamara Ureta
- Paulo Ismael Miranda Barrientos
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
│   ├── fase1_preproc/
│   ├── fase2_hiperparams/
│   └── fase3_test/
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

- **Python 3.11 o superior** (recomendado: Python 3.12). Las versiones fijadas en `requirements.txt` (`pandas` 3.0.x, `numpy` 2.5.x) solo están disponibles para Python ≥ 3.11. Verificar: `python3 --version` en Linux o `py --version` en Windows.
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

### Salidas del preprocesamiento

| Archivo | Contenido |
|---|---|
| `data/preprocessed/base/fold_1..5.parquet` | Particiones del 85 % de entrenamiento, features sin transformar |
| `data/preprocessed/base/test.parquet` | 15 % de prueba, sellado hasta la fase 3 |
| `data/preprocessed/base/metadatos.json` | Cortes temporales, decisiones de limpieza y codificación del objetivo |
| `data/preprocessed/grid/<celda>/r1..r5/` | Variante de cada celda de la rejilla, con `train` y `val` por ronda |
| `data/preprocessed/grid/manifiesto.csv` | Receta que define cada celda (`G00`…`G14`) |

El formato parquet es obligatorio: preserva los tipos `Int8`, `Int16`, `Float32` y `boolean`
de Pandas, que un CSV destruiría al releerlos. Los detalles del esquema de entrenamiento
están en [`.agents/PROPUESTA_ENTRENAMIENTO.md`](.agents/PROPUESTA_ENTRENAMIENTO.md).

## Dataset

UNSW-NB15 (Moustafa y Slay, 2015): tráfico de red capturado en el Cyber Range Lab de la ACCS. Consta de 4 archivos CSV con 2,540,047 registros y 49 columnas.

La primera ejecución del análisis exploratorio descarga el dataset desde Kaggle (`harshwardhanbhangale/unsw-complete-dataset`) a la carpeta `data/raw/`. Esta carpeta está excluida del repositorio (`.gitignore`).

## Decisiones de preprocesamiento

El detalle argumentado de cada decisión, con los datos que la motivan, se encuentra en
[`.agents/PROPUESTA_PREPROCESAMIENTO.md`](.agents/PROPUESTA_PREPROCESAMIENTO.md). En resumen:

| Decisión | Motivo |
|---|---|
| Se conservan los duplicados | El 88 % de `Generic` y el 65 % de `DoS` son duplicados; eliminarlos llevaría `Benign` del 87.35 % al 95.16 % y destruiría la clase más numerosa de ataque |
| Blancos estructurales se imputan a 0 | El blanco codifica el archivo de origen (0 %, 43.86 %, 98.49 %, 98.49 %), no el protocolo del flujo |
| Puertos se derivan a rangos | `Benign` usa 64 581 puertos distintos y `Analysis` solo 3: el valor crudo permite memorizar la asociación puerto-clase |
| `Stime` y `Ltime` se eliminan | Describen el instante de la captura, no el comportamiento del flujo |
| Split cronológico 85/15 + K-Fold | Con partición aleatoria, el contexto de la ventana `ct_*` de una fila de prueba queda dentro de train; el 15 % final queda sellado y la búsqueda usa K-Fold sobre el 85 % |