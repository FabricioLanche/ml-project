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
├── output/
│   ├── *.csv             # Tablas generadas
│   ├── fig*.png          # Figuras del análisis exploratorio
│   └── preprocessed/     # Particiones train/val/test en parquet + metadatos.json
├── requirements.txt      # Dependencias del entorno
└── dataset/              # Datos UNSW-NB15 (no versionado, ver sección Dataset)
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

2. Ejecutar el análisis exploratorio (genera las tablas CSV y figuras en `output/`):

   - **Notebook**: abrir `notebooks/eda.ipynb` en Jupyter / VSCode y ejecutar todas las celdas. Asegurarse de que el kernel use el intérprete del entorno activado (en VSCode: `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir el del `.venv`). Si el kernel usa otro Python, fallará la importación de los paquetes.
   - **Script**: equivalente al notebook, usado durante el desarrollo y debugging.

     ```bash
     python3 notebooks/eda.py
     ```

3. Ejecutar el preprocesamiento. Genera las particiones en `output/preprocessed/` y ejecuta
   seis verificaciones de integridad (presencia de las diez clases, alineación de los cortes
   temporales, ausencia de solapamiento de la ventana `ct_*` entre particiones y ausencia de
   filas idénticas repartidas entre conjuntos).

   ```bash
   python3 notebooks/pre-process.py
   ```

### Salidas del preprocesamiento

| Archivo | Contenido |
|---|---|
| `output/preprocessed/{train,val,test}.parquet` | 45 columnas predictoras, sin duplicados eliminados |
| `output/preprocessed/metadatos.json` | Cortes temporales, modas imputadas, variables transformadas y decisiones aplicadas |
| `output/preprocesamiento_columnas.csv` | Acción aplicada a cada columna y su motivo |
| `output/preprocesamiento_log1p.csv` | Asimetría por variable antes y después de `log1p` |
| `output/preprocesamiento_tabla_clases.csv` | Recuento de cada categoría en cada partición |
| `output/duplicados_por_clase.csv` | Concentración de filas duplicadas por categoría |

El formato parquet es obligatorio: preserva los tipos `Int8`, `Int16`, `Float32` y `boolean`
de Pandas, que un CSV destruiría al releerlos.

## Dataset

UNSW-NB15 (Moustafa y Slay, 2015): tráfico de red capturado en el Cyber Range Lab de la ACCS. Consta de 4 archivos CSV con 2,540,047 registros y 49 columnas.

La primera ejecución del análisis exploratorio descarga el dataset desde Kaggle (`harshwardhanbhangale/unsw-complete-dataset`) a la carpeta `dataset/`. Esta carpeta está excluida del repositorio (`.gitignore`).

## Decisiones de preprocesamiento

El detalle argumentado de cada decisión, con los datos que la motivan, se encuentra en
[`.agents/PROPUESTA_PREPROCESAMIENTO.md`](.agents/PROPUESTA_PREPROCESAMIENTO.md). En resumen:

| Decisión | Motivo |
|---|---|
| Se conservan los duplicados | El 88 % de `Generic` y el 65 % de `DoS` son duplicados; eliminarlos llevaría `Benign` del 87.35 % al 95.16 % y destruiría la clase más numerosa de ataque |
| Blancos estructurales se imputan a 0 | El blanco codifica el archivo de origen (0 %, 43.86 %, 98.49 %, 98.49 %), no el protocolo del flujo |
| Puertos se derivan a rangos | `Benign` usa 64 581 puertos distintos y `Analysis` solo 3: el valor crudo permite memorizar la asociación puerto-clase |
| `Stime` y `Ltime` se eliminan | Describen el instante de la captura, no el comportamiento del flujo |
| Partición cronológica 75/15/10 | Con partición aleatoria, el contexto de la ventana `ct_*` de una fila de prueba queda dentro de train |