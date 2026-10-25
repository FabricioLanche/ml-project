# Modelado con MLlib (P2): interfaz del motor para la capa de validación cruzada

Este documento describe la interfaz de `train.py`, `common.py` y `cli.py` para quien construya la
capa de validación cruzada y comparación de modelos por encima del motor. Todo lo que
sigue corresponde al código tal como está en el repo; si se cambia una firma, este
documento se actualiza en el mismo commit.

## Alcance

- `common.py`: pipeline MLlib compartido (features, clasificadores, evaluación, E/S).
  No importa `train.py` ni depende de `argparse`.
- `train.py`: motor de corridas (`construir_spark`, `correr_ronda`, `correr_suelto`,
  `correr_grid`, `correr_final`). Importable como biblioteca; no parsea argumentos de terminal.
- `cli.py`: adaptador de terminal. Parsea argumentos, llama al motor y vuelca el
  resumen `<out>/metricas.csv`. No entrena ni evalúa nada por su cuenta.
- `k-foldCV.py`: orquestador de punta a punta. Ejecuta la comparación de celdas,
  la búsqueda de hiperparámetros de cada modelo y, al terminar las iteraciones,
  el reentrenamiento final contra `base/test.parquet`.

**Lo que el motor NO hace** (y por eso existe la capa de CV):

- No agrega ni promedia métricas entre rondas: devuelve una fila por corrida y quien
  llama decide cómo promediar/comparar.
- No escribe el `metricas.csv` cuando se usa como biblioteca: eso solo lo hace el
  adaptador (`cli.py`). Desde código, el llamador arma su propio CSV con el esquema de
  la sección *Esquema de salidas*.
- No crea ni destruye sesiones de Spark: ver *Reglas de composición*.

## Orquestador de k-fold

La ejecución completa usa las cinco rondas de cada celda y mantiene un flujo
independiente para `logreg`, `rf` y `svm`:

```bash
python notebooks/k-foldCV.py --iteraciones 1 --driver-memory 4g
```

La fase 1 compara las diez celdas con los parámetros base. La fase 2 busca las
combinaciones de `--params` sobre la celda ganadora de cada modelo. Si no se
proporciona JSON, se usa una rejilla pequeña incorporada al script. El criterio de
selección es, en orden, mayor media de `promedio_F1`, mayor media de
`promedio_recall`, mayor media de `promedio_precision` y menor `DS_F1`.

`--iteraciones` repite ambas fases. La primera fase de la primera iteración usa
los parámetros base; después de cada fase 2, sus hiperparámetros ganadores pasan
a la fase 1 de la siguiente iteración. En cada iteración, la fase 2 usa únicamente
la rejilla configurada para esa ejecución sobre la celda ganadora de su propia
fase 1. Solo los ganadores de la última iteración se reentrenan con todo el 85 %
de entrenamiento. `test.parquet` permanece aislado hasta esa fase final. Un JSON
personalizado tiene esta forma:

```json
{
  "logreg": [{"maxIter": 100, "regParam": 0.1, "elasticNetParam": 0.0}],
  "rf": [{"numTrees": 50, "maxDepth": 10}],
  "svm": [{"maxIter": 200, "regParam": 0.1, "tol": 1e-6}]
}
```

Las salidas se organizan en `results/iteracion_N/fase_1/`, `fase_2/` y
`results/fase_3/<modelo>/`. En fase 3 se ajusta una única receta de
preprocesamiento con todo el 85 % antes de transformar train y test, y se guardan
las métricas por clase y las matrices de confusión.

## Contrato de la interfaz

### `construir_spark(master="local[*]", *, driver_memory=None, shuffle_partitions=32, app_name="unsw-nb15")`

Crea y configura la sesión de Spark (AQE habilitado en masters `local*`, log en WARN).
**El llamador es dueño de la sesión**: debe ejecutar `spark.stop()` en un bloque
`finally`. Ninguna función `correr_*` la detiene.

| Parámetro | Tipo | Notas |
|---|---|---|
| `master` | `str` | default `local[*]` (todos los cores) |
| `driver_memory` | `str \| None` | p. ej. `"4g"`. Si es `None`, Spark usa su default (1 GiB) — insuficiente en instancias de 8 GiB |
| `shuffle_partitions` | `int` | default `32`; el default de Spark (200) es excesivo en un solo nodo |
| `app_name` | `str` | solo identificación en la UI de Spark |

### `correr_ronda(train_df, val_df, arma, params, escala, destino) -> dict[str, float]`

**Unidad granular de trabajo.** Fitea el pipeline en `train_df`, predice `val_df`,
persiste tres artefactos en `destino` (crea el directorio si no existe) y devuelve las
métricas en un dict `{accuracy, promedio_precision, promedio_recall, promedio_F1}` (ver *Métricas de
una ronda*).

| Argumento | Qué recibe |
|---|---|
| `train_df`, `val_df` | DataFrames **ya pasados por `common.cargar_parquet`** (sanea nombres de columnas y castea booleanos). Un DataFrame crudo de `spark.read.parquet` puede fallar en `VectorAssembler` |
| `arma` | `"logreg"`, `"rf"` o `"svm"` (constante `common.ALGOS`) |
| `params` | dict de hiperparámetros que se superponen a los default (`{}` = default) |
| `escala` | `bool`; `True` aplica `StandardScaler` antes del clasificador |
| `destino` | `Path` de salida; escribe `predicciones.parquet`, `modelo/`, `metricas.json` |

### `correr_suelto(spark, entrada, *, val=None, fraccion_val=0.2, semilla=42, arma, params, escala=True, out="results", limite=None) -> list[dict]`

Entrena sobre un parquet suelto. Si `val` falta, parte `entrada` con
`randomSplit([1 - fraccion_val, fraccion_val], seed=semilla)`. `limite` reparte el
tope entre las clases del archivo (hasta `limite // n_clases` filas por clase), de
modo que ninguna clase quede sin ejemplos — **solo para pruebas de humo**, nunca para
resultados finales. Escribe en `<out>/suelto/<arma>/` y devuelve una fila
`{"celda": "suelto", "ronda": "-1", "arma": ..., **metricas}`.

### `correr_grid(spark, grid, *, arma, params, escala=True, out="results", limite=None) -> list[dict]`

Recorre `<grid>/<celda>/r<k>/{train,val}.parquet` (celdas `G*` ordenadas alfabéticamente,
dentro de cada una las rondas `r*` en orden) y llama `correr_ronda` por cada par
celda×ronda. Escribe en `<out>/<celda>/<arma>/r<k>/`. Devuelve una fila por par
`{"celda": ..., "ronda": ..., "arma": ..., **metricas}`. Lanza `SystemExit` si el
directorio no contiene celdas `G*`.

## Reglas de composición

1. **Sesión**: quien llama crea (`construir_spark`) y destruye (`spark.stop()`) la
   sesión. Las funciones `correr_*` reciben el `spark` como primer argumento y lo usan,
   no lo gestionan.
2. **DataFrames de entrada**: pasar siempre por `common.cargar_parquet(spark, ruta)`
   (aplica `sanear_columnas` — espacios/guiones → guion bajo, porque el diccionario de
   UNSW-NB15 trae `ct_src_ ltm` — y `preparar`, que castea booleanos a `double`).
3. **`label` ya viene indexado** (entero 0–9) desde `pre-process.py`; el mapeo índice →
   nombre de clase vive en `data/preprocessed/base/metadatos.json`. Por eso el pipeline
   NO incluye un `StringIndexer` de objetivo.
4. **Columnas de control**: `label`, `attack_cat` y `es_duplicado` nunca entran como
   features (constante `common.CONTROL`).
5. **Sin fugas de datos**: el one-hot y el `StandardScaler` se fitean *dentro* del
   pipeline, por fold. No fitear ningún transformador antes del split de la ronda.
6. **Agregación y CSV**: el motor devuelve listas de filas; promediar por celda,
   comparar armas y escribir el `metricas.csv` agregado es responsabilidad de la capa
   de CV. Esquema de columnas: `celda, ronda, arma` + `common.METRICAS`.
7. **La sesión de cada ronda es la misma**: `correr_grid` reutiliza el `spark`
   recibido en todo el recorrido; no hace falta (ni está soportado) recrear la sesión
   por ronda.

## Algoritmos e hiperparámetros

`common.ALGOS = ("logreg", "rf", "svm")`. Defaults explícitos en
`common.PARAMS_POR_DEFECTO` (una corrida sin `params` es reproducible):

| Arma | Defaults | Notas |
|---|---|---|
| `logreg` | `maxIter=100, regParam=0.0, elasticNetParam=0.0` | LogisticRegression de MLlib |
| `rf` | `numTrees=20, maxDepth=5, seed=42` | Débiles a propósito: son la línea base, no el resultado tuneado. `seed` fija el muestreo del bosque para que dos corridas idénticas den métricas idénticas |
| `svm` | `maxIter=100, regParam=0.0, tol=1e-6` | **Lineal**: MLlib no tiene kernels RBF. Al ser binario, se envuelve en `OneVsRest` (10 modelos binarios por fold). Solo expone `rawPrediction`, no `probability` |

`params` se fusiona sobre los default (`{**default, **params}`); un parámetro inválido
produce el `ValueError` nativo del estimador de Spark.

## Métricas de una ronda

`common.METRICAS = ("accuracy", "promedio_precision", "promedio_recall", "promedio_F1")`. Se calculan
en **una sola pasada**: `common.evaluar` hace un `groupBy(label, prediction).count()` y el
cálculo vive en `common.detalle_desde_matriz`, una función pura (sin Spark) que también se
puede verificar con una matriz de confusión escrita a mano.

| Métrica | Qué mide | Nota |
|---|---|---|
| `accuracy` | aciertos / total | Equivale al `weightedRecall` multiclase, por eso no se reporta aparte |
| `promedio_precision` | precisión por clase promediada sin ponderar | Cada clase tiene el mismo peso |
| `promedio_recall` | recall por clase promediado sin ponderar | Expone ataques que no se detectan |
| `promedio_F1` | F1 por clase promediado sin ponderar | Métrica principal ante el desbalance |

La evaluación detallada puede guardar `metricas_por_clase.csv`, con `precision`, `recall`,
`F1`, `TP`, `TN`, `FP`, `FN` y `support`, además de las matrices de confusión absoluta y
normalizada por fila. El mapa completo de clases puede pasarse a `evaluar` para incluir
clases ausentes con métricas cero.

## Pipeline por fold

Para cada fold, `construir_pipeline` arma, en orden:

1. Por cada columna categórica (string): `StringIndexer(handleInvalid="keep")` →
   `OneHotEncoder(handleInvalid="keep")`. Con `"keep"`, las categorías ausentes en el
   fold-train no descartan filas de validación.
2. `VectorAssembler(numéricas + one-hots → features_raw, handleInvalid="skip")`.
3. Si `escala`: `StandardScaler(withMean=False, withStd=True)` → `features`.
4. El clasificador de la sección anterior.

`common.separar_features` separa categóricas/numéricas por el esquema de Spark después
de `preparar`; no hay lista fija de columnas — el esquema del parquet manda.

## Esquema de salidas

```
<out>/
├── metricas.csv                        # solo lo escribe cli.py; header: celda,ronda,algo,accuracy,promedio_precision,promedio_recall,promedio_F1
└── <celda>/<arma>/r<k>/
    ├── modelo/                         # PipelineModel nativo de Spark (recargable)
    ├── predicciones.parquet            # label, attack_cat, es_duplicado?, prediction, probability|raw_prediction
    ├── metricas.json                   # resumen de la ronda
    ├── metricas_por_clase.csv          # evaluación detallada cuando se solicita
    ├── matriz_confusion.csv
    └── matriz_confusion_normalizada.csv
```

En `correr_suelto`, `<celda>` es literalmente `suelto` y `r<k>` es `-1`.

`predicciones.parquet`: `probability` (logreg/rf) es un **array** de 10 elementos con la
probabilidad por clase; en `svm` se guarda `raw_prediction` (márgenes del OneVsRest)
porque `LinearSVC` no expone probabilidades.

## Recargar un modelo sin reentrenar

El `modelo/` persistido ya incluye indexado, one-hot, ensamblado, escalado y
clasificador. Para clasificar datos nuevos basta:

```python
from pyspark.ml import PipelineModel
import common

modelo = PipelineModel.load("results/fase2/G03/rf/modelo")   # o common.cargar_modelo(...)
nuevos = common.cargar_parquet(spark, "data/preprocessed/base/test.parquet")
pred = common.seleccionar_predicciones(modelo.transform(nuevos))
metricas = common.evaluar(pred)
```

## Requisitos de ejecución

- **Python ≥ 3.12** y **Java 17** (ver `README.md` raíz).
- En la instancia EC2 del laboratorio (m7i-flex.large, 2 vCPU / 8 GiB): usar
  `--driver-memory 4g` — el default de 1 GiB del driver se queda corto.
- El repo `data/preprocessed/` no está versionado: correr `pre-process.py` (o
  clonar/copiar la rejilla) antes de una corrida completa.
- En Windows, PySpark local requiere `winutils.exe` + `HADOOP_HOME`; en la EC2
  (Ubuntu) no aplica.

## Ejemplos mínimos

### Prueba de humo (CLI)

```bash
python notebooks/cli.py --input <ruta>/train.parquet --algo logreg \
    --limite 20000 --driver-memory 4g --out results/humo
```