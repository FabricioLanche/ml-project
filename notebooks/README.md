# Modelado con MLlib (P2): interfaz del motor para la capa de validación cruzada

Este documento describe la interfaz de `train.py` y `common.py` para quien construya la
capa de validación cruzada y comparación de modelos por encima del motor. Todo lo que
sigue corresponde al código tal como está en el repo; si se cambia una firma, este
documento se actualiza en el mismo commit.

## Alcance

- `hiperparametros.py`: base y variaciones de hiperparámetros. `pre-process.py` las
  materializa en `data/preprocessed/grid_hyperparams/hiperparametros.json`.
- `common.py`: pipeline MLlib compartido (features, clasificadores, evaluación, E/S).
  No importa `train.py` ni depende de `argparse`.
- `train.py`: motor del pipeline activo (`construir_spark`, `preparar_fold_features`,
  `evaluar_features`, `correr_final`). Importable como biblioteca.
- `k-foldCV.py`: orquestador de punta a punta. Ejecuta la comparación de celdas,
  la búsqueda de hiperparámetros de cada modelo y, al terminar las iteraciones,
  el reentrenamiento final contra `base/test.parquet`.

**Lo que el motor NO hace** (y por eso existe la capa de CV):

- No agrega ni promedia métricas entre rondas: devuelve una fila por corrida y quien
  llama decide cómo promediar/comparar.
- Las fases 1 y 2 escriben sus propios CSV desde `k-foldCV.py`, con el esquema descrito
  en la documentación del orquestador.
- No crea ni destruye sesiones de Spark: ver *Reglas de composición*.

## Orquestador de k-fold

La ejecución completa usa las cinco rondas de cada celda y mantiene un flujo
independiente para `logreg`, `rf` y `svm`:

```bash
python notebooks/k-foldCV.py --iteraciones 1 --driver-memory 4g
```

La fase 1 compara las cuatro celdas con los parámetros base. La fase 2 busca las
variaciones sobre la celda ganadora de cada modelo. La base y la grilla se leen por
defecto de `data/preprocessed/grid_hyperparams/hiperparametros.json`, generado por
`pre-process.py`; `--hiperparametros` (alias `--params`) permite usar otro JSON. El
criterio de selección es, en orden, mayor media de `promedio_F1`, mayor media de
`promedio_recall`, mayor media de `promedio_precision` y menor `DS_F1`.

`--iteraciones` repite ambas fases. La primera fase de la primera iteración usa
los parámetros base; después de cada fase 2, sus hiperparámetros ganadores pasan
a la fase 1 de la siguiente iteración. En cada iteración, la fase 2 usa únicamente
la grilla configurada para esa ejecución sobre la celda ganadora de su propia
fase 1. Solo los ganadores de la última iteración se reentrenan con todo el 85 %
de entrenamiento. `test.parquet` permanece aislado hasta esa fase final. El JSON
contiene `base` y `grilla`; esta última se genera como producto cartesiano explícito:

| Modelo | Ejes de la grilla | Configuraciones |
|---|---|---|
| `logreg` | `regParam {0, 1e-6, 1e-4, 1e-2}` × `elasticNetParam {0.0, 0.5}` | 8 |
| `rf` | `numTrees {100, 200}` × `maxDepth {10, 18, 25}` × `minInstancesPerNode {1, 10}` × `maxBins {32, 64}` | 24 |
| `svm` | `maxIter {100, 500}` × `regParam {0, 1e-6, 1e-4, 1e-2}` | 8 |

Cada variación se fusiona sobre `base`, así que solo declara los valores que cambia.
Los valores se editan en `notebooks/hiperparametros.py`.

El preprocesamiento compara cuatro recetas: `G00` (base), `G01` (sin `ct_*`),
`G02` (deduplicado) y `G03` (deduplicado + sin `ct_*`). En todas se mantienen los
puertos agrupados por franja IANA y el `proto` completo, que se codifica con one-hot
en el pipeline de Spark. Si se modifica esta grilla, vuelve a ejecutar
`notebooks/pre-process.py` antes de entrenar; el proceso materializa las nuevas celdas
y elimina las obsoletas. Los checkpoints/resultados de la numeración anterior no
corresponden a estas recetas y no deben reanudarse ni compararse con la grilla nueva.
La selección de columnas `log1p` se calcula sobre el fold-train completo de cada ronda,
antes del submuestreo y del deduplicado, y se comparte entre sus celdas; la validación no
participa en esa decisión. En fase 3 se recalcula sobre todo el 85 % de entrenamiento.

Cada fold se persiste como checkpoint en `results/iteracion_N/checkpoints/`. Para
repartir el trabajo entre jobs:

```bash
# Solo fase 1, iteración 1
python notebooks/k-foldCV.py --fase 1 --iteracion 1

# Continuar fase 2 usando la fase 1 ya calculada
python notebooks/k-foldCV.py --fase 2 --iteracion 1 --reanudar

# Repetir una fase concreta
python notebooks/k-foldCV.py --fase 1 2 --iteracion 1
```

Por defecto, cada ejecución recalcula los checkpoints de las fases solicitadas y de
las fases posteriores que dependen de ellas (fase 1 invalida fases 2 y 3; fase 2
invalida fase 3). La limpieza de checkpoints queda limitada a esa iteración. Usa
`--reanudar` para conservar y reutilizar checkpoints compatibles de una ejecución
interrumpida. Al reanudar, el orquestador comprueba que coincidan datos e
hiperparámetros; si cambian, ejecuta desde la fase prerrequisito necesaria sin
`--reanudar`. `--celdas` y `--rondas` permiten smoke tests acotados. Las features se
preparan y liberan fold por fold para limitar el uso de memoria durante las búsquedas.

El orquestador registra eventos con fecha y hora (fases, celdas, preparación de folds
y comienzo/fin de cada entrenamiento, con duración). Se conserva el nivel `WARN` de
Spark para mostrar advertencias y errores, pero se desactivan las barras de progreso
`[Stage ...]` del terminal; la Spark UI web sigue disponible durante la ejecución.
Al inicio también registra los argumentos recibidos, las opciones efectivas (incluidas
`--reanudar`, celdas y rondas), versiones de Python/PySpark y `PYSPARK_SUBMIT_ARGS`
cuando está definida, para contextualizar cada corrida.
Para guardar juntos los mensajes de Python y Spark en WSL, redirige stderr a stdout y
usa `tee` con `set -o pipefail`, de modo que el log incluya errores y el código de salida
refleje también un fallo del entrenamiento.

Para comparar el rendimiento local sobre una misma celda y ronda, ejecuta
`benchmark-local.py`. Corre las configuraciones en serie y en procesos Spark separados,
sin `--reanudar`; usa los Parquet ya materializados y deja los resultados aislados por
configuración para no tocar los experimentos normales. Por defecto mantiene fijos el fold
`G00/r1`, los modelos y las 32 particiones shuffle, y ejecuta `local[*]` con
`--driver-memory 6g`:

```bash
python3 -u notebooks/benchmark-local.py
```

El benchmark usa los Parquet ya materializados, sin muestrear durante el entrenamiento.
Para comparar ejecuciones con otro límite por categoría, primero hay que regenerar los
artefactos desde el preprocesamiento. También se pueden probar varias memorias en
producto cartesiano con los masters, repetir cada configuración o inspeccionar comandos:

```bash
python3 -u notebooks/benchmark-local.py --masters 'local[2]' 'local[4]' 'local[*]' \
  --driver-memories 6g 8g --repeticiones 2
python3 -u notebooks/benchmark-local.py --dry-run
```

Cada ejecución genera un log por configuración y `resumen.csv` en una carpeta nueva
de `logs/benchmark-local-*`. El resumen incluye el tiempo total, el código de salida
y los tiempos por modelo cuando el entrenamiento los alcanzó. Una configuración fallida
queda registrada y no impide probar las demás; el script termina con código distinto de
cero si alguna falló.

El límite de filas por categoría se aplica al materializar cada `train.parquet` de la grilla:
por defecto se conservan hasta `100000` filas por etiqueta `label`, con muestreo
reproducible (semilla 42). Las clases con menos filas se conservan completas. Las
validaciones no se submuestrean y `base/` permanece íntegra para la fase 3. En cada ronda,
las celdas parten de la misma muestra y luego aplican o no deduplicación; los conteos de
filas, grupos únicos y proporción duplicada por clase quedan en
`grid_preprocess/estadisticas_muestreo.csv`, antes y después del deduplicado. Por ello, las
celdas deduplicadas pueden quedar con menos filas que el límite. El límite se cambia al
generar los artefactos; `0` lo desactiva:

```bash
python notebooks/pre-process.py --limite-por-categoria 50000
LIMITE_POR_CATEGORIA=50000 sbatch slurm/01_preprocess.slurm
```

El entrenamiento solo consume los Parquet disponibles y no acepta un límite de muestreo.
Cambiarlo requiere volver a ejecutar el preprocesamiento antes de la búsqueda.

Las salidas se organizan en `results/iteracion_N/fase_1/`, `fase_2/` y
`results/fase_3/<modelo>/`. En fase 3 se ajusta una única receta de
preprocesamiento con todo el 85 % antes de transformar train y test, y se guardan
las métricas por clase y las matrices de confusión.

## Contrato de la interfaz

### `construir_spark(master="local[*]", *, driver_memory=None, shuffle_partitions=32, app_name="unsw-nb15")`

Crea y configura la sesión de Spark (AQE habilitado en masters `local*`, log en WARN).
**El llamador es dueño de la sesión**: debe ejecutar `spark.stop()` en un bloque
`finally`. El motor no detiene la sesión que recibe.

| Parámetro | Tipo | Notas |
|---|---|---|
| `master` | `str` | default `local[*]` (todos los cores) |
| `driver_memory` | `str \| None` | p. ej. `"4g"`. Si es `None`, Spark usa su default (1 GiB) — insuficiente en instancias de 8 GiB |
| `shuffle_partitions` | `int` | default `32`; el default de Spark (200) es excesivo en un solo nodo |
| `app_name` | `str` | solo identificación en la UI de Spark |

## Reglas de composición

1. **Sesión**: quien llama crea (`construir_spark`) y destruye (`spark.stop()`) la
   sesión. Las funciones del motor reciben el `spark` como primer argumento y lo usan,
   no lo gestionan.
2. **DataFrames de entrada**: `common.cargar_parquet(spark, ruta)` lee directamente
   los Parquet materializados por `pre-process.py`. Ese paso ya persiste los nombres
   normalizados y las columnas booleanas como valores numéricos compatibles con Spark ML.
3. **`label` ya viene indexado** (entero 0–9) desde `pre-process.py`; el mapeo índice →
   nombre de clase vive en `data/preprocessed/base/metadatos.json`. Por eso el pipeline
   NO incluye un `StringIndexer` de objetivo.
4. **Columnas de control**: `label`, `attack_cat` y `es_duplicado` nunca entran como
   features (constante `common.CONTROL`).
5. **Sin fugas de datos**: el one-hot y el `StandardScaler` se fitean *dentro* del
   pipeline, por fold. No fitear ningún transformador antes del split de la ronda.
6. **Agregación y CSV**: `k-foldCV.py` calcula los resúmenes por celda y escribe
   `resultado_por_fold.csv` y `resumen_general.csv` para cada fase.
7. **Submuestreo sin fuga**: el límite estratificado se aplica únicamente al train
   antes de ajustar transformaciones; validation se conserva completo.
8. **Features compartidas por fold**: `preparar_fold_features` ajusta las
   transformaciones una sola vez por fold; `evaluar_features` reutiliza esas
   features para cada modelo e hiperparámetro.

## Algoritmos e hiperparámetros

`common.ALGOS = ("logreg", "rf", "svm")`. La base se define en
`notebooks/hiperparametros.py`, se materializa en
`data/preprocessed/grid_hyperparams/hiperparametros.json` y `common.PARAMS_POR_DEFECTO`
la usa como fallback:

| Arma | Base | Notas |
|---|---|---|
| `logreg` | `maxIter=500, regParam=0.0, elasticNetParam=0.0, tol=1e-6, standardization=False, family=multinomial` | LogisticRegression de MLlib |
| `rf` | `numTrees=100, maxDepth=18, seed=42, minInstancesPerNode=1, maxBins=32, impurity=gini, featureSubsetStrategy=sqrt` | Línea base. `seed` fija el muestreo del bosque para que dos corridas idénticas den métricas idénticas |
| `svm` | `maxIter=500, regParam=0.0, tol=1e-6, standardization=False` | **Lineal**: MLlib no tiene kernels RBF. Al ser binario, se envuelve en `OneVsRest` (10 modelos binarios por fold, con `parallelism=5`). Solo expone `rawPrediction`, no `probability` |

Cada configuración se fusiona sobre la base (`{**base, **variacion}`); un parámetro
inválido produce el `ValueError` nativo del estimador de Spark.

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

`common.separar_features` separa categóricas/numéricas según el esquema persistido;
no hay lista fija de columnas — el esquema del parquet manda.

## Esquema de salidas

```
<out>/
└── fase_3/<modelo>/
    ├── modelo/                         # PipelineModel nativo de Spark (recargable)
    ├── predicciones.parquet            # label, attack_cat, es_duplicado?, prediction, probability|raw_prediction
    ├── metricas.json                   # resumen de la ronda
    ├── metricas_por_clase.csv          # evaluación detallada cuando se solicita
    ├── matriz_confusion.csv
    └── matriz_confusion_normalizada.csv
```

`predicciones.parquet`: `probability` (logreg/rf) es un **array** de 10 elementos con la
probabilidad por clase; en `svm` se guarda `raw_prediction` (márgenes del OneVsRest)
porque `LinearSVC` no expone probabilidades.

## Recargar un modelo sin reentrenar

El `modelo/` persistido ya incluye indexado, one-hot, ensamblado, escalado y
clasificador. Para clasificar datos nuevos basta:

```python
from pyspark.ml import PipelineModel
import common

modelo = common.cargar_modelo("results/fase_3/rf/modelo")
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
