"""Máquina de Vectores de Soporte — modelo de margen para clasificación multiclase.

Es el tercer modelo del proyecto y el que aporta una perspectiva distinta: frente a la
frontera lineal de la Regresión Logística y las fronteras por umbral del Random Forest,
el SVM optimiza directamente el margen entre las clases, lo que tiende a producir
separaciones más limpias en problemas con muchas features y muestras acotadas.

## Limitación de la librería y cómo se resuelve

**MLlib no ofrece un SVM multiclase.** `pyspark.ml.classification.LinearSVC` es
estrictamente binario: su atributo `numClasses` vale siempre 2 y su documentación lo
declara como «binary classifier» que optimiza la pérdida hinge con OWLQN. Tampoco expone
una salida `decisionValue`, solo `rawPrediction`, que en el caso binario es un vector de
dos elementos donde el score de la clase positiva es `rawPrediction[1]`.

La clasificación multiclase se resuelve con la estrategia **Uno-contra-Todos**: se
entrenan diez modelos binarios, cada uno tomando una clase como positiva frente a las
nueve restantes, y se predice la clase cuyo margen sea mayor. Es una aproximación
estándar y su limitación es conocida: no optimiza directamente la pérdida multiclase
conjunta, por lo que en problemas con clases muy solapadas puede ser inferior a un SVM
multiclase nativo. Se documenta como limitación del diseño experimental, no como
propiedad del método.

El coste computacional es multiplicar por diez el número de ajustes, lo que se compensa
limitando la rejilla de hiperparámetros.

Ejecución:  python notebooks/SVM.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.ml.classification import LinearSVC
from pyspark.sql import DataFrame, functions as F

import common as C

MODELO = "SVM"


def _binario(df: DataFrame, clase: int) -> DataFrame:
    """Reetiqueta el problema para que la clase `clase` sea la positiva (1) y el resto 0."""
    return df.withColumn("_bin", (F.col(C.LABEL_COL) == clase).cast("double"))


def ajustar_ovr(spark, cfg: dict, train_p: DataFrame, k: int) -> list:
    """Entrena los diez modelos binarios de Uno-Contra-Todos.

    Cada iteracion rehace el `StringIndexer` porque la clase positiva cambia, pero el
    coste de esa etapa es despreciable frente al ajuste hinge que domina el tiempo.
    """
    modelos = []
    for clase in range(k):
        est = LinearSVC(
            featuresCol=C.FEATURES_COL, labelCol="_bin", weightCol=C.PESO_COL,
            regParam=float(cfg["regParam"]), maxIter=int(cfg["maxIter"]),
            standardization=False,  # el StandardScaler del pipeline ya normalizó
        )
        modelos.append(C.ajustar(spark, est,
                                 C.etapas_feature(train_p, estandarizar=True),
                                 _binario(train_p, clase)))
    return modelos


def predecir_ovr(modelos: list, df: DataFrame, k: int) -> DataFrame:
    """Combina los diez márgenes binarios y predice el argmax.

    `rawPrediction[1]` es el score de la clase positiva de cada modelo binario, es decir
    la distancia con signo al hiperplomo de esa clase. El argmax de esos diez valores es
    la predicción multiclase.
    """
    base = df.withColumn("_id", F.monotonically_increasing_id())
    scores = None
    for i, modelo in enumerate(modelos):
        s = modelo.transform(base).select("_id", F.col("rawPrediction")[1].alias(f"s{i}"))
        scores = s if scores is None else scores.join(s, on="_id")

    cols = [F.col(f"s{i}") for i in range(k)]
    scores = scores.withColumn("_scores", F.array(cols))
    # array_position es 1-based; se resta 1 para obtener el índice de clase.
    scores = scores.withColumn(
        "prediction",
        F.array_position(F.col("_scores"), F.array_max(F.col("_scores"))).cast("int") - 1,
    )
    return (base.join(scores.select("_id", "prediction"), on="_id")
                 .select(C.LABEL_COL, C.CLASE_COL, C.DUP_COL, C.PESO_COL, "prediction"))


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{MODELO}: SVM uno-contra-todos")
    ap.add_argument("--usar-ct", action="store_true", default=True,
                    help="conservar la familia ct_* (default: sí)")
    ap.add_argument("--sin-ct", dest="usar_ct", action="store_false",
                    help="ablación: eliminar la familia ct_*")
    ap.add_argument("--fraccion-busqueda", type=float, default=0.15,
                    help="fracción de train usada para la búsqueda en rejilla")
    ap.add_argument("--fraccion-final", type=float, default=1.0,
                    help="fracción de train usada para el ajuste final")
    args = ap.parse_args()

    spark = C.spark_session(app=f"unsw-{MODELO}")
    print(f"=== {MODELO} · Máquina de Vectores de Soporte ===")
    print("MLlib no ofrece SVM multiclase: se usa Uno-Contra-Todos con 10 modelos binarios.")

    datos = C.leer_particiones(spark, usar_ct=args.usar_ct)
    k = 10
    nombres = C.nombres_clases()
    for split, df in datos.items():
        print(f"  {split:5s} {df.count():>9,} filas x {len(df.columns)} columnas")

    train_b = datos["train"]
    if args.fraccion_busqueda < 1.0:
        train_b = train_b.sample(False, args.fraccion_busqueda, 42)
        print(f"\n  búsqueda sobre submuestra: {train_b.count():,} filas de train")

    conteo_b = C.frecuencias(train_b)
    train_bp = C.anadir_pesos(train_b, conteo_b, False)
    val_bp = C.anadir_pesos(datos["val"], conteo_b, False)

    # ---------------- Búsqueda de hiperparámetros ----------------
    # regParam es el parámetro crítico del SVM: con 200 features y solapamiento alto
    # entre clases(minoritarias), una regularización L2 fuerte suele ser necesaria.
    rejilla = [
        {"regParam": r, "maxIter": 100}
        for r in (0.001, 0.01, 0.1, 1.0)
    ]

    print(f"\n--- Búsqueda en rejilla ({len(rejilla)} configuraciones x 10 modelos) ---")
    resultados = []
    for i, cfg in enumerate(rejilla, 1):
        t0 = time.time()
        modelos = ajustar_ovr(spark, cfg, train_bp, k)
        pred = predecir_ovr(modelos, val_bp, k)
        m = C.metricas_desde_cm(C.matriz_confusion(pred, k), k)
        seg = round(time.time() - t0, 1)
        resultados.append({"modelo": MODELO, **cfg, "macro_f1_val": round(m["macro_f1"], 4),
                           "accuracy_val": round(m["accuracy"], 4),
                           "clases_predichas_val": m["clases_predichas"], "segundos": seg})
        print(f"    [{i}/{len(rejilla)}] macro-F1={m['macro_f1']:.4f} "
              f"accuracy={m['accuracy']:.4f} clases={m['clases_predichas']}  {cfg}  ({seg}s)",
              flush=True)
        del modelos
    resultados.sort(key=lambda r: -r["macro_f1_val"])
    C.imprimir("Configuraciones ordenadas", resultados)
    C.guardar_csv("SVM_busqueda.csv", resultados)

    mejor = resultados[0]
    print(f"\n  mejor configuración: {mejor}")

    # ---------------- Selección de la ponderación ----------------
    print("\n--- Comparación de la ponderación (validación) ---")
    train_bpw = C.anadir_pesos(train_b, conteo_b, True)
    val_bpw = C.anadir_pesos(datos["val"], conteo_b, True)
    t0 = time.time()
    modelos_w = ajustar_ovr(spark, mejor, train_bpw, k)
    pred_w = predecir_ovr(modelos_w, val_bpw, k)
    mw = C.metricas_desde_cm(C.matriz_confusion(pred_w, k), k)
    print(f"    con pesos : macro-F1 {mw['macro_f1']:.4f}  accuracy {mw['accuracy']:.4f} "
          f"({round(time.time() - t0)}s)")
    print(f"    sin pesos : macro-F1 {mejor['macro_f1_val']:.4f}  "
          f"accuracy {mejor['accuracy_val']:.4f}")
    ganador = mw["macro_f1"] > mejor["macro_f1_val"]
    print(f"  winner: {C.etiqueta_pesos(ganador)}")
    del modelos_w

    # ---------------- Ajuste final sobre train completo ----------------
    train = datos["train"]
    if args.fraccion_final < 1.0:
        train = train.sample(False, args.fraccion_final, 42)
    conteo = C.frecuencias(train)
    train_p = C.anadir_pesos(train, conteo, ganador)
    test_p = C.anadir_pesos(datos["test"], conteo, ganador)

    print(f"\n--- Ajuste final sobre {train_p.count():,} filas ---")
    t0 = time.time()
    modelos = ajustar_ovr(spark, mejor, train_p, k)
    print(f"  10 modelos binarios ajustados en {round(time.time() - t0)}s")

    # ---------------- Evaluación en prueba ----------------
    print("\n--- Evaluación en conjunto de prueba ---")
    pred = predecir_ovr(modelos, test_p, k)
    cm = C.matriz_confusion(pred, k)
    metricas = C.metricas_desde_cm(cm, k)
    for clave in ("accuracy", "macro_f1", "weighted_f1", "clases_predichas", "n"):
        print(f"  {clave:18s} {metricas[clave]}")

    por_clase = C.metricas_por_clase(cm, k, nombres)
    C.imprimir("Métricas por categoría (test)", por_clase)
    C.guardar_csv("SVM_metricas_clase.csv", por_clase)

    diag = C.diagnostico_duplicados(pred, k, nombres)
    C.imprimir("Diagnóstico duplicadas vs no duplicadas", diag)

    no_dup = C.metricas_subconjunto_no_duplicado(pred, k, nombres)
    C.imprimir("Métricas por categoría sobre filas NO duplicadas (cifra conservadora)", no_dup)
    C.guardar_csv("SVM_metricas_no_duplicadas.csv", no_dup)

    C.guardar_json("SVM_resultados.json", {
        "modelo": MODELO,
        "estrategia_multiclase": "uno-contra-todos (MLlib no ofrece SVM multiclase)",
        "pesos": C.etiqueta_pesos(ganador),
        "configuracion": {k2: v for k2, v in mejor.items() if k2 not in ("modelo",)},
        "metricas_test": metricas,
        "diagnostico_duplicados": diag,
    })
    print(f"\nResultados en {C.MODEL_DIR}/SVM_*")
    spark.stop()


if __name__ == "__main__":
    main()