"""Regresión Logística multinomial — modelo baseline del proyecto.

Es el modelo de referencia porque su estructura lineal lo hace simple de interpretar y
sus coeficientes permiten cuantificar el aporte de cada característica. Su desempeño fija
la vara con la que se miden los dos modelos mas complejos: si el Random Forest o el SVM no
lo superan de forma clara, la complejidad adicional no se justifica.

El conjunto de datos tiene 87 % de tráfico `Benign`, de modo que el desbalance es el
factor dominante. MLlib no expone `classWeight`, pero la Regresión Logística acepta
`weightCol`, que es el mecanismo estándar: se calcula el inverso de la frecuencia de cada
clase sobre el conjunto de entrenamiento y se lo pasa como peso por fila.

Según la decisión del equipo, la ponderación NO se asume. El script evalúa las dos
variantes (con y sin pesos) sobre el conjunto de validación y se reporta la mejor por
macro-F1, accompany siempre del desglose por categoría.

Ejecución:  python notebooks/RegLin.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.ml.classification import LogisticRegression

import common as C

MODELO = "RegLin"


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{MODELO}: regresión logística multinomial")
    ap.add_argument("--usar-ct", action="store_true", default=True,
                    help="conservar la familia ct_* (default: sí)")
    ap.add_argument("--sin-ct", dest="usar_ct", action="store_false",
                    help="ablación: eliminar la familia ct_*")
    ap.add_argument("--fraccion-busqueda", type=float, default=0.20,
                    help="fracción de train usada para la búsqueda en rejilla")
    ap.add_argument("--fraccion-final", type=float, default=1.0,
                    help="fracción de train usada para el ajuste final")
    args = ap.parse_args()

    spark = C.spark_session(app=f"unsw-{MODELO}")
    print(f"=== {MODELO} · Regresión Logística multinomial (MLlib) ===")
    print(f"Spark {spark.version} | family=multinomial | maxBatches iterativo")

    datos = C.leer_particiones(spark, usar_ct=args.usar_ct)
    k = 10
    nombres = C.nombres_clases()
    for split, df in datos.items():
        print(f"  {split:5s} {df.count():>9,} filas x {len(df.columns)} columnas")

    # ---------------- Búsqueda de hiperparámetros ----------------
    # Se barre regParam y elasticNetParam. El criterio es macro-F1 en validación, que es
    # la única métrica que no deja que el modelo destaque ignorando las clases escasas.
    train_b = datos["train"]
    if args.fraccion_busqueda < 1.0:
        train_b = train_b.sample(False, args.fraccion_busqueda, 42)
        print(f"\n  búsqueda sobre submuestra: {train_b.count():,} filas de train")

    rejilla = []
    for reg in (0.0, 0.01, 0.1):
        for elastic in (0.0, 0.5, 1.0):
            rejilla.append({
                "regParam": reg, "elasticNetParam": elastic,
                "family": "multinomial", "maxIter": 100,
                "_estandarizar": True,
            })

    print(f"\n--- Búsqueda en rejilla ({len(rejilla)} configuraciones) ---")
    resultados_por_pesos = {}
    for con_pesos in (False, True):
        print(f"\n  ponderación de clases: {C.etiqueta_pesos(con_pesos)}")
        res = C.buscar(spark, rejilla, train_b, datos["val"], k, con_pesos)
        resultados_por_pesos[con_pesos] = res
        C.imprimir("  Top 3", res[:3])

    # ---------------- Selección de la variante ponderada ----------------
    # El equipo decidió no asumir la ponderación: se comparan ambas variantes en
    # validación y gana la de mayor macro-F1.
    print("\n--- Comparación de la ponderación (validación) ---")
    mejor = {p: res[0] for p, res in resultados_por_pesos.items()}
    ganador = True if mejor[True]["macro_f1_val"] > mejor[False]["macro_f1_val"] else False
    print(f"\n  winner: {C.etiqueta_pesos(ganador)}")
    print(f"    con pesos : macro-F1 {mejor[True]['macro_f1_val']:.4f}")
    print(f"    sin pesos : macro-F1 {mejor[False]['macro_f1_val']:.4f}")
    print("  Nota: macro-F1 pondera igual las diez clases. Si el recall de las minoritarias")
    print("  sube a costa de la precisión de Benign, ese coste queda visible en la tabla")
    print("  por clase que se reporta a continuación; no se oculta.")

    C.guardar_json("RegLin_busqueda.json",
                   {C.etiqueta_pesos(p): resultados_por_pesos[p] for p in resultados_por_pesos})

    # ---------------- Ajuste final sobre train completo ----------------
    train = datos["train"]
    if args.fraccion_final < 1.0:
        train = train.sample(False, args.fraccion_final, 42)
    cfg = mejor[ganador]
    est = LogisticRegression(
        featuresCol=C.FEATURES_COL, labelCol=C.LABEL_COL, weightCol=C.PESO_COL,
        family="multinomial", regParam=float(cfg["regParam"]),
        elasticNetParam=float(cfg["elasticNetParam"]),
        maxIter=100,
    )

    conteo = C.frecuencias(train)
    train_p = C.anadir_pesos(train, conteo, ganador)
    test_p = C.anadir_pesos(datos["test"], conteo, ganador)

    print(f"\n--- Ajuste final sobre {train_p.count():,} filas ---")
    modelo = C.ajustar(spark, est, C.etapas_feature(train_p, estandarizar=True), train_p)
    print("  modelo ajustado")

    # ---------------- Evaluación en prueba ----------------
    print("\n--- Evaluación en conjunto de prueba ---")
    pred = C.predecir(modelo, test_p)
    cm = C.matriz_confusion(pred, k)
    metricas = C.metricas_desde_cm(cm, k)
    for clave in ("accuracy", "macro_f1", "weighted_f1", "clases_predichas", "n"):
        print(f"  {clave:18s} {metricas[clave]}")

    por_clase = C.metricas_por_clase(cm, k, nombres)
    C.imprimir("Métricas por categoría (test)", por_clase)
    C.guardar_csv("RegLin_metricas_clase.csv", por_clase)

    diag = C.diagnostico_duplicados(pred, k, nombres)
    C.imprimir("Diagnóstico duplicadas vs no duplicadas", diag)

    no_dup = C.metricas_subconjunto_no_duplicado(pred, k, nombres)
    C.imprimir("Métricas por categoría sobre filas NO duplicadas (cifra conservadora)", no_dup)
    C.guardar_csv("RegLin_metricas_no_duplicadas.csv", no_dup)

    C.guardar_json("RegLin_resultados.json", {
        "modelo": MODELO,
        "familia": "multinomial",
        "pesos": MSE,
        "configuracion": {k2: v for k2, v in cfg.items() if not k2.startswith("_")},
        "n_features": modelo.stages[-1].numFeatures if hasattr(modelo.stages[-1], "numFeatures") else None,
        "metricas_test": metricas,
        "diagnostico_duplicados": diag,
    })
    print(f"\nResultados en {C.MODEL_DIR}/RegLin_*")
    spark.stop()


if __name__ == "__main__":
    main()