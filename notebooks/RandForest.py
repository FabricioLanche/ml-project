"""Random Forest — modelo de ensamble por árboles.

Es el segundo de los dos modelos de comparación. Frente a la Regresión Logística, que
resuelve el problema con un hiperplano, un bosque aleatorio modela fronteras no lineales
mediante el promedio de muchos árboles entrenados sobre submuestras y subconjuntos de
características. Eso importa aquí porque la relación entre las variables y la familia de
ataque no es lineal: por ejemplo `ct_state_ttl` discrimina con η² = 0.777 entre las diez
categorías, y una combinación de umbrales captura ese patrón mucho mejor que un único
coeficiente.

Dos decisiones de implementación que conviene tener presentes:

1. **No se estandarizan las variables.** Un árbol decide mediante umbrales (`x <= t`), y
   cualquier transformación monótona conserva el orden, de modo que un bosque es
   invariante a `log1p` y a la estandarización. Saltarse el escalado ahorra un paso
   completo sobre 1.9M filas y no cambia nada del resultado. Esto contrasta con la Regresión
   Logística y el SVM, donde el escalado sí es imprescindible.

2. **La profundidad se limita por defecto.** Sin límite, los árboles de un bosque sobre
   1.9M filas pueden llegar a profundidad 20 o más y sobreajustar directamente las ráfagas
   de flujos idénticos que se conservan. Una profundidad acotada es además la
   protección más barata contra el atajo de repetición documentado en el EDA.

Ejecución:  python notebooks/RandForest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.ml.classification import RandomForestClassifier

import common as C

MODELO = "RandForest"


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{MODELO}: bosque aleatorio")
    ap.add_argument("--usar-ct", action="store_true", default=True,
                    help="conservar la familia ct_* (default: sí)")
    ap.add_argument("--sin-ct", dest="usar_ct", action="store_false",
                    help="ablación: eliminar la familia ct_*")
    ap.add_argument("--fraccion-busqueda", type=float, default=0.25,
                    help="fracción de train usada para la búsqueda en rejilla")
    ap.add_argument("--fraccion-final", type=float, default=1.0,
                    help="fracción de train usada para el ajuste final")
    args = ap.parse_args()

    spark = C.spark_session(app=f"unsw-{MODELO}")
    print(f"=== {MODELO} · Random Forest (MLlib) ===")
    print("Sin estandarización: un bosque es invariante a transformaciones monótonas.")

    datos = C.leer_particiones(spark, usar_ct=args.usar_ct)
    k = 10
    nombres = C.nombres_clases()
    for split, df in datos.items():
        print(f"  {split:5s} {df.count():>9,} filas x {len(df.columns)} columnas")

    train_b = datos["train"]
    if args.fraccion_busqueda < 1.0:
        train_b = train_b.sample(False, args.fraccion_busqueda, 42)
        print(f"\n  búsqueda sobre submuestra: {train_b.count():,} filas de train")

    # ---------------- Búsqueda de hiperparámetros ----------------
    # numTrees fija el error de varianza del ensamble; maxDepth controla el ajuste
    # individual de cada árbol; minInstancesPerNode actúa como regularizador y es la
    # defensa más directa contra el sobreajuste de las ráfagas duplicadas.
    rejilla = []
    for prof in (6, 10, 14):
        for min_nodo in (1, 10):
            rejilla.append({
                "numTrees": 60, "maxDepth": prof, "minInstancesPerNode": min_nodo,
                "subsamplingRate": 0.7, "maxBins": 32,
                "_estandarizar": False,
            })

    print(f"\n--- Búsqueda en rejilla ({len(rejilla)} configuraciones) ---")
    resultados_por_pesos = {}
    for con_pesos in (False, True):
        print(f"\n  ponderación de clases: {C.etiqueta_pesos(con_pesos)}")
        res = C.buscar(spark, rejilla, train_b, datos["val"], k, con_pesos)
        resultados_por_pesos[con_pesos] = res
        C.imprimir("  Top 3", res[:3])

    print("\n--- Comparación de la ponderación (validación) ---")
    mejor = {p: res[0] for p, res in resultados_por_pesos.items()}
    ganador = True if mejor[True]["macro_f1_val"] > mejor[False]["macro_f1_val"] else False
    print(f"\n  winner: {C.etiqueta_pesos(ganador)}")
    print(f"    con pesos : macro-F1 {mejor[True]['macro_f1_val']:.4f}")
    print(f"    sin pesos : macro-F1 {mejor[False]['macro_f1_val']:.4f}")

    C.guardar_json("RandForest_busqueda.json",
                   {C.etiqueta_pesos(p): resultados_por_pesos[p] for p in resultados_por_pesos})

    # ---------------- Ajuste final sobre train completo ----------------
    train = datos["train"]
    if args.fraccion_final < 1.0:
        train = train.sample(False, args.fraccion_final, 42)
    cfg = mejor[ganador]
    est = RandomForestClassifier(
        featuresCol=C.FEATURES_COL, labelCol=C.LABEL_COL, weightCol=C.PESO_COL,
        numTrees=int(cfg["numTrees"]), maxDepth=int(cfg["maxDepth"]),
        minInstancesPerNode=int(cfg["minInstancesPerNode"]),
        subsamplingRate=float(cfg["subsamplingRate"]), maxBins=int(cfg["maxBins"]),
        seed=42,
    )

    conteo = C.frecuencias(train)
    train_p = C.anadir_pesos(train, conteo, ganador)
    test_p = C.anadir_pesos(datos["test"], conteo, ganador)

    print(f"\n--- Ajuste final sobre {train_p.count():,} filas ---")
    print(f"  numTrees={cfg['numTrees']} maxDepth={cfg['maxDepth']} "
          f"minInstancesPerNode={cfg['minInstancesPerNode']} pesos={C.etiqueta_pesos(ganador)}")
    modelo = C.ajustar(spark, est, C.etapas_feature(train_p, estandarizar=False), train_p)
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
    C.guardar_csv("RandForest_metricas_clase.csv", por_clase)

    diag = C.diagnostico_duplicados(pred, k, nombres)
    C.imprimir("Diagnóstico duplicadas vs no duplicadas", diag)

    no_dup = C.metricas_subconjunto_no_duplicado(pred, k, nombres)
    C.imprimir("Métricas por categoría sobre filas NO duplicadas (cifra conservadora)", no_dup)
    C.guardar_csv("RandForest_metricas_no_duplicadas.csv", no_dup)

    C.guardar_json("RandForest_resultados.json", {
        "modelo": MODELO,
        "pesos": C.etiqueta_pesos(ganador),
        "configuracion": {k2: v for k2, v in cfg.items() if not k2.startswith("_")},
        "metricas_test": metricas,
        "diagnostico_duplicados": diag,
    })
    print(f"\nResultados en {C.MODEL_DIR}/RandForest_*")
    spark.stop()


if __name__ == "__main__":
    main()