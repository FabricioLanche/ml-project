"""Entrena un modelo de MLlib sobre la rejilla de preprocesamiento (o un parquet suelto).

Punto de entrada de la etapa de modelado (P2). Fitea un modelo por cada ronda ya
materializada, predice el fold de validación y persiste tanto el `PipelineModel`
como las predicciones. El puntaje de validación cruzada es el agregado de las
métricas por ronda (promedio de las 5 rondas en la fase de reporte).

La CLI de este módulo es solo un adaptador para operar en la EC2 (prueba de humo y
corrida completa de la rejilla). El motor es importable: `construir_spark`,
`correr_suelto` y `correr_grid` no dependen de argparse y componen desde código.

Uso como biblioteca (composición programática, p. ej. la comparación con CV)::

    from train import construir_spark, correr_grid

    spark = construir_spark("local[*]", app_name="comparacion-cv")
    filas = correr_grid(spark, "data/preprocessed/grid", algo="rf",
                        params={}, escala=True, out="results/base_rf")

Ejemplos
--------
Prueba de humo sobre un parquet suelto, hiperparámetros por defecto::

    python notebooks/train.py --input <ruta>/train.parquet --algo logreg --limite 20000 --out results/humo

Corrida completa sobre la rejilla (10 celdas × 5 rondas)::

    python notebooks/train.py --grid data/preprocessed/grid --algo rf --out results/base_rf

Corrida tuneada (JSON con los hiperparámetros del clasificador)::

    python notebooks/train.py --grid data/preprocessed/grid --algo rf --params rf_params.json

Salidas
-------
Por cada par (celda, ronda)::

    <out>/<celda>/<algo>/r<k>/modelo/              # PipelineModel nativo de Spark
    <out>/<celda>/<algo>/r<k>/predicciones.parquet # label, attack_cat, prediction, probability
    <out>/<celda>/<algo>/r<k>/metricas.json

más un resumen acumulado en `<out>/metricas.csv`.

Notas
-----
* El modelo guardado ya incluye indexado, one-hot y escalado, así que para
  clasificar datos nuevos basta `common.cargar_modelo(ruta).transform(df)`. Esto es
  lo que necesita la contraparte de evaluación: no hace falta reconstruir el preprocesamiento.
* En Windows, PySpark local requiere `winutils.exe`/`HADOOP_HOME`. En el EC2 de
  AWS Academy (Ubuntu) no aplica.
"""

import argparse
import csv
import json
from pathlib import Path

import common


# ===========================================================================
# Motor: sesión de Spark (parámetros planos; el llamador es dueño de la sesión)
# ===========================================================================
def construir_spark(master="local[*]", *, driver_memory=None, shuffle_partitions=32,
                    app_name="unsw-nb15"):
    """Crea y configura la sesión de Spark. Quien llama decide cuándo hacer `stop()`."""
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    )
    if driver_memory:
        builder = builder.config("spark.driver.memory", driver_memory)
    if master.startswith("local"):
        builder = builder.config("spark.sql.adaptive.enabled", "true")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ===========================================================================
# Motor: una ronda
# ===========================================================================
def correr_ronda(train_df, val_df, algo: str, params: dict, escala: bool,
                 destino: Path) -> dict[str, float]:
    """Fitea en `train_df`, predice `val_df`, persiste modelo y predicciones."""
    pipeline = common.construir_pipeline(train_df, algo, params, escala)
    modelo = pipeline.fit(train_df)

    pred = common.seleccionar_predicciones(modelo.transform(val_df))
    metricas = common.evaluar(pred)

    destino.mkdir(parents=True, exist_ok=True)
    pred.write.mode("overwrite").parquet(str(destino / "predicciones.parquet"))
    common.guardar_modelo(modelo, destino / "modelo")
    (destino / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metricas


# ===========================================================================
# Motor: corridas programáticas (sin argparse)
# ===========================================================================
def correr_suelto(spark, entrada, *, val=None, fraccion_val=0.2, semilla=42,
                  algo: str, params: dict, escala: bool = True,
                  out="results", limite=None) -> list[dict]:
    """Entrena sobre un parquet suelto (si falta `val`, se parte con `fraccion_val`).

    Devuelve las filas de resumen listas para agregación (una por corrida).
    """
    df = common.cargar_parquet(spark, entrada, limite)
    if val:
        train_df = df
        val_df = common.cargar_parquet(spark, val, limite)
    else:
        train_df, val_df = df.randomSplit(
            [1.0 - fraccion_val, fraccion_val], seed=semilla
        )
        print(f"sin --val: se parte {entrada} con fracción de validación {fraccion_val}")

    destino = Path(out) / "suelto" / algo
    metricas = correr_ronda(train_df, val_df, algo, params, escala, destino)
    filas = [{"celda": "suelto", "ronda": "-1", "algo": algo, **metricas}]
    _imprimir("suelto/-1", metricas)
    return filas


def correr_grid(spark, grid, *, algo: str, params: dict, escala: bool = True,
                out="results", limite=None) -> list[dict]:
    """Recorre `<grid>/<celda>/r<k>/{train,val}.parquet` y entrena un modelo por ronda.

    Devuelve las filas de resumen (una por par celda×ronda), en el mismo formato
    que alimenta `metricas.csv`. El progreso se imprime a terminal.
    """
    celdas = sorted(p for p in Path(grid).glob("G*") if p.is_dir())
    if not celdas:
        raise SystemExit(f"no se encontraron celdas G* en {grid}")

    filas: list[dict] = []
    for celda_dir in celdas:
        for ronda_dir in sorted(p for p in celda_dir.glob("r*") if p.is_dir()):
            etiqueta = f"{celda_dir.name}/{ronda_dir.name}"
            print(f"\n>>> {etiqueta}")
            train_df = common.cargar_parquet(
                spark, ronda_dir / "train.parquet", limite
            )
            val_df = common.cargar_parquet(
                spark, ronda_dir / "val.parquet", limite
            )
            destino = Path(out) / celda_dir.name / algo / ronda_dir.name
            metricas = correr_ronda(
                train_df, val_df, algo, params, escala, destino
            )
            filas.append({"celda": celda_dir.name, "ronda": ronda_dir.name,
                          "algo": algo, **metricas})
            _imprimir(etiqueta, metricas)
    return filas


# ===========================================================================
# Reporte / progreso
# ===========================================================================
def _imprimir(etiqueta: str, metricas: dict[str, float]) -> None:
    detalle = "  ".join(f"{k}={v:.4f}" for k, v in metricas.items())
    print(f"    {etiqueta}: {detalle}")


def _escribir_resumen(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(
            fh, fieldnames=["celda", "ronda", "algo", *common.METRICAS]
        )
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nresumen: {ruta}  ({len(filas)} corridas)")


# ===========================================================================
# CLI (adaptador delgado: parsea -> llama al motor -> reporta)
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    origen = parser.add_mutually_exclusive_group(required=True)
    origen.add_argument("--input", help="parquet suelto para entrenar (prueba de humo)")
    origen.add_argument("--grid", help="directorio con <celda>/r<k>/{train,val}.parquet")

    parser.add_argument("--val", help="parquet de validación; si falta, se parte --input")
    parser.add_argument("--fraccion-val", type=float, default=0.2,
                        help="proporción de validación al partir --input (default: 0.2)")
    parser.add_argument("--algo", choices=common.ALGOS, required=True)
    parser.add_argument("--params", help="JSON inline o ruta a un archivo JSON de hiperparámetros")
    parser.add_argument("--out", default="results", help="directorio de salida (default: results)")
    parser.add_argument("--limite", type=int, help="tope de filas por archivo (humo rápido)")
    parser.add_argument("--no-escala", action="store_true", help="omite el StandardScaler")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--master", default="local[*]", help="default: local[*]")
    parser.add_argument("--driver-memory", help="p. ej. 6g; en local suele hacer falta subirlo")
    parser.add_argument("--shuffle-partitions", type=int, default=32,
                        help="default 32: 200 particiones es excesivo para un solo nodo")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    parametros = common.leer_params(args.params)
    escala = not args.no_escala
    spark = construir_spark(
        args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        app_name=f"unsw-nb15-{args.algo}",
    )

    try:
        if args.input:
            filas = correr_suelto(
                spark, args.input, val=args.val, fraccion_val=args.fraccion_val,
                semilla=args.semilla, algo=args.algo, params=parametros,
                escala=escala, out=args.out, limite=args.limite,
            )
        else:
            filas = correr_grid(
                spark, args.grid, algo=args.algo, params=parametros,
                escala=escala, out=args.out, limite=args.limite,
            )
        _escribir_resumen(Path(args.out) / "metricas.csv", filas)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
