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
# Motor: una ronda para un k-fold
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
# Progreso
# ===========================================================================
def _imprimir(etiqueta: str, metricas: dict[str, float]) -> None:
    detalle = "  ".join(f"{k}={v:.4f}" for k, v in metricas.items())
    print(f"    {etiqueta}: {detalle}")
