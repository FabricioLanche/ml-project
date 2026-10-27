"""Benchmark corto de lectura, features persistidas y un clasificador Spark."""

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "notebooks"))
import common
import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="G06")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--shuffle-partitions", type=int, default=16)
    args = parser.parse_args()

    from pyspark import StorageLevel

    spark = train.construir_spark(
        args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        app_name="unsw-benchmark-features",
    )
    ronda = BASE / "data" / "preprocessed" / "grid" / args.cell / f"r{args.fold}"
    inicio = time.perf_counter()
    train_df = common.cargar_parquet(spark, ronda / "train.parquet").limit(args.limit)
    val_df = common.cargar_parquet(spark, ronda / "val.parquet").limit(args.limit)
    feature_pipeline = common.construir_pipeline_caracteristicas(train_df, True)
    feature_model = feature_pipeline.fit(train_df)
    train_features = feature_model.transform(train_df).persist(StorageLevel.MEMORY_AND_DISK)
    val_features = feature_model.transform(val_df).persist(StorageLevel.MEMORY_AND_DISK)
    train_features.count()
    val_features.count()
    carga = time.perf_counter() - inicio

    inicio = time.perf_counter()
    classifier = common.crear_clasificador(
        "logreg",
        {"maxIter": 20, "regParam": 0.0, "elasticNetParam": 0.0},
        "features",
    )
    model = classifier.fit(train_features)
    pred = common.seleccionar_predicciones(model.transform(val_features))
    metricas = common.evaluar(pred)
    entrenamiento = time.perf_counter() - inicio
    print(f"benchmark cell={args.cell} fold={args.fold} limit={args.limit}")
    print(f"features_seconds={carga:.2f} training_seconds={entrenamiento:.2f}")
    print(f"metrics={metricas}")
    train_features.unpersist()
    val_features.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
