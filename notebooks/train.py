from pathlib import Path
from dataclasses import dataclass

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


@dataclass
class FoldFeatures:
    """Features ajustadas solo con train de un fold y reutilizables entre modelos."""

    train: object
    val: object
    modelo: object

    def liberar(self) -> None:
        self.train.unpersist()
        self.val.unpersist()


def preparar_fold_features(spark, ronda, escala=True) -> FoldFeatures:
    """Lee un fold y ajusta una sola vez sus indexadores, encoder y scaler."""
    from pyspark import StorageLevel

    train_df = common.cargar_parquet(spark, ronda / "train.parquet")
    val_df = common.cargar_parquet(spark, ronda / "val.parquet")
    pipeline = common.construir_pipeline_caracteristicas(train_df, escala)
    modelo = pipeline.fit(train_df)
    train_features = modelo.transform(train_df).persist(StorageLevel.MEMORY_AND_DISK)
    val_features = modelo.transform(val_df).persist(StorageLevel.MEMORY_AND_DISK)
    return FoldFeatures(train_features, val_features, modelo)


def evaluar_features(fold: FoldFeatures, algo: str, params: dict,
                     escala: bool) -> dict[str, float]:
    """Entrena solo el clasificador sobre features ya ajustadas."""
    col_features = "features" if escala else "features_raw"
    clasificador = common.crear_clasificador(algo, params, col_features)
    modelo = clasificador.fit(fold.train)
    pred = common.seleccionar_predicciones(modelo.transform(fold.val))
    return common.evaluar(pred)


def correr_final(spark, entradas_train, entrada_test, *, algo: str, params: dict,
                 escala: bool = True, destino="results/fase_3",
                 clases: dict[int, str] | None = None) -> dict[str, float]:
    """Reentrena con todo el entrenamiento disponible y evalúa una vez en test.

    `entradas_train` puede ser una ruta o un iterable de parquets (por ejemplo,
    los cinco folds). El conjunto de test nunca participa en el ajuste.
    """
    if isinstance(entradas_train, (str, Path)):
        entradas_train = [entradas_train]
    train_df = None
    for ruta in entradas_train:
        parte = common.cargar_parquet(spark, ruta)
        train_df = parte if train_df is None else train_df.unionByName(parte)
    test_df = common.cargar_parquet(spark, entrada_test)
    pipeline = common.construir_pipeline(train_df, algo, params, escala)
    modelo = pipeline.fit(train_df)
    pred = common.seleccionar_predicciones(modelo.transform(test_df))

    ruta_salida = Path(destino)
    ruta_salida.mkdir(parents=True, exist_ok=True)
    pred.write.mode("overwrite").parquet(str(ruta_salida / "predicciones.parquet"))
    common.guardar_modelo(modelo, ruta_salida / "modelo")
    metricas = common.evaluar(pred, ruta_salida, clases=clases)
    return metricas
