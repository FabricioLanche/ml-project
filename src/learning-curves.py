"""Curvas de aprendizaje para las configuraciones ganadoras de la fase 2.

El experimento usa un split aleatorio fijo del 85 % de desarrollo. El 15 % de
prueba no se lee en ningún momento. Para cada modelo se reutiliza la celda de
preprocesamiento ganadora y sus hiperparámetros, y se entrena con fracciones
del conjunto de entrenamiento, evaluando siempre sobre la misma validación.

La métrica principal es el macro F1 calculado sobre las diez clases fijas. Si una
clase no aparece en una fracción de entrenamiento, el script falla en lugar de
omitirla del promedio, para que las fracciones sigan siendo comparables.

Salida: ``results/learning_curves/``.
"""

import argparse
import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
UTILIDADES = Path(__file__).resolve().parent / "utilities"
GRID = BASE / "data" / "preprocessed" / "grid_preprocess"
SALIDA = BASE / "results" / "learning_curves"
FRACCIONES = (0.25, 0.50, 0.75, 1.00)
SEMILLA = 42
VALIDACION = 0.20

CLASES = {
    0: "Analysis",
    1: "Backdoors",
    2: "Benign",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms",
}

LOGGER = logging.getLogger("learning-curves")
LOGGER.setLevel(logging.INFO)


def _cargar_utilidad(nombre: str, archivo: str):
    ruta = UTILIDADES / archivo
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


common = _cargar_utilidad("common", "training-utilities.py")
train = _cargar_utilidad("train", "training-functions.py")


def _ganadores(ruta: Path, modelos: list[str]) -> dict:
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    faltantes = set(modelos) - set(contenido)
    if faltantes:
        raise ValueError(f"faltan ganadores para: {', '.join(sorted(faltantes))}")
    return {modelo: contenido[modelo] for modelo in modelos}


def _cargar_desarrollo(spark, celda: str):
    """Une los ``val.parquet`` de las cinco rondas de una celda."""
    partes = []
    for ronda in range(1, 6):
        ruta = GRID / celda / f"r{ronda}" / "val.parquet"
        if not ruta.exists():
            raise FileNotFoundError(f"no existe {ruta}")
        partes.append(spark.read.parquet(str(ruta)))
    desarrollo = partes[0]
    for parte in partes[1:]:
        desarrollo = desarrollo.unionByName(parte)
    return desarrollo.cache()


def _clases_presentes(df) -> set[int]:
    return {int(fila["label"]) for fila in df.select("label").distinct().collect()}


def _verificar_clases(df, contexto: str) -> None:
    faltantes = sorted(set(CLASES) - _clases_presentes(df))
    if faltantes:
        nombres = ", ".join(CLASES[c] for c in faltantes)
        raise ValueError(f"{contexto}: faltan clases {nombres}")


def _fracciones(df, valor: float) -> dict[int, float]:
    return {clase: valor for clase in _clases_presentes(df)}


def _split_fijo(df, validacion: float, semilla: int):
    val = df.sampleBy("label", _fracciones(df, validacion), seed=semilla)
    entreno = df.exceptAll(val)
    _verificar_clases(val, "validación")
    _verificar_clases(entreno, "entrenamiento")
    return entreno.cache(), val.cache()


def _submuestrear(df, fraccion: float, semilla: int):
    if fraccion >= 1.0:
        return df
    muestra = df.sampleBy("label", _fracciones(df, fraccion), seed=semilla)
    _verificar_clases(muestra, f"fracción {fraccion}")
    return muestra.cache()


def _evaluar(train_df, val_df, modelo: str, params: dict, escala: bool) -> dict:
    esquema = params.get("pesoClases", "ninguno")
    train_ajuste = common.agregar_pesos(train_df, esquema)
    pipeline = common.construir_pipeline(train_ajuste, modelo, params, escala)
    ajustado = pipeline.fit(train_ajuste)
    pred = common.seleccionar_predicciones(ajustado.transform(val_df))
    metricas = common.evaluar(pred, clases=CLASES)
    return {
        "modelo": modelo,
        "n_entrenamiento": train_df.count(),
        "n_validacion": val_df.count(),
        **metricas,
    }


def _guardar(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", default=["rf", "logreg", "svm"])
    parser.add_argument("--validacion", type=float, default=VALIDACION)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--iteracion", type=int, default=1)
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory", default=None)
    parser.add_argument("--no-escala", action="store_true")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _args(argv)
    ganadores_path = (
        BASE / "results" / f"iteracion_{args.iteracion}" / "fase_2" / "ganadores.json"
    )
    ganadores = _ganadores(ganadores_path, args.modelos)
    spark = train.construir_spark(args.master, driver_memory=args.driver_memory)
    todas: list[dict] = []
    try:
        for modelo in args.modelos:
            info = ganadores[modelo]
            celda = info["celda"]
            params = json.loads(info["params"])
            LOGGER.info("modelo=%s celda=%s", modelo, celda)
            desarrollo = _cargar_desarrollo(spark, celda)
            entreno, val = _split_fijo(desarrollo, args.validacion, args.semilla)
            filas = []
            for fraccion in FRACCIONES:
                muestra = _submuestrear(entreno, fraccion, args.semilla)
                metricas = _evaluar(
                    muestra, val, modelo, params, not args.no_escala
                )
                fila = {"fraccion": fraccion, "celda": celda, **metricas}
                filas.append(fila)
                todas.append(fila)
                LOGGER.info(
                    "modelo=%s fraccion=%.2f macro_F1=%.4f",
                    modelo, fraccion, fila["promedio_F1"],
                )
                if muestra is not entreno:
                    muestra.unpersist()
            _guardar(args.salida / f"curva_{modelo}.csv", filas)
            entreno.unpersist()
            val.unpersist()
            desarrollo.unpersist()
    finally:
        spark.stop()
    _guardar(args.salida / "resumen.csv", todas)


if __name__ == "__main__":
    main()
