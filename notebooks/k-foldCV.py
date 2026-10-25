"""Orquestador de validación cruzada para las fases 1, 2 y 3.

La fase 1 compara las celdas de preprocesamiento con los hiperparámetros de la
iteración actual. La primera iteración usa los parámetros base y las siguientes
reutilizan el ganador de la fase 2 anterior. La fase 2 busca únicamente las combinaciones declaradas para la celda ganadora.
Ambas fases usan las cinco rondas materializadas por ``pre-process.py``.
El conjunto ``base/test.parquet`` solo se usa al final, después de la última
iteración.
"""

import argparse
import csv
import json
import statistics
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import train
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
DEFAULT_GRID = BASE / "data" / "preprocessed" / "grid"
DEFAULT_BASE = BASE / "data" / "preprocessed" / "base"
DEFAULT_OUT = BASE / "results"

DEFAULT_HYPERPARAMS = {
    "logreg": [
        {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0},
    ],
    "rf": [
        {"numTrees": 20, "maxDepth": 5},
        {"numTrees": 50, "maxDepth": 10},
    ],
    "svm": [
        {"maxIter": 100, "regParam": 0.0, "tol": 1e-6},
        {"maxIter": 200, "regParam": 0.1, "tol": 1e-6},
    ],
}


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--params", type=Path,
                        help="JSON con listas de hiperparámetros por modelo")
    parser.add_argument("--iteraciones", type=int, default=1,
                        help="repeticiones completas de fases 1 y 2 (default: 1)")
    parser.add_argument("--modelos", nargs="+", choices=common.ALGOS,
                        default=list(common.ALGOS))
    parser.add_argument("--no-escala", action="store_true")
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory")
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args(argv)


def _leer_params(ruta):
    if ruta is None:
        return DEFAULT_HYPERPARAMS
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    faltantes = set(common.ALGOS) - set(contenido)
    if faltantes:
        raise ValueError(f"faltan modelos en --params: {', '.join(sorted(faltantes))}")
    if any(not isinstance(v, list) or not v for v in contenido.values()):
        raise ValueError("--params debe mapear cada modelo a una lista no vacía")
    return contenido


def _rondas(celda_dir: Path):
    rondas = sorted(p for p in celda_dir.glob("r*") if p.is_dir())
    if len(rondas) != 5:
        raise ValueError(f"{celda_dir} debe contener exactamente 5 rondas")
    return rondas


def _evaluar_config(spark, rondas, modelo, params, destino, escala):
    filas = []
    for ronda in rondas:
        etiqueta = f"{ronda.parent.name}/{ronda.name}"
        print(f"  >>> {modelo} {etiqueta} params={params}")
        train_df = common.cargar_parquet(spark, ronda / "train.parquet")
        val_df = common.cargar_parquet(spark, ronda / "val.parquet")
        metricas = train.correr_ronda(
            train_df, val_df, modelo, params, escala,
            destino / ronda.name,
        )
        filas.append({
            "modelo": modelo,
            "celda": ronda.parent.name,
            "ronda": ronda.name,
            "params": json.dumps(params, sort_keys=True),
            **metricas,
        })
    return filas


def _resumen(filas):
    nombres = ("promedio_F1", "promedio_recall", "promedio_precision")
    resultado = {nombre: statistics.fmean(f[nombre] for f in filas) for nombre in nombres}
    resultado["DS_F1"] = statistics.stdev(
        (f["promedio_F1"] for f in filas), xbar=resultado["promedio_F1"]
    ) if len(filas) > 1 else 0.0
    resultado["accuracy"] = statistics.fmean(f["accuracy"] for f in filas)
    return resultado


def _escribir_csv(ruta, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        return
    campos = list(filas[0])
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def _orden_seleccion(fila):
    return (
        fila["promedio_F1"],
        fila["promedio_recall"],
        fila["promedio_precision"],
        -fila["DS_F1"],
    )


def _fase1(spark, grid, modelos, params_base, destino, escala):
    filas_fold, resumentes = [], []
    for modelo in modelos:
        for celda in sorted(p for p in grid.glob("G*") if p.is_dir()):
            rondas = _rondas(celda)
            parciales = _evaluar_config(
                spark, rondas, modelo, params_base[modelo],
                destino / celda.name / modelo, escala,
            )
            filas_fold.extend(parciales)
            resumen = _resumen(parciales)
            resumentes.append({
                "modelo": modelo, "celda": celda.name,
                "params": json.dumps(params_base[modelo], sort_keys=True),
                **resumen,
            })
    _escribir_csv(destino / "resultados_por_fold.csv", filas_fold)
    _escribir_csv(destino / "resumen_configuraciones.csv", resumentes)
    return {
        modelo: max((r for r in resumentes if r["modelo"] == modelo),
                    key=_orden_seleccion)
        for modelo in modelos
    }


def _fase2(spark, grid, modelos, ganadores, params_grid, destino, escala):
    filas_fold, resumentes = [], []
    for modelo in modelos:
        celda = grid / ganadores[modelo]["celda"]
        configuraciones = list(params_grid[modelo])
        for indice, params in enumerate(configuraciones, start=1):
            parciales = _evaluar_config(
                spark, _rondas(celda), modelo, params,
                destino / modelo / f"hiperparametro_{indice}", escala,
            )
            filas_fold.extend(parciales)
            resumentes.append({
                "modelo": modelo, "celda": celda.name,
                "hiperparametro": f"hiperparametro_{indice}",
                "params": json.dumps(params, sort_keys=True),
                **_resumen(parciales),
            })
    _escribir_csv(destino / "resultados_por_fold.csv", filas_fold)
    _escribir_csv(destino / "resumen_configuraciones.csv", resumentes)
    return {
        modelo: max((r for r in resumentes if r["modelo"] == modelo),
                    key=_orden_seleccion)
        for modelo in modelos
    }


def _clases(base):
    ruta = base / "metadatos.json"
    if not ruta.exists():
        return None
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    return {int(k): v for k, v in contenido.get("clases_por_indice", {}).items()}


def _cargar_preprocess():
    ruta = Path(__file__).with_name("pre-process.py")
    spec = importlib.util.spec_from_file_location("unsw_preprocess", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _materializar_final(base, celda, destino):
    """Ajusta la receta ganadora en todo train y la aplica al test.

    La rejilla de CV guarda recetas ajustadas por fold. Para la evaluación final
    se necesita una receta única ajustada con el 85 % completo, de modo que no se
    mezclen representaciones de folds ni se use información de test.
    """
    pre = _cargar_preprocess()
    train_parts = [
        pd.read_parquet(base / f"fold_{i}.parquet") for i in range(1, 6)
    ]
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.read_parquet(base / "test.parquet")
    params, train_t = pre.fit_transform(train_df, pre.CELDAS[celda])
    test_t = pre.transform(test_df, pre.CELDAS[celda], params)
    destino.mkdir(parents=True, exist_ok=True)
    train_path = destino / "train.parquet"
    test_path = destino / "test.parquet"
    train_t.to_parquet(train_path, index=False, compression="snappy")
    test_t.to_parquet(test_path, index=False, compression="snappy")
    return train_path, test_path


def main(argv=None):
    args = _args(argv)
    if args.iteraciones < 1:
        raise ValueError("--iteraciones debe ser mayor o igual que 1")
    params_grid = _leer_params(args.params)
    escala = not args.no_escala
    spark = train.construir_spark(
        args.master, driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions, app_name="unsw-nb15-kfold",
    )
    ganadores = None
    try:
        params_fase1 = {
            modelo: dict(common.PARAMS_POR_DEFECTO[modelo])
            for modelo in args.modelos
        }
        for iteracion in range(1, args.iteraciones + 1):
            raiz = args.out / f"iteracion_{iteracion}"
            print(f"\n=== ITERACION {iteracion}/{args.iteraciones}: FASE 1 ===")
            fase1 = _fase1(
                spark, args.grid, args.modelos, params_fase1,
                raiz / "fase_1", escala,
            )
            (raiz / "fase_1" / "ganadores.json").write_text(
                json.dumps(fase1, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n=== ITERACION {iteracion}/{args.iteraciones}: FASE 2 ===")
            ganadores = _fase2(
                spark, args.grid, args.modelos, fase1, params_grid,
                raiz / "fase_2", escala,
            )
            (raiz / "fase_2" / "ganadores.json").write_text(
                json.dumps(ganadores, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            params_fase1 = {
                modelo: json.loads(ganador["params"])
                for modelo, ganador in ganadores.items()
            }

        final = args.out / "fase_3"
        clases = _clases(args.base)
        for modelo, ganador in ganadores.items():
            params = json.loads(ganador["params"])
            destino = final / modelo
            print(f"\n=== FASE 3: {modelo} ({ganador['celda']}) ===")
            entradas_train = destino / "_inputs"
            train_path, test_path = _materializar_final(
                args.base, ganador["celda"], entradas_train
            )
            train.correr_final(
                spark, train_path, test_path,
                algo=modelo, params=params, escala=escala,
                destino=destino, clases=clases,
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()