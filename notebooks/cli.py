"""Adaptador de terminal para el motor de `train.py`.

Cascarón delgado: parsea argumentos, abre la sesión de Spark, invoca
`train.correr_suelto` o `train.correr_grid` y vuelca el resumen a
`<out>/metricas.csv`. Aquí no se entrena ni se evalúa nada: el motor vive en
`train.py` y la lógica de pipeline/evaluación en `common.py`.

Ejemplos
--------
Prueba de humo sobre un parquet suelto, hiperparámetros por defecto::

    python notebooks/cli.py --input <ruta>/train.parquet --algo logreg --limite 20000 --out results/humo

Corrida completa sobre la rejilla (10 celdas × 5 rondas)::

    python notebooks/cli.py --grid data/preprocessed/grid --algo rf --out results/base_rf

Corrida tuneada (JSON con los hiperparámetros del clasificador)::

    python notebooks/cli.py --grid data/preprocessed/grid --algo rf --params rf_params.json
"""

import argparse
import csv
from pathlib import Path

import common
import train


# ===========================================================================
# Reporte
# ===========================================================================
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
    spark = train.construir_spark(
        args.master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
        app_name=f"unsw-nb15-{args.algo}",
    )

    try:
        if args.input:
            filas = train.correr_suelto(
                spark, args.input, val=args.val, fraccion_val=args.fraccion_val,
                semilla=args.semilla, algo=args.algo, params=parametros,
                escala=escala, out=args.out, limite=args.limite,
            )
        else:
            filas = train.correr_grid(
                spark, args.grid, algo=args.algo, params=parametros,
                escala=escala, out=args.out, limite=args.limite,
            )
        _escribir_resumen(Path(args.out) / "metricas.csv", filas)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
