"""Orquestador de validación cruzada para las fases 1, 2 y 3.

La fase 1 compara las celdas de preprocesamiento con los hiperparámetros de la
iteración actual. La primera iteración usa los parámetros base y las siguientes
reutilizan el ganador de la fase 2 anterior. La fase 2 busca únicamente las
combinaciones declaradas para la celda ganadora. Ambas fases usan las rondas
materializadas por ``pre-process.py``. El conjunto ``base/test.parquet`` solo se
usa al final, después de la última iteración.

Cada fold se persiste como checkpoint en ``results/iteracion_N/checkpoints``.
``--reanudar`` reutiliza lo completado y ``--forzar`` lo ignora. ``--fase`` y
``--iteracion`` permiten ejecutar una parte explícita del flujo en jobs separados.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import train
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
DEFAULT_GRID = BASE / "data" / "preprocessed" / "grid_preprocess"
DEFAULT_BASE = BASE / "data" / "preprocessed" / "base"
DEFAULT_HIPERPARAMETROS = (
    BASE / "data" / "preprocessed" / "grid_hyperparams" / "hiperparametros.json"
)
DEFAULT_OUT = BASE / "results"
CHECKPOINT_VERSION = 1


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hiperparametros", "--params", dest="hiperparametros",
                        type=Path, default=DEFAULT_HIPERPARAMETROS,
                        help="JSON con la base y las variaciones de hiperparámetros")
    parser.add_argument("--fase", nargs="+", type=int, choices=(1, 2, 3),
                        default=[1, 2, 3], help="fases a ejecutar (default: 1 2 3)")
    parser.add_argument("--iteracion", type=int, default=None,
                        help="ejecuta solo esta iteración")
    parser.add_argument("--iteraciones", type=int, default=1,
                        help="iteraciones a ejecutar si no se usa --iteracion")
    parser.add_argument("--modelos", nargs="+", choices=common.ALGOS,
                        default=list(common.ALGOS))
    parser.add_argument("--celdas", nargs="+", default=None,
                        help="celdas de preprocesamiento a usar (smoke tests)")
    parser.add_argument("--rondas", nargs="+", type=int, default=None,
                        help="rondas a usar, 1..5 (smoke tests)")
    parser.add_argument("--limite-filas", type=int, default=None,
                        help="limita filas de train/val por ronda (smoke tests)")
    estado = parser.add_mutually_exclusive_group()
    estado.add_argument("--reanudar", action="store_true",
                        help="reutiliza checkpoints completados")
    estado.add_argument("--forzar", action="store_true",
                        help="ignora y reemplaza checkpoints")
    parser.add_argument("--no-escala", action="store_true")
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory")
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args(argv)


def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def _hash_params(params: dict) -> str:
    return hashlib.sha1(_params_key(params).encode("utf-8")).hexdigest()[:12]


def _hash_archivo(ruta: Path) -> str | None:
    if not ruta.exists():
        return None
    return hashlib.sha1(ruta.read_bytes()).hexdigest()


def _firma(args, params_base: dict, params_grid: dict, escala: bool) -> str:
    contenido = {
        "version": CHECKPOINT_VERSION,
        "base": params_base,
        "grilla": params_grid,
        "metadatos": _hash_archivo(args.base / "metadatos.json"),
        "manifiesto": _hash_archivo(args.grid / "manifiesto.csv"),
        "escala": escala,
        "limite_filas": args.limite_filas,
    }
    return hashlib.sha1(
        json.dumps(contenido, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _escribir_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporal.replace(ruta)


def _leer_hiperparametros(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"no existe {ruta}; ejecute notebooks/pre-process.py para generarlo"
        )
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    if "base" in contenido or "grilla" in contenido:
        if not ("base" in contenido and "grilla" in contenido):
            raise ValueError(f"{ruta} debe contener 'base' y 'grilla'")
        base = contenido["base"]
        grilla = contenido["grilla"]
    else:
        base = {
            modelo: dict(common.PARAMS_POR_DEFECTO[modelo])
            for modelo in common.ALGOS
        }
        grilla = contenido
    if not isinstance(base, dict) or not isinstance(grilla, dict):
        raise ValueError("base y grilla deben ser objetos")

    faltantes = set(common.ALGOS) - set(base)
    if faltantes:
        raise ValueError(f"faltan modelos en base: {', '.join(sorted(faltantes))}")
    faltantes = set(common.ALGOS) - set(grilla)
    if faltantes:
        raise ValueError(f"faltan modelos en grilla: {', '.join(sorted(faltantes))}")
    if not all(isinstance(valor, dict) for valor in base.values()):
        raise ValueError("base debe mapear cada modelo a un objeto")
    if not all(isinstance(valor, list) for valor in grilla.values()):
        raise ValueError("grilla debe mapear cada modelo a una lista")
    if not all(
        isinstance(config, dict)
        for configs in grilla.values()
        for config in configs
    ):
        raise ValueError("cada variación de grilla debe ser un objeto")

    grilla = {
        modelo: [{**base[modelo], **config} for config in configs]
        for modelo, configs in grilla.items()
    }
    return base, grilla


def _celdas(grid: Path, seleccion: list[str] | None):
    celdas = sorted(p for p in grid.glob("G*") if p.is_dir())
    if not celdas:
        raise ValueError(f"no hay celdas de preprocesamiento en {grid}")
    if not seleccion:
        return celdas
    pedidas = set(seleccion)
    resultado = [p for p in celdas if p.name in pedidas]
    faltantes = pedidas - {p.name for p in resultado}
    if faltantes:
        raise ValueError(f"celdas inexistentes: {', '.join(sorted(faltantes))}")
    return resultado


def _rondas(celda_dir: Path, seleccion: list[int] | None = None):
    rondas = sorted(
        (p for p in celda_dir.glob("r*") if p.is_dir()),
        key=lambda p: int(p.name[1:]),
    )
    if len(rondas) != 5:
        raise ValueError(f"{celda_dir} debe contener exactamente 5 rondas")
    if not seleccion:
        return rondas
    indices = sorted(set(seleccion))
    if any(indice < 1 or indice > len(rondas) for indice in indices):
        raise ValueError(f"rondas inválidas: {indices}; deben estar entre 1 y {len(rondas)}")
    return [rondas[indice - 1] for indice in indices]


def _cargar_celda_features(spark, rondas, escala, limite_filas=None):
    return [
        train.preparar_fold_features(spark, ronda, escala, limite_filas=limite_filas)
        for ronda in rondas
    ]


def _liberar_celda_features(folds):
    for fold in folds:
        fold.liberar()


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


def _ruta_fase1(root: Path, celda: str, modelo: str, ronda: str) -> Path:
    return root / "fase_1" / celda / modelo / f"{ronda}.json"


def _ruta_fase2(root: Path, modelo: str, celda: str, params: dict, ronda: str) -> Path:
    return root / "fase_2" / modelo / celda / _hash_params(params) / f"{ronda}.json"


def _ruta_fase3(root: Path, modelo: str) -> Path:
    return root / "fase_3" / f"{modelo}.json"


def _escribir_checkpoint(ruta, *, iteracion, fase, modelo, celda, ronda, params, metricas):
    _escribir_json(ruta, {
        "version": CHECKPOINT_VERSION,
        "iteracion": iteracion,
        "fase": fase,
        "modelo": modelo,
        "celda": celda,
        "ronda": ronda,
        "params": params,
        "metricas": metricas,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _leer_checkpoint(ruta, *, fase, modelo, celda, ronda, params):
    if not ruta.exists():
        return None
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    esperado = {
        "version": CHECKPOINT_VERSION,
        "fase": fase,
        "modelo": modelo,
        "celda": celda,
        "ronda": ronda,
    }
    for clave, valor in esperado.items():
        if contenido.get(clave) != valor:
            raise ValueError(
                f"checkpoint incompatible en {ruta}: {clave}={contenido.get(clave)!r}, "
                f"esperado {valor!r}"
            )
    if contenido.get("params") != params:
        raise ValueError(f"checkpoint incompatible en {ruta}: params distintos")
    return contenido


def _fila_checkpoint(contenido, *, hiperparametro="", reutilizado=False):
    return {
        "modelo": contenido["modelo"],
        "celda": contenido["celda"],
        "ronda": contenido["ronda"],
        "params": _params_key(contenido["params"]),
        "hiperparametro": hiperparametro,
        "reutilizado_fase_1": reutilizado,
        **contenido["metricas"],
    }


def _fases_a_limpiar(fases: set[int]) -> set[int]:
    limpiar = set(fases)
    if 1 in fases:
        limpiar.update({2, 3})
    if 2 in fases:
        limpiar.add(3)
    return limpiar


def _preparar_checkpoints(root: Path, firma: str, fases, *, reanudar, forzar):
    ruta_firma = root / "firma.json"
    firma_anterior = None
    if ruta_firma.exists():
        firma_anterior = json.loads(ruta_firma.read_text(encoding="utf-8")).get("firma")
    if forzar:
        a_limpiar = {1, 2, 3} if firma_anterior != firma else _fases_a_limpiar(set(fases))
        for fase in a_limpiar:
            shutil.rmtree(root / f"fase_{fase}", ignore_errors=True)
    existentes = root.exists() and any(p.is_file() for p in root.rglob("*"))
    if existentes and not reanudar and not forzar:
        raise ValueError(f"{root} ya contiene checkpoints; use --reanudar o --forzar")
    if existentes and reanudar:
        if firma_anterior is None:
            raise ValueError(f"{ruta_firma} no existe; no se puede reanudar sin firma")
        if firma_anterior != firma:
            raise ValueError(
                "la firma de checkpoints no coincide con los datos o hiperparámetros "
                "actuales; use --forzar"
            )
    root.mkdir(parents=True, exist_ok=True)
    if not existentes or forzar:
        _escribir_json(ruta_firma, {"firma": firma})


def _cargar_filas_fase1(root: Path, modelos) -> list[dict]:
    filas = []
    for ruta in sorted((root / "fase_1").glob("*/*/r*.json")):
        contenido = json.loads(ruta.read_text(encoding="utf-8"))
        if contenido.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"checkpoint incompatible en {ruta}: versión desconocida")
        if contenido.get("modelo") in modelos:
            filas.append(_fila_checkpoint(contenido))
    return filas


def _cargar_ganadores(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(
            f"no existe {ruta}; ejecute la fase anterior antes de continuar"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def _params_fase1(args, params_base: dict, iteracion: int) -> dict:
    if iteracion == 1:
        return {modelo: dict(params_base[modelo]) for modelo in args.modelos}
    ruta = args.out / f"iteracion_{iteracion - 1}" / "fase_2" / "ganadores.json"
    ganadores = _cargar_ganadores(ruta)
    faltantes = set(args.modelos) - set(ganadores)
    if faltantes:
        raise ValueError(
            f"faltan ganadores de la iteración {iteracion - 1}: "
            f"{', '.join(sorted(faltantes))}"
        )
    return {
        modelo: json.loads(ganadores[modelo]["params"])
        for modelo in args.modelos
    }


def _fase1(spark, grid, modelos, params_base, destino, escala, *,
           checkpoint_root, iteracion, celdas=None, rondas_sel=None,
           limite_filas=None):
    filas_fold, resumentes = [], []
    for celda in _celdas(grid, celdas):
        rondas = _rondas(celda, rondas_sel)
        filas_celda, pendientes = [], []
        for modelo in modelos:
            params = params_base[modelo]
            for ronda in rondas:
                ruta = _ruta_fase1(checkpoint_root, celda.name, modelo, ronda.name)
                contenido = _leer_checkpoint(
                    ruta, fase=1, modelo=modelo, celda=celda.name,
                    ronda=ronda.name, params=params,
                )
                if contenido is None:
                    pendientes.append((modelo, ronda.name, params))
                else:
                    filas_celda.append(_fila_checkpoint(contenido))
        if pendientes:
            print(f"\n=== CELDA {celda.name}: {len(pendientes)} unidades pendientes ===")
            folds = _cargar_celda_features(spark, rondas, escala, limite_filas)
            fold_por_ronda = {ronda.name: fold for ronda, fold in zip(rondas, folds)}
            try:
                for modelo, ronda, params in pendientes:
                    print(f"  >>> fase 1 {modelo} {celda.name}/{ronda} params={params}")
                    metricas = train.evaluar_features(
                        fold_por_ronda[ronda], modelo, params, escala
                    )
                    ruta = _ruta_fase1(checkpoint_root, celda.name, modelo, ronda)
                    _escribir_checkpoint(
                        ruta, iteracion=iteracion, fase=1, modelo=modelo,
                        celda=celda.name, ronda=ronda, params=params,
                        metricas=metricas,
                    )
                    filas_celda.append({
                        "modelo": modelo, "celda": celda.name, "ronda": ronda,
                        "params": _params_key(params), "hiperparametro": "",
                        "reutilizado_fase_1": False, **metricas,
                    })
            finally:
                _liberar_celda_features(folds)
        else:
            print(f"\n=== CELDA {celda.name}: completa, se reutiliza ===")
        filas_celda.sort(key=lambda fila: (fila["modelo"], int(fila["ronda"][1:])))
        for modelo in modelos:
            parciales = [fila for fila in filas_celda if fila["modelo"] == modelo]
            if len(parciales) != len(rondas):
                raise ValueError(
                    f"faltan resultados de {modelo} en {celda.name}: "
                    f"{len(parciales)}/{len(rondas)}"
                )
            resumentes.append({
                "modelo": modelo, "celda": celda.name,
                "params": _params_key(params_base[modelo]),
                "reutilizado_fase_1": False,
                **_resumen(parciales),
            })
            filas_fold.extend(parciales)
    _escribir_csv(destino / "resultado_por_fold.csv", filas_fold)
    _escribir_csv(destino / "resumen_general.csv", resumentes)
    return {
        modelo: max(
            (fila for fila in resumentes if fila["modelo"] == modelo),
            key=_orden_seleccion,
        )
        for modelo in modelos
    }, filas_fold


def _fase2(spark, grid, modelos, ganadores, fase1_filas, params_grid, destino, escala, *,
           checkpoint_root, iteracion, rondas_sel=None, limite_filas=None):
    filas_fold, resumentes = [], []
    por_celda: dict[str, list[str]] = {}
    for modelo in modelos:
        por_celda.setdefault(ganadores[modelo]["celda"], []).append(modelo)

    for celda_nombre in sorted(por_celda):
        celda = grid / celda_nombre
        rondas = _rondas(celda, rondas_sel)
        nombres_ronda = {ronda.name for ronda in rondas}
        modelos_celda = por_celda[celda_nombre]

        base_por_modelo, config_por_modelo = {}, {}
        for modelo in modelos_celda:
            params_base = json.loads(ganadores[modelo]["params"])
            clave_base = _params_key(params_base)
            base_rows = [
                {**fila, "hiperparametro": "hiperparametro_0",
                 "reutilizado_fase_1": True}
                for fila in fase1_filas
                if fila["modelo"] == modelo
                and fila["celda"] == celda_nombre
                and fila["params"] == clave_base
                and fila["ronda"] in nombres_ronda
            ]
            if len(base_rows) != len(rondas):
                raise ValueError(
                    f"fase 1 incompleta para {modelo}/{celda_nombre}: "
                    f"{len(base_rows)}/{len(rondas)} rondas"
                )
            base_por_modelo[modelo] = base_rows
            config_por_modelo[modelo] = [
                params for params in params_grid[modelo]
                if _params_key(params) != clave_base
            ]

        filas_config, pendientes = [], []
        for modelo in modelos_celda:
            for indice, params in enumerate(config_por_modelo[modelo], start=1):
                etiqueta = f"hiperparametro_{indice}"
                for ronda in rondas:
                    ruta = _ruta_fase2(
                        checkpoint_root, modelo, celda_nombre, params, ronda.name
                    )
                    contenido = _leer_checkpoint(
                        ruta, fase=2, modelo=modelo, celda=celda_nombre,
                        ronda=ronda.name, params=params,
                    )
                    if contenido is None:
                        pendientes.append((modelo, params, etiqueta, ronda.name))
                    else:
                        filas_config.append(
                            _fila_checkpoint(contenido, hiperparametro=etiqueta)
                        )
        if pendientes:
            print(f"\n=== FASE 2 {celda_nombre}: {len(pendientes)} unidades pendientes ===")
            folds = _cargar_celda_features(spark, rondas, escala, limite_filas)
            fold_por_ronda = {ronda.name: fold for ronda, fold in zip(rondas, folds)}
            try:
                for modelo, params, etiqueta, ronda in pendientes:
                    print(
                        f"  >>> fase 2 {modelo} {celda_nombre}/{ronda} "
                        f"{etiqueta} params={params}"
                    )
                    metricas = train.evaluar_features(
                        fold_por_ronda[ronda], modelo, params, escala
                    )
                    ruta = _ruta_fase2(
                        checkpoint_root, modelo, celda_nombre, params, ronda
                    )
                    _escribir_checkpoint(
                        ruta, iteracion=iteracion, fase=2, modelo=modelo,
                        celda=celda_nombre, ronda=ronda, params=params,
                        metricas=metricas,
                    )
                    filas_config.append({
                        "modelo": modelo, "celda": celda_nombre, "ronda": ronda,
                        "params": _params_key(params), "hiperparametro": etiqueta,
                        "reutilizado_fase_1": False, **metricas,
                    })
            finally:
                _liberar_celda_features(folds)
        else:
            print(f"\n=== FASE 2 {celda_nombre}: completa, se reutiliza ===")

        for modelo in modelos_celda:
            filas_fold.extend(base_por_modelo[modelo])
            resumentes.append({
                "modelo": modelo, "celda": celda_nombre,
                "hiperparametro": "hiperparametro_0",
                "params": _params_key(json.loads(ganadores[modelo]["params"])),
                "reutilizado_fase_1": True,
                **_resumen(base_por_modelo[modelo]),
            })
            for indice, params in enumerate(config_por_modelo[modelo], start=1):
                etiqueta = f"hiperparametro_{indice}"
                clave = _params_key(params)
                parciales = [
                    fila for fila in filas_config
                    if fila["modelo"] == modelo
                    and fila["params"] == clave
                    and fila["hiperparametro"] == etiqueta
                ]
                if len(parciales) != len(rondas):
                    raise ValueError(
                        f"faltan resultados de {modelo}/{etiqueta}: "
                        f"{len(parciales)}/{len(rondas)}"
                    )
                filas_fold.extend(parciales)
                resumentes.append({
                    "modelo": modelo, "celda": celda_nombre,
                    "hiperparametro": etiqueta,
                    "params": clave,
                    "reutilizado_fase_1": False,
                    **_resumen(parciales),
                })
    _escribir_csv(destino / "resultado_por_fold.csv", filas_fold)
    _escribir_csv(destino / "resumen_general.csv", resumentes)
    return {
        modelo: max(
            (fila for fila in resumentes if fila["modelo"] == modelo),
            key=_orden_seleccion,
        )
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
    pre.escribir_parquet_spark(train_t, train_path)
    pre.escribir_parquet_spark(test_t, test_path)
    return train_path, test_path


def _fase3(spark, base, ganadores, modelos, out, escala, *,
           checkpoint_root, iteracion, reanudar):
    final = out / "fase_3"
    clases = _clases(base)
    for modelo in modelos:
        ganador = ganadores[modelo]
        params = json.loads(ganador["params"])
        destino = final / modelo
        ruta_checkpoint = _ruta_fase3(checkpoint_root, modelo)
        if reanudar and ruta_checkpoint.exists():
            contenido = json.loads(ruta_checkpoint.read_text(encoding="utf-8"))
            if contenido.get("params") != params or contenido.get("celda") != ganador["celda"]:
                raise ValueError(
                    f"checkpoint de fase 3 incompatible para {modelo}; use --forzar"
                )
            if (destino / "metricas.json").exists() and (destino / "modelo").exists():
                print(f"\n=== FASE 3: {modelo} completado, se reutiliza ===")
                continue
        print(f"\n=== FASE 3: {modelo} ({ganador['celda']}) ===")
        entradas_train = destino / "_inputs"
        train_path, test_path = _materializar_final(
            base, ganador["celda"], entradas_train
        )
        metricas = train.correr_final(
            spark, train_path, test_path,
            algo=modelo, params=params, escala=escala,
            destino=destino, clases=clases,
        )
        _escribir_checkpoint(
            ruta_checkpoint, iteracion=iteracion, fase=3, modelo=modelo,
            celda=ganador["celda"], ronda="final", params=params,
            metricas=metricas,
        )


def _iteraciones(args):
    if args.iteracion is not None:
        if args.iteracion < 1:
            raise ValueError("--iteracion debe ser mayor o igual que 1")
        return [args.iteracion]
    if args.iteraciones < 1:
        raise ValueError("--iteraciones debe ser mayor o igual que 1")
    return list(range(1, args.iteraciones + 1))


def main(argv=None):
    args = _args(argv)
    fases = set(args.fase)
    params_base, params_grid = _leer_hiperparametros(args.hiperparametros)
    print(f"Hiperparámetros: {args.hiperparametros}")
    escala = not args.no_escala
    firma = _firma(args, params_base, params_grid, escala)
    iteraciones = _iteraciones(args)
    ultima = iteraciones[-1]

    spark = train.construir_spark(
        args.master, driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions, app_name="unsw-nb15-kfold",
    )
    try:
        for iteracion in iteraciones:
            raiz = args.out / f"iteracion_{iteracion}"
            checkpoint_root = raiz / "checkpoints"
            fases_iter = {fase for fase in fases if fase in (1, 2)}
            if iteracion == ultima and 3 in fases:
                fases_iter.add(3)
            if not fases_iter:
                continue
            _preparar_checkpoints(
                checkpoint_root, firma, fases_iter,
                reanudar=args.reanudar, forzar=args.forzar,
            )

            if 1 in fases_iter:
                print(f"\n=== ITERACION {iteracion}/{ultima}: FASE 1 ===")
                params_fase1 = _params_fase1(args, params_base, iteracion)
                fase1, _ = _fase1(
                    spark, args.grid, args.modelos, params_fase1,
                    raiz / "fase_1", escala,
                    checkpoint_root=checkpoint_root, iteracion=iteracion,
                    celdas=args.celdas, rondas_sel=args.rondas,
                    limite_filas=args.limite_filas,
                )
                _escribir_json(raiz / "fase_1" / "ganadores.json", fase1)

            if 2 in fases_iter:
                print(f"\n=== ITERACION {iteracion}/{ultima}: FASE 2 ===")
                ganadores_fase1 = _cargar_ganadores(
                    raiz / "fase_1" / "ganadores.json"
                )
                fase1_filas = _cargar_filas_fase1(checkpoint_root, args.modelos)
                ganadores = _fase2(
                    spark, args.grid, args.modelos, ganadores_fase1,
                    fase1_filas, params_grid, raiz / "fase_2", escala,
                    checkpoint_root=checkpoint_root, iteracion=iteracion,
                    rondas_sel=args.rondas, limite_filas=args.limite_filas,
                )
                _escribir_json(raiz / "fase_2" / "ganadores.json", ganadores)

            if 3 in fases_iter and iteracion == ultima:
                print(f"\n=== ITERACION {iteracion}/{ultima}: FASE 3 ===")
                ganadores = _cargar_ganadores(
                    raiz / "fase_2" / "ganadores.json"
                )
                _fase3(
                    spark, args.base, ganadores, args.modelos, args.out, escala,
                    checkpoint_root=checkpoint_root, iteracion=iteracion,
                    reanudar=args.reanudar,
                )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
