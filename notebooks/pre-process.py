"""Preprocesamiento y materialización de la rejilla — UNSW-NB15 (P2).

Traduce los hallazgos del EDA y las decisiones acordadas en
`.agents/PROPUESTA_ENTRENAMIENTO.md` a los artefactos que consume la etapa de
entrenamiento. Este script NO entrena ningún modelo: solo limpia, parte y transforma.

Produce, en `data/preprocessed/`:

* `base/`
    - `fold_1.parquet` … `fold_5.parquet`: particionan el 85 % de entrenamiento.
    - `test.parquet`: el 15 % restante, sellado hasta la fase 3.
    - `metadatos.json`: cortes, codificación del objetivo y decisiones de limpieza.
  Los archivos de `base/` guardan las features **sin transformar** (más `label`,
  `attack_cat`, `es_duplicado`, `Ltime` y `fold_id`).

* `grid_preprocess/`
    - `<celda>/r1..r5/{train,val}.parquet`: una variante por celda y por ronda. En la
      ronda `f`, la receta se fitea con los folds distintos de `f` (fold-train) y se
      aplica congelada a `train` y a `val` (el fold `f`).
    - `<celda>/r*/parametros.json`: los valores fiteados (modas y columnas log1p).
    - `manifiesto.csv`: qué receta define cada celda.
    - `estadisticas_muestreo.csv`: filas, grupos únicos y proporción duplicada por
      clase, celda y ronda, antes y después del deduplicado.

* `grid_hyperparams/`
    - `hiperparametros.json`: base y variaciones de hiperparámetros que consume el
      entrenamiento. Se edita en `notebooks/hiperparametros.py`.

Notas de alcance
----------------
* El **one-hot** de las categóricas y el **StandardScaler** son transversales: son
  idénticos en todas las celdas y no forman parte de la rejilla. Se aplican dentro del
  pipeline de MLlib en la etapa de entrenamiento (fit por fold), no aquí. Por eso los
  parquet de `grid_preprocess/` contienen columnas tabulares, no un vector de features.
* Los Parquet se escriben con nombres de columnas normalizados y variables booleanas
  convertidas a valores numéricos, de modo que Spark ML puede leerlos directamente.
* `StandardScaler` se aplicará también a Random Forest por decisión de uniformidad.
* `proto` conserva sus categorías originales en todas las celdas y recibe one-hot
  completo durante el ajuste del pipeline de MLlib.
* El deduplicado (en las celdas que lo activan) se aplica **solo** al fold-train; el
  fold-val se deja íntegro para medir sobre las filas repetidas tal como llegan.
* Cada `train.parquet` se limita por clase a 100,000 filas por defecto, con semilla 42.
  `--limite-por-categoria 0` lo desactiva; la validación y `base/` permanecen completas.
* El muestreo se hace antes del deduplicado para que las cuatro celdas de cada ronda
  partan de las mismas filas.
* La selección de columnas `log1p` se calcula sobre el fold-train completo antes del
  muestreo y se comparte entre celdas; en fase 3 se recalcula con todo el train.

Ejecución:  python3 notebooks/pre-process.py
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from hiperparametros import contenido as contenido_hiperparametros
except ModuleNotFoundError:
    from notebooks.hiperparametros import contenido as contenido_hiperparametros

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "preprocessed"
BASE_DIR = PROC_DIR / "base"
GRID_PREPROC_DIR = PROC_DIR / "grid_preprocess"
GRID_HYPER_DIR = PROC_DIR / "grid_hyperparams"
for _d in (DATA_DIR, BASE_DIR, GRID_PREPROC_DIR, GRID_HYPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "harshwardhanbhangale/unsw-complete-dataset"
REQ = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]

# ===========================================================================
# Esquema de columnas
# ===========================================================================
COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]

DTYPES = {
    "srcip": "string", "dstip": "string", "proto": "string",
    "state": "string", "service": "string", "attack_cat": "string",
    "sport": "Int32", "dsport": "Int32", "sbytes": "Int32", "dbytes": "Int32",
    "sttl": "Int16", "dttl": "Int16", "sloss": "Int16", "dloss": "Int16",
    "Spkts": "Int16", "Dpkts": "Int16", "swin": "Int16", "dwin": "Int16",
    "stcpb": "Int64", "dtcpb": "Int64", "smeansz": "Int16", "dmeansz": "Int16",
    "trans_depth": "Int16", "res_bdy_len": "Int32", "Stime": "Int32", "Ltime": "Int32",
    "ct_state_ttl": "Int8", "ct_flw_http_mthd": "Int8", "ct_ftp_cmd": "Int8",
    "ct_srv_src": "Int8", "ct_srv_dst": "Int8", "ct_dst_ltm": "Int8",
    "ct_src_ ltm": "Int8", "ct_src_dport_ltm": "Int8", "ct_dst_sport_ltm": "Int8",
    "ct_dst_src_ltm": "Int8",
    "dur": "Float32", "Sload": "Float64", "Dload": "Float64",
    "Sjit": "Float32", "Djit": "Float32", "Sintpkt": "Float32", "Dintpkt": "Float32",
    "tcprtt": "Float32", "synack": "Float32", "ackdat": "Float32",
    "is_sm_ips_ports": "boolean", "is_ftp_login": "boolean", "Label": "boolean",
}

NA_VALUES = {
    "sport": ["0x000b", "0x000c", "-"],
    "dsport": ["0xc0a8", "0xcc09", "0x20205321", "-"],
    "ct_ftp_cmd": [" "],
    "is_ftp_login": ["2", "4"],
    "state": ["no"],
}

LABEL_COL = "label"
CLASE_COL = "attack_cat"
DUP_COL = "es_duplicado"
GRUPO_COL = "_grupo_duplicado"
CONTROL = {LABEL_COL, CLASE_COL, DUP_COL, GRUPO_COL}

BLANCOS_ESTRUCTURALES = ["is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd"]
IMPUTAR = ["dsport", "sport", "state"]

CT_COLS = [
    "ct_state_ttl", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_srv_src", "ct_srv_dst",
    "ct_flw_http_mthd", "ct_ftp_cmd",
]

BINARIAS = ["is_sm_ips_ports", "is_ftp_login"]

# Columnas que nunca entran al espacio de características. Ltime se conserva aparte en
# `base/` para ordenar, particionar y auditar; se descarta en `grid_preprocess/`.
ELIMINAR_ESTATICO = {
    "attack_cat": "variable objetivo (se persiste aparte como nombre de clase)",
    "Label": "versión binaria del objetivo (Benign <=> Label=0)",
    "srcip": "identidad de host, 43 valores únicos: memoriza hosts",
    "dstip": "identidad de host, 47 valores únicos: memoriza hosts",
    "Stime": "marca de tiempo absoluta: correlaciona con la sesión de captura",
    "Ltime": "marca de tiempo absoluta: se usa solo para ordenar y particionar",
}

K = 5
FRACCION_TRAIN = 0.85
SEMILLA = 42
LIMITE_POR_CATEGORIA_DEFAULT = 100_000
SEMILLA_SUBMUESTREO = 42
UMBRAL_LOG1P = 1.0

# ===========================================================================
# Rejilla de preprocesamiento (4 celdas, numeración continua G00..G03)
# ===========================================================================
# Solo varían la presencia de ct_* y el deduplicado. Los puertos siempre se agrupan
# por franja IANA y proto conserva todas sus categorías para el one-hot posterior.
# El log1p (umbral 1), StandardScaler y la imputación por moda son transversales.
_BASE_CELL = dict(ct=True, dedup=False)

_OVERRIDES = {
    "G00": {},                              # base
    "G01": dict(ct=False),                  # sin ct_*
    "G02": dict(dedup=True),                # deduplicar
    "G03": dict(dedup=True, ct=False),      # deduplicar + sin ct_*
}
CELDAS = {cid: {**_BASE_CELL, **ov} for cid, ov in _OVERRIDES.items()}


# ===========================================================================
# Grilla de hiperparámetros
# ===========================================================================
def escribir_grilla_hiperparametros() -> Path:
    """Materializa la base y las variaciones de hiperparámetros en JSON."""
    destino = GRID_HYPER_DIR / "hiperparametros.json"
    destino.write_text(
        json.dumps(contenido_hiperparametros(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destino


# ===========================================================================
# Descarga
# ===========================================================================
def asegurar_dataset() -> None:
    if all((DATA_DIR / f).exists() for f in REQ):
        print("Dataset ya presente en data/raw/")
        return
    print("Descargando dataset desde Kaggle…")
    import kagglehub
    tmp = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    for f in REQ:
        origen = tmp / f
        if origen.exists():
            shutil.copy2(origen, DATA_DIR / f)
            print(f"  → {f}")


# ===========================================================================
# Recetas: ajuste (fit) y aplicación (transform)
# ===========================================================================
def _columnas_numericas(df: pd.DataFrame) -> list[str]:
    """Numéricas transformables (excluye control y binarias predefinidas)."""
    return [
        c for c in df.columns
        if c not in CONTROL and c not in BINARIAS
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def _franja(puerto: pd.Series) -> pd.Categorical:
    """Categoría de rango de puerto: bien conocido / registrado / dinámico."""
    valores = np.select(
        [puerto.isna(), puerto <= 1023, puerto <= 49151],
        ["desconocido", "bien_conocido", "registrado"],
        default="dinamico",
    )
    return pd.Categorical(
        valores, categories=["bien_conocido", "registrado", "dinamico", "desconocido"],
    )


def _rellenar_nulos(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Imputa los tokens corruptos con la moda del fold-train (decisión fija).

    Son residuos de parseo (puertos y `state`): ~320 filas, 100 % `Benign`, y la moda
    preserva un valor real en vez de inventar un puerto 0.
    """
    out = df.copy()
    for col in IMPUTAR:
        if col in out.columns:
            out[col] = out[col].fillna(params["modas"][col])
    return out


def _derivar_puertos(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva la franja IANA de los puertos y elimina los valores crudos."""
    out = df
    for pre in ("sport", "dsport"):
        if pre in out.columns:
            out[f"{pre}_franja"] = _franja(out[pre])
    return out.drop(columns=["sport", "dsport"])


def _filtrar_ct(df: pd.DataFrame, celda: dict) -> pd.DataFrame:
    if celda["ct"]:
        return df
    return df.drop(columns=[c for c in CT_COLS if c in df.columns])


def submuestrear_por_categoria(
    train: pd.DataFrame,
    limite_por_categoria: int,
    semilla: int = SEMILLA_SUBMUESTREO,
) -> pd.DataFrame:
    """Limita cada clase de train a una muestra exacta y reproducible."""
    if limite_por_categoria < 0:
        raise ValueError("limite_por_categoria debe ser mayor o igual a cero")
    if limite_por_categoria == 0:
        return train

    muestras = []
    resumen = []
    for clase, grupo in train.groupby(LABEL_COL, observed=True, sort=True):
        cantidad = min(len(grupo), limite_por_categoria)
        muestra = grupo if len(grupo) <= limite_por_categoria else grupo.sample(
            n=limite_por_categoria, random_state=semilla
        )
        muestras.append(muestra)
        resumen.append(f"{clase} {len(grupo):,} → {cantidad:,}")

    print(
        f"  submuestreo materializado por clase (límite "
        f"{limite_por_categoria:,}): " + ", ".join(resumen)
    )
    return pd.concat(muestras, ignore_index=True)


def _resumen_grupos_por_clase(df: pd.DataFrame) -> dict[int, dict]:
    if GRUPO_COL not in df.columns:
        return {}

    resumen = {}
    for etiqueta, grupo in df.groupby(LABEL_COL, observed=True, sort=True):
        duplicadas = int(grupo[GRUPO_COL].duplicated(keep=False).sum())
        resumen[int(etiqueta)] = {
            "clase": str(grupo[CLASE_COL].iloc[0]),
            "filas": int(len(grupo)),
            "grupos_unicos": int(grupo[GRUPO_COL].nunique()),
            "filas_en_grupos_repetidos": duplicadas,
            "proporcion_filas_duplicadas": duplicadas / len(grupo),
        }
    return resumen


def seleccionar_log1p_cols(train: pd.DataFrame) -> list[str]:
    """Selecciona columnas usando la distribución completa del fold-train."""
    params = {"modas": {}}
    for col in IMPUTAR:
        modo = train[col].dropna().mode()
        v = modo.iat[0] if len(modo) else np.nan
        params["modas"][col] = v.item() if hasattr(v, "item") else v

    w = _rellenar_nulos(train, params)
    w = _derivar_puertos(w)
    columnas = []
    for col in _columnas_numericas(w):
        serie = pd.to_numeric(w[col], errors="coerce").astype("float64")
        if abs(float(serie.skew())) > UMBRAL_LOG1P:
            columnas.append(col)
    return columnas


def fit_transform(
    train: pd.DataFrame,
    celda: dict,
    *,
    limite_por_categoria: int | None = None,
    semilla: int = SEMILLA_SUBMUESTREO,
    log1p_cols: list[str] | None = None,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Fitea la receta y devuelve parámetros, train transformado y estadísticas.

    Todas las celdas parten de la misma muestra; el deduplicado se aplica después.
    `limite_por_categoria=None` conserva todas las filas, como en la fase final.
    """
    columnas_log1p = (
        seleccionar_log1p_cols(train) if log1p_cols is None else log1p_cols
    )
    fit = train
    if limite_por_categoria is not None:
        fit = submuestrear_por_categoria(fit, limite_por_categoria, semilla)
    resumen_muestra = _resumen_grupos_por_clase(fit)

    if celda["dedup"]:
        feat = [c for c in fit.columns if c not in CONTROL]
        fit = fit.drop_duplicates(subset=feat)
    resumen_train = _resumen_grupos_por_clase(fit)
    estadisticas = [
        {
            "label": etiqueta,
            "clase": valores["clase"],
            "filas_muestra": valores["filas"],
            "grupos_unicos_muestra": valores["grupos_unicos"],
            "filas_en_grupos_repetidos_muestra": valores["filas_en_grupos_repetidos"],
            "proporcion_filas_duplicadas_muestra": valores["proporcion_filas_duplicadas"],
            "filas_train": resumen_train[etiqueta]["filas"],
            "grupos_unicos_train": resumen_train[etiqueta]["grupos_unicos"],
            "filas_en_grupos_repetidos_train":
                resumen_train[etiqueta]["filas_en_grupos_repetidos"],
            "proporcion_filas_duplicadas_train":
                resumen_train[etiqueta]["proporcion_filas_duplicadas"],
        }
        for etiqueta, valores in resumen_muestra.items()
    ]

    params: dict = {"modas": {}, "log1p_cols": list(columnas_log1p)}
    for col in IMPUTAR:
        modo = fit[col].dropna().mode()
        v = modo.iat[0] if len(modo) else np.nan
        params["modas"][col] = v.item() if hasattr(v, "item") else v

    w = _rellenar_nulos(fit, params)
    w = _derivar_puertos(w)
    w = _filtrar_ct(w, celda)

    w = _aplicar_log1p(w, params)
    return params, w.drop(columns=["Ltime", GRUPO_COL], errors="ignore"), estadisticas


def transform(df: pd.DataFrame, celda: dict, params: dict) -> pd.DataFrame:
    """Aplica una receta ya fiteada (para el fold-val o el test)."""
    out = _rellenar_nulos(df, params)
    out = _derivar_puertos(out)
    out = _filtrar_ct(out, celda)
    out = _aplicar_log1p(out, params)
    return out.drop(columns=["Ltime", GRUPO_COL], errors="ignore")


def preparar_salida_spark(df: pd.DataFrame) -> pd.DataFrame:
    """Adapta un resultado tabular al esquema que consume Spark ML.

    La normalización se hace una sola vez al persistir el Parquet: Spark no
    necesita renombrar columnas ni castear booleanos en cada lectura.
    """
    out = df.copy()
    out.columns = [
        re.sub(r"_+", "_", col.replace(" ", "_").replace("-", "_"))
        for col in out.columns
    ]
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col].dtype):
            out[col] = out[col].astype("float64")
    return out


def escribir_parquet_spark(df: pd.DataFrame, ruta: Path) -> None:
    preparar_salida_spark(df).to_parquet(
        ruta, index=False, compression="snappy"
    )


def validar_salidas_preprocesamiento() -> None:
    """Falla si la materialización no dejó todos los artefactos requeridos por k-fold."""
    requeridos = [
        BASE_DIR / "metadatos.json",
        BASE_DIR / "test.parquet",
        *(BASE_DIR / f"fold_{i}.parquet" for i in range(1, K + 1)),
        GRID_PREPROC_DIR / "manifiesto.csv",
        GRID_HYPER_DIR / "hiperparametros.json",
    ]
    for celda in CELDAS:
        for ronda in range(1, K + 1):
            directorio = GRID_PREPROC_DIR / celda / f"r{ronda}"
            requeridos.extend((
                directorio / "train.parquet",
                directorio / "val.parquet",
                directorio / "parametros.json",
            ))

    faltantes = [ruta for ruta in requeridos if not ruta.is_file() or ruta.stat().st_size == 0]
    if faltantes:
        detalle = "\n".join(f"  - {ruta}" for ruta in faltantes)
        raise FileNotFoundError(
            "Preprocesamiento incompleto; faltan artefactos requeridos por k-fold:\n"
            f"{detalle}"
        )


def _aplicar_log1p(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    for c in params["log1p_cols"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").clip(lower=0).astype("float64")
            df[c] = np.log1p(x).astype("float32")
    return df


# ===========================================================================
# Partición y folds
# ===========================================================================
def corte_alineado(ltime: np.ndarray, fraccion: float) -> int:
    """Índice de corte desplazado a un límite de Ltime.

    Evita partir un grupo de filas con idéntica marca de tiempo (y por tanto un grupo
    de duplicados) entre dos particiones.
    """
    n = len(ltime)
    i = int(fraccion * n)
    while 0 < i < n and ltime[i] == ltime[i - 1]:
        i += 1
    return i


def asignar_folds(label: np.ndarray, grupo: np.ndarray, k: int, semilla: int) -> np.ndarray:
    """Estratificado por clase y agrupado por duplicado.

    Cada grupo (conjunto de filas idénticas) cae entero en un fold, y los grupos se
    reparten de forma codiciosa para equilibrar el recuento de cada clase entre folds.
    """
    info = pd.DataFrame({"grupo": grupo, "label": label})
    tamanos = info.groupby("grupo", observed=True).size()
    etiqueta = info.drop_duplicates("grupo").set_index("grupo")["label"]
    grupos = pd.DataFrame({"label": etiqueta, "tam": tamanos})

    asignacion: dict = {}
    for _, sub in grupos.groupby("label", observed=True):
        sub = sub.sample(frac=1.0, random_state=semilla).sort_values("tam", ascending=False)
        conteo = np.zeros(k, dtype=np.int64)
        for g, tam_g in zip(sub.index.to_numpy(), sub["tam"].to_numpy()):
            j = int(np.argmin(conteo))
            conteo[j] += tam_g
            asignacion[g] = j
    return np.fromiter((asignacion[g] for g in grupo), dtype=np.int8,
                       count=len(grupo))


# ===========================================================================
# 1. Carga y limpieza estática
# ===========================================================================
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limite-por-categoria",
        type=int,
        default=LIMITE_POR_CATEGORIA_DEFAULT,
        help="máximo de filas por clase en train de cada fold; 0 desactiva",
    )
    args = parser.parse_args(argv)
    if args.limite_por_categoria < 0:
        parser.error("--limite-por-categoria debe ser mayor o igual a cero")

    print("=== PREPROCESAMIENTO: split + folds + rejilla (sin entrenamiento) ===")
    print(
        f"Límite de train por categoría: "
        f"{args.limite_por_categoria:,} "
        f"(semilla de muestreo {SEMILLA_SUBMUESTREO})"
        if args.limite_por_categoria
        else "Límite de train por categoría: desactivado"
    )
    ruta_hiper = escribir_grilla_hiperparametros()
    print(f"[0] grilla de hiperparámetros → {ruta_hiper}")
    asegurar_dataset()

    # ---------------- Carga ----------------
    dfs = [
        pd.read_csv(
            DATA_DIR / f"UNSW-NB15_{i}.csv",
            header=None, names=COLUMNS, dtype=DTYPES,
            na_values=NA_VALUES, low_memory=False,
        )
        for i in range(1, 5)
    ]
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    df["attack_cat"] = (
        df["attack_cat"].fillna("Benign").str.strip().replace({"Backdoor": "Backdoors"})
    )
    inc = int(((df["attack_cat"] != "Benign") & (df["Label"] == 0)).sum())
    if inc:
        raise ValueError("attack_cat no se corresponde 1:1 con Label; revisar limpieza.")
    for col in ["state", "service", "attack_cat"]:
        df[col] = df[col].astype("category")

    cols_hash = [c for c in df.columns if c not in ("attack_cat", "Label")]
    df = df.sort_values("Ltime", kind="stable").reset_index(drop=True)
    grupo = pd.util.hash_pandas_object(df[cols_hash], index=False).to_numpy()
    es_dup = df.duplicated(keep=False).to_numpy()
    dup_exceso = int(df.duplicated().sum())
    n = len(df)
    print(f"\n[1] cargado: {n:,} filas · filas duplicadas exactas {dup_exceso:,} "
          f"({dup_exceso / n * 100:.2f} %)" )

    for col in BLANCOS_ESTRUCTURALES:
        relleno = False if df[col].dtype == "boolean" else 0
        df[col] = df[col].fillna(relleno)

    # ---------------- Split 85/15 ----------------
    ltime = df["Ltime"].to_numpy()
    corte = corte_alineado(ltime, FRACCION_TRAIN)
    c_indice = min(corte, n - 1)
    idx_train = np.arange(0, corte)
    idx_test = np.arange(corte, n)
    print(f"[2] corte 85/15 en fila {corte:,} "
          f"(Ltime {pd.to_datetime(int(ltime[c_indice]), unit='s')}); "
          f"train {len(idx_train):,} · test {len(idx_test):,}")

    clases = sorted(df["attack_cat"].unique().tolist())
    mapa = {c: i for i, c in enumerate(clases)}
    label = df["attack_cat"].map(mapa).astype("int8").to_numpy()

    features = [c for c in df.columns if c not in ELIMINAR_ESTATICO]
    cols_base = features + [LABEL_COL, CLASE_COL, DUP_COL, GRUPO_COL, "Ltime"]
    base = df[features].copy()
    base[LABEL_COL] = label
    base[CLASE_COL] = df["attack_cat"].to_numpy()
    base[DUP_COL] = es_dup.astype("int8")
    base[GRUPO_COL] = grupo
    base["Ltime"] = ltime

    # ---------------- Folds ----------------
    fold = np.full(n, -1, dtype="int8")
    fold[idx_train] = asignar_folds(
        label[idx_train], grupo[idx_train], K, SEMILLA
    )
    base["fold_id"] = fold
    print(f"[3] folds asignados (K={K}); "
          + " · ".join(f"f{j+1}={int((fold == j).sum()):,}" for j in range(K)))

    # ---------------- Persistencia de base/ ----------------
    print("\n[4] escribiendo data/preprocessed/base/")
    base_test = base.iloc[idx_test].drop(columns=[GRUPO_COL]).reset_index(drop=True)
    escribir_parquet_spark(base_test, BASE_DIR / "test.parquet")
    tamanos_fold = {}
    for j in range(K):
        sub = base.loc[fold == j].drop(columns=[GRUPO_COL]).reset_index(drop=True)
        escribir_parquet_spark(sub, BASE_DIR / f"fold_{j + 1}.parquet")
        tamanos_fold[f"fold_{j + 1}"] = int(len(sub))
        print(f"  fold_{j + 1}.parquet  {len(sub):>9,} filas × {sub.shape[1]} cols")
    print(f"  test.parquet     {len(base_test):>9,} filas × {base_test.shape[1]} cols")

    # ---------------- Materialización de la rejilla ----------------
    print(f"\n[5] materializando la rejilla de preprocesamiento ({len(CELDAS)} celdas × {K} rondas)")
    for p in GRID_PREPROC_DIR.glob("G*"):
        if p.is_dir() and p.name not in CELDAS:
            shutil.rmtree(p)
            print(f"  celda obsoleta eliminada: {p.name}")
    train_base = base.loc[fold >= 0].reset_index(drop=True)
    manifiesto = []
    estadisticas_muestreo = []
    log1p_por_ronda = {}
    for cid, celda in CELDAS.items():
        celda_dir = GRID_PREPROC_DIR / cid
        celda_dir.mkdir(parents=True, exist_ok=True)
        for j in range(K):
            ronda_dir = celda_dir / f"r{j + 1}"
            ronda_dir.mkdir(parents=True, exist_ok=True)
            fit_df = train_base.loc[train_base["fold_id"] != j, cols_base]
            val_df = train_base.loc[train_base["fold_id"] == j, cols_base]
            if j not in log1p_por_ronda:
                log1p_por_ronda[j] = seleccionar_log1p_cols(fit_df)
            params, train_t, estadisticas = fit_transform(
                fit_df, celda,
                limite_por_categoria=args.limite_por_categoria,
                semilla=SEMILLA_SUBMUESTREO,
                log1p_cols=log1p_por_ronda[j],
            )
            estadisticas_muestreo.extend(
                {"celda": cid, "ronda": j + 1, **fila}
                for fila in estadisticas
            )
            val_t = transform(val_df, celda, params)
            escribir_parquet_spark(train_t, ronda_dir / "train.parquet")
            escribir_parquet_spark(val_t, ronda_dir / "val.parquet")
            with open(ronda_dir / "parametros.json", "w") as fh:
                json.dump(params, fh, ensure_ascii=False, indent=2, default=str)
        manifiesto.append({
            "celda": cid, "ct": celda["ct"], "deduplicar": celda["dedup"],
            "puertos": "solo_franja", "proto": "completo",
        })
        print(f"  {cid}: 5 rondas materializadas")
    pd.DataFrame(manifiesto).to_csv(GRID_PREPROC_DIR / "manifiesto.csv", index=False)
    pd.DataFrame(estadisticas_muestreo).to_csv(
        GRID_PREPROC_DIR / "estadisticas_muestreo.csv", index=False
    )

    # Conteo de tokens corruptos (solo para reporte: se imputan con la moda).
    corrup = {
        col: {
            "totales": int(df[col].isna().sum()),
            "train": int(df.iloc[idx_train][col].isna().sum()),
            "test": int(df.iloc[idx_test][col].isna().sum()),
        }
        for col in IMPUTAR
    }

    # ---------------- Metadatos ----------------
    meta = {
        "filas_totales": int(n),
        "duplicados_conservados": dup_exceso,
        "filas_marcadas_como_duplicadas": int(es_dup.sum()),
        "fracciones": {"train": FRACCION_TRAIN, "test": round(1 - FRACCION_TRAIN, 2)},
        "corte_train_test": int(corte),
        "Ltime_corte_train_test": str(pd.to_datetime(int(ltime[c_indice]), unit="s")),
        "K": K,
        "semilla": SEMILLA,
        "limite_por_categoria": args.limite_por_categoria,
        "semilla_submuestreo": SEMILLA_SUBMUESTREO,
        "submuestreo_comun_antes_de_deduplicar": True,
        "log1p_seleccion_fold_train_completo": True,
        "tamanos_folds": tamanos_fold,
        "columnas_features": features,
        "columnas_control": [LABEL_COL, CLASE_COL, DUP_COL],
        "columnas_eliminadas": ELIMINAR_ESTATICO,
        "blancos_estructurales_imputados_a_cero": BLANCOS_ESTRUCTURALES,
        "tokens_corruptos_imputados": IMPUTAR,
        "filas_tokens_corruptos": corrup,
        "clases": {c: v for c, v in mapa.items()},
        "clases_por_indice": {str(v): c for c, v in mapa.items()},
        "rejilla": manifiesto,
        "umbral_log1p": UMBRAL_LOG1P,
    }
    with open(BASE_DIR / "metadatos.json", "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
    print("\n  metadatos.json escrito")

    # ---------------- Verificaciones ----------------
    print("\n[6] verificaciones")
    conteo_clases = pd.DataFrame({
        "clase": clases,
        "train": [int((label[idx_train] == mapa[c]).sum()) for c in clases],
        "test": [int((label[idx_test] == mapa[c]).sum()) for c in clases],
    })
    print(conteo_clases.to_string(index=False))

    n_cls_train = int(pd.Series(label[idx_train]).nunique())
    n_cls_test = int(pd.Series(label[idx_test]).nunique())
    print(f"  [V1] clases: train={n_cls_train}, test={n_cls_test} (se requieren 10)")

    print("  [V1b] recuento de clases por fold (train):")
    fold_clases = pd.DataFrame(
        {f"fold_{j + 1}": [int(((fold == j) & (label == i)).sum())
                           for i in range(len(clases))]
         for j in range(K)},
        index=clases,
    )
    print(fold_clases.to_string())
    ausentes = {
        f"fold_{j + 1}": [clases[i] for i in range(len(clases))
                          if int(((fold == j) & (label == i)).sum()) == 0]
        for j in range(K)
    }
    sin_alguna = {k: v for k, v in ausentes.items() if v}
    print(f"  folds con clases ausentes: {sin_alguna or 'ninguno (10/10 en cada fold)'}")

    # [V2] Grupos de filas idénticas que cruzan train/test.
    g_train = set(grupo[idx_train].tolist())
    g_test = set(grupo[idx_test].tolist())
    cruce = len(g_train & g_test)
    print(f"  [V2] grupos de filas idénticas que cruzan train/test: {cruce}")
    if cruce:
        raise AssertionError("Hay filas idénticas repartidas entre train y test")

    # [V3] El corte es cronológico y no parte un empate de Ltime. Nota: a diferencia
    # del esquema 75/15/10 (donde val amortiguaba a test), aquí los ~100 primeros
    # registros de test comparten contexto de captura con el final de train; eso no es
    # fuga (el contexto de ct_* es retrospectivo por definición), por lo que no se
    # verifica un porcentaje de solape sino la alineación del corte.
    corte_ok = bool(ltime[c_indice] != ltime[corte - 1])
    print(f"  [V3] corte alineado a límite de Ltime: {corte_ok}")
    if not corte_ok:
        raise AssertionError("El corte 85/15 parte un grupo de Ltime idéntico")

    print("  [V4] tokens corruptos por split (imputados con la moda):")
    for col, c in corrup.items():
        print(f"  {col:8s} {c['totales']:>4} nulos · train={c['train']:>4} · test={c['test']:>4}")

    validar_salidas_preprocesamiento()
    print("\nPreprocesamiento completado y artefactos verificados.")
    print(f"  base/  → 5 folds + test  (sin transformar)")
    print(f"  grid_preprocess/  → {len(CELDAS)} celdas × {K} rondas (train + val por ronda)")
    print("  grid_hyperparams/ → hiperparametros.json")


if __name__ == "__main__":
    main()
