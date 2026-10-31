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
  `attack_cat`, `es_duplicado`, `Ltime` y `fold_id`), con los nombres de columna y los
  tipos originales (sin normalizar): es lo que espera la receta (`fit_transform` /
  `transform`), de modo que aplicarla sobre `base/` equivale a aplicarla en memoria.

* `grid_preprocess/`
    - `<celda>/r1..r5/{train,val}.parquet`: una variante por celda y por ronda. En la
      ronda `f`, la receta se fitea con los folds distintos de `f` (fold-train) y se
      aplica congelada a `train` y a `val` (el fold `f`). Contienen solo features y
      `label`: `attack_cat`, `es_duplicado` y `Ltime` NO se escriben aquí.
    - `<celda>/r*/parametros.json`: los valores fiteados (modas, columnas log1p y
      protocolos frecuentes).
    - `manifiesto.csv`: qué receta define cada celda.

* `grid_hyperparams/`
    - `hiperparametros.json`: base y variaciones de hiperparámetros que consume el
      entrenamiento. Se edita en `notebooks/hiperparametros.py`.

Notas de alcance
----------------
* El **one-hot** de las categóricas y el **StandardScaler** son transversales: son
  idénticos en todas las celdas y no forman parte de la rejilla. Se aplican dentro del
  pipeline de MLlib en la etapa de entrenamiento (fit por fold), no aquí. Por eso los
  parquet de `grid_preprocess/` contienen columnas tabulares, no un vector de features.
* Los Parquet que lee Spark (`grid_preprocess/` y la materialización final) se escriben
  con nombres de columnas normalizados y variables booleanas convertidas a valores
  numéricos, de modo que Spark ML puede leerlos directamente. `base/` NO se normaliza
  (solo lo lee pandas). Todos se comprimen con zstd.
* `StandardScaler` se aplicará también a Random Forest por decisión de uniformidad.
* `proto` colapsa su cola en la categoría `otros`: los protocolos con frecuencia menor
  a `UMBRAL_PROTO` en el fold-train completo (antes del muestreo) pasan a `otros`, y
  también cualquier protocolo de val/test no visto en el fit. La lista fiteada se
  guarda en `parametros.json` y se comparte entre celdas. El one-hot se aplica después
  en el pipeline de MLlib.
* La deduplicación se hace una sola vez, antes del split cronológico, sobre **la vista
  del modelo más la clase**: las features que el modelo realmente recibe (sin IP, sin
  marcas de tiempo, con los puertos reducidos a su franja IANA) y `attack_cat`. Se
  conserva la primera aparición (la más antigua) y se marca `es_duplicado=1` si esa
  fila tenía gemelos en el origen. Filas con las mismas features pero distinta clase
  se conservan (ruido irreducible) y se agrupan en el mismo fold.
* Cada `train.parquet` se limita por clase a 100,000 filas por defecto, con semilla 42.
  `--limite-por-categoria 0` lo desactiva; la validación y `base/` permanecen completas.
* El submuestreo se aplica después de deduplicar, por lo que el límite cuenta filas
  únicas dentro de cada clase.
* La selección de columnas `log1p` y de protocolos frecuentes se calcula sobre el
  fold-train completo antes del muestreo y se comparte entre celdas; en fase 3 se
  recalcula con todo el train.
* Por cada ronda, la receta común se aplica una sola vez generando todas las columnas
  derivadas (ct_* y ambas asimetrías); cada celda solo descarta las columnas que no
  le corresponden. El resultado en disco es idéntico al de fitear celda por celda.

Ejecución:  python3 notebooks/pre-process.py
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
LTIME_COL = "Ltime"
FOLD_COL = "fold_id"
CONTROL = {LABEL_COL, CLASE_COL, DUP_COL, GRUPO_COL, LTIME_COL, FOLD_COL}

# Columnas que se descartan al generar `grid_preprocess/` y la materialización final:
# no son features.
COLS_NO_FEATURE = [LTIME_COL, GRUPO_COL, CLASE_COL, DUP_COL, FOLD_COL]

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
UMBRAL_PROTO = 0.001  # protocolos con < 0.1 % del fold-train pasan a "otros"
CATEGORIA_OTROS = "otros"

# ===========================================================================
# Rejilla de preprocesamiento
# ===========================================================================
# Diseño factorial: presencia/ausencia de cada asimetría, cruzada con conservar/excluir
# ct_*. Así se estiman sus efectos individuales y conjuntos. Se conservan las fuentes.
# Puertos por franja IANA, proto, log1p, escalado e imputación son comunes.
CELDAS = {
    "G00": {"ct": True,  "feature": "base"},
    "G01": {"ct": False, "feature": "base"},
    "G02": {"ct": True,  "feature": "packet_asym"},
    "G03": {"ct": False, "feature": "packet_asym"},
    "G04": {"ct": True,  "feature": "byte_asym"},
    "G05": {"ct": False, "feature": "byte_asym"},
    "G06": {"ct": True,  "feature": "both_asym"},
    "G07": {"ct": False, "feature": "both_asym"},
}

# Receta común: genera todas las columnas derivadas; cada celda descarta las que sobran.
CELDA_COMPLETA = {"ct": True, "feature": "both_asym"}
COLS_ASIMETRIA = {
    "packet_asymmetry": {"packet_asym", "both_asym"},
    "byte_asymmetry": {"byte_asym", "both_asym"},
}


def columnas_a_quitar(celda: dict, columnas) -> list[str]:
    """Columnas de la receta completa que NO pertenecen a la celda."""
    quitar = []
    if not celda["ct"]:
        quitar += [c for c in CT_COLS if c in columnas]
    for col, features_que_la_incluyen in COLS_ASIMETRIA.items():
        if celda["feature"] not in features_que_la_incluyen and col in columnas:
            quitar.append(col)
    return quitar


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
def normalizar_nombre(col: str) -> str:
    """Nombre de columna tal como queda en los Parquet (ct_src_ ltm → ct_src_ltm)."""
    return re.sub(r"_+", "_", col.replace(" ", "_").replace("-", "_"))


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


def seleccionar_proto_frecuentes(train: pd.DataFrame) -> list[str]:
    """Protocolos con frecuencia >= UMBRAL_PROTO en el fold-train completo."""
    frec = train["proto"].astype(str).value_counts(normalize=True)
    return sorted(frec[frec >= UMBRAL_PROTO].index.tolist())


def _agrupar_proto(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Colapsa en `otros` los protocolos fuera de la lista fiteada.

    Modifica `df` in situ: se invoca siempre sobre el DataFrame ya copiado por
    `_rellenar_nulos`.
    """
    proto = df["proto"].astype(str)
    df["proto"] = proto.where(proto.isin(set(params["proto_frecuentes"])), CATEGORIA_OTROS)
    return df


def _derivar_puertos(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva la franja IANA de los puertos y elimina los valores crudos."""
    out = df
    for pre in ("sport", "dsport"):
        if pre in out.columns:
            out[f"{pre}_franja"] = _franja(out[pre])
    return out.drop(columns=["sport", "dsport"])


def _derivar_feature(df: pd.DataFrame, celda: dict) -> pd.DataFrame:
    """Añade una señal de asimetría direccional sin reemplazar sus columnas fuente."""
    feature = celda.get("feature", "base")
    if feature == "base":
        return df
    pares = {
        "packet_asym": [("Spkts", "Dpkts", "packet_asymmetry")],
        "byte_asym": [("sbytes", "dbytes", "byte_asymmetry")],
        "both_asym": [
            ("Spkts", "Dpkts", "packet_asymmetry"),
            ("sbytes", "dbytes", "byte_asymmetry"),
        ],
    }
    if feature not in pares:
        raise ValueError(f"Feature de grilla desconocida: {feature}")

    out = df.copy()
    for origen_a, origen_b, destino in pares[feature]:
        a = pd.to_numeric(out[origen_a], errors="coerce").astype("float64")
        b = pd.to_numeric(out[origen_b], errors="coerce").astype("float64")
        out[destino] = (a - b) / (a + b).clip(lower=1)
    return out


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
        f"  submuestreo por clase (límite {limite_por_categoria:,}): "
        + ", ".join(resumen)
    )
    return pd.concat(muestras, ignore_index=True)


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
    proto_frecuentes: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fitea la receta y devuelve parámetros y train transformado.

    Los duplicados ya se eliminaron antes del split. `limite_por_categoria=None`
    conserva todas las filas, como en la fase final. El resultado no incluye
    `attack_cat`, `es_duplicado`, `Ltime` ni la columna de grupo.
    """
    columnas_log1p = (
        seleccionar_log1p_cols(train) if log1p_cols is None else log1p_cols
    )
    frecuentes = (
        seleccionar_proto_frecuentes(train) if proto_frecuentes is None
        else proto_frecuentes
    )
    fit = train
    if limite_por_categoria is not None:
        fit = submuestrear_por_categoria(fit, limite_por_categoria, semilla)
    params: dict = {
        "modas": {},
        "log1p_cols": list(columnas_log1p),
        "proto_frecuentes": list(frecuentes),
    }
    for col in IMPUTAR:
        modo = fit[col].dropna().mode()
        v = modo.iat[0] if len(modo) else np.nan
        params["modas"][col] = v.item() if hasattr(v, "item") else v

    w = _rellenar_nulos(fit, params)
    w = _agrupar_proto(w, params)
    w = _derivar_puertos(w)
    w = _derivar_feature(w, celda)
    w = _filtrar_ct(w, celda)
    w = _aplicar_log1p(w, params)
    return params, w.drop(columns=COLS_NO_FEATURE, errors="ignore")


def transform(df: pd.DataFrame, celda: dict, params: dict) -> pd.DataFrame:
    """Aplica una receta ya fiteada (para el fold-val o el test)."""
    out = _rellenar_nulos(df, params)
    out = _agrupar_proto(out, params)
    out = _derivar_puertos(out)
    out = _derivar_feature(out, celda)
    out = _filtrar_ct(out, celda)
    out = _aplicar_log1p(out, params)
    return out.drop(columns=COLS_NO_FEATURE, errors="ignore")


def verificar_sin_nulos(df: pd.DataFrame, contexto: str) -> None:
    """Falla si queda algún NaN: VectorAssembler (handleInvalid='error') fallaría."""
    con_nulos = df.columns[df.isna().any()].tolist()
    if con_nulos:
        raise ValueError(f"[{contexto}] columnas con valores nulos: {con_nulos}")


def preparar_salida_spark(df: pd.DataFrame) -> pd.DataFrame:
    """Adapta un resultado tabular al esquema que consume Spark ML.

    Modifica `df` in situ (sin copiarlo): el llamador no debe reutilizarlo con los
    nombres originales. La normalización se hace una sola vez al persistir el Parquet.
    """
    df.columns = [normalizar_nombre(col) for col in df.columns]
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col].dtype):
            df[col] = df[col].astype("float64")
    return df


def escribir_parquet_spark(df: pd.DataFrame, ruta: Path, *, normalizar: bool = True) -> None:
    """Escribe un Parquet con zstd.

    `normalizar=True` (por defecto) adapta nombres y booleanos para Spark ML y modifica
    `df` in situ. `base/` se escribe con `normalizar=False` para conservar el esquema
    original que espera la receta.
    """
    if normalizar:
        preparar_salida_spark(df)
    df.to_parquet(ruta, index=False, compression="zstd")


def validar_salidas_preprocesamiento() -> None:
    """Falla si falta algún artefacto o si la rejilla contiene columnas no permitidas."""
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

    # Los parquet de la rejilla no deben llevar columnas que no son features.
    prohibidas = {normalizar_nombre(c) for c in COLS_NO_FEATURE}
    problemas = []
    for ruta in requeridos:
        if ruta.suffix == ".parquet" and GRID_PREPROC_DIR in ruta.parents:
            nombres = set(pq.read_schema(ruta).names)
            indebidas = sorted(nombres & prohibidas)
            if indebidas:
                problemas.append(f"{ruta}: columnas no permitidas {indebidas}")
            if LABEL_COL not in nombres:
                problemas.append(f"{ruta}: falta la columna '{LABEL_COL}'")
    if problemas:
        raise AssertionError(
            "Parquet de la rejilla con esquema inválido:\n"
            + "\n".join(f"  - {p}" for p in problemas)
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

    Evita partir un grupo de filas con idéntica marca de tiempo entre dos particiones.
    """
    n = len(ltime)
    i = int(fraccion * n)
    while 0 < i < n and ltime[i] == ltime[i - 1]:
        i += 1
    return i


def asignar_folds(label: np.ndarray, grupo: np.ndarray, k: int, semilla: int) -> np.ndarray:
    """Estratificado por clase y agrupado por vista del modelo.

    Cada grupo (filas con las mismas features del modelo; tras deduplicar solo pueden
    diferir en la clase) cae entero en un fold, y los grupos se reparten de forma
    codiciosa para equilibrar el recuento de cada clase entre folds.
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
    # Consistencia 1:1 entre attack_cat y Label, en ambos sentidos.
    inc_a = int(((df["attack_cat"] != "Benign") & (df["Label"] == 0)).sum())
    inc_b = int(((df["attack_cat"] == "Benign") & (df["Label"] == 1)).sum())
    if inc_a or inc_b:
        raise ValueError(
            "attack_cat no se corresponde 1:1 con Label "
            f"(ataque con Label=0: {inc_a}; Benign con Label=1: {inc_b}); revisar limpieza."
        )
    for col in ["state", "service", "attack_cat"]:
        df[col] = df[col].astype("category")

    df = df.sort_values("Ltime", kind="stable").reset_index(drop=True)
    filas_originales = len(df)

    # Blancos estructurales: se rellenan antes de deduplicar para que la vista del
    # modelo coincida con lo que el modelo realmente recibe.
    for col in BLANCOS_ESTRUCTURALES:
        relleno = False if df[col].dtype == "boolean" else 0
        df[col] = df[col].fillna(relleno)

    # ---------------- Deduplicación sobre la vista del modelo ----------------
    # Vista = features sin IP ni tiempos, con los puertos como franja IANA (lo que el
    # modelo recibe). Duplicado = misma vista + misma clase; se conserva la primera
    # aparición (la más antigua, porque df está ordenado por Ltime).
    features = [c for c in df.columns if c not in ELIMINAR_ESTATICO]
    vista = df[[c for c in features if c not in ("sport", "dsport")]].copy()
    vista["sport_franja"] = np.asarray(_franja(df["sport"].astype("float64"))).astype(str)
    vista["dsport_franja"] = np.asarray(_franja(df["dsport"].astype("float64"))).astype(str)

    vista_y = vista.assign(attack_cat=df["attack_cat"].astype(str))
    es_dup_original = vista_y.duplicated(keep=False).to_numpy()
    dup = vista_y.duplicated(keep="first").to_numpy()
    del vista_y
    conservar = ~dup
    dup_exceso = int(dup.sum())
    filas_marcadas = int(es_dup_original.sum())

    df = df.loc[conservar].reset_index(drop=True)
    vista = vista.loc[conservar].reset_index(drop=True)
    # La marca identifica la fila representativa de un grupo repetido en el origen.
    es_dup = es_dup_original[conservar]
    # Grupo = misma vista del modelo (sin la clase): solo agrupa filas con features
    # idénticas y clases distintas, que deben caer en el mismo fold.
    grupo = pd.util.hash_pandas_object(vista, index=False).to_numpy()
    del vista
    n = len(df)
    print(
        f"\n[1] cargado: {filas_originales:,} filas · "
        f"duplicados (vista del modelo + clase) eliminados {dup_exceso:,} "
        f"({dup_exceso / filas_originales * 100:.2f} %) · "
        f"filas únicas conservadas {n:,}"
    )

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
    n_test, c_test = base_test.shape
    escribir_parquet_spark(base_test, BASE_DIR / "test.parquet", normalizar=False)
    del base_test
    tamanos_fold = {}
    for j in range(K):
        sub = base.loc[fold == j].drop(columns=[GRUPO_COL]).reset_index(drop=True)
        tamanos_fold[f"fold_{j + 1}"] = int(len(sub))
        print(f"  fold_{j + 1}.parquet  {len(sub):>9,} filas × {sub.shape[1]} cols")
        escribir_parquet_spark(sub, BASE_DIR / f"fold_{j + 1}.parquet", normalizar=False)
        del sub
    print(f"  test.parquet     {n_test:>9,} filas × {c_test} cols")

    # ---------------- Materialización de la rejilla ----------------
    print(f"\n[5] materializando la rejilla de preprocesamiento ({len(CELDAS)} celdas × {K} rondas)")
    for p in GRID_PREPROC_DIR.glob("G*"):
        if p.is_dir() and p.name not in CELDAS:
            shutil.rmtree(p)
            print(f"  celda obsoleta eliminada: {p.name}")
    estadisticas_obsoletas = GRID_PREPROC_DIR / "estadisticas_muestreo.csv"
    if estadisticas_obsoletas.exists():
        estadisticas_obsoletas.unlink()

    manifiesto = [
        {
            "celda": cid, "ct": celda["ct"], "feature": celda["feature"],
            "puertos": "solo_franja", "proto": "cola_en_otros",
        }
        for cid, celda in CELDAS.items()
    ]
    for cid in CELDAS:
        for j in range(K):
            (GRID_PREPROC_DIR / cid / f"r{j + 1}").mkdir(parents=True, exist_ok=True)

    train_base = base.loc[fold >= 0].reset_index(drop=True)
    for j in range(K):
        fit_df = train_base.loc[train_base["fold_id"] != j, cols_base]
        val_df = train_base.loc[train_base["fold_id"] == j, cols_base]

        # Todo lo común se calcula UNA vez por ronda.
        log1p_cols = seleccionar_log1p_cols(fit_df)
        proto_frec = seleccionar_proto_frecuentes(fit_df)
        print(f"  ronda {j + 1}: {len(log1p_cols)} columnas log1p · "
              f"{len(proto_frec)} protocolos frecuentes")
        params, train_comun = fit_transform(
            fit_df, CELDA_COMPLETA,
            limite_por_categoria=args.limite_por_categoria,
            semilla=SEMILLA_SUBMUESTREO,
            log1p_cols=log1p_cols,
            proto_frecuentes=proto_frec,
        )
        val_comun = transform(val_df, CELDA_COMPLETA, params)
        verificar_sin_nulos(train_comun, f"r{j + 1}/train")
        verificar_sin_nulos(val_comun, f"r{j + 1}/val")

        # Cada celda solo elige sus columnas.
        for cid, celda in CELDAS.items():
            ronda_dir = GRID_PREPROC_DIR / cid / f"r{j + 1}"
            sobran = columnas_a_quitar(celda, train_comun.columns)
            escribir_parquet_spark(train_comun.drop(columns=sobran), ronda_dir / "train.parquet")
            escribir_parquet_spark(val_comun.drop(columns=sobran), ronda_dir / "val.parquet")
            with open(ronda_dir / "parametros.json", "w") as fh:
                json.dump(params, fh, ensure_ascii=False, indent=2, default=str)
        del train_comun, val_comun, fit_df, val_df
    pd.DataFrame(manifiesto).to_csv(GRID_PREPROC_DIR / "manifiesto.csv", index=False)
    print(f"  {len(CELDAS)} celdas × {K} rondas materializadas")

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
        "filas_originales": int(filas_originales),
        "criterio_deduplicacion": (
            "vista_del_modelo_mas_clase: features sin IP ni tiempos, puertos como "
            "franja IANA, mas attack_cat; se conserva la primera aparicion"
        ),
        "copias_exactas_eliminadas": dup_exceso,
        "filas_originales_en_grupos_duplicados": filas_marcadas,
        "filas_conservadas_marcadas_como_duplicadas": int(es_dup.sum()),
        "fracciones": {"train": FRACCION_TRAIN, "test": round(1 - FRACCION_TRAIN, 2)},
        "corte_train_test": int(corte),
        "Ltime_corte_train_test": str(pd.to_datetime(int(ltime[c_indice]), unit="s")),
        "K": K,
        "semilla": SEMILLA,
        "limite_por_categoria": args.limite_por_categoria,
        "semilla_submuestreo": SEMILLA_SUBMUESTREO,
        "deduplicacion_antes_del_split": True,
        "submuestreo_por_categoria_despues_de_deduplicar": True,
        "log1p_seleccion_fold_train_completo": True,
        "proto_cola_en_otros": True,
        "proto_seleccion_fold_train_completo": True,
        "umbral_proto": UMBRAL_PROTO,
        "tamanos_folds": tamanos_fold,
        # Nombres originales, tal como quedan en base/. En la rejilla los nombres están
        # normalizados y sport/dsport se reemplazan por sport_franja/dsport_franja.
        "columnas_features": features,
        "columnas_features_rejilla": [normalizar_nombre(c) for c in features],
        "columnas_control": [LABEL_COL, CLASE_COL, DUP_COL],
        "columnas_no_feature_en_rejilla": [normalizar_nombre(c) for c in COLS_NO_FEATURE],
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

    # [V2] Informativo: tras deduplicar por vista+clase, un grupo que cruza train/test
    # solo puede tener las mismas features con clases distintas (ruido irreducible).
    g_train = set(grupo[idx_train].tolist())
    g_test = set(grupo[idx_test].tolist())
    cruce = len(g_train & g_test)
    filas_test_con_gemelo = int(np.isin(grupo[idx_test], grupo[idx_train]).sum())
    print(f"  [V2] grupos de features iguales (distinta clase) que cruzan train/test: "
          f"{cruce} · filas de test afectadas: {filas_test_con_gemelo:,} "
          f"({filas_test_con_gemelo / max(len(idx_test), 1) * 100:.3f} %)")

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