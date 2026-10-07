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

* `grid/`
    - `<celda>/r1..r5/{train,val}.parquet`: una variante por celda y por ronda. En la
      ronda `f`, la receta se fitea con los folds distintos de `f` (fold-train) y se
      aplica congelada a `train` y a `val` (el fold `f`).
    - `<celda>/r*/parametros.json`: los valores fiteados (modas, p99, columnas log1p,
      categorías de `proto` conservadas).
    - `manifiesto.csv`: qué receta define cada celda.

Notas de alcance
----------------
* El **one-hot** de las categóricas y el **StandardScaler** son transversales: son
  idénticos en todas las celdas y no forman parte de la rejilla. Se aplican dentro del
  pipeline de MLlib en la etapa de entrenamiento (fit por fold), no aquí. Por eso los
  parquet de `grid/` contienen columnas tabulares, no un vector de features.
* `StandardScaler` se aplicará también a Random Forest por decisión de uniformidad.
* El deduplicado (celda P5) se aplica **solo** al fold-train; el fold-val se deja
  íntegro para medir sobre las filas repetidas tal como llegan.

Ejecución:  python3 notebooks/pre-process.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "preprocessed"
BASE_DIR = PROC_DIR / "base"
GRID_DIR = PROC_DIR / "grid"
for _d in (DATA_DIR, BASE_DIR, GRID_DIR):
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
CONTROL = {LABEL_COL, CLASE_COL, DUP_COL}

BLANCOS_ESTRUCTURALES = ["is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd"]
IMPUTAR = ["dsport", "sport", "state"]

CT_COLS = [
    "ct_state_ttl", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_srv_src", "ct_srv_dst",
    "ct_flw_http_mthd", "ct_ftp_cmd",
]

BINARIAS = ["is_sm_ips_ports", "is_ftp_login"]

# Columnas que nunca entran al espacio de características. Ltime se conserva aparte en
# `base/` para ordenar, particionar y auditar; se descarta en `grid/`.
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
UMBRAL_PROTO_COLA = 0.01

# ===========================================================================
# Rejilla de preprocesamiento (14 celdas)
# ===========================================================================
# Ejes: log1p (None | umbral |s|), winsor (p99), ct_* (con/sin), dedup,
# derivados de puerto, cola de proto, imputación de tokens corruptos.
_BASE_CELL = dict(log1p=1.0, winsor=False, ct=True, dedup=False,
                  puertos="ambos", proto="completo", imput="moda")

_OVERRIDES = {
    "G00": {},                                                            # base
    "G01": dict(log1p=None),                                             # sin log1p
    "G02": dict(log1p=3.0, winsor=True),                                 # log1p>3 + winsor
    "G03": dict(winsor=True),                                            # log1p>1 + winsor
    "G05": dict(ct=False),                                               # sin ct_*
    "G06": dict(dedup=True),                                             # deduplicar
    "G07": dict(puertos="solo_franja"),                                  # solo franja
    "G08": dict(puertos="solo_efimero"),                                 # solo efímero
    "G09": dict(puertos="ninguno"),                                      # sin derivados
    "G10": dict(proto="colapsar"),                                       # cola de proto
    "G11": dict(imput="cero"),                                           # imputar a 0
    "G12": dict(log1p=None, ct=False),
    "G13": dict(dedup=True, ct=False),
    "G14": dict(dedup=True, proto="colapsar"),
}
CELDAS = {cid: {**_BASE_CELL, **ov} for cid, ov in _OVERRIDES.items()}


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


def _rellenar_nulos(df: pd.DataFrame, celda: dict, params: dict) -> pd.DataFrame:
    out = df.copy()
    for col in IMPUTAR:
        if col not in out.columns:
            continue
        if col == "state":
            relleno = params["modas"][col]
        else:
            relleno = params["modas"][col] if celda["imput"] == "moda" else 0
        out[col] = out[col].fillna(relleno)
    return out


def _derivar_puertos(df: pd.DataFrame, celda: dict) -> pd.DataFrame:
    """Deriva `efimero`/`franja` según la celda y elimina los puertos crudos."""
    out = df
    opcion = celda["puertos"]
    for pre in ("sport", "dsport"):
        if pre not in out.columns:
            continue
        if opcion in ("ambos", "solo_efimero"):
            out[f"{pre}_efimero"] = (out[pre] >= 1024).astype("Int8")
        if opcion in ("ambos", "solo_franja"):
            out[f"{pre}_franja"] = _franja(out[pre])
    return out.drop(columns=["sport", "dsport"])


def _filtrar_ct(df: pd.DataFrame, celda: dict) -> pd.DataFrame:
    if celda["ct"]:
        return df
    return df.drop(columns=[c for c in CT_COLS if c in df.columns])


def fit_transform(train: pd.DataFrame, celda: dict) -> tuple[dict, pd.DataFrame]:
    """Fitea la receta en `train` y devuelve (parámetros, train transformado).

    El deduplicado (si la celda lo pide) se aplica antes de fitear, porque las
    estadísticas deben salir de los datos con los que realmente se entrena.
    """
    fit = train
    if celda["dedup"]:
        feat = [c for c in train.columns if c not in CONTROL]
        fit = train.drop_duplicates(subset=feat)

    params: dict = {"modas": {}, "p99": {}, "log1p_cols": [], "proto_top": []}
    for col in IMPUTAR:
        modo = fit[col].dropna().mode()
        v = modo.iat[0] if len(modo) else np.nan
        params["modas"][col] = v.item() if hasattr(v, "item") else v

    w = _rellenar_nulos(fit, celda, params)
    w = _derivar_puertos(w, celda)
    w = _filtrar_ct(w, celda)

    if celda["proto"] == "colapsar":
        frec = w["proto"].value_counts(normalize=True)
        params["proto_top"] = frec[frec >= UMBRAL_PROTO_COLA].index.tolist()
        w["proto"] = w["proto"].where(w["proto"].isin(params["proto_top"]), "otros")

    num = _columnas_numericas(w)
    if celda["winsor"]:
        for c in num:
            params["p99"][c] = float(pd.to_numeric(w[c], errors="coerce").quantile(0.99))
        for c in num:
            w[c] = w[c].clip(upper=params["p99"][c])

    if celda["log1p"] is not None:
        for c in num:
            s = pd.to_numeric(w[c], errors="coerce").astype("float64")
            if abs(float(s.skew())) > celda["log1p"]:
                params["log1p_cols"].append(c)

    w = _aplicar_log1p(w, params)
    return params, w.drop(columns=["Ltime"])


def transform(df: pd.DataFrame, celda: dict, params: dict) -> pd.DataFrame:
    """Aplica una receta ya fiteada (para el fold-val o el test)."""
    out = _rellenar_nulos(df, celda, params)
    out = _derivar_puertos(out, celda)
    out = _filtrar_ct(out, celda)
    if celda["proto"] == "colapsar":
        out["proto"] = out["proto"].where(out["proto"].isin(params["proto_top"]), "otros")
    if celda["winsor"]:
        for c in _columnas_numericas(out):
            if c in params["p99"]:
                out[c] = out[c].clip(upper=params["p99"][c])
    out = _aplicar_log1p(out, params)
    return out.drop(columns=["Ltime"])


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
def main() -> None:
    print("=== PREPROCESAMIENTO: split + folds + rejilla (sin entrenamiento) ===")
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
    idx_train = np.arange(0, corte)
    idx_test = np.arange(corte, n)
    print(f"[2] corte 85/15 en fila {corte:,} "
          f"(Ltime {pd.to_datetime(int(ltime[corte]), unit='s')}); "
          f"train {len(idx_train):,} · test {len(idx_test):,}")

    clases = sorted(df["attack_cat"].unique().tolist())
    mapa = {c: i for i, c in enumerate(clases)}
    label = df["attack_cat"].map(mapa).astype("int8").to_numpy()

    features = [c for c in df.columns if c not in ELIMINAR_ESTATICO]
    cols_base = features + [LABEL_COL, CLASE_COL, DUP_COL, "Ltime"]
    base = df[features].copy()
    base[LABEL_COL] = label
    base[CLASE_COL] = df["attack_cat"].to_numpy()
    base[DUP_COL] = es_dup.astype("int8")
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
    base_test = base.iloc[idx_test].reset_index(drop=True)
    base_test.to_parquet(BASE_DIR / "test.parquet", index=False, compression="snappy")
    tamanos_fold = {}
    for j in range(K):
        sub = base.loc[fold == j].reset_index(drop=True)
        sub.to_parquet(BASE_DIR / f"fold_{j + 1}.parquet", index=False, compression="snappy")
        tamanos_fold[f"fold_{j + 1}"] = int(len(sub))
        print(f"  fold_{j + 1}.parquet  {len(sub):>9,} filas × {sub.shape[1]} cols")
    print(f"  test.parquet     {len(base_test):>9,} filas × {base_test.shape[1]} cols")

    # ---------------- Materialización de la rejilla ----------------
    print("\n[5] materializando la rejilla de preprocesamiento (14 celdas × 5 rondas)")
    train_base = base.loc[fold >= 0].reset_index(drop=True)
    manifiesto = []
    for cid, celda in CELDAS.items():
        celda_dir = GRID_DIR / cid
        celda_dir.mkdir(parents=True, exist_ok=True)
        for j in range(K):
            ronda_dir = celda_dir / f"r{j + 1}"
            ronda_dir.mkdir(parents=True, exist_ok=True)
            fit_df = train_base.loc[train_base["fold_id"] != j, cols_base]
            val_df = train_base.loc[train_base["fold_id"] == j, cols_base]
            params, train_t = fit_transform(fit_df, celda)
            val_t = transform(val_df, celda, params)
            train_t.to_parquet(ronda_dir / "train.parquet", index=False, compression="snappy")
            val_t.to_parquet(ronda_dir / "val.parquet", index=False, compression="snappy")
            with open(ronda_dir / "parametros.json", "w") as fh:
                json.dump(params, fh, ensure_ascii=False, indent=2, default=str)
        manifiesto.append({
            "celda": cid, "log1p": celda["log1p"], "winsor_p99": celda["winsor"],
            "ct": celda["ct"], "deduplicar": celda["dedup"],
            "puertos": celda["puertos"], "proto": celda["proto"],
            "imputacion": celda["imput"],
        })
        print(f"  {cid}: 5 rondas materializadas")
    pd.DataFrame(manifiesto).to_csv(GRID_DIR / "manifiesto.csv", index=False)

    # ---------------- Metadatos ----------------
    meta = {
        "filas_totales": int(n),
        "duplicados_conservados": dup_exceso,
        "filas_marcadas_como_duplicadas": int(es_dup.sum()),
        "fracciones": {"train": FRACCION_TRAIN, "test": round(1 - FRACCION_TRAIN, 2)},
        "corte_train_test": int(corte),
        "Ltime_corte_train_test": str(pd.to_datetime(int(ltime[corte]), unit="s")),
        "K": K,
        "semilla": SEMILLA,
        "tamanos_folds": tamanos_fold,
        "columnas_features": features,
        "columnas_control": [LABEL_COL, CLASE_COL, DUP_COL],
        "columnas_eliminadas": ELIMINAR_ESTATICO,
        "blancos_estructurales_imputados_a_cero": BLANCOS_ESTRUCTURALES,
        "tokens_corruptos_imputados": IMPUTAR,
        "clases": {c: v for c, v in mapa.items()},
        "clases_por_indice": {str(v): c for c, v in mapa.items()},
        "rejilla": manifiesto,
        "umbral_proto_cola": UMBRAL_PROTO_COLA,
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
    corte_ok = bool(ltime[corte] != ltime[corte - 1])
    print(f"  [V3] corte alineado a límite de Ltime: {corte_ok}")
    if not corte_ok:
        raise AssertionError("El corte 85/15 parte un grupo de Ltime idéntico")

    print("\nPreprocesamiento completado.")
    print(f"  base/  → 5 folds + test  (sin transformar)")
    print(f"  grid/  → {len(CELDAS)} celdas × {K} rondas (train + val por ronda)")


if __name__ == "__main__":
    main()
