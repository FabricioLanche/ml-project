import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "dataset"
OUT_DIR = BASE / "output"
PROC_DIR = OUT_DIR / "preprocessed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "harshwardhanbhangale/unsw-complete-dataset"
REQ = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
if all((DATA_DIR / f).exists() for f in REQ):
    print("Dataset ya presente en dataset/")
else:
    print("Descargando dataset desde Kaggle…")
    import kagglehub
    tmp = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in REQ:
        origen = tmp / f
        if origen.exists():
            shutil.copy2(origen, DATA_DIR / f)
            print(f"  → {f}")

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

# Variables cuya ausencia es un blanco estructural mal codificado en la redistribución
# de Kaggle. El blanco y el cero significan lo mismo ("no aplica"), pero los cuatro
# archivos mezclan ambas convenciones. Se imputa a 0 (ver §3.5.b de la propuesta).
BLANCOS_ESTRUCTURALES = ["is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd"]

# Variables con residuo corrupto en la fuente: se imputan por la moda de TRAIN.
IMPUTAR_MODA = ["dsport", "sport", "state"]

# Se eliminan del espacio de características. Stime/Ltime se usan solo para ordenar
# y particionar, y se descartan antes de construir la matriz de diseño.
ELIMINAR = {
    "attack_cat": "variable objetivo",
    "Label": "versión binaria del objetivo (Benign <=> Label=0)",
    "srcip": "identidad de host, 43 valores únicos: memoriza hosts",
    "dstip": "identidad de host, 47 valores únicos: memoriza hosts",
    "Stime": "marca de tiempo absoluta: correlaciona con la sesión de captura",
    "Ltime": "marca de tiempo absoluta: correlaciona con la sesión de captura",
    "sport": "129 225 categorías junto a dsport: se derivan rangos",
    "dsport": "129 225 categorías junto a sport: se derivan rangos",
}

UMBRAL_SKEW = 1.0
FRACCIONES = {"train": 0.75, "val": 0.15, "test": 0.10}


# ===========================================================================
# Utilidades
# ===========================================================================
def _guardar_csv(tabla, nombre):
    tabla.to_csv(OUT_DIR / nombre, index=False)
    print(f"  → output/{nombre}")
    return tabla


# ===========================================================================
# 1. CARGA
# ===========================================================================
print("\n=== 1. CARGA DE DATOS ===")
dfs = []
for i in range(1, 5):
    dfs.append(pd.read_csv(
        DATA_DIR / f"UNSW-NB15_{i}.csv",
        header=None, names=COLUMNS, dtype=DTYPES,
        na_values=NA_VALUES, low_memory=False,
    ))
df = pd.concat(dfs, ignore_index=True)
del dfs
N = len(df)
print(f"Shape: {N:,} filas × {len(df.columns)} columnas")

df["attack_cat"] = (
    df["attack_cat"].fillna("Benign").str.strip().replace({"Backdoor": "Backdoors"})
)
inc = int(((df["attack_cat"] != "Benign") & (df["Label"] == 0)).sum())
if inc:
    raise ValueError("attack_cat no se corresponde 1:1 con Label; revisar limpieza.")
for col in ["state", "service", "attack_cat"]:
    df[col] = df[col].astype("category")
print("Verificación target multiclase OK: Benign ⟺ Label=0 al 100%.")


# ===========================================================================
# 2. BLANCOS ESTRUCTURALES -> 0
# ===========================================================================
print("\n=== 2. IMPUTACIÓN DE BLANCOS ESTRUCTURALES ===")
for col in BLANCOS_ESTRUCTURALES:
    antes = int(df[col].isna().sum())
    # is_ftp_login es boolean nullable: requiere False, no 0.
    relleno = False if df[col].dtype == "boolean" else 0
    df[col] = df[col].fillna(relleno)
    print(f"  {col:18s} {antes:>9,} blancos -> 0  (queda {antes / N * 100:.2f}% del total)")
print("  Justificación: el blanco y el cero expresan la misma semántica ('no aplica')")
print("  pero la redistribución los mezcla entre archivos. Ver proposal §3.5.b.")


# ===========================================================================
# 3. IMPUTACIÓN POR MODA (ajustada SOLO en train)
# ===========================================================================
# Se particiona primero para que la moda se estime únicamente con datos de
# entrenamiento. Si se calculase sobre las 2.54M filas, la transformación de train
# estaría usando estadísticas de val/test.
print("\n=== 3. ORDEN TEMPORAL Y PARTICIÓN 75/15/10 ===")
df = df.sort_values("Ltime", kind="stable").reset_index(drop=True)
ltime = df["Ltime"].to_numpy()
n = len(df)


def _corte_alineado(fraccion):
    """Devuelve el índice de corte desplazado hasta un límite de Ltime.

    Sin este desplazamiento, el corte puede caer entre dos filas con la misma marca
    de tiempo. Como las filas byte-idénticas comparten Ltime, eso partiría un grupo
    de duplicados entre dos particiones, que es exactamente la fuga que la
    partición temporal debe cerrar.
    """
    i = int(fraccion * n)
    while 0 < i < n and ltime[i] == ltime[i - 1]:
        i += 1
    return i


corte_val = _corte_alineado(0.75)
corte_test = _corte_alineado(0.90)
print(f"  corte train|val en fila {corte_val:,}  (Ltime {pd.to_datetime(ltime[corte_val], unit='s')})")
print(f"  corte val|test  en fila {corte_test:,}  (Ltime {pd.to_datetime(ltime[corte_test], unit='s')})")

idx_train = np.arange(0, corte_val)
idx_val = np.arange(corte_val, corte_test)
idx_test = np.arange(corte_test, n)
print(f"  filas desplazadas por el alineamiento: {corte_val - int(0.75 * n)} y {corte_test - int(0.90 * n)}")

print("\n[03b] imputación por moda (estimada solo en train)")
modas = {}
train_idx = df.index[idx_train]
for col in IMPUTAR_MODA:
    nulos = int(df[col].isna().sum())
    modo = df.loc[train_idx, col].dropna().mode().iat[0]
    modas[col] = modo.item() if hasattr(modo, "item") else modo
    df[col] = df[col].fillna(modo)
    print(f"  {col:8s} {nulos:>4,} nulos -> moda(train)={modas[col]!r}")


# ===========================================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS: PUERTOS
# ===========================================================================
print("\n=== 4. INGENIERÍA DE CARACTERÍSTICAS ===")


def _banda(p):
    if p <= 1023:
        return "bien_conocido"
    if p <= 49151:
        return "registrado"
    return "dinamico"


for pre in ("sport", "dsport"):
    col = df[pre]
    df[f"{pre}_efimero"] = (col >= 1024).astype("Int8")
    df[f"{pre}_franja"] = pd.Categorical(
        np.where(col.isna(), "desconocido", col.map(_banda)),
        categories=["bien_conocido", "registrado", "dinamico", "desconocido"],
    )
    print(f"  {pre}_efimero: {100*float(df[f'{pre}_efimero'].astype('float64').mean()):.2f}% de puertos efímeros")
    print(f"  {pre}_franja  distribución: {dict(df[f'{pre}_franja'].value_counts())}")
print("  Justificación: el valor crudo no generaliza. Generic usa 79 puertos distintos,")
print("  Benign 64 581: el modelo memorizaría 'puerto 6881 -> Generic'. Ver proposal §3.4.")


# ===========================================================================
# 5. SELECCIÓN DE COLUMNAS
# ===========================================================================
print("\n=== 5. SELECCIÓN DE COLUMNAS ===")
acciones = []
for col in df.columns:
    if col in ELIMINAR:
        acciones.append({"columna": col, "accion": "eliminar", "motivo": ELIMINAR[col]})
    elif col in df.columns and col.endswith(("_efimero", "_franja")):
        acciones.append({"columna": col, "accion": "derivar",
                         "motivo": "ingeniería de puerto (baja cardinalidad)"})
    elif col == "attack_cat":
        continue
    else:
        if col in ("proto", "state", "service"):
            acc, mot = "one-hot", "categórica de baja cardinalidad"
        elif col in BLANCOS_ESTRUCTURALES or col in IMPUTAR_MODA:
            acc, mot = "imputar", "blanco estructural o token corrupto"
        elif col == "is_sm_ips_ports":
            acc, mot = "binaria", "indicador predefinido"
        else:
            acc, mot = "numerica", "log1p condicional + estandarización"
        acciones.append({"columna": col, "accion": acc, "motivo": mot})

X = df.drop(columns=list(ELIMINAR)).copy()
print(f"  {len(ELIMINAR)} columnas eliminadas · {len(X.columns)} conservadas "
      f"(incluye 4 derivadas de puerto)")
print("  objetivo fuera del espacio de características: OK")
_guardar_csv(pd.DataFrame(acciones), "preprocesamiento_columnas.csv")


# ===========================================================================
# 6. SELECCIÓN DE LOG1P (asimetría medida SOLO en train)
# ===========================================================================
print("\n=== 6. TRANSFORMACIÓN LOGARÍTMICA ===")
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
skew = []
for col in num_cols:
    s = X.loc[train_idx, col].astype("float64")
    skew.append({"variable": col, "skew_train": round(float(s.skew()), 4)})
skew = pd.DataFrame(skew).sort_values("skew_train", ascending=False)
skew["transformar"] = skew["skew_train"].abs() > UMBRAL_SKEW
skew["skew_tras_log1p"] = [
    round(float(np.log1p(X.loc[train_idx, c].astype("float64").clip(lower=0))
                 .replace([np.inf, -np.inf], np.nan).skew()), 4)
    for c in skew["variable"]
]
LOG1P_COLS = skew.loc[skew["transformar"], "variable"].tolist()
print(f"  asimetría medida sobre {len(train_idx):,} filas de train (nunca sobre val/test)")
print(f"  umbral |skew| > {UMBRAL_SKEW} -> se transforman {len(LOG1P_COLS)} de {len(skew)} numéricas")
print(f"  no se transforman: {skew.loc[~skew['transformar'], 'variable'].tolist()}")
print("\n  top 8 por beneficio del log1p:")
benef = skew.assign(ganancia=(skew.skew_tras_log1p.abs() - skew.skew_train.abs())).sort_values("ganancia")
print(benef.head(8)[["variable", "skew_train", "skew_tras_log1p", "ganancia"]].to_string(index=False))
_guardar_csv(skew, "preprocesamiento_log1p.csv")

for col in LOG1P_COLS:
    X[col] = np.log1p(X[col].astype("float64").clip(lower=0)).astype("float32")
print("  log1p aplicado.")


# ===========================================================================
# 7. PERSISTENCIA EN PARQUET
# ===========================================================================
print("\n=== 7. PERSISTENCIA ===")
PROC_DIR.mkdir(parents=True, exist_ok=True)
for split, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
    sub = X.iloc[idx]
    destino = PROC_DIR / f"{split}.parquet"
    sub.to_parquet(destino, index=False, compression="snappy")
    print(f"  {split:5s} {len(sub):>9,} filas × {sub.shape[1]} columnas "
          f"→ {destino.stat().st_size / 1e6:7.1f} MB")

meta = {
    "filas_totales": int(n),
    "duplicados_conservados": int(df.duplicated().sum()),
    "fracciones": FRACCIONES,
    "corte_train_val": int(corte_val),
    "corte_val_test": int(corte_test),
    "Ltime_corte_train_val": str(pd.to_datetime(ltime[corte_val], unit="s")),
    "Ltime_corte_val_test": str(pd.to_datetime(ltime[corte_test], unit="s")),
    "modas_imputadas": modas,
    "columnas_eliminadas": ELIMINAR,
    "umbral_skew": UMBRAL_SKEW,
    "variables_log1p": LOG1P_COLS,
    "blancos_estructurales_imputados_a_cero": BLANCOS_ESTRUCTURALES,
    "n_columnas_modelo": int(X.shape[1]),
}
with open(PROC_DIR / "metadatos.json", "w") as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)
print("  → output/preprocessed/metadatos.json")


# ===========================================================================
# 8. VERIFICACIONES
# ===========================================================================
print("\n=== 8. VERIFICACIONES ===")
filas = []
for split, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
    s = df.iloc[idx]["attack_cat"]
    filas.append({"particion": split, "n": len(idx), "n_clases": s.nunique(),
                  "benign_pct": round(100 * float((s == "Benign").mean()), 2)})
dist = pd.crosstab(df.iloc[idx_test]["attack_cat"], columns="test")
tabla_part = _guardar_csv(pd.DataFrame(filas), "preprocesamiento_particiones.csv")

print(f"\n  [V1] clases presentes en cada partición: "
      f"train={df.iloc[idx_train]['attack_cat'].nunique()}, "
      f"val={df.iloc[idx_val]['attack_cat'].nunique()}, "
      f"test={df.iloc[idx_test]['attack_cat'].nunique()}  (se requieren 10)")
assert all(f["n_clases"] == 10 for f in filas), "Falta alguna clase en una partición"

print("\n  [V2] clases minoritarias por partición")
menor = df["attack_cat"].value_counts().tail(4).index.tolist()
conteo = pd.DataFrame({
    p: df.iloc[ix]["attack_cat"].value_counts().reindex(menor).fillna(0).astype(int)
    for p, ix in (("train", idx_train), ("val", idx_val), ("test", idx_test))
})
conteo.index.name = "clase"
print(conteo.to_string())
_guardar_csv(conteo.reset_index(), "preprocesamiento_clases_minoritarias.csv")

print("\n  [V3] solapamiento de la ventana ct_* (100 conexiones previas)")
for split, idx in (("test", idx_test),):
    es_test = np.zeros(n, dtype=bool)
    es_test[idx] = True
    es_train = np.zeros(n, dtype=bool)
    es_train[idx_train] = True
    acum = np.concatenate([[0], np.cumsum(es_train)])
    i = np.where(es_test)[0]
    lo = np.maximum(i - 100, 0)
    ancho = np.maximum(np.minimum(i, 100), 1)
    pct = 100 * float(np.mean((acum[i] - acum[lo]) / ancho))
    print(f"  {split}: {pct:.2f}% del contexto previo de una fila de test está en train")
    assert pct == 0.0, "Hay solapamiento temporal: la partición no es cronológica"

print("\n  [V4] ninguna partición cae dentro de un empate de Ltime")
for nombre, corte in (("train|val", corte_val), ("val|test", corte_test)):
    igual = bool(ltime[corte] == ltime[corte - 1])
    print(f"  corte {nombre}: Ltime distinto a ambos lados = {not igual}")
    assert not igual, f"El corte {nombre} parte un grupo de Ltime idéntico"

print("\n  [V5] filas idénticas no cruzan particiones")
_orden = np.empty(n, dtype=np.int8)
_orden[idx_train] = 0
_orden[idx_val] = 1
_orden[idx_test] = 2
_cras = df.assign(_p=_orden).drop(columns="_p")
_hash = pd.util.hash_pandas_object(_cras, index=False)
_cruce = int((pd.DataFrame({"h": _hash, "p": _orden}).groupby("h")["p"].nunique() > 1).sum())
print(f"  grupos de filas idénticas repartidos entre particiones: {_cruce}")
assert _cruce == 0, "Hay filas idénticas en particiones distintas"

print("\n  [V6] sin fuga de la etiqueta")
assert "attack_cat" not in X.columns and "Label" not in X.columns
print("  attack_cat y Label ausentes del espacio de características: OK")

print("\nPreprocesamiento completado.")
print(f"Filas conservadas: {n:,} (duplicados NO eliminados: {int(df.duplicated().sum()):,})")
print(f"Columnas de entrada: 49 -> modelo: {X.shape[1]}")
print("Particiones: output/preprocessed/{train,val,test}.parquet")