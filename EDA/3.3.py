import pandas as pd
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / ".example-datasets" / "UNSW-NB15"

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
    # --- Nominales ---
    "srcip":         "string",
    "dstip":         "string",
    "proto":         "string",
    "state":         "string",    # category
    "service":       "string",    # category
    "attack_cat":    "string",    # category
    # --- Enteros ---
    "sport":         "Int32",    # 0–65.5k
    "dsport":        "Int32",    # 0–65.5k
    "sbytes":        "Int32",    # 0–14.3M
    "dbytes":        "Int32",    # 0–14.6M
    "sttl":          "Int16",    # 0–255
    "dttl":          "Int16",    # 0–254
    "sloss":         "Int16",    # 0–5.3k
    "dloss":         "Int16",    # 0–5.5k
    "Spkts":         "Int16",    # 0–10.6k
    "Dpkts":         "Int16",    # 0–11k
    "swin":          "Int16",    # 0–255
    "dwin":          "Int16",    # 0–255
    "stcpb":         "Int64",    # 4.29e9
    "dtcpb":         "Int64",    # 4.29e9
    "smeansz":       "Int16",    # 0–1.5k
    "dmeansz":       "Int16",    # 0–1.5k
    "trans_depth":   "Int16",    # 0–172
    "res_bdy_len":   "Int32",    # 0–6.5M
    "Stime":         "Int32",    # 1.42e9
    "Ltime":         "Int32",    # 1.42e9
    "ct_state_ttl":  "Int8",     # 0–6
    "ct_flw_http_mthd": "Int8",  # 0–36
    "ct_ftp_cmd":    "Int8",     # 0–8
    "ct_srv_src":    "Int8",     # 1–67
    "ct_srv_dst":    "Int8",     # 1–67
    "ct_dst_ltm":    "Int8",     # 1–67
    "ct_src_ ltm":   "Int8",     # 1–67
    "ct_src_dport_ltm": "Int8",  # 1–67
    "ct_dst_sport_ltm": "Int8",  # 1–60
    "ct_dst_src_ltm": "Int8",    # 1–67
    # --- Flotantes ---
    "dur":           "Float32",
    "Sload":         "Float64",  # 5.99e9
    "Dload":         "Float64",  # 1.29e8
    "Sjit":          "Float32",
    "Djit":          "Float32",
    "Sintpkt":       "Float32",
    "Dintpkt":       "Float32",
    "tcprtt":        "Float32",
    "synack":        "Float32",
    "ackdat":        "Float32",
    # --- Binarios ---
    "is_sm_ips_ports": "boolean",
    "is_ftp_login":    "boolean",
    "Label":         "boolean",
}

# Valores corruptos/no-numéricos tratados como NaN durante la lectura, por columna.
# Ctrl: los NA del dict aplican SOLO a la columna indicada (no tocan el resto).
NA_VALUES = {
    "sport":          ["0x000b", "0x000c", "-"],
    "dsport":         ["0xc0a8", "0xcc09", "0x20205321", "-"],
    "ct_ftp_cmd":     [" "],
    "is_ftp_login":   ["2", "4"],
    "state":          ["no"],
}

# --- Leer y concatenar los 4 archivos de datos ---
t0 = time.time()
dfs = []
for i in range(1, 5):
    path = BASE / f"UNSW-NB15_{i}.csv"
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        dtype=DTYPES,
        na_values=NA_VALUES,
        low_memory=False,
    )
    print(f"  UNSW-NB15_{i}.csv → {df.shape[0]:,} filas")
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print(f"Lectura + concat: {time.time() - t0:.1f}s")

# --- Post-proceso: normalizar + convertir nominales de baja cardinalidad ---
# (pd.concat en pandas 3.0 degrada category→str si las categorías difieren
# entre archivos, por eso las limpiezas y conversión se hacen tras el concat)

# 1) Limpiar attack_cat: nulos = tráfico normal → 'Benign'; quitar espacios;
#    unificar singular/plural (' Fuzzers '→Fuzzers, 'Backdoor'→Backdoors, ...)
df["attack_cat"] = (
    df["attack_cat"]
    .fillna("Benign")
    .str.strip()
    .replace({"Backdoor": "Backdoors"})
)
print(f"  attack_cat: categorías corregidas → {df['attack_cat'].dropna().nunique()} únicas")

# 1.1) Verificación del target multiclase: los 'Benign' deben coincidir 1:1 con Label=0.
inc_benigno_ataque = int(((df["attack_cat"] == "Benign") & (df["Label"] == 1)).sum())
inc_ataque_benigno = int(((df["attack_cat"] != "Benign") & (df["Label"] == 0)).sum())
print(f"  verificación target: benignos con Label=1 = {inc_benigno_ataque:,}; "
      f"ataques con Label=0 = {inc_ataque_benigno:,}")
if inc_benigno_ataque or inc_ataque_benigno:
    raise ValueError("attack_cat no se corresponde 1:1 con Label; revisar limpieza.")

# 1.2) 'Label' queda perfectamente derivado del target (Label=1 ⟺ attack_cat≠Benign):
#      como variable de entrada sería target leakage → se descarta del pipeline.
#      (El archivo crudo en .example-datasets no se modifica.)
df = df.drop(columns=["Label"])
print("  'Label' eliminada del pipeline (redundante con el target)")

# 2) Convertir nominales de baja cardinalidad a category
CATEGORY_COLS = ["state", "service", "attack_cat"]
for col in CATEGORY_COLS:
    df[col] = df[col].astype("category")
    print(f"  {col}: → category ({len(df[col].cat.categories)} categorías)")

# --- Verificación ---
print(f"\nDataFrame final: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nMemoria: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# =====================================================================
# SECCIÓN 3.3 DEL PAPER — faltantes, clases, calidad y outliers.
# Regla: se computa la tabla completa, se exporta a EDA/output/ (CSV) y
# en consola se muestra el subconjunto que entra al paper.
# =====================================================================

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _guardar(nombre: str, tabla: pd.DataFrame) -> None:
    tabla.to_csv(OUT_DIR / nombre, index=False)
    print(f"  → {OUT_DIR.name}/{nombre}")


# --- 3.3.1  Valores faltantes -------------------------------------------
print("\n=== 3.3.1  VALORES FALTANTES ===")
miss = df.isna().sum()
tabla_miss = pd.DataFrame({
    "columna": miss[miss > 0].index,
    "faltantes": miss[miss > 0].values,
})
tabla_miss["%"] = tabla_miss["faltantes"] / len(df) * 100
tabla_miss = tabla_miss.sort_values("faltantes", ascending=False)
filas_con_faltante = int(df.isna().any(axis=1).sum())
print(f"  Columnas con faltantes: {len(tabla_miss)} de {df.shape[1]}")
print(f"  Filas con ≥1 faltante: {filas_con_faltante:,} "
      f"({filas_con_faltante / len(df) * 100:.2f}%)")
if len(tabla_miss):
    print(tabla_miss.to_string(index=False))
    _guardar("3.3_faltantes.csv", tabla_miss)
print("  Nota: tokens corruptos ya convertidos a NaN en la lectura (NA_VALUES):")
for col, vals in NA_VALUES.items():
    n = int(df[col].isna().sum())
    if n:
        print(f"    {col}: {n:,} ({n / len(df) * 100:.2f}%)  ← {vals}")

# --- 3.3.2  Distribución de clases (target multiclase) ------------------
print("\n=== 3.3.2  DISTRIBUCIÓN DE CLASES (TARGET) ===")
vc = df["attack_cat"].value_counts()
tabla_clases = pd.DataFrame({"clase": vc.index, "n": vc.values})
tabla_clases["%"] = tabla_clases["n"] / len(df) * 100
print(tabla_clases.to_string(index=False))
_guardar("3.3_distribucion_clases.csv", tabla_clases)
n_ben = int((df["attack_cat"] == "Benign").sum())
print(f"  resumen: benignos {n_ben:,} ({n_ben / len(df) * 100:.2f}%) vs "
      f"ataques {len(df) - n_ben:,} ({(len(df) - n_ben) / len(df) * 100:.2f}%)")

# --- 3.3.3  Problemas de calidad -----------------------------------------
print("\n=== 3.3.3  PROBLEMAS DE CALIDAD ===")
dups = int(df.duplicated().sum())
print(f"  Filas duplicadas: {dups:,} ({dups / len(df) * 100:.4f}%)")
constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print(f"  Columnas constantes (nunique==1): {constantes if constantes else 'ninguna'}")
nominales = ["srcip", "dstip", "sport", "dsport", "proto", "state",
             "service", "attack_cat"]
tabla_card = pd.DataFrame({
    "columna": nominales,
    "unicos": [df[c].nunique(dropna=False) for c in nominales],
})
tabla_card["%_unicos"] = tabla_card["unicos"] / len(df) * 100
print(tabla_card.to_string(index=False))
_guardar("3.3_cardinalidad.csv", tabla_card)

# --- 3.3.4  Descriptivos y outliers (IQR) -------------------------------
print("\n=== 3.3.4  DESCRIPTIVOS Y OUTLIERS (IQR) ===")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

desc = df[num_cols].describe().T
print(desc.to_string())
_guardar("3.3_descriptivos.csv",
         desc.reset_index().rename(columns={"index": "columna"}))


def _iqr_outliers(s: pd.Series) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    if q3 - q1 == 0:
        return 0
    inf, sup = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return int(((s < inf) | (s > sup)).sum())


filas = []
for c in num_cols:
    s = df[c].astype("float64")
    n = _iqr_outliers(s)
    filas.append({
        "columna": c,
        "outliers_IQR": n,
        "%": n / len(df) * 100,
        "%_ceros": float((df[c] == 0).mean() * 100),
    })
tabla_out = pd.DataFrame(filas).sort_values("outliers_IQR", ascending=False)
print(f"\n  Top 15 columnas por nº de outliers (tabla completa en CSV):")
print(tabla_out.head(15).to_string(index=False))
_guardar("3.3_outliers_iqr.csv", tabla_out)
