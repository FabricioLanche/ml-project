# =============================================================================
# Análisis exploratorio de datos - UNSW-NB15
# =============================================================================
# Script equivalente al notebook EDA/eda.ipynb, pensado para revisar las
# salidas directamente en terminal (pandas/matplotlib vía .venv).
#
# Dependencias (ya instaladas en .venv): pandas, numpy, matplotlib, kagglehub.
# Uso:  .venv/bin/python EDA/eda.py
# Las figuras se guardan en EDA/output/.
# =============================================================================


# =============================================================================
# 1. Configuración y carga de datos
# -----------------------------------------------------------------------------
# El dataset se descarga de Kaggle (vía `kagglehub`) **la primera vez** y queda
# fijo en la carpeta `dataset/` de la raíz del repo (ignorada por git). Si ya
# está descargada, se reutiliza sin volver a bajar nada.
#
# El esquema de columnas está **extraído manualmente** del diccionario oficial
# (`NUSW-NB15_features.csv`) y se aplica como constantes `COLUMNS`/`DTYPES`. Los
# tokens corruptos (`0x000b`, `-`, `'no'`, …) se convierten a NaN con
# `na_values` durante la lectura.
# =============================================================================

import shutil
from pathlib import Path
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Rutas relativas al propio script (EDA/eda.py → raíz del repo)
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "dataset"
OUT_DIR = BASE / "EDA" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constantes de estilo
AZUL, ROJO = "#2e86c1", "#c0392b"
plt.rc("font", size=10)

# Flujo de carga del dataset de kaggle
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

# --- Lectura y concatenación de los 4 csv's ---
print("\n=== 1. CARGA DE DATOS ===")
dfs = []
for i in range(1, 5):
    path = DATA_DIR / f"UNSW-NB15_{i}.csv"
    dfs.append(
        pd.read_csv(
            path,
            header=None,
            names=COLUMNS,
            dtype=DTYPES,
            na_values=NA_VALUES,
            low_memory=False,
        )
    )
df = pd.concat(dfs, ignore_index=True)
print(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print("Primeras filas:")
print(df.head().to_string())


# =============================================================================
# 2. Limpieza y definición del target multiclase
# -----------------------------------------------------------------------------
# Decisión metodológica del proyecto: los registros normales (`Label=0`) no
# tienen categoría, así que se agrupan en una única clase **`Benign`** (11.ª).
# =============================================================================

print("\n=== 2. LIMPIEZA Y TARGET MULTICLASE ===")

# Los registros sin attack_cat forman la categoría "Benign"
df["attack_cat"] = (
    df["attack_cat"]
    .fillna("Benign")
    .str.strip()
    .replace({"Backdoor": "Backdoors"})
)

for col in ["state", "service", "attack_cat"]:
    df[col] = df[col].astype("category")

print(df["state"].value_counts())
print()
print(df["service"].value_counts())
print()
print(df["attack_cat"].value_counts())


# =============================================================================
# 3. Análisis y exploración del dataset
# -----------------------------------------------------------------------------
# 3.1 Valores faltantes
#
# Tres columnas con ≈53–56 % de NaN son **estructurales** (solo se definen
# para servicios FTP/HTTP); las de «tokens corruptos» vienen de valores
# ilegibles convertidos a NaN en la lectura.
# =============================================================================

print("\n=== 3.1 VALORES FALTANTES ===")
miss = df.isna().sum()
tabla_miss = pd.DataFrame({
    "columna": miss[miss > 0].index,
    "faltantes": miss[miss > 0].values,
})
tabla_miss["%"] = tabla_miss["faltantes"] / len(df) * 100
tabla_miss = tabla_miss.sort_values("faltantes", ascending=False)
print(f"Columnas con datos faltantes: {len(tabla_miss)} de {df.shape[1]}")
print(f"Filas con datos faltantes: {df.isna().any(axis=1).sum():,} "
      f"({df.isna().any(axis=1).mean() * 100:.2f}%)")
print(tabla_miss.to_string(index=False))


# 3.1b) Los NaN de las columnas ftp/http son estructurales (cruce con service)
print("\n--- Cruce del NaN con el servicio ---")
es_ftp = (df["service"] == "ftp").map({True: "ftp", False: "no-ftp"})
tab_ftp = pd.crosstab(es_ftp, df["ct_ftp_cmd"].isna().map({True: "NaN", False: "valor"}),
                      margins=True).rename(index={"All": "total"}, columns={"All": "total"})
print("ct_ftp_cmd NaN x servicio (concentrado en no-ftp?):")
print(tab_ftp.to_string())
print()

es_http = (df["service"] == "http").map({True: "http", False: "no-http"})
tab_http = pd.crosstab(es_http, df["ct_flw_http_mthd"].isna().map({True: "NaN", False: "valor"}),
                       margins=True).rename(index={"All": "total"}, columns={"All": "total"})
print("ct_flw_http_mthd NaN x servicio (concentrado en no-http?):")
print(tab_http.to_string())


# 3.1c) Estadísticos descriptivos de las variables numéricas
# -----------------------------------------------------------------------------
# Resumen numérico (n, promedio, desv. estándar, mín., cuartiles, mediana, máx.
# y % de ceros) para todas las variables numéricas. Permite globalizar el
# comportamiento de las 48 columnas y detectar distribuciones extremas frente
# a la lectura por variable individual.
# =============================================================================

print("\n=== 3.1c DESCRIPTIVOS DE VARIABLES NUMÉRICAS ===")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

filas_desc = []
for c in num_cols:
    s = df[c].astype("float64")
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    filas_desc.append({
        "variable": c,
        "n": int(s.count()),
        "media": round(s.mean(), 4),
        "std": round(s.std(), 4),
        "min": round(s.min(), 4),
        "q1": round(q1, 4),
        "mediana": round(med, 4),
        "q3": round(q3, 4),
        "max": round(s.max(), 4),
        "%_ceros": round(float((s == 0).mean() * 100), 2),
    })
tabla_desc = pd.DataFrame(filas_desc)
print(tabla_desc.to_string(index=False))
tabla_desc.to_csv(OUT_DIR / "3.1c_descriptivos.csv", index=False)
print(f"\n→ {OUT_DIR.name}/3.1c_descriptivos.csv exportado")


# =============================================================================
# 3.2 Distribución de clases (target)
#
# Existe desbalance de las clases: `Benign` 87.4 % vs ataques 12.6 %, y dentro
# de los ataques, `Generic` concentra la mayor parte mientras `Worms` tiene
# solo 174 registros (0.0069 %).
# =============================================================================

print("\n=== 3.2 DISTRIBUCIÓN DE CLASES (TARGET) ===")
vc = df["attack_cat"].value_counts()
tabla_clases = pd.DataFrame({"clase": vc.index, "n": vc.values})
tabla_clases["%"] = tabla_clases["n"] / len(df) * 100
print(tabla_clases.to_string(index=False))


# =============================================================================
# 3.3 Problemas de calidad
#
# - **Filas duplicadas:** hallazgo relevante (~19 % del total). Son duplicados
#   exactos (mismas features y mismo `attack_cat`), por lo que son candidatos a
#   eliminación en la fase de preprocesamiento.
# - **Cardinalidad de nominales:** `srcip`/`dstip` tienen solo 43 y 47 valores
#   únicos (menos de lo esperado para IPs); `sport`/`dsport` son de alta
#   cardinalidad.
# =============================================================================

print("\n=== 3.3 PROBLEMAS DE CALIDAD ===")
dups = int(df.duplicated().sum())
print(f"Filas duplicadas: {dups:,} ({dups / len(df) * 100:.4f}%)")

constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print(f"Columnas constantes (nunique==1): {constantes if constantes else 'ninguna'}")

nominales = ["srcip", "dstip", "sport", "dsport", "proto", "state", "service", "attack_cat"]
tabla_card = pd.DataFrame({
    "columna": nominales,
    "unicos": [df[c].nunique(dropna=False) for c in nominales],
})
tabla_card["%_unicos"] = tabla_card["unicos"] / len(df) * 100
print("\nCardinalidad de variables nominales:")
print(tabla_card.sort_values("unicos", ascending=True).to_string(index=False))


# =============================================================================
# 3.4 Descriptivos y outliers (IQR)
#
# Outliers por rango intercuartílico (Q1/Q3 ± 1.5·IQR). Varias features tienen
# cola pesada (`Sload`, `dur`, `Sjit`) y alto % de ceros, lo que explica los
# conteos grandes de "outliers"; en P2 habrá que decidir transformación/
# tratamiento.
# =============================================================================

print("\n=== 3.4 DESCRIPTIVOS Y OUTLIERS (IQR) ===")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()


def _iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    if q3 - q1 == 0:
        return 0
    inf, sup = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return int(((s < inf) | (s > sup)).sum())


filas = []
for c in num_cols:
    n = _iqr_outliers(df[c].astype("float64"))
    filas.append({
        "columna": c,
        "outliers_IQR": n,
        "%": n / len(df) * 100,
        "%_ceros": float((df[c] == 0).mean() * 100),
    })
tabla_out = pd.DataFrame(filas).sort_values("outliers_IQR", ascending=False)
print("Top 15 columnas por nº de outliers IQR:")
print(tabla_out.head(15).to_string(index=False))


# =============================================================================
# 4. Visualizaciones gráficas
# =============================================================================

# Utilidades de visualización
def _guardar_fig(fig, nombre):
    salida = OUT_DIR / nombre
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"→ {salida.name}")
    return fig


def heatmap(mat, filas, col_etiquetas, titulo, nombre, cmap="RdBu_r", vmin=-1, vmax=1):
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(filas)), max(5, 0.42 * len(filas))))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_etiquetas)))
    ax.set_xticklabels(col_etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(filas, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="correlación")
    ax.set_title(titulo)
    fig.tight_layout()
    return _guardar_fig(fig, nombre)


print("\n=== 4. VISUALIZACIONES (guardadas en EDA/output/) ===")

# fig01 — distribución del target (11 clases)
vc_asc = df["attack_cat"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(vc_asc.index, vc_asc.values, color=[AZUL if c == "Benign" else ROJO for c in vc_asc.index])
ax.set_xscale("log")
ax.set_xlabel("nº de registros (escala log)")
ax.set_title("Distribución del target — 11 clases")
for y, (n, p) in enumerate(zip(vc_asc.values, vc_asc.values / len(df) * 100)):
    ax.text(n, y, f"  {n:,} ({p:.2f}%)", va="center", fontsize=8)
fig.tight_layout()
_guardar_fig(fig, "fig01_distribucion_clases.png")
plt.close(fig)

# fig02 — split binario Benign vs Ataque
n_ben = int((df["attack_cat"] == "Benign").sum())
n_ataq = len(df) - n_ben
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([n_ben, n_ataq], labels=["Benign", "Ataque"], autopct="%.1f%%",
       colors=[AZUL, ROJO], startangle=90, wedgeprops=dict(width=0.35))
ax.set_title("Split binario — Benign vs Ataque")
_guardar_fig(fig, "fig02_benigno_vs_ataque.png")
plt.close(fig)

# fig03 — valores faltantes por columna y origen
miss = df.isna().sum()
miss = miss[miss > 0].sort_values()
origen = {
    "ct_flw_http_mthd": "estructural (HTTP)",
    "is_ftp_login": "estructural FTP/corrupto",
    "ct_ftp_cmd": "estructural FTP/corrupto",
    "sport": "tokens corruptos",
    "dsport": "tokens corruptos",
    "state": "tokens corruptos",
}
paleta = {"estructural (HTTP)": "#27ae60", "estructural FTP/corrupto": "#8e44ad",
          "tokens corruptos": "#f39c12"}
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(miss.index, miss.values / len(df) * 100,
        color=[paleta[origen[c]] for c in miss.index])
ax.set_xlabel("% de filas con valor faltante")
ax.set_title("Valores faltantes por columna")
for y, (c, n) in enumerate(miss.items()):
    ax.text(n / len(df) * 100, y, f"  {n:,} ({n / len(df) * 100:.2f}%)", va="center", fontsize=8)
ax.legend(handles=[Patch(color=v, label=k) for k, v in paleta.items()], fontsize=8, loc="lower right")
fig.tight_layout()
_guardar_fig(fig, "fig03_valores_faltantes.png")
plt.close(fig)

# fig04 — correlación de features numéricas seleccionadas (Label se mantiene para la exploración)
sel = ["dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts",
       "sttl", "dttl", "sloss", "dloss", "res_bdy_len", "Sjit", "Djit",
       "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat"]
mat = df[["Label"] + sel].corr().values
heatmap(
  mat, ["Label"] + sel, ["Label"] + sel,
  "Correlación de features numéricas",
  "fig04_correlacion_features.png"
)
plt.close("all")

# fig05 — features de ventana ct_* vs target (leakage potencial)
ct_cols = ["ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src",
           "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
           "ct_dst_sport_ltm", "ct_dst_src_ltm"]
mat = df[["Label"] + ct_cols].corr().values
heatmap(
  mat, ["Label"] + ct_cols, ["Label"] + ct_cols,
  "Features de ventana ct_* vs. target (leakage potencial)",
  "fig05_ct_vs_target.png"
)
plt.close("all")

# fig06 — histogramas log10 de features clave, por clase
m_ben = df[df["attack_cat"] == "Benign"]
m_ataq = df[df["attack_cat"] != "Benign"]
num_hist = ["dur", "sbytes", "dbytes", "Sload", "Dload", "tcprtt"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), num_hist):
    ax.hist(np.log10(m_ben[col].astype("float64") + 1), bins=60, alpha=0.5,
            label="Benign", color=AZUL, density=True)
    ax.hist(np.log10(m_ataq[col].astype("float64") + 1), bins=60, alpha=0.5,
            label="Ataque", color=ROJO, density=True)
    ax.set_title(col)
    ax.set_xlabel("log10(x+1)")
    ax.set_ylabel("densidad")
    ax.legend(fontsize=8)
fig.suptitle("Distribución de features numéricas clave por clase (muestra 200k)")
fig.tight_layout()
_guardar_fig(fig, "fig06_histogramas_num.png")
plt.close(fig)

# fig06b — boxplots de features clave en escala logarítmica
# Reemplaza al violin de la versión anterior: en variables de cola larga y
# múltiples órdenes de magnitud, el boxplot permite visualizar la dispersión
# y los outliers (fliers). Se usan los datos completos y la configuración
# base de matplotlib.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), num_hist):
    datos = np.log10(df[col].astype("float64").dropna().to_numpy() + 1)
    ax.boxplot(datos, vert=True)
    ax.set_xticks([1])
    ax.set_xticklabels([col], fontsize=10)
    ax.set_ylabel("log10(x+1)", fontsize=9)
fig.suptitle("Boxplot de selección de variables numéricas")
fig.tight_layout()
_guardar_fig(fig, "fig06b_boxplot.png")
plt.close(fig)

# fig07 — variables nominales (con 'Otros' para colas largas)
def tabla_con_otros(s, top_n=12):
    vc = s.value_counts()
    if len(vc) <= top_n:
        return vc.sort_values()
    resto = vc.iloc[top_n:].sum()
    top = vc.iloc[:top_n].sort_values()
    return pd.concat([top, pd.Series({"Otros": resto})]).sort_values()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, col in zip(axes, ["proto", "state", "service"]):
    t = tabla_con_otros(df[col])
    ax.barh(t.index.astype(str), t.values, color=AZUL)
    ax.set_title(f"Distribución de {col}")
    ax.set_xlabel("frecuencia")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
for ax in axes[1:]:
    ax.set_xscale("log")
fig.suptitle("Variables nominales más frecuentes ('Otros' agrupa el resto)")
fig.tight_layout()
_guardar_fig(fig, "fig07_nominales.png")
plt.close(fig)

print("\nFiguras generadas en EDA/output/")