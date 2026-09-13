# =============================================================================
# EDA - UNSW-NB15 · reproduce únicamente los outputs usados en el informe
# =============================================================================
# Script único que genera las salidas referenciadas en main.tex:
#   · Tablas/datos → EDA/output/*.csv  (respaldo de las tablas del informe)
#   · Figuras       → EDA/output/fig01..fig05 (*.png)
#
# Dependencias: pandas, numpy, matplotlib, kagglehub (primera descarga).
# Uso:  .venv/bin/python EDA/eda.py  (o `python3 EDA/eda.py`)
# =============================================================================

import shutil
from pathlib import Path
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "dataset"
OUT_DIR = BASE / "EDA" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constantes de estilo
AZUL, ROJO = "#2e86c1", "#c0392b"
plt.rc("font", size=10)

# --- Carga del dataset (descarga de Kaggle solo la primera vez) -----------
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

# Tokens corruptos de la fuente convertidos a NaN durante la lectura
NA_VALUES = {
    "sport": ["0x000b", "0x000c", "-"],
    "dsport": ["0xc0a8", "0xcc09", "0x20205321", "-"],
    "ct_ftp_cmd": [" "],
    "is_ftp_login": ["2", "4"],
    "state": ["no"],
}

# Origen de cada columna con valores faltantes (usado en la tabla del informe)
ORIGEN = {
    "is_ftp_login": "Estructural (FTP)",
    "ct_ftp_cmd": "Estructural (FTP)",
    "ct_flw_http_mthd": "Estructural (HTTP)",
    "dsport": "Token corrupto",
    "sport": "Token corrupto",
    "state": "Token corrupto",
}


# ===========================================================================
# Carga y limpieza
# ===========================================================================
print("\n=== CARGA DE DATOS ===")
dfs = []
for i in range(1, 5):
    dfs.append(pd.read_csv(
        DATA_DIR / f"UNSW-NB15_{i}.csv",
        header=None, names=COLUMNS, dtype=DTYPES,
        na_values=NA_VALUES, low_memory=False,
    ))
df = pd.concat(dfs, ignore_index=True)
print(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")

df["attack_cat"] = (
    df["attack_cat"].fillna("Benign").str.strip().replace({"Backdoor": "Backdoors"})
)

# Verificación: Benign ≡ Label=0 (coincidencia 1:1)
inc_benigno_ataque = int(((df["attack_cat"] == "Benign") & (df["Label"] == 1)).sum())
inc_ataque_benigno = int(((df["attack_cat"] != "Benign") & (df["Label"] == 0)).sum())
if inc_benigno_ataque or inc_ataque_benigno:
    raise ValueError("attack_cat no se corresponde 1:1 con Label; revisar limpieza.")
print("Verificación target multiclase OK: Benign ⟺ Label=0 al 100%.")

for col in ["state", "service", "attack_cat"]:
    df[col] = df[col].astype("category")

N = len(df)


# ===========================================================================
# Utilidades
# ===========================================================================
def _guardar_fig(fig, nombre):
    salida = OUT_DIR / nombre
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {salida.name}")


def _guardar_csv(tabla, nombre):
    tabla.to_csv(OUT_DIR / nombre, index=False)
    print(f"  → {nombre}")
    return tabla


def _iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    if q3 - q1 == 0:
        return 0
    inf, sup = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return int(((s < inf) | (s > sup)).sum())


def _densidad_log(s, max_puntos=200_000, semilla=42, factor=0.8):
    """KDE binned en log10(x+1) con reflexión en el mínimo (solo numpy).
    Devuelve (x, densidad) para superponer sobre los histogramas."""
    x = np.log10(s.astype("float64").dropna().to_numpy() + 1)
    if len(x) > max_puntos:
        rng = np.random.default_rng(semilla)
        x = rng.choice(x, size=max_puntos, replace=False)
    b = x.min()
    xk = np.concatenate([x, 2 * b - x])
    h = factor * 1.06 * xk.std() * len(xk) ** (-1 / 5)
    span = xk.max() - xk.min()
    n_bins = max(300, min(int(span / (h / 6)), 6000))
    counts, edges = np.histogram(xk, bins=n_bins)
    w = edges[1] - edges[0]
    k = max(3, int(round(3 * h / w)) * 2 + 1)
    t = (np.arange(k) - k // 2) * w
    kernel = np.exp(-0.5 * (t / h) ** 2)
    kernel /= kernel.sum()
    dens = np.convolve(counts, kernel, mode="same") / (len(xk) * w)
    xs = 0.5 * (edges[:-1] + edges[1:])
    mask = xs >= b
    return xs[mask], dens[mask] * 2.0


def _eta2_num_cat(series, labels):
    """Razón de correlación η² entre una variable numérica y una categórica."""
    s = series.astype("float64").dropna()
    lab = labels.loc[s.index]
    if lab.nunique(dropna=False) <= 1 or s.nunique() <= 1:
        return 0.0
    xbar = s.mean()
    ss_total = float(((s - xbar) ** 2).sum())
    ssb = float(
        lab.to_frame("y").assign(x=s).groupby("y")["x"]
        .agg(["mean", "count"])
        .assign(d=lambda t: t["count"] * (t["mean"] - xbar) ** 2)["d"].sum()
    )
    return ssb / ss_total if ss_total > 0 else 0.0


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
    _guardar_fig(fig, nombre)


# ===========================================================================
# 1. DATOS Y TABLAS DEL INFORME (CSV)
# ===========================================================================
print("\n=== 1. EXPORTACIÓN DE TABLAS (CSV) ===")

num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# T01 — Valores faltantes (Tabla 2 del informe: tab:faltantes)
print("\n[01] valores faltantes")
miss = df.isna().sum()
faltantes = pd.DataFrame({
    "columna": miss[miss > 0].index,
    "faltantes": miss[miss > 0].values,
    "origen": [ORIGEN[c] for c in miss[miss > 0].index],
})
faltantes["%"] = faltantes["faltantes"] / N * 100
faltantes = faltantes.sort_values("faltantes", ascending=False)
print(f"  {len(faltantes)} columnas con faltantes · "
      f"{df.isna().any(axis=1).sum():,} filas afectadas "
      f"({df.isna().any(axis=1).mean() * 100:.2f}%)")
_guardar_csv(faltantes, "faltantes.csv")

# T02 — Estadísticos descriptivos (tab:descriptivos)
print("\n[02] descriptivos de variables numéricas")
filas_desc = []
for c in num_cols:
    s = df[c].astype("float64")
    filas_desc.append({
        "variable": c,
        "n": int(s.count()),
        "media": round(s.mean(), 4),
        "mediana": round(s.median(), 4),
        "varianza": round(s.var(), 4),   # muestra (ddof=1), igual que el informe
        "min": round(s.min(), 4),
        "max": round(s.max(), 4),
        "%_ceros": round(float((s == 0).mean() * 100), 2),
    })
_guardar_csv(pd.DataFrame(filas_desc), "descriptivos.csv")

# T03 — Distribución de clases (usada en fig01 y en el texto)
print("\n[03] distribución de clases del target")
vc = df["attack_cat"].value_counts()
clases = pd.DataFrame({"clase": vc.index, "n": vc.values})
clases["%"] = clases["n"] / N * 100
_guardar_csv(clases, "distribucion_clases.csv")

# T04 — Cardinalidad de nominales (tab:cardinalidad)
print("\n[04] cardinalidad de variables nominales")
nominales = ["srcip", "dstip", "sport", "dsport", "proto", "state", "service", "attack_cat"]
cardinalidad = pd.DataFrame({
    "columna": nominales,
    "unicos": [df[c].nunique(dropna=False) for c in nominales],
})
cardinalidad["%_unicos"] = cardinalidad["unicos"] / N * 100
cardinalidad = cardinalidad.sort_values("unicos", ascending=True)
print(f"  filas duplicadas: {int(df.duplicated().sum()):,} "
      f"({df.duplicated().mean() * 100:.2f}%) · columnas constantes: "
      f"{[c for c in df.columns if df[c].nunique(dropna=False) <= 1] or 'ninguna'}")
_guardar_csv(cardinalidad, "cardinalidad.csv")

# T05 — Outliers por IQR (tab:outliers)
print("\n[05] outliers por IQR y % de ceros")
filas_out = []
for c in num_cols:
    n = _iqr_outliers(df[c].astype("float64"))
    filas_out.append({
        "columna": c,
        "outliers_IQR": n,
        "%_outliers": round(n / N * 100, 2),
        "%_ceros": round(float((df[c] == 0).mean() * 100), 2),
    })
outliers = pd.DataFrame(filas_out).sort_values("outliers_IQR", ascending=False)
print("  top 5 por % outliers:")
print(outliers.head(5)[["columna", "%_outliers", "%_ceros"]].to_string(index=False))
_guardar_csv(outliers, "outliers_iqr.csv")

# T06 — η² numéricas vs attack_cat (tab:eta)
print("\n[06] razón de correlación η² vs attack_cat (muestra 500k)")
muestra = df.sample(n=500_000, random_state=42)
eta = pd.DataFrame({
    "columna": num_cols,
    "eta2": [_eta2_num_cat(muestra[c], muestra["attack_cat"]) for c in num_cols],
}).sort_values("eta2", ascending=False)
_guardar_csv(eta, "eta_attack_cat.csv")


# ===========================================================================
# 2. FIGURAS DEL INFORME (renumeradas en orden de aparición)
# ===========================================================================
print("\n=== 2. FIGURAS (guardadas en EDA/output/) ===")

# fig01 — Distribución del target (10 clases)  → fig:clases
print("\n[fig01] distribución del target")
vc_asc = vc.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(vc_asc.index, vc_asc.values,
        color=[AZUL if c == "Benign" else ROJO for c in vc_asc.index])
ax.set_xscale("log")
ax.set_xlabel("nº de registros (escala log)")
ax.set_title("Distribución del target — 10 clases")
for y, (n, p) in enumerate(zip(vc_asc.values, vc_asc.values / N * 100)):
    ax.text(n * 1.15, y, f"{n:,}\n({p:.2f}%)", ha="left", va="center", fontsize=8)
ax.set_xlim(right=vc_asc.values.max() * 5)
fig.tight_layout()
_guardar_fig(fig, "fig01_distribucion_clases.png")

# fig02 — Boxplots top 9 por % de outliers  → fig:boxplot
print("\n[fig02] boxplots top 9 por % de outliers")
out_pct = pd.DataFrame({
    "columna": num_cols,
    "%": [_iqr_outliers(df[c].astype("float64")) / N * 100 for c in num_cols],
}).nlargest(9, "%")
top9 = out_pct["columna"].tolist()
print(f"  top 9: {top9}")
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, col in zip(axes.ravel(), top9):
    datos = np.log10(df[col].astype("float64").dropna().to_numpy() + 1)
    ax.boxplot(datos, vert=True)
    ax.set_xticks([1])
    ax.set_xticklabels([col], fontsize=10)
    ax.set_ylabel("log10(x+1)", fontsize=9)
fig.suptitle("Boxplot del top 9 de variables numéricas por % de outliers")
fig.tight_layout()
_guardar_fig(fig, "fig02_boxplot_outliers.png")

# fig03 — Correlación de Pearson con el target binario  → fig:correlacion
print("\n[fig03] correlación de features numéricas con el target binario")
sel = ["dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts",
       "sttl", "dttl", "sloss", "dloss", "res_bdy_len", "Sjit", "Djit",
       "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat"]
mat = df[["Label"] + sel].corr().values
heatmap(mat, ["Label"] + sel, ["Label"] + sel,
        "Correlación de Pearson con el target binario",
        "fig03_correlacion_features.png")
plt.close("all")

# fig04 — Histogramas del top 9 por correlación con Label + KDE  → fig:hist
print("\n[fig04] histogramas top 9 por correlación con el target")
m_ben = df[df["attack_cat"] == "Benign"]
m_ataq = df[df["attack_cat"] != "Benign"]
corr_label = df[num_cols].corrwith(df["Label"]).dropna().abs().sort_values(ascending=False)
top9_corr = corr_label.head(9).index.tolist()
print(f"  top 9: {top9_corr}")
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, col in zip(axes.ravel(), top9_corr):
    ax.hist(np.log10(m_ben[col].astype("float64") + 1), bins=60, alpha=0.4,
            label="Benign", color=AZUL, density=True)
    ax.hist(np.log10(m_ataq[col].astype("float64") + 1), bins=60, alpha=0.4,
            label="Ataque", color=ROJO, density=True)
    xs, pdf = _densidad_log(m_ben[col])
    ax.plot(xs, pdf, color=AZUL, lw=1.6)
    xs, pdf = _densidad_log(m_ataq[col])
    ax.plot(xs, pdf, color=ROJO, lw=1.6)
    ax.set_title(col)
    ax.set_xlabel("log10(x+1)")
    ax.set_ylabel("densidad")
    ax.legend(fontsize=8)
fig.suptitle("Distribución del top 9 de variables numéricas por correlación con el target binario")
fig.tight_layout()
_guardar_fig(fig, "fig04_histogramas_top9.png")

# fig05 — Correlación de los ct_* con el target  → fig:ct_corr
print("\n[fig05] correlación de variables de ventana ct_* con el target")
ct_cols = ["ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src",
           "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
           "ct_dst_sport_ltm", "ct_dst_src_ltm"]
mat = df[["Label"] + ct_cols].corr().values
heatmap(mat, ["Label"] + ct_cols, ["Label"] + ct_cols,
        "Correlación de las variables de ventana ct_* con el target",
        "fig05_ct_vs_target.png")
plt.close("all")

print("\nOutputs del informe regenerados en EDA/output/")