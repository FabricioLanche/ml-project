# 3.4 — Visualizaciones del EDA (sección 3.4 del paper).
# Dependencias: solo pandas, numpy y matplotlib (sin seaborn/sklearn).
# Misma carga/preproceso que 3.3.py. Las figuras se guardan en EDA/output/.

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

OUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUT_DIR
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

plt.rc("font", size=10)
AZUL, ROJO = "#2e86c1", "#c0392b"


def cargar_datos() -> pd.DataFrame:
    """Idéntico pipeline a 3.3.py: lectura, limpieza y target multiclase."""
    t0 = time.time()
    dfs = []
    for i in range(1, 5):
        path = BASE / f"UNSW-NB15_{i}.csv"
        dfs.append(pd.read_csv(
            path, header=None, names=COLUMNS, dtype=DTYPES,
            na_values=NA_VALUES, low_memory=False,
        ))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Lectura + concat: {time.time() - t0:.1f}s")

    df["attack_cat"] = (
        df["attack_cat"].fillna("Benign").str.strip()
        .replace({"Backdoor": "Backdoors"})
    )
    inc_benigno_ataque = int(((df["attack_cat"] == "Benign") & (df["Label"] == 1)).sum())
    inc_ataque_benigno = int(((df["attack_cat"] != "Benign") & (df["Label"] == 0)).sum())
    if inc_benigno_ataque or inc_ataque_benigno:
        raise ValueError("attack_cat no se corresponde 1:1 con Label; revisar limpieza.")
    df = df.drop(columns=["Label"])
    for col in ["state", "service", "attack_cat"]:
        df[col] = df[col].astype("category")
    df["is_attack"] = (df["attack_cat"] != "Benign").astype(np.int8)
    return df


def _guardar_fig(fig, nombre: str) -> None:
    salida = FIG_DIR / nombre
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {salida.relative_to(Path(__file__).resolve().parent)}")


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


# ============================ CARGA =====================================
df = cargar_datos()
print(f"DataFrame: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# ============================ FIG 01 y 02 — TARGET =======================
vc = df["attack_cat"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
colores = [AZUL if c == "Benign" else ROJO for c in vc.index]
ax.barh(vc.index, vc.values, color=colores)
ax.set_xscale("log")
ax.set_xlabel("nº de registros (escala log)")
ax.set_title("Distribución del target — 11 clases")
for y, (n, p) in enumerate(zip(vc.values, vc.values / len(df) * 100)):
    ax.text(n, y, f"  {n:,} ({p:.2f}%)", va="center", fontsize=8)
fig.tight_layout()
_guardar_fig(fig, "fig01_distribucion_clases.png")

n_ben = int((df["attack_cat"] == "Benign").sum())
n_ataq = len(df) - n_ben
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([n_ben, n_ataq], labels=["Benign", "Ataque"], autopct="%.1f%%",
       colors=[AZUL, ROJO], startangle=90, wedgeprops=dict(width=0.35))
ax.set_title("Split binario — Benign vs Ataque")
_guardar_fig(fig, "fig02_benigno_vs_ataque.png")

# ====================== FIG 03 — VALORES FALTANTES ======================
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
colores = [paleta[origen[c]] for c in miss.index]
ax.barh(miss.index, miss.values / len(df) * 100, color=colores)
ax.set_xlabel("% de filas con valor faltante")
ax.set_title("Valores faltantes por columna")
for y, (c, n) in enumerate(miss.items()):
    ax.text(n / len(df) * 100, y, f"  {n:,} ({n / len(df) * 100:.2f}%)", va="center", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v, label=k) for k, v in paleta.items()], fontsize=8, loc="lower right")
fig.tight_layout()
_guardar_fig(fig, "fig03_valores_faltantes.png")

# ====================== FIG 04 — CORRELACIÓN FEATURES ====================
sel = ["dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts",
       "sttl", "dttl", "sloss", "dloss", "res_bdy_len", "Sjit", "Djit",
       "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat"]
muestra_corr = df.sample(n=min(500_000, len(df)), random_state=42)
mat_corr = muestra_corr[["is_attack"] + sel].corr().values
heatmap(mat_corr, ["is_attack"] + sel, ["is_attack"] + sel,
        "Correlación de features numéricas (muestra 500k)",
        "fig04_correlacion_features.png")

# ================= FIG 05 — ct_* vs target (leakage) =====================
ct_cols = ["ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src",
           "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
           "ct_dst_sport_ltm", "ct_dst_src_ltm"]
mat_ct = muestra_corr[["is_attack"] + ct_cols].corr().values
heatmap(mat_ct, ["is_attack"] + ct_cols, ["is_attack"] + ct_cols,
        "Features de ventana ct_* vs. target (leakage potencial)",
        "fig05_ct_vs_target.png")

# ====================== FIG 06 — HISTOGRAMAS POR CLASE ===================
muestra = df.sample(n=200_000, random_state=42)
m_ben = muestra[muestra["attack_cat"] == "Benign"]
m_ataq = muestra[muestra["attack_cat"] != "Benign"]
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

# ====================== FIG 07 — NOMINALES ===============================
def tabla_con_otros(s, top_n=12):
    vc = s.value_counts()
    if len(vc) <= top_n:
        return vc.sort_values()
    resto = vc.iloc[top_n:].sum()
    top = vc.iloc[:top_n].sort_values()
    top = pd.concat([top, pd.Series({"Otros": resto})])
    return top.sort_values()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, col in zip(axes, ["proto", "state", "service"]):
    t = tabla_con_otros(df[col])
    ax.barh(t.index.astype(str), t.values, color="#2e86c1")
    ax.set_title(f"Distribución de {col}")
    ax.set_xlabel("frecuencia")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
for ax in axes[1:]:
    ax.set_xscale("log")
fig.suptitle("Variables nominales más frecuentes ('Otros' agrupa el resto)")
fig.tight_layout()
_guardar_fig(fig, "fig07_nominales.png")

# ====================== FIG 08 — PCA 2D (numpy SVD) ======================
print("\n  PCA sobre muestra (SVD de numpy, sin sklearn)…")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
num_cols = [c for c in num_cols if c != "is_attack"]
m2 = muestra[num_cols].dropna()
# astype("float64") obligatorio: np.asarray sobre dtypes anulables
# (Int8Dtype/Float32Dtype…) devuelve un array object y svd falla.
Xc = ((m2 - m2.mean()) / m2.std()).astype("float64")
X_arr = Xc.to_numpy()
U, S, Vt = np.linalg.svd(X_arr, full_matrices=False)
proj = (X_arr @ Vt.T)[:, :2]
explained = S ** 2 / (len(Xc) - 1)
pve1, pve2 = explained[0] / explained.sum(), explained[1] / explained.sum()
etiquetas = df.loc[m2.index, "is_attack"]

fig, ax = plt.subplots(figsize=(9, 7))
for val, nombre, color in [(0, "Benign", AZUL), (1, "Ataque", ROJO)]:
    mask = etiquetas == val
    ax.scatter(proj[mask.values, 0], proj[mask.values, 1], s=2, alpha=0.4,
               c=color, label=nombre)
ax.set_xlabel(f"PC1 ({pve1 * 100:.1f}% de varianza)")
ax.set_ylabel(f"PC2 ({pve2 * 100:.1f}% de varianza)")
ax.set_title("PCA 2D de features numéricas (muestra 200k, estandarizadas)")
ax.legend(markerscale=8, fontsize=10)
fig.tight_layout()
_guardar_fig(fig, "fig08_pca.png")

print("\nFiguras generadas en EDA/output/")