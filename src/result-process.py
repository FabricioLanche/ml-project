"""Genera las figuras de la sección de resultados a partir de los CSV de fase 3.

Las tablas de fase 1, fase 2 y por clase ya se construyen directamente desde sus
CSV. Este script produce:

1. Un gráfico de barras agrupadas con los tres modelos y, para cada uno, accuracy,
   precisión macro, recall macro y macro F1.
2. Una matriz de confusión normalizada por modelo.

Salida: ``results/figs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FASE_3 = RAIZ / "results" / "fase_3"
SALIDA = RAIZ / "results" / "figs"

MODELOS = {
    "rf": "Random Forest",
    "logreg": "Regresión logística",
    "svm": "SVM lineal",
}

METRICAS = [
    ("accuracy", "Accuracy"),
    ("promedio_precision", "Precisión macro"),
    ("promedio_recall", "Recall macro"),
    ("promedio_F1", "Macro F1"),
]

COLORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _cargar_metricas() -> dict[str, dict[str, float]]:
    """Lee las métricas finales de cada modelo desde ``metricas.json``."""
    metricas: dict[str, dict[str, float]] = {}
    for modelo in MODELOS:
        ruta = FASE_3 / modelo / "metricas.json"
        with ruta.open(encoding="utf-8") as archivo:
            metricas[modelo] = json.load(archivo)
    return metricas


def graficar_metricas(metricas: dict[str, dict[str, float]]) -> Path:
    """Barras agrupadas con las cuatro métricas para los tres modelos."""
    modelos = list(MODELOS)
    posiciones = np.arange(len(modelos))
    ancho = 0.2
    desplazamiento = (len(METRICAS) - 1) / 2

    fig, eje = plt.subplots(figsize=(10, 4.2))

    for i, (clave, etiqueta) in enumerate(METRICAS):
        valores = [metricas[modelo][clave] for modelo in modelos]
        barras = eje.bar(
            posiciones + (i - desplazamiento) * ancho,
            valores,
            width=ancho,
            label=etiqueta,
            color=COLORES[i],
        )
        for barra, valor in zip(barras, valores):
            eje.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 0.02,
                f"{valor:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    eje.set_xticks(posiciones, [MODELOS[modelo] for modelo in modelos])
    eje.set_ylabel("Valor")
    eje.set_ylim(0, 1.05)
    eje.grid(axis="y", alpha=0.3, linewidth=0.6)
    eje.set_axisbelow(True)
    eje.legend(
        ncol=len(METRICAS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        frameon=False,
    )
    eje.set_title("Métricas finales sobre el conjunto de prueba", pad=28)
    fig.tight_layout()

    destino = SALIDA / "resultados_fase3_metricas.png"
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destino


def graficar_confusion(modelo: str) -> Path:
    """Matriz de confusión normalizada de un modelo."""
    ruta_csv = FASE_3 / modelo / "matriz_confusion_normalizada.csv"
    matriz = pd.read_csv(ruta_csv, index_col=0)
    clases = list(matriz.index)
    valores = matriz.to_numpy()

    fig, eje = plt.subplots(figsize=(6.4, 5.4))
    imagen = eje.imshow(valores, cmap="Blues", vmin=0, vmax=1)

    eje.set_xticks(range(len(clases)), clases, rotation=45, ha="right")
    eje.set_yticks(range(len(clases)), clases)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title(f"{MODELOS[modelo]}: matriz de confusión normalizada")

    for i in range(len(clases)):
        for j in range(len(clases)):
            valor = valores[i, j]
            color = "white" if valor > 0.55 else "black"
            eje.text(
                j,
                i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    barra = fig.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)
    barra.set_label("Proporción de la clase real")
    fig.tight_layout()

    destino = SALIDA / f"resultados_fase3_confusion_{modelo}.png"
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destino


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "savefig.dpi": 300,
        }
    )

    # Elimina la figura antigua de paneles por modelo, si existe.
    (SALIDA / "resultados_fase3_metricas_por_modelo.png").unlink(missing_ok=True)

    metricas = _cargar_metricas()
    rutas = [graficar_metricas(metricas)]
    rutas.extend(graficar_confusion(modelo) for modelo in MODELOS)

    for ruta in rutas:
        print(f"Figura guardada en {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
