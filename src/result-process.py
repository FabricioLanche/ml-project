"""Genera las figuras de la sección de resultados a partir de los CSV de fase 2 y fase 3.

Las tablas de fase 1, fase 2 y por clase ya se construyen directamente desde sus
CSV. Este script produce:

1. Un gráfico de barras agrupadas con los tres modelos y, para cada uno, accuracy,
   precisión macro, recall macro y macro F1.
2. Una curva de macro F1 por fold temporal de la fase 2 para la configuración
   ganadora de cada modelo.
3. Un mapa de calor de F1 por clase y modelo sobre el conjunto de prueba.
4. Una matriz de confusión normalizada por modelo.

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
ITERACION = 1  # Cambiar si se evalúa otra iteración de la fase 2.
FASE_2 = RAIZ / "results" / f"iteracion_{ITERACION}" / "fase_2"
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
COLORES_MODELO = {
    "rf": "#4C72B0",
    "logreg": "#DD8452",
    "svm": "#55A868",
}


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


def _cargar_folds_ganadores() -> dict[str, pd.DataFrame]:
    """Carga los folds de fase 2 para la configuración ganadora de cada modelo.

    ``resultado_por_fold.csv`` contiene únicamente las rondas que terminaron
    correctamente; las ejecuciones fallidas y las reanudaciones del SVM no se
    reconstruyen desde los logs. Cada ronda de ``r1`` a ``r5`` es un bloque
    temporal contiguo que se usa como validación, mientras el modelo se entrena
    con los cuatro bloques restantes.
    """
    with (FASE_2 / "ganadores.json").open(encoding="utf-8") as archivo:
        ganadores = json.load(archivo)
    resultados = pd.read_csv(FASE_2 / "resultado_por_fold.csv")

    folds: dict[str, pd.DataFrame] = {}
    for modelo, info in ganadores.items():
        filtro = (
            (resultados["modelo"] == modelo)
            & (resultados["celda"] == info["celda"])
            & (resultados["hiperparametro"] == info["hiperparametro"])
        )
        tabla = resultados.loc[filtro].sort_values("ronda").copy()
        if len(tabla) != 5:
            raise ValueError(
                f"Se esperaban cinco folds para {modelo} y se encontraron "
                f"{len(tabla)}."
            )
        folds[modelo] = tabla
    return folds


def graficar_folds_temporales() -> Path:
    """Curva de macro F1 por bloque temporal de validación.

    Cada punto corresponde a la métrica de la configuración ganadora de la fase 2
    sobre uno de los cinco folds temporales. La figura documenta la dispersión
    entre bloques y hace visible el comportamiento atípico de ``r2``. El eje
    vertical se recorta en 0.3 para resaltar las diferencias entre bloques.
    """
    folds = _cargar_folds_ganadores()
    rondas = list(next(iter(folds.values()))["ronda"])
    posiciones = np.arange(len(rondas))
    desplazamientos = {
        "rf": (0, 8),
        "logreg": (-14, -13),
        "svm": (14, -13),
    }

    fig, eje = plt.subplots(figsize=(6.4, 4.0))
    for modelo in MODELOS:
        tabla = folds[modelo]
        eje.plot(
            posiciones,
            tabla["promedio_F1"],
            marker="o",
            markersize=5,
            linewidth=2,
            label=MODELOS[modelo],
            color=COLORES_MODELO[modelo],
        )
        for x, valor in zip(posiciones, tabla["promedio_F1"]):
            eje.annotate(
                f"{valor:.2f}",
                (x, valor),
                textcoords="offset points",
                xytext=desplazamientos[modelo],
                ha="center",
                fontsize=6.5,
                color=COLORES_MODELO[modelo],
            )

    eje.set_xticks(posiciones, rondas)
    eje.set_xlabel("Bloque temporal de validación")
    eje.set_ylabel("Macro F1")
    eje.set_xlim(-0.3, len(rondas) - 0.7)
    eje.set_ylim(0.3, 1.06)
    eje.grid(axis="y", alpha=0.3, linewidth=0.6)
    eje.set_axisbelow(True)
    eje.legend(
        ncol=len(MODELOS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    eje.set_title("Macro F1 por fold temporal", pad=28)
    fig.tight_layout()

    destino = SALIDA / "resultados_fase2_folds.png"
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destino


def graficar_heatmap_clases() -> Path:
    """Mapa de calor del F1 por clase y modelo sobre el conjunto de prueba.

    Se construye a partir de ``metricas_por_clase.csv`` de la fase 3. Las filas
    conservan el orden de ``clase_id`` y las columnas el orden de ``MODELOS``.
    """
    tablas: dict[str, pd.DataFrame] = {}
    for modelo in MODELOS:
        ruta = FASE_3 / modelo / "metricas_por_clase.csv"
        tablas[modelo] = pd.read_csv(ruta).sort_values("clase_id")

    clases = tablas["rf"]["clase"].tolist()
    valores = np.column_stack(
        [tablas[modelo]["F1"].to_numpy() for modelo in MODELOS]
    )

    fig, eje = plt.subplots(figsize=(6.4, 4.8))
    imagen = eje.imshow(valores, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    eje.set_xticks(range(len(MODELOS)), [MODELOS[modelo] for modelo in MODELOS])
    eje.set_yticks(range(len(clases)), clases)
    eje.set_xlabel("Modelo")
    eje.set_ylabel("Clase")
    eje.set_title("F1 por clase en el conjunto de prueba", pad=10)

    for i in range(len(clases)):
        for j in range(len(MODELOS)):
            valor = valores[i, j]
            color = "white" if valor > 0.6 else "black"
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
    barra.set_label("F1")
    fig.tight_layout()

    destino = SALIDA / "resultados_fase3_heatmap_clases.png"
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
    rutas = [
        graficar_metricas(metricas),
        graficar_folds_temporales(),
        graficar_heatmap_clases(),
    ]
    rutas.extend(graficar_confusion(modelo) for modelo in MODELOS)

    for ruta in rutas:
        print(f"Figura guardada en {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
