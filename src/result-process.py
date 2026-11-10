"""Genera las figuras de la sección de resultados a partir de los CSV de fase 2 y fase 3.

Las tablas de fase 1, fase 2 y por clase ya se construyen directamente desde sus
CSV. Este script produce:

1. Un gráfico de barras agrupadas con los tres modelos y, para cada uno, accuracy,
   precisión macro, recall macro y macro F1.
2. Una curva de macro F1 por fold temporal de la fase 2 para la configuración
   ganadora de cada modelo.
3. Un mapa de calor de F1 por clase y modelo sobre el conjunto de prueba.
4. Una matriz de confusión normalizada por modelo.
5. Una tabla con los parámetros aprendidos de los modelos guardados.
6. Un CSV con la validación por fold y otro con la comparación entre validación
   y prueba de la configuración ganadora de cada modelo.

Salida: ``results/figs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
PALETA_F1 = LinearSegmentedColormap.from_list(
    "blanco_amarillo_naranja_rojo",
    ["#ffffff", "#ffffb2", "#fd8d3c", "#d73027", "#67000d"],
)


def _cargar_metricas() -> dict[str, dict[str, float]]:
    """Lee las métricas finales de cada modelo desde ``metricas.json``."""
    metricas: dict[str, dict[str, float]] = {}
    for modelo in MODELOS:
        ruta = FASE_3 / modelo / "metricas.json"
        with ruta.open(encoding="utf-8") as archivo:
            metricas[modelo] = json.load(archivo)
    return metricas


def _leer_metadata_etapa(ruta: Path) -> dict:
    """Lee el JSON de metadatos de una etapa de un PipelineModel de Spark."""
    archivos = sorted(ruta.glob("part-*.txt"))
    if not archivos:
        raise FileNotFoundError(f"No hay metadatos en {ruta}")
    with archivos[0].open(encoding="utf-8") as archivo:
        return json.load(archivo)


def _buscar_etapa(etapas: Path, fragmento_clase: str) -> tuple[Path, dict]:
    """Devuelve la carpeta y los metadatos de la primera etapa que coincida."""
    for etapa in sorted(etapas.iterdir()):
        metadata = etapa / "metadata"
        if not metadata.is_dir():
            continue
        datos = _leer_metadata_etapa(metadata)
        if fragmento_clase in datos.get("class", ""):
            return etapa, datos
    raise ValueError(f"No se encontró una etapa {fragmento_clase} en {etapas}")


def _variables_de_entrada(etapas: Path) -> int:
    """Cuenta las variables originales que entran al VectorAssembler."""
    _, datos = _buscar_etapa(etapas, "VectorAssembler")
    return len(datos["paramMap"]["inputCols"])


def _mapa_clases() -> dict[int, str]:
    """Devuelve el nombre de cada clase a partir de su identificador."""
    tabla = pd.read_csv(FASE_3 / "rf" / "metricas_por_clase.csv")
    return dict(zip(tabla["clase_id"], tabla["clase"]))


def _resumen_arboles(arboles: pd.DataFrame) -> pd.DataFrame:
    """Cuenta nodos, hojas y profundidad de cada árbol del Random Forest."""
    filas: list[dict[str, int]] = []
    for tree_id, grupo in arboles.groupby("treeID", sort=True):
        nodos = grupo["nodeData"].tolist()
        hijos = {
            int(nodo["id"]): (int(nodo["leftChild"]), int(nodo["rightChild"]))
            for nodo in nodos
        }
        pila = [(0, 0)]
        profundidad_maxima = 0
        while pila:
            nodo, profundidad = pila.pop()
            profundidad_maxima = max(profundidad_maxima, profundidad)
            izquierdo, derecho = hijos.get(nodo, (-1, -1))
            if izquierdo != -1:
                pila.append((izquierdo, profundidad + 1))
            if derecho != -1:
                pila.append((derecho, profundidad + 1))
        filas.append(
            {
                "treeID": int(tree_id),
                "nodos": len(nodos),
                "hojas": sum(1 for nodo in nodos if nodo["leftChild"] == -1),
                "profundidad": profundidad_maxima,
            }
        )
    return pd.DataFrame(filas)


def extraer_parametros_finales() -> pd.DataFrame:
    """Extrae los parámetros aprendidos de los modelos guardados por PySpark.

    Para regresión logística y SVM se recuperan los coeficientes e interceptos
    desde los archivos ``data`` del modelo. Para Random Forest se resumen las
    estructuras de los 100 árboles: nodos, hojas y profundidad. Los coeficientes
    individuales se guardan en CSV para que puedan auditarse fuera del informe.
    """
    clases = _mapa_clases()
    filas: list[dict[str, object]] = []

    etapas_rf = FASE_3 / "rf" / "modelo" / "stages"
    etapa_rf, _ = _buscar_etapa(etapas_rf, "RandomForestClassificationModel")
    arboles = pd.read_parquet(next((etapa_rf / "data").glob("part-*.parquet")))
    resumen_arboles = _resumen_arboles(arboles)
    resumen_arboles.to_csv(
        SALIDA / "parametros_aprendidos_rf_arboles.csv", index=False, encoding="utf-8"
    )
    filas.append(
        {
            "modelo": MODELOS["rf"],
            "parametros": "Estructura de árboles",
            "cantidad": (
                f"{len(resumen_arboles)} árboles; {len(arboles):,} nodos"
            ),
            "detalle": (
                f"{int(resumen_arboles['hojas'].sum()):,} hojas; "
                f"profundidad máxima {int(resumen_arboles['profundidad'].max())}; "
                f"media {resumen_arboles['profundidad'].mean():.1f}"
            ),
        }
    )

    etapas_lr = FASE_3 / "logreg" / "modelo" / "stages"
    etapa_lr, _ = _buscar_etapa(etapas_lr, "LogisticRegressionModel")
    tabla_lr = pd.read_parquet(next((etapa_lr / "data").glob("part-*.parquet")))
    matriz_cruda = tabla_lr["coefficientMatrix"].iloc[0]
    interceptos_lr = np.asarray(
        tabla_lr["interceptVector"].iloc[0]["values"], dtype=float
    )
    filas_lr = int(matriz_cruda["numRows"])
    columnas_lr = int(matriz_cruda["numCols"])
    valores_lr = np.asarray(matriz_cruda["values"], dtype=float)
    if matriz_cruda["isTransposed"]:
        coeficientes_lr = valores_lr.reshape(filas_lr, columnas_lr)
    else:
        coeficientes_lr = valores_lr.reshape(columnas_lr, filas_lr).T

    detalle_lr: list[dict[str, object]] = []
    for clase_id in range(filas_lr):
        for feature in range(columnas_lr):
            detalle_lr.append(
                {
                    "clase_id": clase_id,
                    "clase": clases.get(clase_id, str(clase_id)),
                    "feature": f"f{feature:02d}",
                    "coeficiente": float(coeficientes_lr[clase_id, feature]),
                    "intercepto": float(interceptos_lr[clase_id]),
                }
            )
    pd.DataFrame(detalle_lr).to_csv(
        SALIDA / "parametros_aprendidos_logreg.csv", index=False, encoding="utf-8"
    )
    filas.append(
        {
            "modelo": MODELOS["logreg"],
            "parametros": "Coeficientes + interceptos",
            "cantidad": (
                f"{coeficientes_lr.size:,} coeficientes + {len(interceptos_lr)} interceptos"
            ),
            "detalle": (
                f"matriz {filas_lr}x{columnas_lr}; "
                f"rango [{coeficientes_lr.min():.2f}, {coeficientes_lr.max():.2f}]; "
                f"L2 {np.linalg.norm(coeficientes_lr):.2f}; "
                f"interceptos [{interceptos_lr.min():.2f}, {interceptos_lr.max():.2f}]"
            ),
        }
    )

    etapas_svm = FASE_3 / "svm" / "modelo" / "stages"
    etapa_svm, _ = _buscar_etapa(etapas_svm, "OneVsRestModel")
    detalle_svm: list[dict[str, object]] = []
    coeficientes_svm: list[np.ndarray] = []
    interceptos_svm: list[float] = []
    for clase_id, carpeta in enumerate(sorted(etapa_svm.glob("model_*"))):
        tabla_svm = pd.read_parquet(next((carpeta / "data").glob("part-*.parquet")))
        coeficientes = np.asarray(
            tabla_svm["coefficients"].iloc[0]["values"], dtype=float
        )
        intercepto = float(tabla_svm["intercept"].iloc[0])
        coeficientes_svm.append(coeficientes)
        interceptos_svm.append(intercepto)
        for feature, valor in enumerate(coeficientes):
            detalle_svm.append(
                {
                    "clase_id": clase_id,
                    "clase": clases.get(clase_id, str(clase_id)),
                    "feature": f"f{feature:02d}",
                    "coeficiente": float(valor),
                    "intercepto": intercepto,
                }
            )
    pd.DataFrame(detalle_svm).to_csv(
        SALIDA / "parametros_aprendidos_svm.csv", index=False, encoding="utf-8"
    )
    coeficientes_svm = np.concatenate(coeficientes_svm)
    interceptos_svm = np.asarray(interceptos_svm, dtype=float)
    filas.append(
        {
            "modelo": MODELOS["svm"],
            "parametros": "Coeficientes + interceptos",
            "cantidad": (
                f"{coeficientes_svm.size:,} coeficientes + {len(interceptos_svm)} interceptos"
            ),
            "detalle": (
                f"{len(interceptos_svm)} clasificadores de {len(coeficientes_svm)//len(interceptos_svm)}; "
                f"{np.count_nonzero(coeficientes_svm):,} no nulos; "
                f"rango [{coeficientes_svm.min():.2f}, {coeficientes_svm.max():.2f}]; "
                f"L2 {np.linalg.norm(coeficientes_svm):.2f}; "
                f"interceptos [{interceptos_svm.min():.2f}, {interceptos_svm.max():.2f}]"
            ),
        }
    )

    return pd.DataFrame(filas)


def exportar_validacion_prueba() -> tuple[Path, Path]:
    """Exporta la validación por fold y su comparación con la prueba final.

    El resumen usa la media y la desviación estándar muestral (``ddof=1``) de
    los cinco folds de la configuración ganadora de la fase 2. Las diferencias
    se calculan con los valores sin redondear como validación menos prueba.
    """
    folds = _cargar_folds_ganadores()
    metricas = _cargar_metricas()
    nombres = {
        "accuracy": "accuracy",
        "promedio_precision": "precision_macro",
        "promedio_recall": "recall_macro",
        "promedio_F1": "macro_F1",
    }

    detalle: list[dict[str, object]] = []
    resumen: list[dict[str, object]] = []
    for modelo in MODELOS:
        tabla = folds[modelo]
        for _, fila in tabla.iterrows():
            registro = {
                "modelo": MODELOS[modelo],
                "celda": fila["celda"],
                "ronda": fila["ronda"],
            }
            for origen, destino in nombres.items():
                registro[destino] = float(fila[origen])
            detalle.append(registro)

        fila_resumen: dict[str, object] = {
            "modelo": MODELOS[modelo],
            "celda": tabla["celda"].iloc[0],
            "n_folds": len(tabla),
        }
        for origen, destino in nombres.items():
            media = float(tabla[origen].mean())
            desviacion = float(tabla[origen].std(ddof=1))
            prueba = float(metricas[modelo][origen])
            fila_resumen[f"validacion_{destino}_media"] = media
            fila_resumen[f"validacion_{destino}_desv"] = desviacion
            fila_resumen[f"prueba_{destino}"] = prueba
            fila_resumen[f"diferencia_{destino}"] = media - prueba
        resumen.append(fila_resumen)

    ruta_detalle = SALIDA / "validacion_por_fold.csv"
    ruta_resumen = SALIDA / "validacion_vs_prueba.csv"
    pd.DataFrame(detalle).to_csv(ruta_detalle, index=False, encoding="utf-8")
    pd.DataFrame(resumen).to_csv(ruta_resumen, index=False, encoding="utf-8")
    return ruta_detalle, ruta_resumen


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
    conservan el orden de ``clase_id``, las columnas el orden de ``MODELOS`` y
    cada etiqueta de fila incluye el soporte de la clase.
    """
    tablas: dict[str, pd.DataFrame] = {}
    for modelo in MODELOS:
        ruta = FASE_3 / modelo / "metricas_por_clase.csv"
        tablas[modelo] = pd.read_csv(ruta).sort_values("clase_id")

    clases = tablas["rf"]["clase"].tolist()
    soportes = dict(zip(tablas["rf"]["clase"], tablas["rf"]["support"]))
    valores = np.column_stack(
        [tablas[modelo]["F1"].to_numpy() for modelo in MODELOS]
    )

    fig, eje = plt.subplots(figsize=(7.2, 5.6))
    imagen = eje.imshow(valores, cmap=PALETA_F1, vmin=0, vmax=1, aspect="auto")

    eje.set_xticks(range(len(MODELOS)), [MODELOS[modelo] for modelo in MODELOS])
    eje.set_yticks(
        range(len(clases)),
        [f"{clase} ({soportes[clase]:,})" for clase in clases],
    )
    eje.set_xlabel("Modelo")
    eje.set_ylabel("Clase real (soporte)")
    eje.set_title("F1 por clase y modelo", pad=10)

    eje.set_xticks(np.arange(-0.5, len(MODELOS), 1), minor=True)
    eje.set_yticks(np.arange(-0.5, len(clases), 1), minor=True)
    eje.grid(which="minor", color="white", linewidth=1.0)
    eje.tick_params(which="minor", length=0)

    for i in range(len(clases)):
        mejor = int(np.argmax(valores[i]))
        for j in range(len(MODELOS)):
            valor = valores[i, j]
            rojo, verde, azul, _ = imagen.cmap(imagen.norm(valor))
            luminancia = 0.299 * rojo + 0.587 * verde + 0.114 * azul
            color = "black" if luminancia > 0.55 else "white"
            eje.text(
                j,
                i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                fontweight="bold" if j == mejor else "normal",
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

    (SALIDA / "parametros_finales.csv").unlink(missing_ok=True)
    parametros = extraer_parametros_finales()
    ruta_parametros = SALIDA / "parametros_aprendidos.csv"
    parametros.to_csv(ruta_parametros, index=False, encoding="utf-8")
    print(f"Tabla guardada en {ruta_parametros.relative_to(RAIZ)}")
    print(parametros.to_string(index=False))

    ruta_folds, ruta_resumen = exportar_validacion_prueba()
    print(f"Detalle por fold guardado en {ruta_folds.relative_to(RAIZ)}")
    print(f"Resumen validación-prueba guardado en {ruta_resumen.relative_to(RAIZ)}")

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
