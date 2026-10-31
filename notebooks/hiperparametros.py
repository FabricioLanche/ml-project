"""Definición única de la base y la grilla de hiperparámetros.

Cómo se usó la literatura
-------------------------
Las fuentes anclan el punto BASE, no los límites de la grilla. Solo se trasladan con
cautela: varias son binarias o de 9 clases, usan otro preprocesamiento y otro balance
de clases, y los parámetros de sklearn no equivalen a los de MLlib.

- Kasongo y Sun (2020): LR ``max_iter=1000``; SVM RBF ``C=1.12`` (no trasladable).
- Ahmad et al. (2021): RF ``n_estimators=100``; ``LinearSVC`` ``C=1e10``;
  ``tol=1e-12`` (extremo, baja confianza).
- Dadavali (2026): RF ``n_estimators=100``, ``random_state=42`` y, según el resumen de
  hiperparámetros del proyecto, ``class_weight='balanced_subsample'``.
- Hussein et al. (2021): RF ``max_depth=18`` (solo accuracy; baja confianza).
- Salmon et al. (2026): RF ``n_estimators=500``, ``max_depth=30`` (resultados
  inusualmente altos y código binario; baja confianza).
- Fatima y Ali (2022): RF ``n_estimators=100``; ``max_depth=80`` no es utilizable
  (Spark limita la profundidad a 30).
- Özbek y Soyak (2025): RF ``n_estimators=200``; profundidad sin límite.

Criterio para elegir qué se explora
-----------------------------------
Un eje entra en la grilla solo si (1) puede mover el macro-F1 de las clases raras, que
es la métrica de selección; (2) es razonable esperar un óptimo interior y no un "más es
mejor" casi seguro; y (3) sus valores son distinguibles entre sí con ~150k filas de
ajuste (tras el tope de 100k por clase).

Ejes que entran:
- ``pesoClases``: es el eje de mayor relevancia esperada. Tras el tope, Benign sigue
  teniendo 100k filas frente a ~100 de Worms; ponderar por clase cambia directamente el
  equilibrio entre clases, que es lo que mide el macro-F1. Además hace que el modelo
  final (entrenado sin tope) vea el mismo equilibrio efectivo que en la validación.
  ``balanceado`` (peso ∝ 1/frecuencia) tiene respaldo en Dadavali; ``raiz`` (peso ∝
  1/√frecuencia) es el punto intermedio. Requiere ``pesos`` en el pipeline (ver
  ``INCLUIR_PESOS``).
- RF ``maxDepth``: controla la capacidad. Con ~150k filas un árbol equilibrado llega a
  ~17 niveles, y solo las ramas desbalanceadas (las de las clases raras) usan más.
- RF ``featureSubsetStrategy``: ``sqrt`` frente a ``onethird`` fija el compromiso entre
  la fuerza de cada árbol y la correlación entre árboles. Con ~50 columnas, ``sqrt``
  ve ~7 por nodo y ``onethird`` ~17.
- LR y SVM ``regParam``: único control de regularización. Con Spark,
  ``regParam ≈ 1/(C·n)``; con n ≈ 150k (el n efectivo de ajuste, no el del train
  completo), ``C=1`` equivale a ~6.5e-6. El efecto empieza a notarse hacia 1e-4
  (C≈0.065) y es fuerte en 1e-2 (C≈6.5e-4), así que la grilla cubre esa transición.
  Para SVM, además, un ``regParam`` positivo suele estabilizar el optimizador.

Ejes que se dejan fijos, y por qué:
- ``numTrees``: más allá de ~100 árboles la mejora es marginal y el coste se duplica;
  100 es además el consenso de la literatura.
- ``maxBins``: las features continuas ya pasaron por ``log1p`` y 32 bins por cuantiles
  son suficientes; sin hipótesis de efecto.
- ``minInstancesPerNode``: valores mayores a 1 tienden a perjudicar justo a las clases
  con pocos ejemplos, y el conjunto de 100 árboles ya regulariza.
- ``elasticNetParam``: L1 no aporta con ~50 columnas y obliga a usar OWLQN, más lento.
- ``maxIter``: más iteraciones solo ayudan a converger; explorarlo mide el optimizador,
  no el modelo. Con ``regParam=0`` es probable que se agote el tope.
- Valores casi equivalentes entre sí (p. ej. ``regParam=1e-6``, que con n≈150k es
  indistinguible de 0).

Ablaciones de features (``stcpb``/``dtcpb``, puertos de destino individuales) no son
hiperparámetros: pertenecen a la rejilla de preprocesamiento.
"""

import json
from itertools import product

#: ``True`` añade el eje ``pesoClases`` (ver docstring). Requiere que el pipeline cree
#: las columnas de pesos (``common.agregar_pesos``) y que ``crear_clasificador`` traduzca
#: ``pesoClases`` a ``weightCol``. Con ``False`` la grilla no usa pesos y no hace falta
#: tocar ``common.py`` ni ``train.py``.
INCLUIR_PESOS = True
ESQUEMAS_PESO = ("ninguno", "raiz", "balanceado")

_PESO_BASE = {"pesoClases": "ninguno"} if INCLUIR_PESOS else {}

BASE: dict[str, dict] = {
    "logreg": {
        # Kasongo y Sun (2020) usan 1000; 500 es un tope por coste.
        "maxIter": 500,
        # Los papers no reportan C; base sin regularización.
        "regParam": 0.0,
        # Los papers no reportan penalty; L2 por convención.
        "elasticNetParam": 0.0,
        # Default de Spark; los papers no lo reportan.
        "tol": 1e-6,
        # Convención del pipeline: ya hay StandardScaler externo.
        "standardization": False,
        # Convención: 10 clases implican softmax multinomial.
        "family": "multinomial",
        **_PESO_BASE,
    },
    "rf": {
        # Ahmad (2021), Dadavali (2026), Hussein (2021), Fatima (2022): 100.
        "numTrees": 100,
        # Hussein (2021): 18 (confianza baja); 30 se evita por coste.
        "maxDepth": 18,
        # Dadavali (2026) y Salmon (2026): 42.
        "seed": 42,
        # Default de Spark; sin respaldo en papers.
        "minInstancesPerNode": 1,
        # Default de Spark; sin respaldo en papers.
        "maxBins": 32,
        # Default de Spark/sklearn; sin respaldo en papers.
        "impurity": "gini",
        # Convención: equivale a max_features='sqrt' de sklearn.
        "featureSubsetStrategy": "sqrt",
        **_PESO_BASE,
    },
    "svm": {
        # Sin respaldo en papers; 500 es un tope operativo.
        "maxIter": 500,
        # Ahmad (2021): C=1e10 en LinearSVC equivale a no regularizar.
        # C=1.12 de Kasongo (2020) es RBF y no se traslada a LinearSVC.
        "regParam": 0.0,
        # Ahmad (2021) usa 1e-12 (extremo, confianza baja);
        # 1e-6 es el default de Spark.
        "tol": 1e-6,
        # Convención del pipeline: ya hay StandardScaler externo.
        "standardization": False,
        **_PESO_BASE,
    },
}


def _producto(base: dict, **ejes: list) -> list[dict]:
    nombres = list(ejes)
    return [
        {**base, **dict(zip(nombres, valores))}
        for valores in product(*(ejes[nombre] for nombre in nombres))
    ]


def _eje_peso(*esquemas: str) -> dict:
    """Eje ``pesoClases`` (vacío si ``INCLUIR_PESOS`` es False)."""
    return {"pesoClases": list(esquemas)} if INCLUIR_PESOS else {}


GRILLA: dict[str, list[dict]] = {
    # 3 regParam × 3 pesos = 9 configuraciones (8 nuevas sobre la base).
    "logreg": _producto(
        BASE["logreg"],
        # 0 = sin regularización; 1e-4 ≈ C 0.065; 1e-2 ≈ C 6.5e-4 (n ≈ 150k).
        regParam=[0.0, 1e-4, 1e-2],
        **_eje_peso("ninguno", "raiz", "balanceado"),
    ),
    # 3 profundidades × 2 estrategias × 2 pesos = 12 configuraciones (11 nuevas).
    "rf": _producto(
        BASE["rf"],
        # 18 es la base; 12 y 25 la acotan por ambos lados (Spark permite hasta 30).
        maxDepth=[12, 18, 25],
        featureSubsetStrategy=["sqrt", "onethird"],
        # Solo "balanceado" (respaldo de Dadavali); RF es poco sensible a la escala.
        **_eje_peso("ninguno", "balanceado"),
    ),
    # 3 regParam × 3 pesos = 9 configuraciones (8 nuevas sobre la base).
    "svm": _producto(
        BASE["svm"],
        # 0 equivale a C=1e10 (Ahmad); los positivos estabilizan el optimizador.
        regParam=[0.0, 1e-4, 1e-2],
        **_eje_peso("ninguno", "raiz", "balanceado"),
    ),
}


def contenido() -> dict:
    """Devuelve la estructura que se persiste en ``hiperparametros.json``."""
    return {"base": BASE, "grilla": GRILLA}


if __name__ == "__main__":
    print(json.dumps(contenido(), indent=2, ensure_ascii=False))