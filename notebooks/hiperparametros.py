"""Definición única de la base y la grilla de hiperparámetros.

Fuentes usadas:
- Kasongo y Sun (2020): LR ``max_iter=1000``; SVM RBF ``C=1.12``.
- Ahmad et al. (2021): RF ``n_estimators=100``; ``LinearSVC`` ``C=1e10``;
  ``tol=1e-12``.
- Dadavali (2026): RF ``n_estimators=100``, ``random_state=42``.
- Hussein et al. (2021): RF ``max_depth=18``.
- Salmon et al. (2026): RF ``n_estimators=500``, ``max_depth=30``,
  ``random_state=42``.
- Fatima y Ali (2022): RF ``n_estimators=100``; ``max_depth=80``.
- Özbek y Soyak (2025): RF ``n_estimators=200``; profundidad sin límite.

Los valores sin fuente se marcan como convención de Spark/sklearn o decisión
del pipeline. ``regParam`` de Spark no equivale a ``C`` de sklearn: la relación
aproximada es ``regParam ~= 1/(C*n)``; con ``n ~= 2.16M``, ``C=1`` daría
``~4.6e-7``. No es una traducción exacta.
"""

import json
from itertools import product

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
    },
}


def _producto(base: dict, **ejes: list) -> list[dict]:
    nombres = list(ejes)
    return [
        {**base, **dict(zip(nombres, valores))}
        for valores in product(*(ejes[nombre] for nombre in nombres))
    ]


GRILLA: dict[str, list[dict]] = {
    "logreg": _producto(
        BASE["logreg"],
        # Sin C reportado; rejilla logarítmica experimental.
        regParam=[0.0, 1e-6, 1e-4, 1e-2],
        # Sin penalty reportado; 0.5 solo como exploración.
        elasticNetParam=[0.0, 0.5],
    ),
    "rf": _producto(
        BASE["rf"],
        # Ahmad (2021) y Dadavali (2026): 100; Özbek (2025): 200.
        # Salmon (2026) usa 500, pero se excluye por coste y baja confianza.
        numTrees=[100, 200],
        # Hussein (2021): 18. Salmon (2026) usa 30, pero Spark limita a 30
        # y resulta caro; 10 y 25 son vecinos de la base.
        maxDepth=[10, 18, 25],
        # Sin respaldo en papers; 1 es default de Spark, 10 regulariza.
        minInstancesPerNode=[1, 10],
        # Sin respaldo en papers; 32 es default de Spark, 64 refina.
        maxBins=[32, 64],
    ),
    "svm": _producto(
        BASE["svm"],
        # Sin respaldo en papers; rango operativo.
        maxIter=[100, 500],
        # Sin C reportado; anclado en C=1e10, que equivale a regParam=0.
        regParam=[0.0, 1e-6, 1e-4, 1e-2],
    ),
}

def contenido() -> dict:
    """Devuelve la estructura que se persiste en ``hiperparametros.json``."""
    return {"base": BASE, "grilla": GRILLA}


if __name__ == "__main__":
    print(json.dumps(contenido(), indent=2, ensure_ascii=False))
