"""Configuración de variantes de preprocesamiento que se comparan en la grilla.

Cada celda declara la inclusión de las features ``ct_*`` y de las asimetrías.
Las transformaciones comunes a todas las celdas se implementan en ``pre-process.py``.
"""

# Diseño factorial: presencia/ausencia de cada asimetría, cruzada con conservar/excluir
# ct_*. Así se estiman sus efectos individuales y conjuntos. Se conservan las fuentes.
# Puertos por franja IANA, proto, log1p, escalado e imputación son comunes.
CELDAS = {
    "G00": {"ct": True,  "feature": "base"},
    "G01": {"ct": False, "feature": "base"},
    "G02": {"ct": True,  "feature": "packet_asym"},
    "G03": {"ct": False, "feature": "packet_asym"},
    "G04": {"ct": True,  "feature": "byte_asym"},
    "G05": {"ct": False, "feature": "byte_asym"},
    "G06": {"ct": True,  "feature": "both_asym"},
    "G07": {"ct": False, "feature": "both_asym"},
}

# Receta común: genera todas las columnas derivadas; cada celda descarta las que sobran.
CELDA_COMPLETA = {"ct": True, "feature": "both_asym"}
COLS_ASIMETRIA = {
    "packet_asymmetry": {"packet_asym", "both_asym"},
    "byte_asymmetry": {"byte_asym", "both_asym"},
}
