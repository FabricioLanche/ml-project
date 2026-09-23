# Detección de ataques de red multiclase sobre UNSW-NB15: análisis exploratorio y evaluación de modelos de Machine Learning

Proyecto de Machine Learning (CS3061, UTEC) para la detección de ataques de red multiclase sobre el dataset UNSW-NB15.

## Objetivo

Desarrollar un clasificador de tráfico de red que distinga el tráfico normal (`Benign`) de 10 familias de ataques, mediante análisis exploratorio de datos y la implementación de modelos de Machine Learning.

## Integrantes

- Fabricio Alonso Lanche Pacsi
- Jhogan Haldo Pachacutec Aguilar
- Anyeli Azumi Tamara Ureta
- Paulo Ismael Miranda Barrientos
- Christofer Renato Perez Torres

## Estructura del repositorio

```
ml-project/
├── notebooks/
│   ├── eda.ipynb     # Análisis exploratorio de datos (principal)
│   └── eda.py        # Versión en script (desarrollo y debugging)
├── output/
│   ├── *.csv         # Tablas generadas (descriptivos, faltantes, clases, etc.)
│   └── fig*.png      # Figuras del análisis exploratorio
├── requirements.txt  # Dependencias del entorno
└── dataset/          # Datos UNSW-NB15 (no versionado, ver sección Dataset)
```

## Reproducción

1. Preparar el entorno e instalar las dependencias (`pandas`, `numpy`, `matplotlib` y `kagglehub` con Python 3.12):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ejecutar el análisis exploratorio (genera las tablas CSV y figuras en `output/`):

   - **Notebook**: abrir `notebooks/eda.ipynb` en Jupyter y ejecutar todas las celdas.
   - **Script**: equivalente, usado durante el desarrollo y debugging.

     ```bash
     python notebooks/eda.py
     ```

## Dataset

UNSW-NB15 (Moustafa y Slay, 2015): tráfico de red capturado en el Cyber Range Lab de la ACCS. Consta de 4 archivos CSV con ~2.5 millones de registros y 49 columnas.

La primera ejecución del análisis exploratorio descarga el dataset desde Kaggle (`harshwardhanbhangale/unsw-complete-dataset`) a la carpeta `dataset/`. Esta carpeta está excluida del repositorio (`.gitignore`).