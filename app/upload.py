# app/upload.py
import pandas as pd

def cargar_dataset(uploaded_file):
    """Carga el dataset desde un archivo subido (CSV o XLSX)."""
    # Identificar extensión del archivo
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato de archivo no soportado.")
    return df
