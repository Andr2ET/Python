# app/preprocesamiento.py
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocesar_datos(df, target, features):
    """
    Realiza el preprocesamiento del dataset:
    - Selecciona la variable objetivo y las características.
    - Remueve nulos (ejemplo simple).
    """
    # Opcional: eliminar filas con valores faltantes
    df = df.dropna(subset=[target] + features)
    X = df[features]
    y = df[target]
    return X, y
