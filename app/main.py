# app/main.py
import streamlit as st
import pandas as pd
import os

# Importar módulos creados
from . import upload, preprocesamiento, modelos, evaluacion, informe

# Configuración de la página
st.set_page_config(page_title="Plataforma de Análisis Predictivo", layout="wide")

# Encabezado y logo
st.image("assets/logo.png", width=150)
st.title("Plataforma de Análisis Predictivo")
st.markdown("Sube tu dataset y selecciona las opciones para ejecutar un análisis predictivo.")

# Sección 1: Subida del dataset
uploaded_file = st.file_uploader("📁 Sube tu archivo CSV", type=["csv", "xlsx"])
if uploaded_file:
    df = upload.cargar_dataset(uploaded_file)
    st.success("✅ Dataset cargado correctamente!")
    st.dataframe(df.head())
    
    # Guardar una copia en la carpeta data (opcional)
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    df.to_csv(file_path, index=False)

    # Sección 2: Selección de columnas y opciones
    st.sidebar.header("Configuración del Análisis")
    target_col = st.sidebar.selectbox("Selecciona la columna objetivo", df.columns)
    
    feature_cols = st.sidebar.multiselect("Selecciona las columnas de características", [col for col in df.columns if col != target_col])
    
    st.sidebar.subheader("Selección de Algoritmos")
    modelos_disponibles = ["Logistic Regression", "Random Forest", "SVM"]
    seleccion_modelos = st.sidebar.multiselect("Elige el/los modelo(s)", modelos_disponibles)
    
    st.sidebar.subheader("Selección de Métricas")
    metricas_disponibles = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    seleccion_metricas = st.sidebar.multiselect("Elige la(s) métrica(s)", metricas_disponibles)
    
    # Botón para ejecutar el análisis
    if st.button("Ejecutar Análisis"):
        # Preprocesar los datos (limpieza, división X e y)
        X, y = preprocesamiento.preprocesar_datos(df, target=target_col, features=feature_cols)
        
        # Entrenar los modelos y obtener predicciones y métricas
        resultados = modelos.ejecutar_modelos(X, y, seleccion_modelos, seleccion_metricas)
        
        # Evaluar y mostrar resultados
        st.header("Resultados del Análisis")
        evaluacion.mostrar_resultados(resultados)
        
        # Generar PDF del informe
        if st.button("Generar Informe PDF"):
            ruta_pdf = informe.generar_informe(resultados)
            with open(ruta_pdf, "rb") as pdf_file:
                st.download_button("Descargar Informe PDF", pdf_file, file_name="informe.pdf")
