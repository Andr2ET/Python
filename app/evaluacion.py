# app/evaluacion.py
import streamlit as st
import pandas as pd

def mostrar_resultados(resultados):
    """Muestra los resultados en forma de tabla y gráficas (ejemplo básico)."""
    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados).T
    st.dataframe(df_resultados)
    
    # Puedes agregar aquí gráficas con matplotlib o plotly para cada métrica
    # Ejemplo (sin gráficos avanzados):
    st.markdown("### Métricas por Modelo")
    for modelo, metricas in resultados.items():
        st.write(f"**{modelo}:**")
        for metrica, valor in metricas.items():
            st.write(f"- {metrica}: {valor:.3f}")
