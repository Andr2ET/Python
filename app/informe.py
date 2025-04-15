# app/informe.py
import os
from fpdf import FPDF

def generar_informe(resultados):
    """Genera un informe PDF con los resultados y lo guarda en la carpeta output."""
    os.makedirs("output", exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Informe de Análisis Predictivo", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    for modelo, metricas in resultados.items():
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Modelo: {modelo}", ln=True)
        for metrica, valor in metricas.items():
            pdf.cell(200, 10, txt=f" - {metrica}: {valor:.3f}", ln=True)
    
    ruta_pdf = os.path.join("output", "informe.pdf")
    pdf.output(ruta_pdf)
    return ruta_pdf
