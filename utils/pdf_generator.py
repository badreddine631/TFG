from fpdf import FPDF
import pandas as pd

def generar_informe(eventos_df, output_path="informe_analisis.pdf"):
    """Genera un informe en PDF con el análisis de eventos macroeconómicos y volatilidad."""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título del informe
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Informe de Análisis del Oro", ln=True, align="C")
    pdf.ln(10)

    # Sección 1: Resumen de eventos macroeconómicos
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="1. Impacto de Eventos Macroeconómicos", ln=True)
    pdf.ln(5)

    # Añadir tabla de impacto de eventos
    pdf.set_font("Arial", size=12)
    for _, row in eventos_df.iterrows():
        evento = row['Evento']
        fecha = row['Fecha'].strftime('%Y-%m-%d') if isinstance(row['Fecha'], pd.Timestamp) else str(row['Fecha'])
        categoria = row.get('Categoría', 'Desconocida')  # Manejar si la columna no existe
        cambio = f"{row.get('Cambio (%)', 'N/A'):.2f}%" if 'Cambio (%)' in row else "N/A"
        pdf.cell(0, 10, txt=f"{evento} ({fecha}) - Categoría: {categoria}, Cambio: {cambio}", ln=True)

    pdf.ln(10)

    # Sección 2: Conclusión
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="2. Conclusión", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Este informe proporciona una visión general del impacto de eventos macroeconómicos en el precio del oro y su volatilidad. La información contenida puede servir de base para la toma de decisiones estratégicas en inversión y gestión de riesgos.")

    # Guardar el archivo PDF
    pdf.output(output_path)

    return output_path
