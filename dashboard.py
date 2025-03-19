import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import os

# Cargar datos
df = pd.read_csv("C:/Users/Usuario/Desktop/TFFFG/price-gold/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
eventos_df = pd.read_csv("C:/Users/Usuario/Desktop/TFFFG/price-gold/eventos_macroeconomicos.csv", parse_dates=["Fecha"])

# Renombrar y configurar índices
df = df.rename(columns={"Date": "Fecha"})
df.set_index("Fecha", inplace=True)

# Configuración de la app
st.set_page_config(
    page_title="Análisis de Impacto en el Oro",
    page_icon=":bar_chart:",
    layout="wide"
)

# Función para cargar CSS
def load_css(css_file):
    with open(css_file, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Cargar el archivo CSS
load_css("style.css")

# Mostrar el encabezado con título y botón
st.markdown("""
    <div class="header">
        <h1>MacroGold Analytics</h1>
        <button class="get-started-btn">Get Started</button>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="center-content">
        <h2>Información basada en datos para los mercados financieros</h2>
        <p>Aprovecha los datos macroeconómicos y el análisis impulsado por IA para tomar decisiones de inversión más inteligentes.</p>
        <button class="explore-btn">Explorar Nuestras Soluciones</button>
    </div>
""", unsafe_allow_html=True)



# **Pestañas principales**
tab_precio, tab_impacto, tab_volatilidad, tab_informe = st.tabs([
    "📊 Precio del Oro",
    "🌍 Impacto de Eventos",
    "📉 Volatilidad",
    "📂 Generar Informe"
])

# **1. Tab: Precio del Oro**
with tab_precio:
    st.sidebar.header("🔍 Filtrar por Fechas")
    start_date = st.sidebar.date_input("Fecha de Inicio", value=df.index.min(), key="dashboard_fecha_inicio")
    end_date = st.sidebar.date_input("Fecha de Fin", value=df.index.max(), key="dashboard_fecha_fin")
    df_filtrado = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    st.subheader("Evolución del Precio del Oro")
    fig_precio = px.line(df_filtrado, x=df_filtrado.index, y="Close", title="Precio del Oro")
    fig_precio.update_layout(xaxis_title="Fecha", yaxis_title="Precio de Cierre (USD)")
    st.plotly_chart(fig_precio)

# **2. Tab: Impacto de Eventos**
with tab_impacto:
    impacto_eventos = []
    for _, evento in eventos_df.iterrows():
        fecha_evento = evento['Fecha']
        rango_antes = df[(df.index >= fecha_evento - pd.Timedelta(days=7)) & (df.index < fecha_evento)]
        rango_despues = df[(df.index > fecha_evento) & (df.index <= fecha_evento + pd.Timedelta(days=7))]
        promedio_antes = rango_antes['Close'].mean()
        promedio_despues = rango_despues['Close'].mean()
        cambio = ((promedio_despues - promedio_antes) / promedio_antes) * 100
        impacto_eventos.append({
            'Evento': evento['Evento'],
            'Fecha': fecha_evento,
            'Categoría': evento['Categoría'],
            'Cambio (%)': cambio
        })
    impacto_eventos_df = pd.DataFrame(impacto_eventos)

    st.subheader("Impacto de Eventos Macroeconómicos")
    fig_eventos = px.bar(impacto_eventos_df, x="Evento", y="Cambio (%)", color="Categoría",
                         title="Impacto de Eventos", text="Cambio (%)")
    fig_eventos.update_layout(xaxis_title="Evento", yaxis_title="Cambio Promedio (%)")
    st.plotly_chart(fig_eventos)
    st.dataframe(impacto_eventos_df)

# **3. Tab: Volatilidad**
with tab_volatilidad:
    volatilidad_eventos = []
    for _, evento in eventos_df.iterrows():
        fecha_evento = evento['Fecha']
        rango_antes = df[(df.index >= fecha_evento - pd.Timedelta(days=7)) & (df.index < fecha_evento)]
        rango_despues = df[(df.index > fecha_evento) & (df.index <= fecha_evento + pd.Timedelta(days=7))]
        volatilidad_antes = rango_antes['Close'].pct_change().std() * 100
        volatilidad_despues = rango_despues['Close'].pct_change().std() * 100
        volatilidad_eventos.append({
            'Evento': evento['Evento'],
            'Fecha': evento['Fecha'],
            'Volatilidad Antes (%)': volatilidad_antes,
            'Volatilidad Después (%)': volatilidad_despues,
            'Categoría': evento['Categoría']
        })
    volatilidad_df = pd.DataFrame(volatilidad_eventos)

    st.subheader("Volatilidad Antes y Después de los Eventos")
    fig_volatilidad = px.bar(volatilidad_df, 
                             x='Evento', 
                             y=['Volatilidad Antes (%)', 'Volatilidad Después (%)'],
                             barmode='group',
                             title="Volatilidad de Eventos")
    st.plotly_chart(fig_volatilidad)
    st.dataframe(volatilidad_df)

# **Función para generar el informe en PDF**
def generar_informe(impacto_eventos_df, volatilidad_df, output_path="informe_analisis.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título del informe
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Informe de Análisis del Oro", ln=True, align="C")
    pdf.ln(10)

    # Resumen de resultados
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="1. Impacto de Eventos Macroeconómicos", ln=True)
    pdf.ln(5)

    # Añadir tabla de impacto de eventos
    for i, row in impacto_eventos_df.iterrows():
        evento = row['Evento']
        fecha = row['Fecha'].strftime('%Y-%m-%d')
        categoria = row['Categoría']
        cambio = f"{row['Cambio (%)']:.2f}%"
        pdf.cell(0, 10, txt=f"{evento} ({fecha}) - Categoría: {categoria}, Cambio: {cambio}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="2. Volatilidad Antes y Después de los Eventos", ln=True)
    pdf.ln(5)

    # Añadir tabla de volatilidad
    for i, row in volatilidad_df.iterrows():
        evento = row['Evento']
        fecha = row['Fecha'].strftime('%Y-%m-%d')
        vol_antes = f"{row['Volatilidad Antes (%)']:.2f}%"
        vol_despues = f"{row['Volatilidad Después (%)']:.2f}%"
        pdf.cell(0, 10, txt=f"{evento} ({fecha}) - Volatilidad Antes: {vol_antes}, Después: {vol_despues}", ln=True)

    pdf.output(output_path)

# **4. Tab: Generar Informe**
with tab_informe:
    if st.button("Generar Informe PDF"):
        generar_informe(impacto_eventos_df, volatilidad_df)
        with open("informe_analisis.pdf", "rb") as file:
            st.download_button(
                label="Descargar Informe PDF",
                data=file,
                file_name="informe_analisis.pdf",
                mime="application/pdf"
            )
