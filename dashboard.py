import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import os

# Definir una variable de estado para cambiar entre páginas
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Función para cambiar de página
def change_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()  # Recargar la interfaz

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

# Contenedor de cuadros clicables usando columnas de Streamlit
st.markdown("### Selecciona un análisis:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Análisis de Tendencias del Mercado", key="btn_market_trends"):
        st.session_state.current_page = "market_trends"
        st.rerun()

with col2:
    if st.button("📉 Información Basada en Datos", key="btn_volatility_analysis"):
        st.session_state.current_page = "data_insights"
        st.rerun()

with col3:
    if st.button("📂 Informes Personalizables", key="btn_generate_report"):
        st.session_state.current_page = "custom_reports"
        st.rerun()

# Mostrar la página correcta según el estado
if st.session_state.current_page == "home":
    st.markdown("<p style='text-align: center;'>Selecciona una opción para ver los análisis.</p>", unsafe_allow_html=True)

elif st.session_state.current_page == "market_trends":
    st.title("📈 Análisis del Precio del Oro y Eventos Macroeconómicos")

    # Filtrar por fechas
    start_date = st.sidebar.date_input("Fecha de Inicio", value=df.index.min(), key="precio_fecha_inicio")
    end_date = st.sidebar.date_input("Fecha de Fin", value=df.index.max(), key="precio_fecha_fin")
    df_filtrado = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    # Mostrar gráfico del precio del oro
    st.subheader("Evolución del Precio del Oro")
    fig_precio = px.line(df_filtrado, x=df_filtrado.index, y="Close", title="Precio del Oro")
    fig_precio.update_layout(xaxis_title="Fecha", yaxis_title="Precio de Cierre (USD)")
    st.plotly_chart(fig_precio)

    # Mostrar eventos macroeconómicos
    st.subheader("Eventos Macroeconómicos")
    st.dataframe(eventos_df)

    # Botón para volver atrás
    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()

elif st.session_state.current_page == "data_insights":
    st.title("📉 Información Basada en Datos")

    # Calcular la volatilidad antes y después de eventos macroeconómicos
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

    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()

elif st.session_state.current_page == "custom_reports":
    st.title("📂 Informes Personalizables")

    # Generar informe en PDF
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

    if st.button("Generar Informe PDF"):
        generar_informe(eventos_df, volatilidad_df)
        with open("informe_analisis.pdf", "rb") as file:
            st.download_button(
                label="Descargar Informe PDF",
                data=file,
                file_name="informe_analisis.pdf",
                mime="application/pdf"
            )

    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
