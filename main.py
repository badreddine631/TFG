import pandas as pd
import streamlit as st
from utils.data_loader import load_data
from bloques import market_trends, data_insights, custom_reports

# Configuración de la app
st.set_page_config(
    page_title="MacroGold Analytics",
    page_icon=":bar_chart:",
    layout="wide"
)

# Definir una variable de estado para cambiar entre páginas
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Función para cambiar de página
def change_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()


with open("assets/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Cargar datos
df, eventos_df, dolar_df, df_oro_ext, df_btc, df_plata, df_oro_eur, df_oro_cny, df_petroleo, df_cad, df_elecciones_usa, df_recesiones, df_crisis_europa, df_crisis_inmo,df_crisis_minerales,  cpi_df, df_oro_mensual, cci_df, fear_greed_df , df_walcl, pib_df = load_data()


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

# Contenedor de cuadros clicables

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Análisis de Tendencias del Mercado"):
        change_page("market_trends")

with col2:
    if st.button("📉 Información Basada en Datos"):
        change_page("data_insights")

with col3:
    if st.button("📂 Informes Personalizables"):
        change_page("custom_reports")

# Cargar la página seleccionada
if st.session_state.current_page == "market_trends":
    market_trends.show(df, eventos_df)
elif st.session_state.current_page == "data_insights":
    data_insights.show(df, eventos_df, dolar_df, df_oro_ext, df_btc, df_plata, df_oro_eur, df_oro_cny, df_petroleo, df_cad, df_elecciones_usa, df_recesiones,df_crisis_europa,df_crisis_inmo, df_crisis_minerales, cpi_df, df_oro_mensual, cci_df,fear_greed_df, df_walcl,pib_df )
elif st.session_state.current_page == "custom_reports":
    custom_reports.show(df, eventos_df)
else:
    st.markdown("<p style='text-align: center;'>Selecciona una opción para ver los análisis.</p>", unsafe_allow_html=True)
