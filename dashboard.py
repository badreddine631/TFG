import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar los datos
df = pd.read_csv("C:\\Users\\Usuario\\Desktop\\TFFFG\\price-gold\\XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
eventos_df = pd.read_csv("C:\\Users\\Usuario\\Desktop\\TFFFG\\price-gold\\XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Fecha"])

# Título del dashboard
st.title("Dashboard Interactivo: Análisis del Oro y Eventos Macroeconómicos")

# Selección de rango de fechas
st.sidebar.header("Seleccione un Rango de Fechas")
start_date = st.sidebar.date_input("Fecha de Inicio", value=df["Date"].min())
end_date = st.sidebar.date_input("Fecha de Fin", value=df["Date"].max())

# Filtrar datos por rango de fechas
df_filtrado = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

# Gráfico interactivo del precio del oro
st.subheader("Precio del Oro a lo Largo del Tiempo")
fig = px.line(df_filtrado, x="Date", y="Close", title="Precio del Oro")
st.plotly_chart(fig)

# Comparar eventos directamente
st.subheader("Impacto de Eventos Macroeconómicos")
fig_eventos = px.bar(
    eventos_df,
    x="Evento",
    y="Cambio (%)",
    title="Impacto Promedio de los Eventos",
    labels={"Evento": "Eventos", "Cambio (%)": "Cambio Promedio (%)"},
)
st.plotly_chart(fig_eventos)
