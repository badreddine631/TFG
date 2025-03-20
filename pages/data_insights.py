import streamlit as st
import plotly.express as px
import pandas as pd

def show(df, eventos_df):
    """Muestra la sección de Información Basada en Datos."""
    st.title("📉 Información Basada en Datos")

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
    fig_volatilidad = px.bar(volatilidad_df, x='Evento', y=['Volatilidad Antes (%)', 'Volatilidad Después (%)'], barmode='group', title="Volatilidad de Eventos")
    st.plotly_chart(fig_volatilidad)
    st.dataframe(volatilidad_df)

    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
