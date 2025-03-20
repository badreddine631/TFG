import pandas as pd
import streamlit as st
import plotly.express as px

def show(df, eventos_df):
    """Muestra la sección de Análisis de Tendencias del Mercado."""
    st.title("📈 Análisis del Precio del Oro y Eventos Macroeconómicos")

    # Filtrar por fechas
    start_date = st.sidebar.date_input("Fecha de Inicio", value=df.index.min(), key="precio_fecha_inicio")
    end_date = st.sidebar.date_input("Fecha de Fin", value=df.index.max(), key="precio_fecha_fin")
    df_filtrado = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    # Mostrar gráfico del precio del oro
    st.subheader("Evolución del Precio del Oro")
    fig_precio = px.line(df_filtrado, x=df_filtrado.index, y="Precio_Oro", title="Precio del Oro")
    fig_precio.update_layout(xaxis_title="Fecha", yaxis_title="Precio de Cierre (USD)")
    st.plotly_chart(fig_precio)

    # Mostrar eventos macroeconómicos
    st.subheader("Eventos Macroeconómicos")
    st.dataframe(eventos_df)

    # Botón para volver atrás
    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
