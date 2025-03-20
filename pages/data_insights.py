import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show(df, eventos_df, dolar_df):
    """Muestra la sección de Información Basada en Datos con gráficos interactivos."""
    
    st.title("📉 Información Basada en Datos")

    # 📌 📊 **Sección 1: Análisis de Volatilidad Antes y Después de los Eventos**
    st.subheader("📈 Volatilidad Antes y Después de los Eventos")

    volatilidad_eventos = []
    for _, evento in eventos_df.iterrows():
        fecha_evento = evento['Fecha']
        rango_antes = df[(df.index >= fecha_evento - pd.Timedelta(days=7)) & (df.index < fecha_evento)]
        rango_despues = df[(df.index > fecha_evento) & (df.index <= fecha_evento + pd.Timedelta(days=7))]
        
        if not rango_antes.empty and not rango_despues.empty:
            volatilidad_antes = rango_antes['Precio_Oro'].pct_change().std() * 100
            volatilidad_despues = rango_despues['Precio_Oro'].pct_change().std() * 100
        else:
            volatilidad_antes, volatilidad_despues = None, None

        volatilidad_eventos.append({
            'Evento': evento['Evento'],
            'Fecha': evento['Fecha'],
            'Volatilidad Antes (%)': volatilidad_antes,
            'Volatilidad Después (%)': volatilidad_despues,
            'Categoría': evento['Categoría']
        })
    
    volatilidad_df = pd.DataFrame(volatilidad_eventos)

    # 📊 **Gráfico interactivo de Volatilidad**
    fig_volatilidad = px.bar(
        volatilidad_df, 
        x='Evento', 
        y=['Volatilidad Antes (%)', 'Volatilidad Después (%)'], 
        barmode='group', 
        title="Volatilidad Antes y Después de los Eventos"
    )
    st.plotly_chart(fig_volatilidad, use_container_width=True)
    st.dataframe(volatilidad_df)

    # 📌 📊 **Sección 2: Media Móvil del Precio del Oro**
    st.subheader("📉 Media Móvil del Precio del Oro")

    # 📌 **Selección de fechas**
    start_date = pd.to_datetime(st.sidebar.date_input("Fecha de Inicio", value=df.index.min(), key="media_fecha_inicio"))
    end_date = pd.to_datetime(st.sidebar.date_input("Fecha de Fin", value=df.index.max(), key="media_fecha_fin"))

    # 📌 **Filtrar datos**
    df_filtrado = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    # 📌 **Cálculo de Medias Móviles**
    df_filtrado["Media Móvil 7 días"] = df_filtrado["Precio_Oro"].rolling(window=7).mean()
    df_filtrado["Media Móvil 30 días"] = df_filtrado["Precio_Oro"].rolling(window=30).mean()
    df_filtrado["Media Móvil 90 días"] = df_filtrado["Precio_Oro"].rolling(window=90).mean()

    # 📊 **Gráfico interactivo de Media Móvil**
    fig_media_movil = px.line(
        df_filtrado, 
        x=df_filtrado.index, 
        y=["Precio_Oro", "Media Móvil 7 días", "Media Móvil 30 días", "Media Móvil 90 días"], 
        labels={"value": "Precio (USD)", "variable": "Tipo"},
        title="Evolución del Precio con Medias Móviles"
    )
    fig_media_movil.update_layout(xaxis_title="Fecha", yaxis_title="Precio (USD)")
    st.plotly_chart(fig_media_movil, use_container_width=True)

    # 📌 📊 **Sección 3: Correlación entre Oro y DXY**
    st.subheader("📈 Correlación entre el Precio del Oro y el Dólar (DXY)")

    # Fusionar los datos en un solo DataFrame
    merged_df = df[["Precio_Oro"]].merge(dolar_df[["DXY"]], left_index=True, right_index=True, how="inner")

    # Calcular la correlación
    correlation = merged_df.corr().iloc[0, 1]
    st.write(f"📌 **Correlación entre Oro y DXY**: `{correlation:.4f}`")

    # 📍 **Scatter plot interactivo**
    fig_scatter = px.scatter(
        merged_df, 
        x="DXY", 
        y="Precio_Oro", 
        opacity=0.5, 
        title="Relación entre el Índice DXY y el Precio del Oro"
    )
    fig_scatter.update_layout(xaxis_title="DXY (Índice del Dólar)", yaxis_title="Precio del Oro (USD)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 📍 **Heatmap de correlación interactivo**
    fig_heatmap = px.imshow(
        merged_df.corr(), 
        text_auto=True, 
        color_continuous_scale="Viridis",  # ✅ COLOR VÁLIDO PARA PLOTLY
        title="Mapa de Calor de Correlación"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # 🔙 **Botón para volver**
    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
