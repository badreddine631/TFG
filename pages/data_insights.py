import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show(df, eventos_df, dolar_df, df_oro_ext, df_btc):
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
    
        # 📌 📊 **Sección 4: Volatilidad Mensual del Oro**
    st.subheader("📊 Volatilidad Mensual del Precio del Oro")

    # Asegurar que el índice es datetime (ya está en tu carga, por si acaso)
    df.index = pd.to_datetime(df.index)

    # Calcular los retornos diarios
    df["Retorno Diario"] = df["Precio_Oro"].pct_change()

    # Agrupar por año y mes y calcular la desviación estándar (volatilidad)
    volatilidad_mensual = df["Retorno Diario"].groupby(pd.Grouper(freq='M')).std() * 100
    volatilidad_mensual = volatilidad_mensual.dropna()

    # Crear DataFrame con fechas formateadas
    volatilidad_mensual_df = pd.DataFrame({
        "Fecha": volatilidad_mensual.index.strftime('%Y-%m'),
        "Volatilidad (%)": volatilidad_mensual.values
    })

    # 📊 Gráfico interactivo de barras
    fig_vol_mensual = px.bar(
        volatilidad_mensual_df,
        x="Fecha",
        y="Volatilidad (%)",
        title="Volatilidad Mensual del Precio del Oro",
        labels={"Volatilidad (%)": "Desviación Estándar (%)"},
    )
    fig_vol_mensual.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_vol_mensual, use_container_width=True)
    
        # 📌 📊 **Sección 5: Comparación entre Oro y Bitcoin**
    st.subheader("💰 Comparación entre Oro y Bitcoin")

    # Selección de fechas para comparar
    start_cmp = pd.to_datetime(
        st.sidebar.date_input("Fecha de inicio comparación Oro vs BTC", value=df_oro_ext.index.min(), key="cmp_fecha_inicio")
    )
    end_cmp = pd.to_datetime(
        st.sidebar.date_input("Fecha de fin comparación Oro vs BTC", value=df_oro_ext.index.max(), key="cmp_fecha_fin")
    )

    # Filtrar por fechas comunes
    oro_filtrado = df_oro_ext[(df_oro_ext.index >= start_cmp) & (df_oro_ext.index <= end_cmp)].copy()
    btc_filtrado = df_btc[(df_btc.index >= start_cmp) & (df_btc.index <= end_cmp)].copy()

    # Unir ambos por fecha
    merged_cmp = pd.merge(oro_filtrado, btc_filtrado, left_index=True, right_index=True, how="inner")

    # Limpiar valores nulos o incorrectos
    merged_cmp = merged_cmp[(merged_cmp["Precio_Oro"] > 0) & (merged_cmp["Precio_BTC"] > 0)].dropna()

    # 📊 Gráfico interactivo de precios absolutos
    fig_abs = px.line(
        merged_cmp,
        x=merged_cmp.index,
        y=["Precio_Oro", "Precio_BTC"],
        labels={"value": "Precio (USD)", "variable": "Activo"},
        title="📉 Evolución del Precio Absoluto: Oro vs Bitcoin"
    )
    st.plotly_chart(fig_abs, use_container_width=True)

    # Normalizar para comparación relativa (Base 100)
    merged_cmp["Oro_norm"] = merged_cmp["Precio_Oro"] / merged_cmp["Precio_Oro"].iloc[0] * 100
    merged_cmp["BTC_norm"] = merged_cmp["Precio_BTC"] / merged_cmp["Precio_BTC"].iloc[0] * 100

    # 📊 Gráfico normalizado (evolución relativa)
    fig_norm = px.line(
        merged_cmp,
        x=merged_cmp.index,
        y=["Oro_norm", "BTC_norm"],
        labels={"value": "Índice Base 100", "variable": "Activo"},
        title="📈 Evolución Relativa: Oro vs Bitcoin (Base 100)"
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    # 📌 Correlación
    corr = merged_cmp[["Precio_Oro", "Precio_BTC"]].corr().iloc[0, 1]
    st.markdown(f"📌 **Correlación entre Oro y Bitcoin**: `{corr:.4f}`")

    # 📍 Pregunta 1: ¿Cuál ha sido más rentable desde {start_cmp.date()}?
    rentabilidad_oro = (merged_cmp["Precio_Oro"].iloc[-1] / merged_cmp["Precio_Oro"].iloc[0] - 1) * 100
    rentabilidad_btc = (merged_cmp["Precio_BTC"].iloc[-1] / merged_cmp["Precio_BTC"].iloc[0] - 1) * 100

    st.markdown("### 📊 Rentabilidad acumulada")
    st.markdown(f"- 🟡 **Oro**: `{rentabilidad_oro:.2f}%`")
    st.markdown(f"- 🔵 **Bitcoin**: `{rentabilidad_btc:.2f}%`")

    # 📍 Pregunta 2: ¿Cuál ha sido más volátil?
    volatilidad_oro = merged_cmp["Precio_Oro"].pct_change().std() * 100
    volatilidad_btc = merged_cmp["Precio_BTC"].pct_change().std() * 100

    st.markdown("### 📉 Volatilidad histórica")
    st.markdown(f"- 🟡 **Oro**: `{volatilidad_oro:.2f}%` desviación estándar diaria")
    st.markdown(f"- 🔵 **Bitcoin**: `{volatilidad_btc:.2f}%` desviación estándar diaria")

    # 📍 Pregunta 3: ¿Tienen relación? Ya respondida arriba con correlación.

    # 📍 Pregunta 4: ¿Qué activo ha resistido mejor durante caídas?
    st.markdown("### 📉 Caída máxima (drawdown)")
    oro_drawdown = (merged_cmp["Precio_Oro"] / merged_cmp["Precio_Oro"].cummax() - 1).min() * 100
    btc_drawdown = (merged_cmp["Precio_BTC"] / merged_cmp["Precio_BTC"].cummax() - 1).min() * 100
    st.markdown(f"- 🟡 **Oro**: caída máxima `{oro_drawdown:.2f}%`")
    st.markdown(f"- 🔵 **Bitcoin**: caída máxima `{btc_drawdown:.2f}%`")

        
  

    # 🔙 **Botón para volver**
    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
