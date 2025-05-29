import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ta
from ta.utils import dropna
from ta.trend import SMAIndicator
from ta.momentum import ROCIndicator



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

    # 📌 📊 **Sección 6: Estacionalidad y Retorno Anual**
    st.subheader("📅 Estacionalidad y Retorno Anual del Precio del Oro")

    # Precio medio mensual (estacionalidad)
    precio_mensual = df["Precio_Oro"].resample("M").mean().reset_index()
    precio_mensual["Año-Mes"] = precio_mensual["Fecha"].dt.to_period("M").astype(str)

    fig_estacionalidad = px.line(
        precio_mensual,
        x="Año-Mes",
        y="Precio_Oro",
        title="📆 Precio Medio Mensual del Oro (Estacionalidad)",
        labels={"Precio_Oro": "Precio Medio (USD)", "Año-Mes": "Mes"},
    )
    fig_estacionalidad.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_estacionalidad, use_container_width=True)

    # Retorno anual acumulado
    df["Retorno Diario"] = df["Precio_Oro"].pct_change()
    retorno_anual = (1 + df["Retorno Diario"]).resample("Y").prod() - 1
    retorno_anual_df = retorno_anual.reset_index()
    retorno_anual_df["Año"] = retorno_anual_df["Fecha"].dt.year
    retorno_anual_df["Retorno %"] = retorno_anual_df["Retorno Diario"] * 100

    fig_retorno = px.bar(
        retorno_anual_df,
        x="Año",
        y="Retorno %",
        title="📈 Retorno Anual Acumulado del Oro",
        labels={"Retorno %": "Retorno (%)"},
    )
    st.plotly_chart(fig_retorno, use_container_width=True)

    # 📌 📊 **Sección 7: Días con Subidas y Bajadas más Extremas**
    st.subheader("🔺 Días con Mayores Subidas y Bajadas del Precio del Oro")

    variacion = df["Retorno Diario"] * 100
    top_subidas = variacion.nlargest(10).round(2)
    top_bajadas = variacion.nsmallest(10).round(2)

    st.markdown("##### 🔼 Top 10 Subidas")
    st.dataframe(top_subidas.to_frame(name="Variación %"))

    st.markdown("##### 🔽 Top 10 Bajadas")
    st.dataframe(top_bajadas.to_frame(name="Variación %"))

    # 📌 📊 **Sección 8: Distribución de Retornos Diarios**
    st.subheader("📊 Distribución de Retornos Diarios del Oro")

    fig_dist = px.histogram(
        df.dropna(), 
        x="Retorno Diario", 
        nbins=100,
        title="Distribución de Retornos Diarios",
        labels={"Retorno Diario": "Retorno Diario (%)"}
    )
    fig_dist.update_layout(bargap=0.01)
    st.plotly_chart(fig_dist, use_container_width=True)

    # 📌 📊 **Sección 9: Eventos Cercanos a Cambios Bruscos**
    st.subheader("📌 Relación entre Eventos y Cambios Significativos")

    # 1. Días más volátiles
    dias_volatiles = variacion[variacion.abs() >= 2]

    categorias_impacto = []
    for fecha in dias_volatiles.index:
        eventos = eventos_df[
            (eventos_df["Fecha"] >= fecha - pd.Timedelta(days=2)) &
            (eventos_df["Fecha"] <= fecha + pd.Timedelta(days=2))
        ]
        for _, e in eventos.iterrows():
            categorias_impacto.append(e.get("Categoría", "Sin categoría"))

    from collections import Counter
    conteo = Counter(categorias_impacto)

    st.markdown("### 🧠 Eventos Asociados a Alta Volatilidad")
    for cat, count in conteo.most_common():
        st.markdown(f"- **{cat}**: {count} días con alta volatilidad asociados")

    # 2. Eventos cercanos a cambios de tendencia
    df["MA5"] = df["Precio_Oro"].rolling(5).mean()
    df["MA20"] = df["Precio_Oro"].rolling(20).mean()
    df["Crossover"] = df["MA5"] - df["MA20"]
    df["Cruzado"] = df["Crossover"].apply(lambda x: 1 if x > 0 else -1)
    df["Cambio_Tendencia"] = df["Cruzado"].diff()

    fechas_cambio = df[df["Cambio_Tendencia"].abs() == 2].index
    cambios_eventos = []

    for fecha in fechas_cambio:
        eventos = eventos_df[
            (eventos_df["Fecha"] >= fecha - pd.Timedelta(days=2)) &
            (eventos_df["Fecha"] <= fecha + pd.Timedelta(days=2))
        ]
        for _, evento in eventos.iterrows():
            cambios_eventos.append({
                "Fecha Cambio": fecha.date(),
                "Evento": evento["Evento"],
                "Fecha Evento": evento["Fecha"].date(),
                "Categoría": evento.get("Categoría", "Sin categoría")
            })

    df_cambios = pd.DataFrame(cambios_eventos)
    st.markdown("### 🔄 Eventos Relacionados con Cambios de Tendencia")
    st.dataframe(df_cambios)
        # 📌 📊 **Sección 8: Patrones de Velas Japonesas (con librería `ta`)**


    st.subheader("🕯️ Patrones de Velas Japonesas (estimados)")

    # Cargar datos originales (sin renombrar 'Close') desde CSV
    df_velas = pd.read_csv("datasets/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
    df_velas.set_index("Date", inplace=True)

    # Asegurarse de tener las columnas correctas como numéricas
    for col in ["Open", "High", "Low", "Close"]:
        df_velas[col] = pd.to_numeric(df_velas[col], errors="coerce")
    df_velas = dropna(df_velas)

    # Selección de fechas
    start_velas = pd.to_datetime(
        st.sidebar.date_input("Fecha de inicio velas (estimación)", value=df_velas.index.min(), key="vela_inicio")
    )
    end_velas = pd.to_datetime(
        st.sidebar.date_input("Fecha de fin velas (estimación)", value=df_velas.index.max(), key="vela_fin")
    )
    df_velas = df_velas[(df_velas.index >= start_velas) & (df_velas.index <= end_velas)].copy()

    # --- Detectar patrones simples con lógica básica ---
    patrones = []

    for i in range(1, len(df_velas)):
        prev = df_velas.iloc[i - 1]
        curr = df_velas.iloc[i]

        if curr["Open"] < curr["Close"] and prev["Open"] > prev["Close"] and curr["Open"] < prev["Close"] and curr["Close"] > prev["Open"]:
            patrones.append("Envolvente Alcista")
        elif curr["Open"] > curr["Close"] and prev["Open"] < prev["Close"] and curr["Open"] > prev["Close"] and curr["Close"] < prev["Open"]:
            patrones.append("Envolvente Bajista")
        elif abs(curr["Close"] - curr["Open"]) <= (curr["High"] - curr["Low"]) * 0.1:
            patrones.append("Doji")
        else:
            patrones.append("")

    df_velas = df_velas.iloc[1:].copy()
    df_velas["Patrón"] = patrones
    df_patrones = df_velas[df_velas["Patrón"] != ""]

    # 📊 Gráfico interactivo con candlestick y patrones
    import plotly.graph_objects as go

    fig = go.Figure(data=[
        go.Candlestick(
            x=df_velas.index,
            open=df_velas["Open"],
            high=df_velas["High"],
            low=df_velas["Low"],
            close=df_velas["Close"],
            name="Precio"
        ),
        go.Scatter(
            x=df_patrones.index,
            y=df_patrones["Close"],
            mode="markers",
            marker=dict(color="red", size=10, symbol="star"),
            name="Patrones detectados",
            text=df_patrones["Patrón"]
        )
    ])

    fig.update_layout(title="🕯️ Patrones Estimados de Velas Japonesas en el Oro", xaxis_title="Fecha", yaxis_title="Precio (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar tabla
    st.markdown("### 📋 Patrones Detectados")
    st.dataframe(df_patrones[["Open", "High", "Low", "Close", "Patrón"]])
    # 📌 📊 **Sección 10: Backtesting - Estrategia de Cruce de Medias Móviles**
    st.subheader("🔟 Backtesting: Estrategia de Cruce de Medias Móviles en Oro")

    # Selección de parámetros por el usuario
    short_window = st.sidebar.slider("Ventana Corta (días)", min_value=5, max_value=50, value=20, key="backtest_ma_corta")
    long_window = st.sidebar.slider("Ventana Larga (días)", min_value=20, max_value=200, value=100, key="backtest_ma_larga")

    # Copia de datos para evitar modificar df original
    df_bt = df.copy()
    df_bt["MA_Corta"] = df_bt["Precio_Oro"].rolling(window=short_window).mean()
    df_bt["MA_Larga"] = df_bt["Precio_Oro"].rolling(window=long_window).mean()
    df_bt = df_bt.dropna()

    # Señales de trading: 1 (compra), 0 (sin posición)
    df_bt["Señal"] = (df_bt["MA_Corta"] > df_bt["MA_Larga"]).astype(int)

    # Calcular retornos
    df_bt["Retorno Diario"] = df_bt["Precio_Oro"].pct_change()
    df_bt["Retorno Estrategia"] = df_bt["Señal"].shift(1) * df_bt["Retorno Diario"]
    df_bt.dropna(inplace=True)

    # Capital acumulado
    df_bt["Capital Estrategia"] = (1 + df_bt["Retorno Estrategia"]).cumprod()
    df_bt["Capital Buy & Hold"] = (1 + df_bt["Retorno Diario"]).cumprod()

    # 📊 Gráfico comparativo
    fig_backtest = px.line(
        df_bt,
        x=df_bt.index,
        y=["Capital Estrategia", "Capital Buy & Hold"],
        title="📈 Evolución del Capital: Estrategia MA vs Buy & Hold",
        labels={"value": "Crecimiento (%)", "variable": "Método"}
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

    # 📊 Métricas clave
    total_estrategia = df_bt["Capital Estrategia"].iloc[-1] - 1
    total_hold = df_bt["Capital Buy & Hold"].iloc[-1] - 1
    vol_estrategia = df_bt["Retorno Estrategia"].std() * (252 ** 0.5)
    vol_hold = df_bt["Retorno Diario"].std() * (252 ** 0.5)
    sharpe_estrategia = total_estrategia / vol_estrategia if vol_estrategia != 0 else 0
    sharpe_hold = total_hold / vol_hold if vol_hold != 0 else 0

    st.markdown("### 📊 Resultados del Backtesting")
    st.markdown(f"- 📈 **Rentabilidad Estrategia MA**: `{total_estrategia:.2%}`")
    st.markdown(f"- 📈 **Rentabilidad Buy & Hold**: `{total_hold:.2%}`")
    st.markdown(f"- ⚠️ **Volatilidad Estrategia MA**: `{vol_estrategia:.2%}`")
    st.markdown(f"- ⚠️ **Volatilidad Buy & Hold**: `{vol_hold:.2%}`")
    st.markdown(f"- 📌 **Sharpe Ratio Estrategia MA**: `{sharpe_estrategia:.2f}`")
    st.markdown(f"- 📌 **Sharpe Ratio Buy & Hold**: `{sharpe_hold:.2f}`")

    st.markdown("💡 *Una estrategia con mejor Sharpe Ratio y menor volatilidad puede ser preferible incluso con rentabilidad similar.*")
    # 📌 📍 Visualización con señales de compra/venta
    st.subheader("📌 Señales de Trading (Cruce de Medias)")

    # Detectar puntos de cambio (cruces)
    df_bt["Cambio"] = df_bt["Señal"].diff()

    compras = df_bt[df_bt["Cambio"] == 1]
    ventas = df_bt[df_bt["Cambio"] == -1]

    import plotly.graph_objects as go

    fig_signals = go.Figure()

    # Línea de precio del oro
    fig_signals.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt["Precio_Oro"],
        mode='lines',
        name="Precio Oro",
        line=dict(color='gold')
    ))

    # Señales de compra
    fig_signals.add_trace(go.Scatter(
        x=compras.index,
        y=compras["Precio_Oro"],
        mode='markers',
        marker=dict(symbol="arrow-up", color="green", size=12),
        name="Compra"
    ))

    # Señales de venta
    fig_signals.add_trace(go.Scatter(
        x=ventas.index,
        y=ventas["Precio_Oro"],
        mode='markers',
        marker=dict(symbol="arrow-down", color="red", size=12),
        name="Venta"
    ))

    # Título y diseño
    fig_signals.update_layout(
        title="📍 Señales de Compra y Venta (Cruce de Medias)",
        xaxis_title="Fecha",
        yaxis_title="Precio Oro (USD)",
        legend=dict(x=0.01, y=0.99),
        height=500
    )

    st.plotly_chart(fig_signals, use_container_width=True)
    # 📌 📊 **Sección: Backtesting - Triple Cruce de Medias Móviles**
    st.subheader("📊 Backtesting: Estrategia de Triple Cruce de Medias Móviles")

    # Copiar DataFrame base
    df_triple = df.copy()

    # Calcular medias móviles
    df_triple["SMA7"] = df_triple["Precio_Oro"].rolling(window=7).mean()
    df_triple["SMA30"] = df_triple["Precio_Oro"].rolling(window=30).mean()
    df_triple["SMA90"] = df_triple["Precio_Oro"].rolling(window=90).mean()

    # Crear condición de compra: SMA7 > SMA30 y ambas > SMA90
    df_triple["Condicion_Compra"] = (df_triple["SMA7"] > df_triple["SMA30"]) & \
                                     (df_triple["SMA30"] > df_triple["SMA90"])

    # Señales de compra y venta
    df_triple["Señal"] = df_triple["Condicion_Compra"].astype(int).diff()

    # Simulación de estrategia (compramos 1 unidad si señal == 1, vendemos si señal == -1)
    capital_inicial = 10000
    en_posicion = False
    capital = capital_inicial
    posiciones = []
    señales_entrada = []
    señales_salida = []

    for fecha, fila in df_triple.iterrows():
        precio = fila["Precio_Oro"]
        señal = fila["Señal"]

        if señal == 1 and not en_posicion:
            unidades = capital / precio
            en_posicion = True
            señales_entrada.append((fecha, precio))
        elif señal == -1 and en_posicion:
            capital = unidades * precio
            en_posicion = False
            señales_salida.append((fecha, precio))
        posiciones.append(capital if not en_posicion else unidades * precio)

    df_triple["Capital"] = posiciones

    # 📈 Visualización de señales
    fig_triple = px.line(df_triple, x=df_triple.index, y="Precio_Oro", title="📉 Precio del Oro con Señales Triple SMA")
    for fecha, precio in señales_entrada:
        fig_triple.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"),
                               name="Compra")
    for fecha, precio in señales_salida:
        fig_triple.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"),
                               name="Venta")
    st.plotly_chart(fig_triple, use_container_width=True)

    # 📊 Resultados del Backtesting - Estrategia Triple SMA
    st.markdown("### 📊 Resultados del Backtesting - Estrategia Triple SMA")

    # Calcular métricas
    rentabilidad_triple = (df_triple["Capital"].iloc[-1] / df_triple["Capital"].iloc[0] - 1) * 100
    rentabilidad_hold = (df["Precio_Oro"].iloc[-1] / df["Precio_Oro"].iloc[0] - 1) * 100

    # Volatilidad anualizada
    ret_diarios_triple = df_triple["Capital"].pct_change().dropna()
    ret_diarios_hold = df["Precio_Oro"].pct_change().dropna()
    volatilidad_triple = ret_diarios_triple.std() * (252**0.5) * 100
    volatilidad_hold = ret_diarios_hold.std() * (252**0.5) * 100

    # Sharpe ratio (sin tasa libre de riesgo)
    sharpe_triple = (ret_diarios_triple.mean() / ret_diarios_triple.std()) * (252**0.5)
    sharpe_hold = (ret_diarios_hold.mean() / ret_diarios_hold.std()) * (252**0.5)

    # Mostrar métricas
    st.markdown("#### 📈 Rentabilidad")
    st.markdown(f"- 🟢 **Estrategia Triple SMA**: `{rentabilidad_triple:.2f}%`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{rentabilidad_hold:.2f}%`")

    st.markdown("#### ⚠️ Volatilidad Anualizada")
    st.markdown(f"- 🟢 **Estrategia Triple SMA**: `{volatilidad_triple:.2f}%`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{volatilidad_hold:.2f}%`")

    st.markdown("#### 🧮 Sharpe Ratio")
    st.markdown(f"- 🟢 **Estrategia Triple SMA**: `{sharpe_triple:.2f}`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{sharpe_hold:.2f}`")
    # 📌 📊 **Sección: Backtesting - Estrategia de Bandas de Bollinger**
    st.subheader("📊 Backtesting: Estrategia de Bandas de Bollinger")

    # Copia base
    df_bollinger = df.copy()

    # Calcular Bollinger Bands
    window = 20
    std_mult = 2
    df_bollinger["MA20"] = df_bollinger["Precio_Oro"].rolling(window=window).mean()
    df_bollinger["Upper"] = df_bollinger["MA20"] + std_mult * df_bollinger["Precio_Oro"].rolling(window=window).std()
    df_bollinger["Lower"] = df_bollinger["MA20"] - std_mult * df_bollinger["Precio_Oro"].rolling(window=window).std()

    # Señales: compra cuando el precio cruza por debajo de la banda inferior, venta cuando cruza por encima de la superior
    df_bollinger["Señal"] = 0
    df_bollinger.loc[df_bollinger["Precio_Oro"] < df_bollinger["Lower"], "Señal"] = 1
    df_bollinger.loc[df_bollinger["Precio_Oro"] > df_bollinger["Upper"], "Señal"] = -1
    df_bollinger["Cambio"] = df_bollinger["Señal"].diff()

    # Backtesting
    capital_inicial = 10000
    capital = capital_inicial
    en_posicion = False
    posiciones = []
    señales_compra = []
    señales_venta = []

    for fecha, fila in df_bollinger.iterrows():
        precio = fila["Precio_Oro"]
        señal = fila["Cambio"]

        if señal == 1 and not en_posicion:
            unidades = capital / precio
            en_posicion = True
            señales_compra.append((fecha, precio))
        elif señal == -1 and en_posicion:
            capital = unidades * precio
            en_posicion = False
            señales_venta.append((fecha, precio))

        posiciones.append(capital if not en_posicion else unidades * precio)

    df_bollinger["Capital"] = posiciones

    # 📈 Gráfico con señales
    fig_bollinger = px.line(df_bollinger, x=df_bollinger.index, y="Precio_Oro", title="📉 Precio del Oro con Señales de Bandas de Bollinger")
    fig_bollinger.add_scatter(x=df_bollinger.index, y=df_bollinger["Upper"], mode="lines", name="Banda Superior", line=dict(dash="dot"))
    fig_bollinger.add_scatter(x=df_bollinger.index, y=df_bollinger["Lower"], mode="lines", name="Banda Inferior", line=dict(dash="dot"))
    for fecha, precio in señales_compra:
        fig_bollinger.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"), name="Compra")
    for fecha, precio in señales_venta:
        fig_bollinger.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"), name="Venta")
    st.plotly_chart(fig_bollinger, use_container_width=True)

    # 📊 Métricas de rendimiento
    st.markdown("### 📊 Resultados del Backtesting - Estrategia Bollinger")

    rentabilidad_bollinger = (df_bollinger["Capital"].iloc[-1] / df_bollinger["Capital"].iloc[0] - 1) * 100
    rentabilidad_hold = (df["Precio_Oro"].iloc[-1] / df["Precio_Oro"].iloc[0] - 1) * 100

    ret_diarios_bollinger = df_bollinger["Capital"].pct_change().dropna()
    ret_diarios_hold = df["Precio_Oro"].pct_change().dropna()

    volatilidad_bollinger = ret_diarios_bollinger.std() * (252**0.5) * 100
    volatilidad_hold = ret_diarios_hold.std() * (252**0.5) * 100

    sharpe_bollinger = (ret_diarios_bollinger.mean() / ret_diarios_bollinger.std()) * (252**0.5)
    sharpe_hold = (ret_diarios_hold.mean() / ret_diarios_hold.std()) * (252**0.5)

    st.markdown("#### 📈 Rentabilidad")
    st.markdown(f"- 🟠 **Estrategia Bollinger**: `{rentabilidad_bollinger:.2f}%`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{rentabilidad_hold:.2f}%`")

    st.markdown("#### ⚠️ Volatilidad Anualizada")
    st.markdown(f"- 🟠 **Estrategia Bollinger**: `{volatilidad_bollinger:.2f}%`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{volatilidad_hold:.2f}%`")

    st.markdown("#### 🧮 Sharpe Ratio")
    st.markdown(f"- 🟠 **Estrategia Bollinger**: `{sharpe_bollinger:.2f}`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{sharpe_hold:.2f}`")
    # 📌 📊 **Sección: Backtesting - Estrategia Golden Cross / Death Cross**
    st.subheader("📊 Backtesting: Golden Cross / Death Cross")

    df_gc = df.copy()
    df_gc["SMA50"] = df_gc["Precio_Oro"].rolling(window=50).mean()
    df_gc["SMA200"] = df_gc["Precio_Oro"].rolling(window=200).mean()

    # Generar señales
    df_gc["Crossover"] = df_gc["SMA50"] > df_gc["SMA200"]
    df_gc["Cambio"] = df_gc["Crossover"].astype(int).diff()

    df_gc["Señal"] = 0
    df_gc.loc[df_gc["Cambio"] == 1, "Señal"] = 1   # Golden Cross (compra)
    df_gc.loc[df_gc["Cambio"] == -1, "Señal"] = -1  # Death Cross (venta)

    # Backtesting
    capital = 10000
    en_posicion = False
    señales_compra = []
    señales_venta = []
    capitales = []

    for fecha, fila in df_gc.iterrows():
        precio = fila["Precio_Oro"]
        señal = fila["Señal"]

        if señal == 1 and not en_posicion:
            unidades = capital / precio
            en_posicion = True
            señales_compra.append((fecha, precio))
        elif señal == -1 and en_posicion:
            capital = unidades * precio
            en_posicion = False
            señales_venta.append((fecha, precio))

        capitales.append(capital if not en_posicion else unidades * precio)

    df_gc["Capital"] = capitales

    # 📉 Gráfico con cruces y señales
    fig_gc = px.line(df_gc, x=df_gc.index, y="Precio_Oro", title="📉 Precio del Oro - Golden Cross / Death Cross")
    fig_gc.add_scatter(x=df_gc.index, y=df_gc["SMA50"], mode="lines", name="SMA 50 días", line=dict(dash="dot"))
    fig_gc.add_scatter(x=df_gc.index, y=df_gc["SMA200"], mode="lines", name="SMA 200 días", line=dict(dash="dot"))
    for fecha, precio in señales_compra:
        fig_gc.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="green", size=10, symbol="arrow-up"), name="Compra")
    for fecha, precio in señales_venta:
        fig_gc.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="red", size=10, symbol="arrow-down"), name="Venta")
    st.plotly_chart(fig_gc, use_container_width=True)

    # 📊 Métricas de rendimiento
    st.markdown("### 📈 Resultados del Backtesting - Estrategia Golden/Death Cross")

    ret_gc = (df_gc["Capital"].iloc[-1] / df_gc["Capital"].iloc[0] - 1) * 100
    ret_hold = (df["Precio_Oro"].iloc[-1] / df["Precio_Oro"].iloc[0] - 1) * 100

    ret_diario_gc = df_gc["Capital"].pct_change().dropna()
    ret_diario_hold = df["Precio_Oro"].pct_change().dropna()

    vol_gc = ret_diario_gc.std() * (252**0.5) * 100
    vol_hold = ret_diario_hold.std() * (252**0.5) * 100

    sharpe_gc = (ret_diario_gc.mean() / ret_diario_gc.std()) * (252**0.5)
    sharpe_hold = (ret_diario_hold.mean() / ret_diario_hold.std()) * (252**0.5)

    st.markdown(f"- 🟢 **Rentabilidad Estrategia**: `{ret_gc:.2f}%`")
    st.markdown(f"- 🔵 **Rentabilidad Buy & Hold**: `{ret_hold:.2f}%`")
    st.markdown(f"- 🟢 **Volatilidad Estrategia**: `{vol_gc:.2f}%`")
    st.markdown(f"- 🔵 **Volatilidad Buy & Hold**: `{vol_hold:.2f}%`")
    st.markdown(f"- 🟢 **Sharpe Ratio Estrategia**: `{sharpe_gc:.2f}`")
    st.markdown(f"- 🔵 **Sharpe Ratio Buy & Hold**: `{sharpe_hold:.2f}`")
    # 📌 📊 **Sección: Backtesting - Estrategia Buy the Dip (Compra en Caídas)**
    st.subheader("📊 Backtesting: Estrategia Buy the Dip")

    df_dip = df.copy()
    df_dip["Retorno_3d"] = df_dip["Precio_Oro"].pct_change(periods=3) * 100

    # Parámetros configurables
    umbral_caida = st.sidebar.slider("Umbral de caída (%)", min_value=1, max_value=10, value=5, step=1)
    ventana_dias = 3  # fijo para esta estrategia, pero puedes hacerlo configurable también

    df_dip["Señal"] = 0
    df_dip.loc[df_dip["Retorno_3d"] <= -umbral_caida, "Señal"] = 1

    # Backtesting
    capital = 10000
    en_posicion = False
    señales_compra = []
    señales_venta = []
    capitales = []

    for fecha, fila in df_dip.iterrows():
        precio = fila["Precio_Oro"]
        señal = fila["Señal"]

        if señal == 1 and not en_posicion:
            unidades = capital / precio
            en_posicion = True
            señales_compra.append((fecha, precio))
        elif señal == 1 and en_posicion:
            # ya en posición, no hacemos nada
            pass
        elif señal == 0 and en_posicion:
            capital = unidades * precio
            en_posicion = False
            señales_venta.append((fecha, precio))

        capitales.append(capital if not en_posicion else unidades * precio)

    df_dip["Capital"] = capitales

    # 📉 Gráfico con señales
    fig_dip = px.line(df_dip, x=df_dip.index, y="Precio_Oro", title="📉 Precio del Oro - Estrategia Buy the Dip")
    for fecha, precio in señales_compra:
        fig_dip.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="green", size=10, symbol="arrow-up"), name="Compra")
    for fecha, precio in señales_venta:
        fig_dip.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="red", size=10, symbol="arrow-down"), name="Venta")
    st.plotly_chart(fig_dip, use_container_width=True)

    # 📊 Métricas de rendimiento
    st.markdown("### 📈 Resultados del Backtesting - Estrategia Buy the Dip")

    ret_dip = (df_dip["Capital"].iloc[-1] / df_dip["Capital"].iloc[0] - 1) * 100
    ret_hold = (df["Precio_Oro"].iloc[-1] / df["Precio_Oro"].iloc[0] - 1) * 100

    ret_diario_dip = df_dip["Capital"].pct_change().dropna()
    ret_diario_hold = df["Precio_Oro"].pct_change().dropna()

    vol_dip = ret_diario_dip.std() * (252**0.5) * 100
    vol_hold = ret_diario_hold.std() * (252**0.5) * 100

    sharpe_dip = (ret_diario_dip.mean() / ret_diario_dip.std()) * (252**0.5)
    sharpe_hold = (ret_diario_hold.mean() / ret_diario_hold.std()) * (252**0.5)

    st.markdown(f"- 🟢 **Rentabilidad Estrategia**: `{ret_dip:.2f}%`")
    st.markdown(f"- 🔵 **Rentabilidad Buy & Hold**: `{ret_hold:.2f}%`")
    st.markdown(f"- 🟢 **Volatilidad Estrategia**: `{vol_dip:.2f}%`")
    st.markdown(f"- 🔵 **Volatilidad Buy & Hold**: `{vol_hold:.2f}%`")
    st.markdown(f"- 🟢 **Sharpe Ratio Estrategia**: `{sharpe_dip:.2f}`")
    st.markdown(f"- 🔵 **Sharpe Ratio Buy & Hold**: `{sharpe_hold:.2f}`")
    # 📌 📊 **Sección: Backtesting - Estrategia de Momentum**
    st.subheader("📊 Backtesting: Estrategia de Momentum")

    df_momentum = df.copy()
    df_momentum["Momentum"] = ta.momentum.ROCIndicator(close=df_momentum["Precio_Oro"], window=10).roc()

    # Generar señales: Comprar cuando Momentum > 0, Vender cuando < 0
    df_momentum["Señal"] = 0
    df_momentum.loc[df_momentum["Momentum"] > 0, "Señal"] = 1
    df_momentum.loc[df_momentum["Momentum"] < 0, "Señal"] = -1

    capital = 10000
    en_posicion = False
    señales_compra = []
    señales_venta = []
    capitales = []

    for fecha, fila in df_momentum.iterrows():
        precio = fila["Precio_Oro"]
        señal = fila["Señal"]

        if señal == 1 and not en_posicion:
            unidades = capital / precio
            en_posicion = True
            señales_compra.append((fecha, precio))
        elif señal == -1 and en_posicion:
            capital = unidades * precio
            en_posicion = False
            señales_venta.append((fecha, precio))

        capitales.append(capital if not en_posicion else unidades * precio)

    df_momentum["Capital"] = capitales

    # 📉 Gráfico con señales
    fig_momentum = px.line(df_momentum, x=df_momentum.index, y="Precio_Oro", title="📉 Precio del Oro - Estrategia Momentum")
    for fecha, precio in señales_compra:
        fig_momentum.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="green", size=10, symbol="arrow-up"), name="Compra")
    for fecha, precio in señales_venta:
        fig_momentum.add_scatter(x=[fecha], y=[precio], mode="markers", marker=dict(color="red", size=10, symbol="arrow-down"), name="Venta")
    st.plotly_chart(fig_momentum, use_container_width=True)

    # 📊 Métricas de rendimiento
    st.markdown("### 📈 Resultados del Backtesting - Estrategia Momentum")

    ret_mom = (df_momentum["Capital"].iloc[-1] / df_momentum["Capital"].iloc[0] - 1) * 100
    ret_hold = (df["Precio_Oro"].iloc[-1] / df["Precio_Oro"].iloc[0] - 1) * 100

    ret_diario_mom = df_momentum["Capital"].pct_change().dropna()
    ret_diario_hold = df["Precio_Oro"].pct_change().dropna()

    vol_mom = ret_diario_mom.std() * (252**0.5) * 100
    vol_hold = ret_diario_hold.std() * (252**0.5) * 100

    sharpe_mom = (ret_diario_mom.mean() / ret_diario_mom.std()) * (252**0.5)
    sharpe_hold = (ret_diario_hold.mean() / ret_diario_hold.std()) * (252**0.5)

    st.markdown(f"- 🟢 **Rentabilidad Estrategia**: `{ret_mom:.2f}%`")
    st.markdown(f"- 🔵 **Rentabilidad Buy & Hold**: `{ret_hold:.2f}%`")
    st.markdown(f"- 🟢 **Volatilidad Estrategia**: `{vol_mom:.2f}%`")
    st.markdown(f"- 🔵 **Volatilidad Buy & Hold**: `{vol_hold:.2f}%`")
    st.markdown(f"- 🟢 **Sharpe Ratio Estrategia**: `{sharpe_mom:.2f}`")
    st.markdown(f"- 🔵 **Sharpe Ratio Buy & Hold**: `{sharpe_hold:.2f}`")
    # 📌 📊 **Sección 7: Estrategia de Volatilidad (Low Volatility Breakout)**
    st.subheader("📊 Estrategia de Volatilidad: Low Volatility Breakout")

    df_vol = df.copy()
    df_vol["Retorno Diario"] = df_vol["Precio_Oro"].pct_change()
    df_vol["Volatilidad 10d"] = df_vol["Retorno Diario"].rolling(window=10).std()
    df_vol["Max_20d"] = df_vol["Precio_Oro"].rolling(window=20).max()

    # Señal de compra: baja volatilidad y ruptura al alza del máximo de 20 días
    df_vol["Signal"] = (df_vol["Volatilidad 10d"] < df_vol["Volatilidad 10d"].rolling(20).mean()) & \
                       (df_vol["Precio_Oro"] > df_vol["Max_20d"].shift(1))
    df_vol["Position"] = df_vol["Signal"].shift(1).fillna(False).astype(int)

    # Backtesting
    df_vol["BuyHold"] = df_vol["Precio_Oro"] / df_vol["Precio_Oro"].iloc[0]
    df_vol["Strategy"] = df_vol["Precio_Oro"].pct_change().fillna(0) * df_vol["Position"]
    df_vol["StrategyValue"] = (1 + df_vol["Strategy"]).cumprod()

    # Fechas para mostrar resumen
    fecha_inicio = df_vol.index.min().date()
    fecha_fin = df_vol.index.max().date()

    st.markdown(f"### 📈 Rentabilidad final desde {fecha_inicio} hasta {fecha_fin}")
    ret_estrategia = (df_vol["StrategyValue"].iloc[-1] - 1) * 100
    ret_hold = (df_vol["BuyHold"].iloc[-1] - 1) * 100
    st.markdown(f"- 🟢 **Estrategia Volatilidad**: `{ret_estrategia:.2f}%`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{ret_hold:.2f}%`")

    # 📊 Gráfico de evolución
    fig_vola = px.line(
        df_vol[["BuyHold", "StrategyValue"]],
        labels={"value": "Índice de Inversión", "variable": "Estrategia"},
        title="Evolución del Valor: Volatility Breakout vs Buy & Hold"
    )
    st.plotly_chart(fig_vola, use_container_width=True)

    # 🔍 Visualización de señales
    st.markdown("### 🟢 Señales de Compra")

    fig_señales = px.line(df_vol, x=df_vol.index, y="Precio_Oro", title="Señales de Compra en Precio del Oro")
    fig_señales.add_scatter(
        x=df_vol[df_vol["Signal"]].index,
        y=df_vol[df_vol["Signal"]]["Precio_Oro"],
        mode="markers",
        marker=dict(symbol="arrow-bar-up", size=10, color="green"),
        name="Señal de Compra"
    )
    fig_señales.update_layout(xaxis_title="Fecha", yaxis_title="Precio (USD)")
    st.plotly_chart(fig_señales, use_container_width=True)
    # 📌 📊 **Sección 8: Estrategia Basada en Eventos Macroeconómicos**
    st.subheader("🌍 Estrategia basada en Eventos Macroeconómicos")

    df_eventos = df.copy()
    df_eventos["Retorno Diario"] = df_eventos["Precio_Oro"].pct_change()
    df_eventos["Estrategia"] = 0

    # Aplicar lógica: comprar 1 día antes del evento, vender a los 3 días (ajustable)
    dias_holding = 3
    for _, evento in eventos_df.iterrows():
        fecha_entrada = evento["Fecha"] - pd.Timedelta(days=1)
        fecha_salida = evento["Fecha"] + pd.Timedelta(days=dias_holding)

        if fecha_entrada in df_eventos.index and fecha_salida in df_eventos.index:
            df_eventos.loc[fecha_entrada:fecha_salida, "Estrategia"] = 1

    # Calcular estrategia de retorno acumulado
    df_eventos["BuyHold"] = df_eventos["Precio_Oro"] / df_eventos["Precio_Oro"].iloc[0]
    df_eventos["Retorno_Estrategia"] = df_eventos["Precio_Oro"].pct_change().fillna(0) * df_eventos["Estrategia"]
    df_eventos["Valor_Estrategia"] = (1 + df_eventos["Retorno_Estrategia"]).cumprod()

    # Resultados
    fecha_inicio = df_eventos.index.min().date()
    fecha_fin = df_eventos.index.max().date()
    st.markdown(f"### 📈 Rentabilidad final desde {fecha_inicio} hasta {fecha_fin}")
    ret_estrategia = (df_eventos["Valor_Estrategia"].iloc[-1] - 1) * 100
    ret_hold = (df_eventos["BuyHold"].iloc[-1] - 1) * 100
    st.markdown(f"- 🟢 **Estrategia Eventos**: `{ret_estrategia:.2f}%`")
    st.markdown(f"- 🔵 **Holding Pasivo**: `{ret_hold:.2f}%`")

    # 📊 Gráfico de comparación
    fig_eventos = px.line(
        df_eventos[["BuyHold", "Valor_Estrategia"]],
        labels={"value": "Índice de Inversión", "variable": "Estrategia"},
        title="Estrategia basada en Eventos vs Buy & Hold"
    )
    st.plotly_chart(fig_eventos, use_container_width=True)

    # 🔍 Visualización de señales
    st.markdown("### 🟢 Señales de Compra basadas en Eventos")

    fig_senales_eventos = px.line(df_eventos, x=df_eventos.index, y="Precio_Oro", title="Señales de Compra por Eventos Macroeconómicos")
    fechas_entrada = [f - pd.Timedelta(days=1) for f in eventos_df["Fecha"] if f - pd.Timedelta(days=1) in df_eventos.index]

    fig_senales_eventos.add_scatter(
        x=fechas_entrada,
        y=df_eventos.loc[fechas_entrada, "Precio_Oro"],
        mode="markers",
        marker=dict(symbol="arrow-bar-up", size=10, color="orange"),
        name="Compra por Evento"
    )
    fig_senales_eventos.update_layout(xaxis_title="Fecha", yaxis_title="Precio del Oro (USD)")
    st.plotly_chart(fig_senales_eventos, use_container_width=True)
    # 📌 📊 **Sección: Reacción del Oro ante Crisis Geopolíticas**
    st.subheader("🌍 Reacción del Oro ante Crisis Geopolíticas")

    # Filtrar eventos geopolíticos
    geo_df = eventos_df[eventos_df["Categoría"] == "Geopolítico"].copy()

    # Fijar ventana +/- 15 días
    dias_ventana = 15
    df_geo = df[["Precio_Oro"]].copy()
    df_geo = df_geo.reset_index()

    # Gráfico de línea con anotaciones
    fig_geo = px.line(df_geo, x="Fecha", y="Precio_Oro", title="📌 Precio del Oro y Eventos Geopolíticos")
    for _, row in geo_df.iterrows():
        if row["Fecha"] in df_geo["Fecha"].values:
            fig_geo.add_annotation(
                x=row["Fecha"],
                y=df_geo[df_geo["Fecha"] == row["Fecha"]]["Precio_Oro"].values[0],
                text=row["Evento"],
                showarrow=True,
                arrowhead=2,
                bgcolor="orange",
                font=dict(size=10),
                yshift=10
            )

    st.plotly_chart(fig_geo, use_container_width=True)
    st.subheader("📉 Análisis Cuantitativo: Retornos Antes y Después de Eventos Geopolíticos")

    # Ventana de análisis (5 días antes y después)
    ventana = 5
    resultados_geo = []

    for _, evento in geo_df.iterrows():
        fecha = evento["Fecha"]

        precio_antes = df[(df.index >= fecha - pd.Timedelta(days=ventana)) & (df.index < fecha)]["Precio_Oro"]
        precio_despues = df[(df.index > fecha) & (df.index <= fecha + pd.Timedelta(days=ventana))]["Precio_Oro"]

        if not precio_antes.empty and not precio_despues.empty:
            retorno_antes = (precio_antes.iloc[-1] / precio_antes.iloc[0] - 1) * 100
            retorno_despues = (precio_despues.iloc[-1] / precio_despues.iloc[0] - 1) * 100
        else:
            retorno_antes, retorno_despues = None, None

        resultados_geo.append({
            "Evento": evento["Evento"],
            "Fecha": fecha.date(),
            "Retorno -5 a 0 días (%)": round(retorno_antes, 2) if retorno_antes is not None else "N/A",
            "Retorno 0 a +5 días (%)": round(retorno_despues, 2) if retorno_despues is not None else "N/A",
        })

    # Mostrar resultados en tabla
    df_retornos_geo = pd.DataFrame(resultados_geo)
    st.dataframe(df_retornos_geo)
    
    st.subheader("📊 Detección de Ráfagas de Volumen en el Oro")

    # Asegurarse que hay columna de volumen
    if "Volumen" not in df.columns:
        st.warning("⚠️ El dataset no contiene una columna 'Volumen'. Esta sección requiere datos de volumen.")
    else:
        # Calcular umbral: volumen superior al percentil 95 como ráfaga
        umbral_volumen = df["Volumen"].quantile(0.95)
        df["Ráfaga Volumen"] = df["Volumen"] > umbral_volumen

        # Gráfico de línea con barras de volumen
        fig_volumen = go.Figure()

        # Precio Oro
        fig_volumen.add_trace(go.Scatter(
            x=df.index,
            y=df["Precio_Oro"],
            mode='lines',
            name='Precio Oro',
            line=dict(color='gold')
        ))

        # Volumen como barras
        fig_volumen.add_trace(go.Bar(
            x=df.index,
            y=df["Volumen"],
            name='Volumen',
            yaxis='y2',
            marker_color='lightblue',
            opacity=0.5
        ))

        # Añadir markers para ráfagas
        rafagas = df[df["Ráfaga Volumen"]]
        fig_volumen.add_trace(go.Scatter(
            x=rafagas.index,
            y=rafagas["Precio_Oro"],
            mode='markers',
            name='Ráfaga de Volumen',
            marker=dict(size=8, color='red', symbol='x'),
        ))

        # Layout con doble eje Y
        fig_volumen.update_layout(
            title="Ráfagas de Volumen y Precio del Oro",
            xaxis_title="Fecha",
            yaxis=dict(title="Precio Oro (USD)", side='left'),
            yaxis2=dict(title="Volumen", overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_volumen, use_container_width=True)

        # Tabla de eventos detectados
        st.markdown("### 📋 Eventos con Ráfagas de Volumen")
        st.dataframe(rafagas[["Precio_Oro", "Volumen"]])
    # 📌 📊 Asociación entre Ráfagas de Volumen y Eventos Macroeconómicos
       # 📌 📊 Asociación entre Ráfagas de Volumen y Eventos Macroeconómicos
    st.subheader("📎 Asociación entre Ráfagas de Volumen y Eventos Macroeconómicos")

    volumen_eventos = []
    for fecha in rafagas.index:
        eventos_cercanos = eventos_df[
            (eventos_df["Fecha"] >= fecha - pd.Timedelta(days=3)) &
            (eventos_df["Fecha"] <= fecha + pd.Timedelta(days=3))
        ]
        for _, evento in eventos_cercanos.iterrows():
            volumen_eventos.append({
                "Fecha Ráfaga": fecha.date(),
                "Volumen": rafagas.loc[fecha, "Volumen"],
                "Evento": evento["Evento"],
                "Fecha Evento": evento["Fecha"].date(),
                "Categoría": evento.get("Categoría", "Sin categoría")
            })

    if volumen_eventos:
        df_vol_eventos = pd.DataFrame(volumen_eventos)
        st.dataframe(df_vol_eventos.sort_values("Fecha Ráfaga"))
    else:
        st.info("No se encontraron eventos macroeconómicos cercanos a las ráfagas de volumen detectadas.")

  

    # 🔙 **Botón para volver**
    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
