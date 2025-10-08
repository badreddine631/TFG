# =====================
# Librerías de sistema / datos
# =====================
import pandas as pd
import numpy as np

# =====================
# Visualización
# =====================
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================
# Indicadores técnicos
# =====================
import ta
from ta.utils import dropna
from ta.trend import SMAIndicator
from ta.momentum import ROCIndicator

# =====================
# Machine Learning
# =====================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# =====================
# Señales y matemáticas
# =====================
from scipy.signal import argrelextrema
from scipy.signal import correlate
from numpy import arange
from statsmodels.tsa.stattools import grangercausalitytests




def show(df, eventos_df, dolar_df, df_oro_ext, df_btc,df_plata, df_oro_eur, df_oro_cny, df_petroleo, df_cad, df_elecciones_usa, df_recesiones, df_crisis_europa, df_crisis_inmo,df_crisis_minerales, cpi_df, df_oro_mensual, cci_df, fear_greed_df, fed_df, pib_df ):
    """Muestra la sección de Información Basada en Datos con selección dinámica."""
    st.title("📉 Información Basada en Datos")

    tabs = st.tabs([
        "📊 Volatilidad", 
        "📈 Tendencias", 
        "🔄 Estrategias", 
        "⚔️ Eventos", 
        "🧪 Cuantitativo", 
        "🧠 Influencias", 
        "📉 Técnicos Avanzados"
    ])

    # 📊 Volatilidad
    with tabs[0]:
        st.header("📊 Volatilidad")
        opciones = [
            "Volatilidad Antes y Después de los Eventos",
            "Volatilidad Mensual del Precio del Oro",
            "Detección de Ráfagas de Volumen en el Oro",
            "Asociación entre Ráfagas de Volumen y Eventos Macroeconómicos",
            "Estrategia de Volatilidad: Low Volatility Breakout",
            "Estudio de Clústeres de Volatilidad en el Oro "
        ]
        seleccion = st.selectbox("Selecciona un análisis de volatilidad", opciones)

        if seleccion == opciones[0]:
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
            
        elif seleccion == opciones[1]:
            # 📌 📊 **Sección 2: Volatilidad Mensual del Oro**
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
    
        elif seleccion == opciones[2]:
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
            
        elif seleccion == opciones[3]:
            st.subheader("📎 Asociación entre Ráfagas de Volumen y Eventos Macroeconómicos")

            if "Volumen" not in df.columns:
                st.warning("⚠️ El dataset no contiene una columna 'Volumen'. Esta sección requiere datos de volumen.")
            else:
                umbral_volumen = df["Volumen"].quantile(0.95)
                df["Ráfaga Volumen"] = df["Volumen"] > umbral_volumen
                rafagas = df[df["Ráfaga Volumen"]]

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


        
        elif seleccion == opciones[4]:
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
    
        elif seleccion == opciones[5]:

            st.subheader("🤖 Estudio de Clústeres de Volatilidad en el Oro")

            df_cluster = df.copy()
            df_cluster["Retorno Diario"] = df_cluster["Precio_Oro"].pct_change()
            df_cluster["Volatilidad 7d"] = df_cluster["Retorno Diario"].rolling(window=7).std()
            df_cluster = df_cluster.dropna().copy()

            # 🔍 Variables para clustering
            X_cluster = df_cluster[["Retorno Diario", "Volatilidad 7d"]].copy()

            # 🔄 Estandarizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            # 🤖 K-Means Clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            df_cluster["Cluster"] = clusters

            # 🎨 Visualización con PCA (2D)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            df_cluster["PCA1"] = pca_result[:, 0]
            df_cluster["PCA2"] = pca_result[:, 1]

            fig_clusters = px.scatter(
                df_cluster,
                x="PCA1",
                y="PCA2",
                color="Cluster",
                title="Agrupación de Clústeres de Volatilidad (PCA)",
                hover_data=["Retorno Diario", "Volatilidad 7d", "Precio_Oro"]
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

            # 📈 Evolución temporal por Clúster
            fig_line_cluster = px.scatter(
                df_cluster,
                x=df_cluster.index,
                y="Volatilidad 7d",
                color="Cluster",
                title="Volatilidad 7d por Clúster a lo largo del Tiempo",
                labels={"Volatilidad 7d": "Volatilidad (7 días)", "index": "Fecha"}
            )
            st.plotly_chart(fig_line_cluster, use_container_width=True)

            # 🧠 Resumen de cada clúster
            resumen = df_cluster.groupby("Cluster")[["Retorno Diario", "Volatilidad 7d"]].agg(["mean", "std", "count"])
            st.markdown("### 📋 Resumen Estadístico por Clúster")
            st.dataframe(resumen)


    # 📈 Tendencias
    with tabs[1]:
        st.header("📈 Tendencias")
        opciones_trend = [
            "Media Móvil del Precio del Oro",
            "Correlación entre el Precio del Oro y el Dólar (DXY)",
            "Comparación entre Oro y Bitcoin",
            "Estacionalidad y Retorno Anual del Precio del Oro",
            "Patrones de Velas Japonesas (estimados)",
            "Comparativa Oro vs Plata ",
            "Comparación del Oro con el Euro (XAU/EUR) ",
            "Comparativa entre Oro y Yuan Chino (CNY) ",
            "Análisis del Ratio Oro/Petróleo ",
            "Análisis del Ratio Oro/Dólar Canadiense (XAU/CAD)",
            "Análisis de Fibonacci en el Precio del Oro",
            "Estudio de Ruptura de Rango Lateral",
            "Análisis de Breakouts del Precio del Oro",
            "Evolución del Precio del Oro en el Fin de Semana (Gaps de Apertura)"
        ]
        seleccion_trend = st.selectbox("Selecciona un análisis de tendencia", opciones_trend)
        
        if seleccion_trend == opciones_trend[0]:
            # 📌 📊 **Sección 1: Media Móvil del Precio del Oro**
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
            
        elif seleccion_trend == opciones_trend[1]:
            # 📌 📊 **Sección 2: Correlación entre Oro y DXY**
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
    
        elif seleccion_trend == opciones_trend[2]:
             # 📌 📊 **Sección 3: Comparación entre Oro y Bitcoin**
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

        elif seleccion_trend == opciones_trend[3]:
            # 📌 📊 **Sección 4: Estacionalidad y Retorno Anual**
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
    
        elif seleccion_trend  == opciones_trend[4]:
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
    
        elif seleccion_trend == opciones_trend[5]:
                    # 📌 📊 Comparativa Oro vs Plata
            st.subheader("📈 Comparativa entre Oro y Plata")

            # Selección de fechas comunes
            start_ovp = pd.to_datetime(
                st.sidebar.date_input("Fecha de inicio comparación Oro vs Plata", value=df.index.min(), key="fecha_inicio_ovp")
            )
            end_ovp = pd.to_datetime(
                st.sidebar.date_input("Fecha de fin comparación Oro vs Plata", value=df.index.max(), key="fecha_fin_ovp")
            )

            # Filtrar ambos datasets por fechas
            oro_filtrado = df[(df.index >= start_ovp) & (df.index <= end_ovp)].copy()
            plata_filtrada = df_plata[(df_plata.index >= start_ovp) & (df_plata.index <= end_ovp)].copy()

            # Unir por índice (fecha)
            merged_ovp = pd.merge(oro_filtrado[["Precio_Oro"]], plata_filtrada[["Precio_Plata"]], left_index=True, right_index=True, how="inner")

            # Eliminar nulos si hay
            merged_ovp.dropna(inplace=True)

            # 📊 Precio absoluto
            fig_abs_ovp = px.line(
                merged_ovp,
                x=merged_ovp.index,
                y=["Precio_Oro", "Precio_Plata"],
                labels={"value": "Precio (USD)", "variable": "Activo"},
                title="📉 Evolución del Precio Absoluto: Oro vs Plata"
            )
            st.plotly_chart(fig_abs_ovp, use_container_width=True)

            # 📊 Evolución Relativa (Base 100)
            merged_ovp["Oro_Base100"] = merged_ovp["Precio_Oro"] / merged_ovp["Precio_Oro"].iloc[0] * 100
            merged_ovp["Plata_Base100"] = merged_ovp["Precio_Plata"] / merged_ovp["Precio_Plata"].iloc[0] * 100

            fig_rel_ovp = px.line(
                merged_ovp,
                x=merged_ovp.index,
                y=["Oro_Base100", "Plata_Base100"],
                labels={"value": "Índice Base 100", "variable": "Activo"},
                title="📈 Evolución Relativa (Base 100): Oro vs Plata"
            )
            st.plotly_chart(fig_rel_ovp, use_container_width=True)

            # 📌 Correlación
            corr_ovp = merged_ovp[["Precio_Oro", "Precio_Plata"]].corr().iloc[0, 1]
            st.markdown(f"📌 **Correlación entre Oro y Plata**: `{corr_ovp:.4f}`")

            # 📌 Rentabilidad
            ret_oro = (merged_ovp["Precio_Oro"].iloc[-1] / merged_ovp["Precio_Oro"].iloc[0] - 1) * 100
            ret_plata = (merged_ovp["Precio_Plata"].iloc[-1] / merged_ovp["Precio_Plata"].iloc[0] - 1) * 100

            st.markdown("### 📊 Rentabilidad Acumulada")
            st.markdown(f"- 🟡 **Oro**: `{ret_oro:.2f}%`")
            st.markdown(f"- ⚪ **Plata**: `{ret_plata:.2f}%`")

            # 📌 Volatilidad diaria
            vol_oro = merged_ovp["Precio_Oro"].pct_change().std() * 100
            vol_plata = merged_ovp["Precio_Plata"].pct_change().std() * 100

            st.markdown("### 📉 Volatilidad Histórica Diaria")
            st.markdown(f"- 🟡 **Oro**: `{vol_oro:.2f}%`")
            st.markdown(f"- ⚪ **Plata**: `{vol_plata:.2f}%`")

            # 📌 Drawdown
            oro_drawdown = (merged_ovp["Precio_Oro"] / merged_ovp["Precio_Oro"].cummax() - 1).min() * 100
            plata_drawdown = (merged_ovp["Precio_Plata"] / merged_ovp["Precio_Plata"].cummax() - 1).min() * 100

            st.markdown("### 📉 Caída Máxima (Drawdown)")
            st.markdown(f"- 🟡 **Oro**: `{oro_drawdown:.2f}%`")
            st.markdown(f"- ⚪ **Plata**: `{plata_drawdown:.2f}%`")

            
        elif seleccion_trend == opciones_trend[6]:
            st.subheader("💶 Comparación del Oro en USD vs EUR")

            # Selección de fechas comunes
            start_eur = pd.to_datetime(
                st.sidebar.date_input("Fecha inicio comparación USD vs EUR", value=max(df.index.min(), df_oro_eur.index.min()), key="eur_fecha_inicio")
            )
            end_eur = pd.to_datetime(
                st.sidebar.date_input("Fecha fin comparación USD vs EUR", value=min(df.index.max(), df_oro_eur.index.max()), key="eur_fecha_fin")
            )

            # Filtrar ambos DataFrames por fechas comunes
            df_usd = df[(df.index >= start_eur) & (df.index <= end_eur)][["Precio_Oro"]].copy()
            df_eur = df_oro_eur[(df_oro_eur.index >= start_eur) & (df_oro_eur.index <= end_eur)][["Precio_Oro_EUR"]].copy()

            # Fusionar por índice (fecha)
            df_comp_eur = df_usd.merge(df_eur, left_index=True, right_index=True, how="inner")

            # 📊 Gráfico interactivo de precios absolutos
            fig_abs_eur = px.line(
                df_comp_eur,
                x=df_comp_eur.index,
                y=["Precio_Oro", "Precio_Oro_EUR"],
                labels={"value": "Precio", "variable": "Divisa"},
                title="📉 Precio del Oro: USD vs EUR"
            )
            st.plotly_chart(fig_abs_eur, use_container_width=True)

            # 📊 Gráfico normalizado (Base 100)
            df_comp_eur["Oro_USD_norm"] = df_comp_eur["Precio_Oro"] / df_comp_eur["Precio_Oro"].iloc[0] * 100
            df_comp_eur["Oro_EUR_norm"] = df_comp_eur["Precio_Oro_EUR"] / df_comp_eur["Precio_Oro_EUR"].iloc[0] * 100

            fig_norm_eur = px.line(
                df_comp_eur,
                x=df_comp_eur.index,
                y=["Oro_USD_norm", "Oro_EUR_norm"],
                labels={"value": "Índice (Base 100)", "variable": "Divisa"},
                title="📈 Evolución Relativa: Oro en USD vs EUR (Base 100)"
            )
            st.plotly_chart(fig_norm_eur, use_container_width=True)

            # Mostrar rentabilidades
            rent_usd = (df_comp_eur["Precio_Oro"].iloc[-1] / df_comp_eur["Precio_Oro"].iloc[0] - 1) * 100
            rent_eur = (df_comp_eur["Precio_Oro_EUR"].iloc[-1] / df_comp_eur["Precio_Oro_EUR"].iloc[0] - 1) * 100

            st.markdown("### 📊 Rentabilidad acumulada en el periodo seleccionado")
            st.markdown(f"- 🇺🇸 **USD**: `{rent_usd:.2f}%`")
            st.markdown(f"- 🇪🇺 **EUR**: `{rent_eur:.2f}%`")

            # Mostrar tabla de datos finales
            st.markdown("### 📋 Tabla de Datos")
            st.dataframe(df_comp_eur.tail(10))
            
        elif seleccion_trend == opciones_trend[7]:
            st.subheader("💴 Comparativa entre el Oro en USD y en Yuan Chino (CNY)")

            # Selección de fechas comunes
            start_cny = pd.to_datetime(
                st.sidebar.date_input("Fecha inicio comparación USD vs CNY", value=max(df.index.min(), df_oro_cny.index.min()), key="cny_fecha_inicio")
            )
            end_cny = pd.to_datetime(
                st.sidebar.date_input("Fecha fin comparación USD vs CNY", value=min(df.index.max(), df_oro_cny.index.max()), key="cny_fecha_fin")
            )

            # Filtrar por fechas
            df_usd_cny = df[(df.index >= start_cny) & (df.index <= end_cny)][["Precio_Oro"]].copy()
            df_cny = df_oro_cny[(df_oro_cny.index >= start_cny) & (df_oro_cny.index <= end_cny)][["Precio_Oro_CNY"]].copy()

            # Merge
            df_comp_cny = df_usd_cny.merge(df_cny, left_index=True, right_index=True, how="inner")

            # 📊 Gráfico de precios absolutos
            fig_abs_cny = px.line(
                df_comp_cny,
                x=df_comp_cny.index,
                y=["Precio_Oro", "Precio_Oro_CNY"],
                labels={"value": "Precio", "variable": "Divisa"},
                title="📉 Precio del Oro: USD vs Yuan Chino (CNY)"
            )
            st.plotly_chart(fig_abs_cny, use_container_width=True)

            # Normalización base 100
            df_comp_cny["Oro_USD_norm"] = df_comp_cny["Precio_Oro"] / df_comp_cny["Precio_Oro"].iloc[0] * 100
            df_comp_cny["Oro_CNY_norm"] = df_comp_cny["Precio_Oro_CNY"] / df_comp_cny["Precio_Oro_CNY"].iloc[0] * 100

            fig_norm_cny = px.line(
                df_comp_cny,
                x=df_comp_cny.index,
                y=["Oro_USD_norm", "Oro_CNY_norm"],
                labels={"value": "Índice Base 100", "variable": "Divisa"},
                title="📈 Evolución Relativa: Oro en USD vs Yuan CNY (Base 100)"
            )
            st.plotly_chart(fig_norm_cny, use_container_width=True)

            # Rentabilidad acumulada
            rent_usd_cny = (df_comp_cny["Precio_Oro"].iloc[-1] / df_comp_cny["Precio_Oro"].iloc[0] - 1) * 100
            rent_cny = (df_comp_cny["Precio_Oro_CNY"].iloc[-1] / df_comp_cny["Precio_Oro_CNY"].iloc[0] - 1) * 100

            st.markdown("### 📊 Rentabilidad acumulada en el periodo seleccionado")
            st.markdown(f"- 🇺🇸 **USD**: `{rent_usd_cny:.2f}%`")
            st.markdown(f"- 🇨🇳 **Yuan (CNY)**: `{rent_cny:.2f}%`")

            # Mostrar tabla resumen
            st.markdown("### 📋 Últimos valores")
            st.dataframe(df_comp_cny.tail(10))

        elif seleccion_trend == opciones_trend[8]:
                
            st.subheader("⛽ Análisis del Ratio Oro/Petróleo")

            # Selección de fechas comunes
            start_oil = pd.to_datetime(
                st.sidebar.date_input("Fecha de inicio (Ratio Oro/Petróleo)", value=df.index.min(), key="inicio_ratio_oil")
            )
            end_oil = pd.to_datetime(
                st.sidebar.date_input("Fecha de fin (Ratio Oro/Petróleo)", value=df.index.max(), key="fin_ratio_oil")
            )

            # Filtrar ambos datasets
            oro_filtrado = df[["Precio_Oro"]].loc[start_oil:end_oil].copy()
            petroleo_filtrado = df_petroleo.loc[start_oil:end_oil].copy()

            # Unir por fecha
            merged_ratio = oro_filtrado.merge(petroleo_filtrado, left_index=True, right_index=True, how="inner")

            # Calcular ratio
            merged_ratio["Ratio Oro/Petróleo"] = merged_ratio["Precio_Oro"] / merged_ratio["Precio_Petroleo"]

            # 📊 Gráfico del ratio
            fig_ratio = px.line(
                merged_ratio,
                x=merged_ratio.index,
                y="Ratio Oro/Petróleo",
                title="📊 Evolución del Ratio Oro/Petróleo",
                labels={"Ratio Oro/Petróleo": "Ratio (XAU/WTI)", "Fecha": "Fecha"},
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

            # 📌 Estadísticas adicionales
            st.markdown("### 📈 Estadísticas del Ratio")
            st.write(f"- Máximo histórico: `{merged_ratio['Ratio Oro/Petróleo'].max():.2f}`")
            st.write(f"- Mínimo histórico: `{merged_ratio['Ratio Oro/Petróleo'].min():.2f}`")
            st.write(f"- Media: `{merged_ratio['Ratio Oro/Petróleo'].mean():.2f}`")

        elif seleccion_trend == opciones_trend[9]:
            st.subheader("💹 Análisis del Ratio Oro/Dólar Canadiense (XAU/CAD)")

            # Selección de fechas
            start_cad = pd.to_datetime(
                st.sidebar.date_input("Fecha de inicio XAU/USD vs XAU/CAD", value=df.index.min(), key="cad_fecha_inicio")
            )
            end_cad = pd.to_datetime(
                st.sidebar.date_input("Fecha de fin XAU/USD vs XAU/CAD", value=df.index.max(), key="cad_fecha_fin")
            )

            # Filtrar datasets
            oro_usd = df[(df.index >= start_cad) & (df.index <= end_cad)][["Precio_Oro"]].copy()
            oro_cad = df_cad[(df_cad.index >= start_cad) & (df_cad.index <= end_cad)][["Precio_Oro_CAD"]].copy()

            # Unir por fecha
            df_cmp_cad = pd.merge(oro_usd, oro_cad, left_index=True, right_index=True, how="inner")

            # Eliminar valores faltantes
            df_cmp_cad = df_cmp_cad.dropna()

            # 📊 Gráfico de precios absolutos
            fig_cad_abs = px.line(
                df_cmp_cad,
                x=df_cmp_cad.index,
                y=["Precio_Oro", "Precio_Oro_CAD"],
                labels={"value": "Precio", "variable": "Activo"},
                title="💰 Precio del Oro en USD vs CAD"
            )
            st.plotly_chart(fig_cad_abs, use_container_width=True)

            # 📈 Normalizar ambos precios (Base 100)
            df_cmp_cad["Oro_USD_norm"] = df_cmp_cad["Precio_Oro"] / df_cmp_cad["Precio_Oro"].iloc[0] * 100
            df_cmp_cad["Oro_CAD_norm"] = df_cmp_cad["Precio_Oro_CAD"] / df_cmp_cad["Precio_Oro_CAD"].iloc[0] * 100

            # 📉 Gráfico normalizado
            fig_cad_norm = px.line(
                df_cmp_cad,
                x=df_cmp_cad.index,
                y=["Oro_USD_norm", "Oro_CAD_norm"],
                labels={"value": "Índice Base 100", "variable": "Moneda"},
                title="📈 Evolución Relativa del Oro: USD vs CAD (Base 100)"
            )
            st.plotly_chart(fig_cad_norm, use_container_width=True)

            # 📌 Estadísticas
            rentabilidad_usd = (df_cmp_cad["Precio_Oro"].iloc[-1] / df_cmp_cad["Precio_Oro"].iloc[0] - 1) * 100
            rentabilidad_cad = (df_cmp_cad["Precio_Oro_CAD"].iloc[-1] / df_cmp_cad["Precio_Oro_CAD"].iloc[0] - 1) * 100

            volatilidad_usd = df_cmp_cad["Precio_Oro"].pct_change().std() * 100
            volatilidad_cad = df_cmp_cad["Precio_Oro_CAD"].pct_change().std() * 100

            drawdown_usd = (df_cmp_cad["Precio_Oro"] / df_cmp_cad["Precio_Oro"].cummax() - 1).min() * 100
            drawdown_cad = (df_cmp_cad["Precio_Oro_CAD"] / df_cmp_cad["Precio_Oro_CAD"].cummax() - 1).min() * 100

            # 📊 Mostrar resultados
            st.markdown("### 📊 Rentabilidad acumulada")
            st.markdown(f"- 🟡 **Oro en USD**: `{rentabilidad_usd:.2f}%`")
            st.markdown(f"- 🔵 **Oro en CAD**: `{rentabilidad_cad:.2f}%`")

            st.markdown("### 📉 Volatilidad histórica")
            st.markdown(f"- 🟡 **Oro en USD**: `{volatilidad_usd:.2f}%` diaria")
            st.markdown(f"- 🔵 **Oro en CAD**: `{volatilidad_cad:.2f}%` diaria")

            st.markdown("### 📉 Caída máxima (drawdown)")
            st.markdown(f"- 🟡 **Oro en USD**: `{drawdown_usd:.2f}%`")
            st.markdown(f"- 🔵 **Oro en CAD**: `{drawdown_cad:.2f}%`")

            # 📎 Correlación
            corr_usd_cad = df_cmp_cad[["Precio_Oro", "Precio_Oro_CAD"]].corr().iloc[0, 1]
            st.markdown(f"### 📎 Correlación entre XAU/USD y XAU/CAD: `{corr_usd_cad:.4f}`")

        elif seleccion_trend == opciones_trend[10]:
            st.subheader("📐 Análisis de Fibonacci en el Precio del Oro")

            # 📌 Selección de fechas para definir el rango del análisis
            start_fib = pd.to_datetime(
                st.sidebar.date_input("Fecha de inicio (Fibonacci)", value=df.index.min(), key="fib_start")
            )
            end_fib = pd.to_datetime(
                st.sidebar.date_input("Fecha de fin (Fibonacci)", value=df.index.max(), key="fib_end")
            )

            df_fib = df[(df.index >= start_fib) & (df.index <= end_fib)].copy()

            if df_fib.empty or len(df_fib) < 2:
                st.warning("⚠️ Selecciona un rango de fechas más amplio para el análisis de Fibonacci.")
            else:
                # 🔍 Detectar el mínimo y máximo en ese rango
                max_price = df_fib["Precio_Oro"].max()
                min_price = df_fib["Precio_Oro"].min()

                # 📈 Calcular los niveles de Fibonacci (comunes)
                diff = max_price - min_price
                levels = {
                    "0.0% (Soporte)": min_price,
                    "23.6%": max_price - 0.236 * diff,
                    "38.2%": max_price - 0.382 * diff,
                    "50.0%": max_price - 0.500 * diff,
                    "61.8%": max_price - 0.618 * diff,
                    "78.6%": max_price - 0.786 * diff,
                    "100.0% (Resistencia)": max_price
                }

                # 📊 Gráfico con niveles de Fibonacci
                fig_fib = go.Figure()

                # Línea de precio
                fig_fib.add_trace(go.Scatter(
                    x=df_fib.index, y=df_fib["Precio_Oro"],
                    mode="lines", name="Precio del Oro", line=dict(color="gold")
                ))

                # Líneas horizontales de Fibonacci
                for name, level in levels.items():
                    fig_fib.add_trace(go.Scatter(
                        x=[df_fib.index.min(), df_fib.index.max()],
                        y=[level, level],
                        mode="lines",
                        name=f"Nivel {name}",
                        line=dict(dash="dash", width=1)
                    ))

                fig_fib.update_layout(
                    title="🔢 Niveles de Retroceso de Fibonacci",
                    xaxis_title="Fecha",
                    yaxis_title="Precio del Oro (USD)",
                    legend_title="Líneas",
                    height=600
                )

                st.plotly_chart(fig_fib, use_container_width=True)

                # 🧾 Mostrar los valores de los niveles
                st.markdown("### 📊 Niveles de Fibonacci calculados:")
                for name, level in levels.items():
                    st.write(f"- `{name}`: `{level:.2f} USD`")

        elif seleccion_trend == opciones_trend[11]:
            st.subheader("📉 Estudio de Ruptura de Rango Lateral en el Precio del Oro")

            # Selección de rango de fechas
            start_rango = pd.to_datetime(
                st.sidebar.date_input("Fecha de inicio (Rango Lateral)", value=df.index.min(), key="rango_start")
            )
            end_rango = pd.to_datetime(
                st.sidebar.date_input("Fecha de fin (Rango Lateral)", value=df.index.max(), key="rango_end")
            )

            df_rango = df[(df.index >= start_rango) & (df.index <= end_rango)].copy()

            if df_rango.empty or len(df_rango) < 10:
                st.warning("📌 Selecciona un rango con suficientes datos.")
            else:
                # Calcular percentiles para definir el canal de rango
                lower_bound = df_rango["Precio_Oro"].quantile(0.25)
                upper_bound = df_rango["Precio_Oro"].quantile(0.75)

                # Detectar rupturas
                df_rango["Ruptura"] = "Dentro del Rango"
                df_rango.loc[df_rango["Precio_Oro"] > upper_bound, "Ruptura"] = "Ruptura Alcista"
                df_rango.loc[df_rango["Precio_Oro"] < lower_bound, "Ruptura"] = "Ruptura Bajista"

                # Contadores
                total_alcistas = (df_rango["Ruptura"] == "Ruptura Alcista").sum()
                total_bajistas = (df_rango["Ruptura"] == "Ruptura Bajista").sum()

                # Mostrar texto de resumen
                st.markdown(f"🔼 Total rupturas alcistas: `{total_alcistas}`")
                st.markdown(f"🔽 Total rupturas bajistas: `{total_bajistas}`")
                st.markdown(f"🎯 Rango lateral definido entre **{lower_bound:.2f} USD** y **{upper_bound:.2f} USD**")

                # 📊 Gráfico
                fig_rango = go.Figure()

                # Precio
                fig_rango.add_trace(go.Scatter(
                    x=df_rango.index,
                    y=df_rango["Precio_Oro"],
                    mode="lines",
                    name="Precio del Oro",
                    line=dict(color="gold")
                ))

                # Límites del rango
                fig_rango.add_trace(go.Scatter(
                    x=[df_rango.index.min(), df_rango.index.max()],
                    y=[upper_bound, upper_bound],
                    mode="lines",
                    name="Límite Superior",
                    line=dict(color="green", dash="dash")
                ))
                fig_rango.add_trace(go.Scatter(
                    x=[df_rango.index.min(), df_rango.index.max()],
                    y=[lower_bound, lower_bound],
                    mode="lines",
                    name="Límite Inferior",
                    line=dict(color="red", dash="dash")
                ))

                # Marcar rupturas
                df_alcista = df_rango[df_rango["Ruptura"] == "Ruptura Alcista"]
                df_bajista = df_rango[df_rango["Ruptura"] == "Ruptura Bajista"]

                fig_rango.add_trace(go.Scatter(
                    x=df_alcista.index,
                    y=df_alcista["Precio_Oro"],
                    mode="markers",
                    name="🔼 Ruptura Alcista",
                    marker=dict(color="green", size=8, symbol="triangle-up")
                ))

                fig_rango.add_trace(go.Scatter(
                    x=df_bajista.index,
                    y=df_bajista["Precio_Oro"],
                    mode="markers",
                    name="🔽 Ruptura Bajista",
                    marker=dict(color="red", size=8, symbol="triangle-down")
                ))

                fig_rango.update_layout(
                    title="📊 Rupturas del Rango Lateral en el Precio del Oro",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USD)",
                    height=600
                )

                st.plotly_chart(fig_rango, use_container_width=True)

                # Mostrar tabla de rupturas si el usuario quiere
                if st.checkbox("📋 Mostrar tabla de rupturas"):
                    st.dataframe(df_rango[df_rango["Ruptura"] != "Dentro del Rango"][["Precio_Oro", "Ruptura"]])

        elif seleccion_trend == opciones_trend[12]:
            st.subheader("🚀 Análisis de Breakouts del Precio del Oro")

            # Parámetros del usuario
            ventana_dias = st.sidebar.slider("Ventana para detectar máximos/mínimos previos (días)", 5, 60, 20, key="ventana_breakout")
            umbral_post_break = st.sidebar.slider("Umbral de confirmación post-breakout (%)", 0.5, 10.0, 2.0, step=0.1, key="umbral_breakout")
            dias_evaluacion = st.sidebar.slider("Días para evaluar el movimiento posterior", 3, 30, 7, key="dias_post_breakout")

            df_break = df.copy().dropna().sort_index()

            # Calcular máximos/mínimos móviles
            df_break["Max_Ant"] = df_break["Precio_Oro"].rolling(window=ventana_dias).max().shift(1)
            df_break["Min_Ant"] = df_break["Precio_Oro"].rolling(window=ventana_dias).min().shift(1)

            df_break["Breakout_Alcista"] = df_break["Precio_Oro"] > df_break["Max_Ant"]
            df_break["Breakout_Bajista"] = df_break["Precio_Oro"] < df_break["Min_Ant"]

            # Detectar breakouts
            df_break["Breakout"] = ""
            df_break.loc[df_break["Breakout_Alcista"], "Breakout"] = "Alcista"
            df_break.loc[df_break["Breakout_Bajista"], "Breakout"] = "Bajista"

            # Evaluar si el breakout fue válido o fallido
            resultados = []

            for fecha, fila in df_break[df_break["Breakout"] != ""].iterrows():
                try:
                    post_periodo = df_break.loc[fecha:].iloc[1:dias_evaluacion+1]
                    precio_inicio = fila["Precio_Oro"]

                    if fila["Breakout"] == "Alcista":
                        max_post = post_periodo["Precio_Oro"].max()
                        cambio_pct = (max_post - precio_inicio) / precio_inicio * 100
                        resultado = "Válido" if cambio_pct >= umbral_post_break else "Fallido"
                    else:
                        min_post = post_periodo["Precio_Oro"].min()
                        cambio_pct = (precio_inicio - min_post) / precio_inicio * 100
                        resultado = "Válido" if cambio_pct >= umbral_post_break else "Fallido"

                    resultados.append((fecha, fila["Breakout"], precio_inicio, cambio_pct, resultado))
                except:
                    continue

            df_resultados = pd.DataFrame(resultados, columns=["Fecha", "Tipo", "Precio_Inicio", "Cambio_%", "Resultado"])
            df_resultados.set_index("Fecha", inplace=True)

            # 📊 Gráfico
            fig_break = go.Figure()
            fig_break.add_trace(go.Scatter(
                x=df_break.index,
                y=df_break["Precio_Oro"],
                name="Precio del Oro",
                line=dict(color="gold")
            ))

            df_valido = df_resultados[df_resultados["Resultado"] == "Válido"]
            df_fallido = df_resultados[df_resultados["Resultado"] == "Fallido"]

            fig_break.add_trace(go.Scatter(
                x=df_valido.index,
                y=df_valido["Precio_Inicio"],
                mode="markers",
                name="Breakouts Válidos",
                marker=dict(color="green", size=9, symbol="triangle-up")
            ))
            fig_break.add_trace(go.Scatter(
                x=df_fallido.index,
                y=df_fallido["Precio_Inicio"],
                mode="markers",
                name="Breakouts Fallidos",
                marker=dict(color="red", size=9, symbol="x")
            ))

            fig_break.update_layout(
                title="🚨 Breakouts del Precio del Oro",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                height=600
            )

            st.plotly_chart(fig_break, use_container_width=True)

            # 📋 Mostrar tabla
            if st.checkbox("📋 Mostrar tabla de breakouts detectados"):
                st.dataframe(df_resultados.style.format({"Cambio_%": "{:.2f}"}))

            # 📊 Resumen
            st.markdown(f"✅ Total de breakouts detectados: `{len(df_resultados)}`")
            st.markdown(f"🔼 Válidos: `{(df_resultados['Resultado'] == 'Válido').sum()}`")
            st.markdown(f"🔻 Fallidos: `{(df_resultados['Resultado'] == 'Fallido').sum()}`")

        elif seleccion_trend == opciones_trend[13]:
            st.subheader("📉 Gaps de Apertura del Oro en Fin de Semana (Viernes vs Lunes)")

            df_gap = pd.read_csv("datasets/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
            df_gap = df_gap.rename(columns={"Date": "Fecha"})
            df_gap.set_index("Fecha", inplace=True)
            df_gap = df_gap.sort_index()

            # Asegurarse de que los datos sean numéricos
            for col in ["Open", "Close"]:
                df_gap[col] = pd.to_numeric(df_gap[col], errors="coerce")
            df_gap = df_gap.dropna(subset=["Open", "Close"])

            # Extraer días de la semana
            df_gap["Día"] = df_gap.index.day_name(locale='es_ES.utf8') if "es_ES.utf8" in plt.rcParams.get('locale', '') else df_gap.index.day_name()

            # Crear lista para guardar los gaps detectados
            gaps = []

            fechas = df_gap.index.tolist()

            for i in range(1, len(fechas)):
                fecha_actual = fechas[i]
                fecha_anterior = fechas[i - 1]

                dia_actual = fecha_actual.weekday()
                dia_anterior = fecha_anterior.weekday()

                # Si hoy es lunes (0) y el día anterior fue viernes (4), calculamos el gap
                if dia_actual == 0 and dia_anterior == 4:
                    cierre_viernes = df_gap.loc[fecha_anterior, "Close"]
                    apertura_lunes = df_gap.loc[fecha_actual, "Open"]
                    gap = apertura_lunes - cierre_viernes
                    gap_pct = (gap / cierre_viernes) * 100
                    tipo = "Alcista" if gap > 0 else "Bajista"
                    gaps.append((fecha_actual, cierre_viernes, apertura_lunes, gap, gap_pct, tipo))

            df_gaps = pd.DataFrame(gaps, columns=["Fecha_Lunes", "Cierre_Viernes", "Apertura_Lunes", "Gap", "Gap_%", "Tipo"])
            df_gaps.set_index("Fecha_Lunes", inplace=True)

            # 📊 Gráfico de gaps en porcentaje
            fig_gap = px.bar(
                df_gaps,
                x=df_gaps.index,
                y="Gap_%",
                color="Tipo",
                title="📉 Gaps de Apertura del Oro los Lunes",
                labels={"Gap_%": "Gap (%)", "Fecha_Lunes": "Fecha"},
                color_discrete_map={"Alcista": "green", "Bajista": "red"},
            )
            fig_gap.update_layout(xaxis_title="Fecha del Lunes", yaxis_title="Gap (%)")
            st.plotly_chart(fig_gap, use_container_width=True)

            # 📈 Promedio y estadísticas
            st.markdown("### 📊 Estadísticas de Gaps")
            promedio_gap = df_gaps["Gap_%"].mean()
            mayor_gap = df_gaps["Gap_%"].max()
            menor_gap = df_gaps["Gap_%"].min()

            st.markdown(f"- 📌 Gap promedio: `{promedio_gap:.2f}%`")
            st.markdown(f"- 🔺 Mayor gap alcista: `{mayor_gap:.2f}%`")
            st.markdown(f"- 🔻 Mayor gap bajista: `{menor_gap:.2f}%`")

            # 📋 Mostrar tabla completa
            if st.checkbox("📋 Mostrar tabla completa de gaps"):
                st.dataframe(df_gaps.style.format({"Gap": "{:.2f}", "Gap_%": "{:.2f}%"}))
    

    # 🔄 Estrategias
    with tabs[2]:
        st.header("🔄 Estrategias")
        opciones_estrategias = [
            "Backtesting: Estrategia de Triple Cruce de Medias Móviles",
            "Backtesting: Estrategia de Cruce de Medias Móviles en Oro",
            "Backtesting: Estrategia de Bandas de Bollinger",
            "Backtesting: Golden Cross / Death Cross",
            "Backtesting: Estrategia Buy the Dip",
            "Backtesting: Estrategia de Momentum",
            "Estrategia basada en Eventos Macroeconómicos",
            "Simulación de Estrategias de Dollar-Cost Averaging (DCA)",
            "Simulación Monte Carlo del Precio Futuro del Oro"
        ]
        seleccion_estrategia = st.selectbox("Selecciona una estrategia", opciones_estrategias)
        
        if seleccion_estrategia == opciones_estrategias[0]:
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

        
        elif seleccion_estrategia == opciones_estrategias[1]:
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
    
        elif seleccion_estrategia == opciones_estrategias[2]:
            # 📌 📊 **Sección 1: Backtesting - Estrategia de Cruce de Medias Móviles**
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
            
                        # ======================================
            # 📊 Análisis con Apalancamiento
            # ======================================
            st.subheader("⚙️ Análisis con Apalancamiento (x2, x5, x10)")

            apalancamientos = [2, 5, 10]
            resultados_apal = []

            for x in apalancamientos:
                df_bt[f"Retorno Estrategia x{x}"] = df_bt["Retorno Estrategia"] * x
                df_bt[f"Capital x{x}"] = (1 + df_bt[f"Retorno Estrategia x{x}"]).cumprod()
                
                rendimiento = df_bt[f"Capital x{x}"].iloc[-1] - 1
                volatilidad = df_bt[f"Retorno Estrategia x{x}"].std() * (252 ** 0.5)
                sharpe = rendimiento / volatilidad if volatilidad != 0 else 0

                resultados_apal.append({
                    "Apalancamiento": f"x{x}",
                    "Rentabilidad": rendimiento,
                    "Volatilidad": volatilidad,
                    "Sharpe": sharpe
                })

            # Mostrar resultados
            for r in resultados_apal:
                st.markdown(f"### 🔹 Apalancamiento {r['Apalancamiento']}")
                st.markdown(f"- 📈 Rentabilidad: `{r['Rentabilidad']:.2%}`")
                st.markdown(f"- ⚠️ Volatilidad: `{r['Volatilidad']:.2%}`")
                st.markdown(f"- 📌 Sharpe Ratio: `{r['Sharpe']:.2f}`")

            st.markdown("💡 *El apalancamiento aumenta tanto el riesgo como la rentabilidad potencial.*")
    
        elif seleccion_estrategia == opciones_estrategias[3]:
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
    
        elif seleccion_estrategia == opciones_estrategias[4]:
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
    
        elif seleccion_estrategia == opciones_estrategias[5]:
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
            
        elif seleccion_estrategia == opciones_estrategias[6]:
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
            
        elif seleccion_estrategia == opciones_estrategias[7]:
                    
            st.subheader("💸 Simulación de Estrategia DCA (Dollar-Cost Averaging) en Oro")

            # 📅 Parámetros seleccionables por el usuario
            start_dca = pd.to_datetime(st.sidebar.date_input("Fecha de inicio DCA", value=df.index.min(), key="dca_start"))
            end_dca = pd.to_datetime(st.sidebar.date_input("Fecha de fin DCA", value=df.index.max(), key="dca_end"))
            freq_dca = st.sidebar.selectbox("Frecuencia de Compra", ["Mensual", "Trimestral", "Anual"])
            cantidad_dca = st.sidebar.number_input("💵 Inversión por periodo (USD)", min_value=10.0, step=10.0, value=100.0)

            # 📆 Convertir frecuencia a resample rule
            freq_map = {"Mensual": "M", "Trimestral": "Q", "Anual": "Y"}
            freq_rule = freq_map[freq_dca]

            # Filtrar datos
            df_dca = df[(df.index >= start_dca) & (df.index <= end_dca)].copy()

            # Generar fechas de inversión
            fechas_inversion = df_dca.resample(freq_rule).first().index
            df_dca["Compra"] = df_dca.index.isin(fechas_inversion)

            # Calcular compras acumuladas
            df_dca["Unidades_Oro"] = 0.0
            df_dca.loc[df_dca["Compra"], "Unidades_Oro"] = cantidad_dca / df_dca.loc[df_dca["Compra"], "Precio_Oro"]

            df_dca["Unidades_Acumuladas"] = df_dca["Unidades_Oro"].cumsum()
            df_dca["Inversion_Acumulada"] = df_dca["Compra"].cumsum() * cantidad_dca
            df_dca["Valor_Portfolio"] = df_dca["Unidades_Acumuladas"] * df_dca["Precio_Oro"]

            # Calcular rentabilidad
            df_dca["Rentabilidad_DCA"] = df_dca["Valor_Portfolio"] / df_dca["Inversion_Acumulada"] - 1
            df_dca.dropna(inplace=True)

            # 📊 Gráfico de crecimiento de inversión vs valor
            fig_dca = px.line(
                df_dca,
                x=df_dca.index,
                y=["Inversion_Acumulada", "Valor_Portfolio"],
                labels={"value": "USD", "variable": "Serie"},
                title="💰 Evolución de la Inversión DCA vs Valor del Portafolio"
            )
            st.plotly_chart(fig_dca, use_container_width=True)

            # 📌 Métricas clave
            final_inversion = df_dca["Inversion_Acumulada"].iloc[-1]
            final_valor = df_dca["Valor_Portfolio"].iloc[-1]
            rentabilidad_total = df_dca["Rentabilidad_DCA"].iloc[-1] * 100

            st.markdown("### 📊 Resultados Finales de la Estrategia DCA")
            st.markdown(f"- 💸 Inversión Total: `{final_inversion:,.2f} USD`")
            st.markdown(f"- 💰 Valor Final del Portafolio: `{final_valor:,.2f} USD`")
            st.markdown(f"- 📈 Rentabilidad Acumulada: `{rentabilidad_total:.2f}%`")
            st.markdown("💡 *DCA puede ayudar a reducir el riesgo del market timing al promediar los precios de compra.*")

        elif seleccion_estrategia == opciones_estrategias[8]:
                
            st.subheader("🎲 Simulación Monte Carlo del Precio Futuro del Oro")

            # Parámetros seleccionables por el usuario
            num_simulaciones = st.sidebar.slider("Número de simulaciones", min_value=100, max_value=5000, step=100, value=1000)
            dias_simulacion = st.sidebar.slider("Horizonte temporal (días)", min_value=30, max_value=365*2, step=30, value=365)
            precio_inicial = df["Precio_Oro"].iloc[-1]

            st.markdown(f"🔢 Precio actual del oro: **${precio_inicial:.2f} USD**")
            st.markdown(f"📅 Horizonte: **{dias_simulacion} días** | 🔁 Simulaciones: **{num_simulaciones}**")

            # Calcular retornos logarítmicos
            df_mc = df.copy()
            df_mc["LogRetorno"] = np.log(df_mc["Precio_Oro"] / df_mc["Precio_Oro"].shift(1))
            mu = df_mc["LogRetorno"].mean()
            sigma = df_mc["LogRetorno"].std()

            # Simulación Monte Carlo
            resultados = np.zeros((dias_simulacion, num_simulaciones))
            for i in range(num_simulaciones):
                precios = [precio_inicial]
                for _ in range(1, dias_simulacion):
                    drift = mu - 0.5 * sigma**2
                    shock = sigma * np.random.normal()
                    precio = precios[-1] * np.exp(drift + shock)
                    precios.append(precio)
                resultados[:, i] = precios

            # Crear DataFrame para graficar
            fechas_futuras = pd.date_range(df.index[-1], periods=dias_simulacion, freq="D")
            df_resultados = pd.DataFrame(resultados, index=fechas_futuras)

            # 📊 Gráfico de simulaciones
            fig_mc = px.line(df_resultados.iloc[:, :200],  # Solo primeras 200 simulaciones por claridad
                            title="🎲 Simulaciones Monte Carlo del Precio del Oro",
                            labels={"value": "Precio Oro (USD)", "index": "Fecha"})
            st.plotly_chart(fig_mc, use_container_width=True)

            # 📈 Intervalos de confianza
            percentiles = df_resultados.quantile([0.05, 0.5, 0.95], axis=1).T
            percentiles["Fecha"] = percentiles.index
            percentiles.columns = ["P5", "Mediana", "P95", "Fecha"]

            fig_conf = go.Figure()
            fig_conf.add_trace(go.Scatter(
                x=percentiles["Fecha"], y=percentiles["P95"],
                name="Percentil 95%", line=dict(color='lightgreen')
            ))
            fig_conf.add_trace(go.Scatter(
                x=percentiles["Fecha"], y=percentiles["Mediana"],
                name="Mediana", line=dict(color='gold')
            ))
            fig_conf.add_trace(go.Scatter(
                x=percentiles["Fecha"], y=percentiles["P5"],
                name="Percentil 5%", line=dict(color='lightcoral')
            ))

            fig_conf.update_layout(
                title="📈 Intervalos de Confianza del Precio Simulado del Oro",
                xaxis_title="Fecha", yaxis_title="Precio (USD)",
                legend=dict(x=0.01, y=0.99)
            )

            st.plotly_chart(fig_conf, use_container_width=True)

            # Métricas al final del período
            precio_esperado = df_resultados.iloc[-1].mean()
            p5 = df_resultados.iloc[-1].quantile(0.05)
            p95 = df_resultados.iloc[-1].quantile(0.95)

            st.markdown("### 📊 Resultados Esperados al Final del Horizonte")
            st.markdown(f"- 🔮 **Precio Esperado**: `{precio_esperado:.2f} USD`")
            st.markdown(f"- 📉 **5% más bajo (riesgo)**: `{p5:.2f} USD`")
            st.markdown(f"- 📈 **5% más alto (optimista)**: `{p95:.2f} USD`")
            st.markdown("💡 *La simulación Monte Carlo no predice el futuro, pero ayuda a estimar un rango probable de evolución.*")

            
    # ⚔️ Eventos
    with tabs[3]:
        st.header("⚔️ Eventos")
        opciones_eventos = [
            "Relación entre Eventos y Cambios Significativos",
            "Reacción del Oro ante Crisis Geopolíticas",
            "Análisis Cuantitativo: Retornos Antes y Después de Eventos Geopolítico",
            "Análisis del Oro en Elecciones Presidenciales USA",
            "Efecto de Recesiones en el Precio del Oro",
            "Reacción del Oro a Crisis Financieras en Europa",
            "Influencia de Crisis Inmobiliarias en el Oro",
            "Estudio del Oro frente a Choques de Oferta en Minerales"
        ]
        seleccion_evento = st.selectbox("Selecciona un análisis de eventos", opciones_eventos)
        if seleccion_evento == opciones_eventos[0]:
             # 📌 📊 **Sección 1: Eventos Cercanos a Cambios Bruscos**
            st.subheader("📌 Relación entre Eventos y Cambios Significativos")

            # 1. Días más volátiles
            variacion = df["Precio_Oro"].pct_change() * 100
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
        
        elif seleccion_evento == opciones_eventos[1]:
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
    
        elif seleccion_evento == opciones_eventos[2]:
            st.subheader("📉 Análisis Cuantitativo: Retornos Antes y Después de Eventos Geopolíticos")
            
            geo_df = eventos_df[eventos_df["Categoría"] == "Geopolítico"].copy()

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
    
        elif seleccion_evento == opciones_eventos[3]:
            st.subheader("🇺🇸 Análisis del Oro en Elecciones Presidenciales USA")

            volatilidad_eventos = []
            for _, evento in df_elecciones_usa.iterrows():
                fecha_evento = evento["Fecha"]
                rango_antes = df[(df.index >= fecha_evento - pd.Timedelta(days=10)) & (df.index < fecha_evento)]
                rango_despues = df[(df.index > fecha_evento) & (df.index <= fecha_evento + pd.Timedelta(days=10))]

                if not rango_antes.empty and not rango_despues.empty:
                    vol_antes = rango_antes["Precio_Oro"].pct_change().std() * 100
                    vol_despues = rango_despues["Precio_Oro"].pct_change().std() * 100
                    ret_antes = (rango_antes["Precio_Oro"].iloc[-1] / rango_antes["Precio_Oro"].iloc[0] - 1) * 100
                    ret_despues = (rango_despues["Precio_Oro"].iloc[-1] / rango_despues["Precio_Oro"].iloc[0] - 1) * 100
                else:
                    vol_antes, vol_despues, ret_antes, ret_despues = None, None, None, None

                volatilidad_eventos.append({
                    "Fecha": fecha_evento.date(),
                    "Evento": evento["Evento"],
                    "Presidente Electo": evento["Presidente_Electo"],
                    "Partido": evento["Partido"],
                    "Volatilidad Antes (%)": vol_antes,
                    "Volatilidad Después (%)": vol_despues,
                    "Retorno Antes (%)": ret_antes,
                    "Retorno Después (%)": ret_despues
                })

            df_vol_elec = pd.DataFrame(volatilidad_eventos).dropna()

            st.dataframe(df_vol_elec)

            # 📊 Gráfico comparativo de volatilidad
            fig_vol = px.bar(
                df_vol_elec,
                x="Fecha",
                y=["Volatilidad Antes (%)", "Volatilidad Después (%)"],
                title="Volatilidad del Oro antes y después de Elecciones USA",
                barmode="group",
                labels={"value": "Volatilidad (%)", "variable": "Periodo"},
                color_discrete_map={
                    "Volatilidad Antes (%)": "gray",
                    "Volatilidad Después (%)": "gold"
                }
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # 📊 Gráfico de retorno antes vs después
            fig_ret = px.bar(
                df_vol_elec,
                x="Fecha",
                y=["Retorno Antes (%)", "Retorno Después (%)"],
                title="Retorno del Oro antes y después de Elecciones USA",
                barmode="group",
                labels={"value": "Retorno (%)", "variable": "Periodo"},
                color_discrete_map={
                    "Retorno Antes (%)": "lightblue",
                    "Retorno Después (%)": "darkblue"
                }
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        elif seleccion_evento == opciones_eventos[4]:
            st.subheader("📉 Efecto de Recesiones en el Precio del Oro")

            resultados_recesiones = []

            for _, recesion in df_recesiones.iterrows():
                inicio = recesion["Fecha_Inicio"]
                fin = recesion["Fecha_Fin"]

                precio_inicio = df.loc[inicio:inicio + pd.Timedelta(days=2)]["Precio_Oro"].dropna().iloc[0] if not df.loc[inicio:inicio + pd.Timedelta(days=2)]["Precio_Oro"].dropna().empty else None
                precio_fin = df.loc[fin - pd.Timedelta(days=2):fin]["Precio_Oro"].dropna().iloc[-1] if not df.loc[fin - pd.Timedelta(days=2):fin]["Precio_Oro"].dropna().empty else None

                if precio_inicio is not None and precio_fin is not None:
                    retorno = (precio_fin / precio_inicio - 1) * 100
                else:
                    retorno = None

                resultados_recesiones.append({
                    "Recesión": recesion["Nombre"],
                    "Fecha Inicio": inicio.date(),
                    "Fecha Fin": fin.date(),
                    "Retorno del Oro (%)": retorno
                })

            df_recesion_resultado = pd.DataFrame(resultados_recesiones).dropna()

            st.dataframe(df_recesion_resultado)

            # 📊 Gráfico de retornos durante las recesiones
            fig = px.bar(
                df_recesion_resultado,
                x="Recesión",
                y="Retorno del Oro (%)",
                title="Variación del Precio del Oro durante Recesiones Mundiales",
                color="Retorno del Oro (%)",
                color_continuous_scale="reds",
            )
            st.plotly_chart(fig, use_container_width=True)
    
        elif seleccion_evento == opciones_eventos[5]:
                    
            st.subheader("💶 Efecto de Crisis Financieras en Europa en el Precio del Oro")

            retornos_crisis = []
            for _, evento in df_crisis_europa.iterrows():
                fecha_inicio = evento["Fecha de Inicio"]
                fecha_fin = evento["Fecha de Fin"]

                rango_precio = df[(df.index >= fecha_inicio) & (df.index <= fecha_fin)]
                
                if not rango_precio.empty:
                    retorno_oro = (rango_precio["Precio_Oro"].iloc[-1] / rango_precio["Precio_Oro"].iloc[0] - 1) * 100
                else:
                    retorno_oro = None

                retornos_crisis.append({
                    "Crisis": evento["Evento"],
                    "Fecha Inicio": fecha_inicio.date(),
                    "Fecha Fin": fecha_fin.date(),
                    "Países Afectados": evento["Países Afectados"],
                    "Causa Principal": evento["Causa Principal"],
                    "Retorno del Oro (%)": retorno_oro
                })

            df_ret_crisis = pd.DataFrame(retornos_crisis).dropna()
            st.dataframe(df_ret_crisis)

            fig_crisis = px.bar(
                df_ret_crisis,
                x="Crisis",
                y="Retorno del Oro (%)",
                color="Retorno del Oro (%)",
                title="Variación del Precio del Oro durante Crisis Financieras en Europa",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_crisis, use_container_width=True)

        elif seleccion_evento == opciones_eventos[6]:
            
            st.write("Influencia de Crisis Inmobiliarias en el Oro 🚧")


            retornos_inmo = []
            for _, evento in df_crisis_inmo.iterrows():
                fecha_inicio = evento["Fecha_Inicio"]
                fecha_fin = evento["Fecha_Fin"]

                rango_precio = df[(df.index >= fecha_inicio) & (df.index <= fecha_fin)]
                
                if not rango_precio.empty:
                    retorno = (rango_precio["Precio_Oro"].iloc[-1] / rango_precio["Precio_Oro"].iloc[0] - 1) * 100
                else:
                    retorno = None

                retornos_inmo.append({
                    "Crisis Inmobiliaria": evento["Evento"],
                    "Fecha Inicio": fecha_inicio.date(),
                    "Fecha Fin": fecha_fin.date(),
                    "Países Afectados": evento["Países_Afectados"],
                    "Causa": evento["Causa_Principal"],
                    "Retorno del Oro (%)": retorno
                })

            df_ret_inmo = pd.DataFrame(retornos_inmo).dropna()
            st.dataframe(df_ret_inmo)

            fig_inmo = px.bar(
                df_ret_inmo,
                x="Crisis Inmobiliaria",
                y="Retorno del Oro (%)",
                color="Retorno del Oro (%)",
                title="Variación del Precio del Oro durante Crisis Inmobiliarias",
                color_continuous_scale="Purples"
            )
            st.plotly_chart(fig_inmo, use_container_width=True)

        elif seleccion_evento == opciones_eventos[7]:
            
            st.subheader("⛏️ Efecto de Crisis de Materias Primas Críticas en el Precio del Oro")

            resultados_minerales = []
            for _, evento in df_crisis_minerales.iterrows():
                fecha_inicio = evento["Fecha Inicio"]
                fecha_fin = evento["Fecha Fin"]

                datos_periodo = df[(df.index >= fecha_inicio) & (df.index <= fecha_fin)]

                if not datos_periodo.empty:
                    retorno = (datos_periodo["Precio_Oro"].iloc[-1] / datos_periodo["Precio_Oro"].iloc[0] - 1) * 100
                else:
                    retorno = None

                resultados_minerales.append({
                    "Crisis": evento["Evento"],
                    "Fecha Inicio": fecha_inicio.date(),
                    "Fecha Fin": fecha_fin.date(),
                    "Minerales Afectados": evento["Minerales Afectados"],
                    "Causa": evento["Causa"],
                    "Países Afectados": evento["Países Afectados"],
                    "Retorno del Oro (%)": retorno
                })

            df_ret_minerales = pd.DataFrame(resultados_minerales).dropna()
            st.dataframe(df_ret_minerales)

            fig_minerales = px.bar(
                df_ret_minerales,
                x="Crisis",
                y="Retorno del Oro (%)",
                color="Retorno del Oro (%)",
                title="📊 Variación del Precio del Oro ante Crisis de Minerales Críticos",
                color_continuous_scale="Magma"
            )
            st.plotly_chart(fig_minerales, use_container_width=True)

    # 🧪 Cuantitativo
    with tabs[4]:
        st.header("🧪 Cuantitativo")
        opciones_cuant = [
            "Distribución de Retornos Diarios del Oro",
            "Días con Mayores Subidas y Bajadas del Precio del Oro",
            "Análisis de Retornos Semanales del Oro ",
            "Estudio de Regresión Lineal entre Oro y IPC "
        ]
        seleccion_cuant = st.selectbox("Selecciona un análisis cuantitativo", opciones_cuant)
        
        if seleccion_cuant == opciones_cuant[0]:
            # 📌 📊 **Sección 8: Distribución de Retornos Diarios**
            st.subheader("📊 Distribución de Retornos Diarios del Oro")
            
            df["Retorno Diario"] = df["Precio_Oro"].pct_change() * 100
            df = df.dropna(subset=["Retorno Diario"])


            fig_dist = px.histogram(
                df.dropna(), 
                x="Retorno Diario", 
                nbins=100,
                title="Distribución de Retornos Diarios",
                labels={"Retorno Diario": "Retorno Diario (%)"}
            )
            fig_dist.update_layout(bargap=0.01)
            st.plotly_chart(fig_dist, use_container_width=True)
    
        elif seleccion_cuant == opciones_cuant[1]:
            # 📌 📊 **Sección 2: Días con Subidas y Bajadas más Extremas**
            st.subheader("🔺 Días con Mayores Subidas y Bajadas del Precio del Oro")
            
            df["Retorno Diario"] = df["Precio_Oro"].pct_change() * 100
            df = df.dropna(subset=["Retorno Diario"])


            variacion = df["Retorno Diario"] * 100
            top_subidas = variacion.nlargest(10).round(2)
            top_bajadas = variacion.nsmallest(10).round(2)

            st.markdown("##### 🔼 Top 10 Subidas")
            st.dataframe(top_subidas.to_frame(name="Variación %"))

            st.markdown("##### 🔽 Top 10 Bajadas")
            st.dataframe(top_bajadas.to_frame(name="Variación %"))
        
        elif seleccion_cuant == opciones_cuant[2]:
                    
            st.subheader("📆 Análisis de Retornos Semanales del Oro")

            # Resamplear por semana y calcular precio final de la semana
            df_weekly = df["Precio_Oro"].resample("W").last()

            # Calcular retorno semanal en porcentaje
            retornos_semanales = df_weekly.pct_change().dropna() * 100

            # Crear dataframe para análisis
            df_retornos = pd.DataFrame({
                "Retorno Semanal (%)": retornos_semanales,
                "Semana": retornos_semanales.index
            })

            # Mostrar estadísticas clave
            st.write("📈 **Estadísticas de Retornos Semanales**")
            st.dataframe(df_retornos["Retorno Semanal (%)"].describe().round(4))

            # Semanas con mayor y menor retorno
            st.write("🔼 **Top 5 Semanas con Mayor Subida**")
            st.dataframe(df_retornos.sort_values("Retorno Semanal (%)", ascending=False).head(5))

            st.write("🔽 **Top 5 Semanas con Mayor Caída**")
            st.dataframe(df_retornos.sort_values("Retorno Semanal (%)").head(5))

            # Gráfico de línea de retornos
            st.plotly_chart(
                px.line(df_retornos, x="Semana", y="Retorno Semanal (%)", title="📊 Retornos Semanales del Precio del Oro"),
                use_container_width=True
            )

            # Histograma de distribución
            st.plotly_chart(
                px.histogram(df_retornos, x="Retorno Semanal (%)", nbins=100, marginal="rug",
                            title="📉 Distribución de Retornos Semanales del Oro"),
                use_container_width=True
    )
            
        elif seleccion_cuant == opciones_cuant[3]:
            
            st.subheader("📈 Regresión Lineal entre Precio del Oro e IPC (EE.UU.)")

            # 1. Unir por fecha
            df_regresion = df_oro_mensual.join(cpi_df, how="inner")  # Índice = Fecha

            # 2. Eliminar nulos
            df_regresion = df_regresion.dropna(subset=["Precio_Oro", "IPC"])

            # 3. Variables
            X = df_regresion[["IPC"]].values
            y = df_regresion["Precio_Oro"].values

            # 4. Entrenar modelo
            modelo = LinearRegression()
            modelo.fit(X, y)
            y_pred = modelo.predict(X)

            # 5. Métricas
            pendiente = modelo.coef_[0]
            intercepto = modelo.intercept_
            r2 = modelo.score(X, y)

            # 6. Mostrar métricas
            st.markdown("### 📊 Resultados de la Regresión")
            st.markdown(f"""
            - **Coeficiente (pendiente)**: `{pendiente:.4f}`  
            - **Intercepto**: `{intercepto:.4f}`  
            - **R² Score**: `{r2:.4f}`
            """)

            # 7. Gráfico interactivo
            fig = px.scatter(
                df_regresion,
                x="IPC",
                y="Precio_Oro",
                title="Relación entre IPC y Precio del Oro",
                labels={"IPC": "Índice de Precios al Consumidor (EE.UU.)", "Precio_Oro": "Precio del Oro (USD/oz)"},
                opacity=0.7
            )

            # Añadir línea de regresión
            fig.add_traces(
                go.Scatter(
                    x=df_regresion["IPC"],
                    y=y_pred,
                    mode="lines",
                    name="Línea de regresión",
                    line=dict(color="red")
                )
            )

            fig.update_layout(
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔍 Proyección de Precio del Oro según IPC")

            nuevo_ipc = st.slider("Selecciona un valor hipotético del IPC (EE.UU.):", 
                                min_value=int(df_regresion["IPC"].min()), 
                                max_value=int(df_regresion["IPC"].max()) + 20, 
                                value=int(df_regresion["IPC"].mean()))
            
            prediccion = modelo.predict([[nuevo_ipc]])[0]
            
            st.markdown(f"📌 Si el IPC alcanza `{nuevo_ipc}`, el modelo predice un precio del oro de aproximadamente: **${prediccion:,.2f} USD/oz**")
            
            st.markdown("### 🧮 Análisis de Residuales")

            df_regresion["Residuales"] = y - y_pred

            fig_resid = px.scatter(
                df_regresion,
                x=df_regresion.index,
                y="Residuales",
                title="Residuales del Modelo de Regresión Lineal",
                labels={"Residuales": "Error de Predicción (USD/oz)", "index": "Fecha"},
                template="plotly_dark"
            )
            fig_resid.add_hline(y=0, line_dash="dot", line_color="red")

            st.plotly_chart(fig_resid, use_container_width=True)
            
            

            # Mostrar sección
            st.markdown("### 📉 Evolución Temporal del R² (Relación Oro - IPC)")

            # Slider
            ventana = st.slider("Tamaño de ventana móvil (meses):", 12, 60, 24)

            # Preparar lista vacía
            r2_scores = []

            # Iterar sobre las ventanas móviles
            for i in range(len(df_regresion) - ventana + 1):
                sub_df = df_regresion.iloc[i:i+ventana]
                X_roll = sub_df[["IPC"]].values
                y_roll = sub_df["Precio_Oro"].values

                if len(X_roll) == ventana and len(y_roll) == ventana:
                    modelo_roll = LinearRegression().fit(X_roll, y_roll)
                    r2_roll = modelo_roll.score(X_roll, y_roll)
                    r2_scores.append((sub_df.index[-1], r2_roll))

            # Convertir resultados a DataFrame
            r2_df = pd.DataFrame(r2_scores, columns=["Fecha", "R2"])
            r2_df.set_index("Fecha", inplace=True)

            # Graficar con Plotly
            fig_r2 = px.line(
                r2_df,
                y="R2",
                title=f"R² Rolling ({ventana} meses): Precio del Oro vs IPC",
                labels={"R2": "R² Score"},
                template="plotly_dark"
            )
            fig_r2.update_layout(yaxis_range=[0, 1])

            # Mostrar
            st.plotly_chart(fig_r2, use_container_width=True)

            
            correlacion = df_regresion["IPC"].corr(df_regresion["Precio_Oro"])
            st.markdown(f"### 🔗 Correlación Pearson: `{correlacion:.4f}`")
            
            st.markdown("""
            ### 📘 Interpretación Económica
            
            - Un coeficiente de `12.94` indica que, **por cada punto que sube el IPC (EE.UU.), el oro aumenta aproximadamente $12.94 USD/oz** en promedio.
            - Esto sugiere una **relación positiva** entre inflación y oro, lo que **refuerza su papel como activo refugio**.
            - El `R² de 0.76` indica que el **76% de la variabilidad en el precio del oro mensual puede explicarse por el IPC**.

            ⚠️ *Nota:* Este modelo es lineal y no captura eventos geopolíticos, manipulación de mercados o políticas monetarias inesperadas.
            """)








    # 🧠 Influencias
    with tabs[5]:
        st.header("🧠 Indicadores e Influencias Externas")
        opciones_influencias = [
            "Influencia del Índice de Confianza del Consumidor (CCI) ",
            "Índice de Miedo y Avaricia (Fear & Greed Index) vs. Oro ",
            "Influencia del Balance de la FED en el Oro ",
            "Influencia del PIB de EE.UU. en el Precio del Oro ",
            "Oro en Periodos de Caída del S&P 500 "
            
        ]
        seleccion_influencia = st.selectbox("Selecciona un indicador o influencia", opciones_influencias)
        
        if seleccion_influencia == opciones_influencias[0]:

            st.subheader("📉 Influencia del Índice de Confianza del Consumidor (CCI)")

            st.markdown("""
            Este análisis examina cómo los cambios en el **Índice de Confianza del Consumidor (CCI)** pueden correlacionarse con las variaciones en el precio del oro.
            Se analizan tanto el índice general como sus componentes: *Situación Actual* y *Expectativas*.
            """)

            # Combinar CCI y Precio del Oro
            df_cci = df[["Precio_Oro"]].join(cci_df, how="inner")

            # Visualización conjunta
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_cci.index, y=df_cci["Precio_Oro"], name="Precio del Oro", line=dict(color="gold")), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_cci.index, y=df_cci["CCI"], name="CCI", line=dict(color="blue")), secondary_y=True)

            fig.update_layout(title="Precio del Oro vs Índice de Confianza del Consumidor (CCI)", template="plotly_dark")
            fig.update_yaxes(title_text="Precio del Oro", secondary_y=False)
            fig.update_yaxes(title_text="CCI", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

            # Correlación
            corr = df_cci[["Precio_Oro", "CCI", "Situacion_Actual", "Expectativas"]].corr()
            st.markdown("### 📈 Matriz de Correlación")
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"))
            
            # 1. Regresión lineal entre CCI y Precio del Oro

            st.subheader("📈 Regresión Lineal entre CCI y Precio del Oro")

            # Eliminar NaNs
            df_regresion = df_cci[["CCI", "Precio_Oro"]].dropna()

            # Ajustar modelo
            modelo_1 = LinearRegression()
            modelo_1.fit(df_regresion[["CCI"]], df_regresion["Precio_Oro"])

            # Predicción
            df_regresion["Predicción"] = modelo_1.predict(df_regresion[["CCI"]])

            # Plot
            fig1 = px.scatter(
                df_regresion,
                x="CCI",
                y="Precio_Oro",
                trendline="ols",
                labels={"CCI": "Índice de Confianza del Consumidor", "Precio_Oro": "Precio del Oro (USD/oz)"},
                title="Relación entre CCI y Precio del Oro"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Mostrar coeficiente y R²
            st.markdown(f"**Coeficiente:** {modelo_1.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_1.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_1.score(df_regresion[['CCI']], df_regresion['Precio_Oro']):.4f}")


            # 2. Regresión lineal entre Var_CCI y Var_Precio_Oro

            st.subheader("📉 Regresión Lineal entre Variaciones Mensuales de CCI y del Precio del Oro")
            
            # Calcular variaciones mensuales (%)
            df_cci["Var_CCI"] = df_cci["CCI"].pct_change() * 100
            df_cci["Var_Precio_Oro"] = df_cci["Precio_Oro"].pct_change() * 100

            # Eliminar NaNs
            df_var = df_cci[["Var_CCI", "Var_Precio_Oro"]].dropna()
            

            # Ajustar modelo
            modelo_2 = LinearRegression()
            modelo_2.fit(df_var[["Var_CCI"]], df_var["Var_Precio_Oro"])

            # Predicción
            df_var["Predicción"] = modelo_2.predict(df_var[["Var_CCI"]])

            # Plot
            fig2 = px.scatter(
                df_var,
                x="Var_CCI",
                y="Var_Precio_Oro",
                trendline="ols",
                labels={"Var_CCI": "Variación mensual del CCI", "Var_Precio_Oro": "Variación mensual del Oro (%)"},
                title="Relación entre Variación del CCI y del Precio del Oro"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Mostrar coeficiente y R²
            st.markdown(f"**Coeficiente:** {modelo_2.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_2.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_2.score(df_var[['Var_CCI']], df_var['Var_Precio_Oro']):.4f}")
            
            # 3. Test de Causalidad de Granger

            st.subheader("🧪 Test de Causalidad de Granger")

            st.markdown("""
            Este test evalúa si los cambios en el **CCI** pueden ayudar a predecir los cambios futuros en el **precio del oro**.

            > H0 (nula): El CCI **no** causa (en el sentido de Granger) al precio del oro.
            """)

            

            # Preparar el DataFrame para el test (sin NaNs)
            df_gc = df_cci[["Var_Precio_Oro", "Var_CCI"]].dropna()

            # Ejecutar el test de Granger para diferentes lags
            max_lags = st.slider("Selecciona el número máximo de retardos (lags):", 1, 12, 6)

            with st.expander("Resultados detallados del Test de Granger"):
                granger_result = grangercausalitytests(df_gc, maxlag=max_lags, verbose=True)

            # Mostrar tabla resumen con p-values
            st.markdown("**P-values por lag:**")
            p_values = {f"Lag {lag}": round(result[0]['ssr_ftest'][1], 4) for lag, result in granger_result.items()}
            st.dataframe(pd.DataFrame.from_dict(p_values, orient="index", columns=["p-value"]))
            
            
             # 4. Análisis de Retardos: Cross-correlation

            st.subheader("⏳ Análisis de Retardos (Cross-Correlation entre CCI y Oro)")

            st.markdown("""
            Se mide en qué **retardo (lag)** el CCI y el Precio del Oro están más correlacionados.

            Valores positivos indican que **CCI va primero**, negativos que **el oro lidera**.
            """)

            # Obtener las dos series (ya con dropna hecho)
            serie_cci = df_cci["Var_CCI"].dropna()
            serie_oro = df_cci["Var_Precio_Oro"].dropna()

            # Asegurar misma longitud (recorte por el índice)
            min_len = min(len(serie_cci), len(serie_oro))
            serie_cci = serie_cci[-min_len:]
            serie_oro = serie_oro[-min_len:]

            lags = arange(-12, 13, 1)  # ±12 meses
            correlaciones = [serie_cci.corr(serie_oro.shift(lag)) for lag in lags]

            # Plot
            fig_cc = px.line(
                x=lags,
                y=correlaciones,
                labels={"x": "Lag (meses)", "y": "Correlación"},
                title="Correlación Cruzada entre Variaciones de CCI y Precio del Oro"
            )
            fig_cc.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_cc, use_container_width=True)

            # Mejor correlación
            mejor_lag = lags[np.argmax(correlaciones)]
            st.markdown(f"**Mayor correlación:** {max(correlaciones):.2f} en lag {mejor_lag} meses")




            
        elif seleccion_influencia == opciones_influencias[1]:

            st.subheader("😨😈 Influencia del Índice de Miedo y Avaricia (Fear & Greed Index)")

            st.markdown("""
            El **Fear & Greed Index** refleja el sentimiento general de los inversores. Este análisis explora si existe alguna **correlación o capacidad predictiva** entre este índice y el **precio del oro**.
            
            Valores altos del índice indican avaricia (optimismo), y valores bajos, miedo (pesimismo).
            """)

            # Unir con el precio del oro
            df_fg = df[["Precio_Oro"]].join(fear_greed_df, how="inner")

            # Gráfico dual
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_fg.index, y=df_fg["Precio_Oro"], name="Precio del Oro", line=dict(color="gold")), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_fg.index, y=df_fg["Fear_Greed"], name="Fear & Greed Index", line=dict(color="crimson")), secondary_y=True)
            fig.update_layout(title="Precio del Oro vs Fear & Greed Index", template="plotly_dark")
            fig.update_yaxes(title_text="Precio del Oro", secondary_y=False)
            fig.update_yaxes(title_text="Fear & Greed", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Matriz de correlación
            corr = df_fg[["Precio_Oro", "Fear_Greed"]].corr()
            st.markdown("### 🔗 Matriz de Correlación")
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"))

            # Calcular variaciones mensuales (%)
            df_fg["Var_FG"] = df_fg["Fear_Greed"].pct_change() * 100
            df_fg["Var_Precio_Oro"] = df_fg["Precio_Oro"].pct_change() * 100

            # Eliminar valores infinitos y NaNs
            df_fg = df_fg.replace([np.inf, -np.inf], np.nan).dropna(subset=["Var_FG", "Var_Precio_Oro"])

            # Preparar DataFrame final
            df_var = df_fg[["Var_FG", "Var_Precio_Oro"]]


            # Regresión lineal entre el índice y el precio
            st.subheader("📉 Regresión Lineal entre Fear & Greed y Precio del Oro")
            df_reg = df_fg[["Fear_Greed", "Precio_Oro"]].dropna()
            modelo_fg = LinearRegression()
            modelo_fg.fit(df_reg[["Fear_Greed"]], df_reg["Precio_Oro"])
            df_reg["Predicción"] = modelo_fg.predict(df_reg[["Fear_Greed"]])
            fig1 = px.scatter(df_reg, x="Fear_Greed", y="Precio_Oro", trendline="ols",
                            title="Relación entre Fear & Greed y el Precio del Oro",
                            labels={"Fear_Greed": "Fear & Greed Index", "Precio_Oro": "Precio del Oro (USD/oz)"})
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(f"**Coeficiente:** {modelo_fg.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_fg.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_fg.score(df_reg[['Fear_Greed']], df_reg['Precio_Oro']):.4f}")

            # Regresión entre variaciones
            st.subheader("📈 Regresión entre Variaciones (%) del Fear & Greed y del Precio del Oro")
            df_var = df_fg[["Var_FG", "Var_Precio_Oro"]].dropna()
            modelo_var = LinearRegression()
            modelo_var.fit(df_var[["Var_FG"]], df_var["Var_Precio_Oro"])
            df_var["Predicción"] = modelo_var.predict(df_var[["Var_FG"]])
            fig2 = px.scatter(df_var, x="Var_FG", y="Var_Precio_Oro", trendline="ols",
                            title="Relación entre Variaciones del Fear & Greed y del Precio del Oro",
                            labels={"Var_FG": "Variación mensual del Fear & Greed", "Var_Precio_Oro": "Variación mensual del Oro (%)"})
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(f"**Coeficiente:** {modelo_var.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_var.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_var.score(df_var[['Var_FG']], df_var['Var_Precio_Oro']):.4f}")
            
            
                        # --- TEST DE CAUSALIDAD DE GRANGER ---
            st.subheader("🔁 Test de Causalidad de Granger")

            st.markdown("""
            Este test evalúa si las **variaciones del Índice de Miedo y Avaricia (FGI)** pueden **predecir los cambios en el precio del oro** (o viceversa).
            """)

            # Seleccionar número máximo de retardos para probar
            max_lag = st.slider("Selecciona el número máximo de retardos (lags):", 1, 12, 6)

            # Construir DataFrame para test (debe tener las 2 series)
            granger_df = df_fg[["Var_FG", "Var_Precio_Oro"]].dropna()

            # Ejecutar test
            try:
                granger_result = grangercausalitytests(granger_df, maxlag=max_lag, verbose=False)

                st.markdown("**Resultados por número de retardos:**")
                for lag in range(1, max_lag+1):
                    p_value = granger_result[lag][0]["ssr_ftest"][1]
                    st.markdown(f"- Lag {lag}: p-valor = `{p_value:.4f}` → {'❗ Causalidad probable' if p_value < 0.05 else 'No significativa'}")
            except Exception as e:
                st.error(f"Error al realizar el test de Granger: {e}")
                
                
                            # --- ANÁLISIS DE RETARDOS (Cross-Correlation) ---
            st.subheader("📊 Análisis de Retardos (Cross-Correlation)")

            st.markdown("""
            Este gráfico muestra cómo se correlacionan las **variaciones del FGI** con las del **oro** en diferentes retardos (lags).
            Permite observar si **los movimientos en el FGI anticipan (o siguen) los del oro**.
            """)

            # Normalizar datos
            df_norm = df_fg[["Var_FG", "Var_Precio_Oro"]].dropna()
            var_fg = (df_norm["Var_FG"] - df_norm["Var_FG"].mean()) / df_norm["Var_FG"].std()
            var_oro = (df_norm["Var_Precio_Oro"] - df_norm["Var_Precio_Oro"].mean()) / df_norm["Var_Precio_Oro"].std()

            # Calcular correlaciones cruzadas para diferentes lags
            max_lag = 12
            lags = np.arange(-max_lag, max_lag+1)
            correlations = [var_fg.corr(var_oro.shift(lag)) for lag in lags]

            # Mostrar gráfico
            fig_lag = go.Figure()
            fig_lag.add_trace(go.Bar(x=lags, y=correlations, marker_color="orange"))
            fig_lag.update_layout(
                title="Correlación Cruzada entre Variaciones de Fear & Greed y del Oro",
                xaxis_title="Retardo (días)",
                yaxis_title="Correlación",
                template="plotly_dark"
            )
            st.plotly_chart(fig_lag, use_container_width=True)



            
        elif seleccion_influencia == opciones_influencias[2]:

            st.subheader("🏦 Influencia del Balance de la FED (WALCL) en el Precio del Oro")

            st.markdown("""
            Este análisis explora la posible relación entre el **balance semanal de la Reserva Federal (FED)** y el **precio del oro**.
            Se analiza si un aumento en el balance puede estar asociado con movimientos del oro como activo refugio frente a la expansión monetaria.
            """)

            # --- Preparar DataFrame fusionado ---
            # Reindexar el FED al índice diario del oro
            fed_df_daily = fed_df.reindex(df.index, method="ffill")  # Rellena valores hasta la próxima semana

            # Unir ambos
            df_fed = df[["Precio_Oro"]].join(fed_df_daily, how="inner")


            # --- Visualización conjunta ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_fed.index, y=df_fed["Precio_Oro"], name="Precio del Oro", line=dict(color="gold")), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_fed.index, y=df_fed["FED_Balance"], name="Balance FED (WALCL)", line=dict(color="purple")), secondary_y=True)

            fig.update_layout(title="Precio del Oro vs Balance de la FED", template="plotly_dark")
            fig.update_yaxes(title_text="Precio del Oro", secondary_y=False)
            fig.update_yaxes(title_text="Balance FED (millones USD)", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

            # --- Correlación simple ---
            st.markdown("### 📊 Matriz de Correlación")
            st.dataframe(df_fed.corr().style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"))

            # --- Regresión Lineal ---
            st.subheader("📈 Regresión Lineal entre Balance FED y Precio del Oro")

            df_reg = df_fed.dropna()
            modelo_fed = LinearRegression()
            modelo_fed.fit(df_reg[["FED_Balance"]], df_reg["Precio_Oro"])
            df_reg["Predicción"] = modelo_fed.predict(df_reg[["FED_Balance"]])

            fig2 = px.scatter(
                df_reg,
                x="FED_Balance",
                y="Precio_Oro",
                trendline="ols",
                labels={"FED_Balance": "Balance FED (millones USD)", "Precio_Oro": "Precio del Oro (USD/oz)"},
                title="Regresión Lineal: Balance FED vs Precio del Oro"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(f"**Coeficiente:** {modelo_fed.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_fed.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_fed.score(df_reg[['FED_Balance']], df_reg['Precio_Oro']):.4f}")

            # --- Variaciones mensuales ---
            st.subheader("📉 Variaciones Mensuales de Balance FED y Precio del Oro")

            df_fed["Var_FED"] = df_fed["FED_Balance"].pct_change() * 100
            df_fed["Var_Precio_Oro"] = df_fed["Precio_Oro"].pct_change() * 100
            df_var = df_fed[["Var_FED", "Var_Precio_Oro"]].dropna()

            modelo_var = LinearRegression()
            modelo_var.fit(df_var[["Var_FED"]], df_var["Var_Precio_Oro"])
            df_var["Predicción"] = modelo_var.predict(df_var[["Var_FED"]])

            fig3 = px.scatter(
                df_var,
                x="Var_FED",
                y="Var_Precio_Oro",
                trendline="ols",
                title="Relación entre Variación del Balance FED y del Oro",
                labels={"Var_FED": "Variación (%) del Balance FED", "Var_Precio_Oro": "Variación (%) del Oro"}
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown(f"**Coeficiente:** {modelo_var.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_var.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_var.score(df_var[['Var_FED']], df_var['Var_Precio_Oro']):.4f}")
            
                        # --- Granger ---
            st.subheader("🔁 Test de Causalidad de Granger")

            st.markdown("¿El crecimiento del balance de la FED precede al movimiento del precio del oro?")

            max_lag = st.slider("Selecciona el número máximo de retardos (lags):", 1, 12, 6)
            df_granger = df_fed[["Var_FED", "Var_Precio_Oro"]].dropna()

            try:
                granger_res = grangercausalitytests(df_granger, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag+1):
                    pval = granger_res[lag][0]["ssr_ftest"][1]
                    st.markdown(f"- Lag {lag}: p-valor = `{pval:.4f}` → {'❗ Causalidad probable' if pval < 0.05 else 'No significativa'}")
            except Exception as e:
                st.error(f"Error al realizar el test de Granger: {e}")

            # --- Cross-Correlation ---
            st.subheader("📊 Análisis de Retardos (Cross-Correlation)")

            var_fed = (df_granger["Var_FED"] - df_granger["Var_FED"].mean()) / df_granger["Var_FED"].std()
            var_oro = (df_granger["Var_Precio_Oro"] - df_granger["Var_Precio_Oro"].mean()) / df_granger["Var_Precio_Oro"].std()

            lags = np.arange(-12, 13)
            corr_lags = [var_fed.corr(var_oro.shift(lag)) for lag in lags]

            fig_lags = go.Figure()
            fig_lags.add_trace(go.Bar(x=lags, y=corr_lags, marker_color="purple"))
            fig_lags.update_layout(
                title="Correlación Cruzada entre Variación del Balance FED y del Oro",
                xaxis_title="Retardo (semanas)",
                yaxis_title="Correlación",
                template="plotly_dark"
            )
            st.plotly_chart(fig_lags, use_container_width=True)


            
        elif seleccion_influencia == opciones_influencias[3]:


            st.subheader("💰 Influencia del PIB de EE.UU. en el Precio del Oro")

            st.markdown("""
            Este análisis examina cómo el **Producto Interior Bruto (PIB)** de Estados Unidos afecta al comportamiento del **precio del oro**.
            
            Aunque el PIB es un indicador de frecuencia anual, su evolución puede reflejarse en las expectativas económicas y monetarias que influyen en los mercados financieros, incluyendo los metales preciosos.
            """)

            # Reindexar el PIB anual al índice diario del oro usando forward fill
            pib_df_daily = pib_df.reindex(df.index, method='ffill')

            # Combinar con el precio del oro
            df_pib = df[["Precio_Oro"]].join(pib_df_daily, how="inner")

            # 📊 Visualización comparativa
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_pib.index, y=df_pib["Precio_Oro"], name="Precio del Oro", line=dict(color="gold")), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_pib.index, y=df_pib["PIB"], name="PIB USA", line=dict(color="green")), secondary_y=True)

            fig.update_layout(title="Precio del Oro vs PIB de EE.UU.", template="plotly_dark")
            fig.update_yaxes(title_text="Precio del Oro (USD/oz)", secondary_y=False)
            fig.update_yaxes(title_text="PIB (Millones de €)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # 📈 Matriz de Correlación
            st.markdown("### 📈 Matriz de Correlación")
            corr = df_pib[["Precio_Oro", "PIB", "Crecimiento_PIB"]].corr()
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"))

            # 📉 Regresión Lineal entre PIB y Precio del Oro
            st.subheader("📉 Regresión Lineal entre PIB y Precio del Oro")

            df_regresion = df_pib[["PIB", "Precio_Oro"]].dropna()

            modelo = LinearRegression()
            modelo.fit(df_regresion[["PIB"]], df_regresion["Precio_Oro"])
            df_regresion["Predicción"] = modelo.predict(df_regresion[["PIB"]])

            fig_reg = px.scatter(
                df_regresion,
                x="PIB",
                y="Precio_Oro",
                trendline="ols",
                labels={"PIB": "PIB de EE.UU. (Millones €)", "Precio_Oro": "Precio del Oro (USD/oz)"},
                title="Relación entre PIB de EE.UU. y el Precio del Oro"
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            st.markdown(f"**Coeficiente:** {modelo.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo.score(df_regresion[['PIB']], df_regresion['Precio_Oro']):.4f}")

            # 📉 Variación mensual
            st.subheader("📉 Regresión entre Variaciones de PIB y del Precio del Oro")

            df_pib["Var_PIB"] = df_pib["PIB"].pct_change() * 100
            df_pib["Var_Oro"] = df_pib["Precio_Oro"].pct_change() * 100
            df_var = df_pib[["Var_PIB", "Var_Oro"]].dropna()

            modelo_var = LinearRegression()
            modelo_var.fit(df_var[["Var_PIB"]], df_var["Var_Oro"])
            df_var["Predicción"] = modelo_var.predict(df_var[["Var_PIB"]])

            fig_var = px.scatter(
                df_var,
                x="Var_PIB",
                y="Var_Oro",
                trendline="ols",
                labels={"Var_PIB": "Variación del PIB (%)", "Var_Oro": "Variación del Precio del Oro (%)"},
                title="Variaciones del PIB vs Variaciones del Oro"
            )
            st.plotly_chart(fig_var, use_container_width=True)

            st.markdown(f"**Coeficiente:** {modelo_var.coef_[0]:.4f}")
            st.markdown(f"**Intercepto:** {modelo_var.intercept_:.2f}")
            st.markdown(f"**R² (Bondad de ajuste):** {modelo_var.score(df_var[['Var_PIB']], df_var['Var_Oro']):.4f}")

            # ⏱ Cross-correlation lag analysis
            st.subheader("⏱ Análisis de Retardos Cruzados (Lags)")

            lags = range(-12, 13)
            correlations = [df_pib["PIB"].shift(lag).corr(df_pib["Precio_Oro"]) for lag in lags]

            fig_lags = go.Figure()
            fig_lags.add_trace(go.Scatter(x=list(lags), y=correlations, mode="lines+markers", name="Correlación"))
            fig_lags.update_layout(
                title="Correlación Cruzada entre PIB y Precio del Oro (por Lags)",
                xaxis_title="Retardo (meses)",
                yaxis_title="Correlación",
                template="plotly_dark"
            )
            st.plotly_chart(fig_lags, use_container_width=True)

            # ⏳ Test de Causalidad de Granger
            st.subheader("⏳ Test de Causalidad de Granger")

            max_lag = 12
            df_test = df_pib[["Precio_Oro", "PIB"]].dropna()

            with st.expander("Ver Resultados del Test de Granger"):
                
                resultado = grangercausalitytests(df_test, maxlag=max_lag, verbose=False)

                resultados = []
                for lag in range(1, max_lag+1):
                    p_value = resultado[lag][0]["ssr_ftest"][1]
                    resultados.append({"Lag": lag, "p-valor": p_value})

                df_resultados = pd.DataFrame(resultados)
                st.dataframe(df_resultados.style.highlight_min("p-valor", color="lightgreen").format({"p-valor": "{:.4f}"}))
                st.markdown("""
                **Interpretación:** Si el p-valor es menor a 0.05, hay evidencia estadística de que el PIB Granger-causa el precio del oro con ese retardo.
                """)

            
        elif seleccion_influencia == opciones_influencias[4]:

            st.subheader("📉 Comportamiento del Oro en Periodos de Caída del S&P 500")

            st.markdown("""
            Este análisis compara el comportamiento del **oro** durante los periodos en los que el **S&P 500** experimentó caídas significativas.
            Se busca evaluar si el oro actúa como un **activo refugio** cuando el mercado bursátil atraviesa crisis o correcciones.
            """)

            # Lista de periodos de caída del S&P 500
            periodos_caida_sp500 = [
                ("2007-10-09", "2009-03-09"),
                ("2010-04-23", "2010-07-02"),
                ("2011-04-29", "2011-10-03"),
                ("2012-04-02", "2012-06-01"),
                ("2015-05-21", "2016-02-11"),
                ("2018-01-26", "2018-02-08"),
                ("2018-09-20", "2018-12-24"),
                ("2020-02-19", "2020-03-23"),
                ("2022-01-03", "2022-10-13"),
                ("2023-02-02", "2023-03-13"),
                ("2006-05-05", "2006-06-13"),
                ("2008-05-19", "2008-11-20"),
                ("2011-07-07", "2011-08-19"),
                ("2013-05-21", "2013-06-24"),
                ("2014-09-19", "2014-10-15"),
                ("2016-12-13", "2017-01-12"),
                ("2019-04-30", "2019-06-03"),
                ("2023-07-31", "2023-10-27")
            ]

            resultados = []

            for inicio, fin in periodos_caida_sp500:
                try:
                    precio_inicio = df.loc[inicio]["Precio_Oro"]
                    precio_fin = df.loc[fin]["Precio_Oro"]
                    retorno = ((precio_fin - precio_inicio) / precio_inicio) * 100

                    resultados.append({
                        "Inicio": inicio,
                        "Fin": fin,
                        "Precio Inicial": precio_inicio,
                        "Precio Final": precio_fin,
                        "Variación (%)": round(retorno, 2)
                    })
                except:
                    continue  # Por si alguna fecha no está en el dataset de oro

            df_resultados = pd.DataFrame(resultados)

            st.markdown("### 📊 Evolución del Oro en Crisis del S&P 500")
            st.dataframe(df_resultados.style.background_gradient(cmap="RdYlGn", subset=["Variación (%)"]).format({"Precio Inicial": "{:.2f}", "Precio Final": "{:.2f}", "Variación (%)": "{:.2f}"}))

            # 📈 Gráfico de barras
            fig = px.bar(
                df_resultados,
                x="Inicio",
                y="Variación (%)",
                title="Variación del Precio del Oro durante Caídas del S&P 500",
                labels={"Variación (%)": "Retorno (%)"},
                color="Variación (%)",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Resumen estadístico
            media = df_resultados["Variación (%)"].mean()
            st.markdown(f"### 🧠 Promedio de rendimiento del oro en estas caídas: **{media:.2f}%**")

            if media > 0:
                st.success("✅ El oro ha mostrado tendencia a subir durante caídas del mercado bursátil.")
            else:
                st.warning("⚠️ El oro no ha actuado consistentemente como refugio en estos periodos.")

            



    # 📉 Técnicos Avanzados
    with tabs[6]:
        st.header("📉 Indicadores Técnicos Avanzados")
        opciones_tecnicos = [
            "Detección de Divergencias entre RSI y el Precio del Oro ",
            "Detección de Divergencias MACD en el Oro ",
            "Ciclos del Oro según la Teoría de Kondratiev ",
            "Análisis de Volumen Anómalo en Oro ",
            "Análisis de Ciclos de Halving de Bitcoin y su Relación con el Oro "
        ]
        seleccion_tecnico = st.selectbox("Selecciona un análisis técnico avanzado", opciones_tecnicos)
        
        if seleccion_tecnico == opciones_tecnicos[0]:
            
            st.subheader("📉 Detección de Divergencias entre RSI y el Precio del Oro")

            st.markdown("""
            Las **divergencias** ocurren cuando el precio del activo y un indicador como el RSI se mueven en direcciones opuestas, lo que puede anticipar posibles cambios de tendencia.
            """)

            # Parámetros de usuario
            ventana_rsi = st.slider("Ventana RSI", 5, 30, 14)
            lookback = st.slider("Ventana para detectar extremos (mín/máx)", 5, 30, 14)

            # Calcular RSI
            delta = df["Precio_Oro"].diff()
            ganancia = delta.clip(lower=0)
            perdida = -delta.clip(upper=0)

            media_ganancia = ganancia.rolling(window=ventana_rsi).mean()
            media_perdida = perdida.rolling(window=ventana_rsi).mean()
            rs = media_ganancia / media_perdida
            rsi = 100 - (100 / (1 + rs))

            df["RSI"] = rsi

            # Detectar extremos
            

            df["Max_Precio"] = df["Precio_Oro"][argrelextrema(df["Precio_Oro"].values, np.greater_equal, order=lookback)[0]]
            df["Min_Precio"] = df["Precio_Oro"][argrelextrema(df["Precio_Oro"].values, np.less_equal, order=lookback)[0]]
            df["Max_RSI"] = df["RSI"][argrelextrema(df["RSI"].values, np.greater_equal, order=lookback)[0]]
            df["Min_RSI"] = df["RSI"][argrelextrema(df["RSI"].values, np.less_equal, order=lookback)[0]]



            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Precio del Oro", "RSI del Precio del Oro")
            )

            # === Subplot 1: Precio del Oro
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Precio_Oro"],
                mode="lines",
                name="Precio del Oro",
                line=dict(color="gold")
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Max_Precio"],
                mode="markers",
                name="Máximos locales",
                marker=dict(color="red", size=6),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Min_Precio"],
                mode="markers",
                name="Mínimos locales",
                marker=dict(color="green", size=6),
            ), row=1, col=1)

            # === Subplot 2: RSI
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="blue")
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Max_RSI"],
                mode="markers",
                name="Máximos RSI",
                marker=dict(color="red", size=6),
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Min_RSI"],
                mode="markers",
                name="Mínimos RSI",
                marker=dict(color="green", size=6),
            ), row=2, col=1)

            # Líneas horizontales de referencia RSI
            fig.add_shape(type="line", x0=df.index.min(), x1=df.index.max(), y0=70, y1=70,
                          line=dict(color="red", dash="dash"), row=2, col=1)
            fig.add_shape(type="line", x0=df.index.min(), x1=df.index.max(), y0=30, y1=30,
                          line=dict(color="green", dash="dash"), row=2, col=1)

            fig.update_layout(
                height=700,
                showlegend=True,
                template="plotly_dark",
                margin=dict(t=60, b=20),
                title_text="Divergencias RSI vs Precio del Oro",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            🔍 **Interpretación**:
            - Divergencia bajista: Precio hace un nuevo máximo, pero el RSI no lo confirma (máximos decrecientes).
            - Divergencia alcista: Precio hace un nuevo mínimo, pero el RSI no lo confirma (mínimos crecientes).
            Estas señales pueden anticipar **cambios de tendencia**.
            """)

            
        elif seleccion_tecnico == opciones_tecnicos[1]:


            st.subheader("📊 Detección de Divergencias entre MACD y el Precio del Oro")

            # Parámetros del MACD
            short_ema = df["Precio_Oro"].ewm(span=12, adjust=False).mean()
            long_ema = df["Precio_Oro"].ewm(span=26, adjust=False).mean()
            macd_line = short_ema - long_ema
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            df["MACD"] = macd_line
            df["Signal"] = signal_line

            # Detección de máximos/mínimos locales
            
            df["max_price"] = df["Precio_Oro"][(argrelextrema(df["Precio_Oro"].values, np.greater_equal, order=5)[0])]
            df["min_price"] = df["Precio_Oro"][(argrelextrema(df["Precio_Oro"].values, np.less_equal, order=5)[0])]
            df["max_macd"] = df["MACD"][(argrelextrema(df["MACD"].values, np.greater_equal, order=5)[0])]
            df["min_macd"] = df["MACD"][(argrelextrema(df["MACD"].values, np.less_equal, order=5)[0])]

            # Crear gráfico con Plotly
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, subplot_titles=("Precio del Oro", "MACD"))

            # --- Precio del Oro ---
            fig.add_trace(go.Scatter(x=df.index, y=df["Precio_Oro"], mode='lines', name="Precio del Oro", line=dict(color="gold")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["max_price"], mode='markers', name="Máximos Precio", marker=dict(color="red", size=6)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["min_price"], mode='markers', name="Mínimos Precio", marker=dict(color="green", size=6)), row=1, col=1)

            # --- MACD y señal ---
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name="MACD", line=dict(color="blue")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode='lines', name="Línea de Señal", line=dict(color="orange")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["max_macd"], mode='markers', name="Máximos MACD", marker=dict(color="red", size=6)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["min_macd"], mode='markers', name="Mínimos MACD", marker=dict(color="green", size=6)), row=2, col=1)

            # Layout
            fig.update_layout(
                height=700,
                title_text="📉 Divergencias MACD vs Precio del Oro",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Interpretación
            with st.expander("🔎 Interpretación"):
                st.markdown("""
                - **Divergencia bajista**: el precio hace un nuevo máximo pero el MACD no lo confirma (máximos decrecientes).
                - **Divergencia alcista**: el precio hace un nuevo mínimo pero el MACD hace un mínimo más alto.
                - Estas divergencias pueden anticipar posibles cambios de tendencia.
                """)

        elif seleccion_tecnico == opciones_tecnicos[2]:

            st.subheader("🌐 Ciclos del Oro según la Teoría de Kondratiev")
            st.markdown("""
            La Teoría de Kondratiev propone la existencia de ciclos económicos de largo plazo (~50 años), divididos en 4 fases: **expansión (primavera)**, **auge inflacionario (verano)**, **estancamiento (otoño)** y **crisis (invierno)**.
            
            En este gráfico se representan visualmente estas fases aplicadas al precio del oro para identificar posibles patrones históricos.
            """)

            # Selección de rango de años para analizar
            fecha_inicio = st.date_input("Selecciona la fecha de inicio", value=pd.to_datetime("2004-01-01"))
            fecha_fin = st.date_input("Selecciona la fecha de fin", value=df.index[-1])
            df_filtrado = df.loc[fecha_inicio:fecha_fin]

            # Definir ciclos estimados manualmente (puedes ajustarlos si tienes fechas más concretas)
            fases = [
                {"fase": "Primavera", "inicio": "2004-01-01", "fin": "2007-12-31", "color": "green"},
                {"fase": "Verano", "inicio": "2008-01-01", "fin": "2012-12-31", "color": "red"},
                {"fase": "Otoño", "inicio": "2013-01-01", "fin": "2018-12-31", "color": "orange"},
                {"fase": "Invierno", "inicio": "2019-01-01", "fin": "2024-12-31", "color": "blue"},
            ]

            fig = go.Figure()

            # Precio del oro
            fig.add_trace(go.Scatter(
                x=df_filtrado.index,
                y=df_filtrado["Precio_Oro"],
                mode="lines",
                name="Precio del Oro",
                line=dict(color="gold")
            ))

            # Añadir bandas de color por fases
            for fase in fases:
                fig.add_vrect(
                    x0=fase["inicio"], x1=fase["fin"],
                    fillcolor=fase["color"], opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=fase["fase"], annotation_position="top left"
                )

            fig.update_layout(
                title="📉 Ciclos de Kondratiev en el Precio del Oro",
                xaxis_title="Fecha",
                yaxis_title="Precio del Oro (USD/oz)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Interpretación
            with st.expander("🧠 Interpretación sugerida"):
                st.markdown("""
                - **Primavera**: Inicio del ciclo, crecimiento sostenido (p. ej., 2004–2007).
                - **Verano**: Auge e inflación, oro como refugio (pico 2011).
                - **Otoño**: Estancamiento, lateralidad o burbuja.
                - **Invierno**: Crisis global, oro tiende a subir nuevamente (post-COVID).
                """)

            
        elif seleccion_tecnico == opciones_tecnicos[3]:

            st.subheader("📊 Análisis de Volumen Anómalo en el Oro")

            st.markdown("""
            Este análisis permite detectar momentos donde el **volumen negociado** del oro ha sido **excepcionalmente alto** respecto a su comportamiento habitual.
            
            Estos picos pueden estar relacionados con eventos relevantes del mercado o movimientos institucionales.
            """)

            # Parámetros dinámicos
            ventana = st.slider("Ventana de media móvil (días):", 5, 100, 30)
            umbral_std = st.slider("Umbral de desviación estándar:", 1.0, 5.0, 2.0, step=0.1)

            # Cálculo de media y std
            df["Volumen_MM"] = df["Volumen"].rolling(window=ventana).mean()
            df["Volumen_STD"] = df["Volumen"].rolling(window=ventana).std()

            # Detección de anomalías
            df["Anomalo"] = df["Volumen"] > (df["Volumen_MM"] + umbral_std * df["Volumen_STD"])

            # Gráfico interactivo con Plotly
            fig = go.Figure()

            # Volumen normal
            fig.add_trace(go.Bar(
                x=df.index,
                y=df["Volumen"],
                name="Volumen",
                marker_color="rgba(0, 150, 255, 0.4)"
            ))

            # Volumen anómalo
            fig.add_trace(go.Scatter(
                x=df[df["Anomalo"]].index,
                y=df[df["Anomalo"]]["Volumen"],
                mode="markers",
                name="Volumen Anómalo",
                marker=dict(color="red", size=7, symbol="x")
            ))

            # Línea media móvil
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Volumen_MM"],
                mode="lines",
                name="Media Móvil Volumen",
                line=dict(color="orange", width=2)
            ))

            fig.update_layout(
                title="📈 Volumen Diario del Oro con Detección de Anomalías",
                xaxis_title="Fecha",
                yaxis_title="Volumen",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Interpretación
            with st.expander("📌 Interpretación"):
                st.markdown(f"""
                - Se considera **anómalo** un volumen que supere la media de {ventana} días más {umbral_std} desviaciones estándar.
                - Picos de volumen como estos suelen estar asociados a eventos clave, movimientos institucionales o manipulación del mercado.
                - Puedes ajustar los umbrales para hacer el filtro más o menos sensible.
                """)

            
        elif seleccion_tecnico == opciones_tecnicos[4]:

            st.subheader("⛏️ Análisis de Ciclos de Halving de Bitcoin y su Relación con el Oro")

            st.markdown("""
            Este análisis superpone los **momentos del Halving de Bitcoin** sobre el gráfico del **precio del Oro**, 
            para identificar posibles patrones cíclicos, correlaciones o reacciones compartidas.
            
            Se muestran además los retornos del Oro tras cada Halving en un plazo configurable.
            """)

            # Lista de fechas de Halving de Bitcoin
            fechas_halving = [
                "2012-11-28",  # Primer halving
                "2016-07-09",  # Segundo halving
                "2020-05-11",  # Tercer halving
                "2024-04-20"   # Cuarto halving
            ]
            fechas_halving = pd.to_datetime(fechas_halving)

            # Selección de horizonte de análisis post-halving
            meses_horizonte = st.slider("Horizonte tras Halving (meses):", 3, 48, 12)
            dias_horizonte = meses_horizonte * 30

            # Crear figura con Plotly
            fig = go.Figure()

            # Precio del oro
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Precio_Oro"],
                mode="lines",
                name="Precio del Oro",
                line=dict(color="gold", width=2)
            ))

            # Marcar los eventos de halving
            for fecha in fechas_halving:
                # Buscar la fecha más próxima en el índice (igual o posterior)
                fecha_real = df.index[df.index.get_indexer([fecha], method="bfill")][0]

                # Añadir marcador del Halving
                fig.add_trace(go.Scatter(
                    x=[fecha_real],
                    y=[df.loc[fecha_real, "Precio_Oro"]],
                    mode="markers+text",
                    name="Halving BTC",
                    text=["Halving"],
                    textposition="top center",
                    marker=dict(color="blue", size=10, symbol="diamond")
                ))

                # Zona post-halving
                fecha_fin = fecha_real + pd.Timedelta(days=dias_horizonte)
                fig.add_vrect(
                    x0=fecha_real,
                    x1=fecha_fin,
                    fillcolor="rgba(0, 100, 255, 0.1)",
                    layer="below",
                    line_width=0
                )

            fig.update_layout(
                title=f"Evolución del Precio del Oro con Ciclos de Halving de Bitcoin",
                xaxis_title="Fecha",
                yaxis_title="Precio del Oro (USD/oz)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cálculo de retorno tras cada halving
            st.markdown("### 📊 Rentabilidad del Oro tras cada Halving")
            resultados = []

            for fecha in fechas_halving:
                # Buscar la fecha más cercana al halving
                fecha_real = df.index[df.index.get_indexer([fecha], method="bfill")][0]
                precio_inicio = df.loc[fecha_real, "Precio_Oro"]

                # Fecha fin del horizonte
                fecha_post = fecha_real + pd.Timedelta(days=dias_horizonte)

                # Buscar la fecha real más cercana para el final
                if fecha_post <= df.index[-1]:
                    fecha_post_real = df.index[df.index.get_indexer([fecha_post], method="ffill")][0]
                    precio_despues = df.loc[fecha_post_real, "Precio_Oro"]
                    retorno = ((precio_despues - precio_inicio) / precio_inicio) * 100

                    resultados.append({
                        "Fecha Halving": fecha_real.strftime("%Y-%m-%d"),
                        f"Retorno {meses_horizonte}m": f"{retorno:.2f}%"
                    })

            if resultados:
                st.dataframe(pd.DataFrame(resultados))
            else:
                st.warning("No hay suficientes datos para calcular rentabilidades.")



            
    # 🔙 **Botón para volver**
    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
