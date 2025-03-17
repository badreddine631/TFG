import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar los datos
df = pd.read_csv("C:/Users/Usuario/Desktop/TFFFG/price-gold/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
eventos_df = pd.read_csv("C:/Users/Usuario/Desktop/TFFFG/price-gold/eventos_macroeconomicos.csv", parse_dates=["Fecha"])

# Asegurarnos de que las columnas estén correctamente configuradas
df = df.rename(columns={"Date": "Fecha"})
df.set_index("Fecha", inplace=True)

# Calcular el impacto promedio antes y después de los eventos
impacto_eventos = []
for _, evento in eventos_df.iterrows():
    fecha_evento = evento['Fecha']
    nombre_evento = evento['Evento']

    # Seleccionar rangos de 7 días antes y después del evento
    rango_antes = df[(df.index >= fecha_evento - pd.Timedelta(days=7)) & (df.index < fecha_evento)]
    rango_despues = df[(df.index > fecha_evento) & (df.index <= fecha_evento + pd.Timedelta(days=7))]

    promedio_antes = rango_antes['Close'].mean()
    promedio_despues = rango_despues['Close'].mean()

    # Calcular el cambio porcentual
    cambio = ((promedio_despues - promedio_antes) / promedio_antes) * 100

    # Guardar los resultados
    impacto_eventos.append({
        'Evento': nombre_evento,
        'Fecha': fecha_evento,
        'Categoría': evento['Categoría'],
        'Cambio (%)': cambio
    })

# Convertir a DataFrame el impacto calculado
impacto_eventos_df = pd.DataFrame(impacto_eventos)

# Título del dashboard
st.title("Dashboard Interactivo: Análisis del Oro y Eventos Macroeconómicos")

# Sección de selección de rango de fechas
st.sidebar.header("Seleccione un Rango de Fechas")
start_date = st.sidebar.date_input("Fecha de Inicio", value=df.index.min())
end_date = st.sidebar.date_input("Fecha de Fin", value=df.index.max())

# Filtrar datos por rango de fechas
df_filtrado = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

# Gráfico interactivo del precio del oro
st.subheader("Precio del Oro a lo Largo del Tiempo")
fig_precio = px.line(df_filtrado, x=df_filtrado.index, y="Close", title="Precio del Oro")
fig_precio.update_layout(xaxis_title="Fecha", yaxis_title="Precio de Cierre (USD)")
st.plotly_chart(fig_precio)

# Gráfico de impacto de eventos macroeconómicos
st.subheader("Impacto de Eventos Macroeconómicos")
if 'Cambio (%)' in impacto_eventos_df.columns:
    fig_eventos = px.bar(impacto_eventos_df, x="Evento", y="Cambio (%)", color="Categoría",
                         title="Impacto de Eventos Macroeconómicos", text="Cambio (%)")
    fig_eventos.update_layout(xaxis_title="Evento", yaxis_title="Cambio Promedio (%)")
    st.plotly_chart(fig_eventos)
else:
    st.error("No se encontró la columna 'Cambio (%)'. Verifica los cálculos.")

# Mostrar la tabla de eventos e impactos
st.subheader("Tabla de Impacto de Eventos")
st.dataframe(impacto_eventos_df)

# Nota final
st.info("Este dashboard permite analizar el impacto de eventos macroeconómicos seleccionados sobre el precio del oro.")
