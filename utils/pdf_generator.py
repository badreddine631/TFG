from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from fpdf import FPDF

try:
    import ta
except ImportError:  # pragma: no cover - la librería ta es opcional en algunos entornos
    ta = None  # type: ignore


@dataclass
class AnalysisInputs:
    """Agrupa todos los datasets necesarios para construir los análisis."""

    df: pd.DataFrame
    eventos_df: pd.DataFrame
    dolar_df: pd.DataFrame
    df_oro_ext: pd.DataFrame
    df_btc: pd.DataFrame
    df_plata: pd.DataFrame
    df_oro_eur: pd.DataFrame
    df_oro_cny: pd.DataFrame
    df_petroleo: pd.DataFrame
    df_cad: pd.DataFrame
    df_elecciones_usa: pd.DataFrame
    df_recesiones: pd.DataFrame
    df_crisis_europa: pd.DataFrame
    df_crisis_inmo: pd.DataFrame
    df_crisis_minerales: pd.DataFrame
    cpi_df: pd.DataFrame
    df_oro_mensual: pd.DataFrame
    cci_df: pd.DataFrame
    fear_greed_df: pd.DataFrame
    df_walcl: pd.DataFrame
    pib_df: pd.DataFrame


@dataclass
class AnalysisResult:
    """Representa la salida de un análisis concreto."""

    title: str
    description: str
    table: Optional[pd.DataFrame] = None


AnalysisCallable = Callable[[AnalysisInputs], AnalysisResult]


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def _as_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Fecha" in df.columns:
            df = df.copy()
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df.set_index("Fecha")
        else:
            raise ValueError("El DataFrame proporcionado debe tener índice de fechas o columna 'Fecha'.")
    return df.sort_index()


def _format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def _safe_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Funciones de análisis
# ---------------------------------------------------------------------------

def _analysis_volatilidad_eventos(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    eventos = inputs.eventos_df.copy()

    resultados: List[Dict[str, object]] = []
    for _, evento in eventos.iterrows():
        fecha_evento = pd.to_datetime(evento.get("Fecha"))
        if pd.isna(fecha_evento):
            continue
        ventana_antes = df.loc[fecha_evento - pd.Timedelta(days=7) : fecha_evento - pd.Timedelta(days=1)]
        ventana_despues = df.loc[fecha_evento + pd.Timedelta(days=1) : fecha_evento + pd.Timedelta(days=7)]

        vol_antes = _safe_pct_change(ventana_antes.get("Precio_Oro", pd.Series(dtype=float))).std() * 100
        vol_despues = _safe_pct_change(ventana_despues.get("Precio_Oro", pd.Series(dtype=float))).std() * 100

        resultados.append(
            {
                "Evento": evento.get("Evento", "Desconocido"),
                "Fecha": fecha_evento.date(),
                "Categoría": evento.get("Categoría", "General"),
                "Volatilidad Antes (%)": vol_antes,
                "Volatilidad Después (%)": vol_despues,
            }
        )

    tabla = pd.DataFrame(resultados)
    resumen = "Analiza la volatilidad de 7 días alrededor de cada evento macroeconómico registrado."
    return AnalysisResult(
        title="Volatilidad antes y después de eventos",
        description=resumen,
        table=tabla,
    )


def _analysis_volatilidad_mensual(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    retornos = _safe_pct_change(df["Precio_Oro"]).dropna()
    volatilidad_mensual = retornos.groupby(pd.Grouper(freq="M")).std() * 100
    tabla = (
        volatilidad_mensual.sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    tabla.columns = ["Mes", "Volatilidad (%)"]
    descripcion = (
        "Desviación estándar mensual de los retornos diarios del oro. Se muestran los 12 meses más volátiles del periodo."
    )
    return AnalysisResult(
        title="Volatilidad mensual del precio del oro",
        description=descripcion,
        table=tabla,
    )


def _analysis_rafagas_volumen(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    if "Volumen" not in df.columns:
        return AnalysisResult(
            title="Ráfagas de volumen",
            description="El dataset de oro no contiene la columna 'Volumen', por lo que no es posible evaluar ráfagas.",
            table=None,
        )
    umbral = df["Volumen"].quantile(0.95)
    rafagas = df[df["Volumen"] >= umbral]
    porcentaje = (len(rafagas) / len(df)) * 100 if len(df) else np.nan
    tabla = rafagas[["Precio_Oro", "Volumen"]].tail(20).reset_index()
    tabla.columns = ["Fecha", "Precio Oro", "Volumen"]
    descripcion = (
        "Identifica los días en los que el volumen supera el percentil 95 del histórico. "
        f"Estas sesiones representan el {porcentaje:.2f}% del total y pueden anticipar movimientos significativos."
    )
    return AnalysisResult(
        title="Detección de ráfagas de volumen",
        description=descripcion,
        table=tabla,
    )


def _analysis_media_movil(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    precios = df["Precio_Oro"].copy()
    sma_50 = precios.rolling(window=50).mean()
    sma_200 = precios.rolling(window=200).mean()
    cruces = (sma_50 > sma_200).astype(int).diff()
    ult_cruce = cruces.dropna().iloc[-1] if not cruces.dropna().empty else np.nan
    tipo_cruce = "alcista" if ult_cruce == 1 else "bajista" if ult_cruce == -1 else "sin cambios recientes"
    fecha_cruce = cruces.dropna().index[-1].date() if not cruces.dropna().empty else "N/A"
    distancia = ((precios.iloc[-1] / sma_200.iloc[-1]) - 1) * 100 if not pd.isna(sma_200.iloc[-1]) else np.nan
    descripcion = (
        "Compara la media móvil de 50 sesiones con la de 200 para detectar la tendencia dominante. "
        f"El último cruce fue {tipo_cruce} el {fecha_cruce}. El precio actual está a {_format_percentage(distancia)} de la SMA200."
    )
    tabla = pd.DataFrame(
        {
            "Fecha": precios.tail(5).index.date,
            "Precio": precios.tail(5).values,
            "SMA 50": sma_50.tail(5).values,
            "SMA 200": sma_200.tail(5).values,
        }
    )
    return AnalysisResult(
        title="Media móvil 50/200 sesiones",
        description=descripcion,
        table=tabla,
    )


def _analysis_correlacion_dxy(inputs: AnalysisInputs) -> AnalysisResult:
    oro = _as_datetime_index(inputs.df)["Precio_Oro"].copy()
    dolar = _as_datetime_index(inputs.dolar_df)["DXY"].copy()
    df = pd.concat([oro, dolar], axis=1, join="inner").dropna()
    retornos = df.pct_change().dropna()
    correlacion = retornos["Precio_Oro"].corr(retornos["DXY"])
    descripcion = (
        "Calcula la correlación entre los retornos diarios del oro y el índice dólar (DXY). "
        f"La correlación histórica es {correlacion:.2f}, donde valores negativos indican que el oro tiende a subir cuando el dólar se debilita."
    )
    tabla = retornos.tail(10).reset_index()
    tabla.columns = ["Fecha", "Retorno Oro", "Retorno DXY"]
    return AnalysisResult(
        title="Correlación Oro vs DXY",
        description=descripcion,
        table=tabla,
    )


def _analysis_oro_vs_btc(inputs: AnalysisInputs) -> AnalysisResult:
    oro = _as_datetime_index(inputs.df)["Precio_Oro"].copy()
    btc = _as_datetime_index(inputs.df_btc)["Precio_BTC"].copy()
    combinado = pd.concat([oro, btc], axis=1, join="inner").dropna()
    retornos = combinado.pct_change().dropna()
    correlacion = retornos["Precio_Oro"].corr(retornos["Precio_BTC"])
    ratio_actual = combinado["Precio_Oro"].iloc[-1] / combinado["Precio_BTC"].iloc[-1]
    descripcion = (
        "Evalúa la relación entre el oro y Bitcoin a través de la correlación de retornos diarios. "
        f"La correlación es {correlacion:.2f}. El ratio precio oro / precio BTC actual es {ratio_actual:.4f}."
    )
    tabla = retornos.tail(10).reset_index()
    tabla.columns = ["Fecha", "Retorno Oro", "Retorno BTC"]
    return AnalysisResult(
        title="Comparativa Oro vs Bitcoin",
        description=descripcion,
        table=tabla,
    )


def _analysis_backtest_medias(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    precios = df["Precio_Oro"].copy()
    retornos = precios.pct_change().fillna(0)
    sma_corta = precios.rolling(50).mean()
    sma_larga = precios.rolling(200).mean()
    posicion = (sma_corta > sma_larga).astype(int)
    estrategia = (1 + (posicion.shift(1).fillna(0) * retornos)).cumprod()
    buy_hold = (1 + retornos).cumprod()
    resumen = pd.DataFrame(
        {
            "Metric": ["Retorno Estrategia", "Retorno Buy & Hold", "Max Drawdown Estrategia", "Max Drawdown Buy & Hold"],
            "Valor": [
                estrategia.iloc[-1] - 1,
                buy_hold.iloc[-1] - 1,
                (estrategia.cummax() - estrategia).max(),
                (buy_hold.cummax() - buy_hold).max(),
            ],
        }
    )
    descripcion = (
        "Backtest de una estrategia que permanece invertida cuando la SMA50 está por encima de la SMA200. "
        "Se compara frente a mantener oro de forma continua."
    )
    return AnalysisResult(
        title="Backtest cruce de medias 50/200",
        description=descripcion,
        table=resumen,
    )


def _analysis_dca(inputs: AnalysisInputs) -> AnalysisResult:
    precios_mensuales = _as_datetime_index(inputs.df)["Precio_Oro"].resample("M").last().dropna()
    inversion_mensual = 100.0
    unidades = (inversion_mensual / precios_mensuales).fillna(0)
    acumuladas = unidades.cumsum()
    capital_invertido = inversion_mensual * len(precios_mensuales)
    valor_final = acumuladas.iloc[-1] * precios_mensuales.iloc[-1]
    rentabilidad = (valor_final / capital_invertido - 1) * 100 if capital_invertido else np.nan
    descripcion = (
        "Simula un plan de inversión periódica (DCA) aportando 100 USD al oro cada mes. "
        f"El capital invertido asciende a {capital_invertido:,.0f} USD y el valor final a {valor_final:,.0f} USD, "
        f"lo que supone una rentabilidad del {_format_percentage(rentabilidad)}."
    )
    tabla = pd.DataFrame(
        {
            "Fecha": precios_mensuales.tail(12).index.to_period("M").astype(str),
            "Precio": precios_mensuales.tail(12).values,
            "Unidades adquiridas": unidades.tail(12).values,
        }
    )
    return AnalysisResult(
        title="Simulación Dollar-Cost Averaging",
        description=descripcion,
        table=tabla,
    )


def _analysis_momentum(inputs: AnalysisInputs) -> AnalysisResult:
    precios_mensuales = _as_datetime_index(inputs.df)["Precio_Oro"].resample("M").last().dropna()
    momentum = precios_mensuales.pct_change(periods=12)
    retornos = precios_mensuales.pct_change().fillna(0)
    posicion = (momentum > 0).astype(int)
    estrategia = (1 + posicion.shift(1).fillna(0) * retornos).cumprod()
    buy_hold = (1 + retornos).cumprod()
    tabla = pd.DataFrame(
        {
            "Serie": ["Estrategia Momentum", "Buy & Hold"],
            "Retorno acumulado (%)": [(estrategia.iloc[-1] - 1) * 100, (buy_hold.iloc[-1] - 1) * 100],
        }
    )
    descripcion = (
        "Estrategia mensual que permanece invertida cuando el retorno a 12 meses del oro es positivo. "
        "Se compara la rentabilidad acumulada frente a comprar y mantener."
    )
    return AnalysisResult(
        title="Estrategia de Momentum 12M",
        description=descripcion,
        table=tabla,
    )


def _analysis_eventos_cambios(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    eventos = inputs.eventos_df.copy()
    resultados: List[Dict[str, object]] = []
    for _, evento in eventos.iterrows():
        fecha_evento = pd.to_datetime(evento.get("Fecha"))
        if pd.isna(fecha_evento):
            continue
        ventana = df.loc[fecha_evento - pd.Timedelta(days=5) : fecha_evento + pd.Timedelta(days=5)]
        if ventana.empty:
            continue
        precio_antes = ventana.iloc[0]["Precio_Oro"]
        precio_despues = ventana.iloc[-1]["Precio_Oro"]
        cambio = (precio_despues / precio_antes - 1) * 100
        resultados.append(
            {
                "Evento": evento.get("Evento", "Desconocido"),
                "Fecha": fecha_evento.date(),
                "Cambio 5d (%)": cambio,
            }
        )
    tabla = pd.DataFrame(resultados)
    descripcion = "Cambios porcentuales del oro cinco días antes y después de cada evento macroeconómico registrado."
    return AnalysisResult(
        title="Impacto de eventos macroeconómicos",
        description=descripcion,
        table=tabla,
    )


def _analysis_recesiones(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    recesiones = inputs.df_recesiones.copy()
    resultados: List[Dict[str, object]] = []
    for _, recesion in recesiones.iterrows():
        inicio = pd.to_datetime(recesion.get("Fecha_Inicio"))
        fin = pd.to_datetime(recesion.get("Fecha_Fin"))
        if pd.isna(inicio) or pd.isna(fin):
            continue
        rango = df.loc[inicio:fin]
        if rango.empty:
            continue
        retorno = (rango["Precio_Oro"].iloc[-1] / rango["Precio_Oro"].iloc[0] - 1) * 100
        resultados.append(
            {
                "Recesión": recesion.get("Nombre", "Recesión"),
                "Inicio": inicio.date(),
                "Fin": fin.date(),
                "Retorno Oro (%)": retorno,
            }
        )
    tabla = pd.DataFrame(resultados)
    descripcion = "Retornos del oro durante las recesiones globales del periodo analizado."
    return AnalysisResult(
        title="Comportamiento del oro en recesiones",
        description=descripcion,
        table=tabla,
    )


def _analysis_crisis_europa(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    crisis = inputs.df_crisis_europa.copy()
    resultados: List[Dict[str, object]] = []
    for _, evento in crisis.iterrows():
        inicio = pd.to_datetime(evento.get("Fecha de Inicio"))
        fin = pd.to_datetime(evento.get("Fecha de Fin"))
        if pd.isna(inicio) or pd.isna(fin):
            continue
        rango = df.loc[inicio:fin]
        if rango.empty:
            continue
        retorno = (rango["Precio_Oro"].iloc[-1] / rango["Precio_Oro"].iloc[0] - 1) * 100
        resultados.append(
            {
                "Crisis": evento.get("Evento", "Crisis"),
                "Inicio": inicio.date(),
                "Fin": fin.date(),
                "Retorno Oro (%)": retorno,
            }
        )
    tabla = pd.DataFrame(resultados)
    descripcion = "Evaluación de las principales crisis financieras europeas y su efecto en el precio del oro."
    return AnalysisResult(
        title="Impacto de crisis financieras europeas",
        description=descripcion,
        table=tabla,
    )


def _analysis_distribucion_retornos(inputs: AnalysisInputs) -> AnalysisResult:
    retornos = _safe_pct_change(_as_datetime_index(inputs.df)["Precio_Oro"]).dropna()
    tabla = pd.DataFrame(
        {
            "Métrica": ["Media diaria", "Mediana", "Volatilidad", "Curtosis", "Asimetría"],
            "Valor": [
                retornos.mean() * 100,
                retornos.median() * 100,
                retornos.std() * 100,
                retornos.kurtosis(),
                retornos.skew(),
            ],
        }
    )
    descripcion = "Estadísticas descriptivas de los retornos diarios del oro."

    return AnalysisResult(
        title="Distribución de retornos diarios",
        description=descripcion,
        table=tabla,
    )


def _analysis_top_bottom_days(inputs: AnalysisInputs) -> AnalysisResult:
    retornos = _safe_pct_change(_as_datetime_index(inputs.df)["Precio_Oro"]).dropna()
    mejores = retornos.nlargest(5).reset_index()
    peores = retornos.nsmallest(5).reset_index()
    mejores["Tipo"] = "Subidas"
    peores["Tipo"] = "Caídas"
    tabla = pd.concat([mejores, peores], ignore_index=True)
    tabla.columns = ["Fecha", "Retorno", "Tipo"]
    tabla["Retorno"] = tabla["Retorno"] * 100
    descripcion = "Días con los mayores movimientos porcentuales del precio del oro."
    return AnalysisResult(
        title="Días con mayores subidas y bajadas",
        description=descripcion,
        table=tabla,
    )


def _analysis_retornos_semanales(inputs: AnalysisInputs) -> AnalysisResult:
    retornos_semanales = _as_datetime_index(inputs.df)["Precio_Oro"].resample("W").last().pct_change().dropna()
    tabla = pd.DataFrame(
        {
            "Métrica": ["Retorno medio semanal", "Volatilidad semanal"],
            "Valor": [retornos_semanales.mean() * 100, retornos_semanales.std() * 100],
        }
    )
    descripcion = "Resumen estadístico de los retornos semanales del oro."
    return AnalysisResult(
        title="Retornos semanales del oro",
        description=descripcion,
        table=tabla,
    )


def _analysis_cpi(inputs: AnalysisInputs) -> AnalysisResult:
    oro = _as_datetime_index(inputs.df)["Precio_Oro"].resample("M").last()
    ipc = _as_datetime_index(inputs.cpi_df)["IPC"]
    combinado = pd.concat([oro, ipc], axis=1, join="inner").dropna()
    retornos_oro = combinado["Precio_Oro"].pct_change()
    variacion_ipc = combinado["IPC"].pct_change()
    correlacion = retornos_oro.corr(variacion_ipc)
    descripcion = (
        "Se calcula la variación porcentual mensual del IPC y su correlación con los retornos del oro. "
        f"La correlación es {correlacion:.2f}."
    )
    tabla = pd.DataFrame(
        {
            "Métrica": ["Correlación", "Retorno medio oro (%)", "Variación media IPC (%)"],
            "Valor": [
                correlacion,
                retornos_oro.mean() * 100,
                variacion_ipc.mean() * 100,
            ],
        }
    )
    return AnalysisResult(
        title="Relación Oro - IPC",
        description=descripcion,
        table=tabla,
    )


def _analysis_cci(inputs: AnalysisInputs) -> AnalysisResult:
    oro = _as_datetime_index(inputs.df)["Precio_Oro"].resample("M").last()
    cci = _as_datetime_index(inputs.cci_df).iloc[:, 0]
    combinado = pd.concat([oro, cci], axis=1, join="inner").dropna()
    retornos_oro = combinado["Precio_Oro"].pct_change()
    variacion_cci = combinado.iloc[:, 1].pct_change()
    correlacion = retornos_oro.corr(variacion_cci)
    descripcion = (
        "Correlación entre la confianza del consumidor y los retornos del oro. "
        f"La correlación de variaciones mensuales es {correlacion:.2f}."
    )
    tabla = combinado.tail(12).reset_index()
    tabla.columns = ["Fecha", "Precio Oro", "CCI"]
    return AnalysisResult(
        title="Influencia del CCI en el oro",
        description=descripcion,
        table=tabla,
    )


def _analysis_fear_greed(inputs: AnalysisInputs) -> AnalysisResult:
    oro = _as_datetime_index(inputs.df)["Precio_Oro"]
    fear = _as_datetime_index(inputs.fear_greed_df)["Fear_Greed"]
    combinado = pd.concat([oro, fear], axis=1, join="inner").dropna()
    retornos = combinado["Precio_Oro"].pct_change()
    variacion_fear = combinado["Fear_Greed"].pct_change()
    correlacion = retornos.corr(variacion_fear)
    descripcion = (
        "Relación entre los cambios del índice Fear & Greed y los retornos diarios del oro. "
        f"La correlación estimada es {correlacion:.2f}."
    )
    tabla = combinado.tail(15).reset_index()
    tabla.columns = ["Fecha", "Precio Oro", "Fear & Greed"]
    return AnalysisResult(
        title="Influencia del Fear & Greed Index",
        description=descripcion,
        table=tabla,
    )


def _analysis_rsi(inputs: AnalysisInputs) -> AnalysisResult:
    if ta is None:
        return AnalysisResult(
            title="Análisis RSI",
            description="La librería 'ta' no está disponible en el entorno y es necesaria para calcular el RSI.",
            table=None,
        )
    df = _as_datetime_index(inputs.df)
    rsi = ta.momentum.RSIIndicator(df["Precio_Oro"], window=14).rsi()
    sobrecompra = (rsi > 70).sum()
    sobreventa = (rsi < 30).sum()
    descripcion = (
        "Cálculo del RSI (14) para identificar periodos de sobrecompra y sobreventa. "
        f"Se registran {sobrecompra} sesiones en sobrecompra y {sobreventa} en sobreventa."
    )
    tabla = pd.DataFrame({"Fecha": rsi.tail(10).index, "RSI": rsi.tail(10).values})
    tabla["Fecha"] = tabla["Fecha"].dt.date
    return AnalysisResult(
        title="Indicador RSI",
        description=descripcion,
        table=tabla,
    )


def _analysis_macd(inputs: AnalysisInputs) -> AnalysisResult:
    if ta is None:
        return AnalysisResult(
            title="Análisis MACD",
            description="La librería 'ta' no está disponible en el entorno y es necesaria para calcular el MACD.",
            table=None,
        )
    df = _as_datetime_index(inputs.df)
    indicador = ta.trend.MACD(df["Precio_Oro"], window_slow=26, window_fast=12, window_sign=9)
    macd = indicador.macd()
    signal = indicador.macd_signal()
    hist = indicador.macd_diff()
    cruces = np.sign(macd - signal).diff()
    ultimo_cruce = cruces.dropna().iloc[-1] if not cruces.dropna().empty else 0
    tipo_cruce = "alcista" if ultimo_cruce > 0 else "bajista" if ultimo_cruce < 0 else "sin cruce reciente"
    descripcion = (
        "Indicador MACD (12-26-9) aplicado al precio del oro. "
        f"El último cruce MACD/señal fue {tipo_cruce}."
    )
    tabla = pd.DataFrame(
        {
            "Fecha": macd.tail(10).index.date,
            "MACD": macd.tail(10).values,
            "Señal": signal.tail(10).values,
            "Histograma": hist.tail(10).values,
        }
    )
    return AnalysisResult(
        title="Indicador MACD",
        description=descripcion,
        table=tabla,
    )


def _analysis_fibonacci(inputs: AnalysisInputs) -> AnalysisResult:
    df = _as_datetime_index(inputs.df)
    ultimo_ano = df.last("365D")
    if ultimo_ano.empty:
        ultimo_ano = df
    maximo = ultimo_ano["Precio_Oro"].max()
    minimo = ultimo_ano["Precio_Oro"].min()
    niveles = {
        "Nivel 0%": maximo,
        "Nivel 23.6%": maximo - 0.236 * (maximo - minimo),
        "Nivel 38.2%": maximo - 0.382 * (maximo - minimo),
        "Nivel 50%": (maximo + minimo) / 2,
        "Nivel 61.8%": maximo - 0.618 * (maximo - minimo),
        "Nivel 100%": minimo,
    }
    tabla = pd.DataFrame(list(niveles.items()), columns=["Nivel", "Precio"])
    descripcion = (
        "Niveles de retroceso de Fibonacci calculados con el rango de precios del último año. "
        "Permiten identificar zonas potenciales de soporte y resistencia."
    )
    return AnalysisResult(
        title="Niveles de Fibonacci",
        description=descripcion,
        table=tabla,
    )


# ---------------------------------------------------------------------------
# Catálogo de análisis disponibles
# ---------------------------------------------------------------------------

ANALYSIS_CATALOG: Dict[str, List[Dict[str, object]]] = {
    "Volatilidad": [
        {
            "id": "vol_eventos",
            "name": "Volatilidad antes/después de eventos",
            "description": "Compara la volatilidad de 7 días antes y después de los eventos macroeconómicos.",
            "func": _analysis_volatilidad_eventos,
        },
        {
            "id": "vol_mensual",
            "name": "Volatilidad mensual",
            "description": "Desviación estándar mensual de los retornos diarios del oro.",
            "func": _analysis_volatilidad_mensual,
        },
        {
            "id": "vol_rafagas",
            "name": "Ráfagas de volumen",
            "description": "Detecta sesiones con volumen excepcionalmente alto.",
            "func": _analysis_rafagas_volumen,
        },
    ],
    "Tendencias": [
        {
            "id": "tend_sma",
            "name": "Medias móviles 50/200",
            "description": "Evalúa la tendencia principal mediante medias móviles.",
            "func": _analysis_media_movil,
        },
        {
            "id": "tend_dxy",
            "name": "Correlación Oro-DXY",
            "description": "Mide la relación entre el oro y el índice dólar.",
            "func": _analysis_correlacion_dxy,
        },
        {
            "id": "tend_btc",
            "name": "Oro vs Bitcoin",
            "description": "Analiza la interacción entre el oro y Bitcoin.",
            "func": _analysis_oro_vs_btc,
        },
    ],
    "Estrategias": [
        {
            "id": "strat_medias",
            "name": "Backtest cruces de medias",
            "description": "Estrategia basada en SMA50/SMA200.",
            "func": _analysis_backtest_medias,
        },
        {
            "id": "strat_dca",
            "name": "Simulación DCA",
            "description": "Aporte periódico mensual de capital.",
            "func": _analysis_dca,
        },
        {
            "id": "strat_momentum",
            "name": "Momentum 12M",
            "description": "Permanece invertido cuando el momentum a 12 meses es positivo.",
            "func": _analysis_momentum,
        },
    ],
    "Eventos": [
        {
            "id": "eventos_cambios",
            "name": "Impacto de eventos",
            "description": "Retornos alrededor de eventos macroeconómicos.",
            "func": _analysis_eventos_cambios,
        },
        {
            "id": "eventos_recesiones",
            "name": "Recesiones globales",
            "description": "Comportamiento del oro en recesiones.",
            "func": _analysis_recesiones,
        },
        {
            "id": "eventos_crisis_europa",
            "name": "Crisis financieras europeas",
            "description": "Impacto de crisis europeas en el oro.",
            "func": _analysis_crisis_europa,
        },
    ],
    "Cuantitativo": [
        {
            "id": "quant_distribucion",
            "name": "Distribución de retornos",
            "description": "Estadísticas descriptivas de retornos diarios.",
            "func": _analysis_distribucion_retornos,
        },
        {
            "id": "quant_top_bottom",
            "name": "Top/Bottom días",
            "description": "Mayores subidas y bajadas del periodo.",
            "func": _analysis_top_bottom_days,
        },
        {
            "id": "quant_retornos_semanales",
            "name": "Retornos semanales",
            "description": "Resumen de retornos a frecuencia semanal.",
            "func": _analysis_retornos_semanales,
        },
    ],
    "Influencias": [
        {
            "id": "infl_ipc",
            "name": "Inflación (IPC)",
            "description": "Relación entre IPC y precio del oro.",
            "func": _analysis_cpi,
        },
        {
            "id": "infl_cci",
            "name": "Confianza del consumidor",
            "description": "Interacción entre CCI y oro.",
            "func": _analysis_cci,
        },
        {
            "id": "infl_fear_greed",
            "name": "Fear & Greed Index",
            "description": "Influencia del sentimiento de mercado.",
            "func": _analysis_fear_greed,
        },
    ],
    "Técnicos": [
        {
            "id": "tec_rsi",
            "name": "Indicador RSI",
            "description": "Detección de sobrecompra y sobreventa.",
            "func": _analysis_rsi,
        },
        {
            "id": "tec_macd",
            "name": "Indicador MACD",
            "description": "Cruces de tendencia del MACD.",
            "func": _analysis_macd,
        },
        {
            "id": "tec_fibonacci",
            "name": "Niveles Fibonacci",
            "description": "Zonas potenciales de soporte/resistencia.",
            "func": _analysis_fibonacci,
        },
    ],
}


def get_available_analyses() -> Dict[str, List[Dict[str, object]]]:
    """Devuelve el catálogo completo de análisis disponibles."""

    return ANALYSIS_CATALOG


# ---------------------------------------------------------------------------
# Generación de PDF
# ---------------------------------------------------------------------------

def _add_section(pdf: FPDF, result: AnalysisResult) -> None:
    pdf.set_font("Arial", style="B", size=14)
    pdf.multi_cell(0, 10, result.title)
    pdf.ln(1)
    pdf.set_font("Arial", size=11)
    for paragraph in result.description.split("\n"):
        pdf.multi_cell(0, 6, paragraph)
    pdf.ln(2)
    if result.table is not None and not result.table.empty:
        pdf.set_font("Courier", size=9)
        table_str = result.table.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        for line in table_str.split("\n"):
            pdf.multi_cell(0, 5, line)
        pdf.ln(3)


def generar_informe(
    inputs: AnalysisInputs,
    selected_analysis_ids: Iterable[str],
    output_path: str = "informe_analisis.pdf",
) -> str:
    """Genera un informe PDF con los análisis seleccionados."""

    selected_ids = list(dict.fromkeys(selected_analysis_ids))
    if not selected_ids:
        raise ValueError("Debe seleccionar al menos un análisis para generar el informe.")

    id_to_callable: Dict[str, AnalysisCallable] = {
        item["id"]: item["func"]  # type: ignore[index]
        for seccion in ANALYSIS_CATALOG.values()
        for item in seccion
    }

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, txt="Informe de Análisis del Oro", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(
        0,
        6,
        "Este informe ha sido generado automáticamente desde MacroGold Analytics a partir de los módulos "
        "de Información Basada en Datos."
    )
    pdf.ln(5)

    for analysis_id in selected_ids:
        if analysis_id not in id_to_callable:
            continue
        resultado = id_to_callable[analysis_id](inputs)
        _add_section(pdf, resultado)

    pdf.output(output_path)
    return output_path
