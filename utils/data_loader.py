import pandas as pd

def load_data():
    """Carga y procesa todos los datos necesarios para la aplicación."""
    
    # === 1. Oro principal (2004-2024) ===
    df = pd.read_csv("datasets/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])

    # Renombrar y preparar índice
    df = df.rename(columns={
        "Date": "Fecha",
        "Close": "Precio_Oro",
        "Volume": "Volumen"  # Aseguramos que el nombre sea uniforme
    })

    # Convertir columna de volumen a numérico (por si acaso viene como texto)
    df["Volumen"] = pd.to_numeric(df["Volumen"], errors="coerce")

    # Establecer la fecha como índice
    df.set_index("Fecha", inplace=True)
    
        # === Oro mensual ===
    df_oro_mensual = pd.read_csv("datasets/XAU_1Month_data_2004_to_2024-09-20.csv", parse_dates=["Date"])

    # Renombrar columnas clave
    df_oro_mensual = df_oro_mensual.rename(columns={
        "Date": "Fecha",
        "Close": "Precio_Oro",
        "Volume": "Volumen"
    })

    # Asegurar que el volumen es numérico
    df_oro_mensual["Volumen"] = pd.to_numeric(df_oro_mensual["Volumen"], errors="coerce")

    # Establecer la fecha como índice
    df_oro_mensual.set_index("Fecha", inplace=True)

  
    
    
    # === Plata (XAG/USD) ===
    df_plata_raw = pd.read_csv("datasets/Datos históricos XAG_USD.csv", sep=",", encoding="utf-8")

    # Limpiar nombres de columnas
    df_plata_raw.columns = df_plata_raw.columns.str.strip().str.replace('"', '').str.replace("﻿", "")

    # Convertir la columna de fecha
    df_plata_raw["Fecha"] = pd.to_datetime(df_plata_raw["Fecha"], dayfirst=True)
    df_plata_raw = df_plata_raw.sort_values("Fecha")

    # Quedarse solo con columnas necesarias
    df_plata = df_plata_raw[["Fecha", "Último"]].rename(columns={"Último": "Precio_Plata"})

    # 🔁 Convertir precios al formato numérico correcto (coma decimal)
    df_plata["Precio_Plata"] = df_plata["Precio_Plata"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_plata["Precio_Plata"] = pd.to_numeric(df_plata["Precio_Plata"], errors="coerce")

    # Filtrar solo precios válidos
    df_plata = df_plata[(df_plata["Precio_Plata"] > 0) & (~df_plata["Precio_Plata"].isna())]

    # Indexar por fecha
    df_plata.set_index("Fecha", inplace=True)



    # === 2. Eventos macroeconómicos ===
    eventos_df = pd.read_csv("datasets/eventos_macroeconomicos.csv", parse_dates=["Fecha"])
    

    # === 3. Dólar (Índice DXY) ===
    dolar_df = pd.read_csv("datasets/Datos_historicos_Indice_dolar.csv", parse_dates=["Fecha"], dayfirst=True)
    dolar_df = dolar_df.rename(columns={"Último": "DXY"})
    dolar_df["DXY"] = pd.to_numeric(dolar_df["DXY"].astype(str).str.replace(",", "."), errors="coerce")
    dolar_df.dropna(subset=["DXY"], inplace=True)
    dolar_df.set_index("Fecha", inplace=True)

    # === 4. Oro adicional (2010+) ===
    df_oro_ext = pd.read_csv("datasets/Datos históricos XAU_USD_2010.csv", sep=",", encoding="utf-8")
    df_oro_ext.columns = df_oro_ext.columns.str.strip().str.replace('"', '').str.replace("﻿", "")
    df_oro_ext["Fecha"] = pd.to_datetime(df_oro_ext["Fecha"], dayfirst=True)
    df_oro_ext = df_oro_ext.sort_values("Fecha")
    df_oro_ext = df_oro_ext[["Fecha", "Último"]].rename(columns={"Último": "Precio_Oro"})

    # 🔁 Convertir formato europeo: 94.817,37 → 94817.37
    df_oro_ext["Precio_Oro"] = df_oro_ext["Precio_Oro"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_oro_ext["Precio_Oro"] = pd.to_numeric(df_oro_ext["Precio_Oro"], errors="coerce")
    df_oro_ext = df_oro_ext[(df_oro_ext["Precio_Oro"] > 0) & (~df_oro_ext["Precio_Oro"].isna())]
    df_oro_ext.set_index("Fecha", inplace=True)

    # === 5. Bitcoin ===
    df_btc = pd.read_csv("datasets/Datos históricos BTC_USD Bitfinex.csv", sep=",", encoding="utf-8")
    df_btc.columns = df_btc.columns.str.strip().str.replace('"', '').str.replace("﻿", "")
    df_btc["Fecha"] = pd.to_datetime(df_btc["Fecha"], dayfirst=True)
    df_btc = df_btc.sort_values("Fecha")
    df_btc = df_btc[["Fecha", "Último"]].rename(columns={"Último": "Precio_BTC"})

    # 🔁 Convertir formato europeo
    df_btc["Precio_BTC"] = df_btc["Precio_BTC"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_btc["Precio_BTC"] = pd.to_numeric(df_btc["Precio_BTC"], errors="coerce")
    df_btc = df_btc[(df_btc["Precio_BTC"] > 0) & (~df_btc["Precio_BTC"].isna())]
    df_btc.set_index("Fecha", inplace=True)
    
    
        # === Oro en Euros (XAU/EUR) ===
    df_oro_eur_raw = pd.read_csv("datasets/Datos históricos XAU_EUR.csv", sep=",", encoding="utf-8")

    # Limpiar nombres de columnas
    df_oro_eur_raw.columns = df_oro_eur_raw.columns.str.strip().str.replace('"', '').str.replace("﻿", "")

    # Convertir la columna de fecha
    df_oro_eur_raw["Fecha"] = pd.to_datetime(df_oro_eur_raw["Fecha"], dayfirst=True)
    df_oro_eur_raw = df_oro_eur_raw.sort_values("Fecha")

    # Selección de columnas relevantes
    df_oro_eur = df_oro_eur_raw[["Fecha", "Último"]].rename(columns={"Último": "Precio_Oro_EUR"})

    # Convertir precios al formato numérico correcto (coma decimal → punto decimal)
    df_oro_eur["Precio_Oro_EUR"] = df_oro_eur["Precio_Oro_EUR"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_oro_eur["Precio_Oro_EUR"] = pd.to_numeric(df_oro_eur["Precio_Oro_EUR"], errors="coerce")
    df_oro_eur = df_oro_eur[(df_oro_eur["Precio_Oro_EUR"] > 0) & (~df_oro_eur["Precio_Oro_EUR"].isna())]

    # Establecer fecha como índice
    df_oro_eur.set_index("Fecha", inplace=True)
    
        # === Oro en Yuan Chino (XAU/CNY) ===
    df_oro_cny_raw = pd.read_csv("datasets/Datos históricos XAU_CNY.csv", sep=",", encoding="utf-8")

    # Limpiar nombres de columnas
    df_oro_cny_raw.columns = df_oro_cny_raw.columns.str.strip().str.replace('"', '').str.replace("﻿", "")

    # Convertir fechas y ordenar
    df_oro_cny_raw["Fecha"] = pd.to_datetime(df_oro_cny_raw["Fecha"], dayfirst=True)
    df_oro_cny_raw = df_oro_cny_raw.sort_values("Fecha")

    # Seleccionar solo columna de interés
    df_oro_cny = df_oro_cny_raw[["Fecha", "Último"]].rename(columns={"Último": "Precio_Oro_CNY"})

    # Convertir a formato numérico (coma → punto decimal)
    df_oro_cny["Precio_Oro_CNY"] = df_oro_cny["Precio_Oro_CNY"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_oro_cny["Precio_Oro_CNY"] = pd.to_numeric(df_oro_cny["Precio_Oro_CNY"], errors="coerce")
    df_oro_cny = df_oro_cny[(df_oro_cny["Precio_Oro_CNY"] > 0) & (~df_oro_cny["Precio_Oro_CNY"].isna())]

    # Establecer fecha como índice
    df_oro_cny.set_index("Fecha", inplace=True)

    # === Petróleo Crudo WTI ===
    df_petroleo_raw = pd.read_csv("datasets/Datos históricos Futuros petróleo crudo WTI.csv", sep=",", encoding="utf-8")

    # Limpiar nombres de columnas
    df_petroleo_raw.columns = df_petroleo_raw.columns.str.strip().str.replace('"', '').str.replace("﻿", "")

    # Convertir la columna de fecha
    df_petroleo_raw["Fecha"] = pd.to_datetime(df_petroleo_raw["Fecha"], dayfirst=True)
    df_petroleo_raw = df_petroleo_raw.sort_values("Fecha")

    # Quedarse solo con columnas necesarias
    df_petroleo = df_petroleo_raw[["Fecha", "Último"]].rename(columns={"Último": "Precio_Petroleo"})

    # 🔁 Convertir precios al formato numérico correcto (coma decimal → punto decimal)
    df_petroleo["Precio_Petroleo"] = df_petroleo["Precio_Petroleo"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_petroleo["Precio_Petroleo"] = pd.to_numeric(df_petroleo["Precio_Petroleo"], errors="coerce")

    # Filtrar solo precios válidos
    df_petroleo = df_petroleo[(df_petroleo["Precio_Petroleo"] > 0) & (~df_petroleo["Precio_Petroleo"].isna())]

    # Indexar por fecha
    df_petroleo.set_index("Fecha", inplace=True)
    
        # === Oro en Dólar Canadiense (XAU/CAD) ===
    df_cad_raw = pd.read_csv("datasets/Datos históricos XAU_CAD.csv", sep=",", encoding="utf-8")

    # Limpiar nombres de columnas
    df_cad_raw.columns = df_cad_raw.columns.str.strip().str.replace('"', '').str.replace("﻿", "")

    # Convertir la columna de fecha
    df_cad_raw["Fecha"] = pd.to_datetime(df_cad_raw["Fecha"], dayfirst=True)
    df_cad_raw = df_cad_raw.sort_values("Fecha")

    # Quedarse solo con columnas necesarias
    df_cad = df_cad_raw[["Fecha", "Último"]].rename(columns={"Último": "Precio_Oro_CAD"})

    # 🔁 Convertir precios al formato numérico correcto (coma decimal → punto decimal)
    df_cad["Precio_Oro_CAD"] = df_cad["Precio_Oro_CAD"].astype(str) \
        .str.replace(".", "", regex=False) \
        .str.replace(",", ".", regex=False)

    df_cad["Precio_Oro_CAD"] = pd.to_numeric(df_cad["Precio_Oro_CAD"], errors="coerce")

    # Filtrar solo precios válidos
    df_cad = df_cad[(df_cad["Precio_Oro_CAD"] > 0) & (~df_cad["Precio_Oro_CAD"].isna())]

    # Indexar por fecha
    df_cad.set_index("Fecha", inplace=True)
    
    # === Eventos: Elecciones Presidenciales USA ===
    df_elecciones_usa = pd.read_csv("datasets/eventos_elecciones_presidenciales_usa_2004_2024.csv", sep=",")
    df_elecciones_usa["Fecha"] = pd.to_datetime(df_elecciones_usa["Fecha"], errors="coerce")
    df_elecciones_usa = df_elecciones_usa.dropna(subset=["Fecha"])
    
        # === Eventos: Recesiones Mundiales ===
    df_recesiones = pd.read_csv("datasets/eventos_recesiones_mundiales_2004_2024.csv", sep=",")
    df_recesiones["Fecha_Inicio"] = pd.to_datetime(df_recesiones["Fecha_Inicio"], errors="coerce")
    df_recesiones["Fecha_Fin"] = pd.to_datetime(df_recesiones["Fecha_Fin"], errors="coerce")
    df_recesiones = df_recesiones.dropna(subset=["Fecha_Inicio", "Fecha_Fin"])
    
    df_crisis_europa = pd.read_csv("datasets/eventos_crisis_financieras_europa_2004_2024.csv",  sep=",")
    df_crisis_europa["Fecha de Inicio"] = pd.to_datetime(df_crisis_europa["Fecha de Inicio"], errors="coerce")
    df_crisis_europa["Fecha de Fin"] = pd.to_datetime(df_crisis_europa["Fecha de Fin"], errors="coerce")
    df_crisis_europa = df_crisis_europa.dropna(subset=["Fecha de Inicio", "Fecha de Fin"])
    
    df_crisis_inmo = pd.read_csv("datasets/eventos_crisis_inmobiliarias_2004_2024.csv")
    df_crisis_inmo["Fecha_Inicio"] = pd.to_datetime(df_crisis_inmo["Fecha_Inicio"], errors="coerce")
    df_crisis_inmo["Fecha_Fin"] = pd.to_datetime(df_crisis_inmo["Fecha_Fin"], errors="coerce")
    df_crisis_inmo["Fecha_Fin"] = df_crisis_inmo["Fecha_Fin"].fillna(pd.Timestamp.today())
    
    df_crisis_minerales = pd.read_csv("datasets/eventos_choques_oferta_minerales_2004_2024.csv")
    df_crisis_minerales["Fecha Inicio"] = pd.to_datetime(df_crisis_minerales["Fecha Inicio"], errors="coerce")
    df_crisis_minerales["Fecha Fin"] = pd.to_datetime(df_crisis_minerales["Fecha Fin"], errors="coerce")
    df_crisis_minerales["Fecha Fin"] = df_crisis_minerales["Fecha Fin"].fillna(pd.Timestamp.today())
    
    # Cargar IPC (formato mensual por año)
    cpi_df = pd.read_csv("datasets/cpi_mensual_2004_2024.csv", parse_dates=["Fecha"])
    cpi_df.set_index("Fecha", inplace=True)

    # Renombrar columna si no se llama 'IPC'
    cpi_df.columns = ["IPC"]
    
        # Cargar índice de confianza del consumidor (CCI)
    cci_df = pd.read_csv("datasets/CCI.csv", sep=",", parse_dates=["Fecha"])
    cci_df.set_index("Fecha", inplace=True)
    cci_df = cci_df.sort_index()
    
    # Fear & Greed Index
    fear_greed_df = pd.read_csv("datasets/fear-greed-2011-2023.csv", sep=",", parse_dates=["Date"])
    fear_greed_df.set_index("Date", inplace=True)
    fear_greed_df.sort_index(inplace=True)
    fear_greed_df.rename(columns={"Fear Greed": "Fear_Greed"}, inplace=True)
    
    
    df_walcl = pd.read_csv("datasets/WALCL.csv", sep=",")
    df_walcl["observation_date"] = pd.to_datetime(df_walcl["observation_date"])
    df_walcl.set_index("observation_date", inplace=True)
    df_walcl.rename(columns={"WALCL": "FED_Balance"}, inplace=True)
    df_walcl = df_walcl.resample("W").mean()  # Datos semanales
    
        # PIB de EE.UU.
    pib_df = pd.read_csv("datasets/PIB_USA_2004_2024.csv")
    pib_df["Fecha"] = pd.to_datetime(pib_df["Fecha"])
    pib_df.set_index("Fecha", inplace=True)
    pib_df.rename(columns={"PIB_Millones_Euros": "PIB", "Crecimiento_Anual": "Crecimiento_PIB"}, inplace=True)













    return df, eventos_df, dolar_df, df_oro_ext, df_btc, df_plata, df_oro_eur, df_oro_cny, df_petroleo, df_cad, df_elecciones_usa, df_recesiones, df_crisis_europa, df_crisis_inmo, df_crisis_minerales, cpi_df, df_oro_mensual, cci_df, fear_greed_df, df_walcl, pib_df
