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



    return df, eventos_df, dolar_df, df_oro_ext, df_btc
