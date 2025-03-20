import pandas as pd

def load_data():
    """Carga y procesa los datos necesarios para la aplicación."""
    
    # Cargar datos del oro
    df = pd.read_csv("datasets/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
    eventos_df = pd.read_csv("datasets/eventos_macroeconomicos.csv", parse_dates=["Fecha"])
    
    # Cargar datos del índice DXY
    dolar_df = pd.read_csv("datasets/Datos_historicos_Indice_dolar.csv", parse_dates=["Fecha"], dayfirst=True)

    # Convertir columnas a valores numéricos correctamente
    df = df.rename(columns={"Date": "Fecha", "Close": "Precio_Oro"})
    df.set_index("Fecha", inplace=True)

    # Procesar índice DXY
    dolar_df = dolar_df.rename(columns={"Último": "DXY"})
    dolar_df["DXY"] = pd.to_numeric(dolar_df["DXY"].astype(str).str.replace(",", "."), errors='coerce')
    dolar_df.dropna(subset=["DXY"], inplace=True)  # Eliminar filas que no pudieron convertirse
    dolar_df.set_index("Fecha", inplace=True)

    return df, eventos_df, dolar_df
