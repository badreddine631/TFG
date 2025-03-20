import pandas as pd

def load_data():
    """Carga y procesa los datos necesarios para la aplicación."""
    df = pd.read_csv("C:/Users/Usuario/Desktop/TFFFG/price-gold/XAU_1d_data_2004_to_2024-09-20.csv", parse_dates=["Date"])
    eventos_df = pd.read_csv("C:/Users/Usuario/Desktop/TFFFG/price-gold/eventos_macroeconomicos.csv", parse_dates=["Fecha"])

    df = df.rename(columns={"Date": "Fecha"})
    df.set_index("Fecha", inplace=True)

    return df, eventos_df
