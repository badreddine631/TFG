import streamlit as st
from utils.pdf_generator import generar_informe

def show(df, eventos_df):
    """Muestra la sección de generación de informes."""
    st.title("📂 Informes Personalizables")

    if st.button("Generar Informe PDF"):
        output_path = generar_informe(eventos_df)
        with open(output_path, "rb") as file:
            st.download_button("Descargar Informe PDF", file, file_name="informe_analisis.pdf", mime="application/pdf")

    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
