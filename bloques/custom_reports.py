import streamlit as st

from utils.pdf_generator import (
    AnalysisInputs,
    generar_informe,
    get_available_analyses,
)


def show(
    df,
    eventos_df,
    dolar_df,
    df_oro_ext,
    df_btc,
    df_plata,
    df_oro_eur,
    df_oro_cny,
    df_petroleo,
    df_cad,
    df_elecciones_usa,
    df_recesiones,
    df_crisis_europa,
    df_crisis_inmo,
    df_crisis_minerales,
    cpi_df,
    df_oro_mensual,
    cci_df,
    fear_greed_df,
    df_walcl,
    pib_df,
):
    """Muestra la sección de generación de informes personalizados."""

    st.title("📂 Informes Personalizables")
    st.markdown(
        "Selecciona los análisis que deseas incluir en el informe PDF. "
        "Los resultados se basan en los mismos módulos disponibles en Información Basada en Datos."
    )

    catalogo = get_available_analyses()

    with st.form("form_informe_personalizado"):
        seleccionados = []
        for seccion, items in catalogo.items():
            with st.expander(seccion, expanded=False):
                for item in items:
                    marcado = st.checkbox(item["name"], key=f"{seccion}_{item['id']}")
                    st.caption(item["description"])
                    if marcado:
                        seleccionados.append(item["id"])
        generar = st.form_submit_button("Generar Informe PDF")

    if generar:
        if not seleccionados:
            st.warning("Debes seleccionar al menos un análisis para generar el informe.")
        else:
            inputs = AnalysisInputs(
                df=df,
                eventos_df=eventos_df,
                dolar_df=dolar_df,
                df_oro_ext=df_oro_ext,
                df_btc=df_btc,
                df_plata=df_plata,
                df_oro_eur=df_oro_eur,
                df_oro_cny=df_oro_cny,
                df_petroleo=df_petroleo,
                df_cad=df_cad,
                df_elecciones_usa=df_elecciones_usa,
                df_recesiones=df_recesiones,
                df_crisis_europa=df_crisis_europa,
                df_crisis_inmo=df_crisis_inmo,
                df_crisis_minerales=df_crisis_minerales,
                cpi_df=cpi_df,
                df_oro_mensual=df_oro_mensual,
                cci_df=cci_df,
                fear_greed_df=fear_greed_df,
                df_walcl=df_walcl,
                pib_df=pib_df,
            )

            output_path = generar_informe(inputs, seleccionados)
            with open(output_path, "rb") as file:
                pdf_bytes = file.read()

            st.success("Informe generado correctamente.")
            st.download_button(
                "Descargar Informe PDF",
                data=pdf_bytes,
                file_name="informe_analisis.pdf",
                mime="application/pdf",
            )

    if st.button("⬅ Volver al Inicio"):
        st.session_state.current_page = "home"
        st.rerun()
