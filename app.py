import streamlit as st
import pandas as pd
import json
import joblib
import logging
import nltk
nltk.download('stopwords')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.interpret_clusters import generate_features_extended

import os
from io import StringIO

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CARGAR MODELOS CON CACHÉ PARA MAYOR RENDIMIENTO
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    model_shorts_en = joblib.load("models/model_shorts_en.joblib")
    model_shorts_es = joblib.load("models/model_shorts_es.joblib")
    model_videos_en = joblib.load("models/model_videos_en.joblib")
    model_videos_es = joblib.load("models/model_videos_es.joblib")
    return model_shorts_en, model_shorts_es, model_videos_en, model_videos_es

model_shorts_en, model_shorts_es, model_videos_en, model_videos_es = load_models()

# -----------------------------------------------------------------------------
# CARGAR OBJETOS ADICIONALES (FEATURES, PALABRAS TOP, ETC.)
# -----------------------------------------------------------------------------
with open("otros_objetos/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

with open("otros_objetos/feature_columns_SH.json", "r") as f:
    feature_columns_SH = json.load(f)

with open("otros_objetos/palabras_top.json", "r") as f:
    palabras_top = json.load(f)

with open("otros_objetos/palabras_top_SH.json", "r") as f:
    palabras_top_SH = json.load(f)

# -----------------------------------------------------------------------------
# FUNCIÓN PARA EVALUAR UNA FRASE CON UN MODELO DADO
# -----------------------------------------------------------------------------
def eval_frase(frase, modelo, tipo="Shorts", idioma="Spanish"):
    """
    Evalúa una frase con el modelo específico.
    Ajusta las columnas para evitar ValueError por mismatch.
    """
    # 1. Generar DataFrame con la frase
    df_probar = pd.DataFrame({'titulo': [frase], 'language': [idioma]})

    # 2. Generar features
    if tipo == "Shorts":
        features_temp, _ = generate_features_extended(
            df_probar,
            title_column='titulo',
            most_common_words=[(x, 0) for x in palabras_top_SH.values()]
        )
        columnas_modelo = feature_columns_SH
    else:  # "Videos"
        features_temp, _ = generate_features_extended(
            df_probar,
            title_column='titulo',
            most_common_words=[(x, 0) for x in palabras_top.values()]
        )
        columnas_modelo = feature_columns

    # Asegurarnos de que las columnas coincidan exactamente con las que el modelo espera
    # 1) Creamos cualquier columna faltante con valor 0
    for col in columnas_modelo:
        if col not in features_temp.columns:
            features_temp[col] = 0
    # 2) Descartamos columnas extra que el modelo no tenga
    features_temp = features_temp[[c for c in columnas_modelo if c in features_temp.columns]]

    # 3. Hacer predicción (suponiendo que es un RandomForestClassifier)
    proba = modelo.predict_proba(features_temp)
    score = proba[0][1]
    return float(score)

# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE LA APP
# -----------------------------------------------------------------------------
def main():
    # ---------------
    # Barra Lateral
    # ---------------
    st.sidebar.title("Parámetros y Descripción")
    st.sidebar.write(
        """
        ### Sobre la Aplicación
        - Evalúa títulos con distintos **modelos** (Shorts/Videos, Español/Inglés).
        - Carga masiva de frases por archivo CSV o TXT.
        - Descarga de resultados en CSV.
        - **Nota**: Los modelos usan RandomForestClassifier.
        """
    )

    st.sidebar.write("### Selección de Modelos (Activados por defecto)")

    # Checkboxes en la barra lateral
    usar_shorts_es = st.sidebar.checkbox("Shorts ES", value=True)
    usar_shorts_en = st.sidebar.checkbox("Shorts EN", value=True)
    usar_videos_es = st.sidebar.checkbox("Videos ES", value=True)
    usar_videos_en = st.sidebar.checkbox("Videos EN", value=True)

    # ---------------
    # Contenido Principal
    # ---------------
    st.title("App de Evaluación de Títulos")

    # Usamos tabs para separar la evaluación individual y la evaluación por archivo
    tab1, tab2 = st.tabs(["Evaluación Interactiva", "Evaluar Archivo"])

    # PESTAÑA 1 ----------------------------------------------------------------------------------
    with tab1:
        st.header("Ventana 1: Evaluación de una Frase")

        # Entrada de texto
        frase = st.text_input("Ingresa el título:", "Los 5 Secretos del Liderazgo Efectivo")

        if st.button("Evaluar Frase"):
            # Para evitar repeticiones, guardamos los scores en un dict
            scores_dict = {
                "Shorts ES": None,
                "Shorts EN": None,
                "Videos ES": None,
                "Videos EN": None
            }

            with st.spinner("Evaluando..."):
                # Calculamos cada modelo, si está activo
                if usar_shorts_es:
                    scores_dict["Shorts ES"] = eval_frase(frase, model_shorts_es, "Shorts", "Spanish")
                if usar_shorts_en:
                    scores_dict["Shorts EN"] = eval_frase(frase, model_shorts_en, "Shorts", "English")
                if usar_videos_es:
                    scores_dict["Videos ES"] = eval_frase(frase, model_videos_es, "Videos", "Spanish")
                if usar_videos_en:
                    scores_dict["Videos EN"] = eval_frase(frase, model_videos_en, "Videos", "English")

            # Convertimos a DataFrame
            df_result_individual = pd.DataFrame([scores_dict])
            st.subheader("Resultados de la evaluación")
            st.dataframe(df_result_individual, use_container_width=True)

    # PESTAÑA 2 ----------------------------------------------------------------------------------
    with tab2:
        st.header("Ventana 2: Evaluar por Archivo")

        # Sube un archivo CSV o TXT
        archivo_subido = st.file_uploader("Sube tu archivo CSV o TXT (1 frase por línea)", type=["csv", "txt"])

        # Slider para definir cuántos resultados mostrar (1 a 100)
        top_n = st.slider("¿Cuántos resultados quieres ver?", min_value=1, max_value=100, value=10)

        if archivo_subido is not None:
            # Intentamos leer el archivo
            try:
                df_frases = pd.read_csv(archivo_subido, header=None, names=["frase"])
            except:
                # Si falla, leemos como TXT
                archivo_subido.seek(0)
                lineas = archivo_subido.read().decode("utf-8").splitlines()
                df_frases = pd.DataFrame({"frase": lineas})

            st.write(f"**Se cargaron {len(df_frases)} frases.**")

            if st.button("Evaluar Archivo"):
                # Estructura para resultados
                resultados = []

                with st.spinner("Procesando archivo..."):
                    for frase_texto in df_frases["frase"]:
                        # Inicializamos valores
                        s_shorts_es = eval_frase(frase_texto, model_shorts_es, "Shorts", "Spanish") if usar_shorts_es else 0.0
                        s_shorts_en = eval_frase(frase_texto, model_shorts_en, "Shorts", "English") if usar_shorts_en else 0.0
                        s_videos_es = eval_frase(frase_texto, model_videos_es, "Videos", "Spanish") if usar_videos_es else 0.0
                        s_videos_en = eval_frase(frase_texto, model_videos_en, "Videos", "English") if usar_videos_en else 0.0

                        score_total = s_shorts_es + s_shorts_en + s_videos_es + s_videos_en
                        resultados.append({
                            "frase": frase_texto,
                            "Shorts ES": s_shorts_es if usar_shorts_es else None,
                            "Shorts EN": s_shorts_en if usar_shorts_en else None,
                            "Videos ES": s_videos_es if usar_videos_es else None,
                            "Videos EN": s_videos_en if usar_videos_en else None,
                            "Score Total": score_total
                        })

                # Ordenamos por 'Score Total' de mayor a menor
                resultados_ordenados = sorted(resultados, key=lambda x: x["Score Total"], reverse=True)
                # Tomamos top_n
                top_n_resultados = resultados_ordenados[:top_n]

                df_top = pd.DataFrame(top_n_resultados)
                st.subheader(f"Top {top_n} resultados")
                st.dataframe(df_top, use_container_width=True)

                # -------------------------------------------------------
                # OPCIÓN: DESCARGAR RESULTADOS COMO CSV
                # -------------------------------------------------------
                csv_buffer = StringIO()
                df_top.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Descargar CSV de Resultados",
                    data=csv_buffer.getvalue(),
                    file_name="resultados.csv",
                    mime="text/csv"
                )

# -----------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
