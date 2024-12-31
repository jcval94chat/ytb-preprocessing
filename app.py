import streamlit as st
import pickle
import pandas as pd
import json
from nltk.corpus import stopwords
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.interpret_clusters import generate_features_extended
import nltk
nltk.download('stopwords')

import logging
import joblib
import os

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de severidad
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato del mensaje
    handlers=[
        logging.FileHandler("model_verification.log"),  # Archivo para logs
        logging.StreamHandler()                        # También a consola
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------
# CARGA DE MODELOS
# -----------------------------------
model_shorts_en = joblib.load("models/model_shorts_en.joblib")
model_shorts_es = joblib.load("models/model_shorts_es.joblib")
model_videos_en = joblib.load("models/model_videos_en.joblib")
model_videos_es = joblib.load("models/model_videos_es.joblib")


# -----------------------------------
# CARGA DE OBJETOS ADICIONALES
# -----------------------------------
with open("otros_objetos/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

with open("otros_objetos/feature_columns_SH.json", "r") as f:
    feature_columns_SH = json.load(f)

with open("otros_objetos/palabras_top.json", "r") as f:
    palabras_top = json.load(f)

with open("otros_objetos/palabras_top_SH.json", "r") as f:
    palabras_top_SH = json.load(f)


def eval_frase(frase, modelo, tipo="Shorts", idioma="Spanish"):
    """
    Evalúa una frase con un modelo específico. 
    'tipo' puede ser "Shorts" o "Videos".
    'idioma' puede ser "Spanish" o "English".
    """
    # Construir un DataFrame temporal
    df_probar = pd.DataFrame({'titulo': [frase], 'language': [idioma]})

    # Generar features
    if tipo == "Shorts":
        features_2, _ = generate_features_extended(
            df_probar,
            title_column='titulo',
            most_common_words=[(x, 0) for x in palabras_top_SH.values()]
        )
        columnas = feature_columns_SH
    else:  # "Videos"
        features_2, _ = generate_features_extended(
            df_probar,
            title_column='titulo',
            most_common_words=[(x, 0) for x in palabras_top.values()]
        )
        columnas = feature_columns

    # Hacemos la predicción (asumiendo que el modelo es un clasificador)
    proba = modelo.predict_proba(features_2[columnas])
    score = proba[0][1]  # prob. de la clase "positiva"
    return float(score)


def main():
    st.title("App de Evaluación de Títulos")

    # Creamos las pestañas
    tab1, tab2 = st.tabs(["Evaluación Interactiva", "Evaluar por Archivo"])

    # ------------------------------
    # PESTAÑA 1: EVALUACIÓN INTERACTIVA
    # ------------------------------
    with tab1:
        st.header("Ventana 1: Evaluación Interactiva")

        # Entrada de texto
        frase = st.text_input("Ingresa el título", value="Los 5 Secretos del Liderazgo Efectivo")

        st.write("**Selecciona los modelos a evaluar:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            usar_shorts_es = st.checkbox("Shorts ES", value=True)
        with col2:
            usar_shorts_en = st.checkbox("Shorts EN", value=True)
        with col3:
            usar_videos_es = st.checkbox("Videos ES", value=True)
        with col4:
            usar_videos_en = st.checkbox("Videos EN", value=True)

        if st.button("Evaluar Frase"):
            # Variables para almacenar los scores (None si no se usa)
            score_sh_es = None
            score_sh_en = None
            score_v_es = None
            score_v_en = None

            # Calcular cada modelo activo
            if usar_shorts_es:
                score_sh_es = eval_frase(frase, model_shorts_es, tipo="Shorts", idioma="Spanish")
            if usar_shorts_en:
                score_sh_en = eval_frase(frase, model_shorts_en, tipo="Shorts", idioma="English")
            if usar_videos_es:
                score_v_es = eval_frase(frase, model_videos_es, tipo="Videos", idioma="Spanish")
            if usar_videos_en:
                score_v_en = eval_frase(frase, model_videos_en, tipo="Videos", idioma="English")

            # Construimos un DF con una fila y columnas de cada modelo
            data_result = {
                "Shorts ES": [score_sh_es],
                "Shorts EN": [score_sh_en],
                "Videos ES": [score_v_es],
                "Videos EN": [score_v_en]
            }
            df_result = pd.DataFrame(data_result)

            st.write("### Resultados en tabla")
            st.dataframe(df_result, use_container_width=True)

    # ------------------------------
    # PESTAÑA 2: EVALUACIÓN POR ARCHIVO
    # ------------------------------
    with tab2:
        st.header("Ventana 2: Evaluar Archivo con Frases")
        st.write("Sube un archivo CSV o TXT con **una frase por renglón**. Se calculará la suma de scores de los modelos seleccionados y luego se mostrarán los resultados.")

        # Checkboxes (horizontal) para seleccionar modelos - por defecto activados
        st.write("**Selecciona los modelos a evaluar:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            usar_shorts_es_2 = st.checkbox("Shorts ES (2)", value=True)
        with c2:
            usar_shorts_en_2 = st.checkbox("Shorts EN (2)", value=True)
        with c3:
            usar_videos_es_2 = st.checkbox("Videos ES (2)", value=True)
        with c4:
            usar_videos_en_2 = st.checkbox("Videos EN (2)", value=True)

        # Subir archivo (CSV/TXT)
        archivo_subido = st.file_uploader("Sube tu archivo CSV o TXT", type=["csv", "txt"])

        # Slider para definir cuántos resultados mostrar (1 a 100)
        top_n = st.slider("¿Cuántos resultados mostrar?", min_value=1, max_value=100, value=10)

        if archivo_subido is not None:
            # Intentamos leer como CSV
            try:
                df_frases = pd.read_csv(archivo_subido, header=None, names=["frase"])
            except:
                # Si falla, leemos como TXT
                archivo_subido.seek(0)
                lineas = archivo_subido.read().decode("utf-8").splitlines()
                df_frases = pd.DataFrame({"frase": lineas})

            st.write(f"**Se han cargado {len(df_frases)} frases.**")

            if st.button("Evaluar Archivo"):
                resultados = []

                for _, row in df_frases.iterrows():
                    frase_texto = row["frase"]

                    # Evaluamos con cada modelo seleccionado y sumamos
                    score_sh_es = eval_frase(frase_texto, model_shorts_es, "Shorts", "Spanish") if usar_shorts_es_2 else 0
                    score_sh_en = eval_frase(frase_texto, model_shorts_en, "Shorts", "English") if usar_shorts_en_2 else 0
                    score_v_es  = eval_frase(frase_texto, model_videos_es, "Videos", "Spanish") if usar_videos_es_2 else 0
                    score_v_en  = eval_frase(frase_texto, model_videos_en, "Videos", "English") if usar_videos_en_2 else 0

                    score_total = score_sh_es + score_sh_en + score_v_es + score_v_en

                    resultados.append({
                        "frase": frase_texto,
                        "Shorts ES": score_sh_es if usar_shorts_es_2 else None,
                        "Shorts EN": score_sh_en if usar_shorts_en_2 else None,
                        "Videos ES": score_v_es if usar_videos_es_2 else None,
                        "Videos EN": score_v_en if usar_videos_en_2 else None,
                        "Score Total": score_total
                    })

                # Ordenamos por Score Total descendente
                resultados_ordenados = sorted(resultados, key=lambda x: x["Score Total"], reverse=True)

                # Tomamos top_n
                top_n_resultados = resultados_ordenados[:top_n]

                # Convertimos a DataFrame
                df_top = pd.DataFrame(top_n_resultados)

                st.write(f"### Mejores {top_n} resultados")
                # Mostramos el DataFrame (puede reordenar columnas con st.data_editor en versiones recientes)
                st.data_editor(
                    df_top,
                    use_container_width=True
                )

                # (Opcional) Expandir detalles si lo requieres
                # with st.expander("Ver detalles"):
                #     st.write(df_top)


if __name__ == "__main__":
    main()
