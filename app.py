# app.py
import streamlit as st
import pickle
import pandas as pd
import json
from nltk.corpus import stopwords
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Importar tu lógica de features
from utils.interpret_clusters import generate_features_extended
import nltk
nltk.download('stopwords')

import logging
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de severidad
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato del mensaje
    handlers=[
        logging.FileHandler("model_verification.log"),  # Archivo donde se guardarán los logs
        logging.StreamHandler()  # También se mostrarán en la consola
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

def verificar_tipo_modelo(model, nombre_modelo):
    """Función de verificación (opcional)"""
    if isinstance(model, RandomForestClassifier):
        tipo = "RandomForestClassifier"
    elif isinstance(model, RandomForestRegressor):
        tipo = "RandomForestRegressor"
    else:
        tipo = type(model).__name__
    print(f"El modelo '{nombre_modelo}' es de tipo: {tipo}")

# (Opcional) Verificar tipo de cada modelo
verificar_tipo_modelo(model_shorts_en, "model_shorts_en")
verificar_tipo_modelo(model_shorts_es, "model_shorts_es")
verificar_tipo_modelo(model_videos_en, "model_videos_en")
verificar_tipo_modelo(model_videos_es, "model_videos_es")

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
    df_probar = pd.DataFrame({'titulo': [frase], 'language': [idioma]})

    # Según el tipo, usamos feature_columns diferentes
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

    proba = modelo.predict_proba(features_2[columnas])
    score = proba[0][1]  # prob. de la clase "positiva"
    return float(score)

def main():
    st.title("App de Evaluación de Títulos")

    # Creamos las pestañas
    tab1, tab2 = st.tabs(["Evaluación Interactiva", "Evaluar por Archivo"])

    # ------------------------------------------------------------
    # PESTAÑA 1: EVALUACIÓN INTERACTIVA
    # ------------------------------------------------------------
    with tab1:
        st.header("Ventana 1: Evaluación Interactiva")
        st.write("Elige uno o varios modelos y evalúa de forma individual.")

        # Entrada de texto
        frase = st.text_input("Ingresa el título", value="Los 5 Secretos del Liderazgo Efectivo")

        # Mostramos 4 checkboxes en horizontal, uno para cada modelo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            usar_shorts_es = st.checkbox("Shorts ES", value=False)
        with col2:
            usar_shorts_en = st.checkbox("Shorts EN", value=False)
        with col3:
            usar_videos_es = st.checkbox("Videos ES", value=False)
        with col4:
            usar_videos_en = st.checkbox("Videos EN", value=False)

        if st.button("Evaluar Frase"):
            # Para cada modelo activo, calculamos score y lo mostramos
            if usar_shorts_es:
                score_sh_es = eval_frase(frase, model_shorts_es, tipo="Shorts", idioma="Spanish")
                st.success(f"[Shorts ES] Score: {score_sh_es:.4f}")

            if usar_shorts_en:
                score_sh_en = eval_frase(frase, model_shorts_en, tipo="Shorts", idioma="English")
                st.success(f"[Shorts EN] Score: {score_sh_en:.4f}")

            if usar_videos_es:
                score_v_es = eval_frase(frase, model_videos_es, tipo="Videos", idioma="Spanish")
                st.success(f"[Videos ES] Score: {score_v_es:.4f}")

            if usar_videos_en:
                score_v_en = eval_frase(frase, model_videos_en, tipo="Videos", idioma="English")
                st.success(f"[Videos EN] Score: {score_v_en:.4f}")

    # ------------------------------------------------------------
    # PESTAÑA 2: EVALUAR ARCHIVO DE FRASES
    # ------------------------------------------------------------
    with tab2:
        st.header("Ventana 2: Evaluar Archivo con Frases")
        st.write("Sube un archivo CSV o TXT con una frase por renglón. "
                 "Selecciona los modelos a utilizar y obtén un ranking de hasta 100 frases.")

        # Checkboxes en horizontal para los cuatro modelos
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            usar_shorts_es_2 = st.checkbox("Shorts ES (2)", value=False)
        with col2:
            usar_shorts_en_2 = st.checkbox("Shorts EN (2)", value=False)
        with col3:
            usar_videos_es_2 = st.checkbox("Videos ES (2)", value=False)
        with col4:
            usar_videos_en_2 = st.checkbox("Videos EN (2)", value=False)

        # Subida de archivo
        archivo_subido = st.file_uploader("Sube tu archivo CSV o TXT", type=["csv", "txt"])

        # Slider para definir cuántos resultados (Top N) mostrar
        top_n = st.slider("Elige cuántos resultados mostrar", min_value=1, max_value=100, value=10)

        if archivo_subido is not None:
            # Intentamos leer como CSV; si falla, leemos como TXT
            try:
                df_frases = pd.read_csv(archivo_subido, header=None, names=["frase"])
            except:
                archivo_subido.seek(0)  # reiniciamos cursor para leer de nuevo
                lineas = archivo_subido.read().decode("utf-8").splitlines()
                df_frases = pd.DataFrame({"frase": lineas})

            st.write(f"Se han cargado {len(df_frases)} frases.")

            if st.button("Evaluar Frases"):
                resultados = []

                for _, row in df_frases.iterrows():
                    frase_texto = row["frase"]
                    # Evaluamos con cada modelo (si está seleccionado). Si no, ponemos 0.
                    s_sh_es = eval_frase(frase_texto, model_shorts_es, "Shorts", "Spanish") \
                              if usar_shorts_es_2 else 0
                    s_sh_en = eval_frase(frase_texto, model_shorts_en, "Shorts", "English") \
                              if usar_shorts_en_2 else 0
                    s_vid_es = eval_frase(frase_texto, model_videos_es, "Videos", "Spanish") \
                               if usar_videos_es_2 else 0
                    s_vid_en = eval_frase(frase_texto, model_videos_en, "Videos", "English") \
                               if usar_videos_en_2 else 0

                    total_score = s_sh_es + s_sh_en + s_vid_es + s_vid_en

                    resultados.append({
                        "frase": frase_texto,
                        "Shorts ES": s_sh_es,
                        "Shorts EN": s_sh_en,
                        "Videos ES": s_vid_es,
                        "Videos EN": s_vid_en,
                        "Score Total": total_score
                    })

                # Ordenamos desc por Score Total
                resultados_ordenados = sorted(resultados, key=lambda x: x["Score Total"], reverse=True)
                # Tomamos los N que definimos en el slider
                top_n_resultados = resultados_ordenados[:top_n]

                st.write(f"## Top {top_n} frases")

                # Convertimos a DataFrame para mostrarlo
                df_top = pd.DataFrame(top_n_resultados)

                # Mostramos en un 'data_editor' para que el usuario pueda reordenar columnas
                st.data_editor(
                    df_top,
                    disable_column_reordering=False,
                    use_container_width=True
                )

                # Si quieres mostrar detalles de cada fila, podrías usar un expander
                # (aunque aquí cada columna ya está a la vista).
                # with st.expander("Ver detalles..."):
                #     st.write(df_top)


if __name__ == "__main__":
    main()
