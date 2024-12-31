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

# Función para verificar el tipo de modelo (opcional, para tu registro en logs)
def verificar_tipo_modelo(model, nombre_modelo):
    if isinstance(model, RandomForestClassifier):
        tipo = "RandomForestClassifier"
    elif isinstance(model, RandomForestRegressor):
        tipo = "RandomForestRegressor"
    else:
        tipo = type(model).__name__
    print(f"El modelo '{nombre_modelo}' es de tipo: {tipo}")

# Verificar cada modelo (opcional)
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
    # Construir un DF temporal
    df_probar = pd.DataFrame({'titulo': [frase], 'language': [idioma]})

    # Generar features
    if tipo == "Shorts":
        # Usamos las palabras_top de Shorts (si así lo defines)
        features_2, _ = generate_features_extended(
            df_probar,
            title_column='titulo',
            most_common_words=[(x, 0) for x in palabras_top_SH.values()]
        )
        # Seleccionamos las columnas de Shorts
        columnas = feature_columns_SH
    else:  # "Videos"
        features_2, _ = generate_features_extended(
            df_probar,
            title_column='titulo',
            most_common_words=[(x, 0) for x in palabras_top.values()]
        )
        columnas = feature_columns

    # Hacemos la predicción
    proba = modelo.predict_proba(features_2[columnas])
    score = proba[0][1]  # asumimos [0][1] es la prob. de la clase "positiva"
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
        st.write("Elige los modelos y evalúa de forma individual.")

        # Entrada de texto
        frase = st.text_input("Ingresa el título", value="Los 5 Secretos del Liderazgo Efectivo")

        # Checkboxes (o botones) para los 4 modelos
        usar_shorts_es = st.checkbox("Modelo Shorts - Español", value=False)
        usar_shorts_en = st.checkbox("Modelo Shorts - Inglés", value=False)
        usar_videos_es = st.checkbox("Modelo Videos - Español", value=False)
        usar_videos_en = st.checkbox("Modelo Videos - Inglés", value=False)

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

    # ------------------------------
    # PESTAÑA 2: EVALUACIÓN POR ARCHIVO
    # ------------------------------
    with tab2:
        st.header("Ventana 2: Evaluar Archivo con Frases")

        # Instrucciones
        st.write("Sube un archivo CSV o TXT con **una frase por renglón**. Se calculará la suma de scores de los modelos seleccionados. Luego se mostrarán las 10 mejores frases.")

        # Checkboxes para seleccionar modelos
        usar_shorts_es_2 = st.checkbox("Modelo Shorts - Español (2)", value=False)
        usar_shorts_en_2 = st.checkbox("Modelo Shorts - Inglés (2)", value=False)
        usar_videos_es_2 = st.checkbox("Modelo Videos - Español (2)", value=False)
        usar_videos_en_2 = st.checkbox("Modelo Videos - Inglés (2)", value=False)

        # Selectbox para idioma (para simplificar, asumimos que todas las frases son del mismo idioma)
        idioma_archivo = st.selectbox("Idioma de las frases en el archivo:", ["Spanish", "English"])

        # Cargamos el archivo
        archivo_subido = st.file_uploader("Sube tu archivo CSV o TXT", type=["csv", "txt"])

        if archivo_subido is not None:
            # Leemos el contenido
            try:
                # Intentamos leerlo como CSV
                df_frases = pd.read_csv(archivo_subido, header=None, names=["frase"])
            except:
                # Si falla, tal vez es un TXT
                archivo_subido.seek(0)  # reiniciamos el cursor
                lineas = archivo_subido.read().decode("utf-8").splitlines()
                df_frases = pd.DataFrame({"frase": lineas})

            # Mostramos cuántas frases se han cargado
            st.write(f"Se han cargado {len(df_frases)} frases.")

            # Botón para procesar
            if st.button("Evaluar Frases"):
                resultados = []
                
                for i, row in df_frases.iterrows():
                    frase_texto = row["frase"]
                    scores_individuales = []

                    # Sumamos todos los modelos activos
                    total_score = 0.0
                    
                    if usar_shorts_es_2:
                        s = eval_frase(frase_texto, model_shorts_es, "Shorts", "Spanish")
                        total_score += s
                        scores_individuales.append(("Shorts ES", s))

                    if usar_shorts_en_2:
                        s = eval_frase(frase_texto, model_shorts_en, "Shorts", "English")
                        total_score += s
                        scores_individuales.append(("Shorts EN", s))

                    if usar_videos_es_2:
                        s = eval_frase(frase_texto, model_videos_es, "Videos", "Spanish")
                        total_score += s
                        scores_individuales.append(("Videos ES", s))

                    if usar_videos_en_2:
                        s = eval_frase(frase_texto, model_videos_en, "Videos", "English")
                        total_score += s
                        scores_individuales.append(("Videos EN", s))

                    resultados.append({
                        "frase": frase_texto,
                        "score_total": total_score,
                        "detalles": scores_individuales
                    })

                # Ordenamos por score_total (descendente)
                resultados_ordenados = sorted(resultados, key=lambda x: x["score_total"], reverse=True)

                # Tomamos top 10
                top_10 = resultados_ordenados[:10]

                st.write("## Top 10 Frases")
                st.write("A continuación se muestran las 10 frases con mayor puntuación acumulada.")
                
                # Mostramos en una tabla
                # Podrías mostrarlo con st.dataframe, st.table o formatearlo
                df_top_10 = pd.DataFrame([{
                    "Frase": r["frase"],
                    "Score Total": r["score_total"]
                } for r in top_10])

                st.dataframe(df_top_10)

                # Si deseas, también puedes mostrar los scores individuales de cada modelo
                # en la misma tabla o en un expander. Ejemplo:
                with st.expander("Ver detalles de cada frase (scores por modelo)"):
                    for elem in top_10:
                        st.write(f"**Frase:** {elem['frase']}")
                        st.write(f"**Score total:** {elem['score_total']:.4f}")
                        for (nombre_modelo, s) in elem["detalles"]:
                            st.write(f"  - {nombre_modelo}: {s:.4f}")
                        st.write("---")


if __name__ == "__main__":
    main()
