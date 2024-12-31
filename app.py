# app.py
import streamlit as st
import pickle
import pandas as pd
import json
from nltk.corpus import stopwords
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Importar tu lógica de features si la tienes en utils/
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
# CARGA DE MODELOS (.pkl)
# -----------------------------------
# model_shorts_en = pickle.load(open("models/model_shorts_en.pkl", "rb"))
# model_shorts_es = pickle.load(open("models/model_shorts_es.pkl", "rb"))
# model_videos_en = pickle.load(open("models/model_videos_en.pkl", "rb"))
# model_videos_es = pickle.load(open("models/model_videos_es.pkl", "rb"))

model_shorts_en = joblib.load("models/model_shorts_en.joblib")
model_shorts_es = joblib.load("models/model_shorts_es.joblib")
model_videos_en = joblib.load("models/model_videos_en.joblib")
model_videos_es = joblib.load("models/model_videos_es.joblib")


# Función para verificar el tipo de modelo
def verificar_tipo_modelo(model, nombre_modelo):
    if isinstance(model, RandomForestClassifier):
        tipo = "RandomForestClassifier"
    elif isinstance(model, RandomForestRegressor):
        tipo = "RandomForestRegressor"
    else:
        tipo = type(model).__name__
    print(f"El modelo '{nombre_modelo}' es de tipo: {tipo}")

# Verificar cada modelo
verificar_tipo_modelo(model_shorts_en, "model_shorts_en")
verificar_tipo_modelo(model_shorts_es, "model_shorts_es")
verificar_tipo_modelo(model_videos_en, "model_videos_en")
verificar_tipo_modelo(model_videos_es, "model_videos_es")

# -----------------------------------
# CARGA DE OBJETOS ADICIONALES (JSON)
# -----------------------------------
with open("otros_objetos/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

with open("otros_objetos/feature_columns_SH.json", "r") as f:
    feature_columns_SH = json.load(f)

with open("otros_objetos/palabras_top.json", "r") as f:
    palabras_top = json.load(f)

with open("otros_objetos/palabras_top_SH.json", "r") as f:
    palabras_top_SH = json.load(f)


def eval_frase(frase, idioma="Spanish", tipo="Shorts"):
    """
    Evalúa la frase usando uno de los cuatro modelos, según el idioma y tipo.
    """
    # Preparar un DataFrame
    df_probar = pd.DataFrame({'titulo': frase, 'language': idioma}, index=[0])

    # Generar features
    features_2, _ = generate_features_extended(
        df_probar,
        title_column='titulo',
        most_common_words=[(x, 0) for x in palabras_top.values()]  # ejemplo
    )

    # Seleccionar modelo
    if tipo == "Shorts":
        if idioma == "Spanish":
            chosen_model = model_shorts_es
        else:
            chosen_model = model_shorts_en
    else:  # tipo = "Videos"
        if idioma == "Spanish":
            chosen_model = model_videos_es
        else:
            chosen_model = model_videos_en
    
    # Hacer la predicción
    proba = chosen_model.predict_proba(features_2[feature_columns])
    score = proba[0][1]  # asumimos [0][1] es prob. de la clase "positiva"
    return float(score)

# -----------------------------------
# INTERFAZ STREAMLIT
# -----------------------------------
def main():
    st.title("Evaluar Título con RandomForest")
    st.write("Ejemplo de uso de modelos Shorts/Videos en Español/Inglés.")

    # Inputs de usuario
    frase = st.text_input("Ingresa el título", value="Los 5 Secretos del Liderazgo Efectivo")
    idioma = st.selectbox("Idioma:", ["Spanish", "English"])
    tipo = st.selectbox("Tipo de contenido:", ["Shorts", "Videos"])

    if st.button("Evaluar"):
        score = eval_frase(frase, idioma=idioma, tipo=tipo)
        st.success(f"Probabilidad de éxito: {score:.4f}")

if __name__ == "__main__":
    main()
