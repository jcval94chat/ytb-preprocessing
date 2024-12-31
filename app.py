import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import logging
import os
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# from utils.interpret_clusters import generate_features_extended
# Suponiendo que tienes la función generate_features_extended importada:
def generate_features_extended(df, title_column='titulo', most_common_words=None):
    """
    Ejemplo sencillo de la función generate_features_extended.
    Retorna (df_features, algo_mas).
    Aquí lo dejamos muy básico para ilustrar.
    """
    # Suponiendo que "most_common_words" es una lista de tuplas (word, freq)
    # Generaremos features muy simples (longitud, contadores, etc.)
    df_features = pd.DataFrame()
    df_features["longitud"] = df[title_column].str.len()
    # ... Aquí vendría tu lógica real ...
    return df_features, None

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------
# CARGA DE OBJETOS ADICIONALES
# -----------------------------------
# Ejemplo: supongamos que tienes tus listas JSON de features
feature_columns = ["longitud"]  # Simplificado
feature_columns_SH = ["longitud"]  # Simplificado
palabras_top = {}   # Simplificado
palabras_top_SH = {}  # Simplificado

# -----------------------------------
# CACHING DE MODELOS
# -----------------------------------
@st.cache_resource
def load_models():
    # Carga tus modelos de la carpeta "models"
    # Aquí ponemos un ejemplo ficticio
    model_shorts_en = joblib.load("models/model_shorts_en.joblib")
    model_shorts_es = joblib.load("models/model_shorts_es.joblib")
    model_videos_en = joblib.load("models/model_videos_en.joblib")
    model_videos_es = joblib.load("models/model_videos_es.joblib")
    return model_shorts_en, model_shorts_es, model_videos_en, model_videos_es

model_shorts_en, model_shorts_es, model_videos_en, model_videos_es = load_models()

# Función para evaluar una frase con un modelo
def eval_frase(frase, modelo, tipo="Shorts", idioma="Spanish"):
    """
    'tipo' puede ser "Shorts" o "Videos".
    'idioma' puede ser "Spanish" o "English".
    """
    df_probar = pd.DataFrame({'titulo': [frase], 'language': [idioma]})

    # Generar features (esto depende de tu lógica real)
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

    # Suponiendo que es un clasificador con predict_proba
    proba = modelo.predict_proba(features_2[columnas])
    score = proba[0][1]
    return float(score)

def main():
    # -----------------------------------
    # SIDEBAR
    # -----------------------------------
    st.sidebar.title("Parámetros y Descripción")

    # Descripción
    st.sidebar.markdown(
        """
        **Bienvenido(a)**  
        En esta aplicación puedes:
        - Evaluar una sola frase con varios modelos.  
        - Subir un archivo con múltiples frases.  
        - Descargar los resultados en CSV.  

        **Modelos disponibles**:  
        - Shorts (ES/EN)  
        - Videos (ES/EN)
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Selecciona los modelos a usar")

    # Aquí podrías usar toggles si instalas librerías externas. 
    # De momento, usamos checkboxes (activados por defecto).
    usar_shorts_es = st.sidebar.checkbox("Shorts ES", value=True)
    usar_shorts_en = st.sidebar.checkbox("Shorts EN", value=True)
    usar_videos_es = st.sidebar.checkbox("Videos ES", value=True)
    usar_videos_en = st.sidebar.checkbox("Videos EN", value=True)

    st.sidebar.markdown("---")

    # Slider para top_n de la segunda pestaña (en la barra lateral)
    top_n = st.sidebar.slider("¿Cuántos resultados mostrar?", min_value=1, max_value=100, value=10)

    # -----------------------------------
    # MAIN TABS
    # -----------------------------------
    st.title("App de Evaluación de Títulos")

    tab1, tab2 = st.tabs(["Evaluación de UNA Frase", "Evaluación por Archivo"])

    # ================================
    # TAB 1: Evaluación Interactiva
    # ================================
    with tab1:
        st.header("Evaluar una sola frase")
        frase = st.text_input("Ingresa el título", value="Los 5 Secretos del Liderazgo Efectivo")

        if st.button("Evaluar Frase"):
            with st.spinner("Evaluando la frase..."):
                # Calculamos cada modelo que esté activo
                score_sh_es = eval_frase(frase, model_shorts_es, "Shorts", "Spanish") if usar_shorts_es else None
                score_sh_en = eval_frase(frase, model_shorts_en, "Shorts", "English") if usar_shorts_en else None
                score_v_es  = eval_frase(frase, model_videos_es, "Videos", "Spanish") if usar_videos_es else None
                score_v_en  = eval_frase(frase, model_videos_en, "Videos", "English") if usar_videos_en else None

            # Mostramos en un DataFrame
            df_result = pd.DataFrame({
                "Shorts ES": [score_sh_es],
                "Shorts EN": [score_sh_en],
                "Videos ES": [score_v_es],
                "Videos EN": [score_v_en]
            })
            st.write("### Resultado para la frase ingresada")
            st.dataframe(df_result, use_container_width=True)

            # (Opcional) Botón para descargar este resultado en CSV
            csv_data = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar resultado (CSV)",
                data=csv_data,
                file_name="resultado_frase.csv",
                mime="text/csv"
            )

    # ================================
    # TAB 2: Evaluación por Archivo
    # ================================
    with tab2:
        st.header("Evaluar un archivo con frases")
        st.write("Sube un archivo CSV o TXT con **una frase por renglón**.")

        # Subir archivo
        archivo_subido = st.file_uploader("Selecciona archivo (CSV o TXT)", type=["csv", "txt"])

        if archivo_subido is not None:
            # Intentamos leerlo como CSV
            try:
                df_frases = pd.read_csv(archivo_subido, header=None, names=["frase"])
            except:
                # Si falla, tal vez es TXT
                archivo_subido.seek(0)
                lineas = archivo_subido.read().decode("utf-8").splitlines()
                df_frases = pd.DataFrame({"frase": lineas})

            st.write(f"**Se han cargado {len(df_frases)} frases.**")

            if st.button("Evaluar Archivo"):
                with st.spinner("Evaluando todas las frases..."):
                    resultados = []
                    for i, row in df_frases.iterrows():
                        frase_texto = row["frase"]
                        # Calcular los scores, 0 si no se usa o None para distinguir
                        s_sh_es = eval_frase(frase_texto, model_shorts_es, "Shorts", "Spanish") if usar_shorts_es else None
                        s_sh_en = eval_frase(frase_texto, model_shorts_en, "Shorts", "English") if usar_shorts_en else None
                        s_v_es  = eval_frase(frase_texto, model_videos_es,  "Videos", "Spanish") if usar_videos_es else None
                        s_v_en  = eval_frase(frase_texto, model_videos_en,  "Videos", "English") if usar_videos_en else None

                        # Suma el valor numérico, asumiendo 0 cuando es None
                        total_score = (s_sh_es or 0) + (s_sh_en or 0) + (s_v_es or 0) + (s_v_en or 0)

                        resultados.append({
                            "frase": frase_texto,
                            "Shorts ES": s_sh_es,
                            "Shorts EN": s_sh_en,
                            "Videos ES": s_v_es,
                            "Videos EN": s_v_en,
                            "Score Total": total_score
                        })

                # Ordenar por score total descendente
                resultados_ordenados = sorted(resultados, key=lambda x: x["Score Total"], reverse=True)
                # Tomar top_n
                top_n_resultados = resultados_ordenados[:top_n]

                # Convertir a DataFrame
                df_top = pd.DataFrame(top_n_resultados)
                st.write(f"### Mejores {top_n} resultados")
                st.dataframe(df_top, use_container_width=True)

                # Botón para descargar resultados
                csv_full = pd.DataFrame(resultados_ordenados).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Descargar TODOS los resultados (CSV)",
                    data=csv_full,
                    file_name="resultados_completos.csv",
                    mime="text/csv"
                )

                # (Opcional) Si solo quieres descargar el top_n
                csv_top = df_top.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Descargar TOP {top_n} resultados (CSV)",
                    data=csv_top,
                    file_name=f"top_{top_n}_resultados.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
