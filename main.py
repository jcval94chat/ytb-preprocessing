# main.py

import joblib
import logging
import os
import traceback
import pandas as pd
import spacy
import nltk
import math
import json
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ----------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('youtube_data.log', mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# ----------------------------------------------------------------------------

# Descarga de recursos nltk (si no se hubieran descargado antes)
nltk.download("punkt")
nltk.download("stopwords")

# ----------------------------------------------------------------------------
# IMPORTS DE FUNCIONES DE UTILERÍA
# ----------------------------------------------------------------------------
from utils.google_utils import (
    get_sheets_data_from_folder,
    upload_dataframe_to_google_sheet
)
from utils.text_processing import (
    detect_language,
    calculate_avg_daily_metrics,
    get_best_words_df,
    get_top_bottom_videos
)
from InsideForest import trees, models, regions, labels

# Estas funciones asumen que tu proyecto las tiene definidas en `utils/interpret_clusters.py` (o donde corresponda):
from utils.interpret_clusters import (
    get_model_interpret,
    get_dict_descripciones
)

# from utils.model_manager import (
#     guardar_modelo, 
#     cargar_todos_los_modelos, 
#     predecir_con_onnx
# )

# ----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------------

def main():
    """
    Script principal que:
      1) Obtiene datos de Google Sheets (carpeta en Drive),
      2) Elimina duplicados,
      3) Procesa DataFrame (detección de idioma, métricas diarias, etc.),
      4) Separa en VIDEOS/SHORTS,
      5) Extrae "mejores palabras", top & bottom videos,
      6) Realiza interpretación de modelo con InsideForest,
      7) Limpia cluster_descripcion y sube resultados a Google Sheets.
    """

    # 1. Leer secrets de variables de entorno (definidas en GitHub Actions, por ejemplo)
    folder_id = os.environ.get("SECRET_FOLDER_ID", None)
    creds_file = os.environ.get("SECRET_CREDS_FILE", None)
    spreadsheet_id_postprocess = os.environ.get("SPREADSHEET_ID_POSTPROCESS", None)

    if not folder_id or not creds_file:
        logger.error("No se pudieron obtener 'folder_id' o 'creds_file' desde los secrets.")
        return  # Terminamos, pues no hay cómo continuar
    
    # En lugar de hardcodear "es_core_news_sm", tomamos la env var
    es_model = os.environ.get("SPACY_ES_MODEL", "es_core_news_sm")
    en_model = os.environ.get("SPACY_EN_MODEL", "en_core_web_sm")
    
    
    # 4.1. Cargar modelos de lenguaje spaCy (asígnales a variables globales si deseas)
    #      Si los usas en otras partes, podrías hacerlo aquí o en un módulo aparte
    logger.info("Cargando modelos spaCy (es_core_news_sm, en_core_web_sm)...")
    # Ahora spaCy cargará esos modelos. 
    # Ya están instalados (descargados) en el paso anterior del workflow.
    nlp_es = spacy.load(es_model)
    nlp_en = spacy.load(en_model)

    
    # 2. Obtener DataFrame desde Google Sheets en una carpeta de Drive
    logger.info(f"Obteniendo datos de la carpeta con ID='{folder_id}'...")
    combined_df = get_sheets_data_from_folder(
        folder_id=folder_id,
        creds_file=creds_file,
        days=30,        # Ajusta según tus necesidades
        max_files=60,   # Límite de archivos a leer
        sleep_seconds=2 # Pausa entre lecturas para no saturar la API
    )
    if combined_df is None:
        logger.warning("No se obtuvo ningún DataFrame (None). Abortando proceso.")
        return

    logger.info(f"Dimensiones iniciales del DataFrame: {combined_df.shape}")

    # 3. Ejemplo: eliminar duplicados
    logger.info("Eliminando duplicados por ['video_id','execution_date']...")
    combined_df.drop_duplicates(subset=['video_id','execution_date'], inplace=True)
    logger.info(f"Dimensiones tras drop_duplicates: {combined_df.shape}")

    # 4.2. Eliminar duplicados por ['title','upload_date','channel_name']
    combined_df.drop_duplicates(subset=['title','upload_date','channel_name'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    logger.info(f"Después de eliminar duplicados (title/upload_date/channel): {combined_df.shape}")

    # 4.3. Separar en df_SHORTS y df_VIDEOS
    df_ALL = combined_df.copy()
    df_SHORTS = df_ALL[df_ALL['duration_seconds'].astype(int) <= 62].copy()
    df_VIDEOS = df_ALL[df_ALL['duration_seconds'].astype(int) > 62].copy()
    logger.info(f"Se obtuvieron {df_VIDEOS.shape[0]} 'VIDEOS' y {df_SHORTS.shape[0]} 'SHORTS'.")

    # 4.4. Detectar idioma, calcular métricas, etc. para VIDEOS
    df_VIDEOS = detect_language(df_VIDEOS)
    df_VIDEOS = calculate_avg_daily_metrics(df_VIDEOS)

    # 5. get_best_words_df para VIDEOS
    df_indice_all, df_S_dict = get_best_words_df(df_VIDEOS, stopw=False)
    df_indice_all = df_indice_all[df_indice_all['language'] != 'Other']
    logger.info(f"df_indice_all (VIDEOS) => {df_indice_all.shape}")

    df_S_df_ = pd.concat([v for k, v in df_S_dict.items()], axis=0)

    # 5.1. Top & bottom videos por canal (VIDEOS)
    top_5, bottom_5 = get_top_bottom_videos(
        df_S_df_, channel_column="channel_name", metric_column="LongTerm_Success", top_n=6
    )
    Extreme_Titles = pd.concat([top_5, bottom_5], axis=0).drop_duplicates(subset=['channel_name','title'])
    logger.info(f"Se extrajeron Extreme_Titles (VIDEOS) => {Extreme_Titles.shape}")

    # 6. Lo mismo para SHORTS
    df_SHORTS = detect_language(df_SHORTS)
    df_SHORTS = calculate_avg_daily_metrics(df_SHORTS)

    df_indice_all_SHORTS, df_S_dict_SHORTS = get_best_words_df(df_SHORTS, stopw=False)
    df_indice_all_SHORTS = df_indice_all_SHORTS[df_indice_all_SHORTS['language'] != 'Other']

    top_5_s, bottom_5_s = get_top_bottom_videos(
        df_S_df_, channel_column="channel_name", metric_column="LongTerm_Success", top_n=10
    )
    Extreme_Titles = pd.concat([top_5_s, bottom_5_s], axis=0).drop_duplicates(subset=['channel_name','title'])
    logger.info(f"Se extrajeron Extreme_Titles (SHORTS) => {Extreme_Titles.shape}")

    df_S_df_SHORTS = pd.concat([v for k, v in df_S_dict_SHORTS.items()], axis=0)
    top_5_SHORTS, bottom_5_SHORTS = get_top_bottom_videos(
        df_S_df_SHORTS, channel_column="channel_name", metric_column="LongTerm_Success", top_n=10
    )
    Extreme_Titles_SHORTS = pd.concat([top_5_SHORTS, bottom_5_SHORTS], axis=0).drop_duplicates(subset=['channel_name','title'])

    # 7. (Opcional) InsideForest: arboles, modelos, regiones, labels
    logger.info("Interpretabilidad de los modelos")

    # 8. Interpretación y clusterización (ejemplo)
    (df_titulos_full_info, mejores_clusters, df_datos_clusterizados,
     palabras_top, feature_columns, adfuhie_dafs, lista_modelos) = get_model_interpret(df_S_dict)

    (df_titulos_full_info_SH, mejores_clusters_SH, df_datos_clusterizados_SH,
     palabras_top_SH, feature_columns_SH, adfuhie_dafs_SH, lista_modelos_SH) = get_model_interpret(df_S_dict_SHORTS)

    # 8.1. Limpieza de descripciones de clusters
    logger.info("Generando diccionarios de descripciones limpias para clusters...")
    cluster_cleaned_dict = get_dict_descripciones(adfuhie_dafs, palabras_top, feature_columns, df_datos_clusterizados)
    cluster_cleaned_dict_SH = get_dict_descripciones(adfuhie_dafs_SH, palabras_top_SH, feature_columns_SH, df_datos_clusterizados_SH)

    df_datos_clusterizados['cluster_descripcion'] = df_datos_clusterizados['cluster_descripcion'].replace(cluster_cleaned_dict)
    df_titulos_full_info['cluster_descripcion'] = df_titulos_full_info['cluster_descripcion'].replace(cluster_cleaned_dict)
    df_datos_clusterizados_SH['cluster_descripcion'] = df_datos_clusterizados_SH['cluster_descripcion'].replace(cluster_cleaned_dict_SH)
    df_titulos_full_info_SH['cluster_descripcion'] = df_titulos_full_info_SH['cluster_descripcion'].replace(cluster_cleaned_dict_SH)
    mejores_clusters['cluster_descripcion'] = mejores_clusters['cluster_descripcion'].replace(cluster_cleaned_dict)
    mejores_clusters_SH['cluster_descripcion'] = mejores_clusters_SH['cluster_descripcion'].replace(cluster_cleaned_dict_SH)

    # 9. Subir DataFrames a Google Sheets (opcional)
    if spreadsheet_id_postprocess:
        logger.info("Subiendo DataFrames a Google Sheets...")

        # Ejemplos de subidas
        exito = upload_dataframe_to_google_sheet(Extreme_Titles, creds_file, spreadsheet_id_postprocess, 'titulos')
        exito = upload_dataframe_to_google_sheet(df_indice_all, creds_file, spreadsheet_id_postprocess, 'palabras')
        exito = upload_dataframe_to_google_sheet(df_titulos_full_info, creds_file, spreadsheet_id_postprocess, 'titulos_cluster')
        exito = upload_dataframe_to_google_sheet(mejores_clusters, creds_file, spreadsheet_id_postprocess, 'clusters')

        exito = upload_dataframe_to_google_sheet(Extreme_Titles_SHORTS, creds_file, spreadsheet_id_postprocess, 'titulos_shorts')
        exito = upload_dataframe_to_google_sheet(df_indice_all_SHORTS, creds_file, spreadsheet_id_postprocess, 'palabras_shorts')
        exito = upload_dataframe_to_google_sheet(df_titulos_full_info_SH, creds_file, spreadsheet_id_postprocess, 'titulos_cluster_shorts')
        exito = upload_dataframe_to_google_sheet(mejores_clusters_SH, creds_file, spreadsheet_id_postprocess, 'clusters_shorts')

    logger.info("¡Proceso finalizado con éxito!")

    # Verificar el tipo antes de guardar
    logger.info("Verificando tipos de modelos antes de guardar:")
    for idx, m in enumerate(lista_modelos):
        logger.info(f"[VIDEOS] lista_modelos[{idx}] es de tipo: {type(m)}")

    for idx, m in enumerate(lista_modelos_SH):
        logger.info(f"[SHORTS] lista_modelos_SH[{idx}] es de tipo: {type(m)}")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    joblib.dump(lista_modelos[0], "models/model_videos_es.joblib")
    joblib.dump(lista_modelos[1], "models/model_videos_en.joblib")
    joblib.dump(lista_modelos_SH[0], "models/model_shorts_es.joblib")
    joblib.dump(lista_modelos_SH[1], "models/model_shorts_en.joblib")

    logger.info("¡Modelos guardados (o actualizados) en la carpeta 'models/'!")

    # Asegúrate de que la carpeta existe
    if not os.path.exists("models"):
        os.makedirs("otros_objetos")
    
    # Guardar como JSON
    with open("otros_objetos/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    
    with open("otros_objetos/feature_columns_SH.json", "w") as f:
        json.dump(feature_columns_SH, f)
    
    with open("otros_objetos/palabras_top.json", "w") as f:
        json.dump(palabras_top, f)
    
    with open("otros_objetos/palabras_top_SH.json", "w") as f:
        json.dump(palabras_top_SH, f)


# ----------------------------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
