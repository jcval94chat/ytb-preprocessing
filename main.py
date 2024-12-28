# main.py

import logging
import os

import pandas as pd
import spacy
import nltk

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
from utils.model_utils import (
    get_model_interpret
)
# Nota: Si tienes otras funciones en interpret_clusters, etc., impórtalas según necesites

# ----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ----------------------------------------------------------------------------

def main():
    """
    Ejemplo de script principal que:
      1) Obtiene datos de Google Sheets (carpeta en Drive),
      2) Elimina duplicados,
      3) Procesa DataFrame (detección de idioma, métricas diarias, etc.),
      4) Extrae "mejores palabras" y "top/bottom videos",
      5) Sube resultados (opcional) a Google Sheets.
    """

    # 1. Leer secrets de variables de entorno (definidas en GitHub Actions)
    folder_id = os.environ.get("SECRET_FOLDER_ID", None)
    creds_file = os.environ.get("SECRET_CREDS_FILE", None)

    if not folder_id or not creds_file:
        logger.error("No se pudieron obtener 'folder_id' o 'creds_file' desde los secrets.")
        return  # Terminamos, pues no hay cómo continuar

    # 2. Obtener DataFrame desde Google Sheets en una carpeta de Drive
    logger.info(f"Obteniendo datos de la carpeta con ID='{folder_id}'...")
    combined_df = get_sheets_data_from_folder(
        folder_id=folder_id,
        creds_file=creds_file,
        days=30,
        max_files=60,
        sleep_seconds=2
    )
    if combined_df is None:
        logger.warning("No se obtuvo ningún DataFrame (None). Abortando proceso.")
        return

    logger.info(f"Dimensiones iniciales del DataFrame: {combined_df.shape}")

    # 3. Ejemplo: eliminar duplicados
    logger.info("Eliminando duplicados por ['video_id','execution_date']...")
    combined_df.drop_duplicates(subset=['video_id','execution_date'], inplace=True)
    logger.info(f"Dimensiones tras drop_duplicates: {combined_df.shape}")

    # 4. Procesamiento adicional (ej. detección de idioma, cálculo de métricas)
    logger.info("Detectando idioma y calculando métricas diarias...")
    combined_df = detect_language(combined_df)
    combined_df = calculate_avg_daily_metrics(combined_df)

    # 5. Dividir en VIDEOS y SHORTS (ejemplo)
    df_videos = combined_df[combined_df['duration_seconds'].astype(int) > 62].copy()
    df_shorts = combined_df[combined_df['duration_seconds'].astype(int) <= 62].copy()

    logger.info(f"Se obtuvieron {df_videos.shape[0]} 'VIDEOS' y {df_shorts.shape[0]} 'SHORTS'.")

    # 6. Ejemplo de extracción de mejores palabras
    df_indice_all, df_S_dict = get_best_words_df(df_videos, stopw=False)
    logger.info(f"get_best_words_df() generó {df_indice_all.shape[0]} filas en df_indice_all")

    # 7. Top & bottom videos
    df_S_df_videos = pd.concat([v for k, v in df_S_dict.items()], axis=0)
    top_5_v, bottom_5_v = get_top_bottom_videos(
        df_S_df_videos, channel_column="channel_name",
        metric_column="LongTerm_Success", top_n=5
    )
    extreme_titles_videos = pd.concat([top_5_v, bottom_5_v], axis=0).drop_duplicates(subset=['channel_name','title'])
    logger.info("Extraídos top y bottom videos para 'VIDEOS'")

    # Repetir para shorts si deseas
    df_indice_all_shorts, df_S_dict_shorts = get_best_words_df(df_shorts, stopw=False)
    df_S_df_shorts = pd.concat([v for k, v in df_S_dict_shorts.items()], axis=0)
    top_5_s, bottom_5_s = get_top_bottom_videos(
        df_S_df_shorts, channel_column="channel_name",
        metric_column="LongTerm_Success", top_n=5
    )
    extreme_titles_shorts = pd.concat([top_5_s, bottom_5_s], axis=0).drop_duplicates(subset=['channel_name','title'])

    # 8. Ejemplo de "interpretación de modelo" (si usas InsideForest)
    #    Nota: get_model_interpret() es solo un ejemplo de la lógica que
    #    habías descrito. Ajusta a tu caso real.
    df_titulos_full_info, mejores_clusters, df_datos_clusterizados, palabras_top, feature_columns, adfuhie_dafs = \
        get_model_interpret(df_S_dict)

    logger.info("Interpretación de clusters completada.")

    # 9. (Opcional) Subir resultados a Google Sheets
    logger.info("Subiendo resultados a Google Sheets (opcional)...")
    spreadsheet_id_postprocess = os.environ.get("SPREADSHEET_ID_POSTPROCESS", None)  # otro secret, por ejemplo
    if spreadsheet_id_postprocess:
        upload_dataframe_to_google_sheet(extreme_titles_videos, creds_file, spreadsheet_id_postprocess, 'titulos_videos')
        upload_dataframe_to_google_sheet(df_indice_all, creds_file, spreadsheet_id_postprocess, 'palabras_videos')
        upload_dataframe_to_google_sheet(df_titulos_full_info, creds_file, spreadsheet_id_postprocess, 'titulos_cluster_videos')
        upload_dataframe_to_google_sheet(mejores_clusters, creds_file, spreadsheet_id_postprocess, 'clusters_videos')
        
        # Y lo mismo para shorts
        upload_dataframe_to_google_sheet(extreme_titles_shorts, creds_file, spreadsheet_id_postprocess, 'titulos_shorts')
        upload_dataframe_to_google_sheet(df_indice_all_shorts, creds_file, spreadsheet_id_postprocess, 'palabras_shorts')
        # etc...

    logger.info("Proceso finalizado con éxito.")

# ----------------------------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
