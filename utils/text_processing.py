# utils/text_processing.py

import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords
import re

# ---------------------------------------------------------------------
# Funciones de limpieza / análisis de texto
# ---------------------------------------------------------------------

def detect_language(df):
    """
    Detecta el idioma de cada fila combinando 'title' y 'description'.
    Agrega la columna 'language' con valores 'Spanish', 'English' u 'Other'.
    """
    DetectorFactory.seed = 0
    def detect_lang(text):
        try:
            return detect(text)
        except:
            return 'unknown'

    df['title_desc'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['language'] = df['title_desc'].apply(detect_lang)

    language_map = {'es': 'Spanish', 'en': 'English'}
    df['language'] = df['language'].map(language_map).fillna('English')  # Asumes default English
    return df

def calculate_avg_daily_metrics(df):
    """
    Calcula las métricas promedio diario (views, likes, comments) basado en la fecha de subida y la de ejecución.
    """
    df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
    df['execution_date'] = pd.to_datetime(df['execution_date'], errors='coerce')

    df['days_since_upload'] = (df['execution_date'] - df['upload_date']).dt.days
    df['days_since_upload'] = df['days_since_upload'].fillna(0).astype(int)
    df['days_since_upload'] += 2  # evitar div/0

    df['views'] = df['views'].fillna(0).astype(int)
    df['likes'] = df['likes'].fillna(0).astype(int)
    df['comments'] = df['comments'].fillna(0).astype(int)

    df['avg_daily_views'] = df['views'] / df['days_since_upload']
    df['avg_daily_likes'] = df['likes'] / df['days_since_upload']
    df['avg_daily_comments'] = df['comments'] / df['days_since_upload']

    return df

# ---------------------------------------------------------------------
# Funciones para agrupar / seleccionar top & bottom
# ---------------------------------------------------------------------

def get_top_bottom_videos(df, channel_column, metric_column, top_n=5):
    """
    Obtiene el top y bottom N de videos por canal según una métrica específica.
    """
    top_videos = df.groupby(channel_column).apply(
        lambda group: group.nlargest(top_n, metric_column)
    ).reset_index(drop=True)

    bottom_videos = df.groupby(channel_column).apply(
        lambda group: group.nsmallest(top_n, metric_column)
    ).reset_index(drop=True)

    cols_sub = [channel_column,"title",metric_column,'language','days_since_upload']
    return top_videos[cols_sub], bottom_videos[cols_sub]

# ---------------------------------------------------------------------
# Capa final: unifica la extracción de "mejores palabras"
# ---------------------------------------------------------------------

from .model_utils import get_best_words  # Importamos la función que entrena

def get_best_words_df(df_VIDEOS, stopw=False):
    """
    Usa get_best_words() internamente para cada idioma.
    Retorna:
      df_indice_all: DataFrame final con índice de efectividad
      df_S_dict: Diccionario con DataFrames transformados por idioma
    """
    # Detecta idiomas en df_VIDEOS (ya asumimos que se ha hecho detect_language fuera, pero por si acaso):
    languages = df_VIDEOS['language'].unique()

    df_indice_list = []
    df_S_dict = {}

    for lang in languages:
        df_VIDEOS_lang = df_VIDEOS[df_VIDEOS['language'] == lang].copy()
        if df_VIDEOS_lang.empty:
            continue

        # Obtener las mejores palabras para el idioma
        mejores_palabras_lang, df_S_lang = get_best_words(df_VIDEOS_lang, lang, stopw)

        if not mejores_palabras_lang:
            continue

        # Unificar
        df_indice_lang = unify_repeated_words(mejores_palabras_lang)
        df_indice_lang = calcular_indice_efectividad(df_indice_lang)
        df_indice_lang['language'] = lang

        df_indice_list.append(df_indice_lang)
        df_S_dict[lang] = df_S_lang

    if df_indice_list:
        df_indice_all = pd.concat(df_indice_list, ignore_index=True)
    else:
        df_indice_all = pd.DataFrame(columns=['category','word','count_best','count_worst','indice_efectividad','language'])

    return df_indice_all, df_S_dict

# ---------------------------------------------------------------------
# Sub-funciones de unificación de palabras, etc.
# ---------------------------------------------------------------------

def unify_repeated_words(mejores_palabras, top_n=150):
    """
    Consolida palabras más repetidas de las categorías (title, tags, description).
    """
    categories = ["title", "tags", "description"]
    pattern = "Log_avg_daily_likes, Log_avg_daily_views"

    frames = []
    for category in categories:
        key = f"{category}_{pattern}"
        if key not in mejores_palabras:
            continue

        df_temp = pd.DataFrame(mejores_palabras[key]).T

        df_best = df_temp.head(top_n)
        df_worst = df_temp.tail(top_n)

        most_repeated_best = get_most_repeated_words(df_best, 0, top_n=top_n).copy()
        most_repeated_worst = get_most_repeated_words(df_worst, 0, top_n=top_n).copy()

        most_repeated_best = most_repeated_best.reset_index().rename(columns={'index': 'word', 0:'count'})
        most_repeated_worst = most_repeated_worst.reset_index().rename(columns={'index': 'word', 0:'count'})

        most_repeated_best["type"] = "best"
        most_repeated_worst["type"] = "worst"
        most_repeated_best["category"] = category
        most_repeated_worst["category"] = category

        most_repeated_best = most_repeated_best[['category','type','word','count']]
        most_repeated_worst = most_repeated_worst[['category','type','word','count']]

        frames.append(most_repeated_best)
        frames.append(most_repeated_worst)

    if frames:
        final_df = pd.concat(frames, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=['category','type','word','count'])
    return final_df

def get_most_repeated_words(df, column_name, top_n=10):
    """
    Obtiene las palabras más repetidas en la columna dada.
    """
    all_text = ' '.join(df[column_name].astype(str).tolist())
    word_counts = {}
    for word in all_text.split():
        w = word.lower()
        word_counts[w] = word_counts.get(w, 0) + 1
    word_counts_series = pd.Series(word_counts).sort_values(ascending=False)
    return word_counts_series.head(top_n)

def calcular_indice_efectividad(df):
    """
    Calcula el Índice de Efectividad para cada palabra en cada categoría,
    asumiendo que el DataFrame tiene ['category','type','word','count'].
    """
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
    df_best = df[df['type'] == 'best'].groupby(['category','word'])['count'].sum().reset_index()
    df_best.rename(columns={'count': 'count_best'}, inplace=True)

    df_worst = df[df['type'] == 'worst'].groupby(['category','word'])['count'].sum().reset_index()
    df_worst.rename(columns={'count': 'count_worst'}, inplace=True)

    df_combined = pd.merge(df_best, df_worst, on=['category','word'], how='outer')
    df_combined['count_best'] = df_combined['count_best'].fillna(0)
    df_combined['count_worst'] = df_combined['count_worst'].fillna(0)

    df_combined['indice_efectividad'] = (df_combined['count_best']+1) / (df_combined['count_worst'] + 1)
    df_combined.sort_values(by=['category','indice_efectividad'], ascending=[True,False], inplace=True)

    return df_combined[['category','word','count_best','count_worst','indice_efectividad']]
