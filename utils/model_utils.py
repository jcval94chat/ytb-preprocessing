# utils/model_utils.py

import pandas as pd
import numpy as np
import random
import itertools
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import spacy
# Asegúrate de cargar nlp_es y nlp_en en main o donde corresponda

# -----------------------------------------------------------------------------------
# Funciones principales que se usan dentro de "get_best_words"
# -----------------------------------------------------------------------------------

def process_text(text, nlp, include_stopwords=True):
    """
    Procesa un texto aplicando minúsculas, lematización y
    filtrado de caracteres alfabéticos. Quita stopwords según `include_stopwords`.
    """
    doc = nlp(text.lower())
    if include_stopwords:
        tokens = [token.lemma_ for token in doc if token.is_alpha]
    else:
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    if len(tokens) == 0:
        return 'no title'
    return ' '.join(tokens)

def preprocess_text_column(df, column, language, stopw, nlp_es, nlp_en):
    """
    Preprocesa columna (title, description, tags) en un DataFrame
    usando spaCy según idioma.
    """
    if language == 'Spanish':
        nlp = nlp_es
    elif language == 'English':
        nlp = nlp_en
    else:
        return df

    df[column] = df[column].fillna('').apply(lambda x: process_text(x, nlp, include_stopwords=stopw))
    return df

def train_model(df, language, objetivos=['avg_daily_likes'], tipo_texto='title', stopw=True,
                nlp_es=None, nlp_en=None):
    """
    Entrena un RandomForestRegressor con TF-IDF.
    Retorna (df_lang, model, X_train, X_test, y_train, y_test, vectorizer_title, feature_names, importances).
    """
    df_lang = df[df['language'] == language].dropna(subset=objetivos)
    if df_lang.empty:
        return (None,)*9

    df_lang = preprocess_text_column(df_lang, tipo_texto, language, stopw, nlp_es, nlp_en)
    df_lang = df_lang[df_lang[tipo_texto].str.strip() != '']
    if df_lang.empty:
        return (None,)*9

    vectorizer_title = TfidfVectorizer(max_features=5000)
    X_title = vectorizer_title.fit_transform(df_lang[tipo_texto])
    X = X_title
    y = df_lang[objetivos]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[{language} - {tipo_texto}] MAE: {mae:.2f} | MSE: {mse:.2f}")

    feature_names = vectorizer_title.get_feature_names_out()
    importances = model.feature_importances_

    return (df_lang, model, X_train, X_test, y_train, y_test, vectorizer_title, feature_names, importances)

def predict_avg_daily_likes(model, vectorizer_title, text, language, stopw, nlp_es, nlp_en):
    """
    Predice avg_daily_likes para un texto usando el modelo entrenado.
    """
    if language == 'Spanish':
        nlp = nlp_es
    else:
        nlp = nlp_en

    processed_text = process_text(text, nlp, include_stopwords=stopw)
    X_title = vectorizer_title.transform([processed_text])
    predicted = model.predict(X_title)
    return predicted[0]

def generate_top_texts(model, vectorizer_title, predict_function, feature_names, importances,
                       language, top_n=5, sample_fraction=0.1, stopw=True, nlp_es=None, nlp_en=None):
    """
    Genera textos que maximizan la predicción, combinando palabras más importantes.
    """
    indices = np.argsort(importances)[::-1]
    top_words = [feature_names[idx] for idx in indices[:15]]

    max_text_length = 6
    all_combinations = []

    for L in range(1, max_text_length + 1):
        total_combos = list(itertools.combinations(top_words, L))
        num_combos = len(total_combos)
        num_samples = max(1, int(num_combos * sample_fraction))
        sampled_combos = random.sample(total_combos, min(num_samples, num_combos))
        all_combinations.extend(sampled_combos)

    all_texts = [' '.join(combo) for combo in all_combinations]
    all_texts = list(set(all_texts))

    predictions = []
    for text_ in all_texts:
        pred_likes = predict_function(model, vectorizer_title, text_, language, stopw, nlp_es, nlp_en)
        predictions.append((text_, pred_likes))

    predictions.sort(key=lambda x: x[1][0], reverse=True)
    if top_n is None:
        top_texts = [text_ for (text_, _) in predictions]
        top_preds = [pred for (_, pred) in predictions]
    else:
        top_texts = [text_ for (text_, _) in predictions[:top_n]]
        top_preds = [pred for (_, pred) in predictions[:top_n]]

    return top_texts, top_preds

# -----------------------------------------------------------------------------------
# Función final que usa las de arriba para obtener "mejores palabras"
# -----------------------------------------------------------------------------------

def get_best_words(df, lang_='Spanish', stopw=True):
    """
    Calcula mejores palabras entrenando RF en 'title','description','tags'.
    """
    objetivos = ['avg_daily_likes','avg_daily_views']

    # Agrega columnas logarítmicas
    for var in objetivos:
        df['Log_' + var] = np.log(df[var] + 0.1)

    # Normaliza por canal (ejemplo muy simple)
    df = normalize_log_columns_by_channel(df)
    df['top_video'] = ((df['Log_avg_daily_likes'] > 0) & (df['Log_avg_daily_views'] > 0)).astype(int)
    df['Success_Value'] = df['Log_avg_daily_likes']*100 + ((df['Log_avg_daily_views']>0)*100)

    # LongTerm
    df['LongTerm_Success'] = calculate_long_term_success(df, 'Success_Value', 'days_since_upload')
    objetivos_logs = [x for x in df.columns if 'Log_' in x]
    mejores_palabras = {}

    # Cargar los modelos spaCy en memoria:
    # (Ojo, en la práctica convendría cargarlos fuera de la función para no recargarlos cada vez)
    nlp_es = spacy.load("es_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")

    for tipo_texto in ['title','description','tags']:
        # Entrenamos
        (df_lang, model, X_train, X_test, y_train, y_test,
         vectorizer_title, feature_names, importances) = train_model(
             df, lang_, objetivos_logs, tipo_texto, stopw, nlp_es, nlp_en
         )
        if model is None:
            continue

        # Generamos ejemplos
        top_texts, top_predictions = generate_top_texts(
            model, vectorizer_title, predict_avg_daily_likes,
            feature_names, importances, lang_,
            top_n=500, sample_fraction=0.1, stopw=stopw,
            nlp_es=nlp_es, nlp_en=nlp_en
        )

        # Guardamos
        obj_key = ", ".join(objetivos_logs)
        mejores_palabras[tipo_texto + "_" + obj_key] = [top_texts, top_predictions]

    return mejores_palabras, df

def normalize_log_columns_by_channel(df):
    """
    Normaliza columnas Log_ por 'channel_name'.
    """
    for column in df.columns:
        if column.startswith('Log_'):
            df[column] = df.groupby('channel_name')[column].transform(lambda x: (x - x.mean()) / (x.std()+1e-9))
    return df

def calculate_long_term_success(df, success_column, days_column):
    """
    Calcula una métrica de éxito a largo plazo basada en log y raíz cuadrada de días.
    """
    if success_column not in df.columns or days_column not in df.columns:
        return 0

    log_factor = np.log(df[days_column] + 2)
    sqrt_factor = np.sqrt(df[days_column] + 1)
    combined_factor = (log_factor + sqrt_factor)/2

    return df[success_column]*combined_factor
