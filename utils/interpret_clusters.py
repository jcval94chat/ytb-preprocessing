# utils/interpret_clusters.py

import re
from InsideForest import trees, regions
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter
import langid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def get_model_interpret(df_S_dict):
    """
    Ejecuta el pipeline con InsideForest, regiones, etc.
    Crea clusterización y saca mejores/peores títulos.
    """

    # Preparamos dos "idiomas"
    # Asumiendo que df_S_dict es un diccionario con 'English' y 'Spanish'
    dfs_ls_ = [v for k, v in df_S_dict.items()]

    # Instanciar
    arbolesPY = trees('py')
    regiones = regions()

    # 1) get_model_idioma
    idioma_es_en, feature_columns, palabras_top = get_model_idioma(dfs_ls_)
    # lista de modelos
    lista_de_modelos = [model for (model, _) in idioma_es_en]
    
    # Tomamos cada una
    model_en, df_en = idioma_es_en[1]
    model_es, df_es = idioma_es_en[0]

    var_obj = 'target'

    # 2) Lógica de separacion
    separacion_dim_en = arbolesPY.get_branches(df_en[feature_columns+[var_obj]], var_obj, model_en)
    df___ = df_en[feature_columns+[var_obj]]
    df_reres = regiones.prio_ranges(separacion_dim_en, df___)
    df_datos_clusterizados = regiones.labels(df___, df_reres, True)

    # 3) Categorizar
    adfuhie_dafs = [x for x in df_datos_clusterizados['cluster_descripcion'].unique().tolist() if isinstance(x, str)]

    # 4) Sumar "best/worst" titles
    (df_mejores_titulos_en, df_peores_titulos_en,
     mejores_titulos_en_agg, peores_titulos_en_agg) = get_best_worst_titles(df_en, df_datos_clusterizados, var_obj)

    (df_mejores_titulos_es, df_peores_titulos_es,
     mejores_titulos_es_agg, peores_titulos_es_agg) = get_best_worst_titles(df_es, df_datos_clusterizados, var_obj)

    df_mejores_titulos_en['Type'] = 'Mejores'
    df_peores_titulos_en['Type'] = 'Peores'
    df_titulos_en = pd.concat([df_mejores_titulos_en, df_peores_titulos_en])

    df_mejores_titulos_es['Type'] = 'Mejores'
    df_peores_titulos_es['Type'] = 'Peores'
    df_titulos_es = pd.concat([df_mejores_titulos_es, df_peores_titulos_es])

    df_titulos_en['language'] = 'English'
    df_titulos_es['language'] = 'Spanish'
    df_titulos_full_info = pd.concat([df_titulos_en, df_titulos_es])

    mejores_titulos_en_agg['language'] = 'English'
    mejores_titulos_es_agg['language'] = 'Spanish'
    mejores_clusters = pd.concat([mejores_titulos_en_agg, mejores_titulos_es_agg])

    return (df_titulos_full_info, mejores_clusters, df_datos_clusterizados,
            palabras_top, feature_columns, adfuhie_dafs, lista_de_modelos)

def get_model_idioma(dfs_ls_):
    """
    Simplificado: entrena un modelo "RandomForestClassifier" en cada DataFrame (asumiendo 2: ES y EN).
    Retorna: [ (model_es, df_es), (model_en, df_en) ], feature_columns, palabras_top
    """
    import pandas as pd
    # Ejemplo ficticio
    if len(dfs_ls_) < 2:
        return [], [], {}

    # Para simplificar, supón que: dfs_ls_[0] => Spanish, dfs_ls_[1] => English
    # En la práctica, deberías filtrar por df['language'] etc.
    df_es = prepare_lang(dfs_ls_[0])
    df_en = prepare_lang(dfs_ls_[1])

    # Extraer features
    features_es, palabras_top_es = generate_features_extended(df_es, "title")
    feature_cols = [c for c in features_es.columns if c not in ["common_words","language"]]
    df_es = pd.concat([df_es, features_es], axis=1)

    # Entrenar (este es un ej. muy simplificado)
    model_es = train_model_lg(df_es, "target", feature_cols)

    # Repetimos para df_en
    features_en, palabras_top_en = generate_features_extended(df_en, "title")
    df_en = pd.concat([df_en, features_en], axis=1)
    model_en = train_model_lg(df_en, "target", feature_cols)

    # Mezclamos los top "palabras_top" (no es muy esencial)
    palabras_top = {}
    palabras_top.update(palabras_top_es)
    palabras_top.update(palabras_top_en)

    return [(model_es, df_es), (model_en, df_en)], feature_cols, palabras_top

# Funciones complementarias

def prepare_lang(df_S):
    """
    Ajusta language a 'en' o 'es', y crea 'target' en base a mediana de 'LongTerm_Success'.
    """
    import numpy as np
    df = df_S.copy()
    df['language'] = df['language'].replace('English', 'en')
    df['language'] = df['language'].replace('Spanish','es')
    df["target"] = (df["LongTerm_Success"] > df["LongTerm_Success"].median()).astype(int)
    return df


def generate_features_extended(df, title_column, most_common_words=None):
    """
    Genera features para un clasificador (ej. RandomForestClassifier).
    """
    import pandas as pd
    import re

    features = pd.DataFrame()
    features["title_length"] = df[title_column].apply(len)
    features["uppercase_count"] = df[title_column].apply(lambda x: sum(1 for c in x if c.isupper()))
    features["has_numbers"] = df[title_column].apply(lambda x: int(bool(re.search(r"\d", x))))
    features["has_money_symbols"] = df[title_column].apply(lambda x: int(bool(re.search(r"[$€£]", x))))

    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"
                               "\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF"
                               "\U0001F700-\U0001F77F"
                               "\U0001F780-\U0001F7FF"
                               "\U0001F800-\U0001F8FF"
                               "\U0001F900-\U0001F9FF"
                               "\U0001FA00-\U0001FA6F"
                               "\U0001FA70-\U0001FAFF"
                               "\U00002702-\U000027B0"
                               "\U000024C2-\U0001F251]+", flags=re.UNICODE)
    features["has_emojis"] = df[title_column].apply(lambda x: int(bool(emoji_pattern.search(x))))

    if "language" not in df.columns:
        df["language"] = df[title_column].apply(lambda x: langid.classify(x)[0])

    stop_words_en = set(stopwords.words("english"))
    stop_words_es = set(stopwords.words("spanish"))

    def get_stop_words_by_language(lang):
        if lang == "en":
            return stop_words_en
        elif lang == "es":
            return stop_words_es
        else:
            return set()

    def get_most_common_words_func(title, language, top_n=3):
        sw = get_stop_words_by_language(language)
        words = [w.lower() for w in re.findall(r'\w+', title) if w.lower() not in sw]
        common_ = [w for w, _ in Counter(words).most_common(top_n)]
        return common_

    features["common_words"] = df.apply(
        lambda row: get_most_common_words_func(row[title_column], row["language"]), axis=1
    )

    if most_common_words is None:
        most_common_words = Counter([w for subl in features["common_words"] for w in subl]).most_common(10)

    # Añadir columnas "has_word_x"
    palabras_ = {}
    for i, (word, _) in enumerate(most_common_words):
        colname = f"has_word_{i+1}"
        features[colname] = df[title_column].str.contains(rf"\b{word}\b", case=False).astype(int)
        palabras_[colname] = word

    # Nuevas variables
    def tokenize(t):
        return re.findall(r'\w+', t)

    def count_stopwords_fn(t, lang):
        sw = get_stop_words_by_language(lang)
        return sum(1 for w in tokenize(t) if w.lower() in sw)

    features["word_count"] = df[title_column].apply(lambda x: len(tokenize(x)))

    def avg_word_length(t):
        wds = tokenize(t)
        if not wds: return 0
        return sum(len(w) for w in wds)/len(wds)
    features["avg_word_length"] = df[title_column].apply(avg_word_length)

    def uppercase_ratio(t):
        wds = tokenize(t)
        if not wds: return 0
        up_ = [w for w in wds if w.isupper()]
        return len(up_)/len(wds)
    features["uppercase_ratio"] = df[title_column].apply(uppercase_ratio)

    def first_three_uppercase(t):
        wds = tokenize(t)
        first_three = wds[:3]
        if len(first_three) < 3:
            return 0
        return int(all(w.isupper() for w in first_three))
    features["first_three_uppercase"] = df[title_column].apply(first_three_uppercase)

    features["stopword_count"] = df.apply(
        lambda row: count_stopwords_fn(row[title_column], row["language"]), axis=1
    )

    def stopword_ratio_fn(row):
        if row["word_count"] > 0:
            return row["stopword_count"]/row["word_count"]
        return 0
    features["stopword_ratio"] = features.apply(stopword_ratio_fn, axis=1)

    punct_pattern = re.compile(r'[^\w\s]')
    features["punctuation_count"] = df[title_column].apply(lambda x: len(punct_pattern.findall(x)))

    features["has_question_mark"] = df[title_column].apply(lambda x: int('?' in x))
    features["has_exclamation_mark"] = df[title_column].apply(lambda x: int('!' in x))

    def uppercase_words_count(t):
        wds = tokenize(t)
        return sum(1 for w in wds if w.isupper())
    features["uppercase_words_count"] = df[title_column].apply(uppercase_words_count)

    def lexical_diversity(t):
        wds = [w.lower() for w in tokenize(t)]
        u = set(wds)
        return len(u)/len(wds) if wds else 0
    features["lexical_diversity"] = df[title_column].apply(lexical_diversity)

    def alpha_ratio(t):
        alphas = sum(1 for c in t if c.isalpha())
        total = len(t)
        return alphas/total if total>0 else 0
    features["alpha_ratio"] = df[title_column].apply(alpha_ratio)

    return features, palabras_

def train_model_lg(df, target_column, feature_columns):
    """
    Entrena un RandomForestClassifier y retorna el modelo.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Reporte para idioma '{df['language'].iloc[0]}':")
    print(classification_report(y_test, y_pred))
    return model

# ----------------------------------------------------------------------

def get_best_worst_titles(df_en, df_datos_clusterizados, var_obj):
    """
    Retorna dataframes con títulos 'mejores' y 'peores' según la variable objetivo.
    """

    df_en_resumen = df_en[['title','channel_name',var_obj]].reset_index(drop=True)
    df_en_resumen['cluster_descripcion'] = df_datos_clusterizados['cluster_descripcion'].astype(str)

    pt = pd.pivot_table(df_en_resumen, aggfunc={'target':['mean','sum','count']},
                        index='cluster_descripcion')
    pt = pt['target'].reset_index()

    pt.sort_values(by=['mean','count'], ascending=False, inplace=True)

    mejores_titulos = pt[(pt['mean']>0.6)&(pt['count']>4)]
    peores_titulos = pt[(pt['mean']<0.3)&(pt['count']>4)]

    df_mejores_titulos = df_en_resumen[df_en_resumen['cluster_descripcion'].isin(mejores_titulos['cluster_descripcion'])]
    df_peores_titulos = df_en_resumen[df_en_resumen['cluster_descripcion'].isin(peores_titulos['cluster_descripcion'])]

    return (df_mejores_titulos, df_peores_titulos, mejores_titulos, peores_titulos)

# ----------------------------------------------------------------------

def get_dict_descripciones(adfuhie_dafs, palabras_top, feature_columns, df_datos_clusterizados, decimals=2):
    """
    Crea un mapeo de descripciones 'crudas' a descripciones 'binarizadas' y formateadas.
    Reemplaza las condiciones y columnas binarias con algo más limpio.
    """

    # 1) Limpieza de floats
    descripciones_limpias = limpiar_descripciones(adfuhie_dafs, decimals=decimals)

    # 2) Determinar qué columnas son binarias
    cuenta_valores = {}
    for f in feature_columns:
        # Chequeo de si los valores son un set {0,1} => True si binario
        vals = df_datos_clusterizados[f].dropna().unique()
        cuenta_valores[f] = (set(vals) <= {0,1})

    # 3) Binarizar
    descripciones_binarizadas = binarizar_columnas(descripciones_limpias, cuenta_valores)

    # 4) Reemplazar "has_word_x" => "has_word_x: <palabra>"
    dict_has_word = {k: f"{k}: {v}" for k,v in palabras_top.items()}
    descripciones_binarizadas = [replace_words(x, dict_has_word) for x in descripciones_binarizadas]

    # 5) Formato final (global)
    resultado_3col = formatear_descripciones_binarizadas_global(descripciones_binarizadas, n_cols=5)
    return dict(zip(adfuhie_dafs, resultado_3col))

# ----------------------------------------------------------------------
# Subfunciones "privadas":
# ----------------------------------------------------------------------

def limpiar_descripciones(descriptions, decimals=2):
    pattern = re.compile(r'(?<![0-9])-?\d+(\.\d+)?(?![0-9])')
    cleaned = []
    for desc in descriptions:
        def replacer(m):
            numero_str = m.group(0)
            numero_f = float(numero_str)
            if numero_f.is_integer():
                return str(int(numero_f))
            else:
                return f"{round(numero_f, decimals):.{decimals}f}"
        cleaned.append(pattern.sub(replacer, desc))
    return cleaned

def binarizar_columnas(descriptions, cuenta_valores):
    nuevas_desc = []
    for desc in descriptions:
        condiciones = desc.split(" AND ")
        nuevas_cond = []
        for cond in condiciones:
            partes = [p.strip() for p in cond.split("<=")]
            if len(partes) == 3:
                x_str, col, y_str = partes
                try: x_val = float(x_str)
                except: x_val = None
                try: y_val = float(y_str)
                except: y_val = None

                if col in cuenta_valores and cuenta_valores[col]:
                    if (y_val is not None and y_val <= 0.5):
                        nuevas_cond.append(f"{col} = 0")
                    elif (x_val is not None and x_val >= 0.5):
                        nuevas_cond.append(f"{col} = 1")
                    else:
                        nuevas_cond.append(cond)
                else:
                    nuevas_cond.append(cond)
            else:
                nuevas_cond.append(cond)
        nuevas_desc.append(" AND ".join(nuevas_cond))
    return nuevas_desc

def parse_condition(cond):
    cond = cond.strip()
    patron_binario = re.compile(r'^([\w\d_]+)\s*=\s*([\-\d\.]+)$')
    m_bin = patron_binario.match(cond)
    if m_bin:
        col = m_bin.group(1)
        val = m_bin.group(2)
        return col, val

    patron_rango = re.compile(r'^([\-\d\.]+)\s*<=\s*([\w\d_]+)\s*<=\s*([\-\d\.]+)$')
    m_ran = patron_rango.match(cond)
    if m_ran:
        x_val = m_ran.group(1)
        col = m_ran.group(2)
        y_val = m_ran.group(3)
        return col, f"[{x_val}, {y_val}]"
    return cond, ""

def preparar_bloques(descriptions, n_cols=2):
    all_desc = []
    for desc in descriptions:
        conds = desc.split(" AND ")
        parsed = []
        for c in conds:
            col, val = parse_condition(c)
            if val == "":
                parsed.append(col)
            else:
                parsed.append(f"{col}: {val}")
        filas = []
        for i in range(0,len(parsed),n_cols):
            filas.append(parsed[i:i+n_cols])
        all_desc.append(filas)
    return all_desc

def calcular_anchos_global(all_desc, n_cols=2):
    col_widths = [0]*n_cols
    for desc in all_desc:
        for fila in desc:
            for idx, item in enumerate(fila):
                length_item = len(item)
                if length_item>col_widths[idx]:
                    col_widths[idx] = length_item
    return col_widths

def formatear_descripciones_binarizadas_global(descriptions, n_cols=2):
    all_desc = preparar_bloques(descriptions, n_cols=n_cols)
    col_widths = calcular_anchos_global(all_desc, n_cols=n_cols)
    total_desc = len(all_desc)
    digitos = len(str(total_desc))
    bloques = []

    for i, desc in enumerate(all_desc):
        prefijo_num = f"{(i+1):>{digitos}}. "
        lineas = []
        for row_idx, fila in enumerate(desc):
            row_parts = []
            for col_idx, item in enumerate(fila):
                ancho = col_widths[col_idx]
                row_parts.append(f"{item:<{ancho}}")
            row_str = "   ".join(row_parts)
            if row_idx < len(desc)-1:
                row_str += " &"
            if row_idx==0:
                final_line = prefijo_num + row_str
            else:
                final_line = " "*len(prefijo_num) + row_str
            lineas.append(final_line)
        bloques.append("\n".join(lineas))

    return bloques

def replace_words(text, replacement_dict):
    for original, replacement in replacement_dict.items():
        text = text.replace(original, replacement)
    return text
