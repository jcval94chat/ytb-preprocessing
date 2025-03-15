# utils/google_utils.py

import logging
import time
import pandas as pd
import numpy as np
import gspread
import traceback
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def authenticate_google_services(creds_file):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    return credentials

def parse_timestamp_from_name(name):
    try:
        timestamp_str = name.split("(Copia")[-1].strip(" )")
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M-%S")
    except:
        return None

def get_sheets_data_from_folder(folder_id, creds_file, days=30, max_files=60, sleep_seconds=2):
    """
    Obtiene datos filtrados por fecha y limita el número de archivos
    a leer en una carpeta de Google Drive (cada archivo es una Google Sheet).
    """
    credentials = authenticate_google_services(creds_file)
    drive_service = build("drive", "v3", credentials=credentials)

    logger.info(f"Buscando archivos en folder_id={folder_id} ...")
    query = f"'{folder_id}' in parents"
    results = drive_service.files().list(q=query).execute()
    files = results.get('files', [])
    if not files:
        logger.warning("No se encontraron archivos en la carpeta de Drive.")
        return None

    import gspread
    client = gspread.authorize(credentials)

    file_timestamps = [(f, parse_timestamp_from_name(f['name'])) for f in files]
    file_timestamps = [x for x in file_timestamps if x[1] is not None]
    if not file_timestamps:
        logger.warning("No se encontraron archivos con timestamp. Se procederá sin filtrar por fecha.")
        filtered_files = files[:max_files]
    else:
        max_ts = max(ts for _, ts in file_timestamps if ts is not None)
        cutoff_date = max_ts - timedelta(days=days)
        file_timestamps = [(f, ts) for (f, ts) in file_timestamps if ts >= cutoff_date]

        processed_timestamps = []
        filtered_files = []
        for f, ts in sorted(file_timestamps, key=lambda x: x[1], reverse=True):
            if all(abs((ts - t).total_seconds())>1800 for t in processed_timestamps):
                filtered_files.append(f)
                processed_timestamps.append(ts)
            if len(filtered_files)>=max_files:
                break

    dataframes = []
    for i, file in enumerate(filtered_files):
        if i>0:
            time.sleep(sleep_seconds)
        try:
            sheet = client.open_by_key(file['id'])
            worksheet = sheet.get_worksheet(0)
            data = worksheet.get_all_values()
            df = pd.DataFrame(data[1:], columns=data[0])
            dataframes.append(df)
            logger.info(f"Leído archivo: {file['name']} con {df.shape[0]} filas.")
        except Exception as e:
            logger.error(f"Error leyendo {file['name']}: {str(e)}")

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        logger.warning("No se pudieron leer archivos o no hay datos.")
        return None

# La función upload_dataframe_to_google_sheet iría aquí también
# (omitida en este snippet para brevedad).

def sanitize_dataframe(df):
    """
    Reemplaza valores no JSON-compliant en el DataFrame:
    - Reemplaza inf, -inf y NaN con None.
    - Convierte datetime a cadenas.
    """
    df_replaced = df.replace([np.inf, -np.inf, np.nan], None)
    for col in df_replaced.select_dtypes(include=['datetime', 'datetime64[ns]']).columns:
        df_replaced[col] = df_replaced[col].astype(str)

    # Verificar si quedan valores problemáticos:
    # (Opcional, se podría omitir si no te hace falta la comprobación extra)
    return df_replaced

def upload_dataframe_to_google_sheet(df, creds_file, spreadsheet_id, sheet_name='Sheet1'):
    """
    Sube un DataFrame de pandas a una hoja de cálculo de Google Sheets.
    """
    try:
        df_sanitized = sanitize_dataframe(df)

        # Información de los primeros 3 registros
        preview_data = df_sanitized.head(3)
        logging.info(f"Primeros 3 registros del DataFrame:\n{preview_data}")
        
        credentials = authenticate_google_services(creds_file)
        client = gspread.authorize(credentials)

        spreadsheet = client.open_by_key(spreadsheet_id)

        # Selecciona worksheet
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")

        sheet.clear()

        data = [df_sanitized.columns.values.tolist()] + df_sanitized.values.tolist()
        sheet.update(data)

        logging.info(f"Datos subidos correctamente a '{sheet_name}' en la hoja '{spreadsheet_id}'.")
        return True

    except Exception as e:
        logging.error(f"Error al subir DataFrame a Google Sheets: {str(e)}")
        logging.error(traceback.format_exc())
        return False


