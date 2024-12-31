# # model_manager.py

# import joblib
# import os
# import logging
# from sklearn.ensemble import RandomForestClassifier
# from skl2onnx import convert_sklearn, load_model as load_onnx_model
# from skl2onnx.common.data_types import FloatTensorType
# import onnxruntime as rt
# import numpy as np

# # Configuración del logging
# logging.basicConfig(
#     level=logging.INFO,  # Nivel de severidad
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Formato del mensaje
#     handlers=[
#         logging.FileHandler("model_management.log"),  # Archivo donde se guardarán los logs
#         logging.StreamHandler()  # También se mostrarán en la consola
#     ]
# )

# logger = logging.getLogger(__name__)

# def guardar_modelo(model, nombre_modelo, ruta_modelo, ruta_onnx):
#     """
#     Guarda el modelo en formato joblib y ONNX.

#     Parámetros:
#     - model: Instancia de RandomForestClassifier.
#     - nombre_modelo: Nombre base del modelo para los archivos.
#     - ruta_modelo: Ruta donde se guardará el archivo joblib.
#     - ruta_onnx: Ruta donde se guardará el archivo ONNX.
#     """
#     try:
#         # Guardar con joblib
#         joblib.dump(model, ruta_modelo)
#         logger.info(f"Modelo '{nombre_modelo}' guardado exitosamente en '{ruta_modelo}'.")

#         # Convertir a ONNX
#         initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
#         onnx_model = convert_sklearn(model, initial_types=initial_type)

#         # Guardar el modelo ONNX
#         with open(ruta_onnx, "wb") as f:
#             f.write(onnx_model.SerializeToString())
#         logger.info(f"Modelo '{nombre_modelo}' convertido y guardado en formato ONNX en '{ruta_onnx}'.")

#     except Exception as e:
#         logger.error(f"Error al guardar el modelo '{nombre_modelo}': {e}")

# def cargar_modelo(nombre_modelo, ruta_modelo_joblib, ruta_modelo_onnx):
#     """
#     Carga el modelo desde joblib o, si no es RandomForestClassifier, desde ONNX.

#     Parámetros:
#     - nombre_modelo: Nombre base del modelo.
#     - ruta_modelo_joblib: Ruta del archivo joblib.
#     - ruta_modelo_onnx: Ruta del archivo ONNX.

#     Retorna:
#     - model: Instancia de RandomForestClassifier o una sesión de ONNXRuntime.
#     """
#     model = None
#     try:
#         # Intentar cargar con joblib
#         model = joblib.load(ruta_modelo_joblib)
#         logger.info(f"Modelo '{nombre_modelo}' cargado exitosamente desde '{ruta_modelo_joblib}'.")

#         # Verificar el tipo
#         if isinstance(model, RandomForestClassifier):
#             logger.info(f"Modelo '{nombre_modelo}' es una instancia de RandomForestClassifier.")
#             return model
#         else:
#             logger.warning(f"Modelo '{nombre_modelo}' no es RandomForestClassifier (tipo: {type(model)}). Intentando cargar desde ONNX.")
#             model = None  # Resetear model para intentar cargar desde ONNX

#     except FileNotFoundError:
#         logger.error(f"Archivo joblib '{ruta_modelo_joblib}' no encontrado.")
#     except Exception as e:
#         logger.error(f"Error al cargar con joblib el modelo '{nombre_modelo}': {e}")

#     # Intentar cargar desde ONNX si joblib falló o no es RF
#     if not model:
#         try:
#             if not os.path.isfile(ruta_modelo_onnx):
#                 logger.error(f"Archivo ONNX '{ruta_modelo_onnx}' no existe.")
#                 return None

#             onnx_model = load_onnx_model(ruta_modelo_onnx)
#             logger.info(f"Modelo '{nombre_modelo}' cargado exitosamente desde '{ruta_modelo_onnx}' en formato ONNX.")

#             # Configurar sesión de ONNX Runtime
#             sess = rt.InferenceSession(ruta_modelo_onnx)
#             model = sess
#             logger.info(f"Sesión de ONNX Runtime para '{nombre_modelo}' creada exitosamente.")
#             return model
#         except Exception as e:
#             logger.error(f"Error al cargar el modelo ONNX '{ruta_modelo_onnx}': {e}")
#             return None

# def predecir_con_onnx(model_onnx, datos):
#     """
#     Realiza predicciones usando un modelo ONNX.

#     Parámetros:
#     - model_onnx: Sesión de ONNX Runtime.
#     - datos: Datos de entrada como numpy array.

#     Retorna:
#     - Predicciones.
#     """
#     try:
#         input_name = model_onnx.get_inputs()[0].name
#         label_name = model_onnx.get_outputs()[0].name
#         predicciones = model_onnx.run([label_name], {input_name: datos.astype(np.float32)})[0]
#         return predicciones
#     except Exception as e:
#         logger.error(f"Error al realizar predicciones con ONNX: {e}")
#         return None

# def cargar_todos_los_modelos():
#     """
#     Carga todos los modelos guardados en 'models/'.

#     Retorna:
#     - Diccionario con modelos cargados para videos y shorts en ambos idiomas.
#     """
#     modelos = {}
#     model_paths = {
#         "model_videos_es": ("models/model_videos_es.joblib", "models/model_videos_es.onnx"),
#         "model_videos_en": ("models/model_videos_en.joblib", "models/model_videos_en.onnx"),
#         "model_shorts_es": ("models/model_shorts_es.joblib", "models/model_shorts_es.onnx"),
#         "model_shorts_en": ("models/model_shorts_en.joblib", "models/model_shorts_en.onnx"),
#     }

#     for nombre, (ruta_joblib, ruta_onnx) in model_paths.items():
#         model = cargar_modelo(nombre, ruta_joblib, ruta_onnx)
#         if model:
#             modelos[nombre] = model
#         else:
#             logger.error(f"No se pudo cargar el modelo '{nombre}' desde ninguna de las fuentes.")

#     return modelos
