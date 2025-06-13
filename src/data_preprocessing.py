import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path # Asegurarse que Path está importado para las nuevas funciones

# Importaciones de módulos hermanos (dentro de src)
from src.utils import strip_accents # Asumiendo que esta función está en utils.py
from src.config import (
    SYMPTOM_SPLIT_PATTERN,
    SYMPTOM_EXTRACTION_PATTERN,
    DURATION_EXTRACTION_PATTERN,
    INTENSITY_EXTRACTION_PATTERN,
    ADJECTIVE_CLEANUP_PATTERN,
    INTENSITY_SCALE_MAP,
    SYMPTOM_GROUPS_MAP
    # Nota: Si config.py también define IMAGE_EXTENSIONS y las rutas de carpetas de imágenes,
    # podrían importarse aquí si se usaran directamente, pero es mejor pasarlas como argumentos.
)
# ─────────────────────── Preprocesador tabular (ColumnTransformer) ───────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler,
    OneHotEncoder, LabelEncoder,
)

def build_clinical_preprocessor(
    num_feats: list[str],
    sym_feats: list[str],
    cat_feats: list[str],
    scaler: str = "minmax",
) -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer que escala numéricas y OHE-iza categóricas.

    Parameters
    ----------
    num_feats  : columnas numéricas "básicas"  (age, duration_days, …)
    sym_feats  : columnas de síntomas          (prefijadas con 'sym_')
    cat_feats  : columnas categóricas para OHE (p.e. 'condition_code')
    scaler     : 'minmax' | 'standard'
    """
    scaler_cls = MinMaxScaler if scaler == "minmax" else StandardScaler
    return ColumnTransformer(
        [
            ("num", scaler_cls(), num_feats + sym_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), cat_feats),
        ],
        remainder="drop",
    )

def label_encode_column(
    df: pd.DataFrame,
    col: str,
    ordered: bool = True,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Codifica una columna categórica y añade una columna '{col}_code'.

    Devuelve el DataFrame (misma referencia) y el LabelEncoder.
    """
    le = LabelEncoder().fit(sorted(df[col].unique()) if ordered else df[col])
    df[f"{col}_code"] = le.transform(df[col])
    return df, le

# --- Funciones de Carga y Preprocesamiento Inicial de Datos Clínicos ---

def load_clinical_data(file_path: str, separator: str = ';') -> Optional[pd.DataFrame]:
    """
    Carga un conjunto de datos clínicos desde un archivo CSV.
    Maneja errores comunes como archivo no encontrado.
    Retorna un DataFrame de Pandas si la carga es exitosa, de lo contrario None.
    """
    try:
        df = pd.read_csv(file_path, sep=separator)
        print(f"Dataset cargado exitosamente desde: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {file_path}")
    except Exception as e:
        print(f"Ocurrió un error al cargar el dataset desde {file_path}: {e}")
    return None

# --- Funciones Auxiliares para el Parseo de Notas Clínicas (uso interno) ---

def _canonical_symptom(raw_symptom: str) -> str:
    """
    Normaliza una cadena de síntoma crudo.
    Convierte a minúsculas, elimina acentos, adjetivos calificativos iniciales
    y plurales comunes (ej. 'headaches' -> 'headach', 'seizures' -> 'seizure').
    Esta es una función interna (prefijo '_').
    """
    # Elimina acentos, convierte a minúsculas y quita espacios extra
    text = strip_accents(raw_symptom).lower().strip()
    # Elimina adjetivos comunes del inicio del síntoma
    text = re.sub(ADJECTIVE_CLEANUP_PATTERN, '', text, flags=re.IGNORECASE).strip()
    # Normaliza plurales comunes a singular (casos específicos)
    text = re.sub(r'ches$', 'ch', text)  # Ej: headaches -> headach
    text = re.sub(r's$', '', text)       # Ej: seizures -> seizure
    return text

def _duration_to_days(duration_text: str) -> Optional[int]:
    """
    Convierte una descripción textual de duración (ej. "a few weeks", "2 months")
    en un número aproximado de días.
    Maneja varias frases comunes y patrones numéricos. Retorna None si no se puede convertir.
    Esta es una función interna (prefijo '_').
    """
    text = duration_text.lower() # Normaliza a minúsculas para la comparación
    # Mapeo de frases comunes a días
    if 'few week' in text: return 21         # "unas pocas semanas"
    if 'several week' in text: return 35     # "varias semanas"
    if 'past month' in text or 'last month' in text: return 30 # "mes pasado"
    if any(p in text for p in ['few month', 'last few month', 'past few month']): return 90 # "unos pocos meses"
    if 'several month' in text: return 120    # "varios meses"
    if 'more than 6 month' in text: return 210 # "más de 6 meses"
    if any(p in text for p in ['past year', 'last year', 'this year']): return 365 # "año pasado/este año"
    
    # Extracción de duraciones numéricas (semanas, meses, años)
    match_weeks = re.search(r'(\d+)\s*week', text)
    if match_weeks: return int(match_weeks.group(1)) * 7
    
    match_months = re.search(r'(\d+)\s*month', text)
    if match_months: return int(match_months.group(1)) * 30
    
    match_years = re.search(r'(\d+)\s*year', text)
    if match_years: return int(match_years.group(1)) * 365
    
    # Extracción de rangos (ej. "2-3 semanas") - toma el promedio
    match_range_weeks = re.search(r'(\d+)\s*-\s*(\d+)\s*week', text)
    if match_range_weeks:
        return (int(match_range_weeks.group(1)) + int(match_range_weeks.group(2))) // 2 * 7
    
    return None # Si no se encuentra un patrón reconocible

def _parse_single_clinical_note(note: str) -> Dict[str, any]:
    """
    Parsea una única nota clínica (cadena de texto) para extraer:
    - Lista de síntomas canónicos (únicos y en orden de aparición).
    - Nivel de intensidad global de los síntomas.
    - Duración aproximada de los síntomas en días.
    Retorna un diccionario con estas tres claves. Maneja notas no válidas (NaN).
    Esta es una función interna (prefijo '_').
    """
    # Manejo de entradas no válidas (ej. NaN)
    if not isinstance(note, str):
        return {'symptoms_list': [], 'intensity_lvl': np.nan, 'duration_days': np.nan}

    lower_note = note.lower() # Normaliza a minúsculas
    
    # Búsqueda de patrones de síntomas, intensidad y duración usando regex
    match_symptoms = re.search(SYMPTOM_EXTRACTION_PATTERN, lower_note, re.IGNORECASE)
    match_intensity = re.search(INTENSITY_EXTRACTION_PATTERN, lower_note, re.IGNORECASE)
    match_duration = re.search(DURATION_EXTRACTION_PATTERN, lower_note, re.IGNORECASE)

    # Extracción y procesamiento de síntomas
    raw_symptoms_text = match_symptoms.group(1) if match_symptoms else ""
    symptom_list_raw = re.split(SYMPTOM_SPLIT_PATTERN, raw_symptoms_text) # Divide por comas, "and", etc.
    
    processed_symptoms = [
        _canonical_symptom(s) for s in symptom_list_raw if s and s.strip() # Normaliza cada síntoma
    ]
    unique_symptoms = list(dict.fromkeys(processed_symptoms)) # Mantiene orden y elimina duplicados

    # Extracción y mapeo de intensidad
    intensity_raw_text = match_intensity.group(1).strip().lower() if match_intensity else None
    intensity_level = INTENSITY_SCALE_MAP.get(intensity_raw_text, np.nan) # Mapea a escala numérica
    
    # Extracción y conversión de duración
    duration_raw_text = match_duration.group(1) if match_duration else ""
    duration_in_days = _duration_to_days(duration_raw_text) if duration_raw_text else np.nan
    
    return {
        'symptoms_list': unique_symptoms,
        'intensity_lvl': intensity_level,
        'duration_days': duration_in_days
    }

# --- Funciones Principales de Preprocesamiento de Notas Clínicas y Síntomas ---

def process_clinical_notes(df: pd.DataFrame, note_column: str = 'Clinical Note') -> Tuple[pd.DataFrame, List[str]]:
    """
    Procesa la columna de notas clínicas de un DataFrame.
    Aplica la función `_parse_single_clinical_note` a cada nota para extraer
    síntomas, intensidad y duración.
    Luego, crea nuevas columnas binarias/de intensidad para cada síntoma único encontrado.
    Retorna el DataFrame modificado y una lista de los nombres de las nuevas columnas de síntomas.
    """
    # Validación de la columna de notas
    if note_column not in df.columns:
        raise ValueError(f"Columna '{note_column}' no encontrada en el DataFrame.")

    # Aplica el parseo a cada nota y expande los resultados en nuevas columnas del DataFrame
    parsed_notes_df = df[note_column].apply(_parse_single_clinical_note).apply(pd.Series)
    
    # Concatena las nuevas columnas (symptoms_list, intensity_lvl, duration_days) al DataFrame original
    df_with_parsed_info = pd.concat([df, parsed_notes_df], axis=1)

    # Identifica todos los síntomas únicos extraídos en todo el dataset
    all_extracted_symptoms = sorted(
        {s for sublist in df_with_parsed_info['symptoms_list'].dropna() for s in sublist}
    )
    
    # Crea nombres para las nuevas columnas de síntomas (ej. 'sym_headache')
    symptom_feature_cols = [f"sym_{s}" for s in all_extracted_symptoms]

    # Inicializa una matriz para las intensidades de los síntomas (0 por defecto)
    symptom_intensity_matrix = pd.DataFrame(0, index=df.index, columns=symptom_feature_cols, dtype='float32')

    # Asigna la intensidad extraída a las columnas de síntomas correspondientes para cada paciente
    for idx, row in df_with_parsed_info.iterrows():
        symptoms = row['symptoms_list']
        level = row['intensity_lvl']
        if isinstance(symptoms, list) and not pd.isna(level): # Si hay síntomas y nivel de intensidad
            for s_name in symptoms:
                col_name = f"sym_{s_name}"
                if col_name in symptom_intensity_matrix.columns: # Verifica que el síntoma esté en las columnas creadas
                    symptom_intensity_matrix.loc[idx, col_name] = level
    
    # Concatena la matriz de intensidades de síntomas al DataFrame
    df_final = pd.concat([df_with_parsed_info, symptom_intensity_matrix], axis=1)
    
    # Opcionalmente, se podrían eliminar columnas intermedias como 'symptoms_list', 'intensity_lvl' aquí
    # si ya no son necesarias, pero se mantienen por ahora para inspección.
    # Ejemplo: df_final = df_final.drop(columns=['symptoms_list', 'intensity_lvl'])
    
    return df_final, symptom_feature_cols # Retorna el DataFrame procesado y la lista de columnas de síntomas


def group_symptoms_by_mapping(df: pd.DataFrame, symptom_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Agrupa columnas de síntomas basándose en un mapeo predefinido (SYMPTOM_GROUPS_MAP de config).
    Para cada grupo, crea una nueva columna de síntoma agrupado tomando la intensidad máxima
    de sus componentes. Elimina las columnas originales que fueron agrupadas.
    Retorna el DataFrame con los síntomas agrupados y la lista actualizada de columnas de síntomas.
    """
    df_processed = df.copy() # Trabaja sobre una copia para evitar modificar el original directamente en esta etapa
    original_sym_cols_to_potentially_drop = [] # Lista para rastrear columnas originales que se agrupan
    created_group_cols_names = [] # Lista para rastrear los nombres de las nuevas columnas de grupo

    # Itera sobre el mapeo de grupos de síntomas
    for group_name_base, original_symptoms_bases in SYMPTOM_GROUPS_MAP.items():
        new_group_col_name = f"sym_{group_name_base}" # Nombre de la nueva columna agrupada
        created_group_cols_names.append(new_group_col_name)
        
        # Identifica las columnas existentes en el DataFrame que pertenecen a este grupo
        cols_to_group_full_names = [
            f"sym_{s_base}" for s_base in original_symptoms_bases 
            if f"sym_{s_base}" in df_processed.columns
        ]
        
        if not cols_to_group_full_names: # Si no hay columnas para este grupo, advertir y continuar
            print(f"Advertencia: No se encontraron columnas para el grupo '{group_name_base}'. Saltando.")
            continue
        
        # Agrega tomando la intensidad máxima de los síntomas componentes
        df_processed[new_group_col_name] = df_processed[cols_to_group_full_names].max(axis=1)
        original_sym_cols_to_potentially_drop.extend(cols_to_group_full_names) # Añade a la lista de posibles eliminaciones

    # Determina qué columnas originales eliminar
    final_cols_to_drop = []
    unique_original_cols_to_consider_dropping = list(set(original_sym_cols_to_potentially_drop))

    for col_to_drop in unique_original_cols_to_consider_dropping:
        # Solo elimina una columna original si no es ella misma el nombre de una nueva columna de grupo
        # (esto maneja casos donde una columna original se convierte en la columna agrupada, ej. 'sym_seizure')
        if col_to_drop not in created_group_cols_names:
            final_cols_to_drop.append(col_to_drop)

    # Elimina las columnas originales que fueron efectivamente agrupadas y no son columnas de grupo
    cols_actually_dropped_from_df = [col for col in final_cols_to_drop if col in df_processed.columns]
    if cols_actually_dropped_from_df:
        df_processed.drop(columns=cols_actually_dropped_from_df, inplace=True)
        print(f"Columnas originales agrupadas y eliminadas: {cols_actually_dropped_from_df}")

    # Actualiza la lista final de columnas de síntomas (incluye agrupadas y no agrupadas restantes)
    current_sym_cols_in_df = [col for col in df_processed.columns if col.startswith('sym_')]
    final_symptom_cols = sorted(list(set(current_sym_cols_in_df))) # Ordena alfabéticamente
    
    return df_processed, final_symptom_cols

# --- Funciones para Enriquecimiento y Guardado de Datos ---

def build_case_id_to_image_map(
    image_folders_map: Dict[str, Path], # Mapeo de nombre de carpeta de tumor a su ruta Path
    image_extensions: List[str],       # Lista de extensiones de imagen a buscar (ej. ['*.jpg'])
    get_image_paths_func               # Función para obtener rutas de imágenes (ej. utils.get_image_paths)
) -> Dict[str, str]:
    """
    Construye un diccionario que mapea un 'Case ID' (extraído del nombre del archivo de imagen)
    a la ruta absoluta de la imagen correspondiente.
    Filtra imágenes que contengan la palabra "copy" en su nombre (insensible a mayúsculas).
    Toma la primera imagen válida encontrada por 'Case ID' si hay múltiples.
    """
    print("\n--- Construyendo mapeo de Case ID a rutas de imágenes ---")
    case_id_to_image_path = {} # Diccionario para almacenar el mapeo
    total_images_found = 0     # Contador total de archivos de imagen encontrados
    images_skipped_copy = 0    # Contador de imágenes omitidas por contener "copy"

    # Itera sobre cada tipo de tumor y su carpeta de imágenes correspondiente
    for _tumor_type_folder_name, folder_path in image_folders_map.items():
        if folder_path.exists(): # Verifica que la carpeta exista
            # Obtiene todas las rutas de imágenes en la carpeta actual
            image_paths_current_folder = get_image_paths_func(folder_path, image_extensions)
            for img_path in image_paths_current_folder:
                total_images_found += 1
                # Omite archivos si "copy" está en el nombre (prevención de duplicados de data augmentation)
                if "copy" in img_path.name.lower():
                    images_skipped_copy += 1
                    continue
                
                # Extrae el 'Case ID' del nombre del archivo (sin la extensión)
                case_id_from_img = img_path.stem 
                # Si este 'Case ID' no ha sido mapeado aún, añade la ruta de la imagen
                if case_id_from_img not in case_id_to_image_path:
                    case_id_to_image_path[case_id_from_img] = str(img_path.resolve()) # Guarda la ruta absoluta
        else:
            print(f"Advertencia: La carpeta de imágenes {folder_path} no existe y será omitida.")

    print(f"Total de archivos de imagen escaneados: {total_images_found}")
    print(f"Imágenes omitidas por contener 'copy' en el nombre: {images_skipped_copy}")
    print(f"Número de Case IDs únicos con una ruta de imagen asignada: {len(case_id_to_image_path)}")
    return case_id_to_image_path

def save_processed_dataframe(df: pd.DataFrame, output_dir: Path, filename: str):
    """
    Guarda el DataFrame proporcionado en un archivo CSV en el directorio de salida especificado.
    Crea el directorio de salida si no existe.
    Muestra las primeras filas del DataFrame guardado.
    """
    print("\n--- Guardando DataFrame procesado ---")
    # Asegura que el directorio de salida exista, creándolo si es necesario
    output_dir.mkdir(parents=True, exist_ok=True)
    # Construye la ruta completa del archivo de salida
    processed_df_path = output_dir / filename
    # Guarda el DataFrame en formato CSV sin el índice
    df.to_csv(processed_df_path, index=False)
    print(f"\nDataFrame procesado guardado en: {processed_df_path}")
    display(df.head()) # Muestra las primeras filas (útil en notebooks)