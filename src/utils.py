import pandas as pd
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional # Asegurarse que Optional está aquí
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations

# --- Funciones de Utilidad General y Preprocesamiento de DataFrames ---

def strip_accents(text: str) -> str:
    """
    Elimina acentos y normaliza texto a caracteres ASCII.
    Retorna el texto sin acentos si es una cadena, de lo contrario, el texto original.
    """
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def get_image_paths(folder_path: Path, extensions: List[str]) -> List[Path]:
    """
    Recopila todas las rutas de archivos de imagen de una carpeta específica
    que coincidan con las extensiones proporcionadas.
    Retorna una lista de objetos Path correspondientes a las imágenes encontradas.
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(folder_path.glob(ext)))
    return image_paths

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los nombres de las columnas a minúsculas y reemplaza espacios con guiones bajos.
    Devuelve el DataFrame con los nombres de columna modificados (modifica el DataFrame original).
    """
    # Modifica el DataFrame original directamente
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    print("\n--- Renombrando columnas para estandarización ---")
    print(f"Columnas renombradas.")
    return df # Retorna el DataFrame modificado

def reorder_dataframe_columns(
    df: pd.DataFrame,
    id_col_name: str = 'case_id',
    target_col_names: List[str] = ['condition', 'treatment'],
    symptom_prefix: str = 'sym_',
    clinical_note_col_name: str = 'clinical_note' # Permitir que no exista
) -> pd.DataFrame:
    """
    Reordena las columnas del DataFrame según una lógica predefinida:
    ID, Otras Features (alfabéticamente), Síntomas (alfabéticamente), 
    [Nota Clínica si existe], Targets (en el orden proporcionado).
    Retorna un nuevo DataFrame con las columnas reordenadas o el original si hay errores.
    """
    print("\n--- Reordenando columnas ---")
    current_cols = df.columns.tolist()

    # Verificar que la columna de ID exista
    if id_col_name not in current_cols:
        print(f"Advertencia: Columna de ID ('{id_col_name}') no encontrada. No se aplicará reordenamiento basado en ID.")
        # Podríamos decidir devolver el df original o continuar sin la columna ID en el nuevo orden.
        # Por ahora, se intentará reordenar el resto.
        id_cols = []
    else:
        id_cols = [id_col_name]
        
    # Verificar columnas objetivo
    existing_target_cols = [col for col in target_col_names if col in current_cols]
    if len(existing_target_cols) != len(target_col_names):
        missing_targets = set(target_col_names) - set(existing_target_cols)
        print(f"Advertencia: No todas las columnas objetivo ({target_col_names}) fueron encontradas. Faltan: {missing_targets}. Se reordenarán las presentes.")
    
    target_cols_to_move = existing_target_cols # Solo las que existen

    symptom_cols = sorted([col for col in current_cols if col.startswith(symptom_prefix)])
    
    # Columnas de nota clínica (si existe)
    note_col_list = []
    if clinical_note_col_name in current_cols:
        note_col_list = [clinical_note_col_name]

    # Columnas que no son ID, ni target, ni síntomas, ni la nota clínica
    other_feature_cols = sorted([
        col for col in current_cols 
        if col not in id_cols + target_cols_to_move + symptom_cols + note_col_list
    ])
    
    # Construye el nuevo orden
    new_order = id_cols + other_feature_cols + symptom_cols + note_col_list + target_cols_to_move
                
    # Validar que todas las columnas originales están en el nuevo orden y no hay duplicados
    if len(set(new_order)) == len(current_cols) and all(col in new_order for col in current_cols):
        df_reordered = df[new_order]
        print(f"Columnas reordenadas. Nuevo orden: {df_reordered.columns.tolist()}")
        return df_reordered
    else:
        print("Error: El reordenamiento de columnas no pudo ser validado (inconsistencia o pérdida de columnas). Se devuelve el DataFrame original.")
        print(f"Orden propuesto: {new_order}")
        print(f"Columnas actuales: {current_cols}")
        return df

# --- Funciones para Análisis Exploratorio de Datos (EDA) ---

def display_counts_and_percentages(df: pd.DataFrame, column_name: str, dropna: bool = False) -> pd.DataFrame:
    """
    Calcula y retorna un DataFrame con los conteos y porcentajes de valores únicos
    para una columna categórica especificada de un DataFrame.
    La visualización del DataFrame resultante se realiza en el notebook.
    """
    counts = df[column_name].value_counts(dropna=dropna)
    percentages = (counts / len(df) * 100).round(2)
    summary_df = pd.DataFrame({
        'Frecuencia': counts,
        'Porcentaje (%)': percentages
    })
    return summary_df

def plot_categorical_distribution(df: pd.DataFrame, column_name: str, palette: str = 'pastel', **kwargs):
    """
    Genera y muestra un gráfico de barras (countplot) para visualizar la distribución
    de una variable categórica.
    Permite personalizar título, etiquetas, tamaño y rotación de ticks.
    """
    plt.figure(figsize=kwargs.get('figsize', (7, 4)))
    order = kwargs.get('order', df[column_name].value_counts().index)
    sns.countplot(data=df, x=column_name, palette=palette, order=order)
    plt.title(kwargs.get('title', f'Distribución de {column_name}'), fontsize=12)
    plt.xlabel(kwargs.get('xlabel', column_name), fontsize=10)
    plt.ylabel(kwargs.get('ylabel', 'Frecuencia'), fontsize=10)
    if kwargs.get('xtick_rotation'):
        plt.xticks(rotation=kwargs.get('xtick_rotation'), ha='right')
    plt.tight_layout()
    plt.show()

def plot_numerical_distribution(df: pd.DataFrame, column_name: str, bins: int = 20, **kwargs):
    """
    Genera y muestra un histograma (con KDE) y un boxplot para visualizar
    la distribución de una variable numérica.
    Omite valores nulos en los gráficos e informa si existen.
    """
    fig, axes = plt.subplots(1, 2, figsize=kwargs.get('figsize', (10, 4)))
    # Histograma y KDE
    sns.histplot(df[column_name].dropna(), bins=bins, kde=True, ax=axes[0], color=kwargs.get('hist_color', 'skyblue'))
    axes[0].set_title(kwargs.get('hist_title', f'Distribución de {column_name}'), fontsize=12)
    axes[0].set_xlabel(kwargs.get('xlabel', column_name), fontsize=10)
    axes[0].set_ylabel(kwargs.get('ylabel_hist', 'Frecuencia'), fontsize=10)

    # Boxplot
    sns.boxplot(y=df[column_name].dropna(), ax=axes[1], color=kwargs.get('box_color', 'lightgreen'))
    axes[1].set_title(kwargs.get('box_title', f'Boxplot de {column_name}'), fontsize=12)
    axes[1].set_ylabel(kwargs.get('ylabel_box', column_name), fontsize=10)
    
    plt.tight_layout()
    plt.show()

    # Informar sobre nulos omitidos
    if df[column_name].isnull().sum() > 0:
        print(f"Nota: La columna '{column_name}' contiene {df[column_name].isnull().sum()} valores nulos que fueron omitidos en los gráficos.")

# --- Funciones de Formateo Específicas del Proyecto ---

def bucket_duration_to_readable(days: float | int | None) -> str:
    """
    Convierte una duración en días a una categoría de tiempo legible (ej. "<1 m", "1–3 m").
    Maneja valores nulos retornando "unknown".
    """
    if pd.isna(days): return "unknown"
    if days < 30: return "<1 m" # Menos de 1 mes
    if days < 90: return "1–3 m" # Entre 1 y 3 meses
    if days < 180: return "3–6 m" # Entre 3 y 6 meses
    if days < 365: return "6–12 m" # Entre 6 y 12 meses
    return ">12 m" # Más de 12 meses

def format_symptoms_for_display(row: pd.Series, sym_cols: List[str]) -> str:
    """
    Formatea una lista de síntomas y sus intensidades (de una fila de DataFrame)
    en una cadena legible, mostrando solo síntomas con intensidad > 0.
    Ejemplo de salida: "sintoma1: 3, sintoma2: 5". Retorna "—" si no hay síntomas.
    """
    pairs = [
        f"{col[4:]}: {int(row[col])}" # Elimina el prefijo 'sym_' del nombre de la columna
        for col in sym_cols
        if row[col] > 0 # Solo incluye síntomas presentes (intensidad > 0)
    ]
    return ", ".join(pairs) if pairs else "—"

# --- Funciones de Visualización para Análisis Bivariado y de Síntomas ---

def plot_bivariate_categorical(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    normalize_by: Optional[str] = 'index',
    plot_kind: str = 'bar',
    stacked: bool = False,
    figsize: tuple = (10, 6),
    colormap: Optional[str] = 'viridis',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Frecuencia"
):
    """
    Analiza la relación entre dos variables categóricas, mostrando tablas de contingencia
    (conteos y porcentajes normalizados) y un gráfico de barras.
    """
    # Validación de existencia de columnas
    if col_x not in df.columns or col_y not in df.columns:
        print(f"Error: Columnas {col_x} o {col_y} no encontradas en el DataFrame.")
        return

    print(f"\n--- Relación entre {col_x} y {col_y} ---")
    
    # Cálculo y visualización de tabla de contingencia (conteos absolutos)
    contingency_table = pd.crosstab(df[col_x], df[col_y])
    print("Tabla de Contingencia (Conteos):")
    display(contingency_table) # La función display es para notebooks

    # Cálculo y visualización de tabla de contingencia (porcentajes) si se especifica normalización
    if normalize_by:
        normalized_table = pd.crosstab(df[col_x], df[col_y], normalize=normalize_by)
        print(f"\nTabla de Contingencia (Porcentaje por {normalize_by}):")
        display(normalized_table.mul(100).round(1).astype(str) + '%')
    
    # Configuración de títulos y etiquetas para el gráfico
    if title is None:
        title = f'Distribución de {col_y} por {col_x}'
    if xlabel is None:
        xlabel = col_x

    # Generación del gráfico
    contingency_table.plot(
        kind=plot_kind,
        stacked=stacked,
        figsize=figsize,
        colormap=colormap
    )
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=col_y) # Leyenda basada en la segunda variable categórica
    plt.tight_layout()
    plt.show()

def plot_bivariate_categorical_numerical(
    df: pd.DataFrame,
    col_cat: str,
    col_num: str,
    plot_type: str = 'boxplot', 
    figsize: tuple = (10, 5),
    palette: Optional[str] = 'Set2',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
):
    """
    Analiza la relación entre una variable categórica y una numérica, mostrando
    un boxplot o violinplot y estadísticas descriptivas agrupadas.
    """
    # Validación de existencia de columnas
    if col_cat not in df.columns or col_num not in df.columns:
        print(f"Error: Columnas {col_cat} o {col_num} no encontradas en el DataFrame.")
        return

    print(f"\n--- Relación entre {col_cat} y {col_num} ---")

    # Configuración de títulos y etiquetas
    if title is None:
        title = f'Distribución de {col_num} por {col_cat}'
    if xlabel is None:
        xlabel = col_cat
    if ylabel is None:
        ylabel = col_num

    # Generación del gráfico (boxplot o violinplot)
    plt.figure(figsize=figsize)
    if plot_type == 'boxplot':
        sns.boxplot(x=col_cat, y=col_num, data=df, palette=palette)
    elif plot_type == 'violinplot':
        sns.violinplot(x=col_cat, y=col_num, data=df, palette=palette)
    else:
        print(f"Error: Tipo de gráfico '{plot_type}' no soportado. Use 'boxplot' o 'violinplot'.")
        return
        
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Muestra de estadísticas descriptivas agrupadas
    print(f"\nEstadísticas descriptivas de {col_num} por {col_cat}:")
    display(df.groupby(col_cat)[col_num].describe()) # display para notebooks

def plot_symptom_correlation_matrix(
    df: pd.DataFrame,
    symptom_cols: List[str],
    figsize_scale: tuple = (0.7, 0.6), # Escala para ajustar tamaño del heatmap
    annot: bool = True, # Mostrar anotaciones (valores de correlación)
    fmt: str = ".1f" # Formato de las anotaciones
):
    """
    Genera y muestra un heatmap de la matriz de correlación entre las intensidades
    de las columnas de síntomas especificadas.
    """
    # Validación de columnas de síntomas
    if not symptom_cols or not all(col in df.columns for col in symptom_cols):
        print("Error: Lista de columnas de síntomas vacía o algunas columnas no encontradas.")
        return

    print("\n--- Matriz de Correlación de Intensidades de Síntomas ---")
    # Cálculo de la matriz de correlación
    symptoms_intensity_df = df[symptom_cols]
    correlation_matrix = symptoms_intensity_df.corr()

    # Generación del heatmap
    plt.figure(figsize=(
        max(10, len(symptom_cols) * figsize_scale[0]), # Ancho dinámico
        max(8, len(symptom_cols) * figsize_scale[1])   # Alto dinámico
    ))
    sns.heatmap(
        correlation_matrix,
        annot=annot,
        fmt=fmt,
        cmap='coolwarm', # Paleta de colores para correlaciones (-1 a 1)
        vmin=-1, vmax=1, # Rango de la barra de color
        linewidths=.5,
        cbar_kws={'label': 'Coeficiente de Correlación'},
        xticklabels=[col.replace('sym_', '') for col in symptom_cols], # Nombres limpios para ejes
        yticklabels=[col.replace('sym_', '') for col in symptom_cols]
    )
    plt.title('Matriz de Correlación de Intensidades de Síntomas', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    # Muestra la matriz de correlación numérica con formato
    print("\nMatriz de Correlación (valores numéricos):")
    display(correlation_matrix.style.format("{:.2f}")) # display para notebooks

def plot_symptom_cooccurrence_matrix(
    df: pd.DataFrame,
    symptom_cols: List[str],
    figsize_scale: tuple = (0.7, 0.6),
    annot: bool = True,
    fmt: str = ".0f" # Formato para conteos enteros
):
    """
    Genera y muestra un heatmap de la matriz de co-ocurrencia de síntomas,
    basado en la presencia/ausencia de los mismos (intensidad > 0).
    """
    # Validación de columnas de síntomas
    if not symptom_cols or not all(col in df.columns for col in symptom_cols):
        print("Error: Lista de columnas de síntomas vacía o algunas columnas no encontradas.")
        return

    print("\n--- Matriz de Co-ocurrencia de Síntomas (Presencia/Ausencia) ---")
    # Creación de DataFrame binario de presencia de síntomas
    symptoms_presence_df = (df[symptom_cols] > 0).astype(int)
    
    # Inicialización de la matriz de co-ocurrencia
    cooccurrence_df = pd.DataFrame(
        index=symptoms_presence_df.columns,
        columns=symptoms_presence_df.columns,
        dtype=float # Usar float para heatmap, aunque los valores sean enteros
    )
    # Llenado de la diagonal con la frecuencia de cada síntoma
    for s_col in symptoms_presence_df.columns:
        cooccurrence_df.loc[s_col, s_col] = symptoms_presence_df[s_col].sum()
    
    # Llenado de las celdas fuera de la diagonal con conteos de co-ocurrencia
    for col1, col2 in combinations(symptoms_presence_df.columns, 2):
        cooccurrence_count = (symptoms_presence_df[col1] & symptoms_presence_df[col2]).sum()
        cooccurrence_df.loc[col1, col2] = cooccurrence_count
        cooccurrence_df.loc[col2, col1] = cooccurrence_count # Matriz simétrica

    # Generación del heatmap
    plt.figure(figsize=(
        max(10, len(symptom_cols) * figsize_scale[0]),
        max(8, len(symptom_cols) * figsize_scale[1])
    ))
    sns.heatmap(
        cooccurrence_df.astype(float), # Asegurar tipo float para cmap
        annot=annot,
        fmt=fmt,
        cmap='YlGnBu', # Paleta de colores adecuada para conteos
        linewidths=.5,
        cbar_kws={'label': 'Conteo de Co-ocurrencia'},
        xticklabels=[col.replace('sym_', '') for col in symptom_cols],
        yticklabels=[col.replace('sym_', '') for col in symptom_cols]
    )
    plt.title('Heatmap de Co-ocurrencia de Síntomas (Conteos)', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    # Muestra la tabla de co-ocurrencia numérica
    cooccurrence_df_display = cooccurrence_df.copy()
    cooccurrence_df_display.index = [idx.replace('sym_', '') for idx in cooccurrence_df_display.index]
    cooccurrence_df_display.columns = [col.replace('sym_', '') for col in cooccurrence_df_display.columns]
    print("\nTabla de Co-ocurrencia (Conteos):")
    display(cooccurrence_df_display.round(0).astype(int)) # display para notebooks

def plot_symptom_profile_by_group(
    df: pd.DataFrame,
    symptom_cols: List[str],
    group_by_col: str, # Columna por la cual agrupar (ej. 'condition', 'treatment')
    figsize_scale_intensity: tuple = (2.5, 0.6),
    figsize_scale_occurrence: tuple = (2.5, 0.6)
):
    """
    Genera y muestra dos heatmaps para el perfil de síntomas:
    1. Intensidad mediana de síntomas por grupo.
    2. Porcentaje de ocurrencia de síntomas por grupo.
    """
    # Validación de columnas
    if not symptom_cols or not all(col in df.columns for col in symptom_cols):
        print("Error: Lista de columnas de síntomas vacía o algunas columnas no encontradas.")
        return
    if group_by_col not in df.columns:
        print(f"Error: Columna de agrupación '{group_by_col}' no encontrada.")
        return

    print(f"\n--- Perfil de Síntomas por {group_by_col} ---")
    
    # Preparación de datos para los heatmaps
    groups = df[group_by_col].unique()
    symptom_names_clean = [s.replace('sym_', '') for s in symptom_cols]

    median_intensity_matrix = pd.DataFrame(index=symptom_names_clean, columns=groups, dtype=float)
    occurrence_percentage_matrix = pd.DataFrame(index=symptom_names_clean, columns=groups, dtype=float)
    occurrence_count_matrix = pd.DataFrame(index=symptom_names_clean, columns=groups, dtype=int)

    # Cálculo de métricas para cada síntoma y grupo
    for sym_col, sym_name_clean in zip(symptom_cols, symptom_names_clean):
        for group_val in groups:
            df_group = df[df[group_by_col] == group_val]
            # Manejo de grupos vacíos (poco probable pero seguro)
            if df_group.empty:
                median_intensity_matrix.loc[sym_name_clean, group_val] = np.nan
                occurrence_percentage_matrix.loc[sym_name_clean, group_val] = 0.0
                occurrence_count_matrix.loc[sym_name_clean, group_val] = 0
                continue

            symptom_present_in_group = df_group[df_group[sym_col] > 0]
            
            # Intensidad mediana (NaN si el síntoma no ocurre en el grupo)
            median_intensity_matrix.loc[sym_name_clean, group_val] = symptom_present_in_group[sym_col].median() if not symptom_present_in_group.empty else np.nan
            
            # Porcentaje y conteo de ocurrencia
            occurrence_count = len(symptom_present_in_group)
            total_in_group = len(df_group)
            occurrence_percentage_matrix.loc[sym_name_clean, group_val] = (occurrence_count / total_in_group * 100) if total_in_group > 0 else 0.0
            occurrence_count_matrix.loc[sym_name_clean, group_val] = occurrence_count

    # Generación del Heatmap de Intensidad Mediana
    plt.figure(figsize=(
        max(10, len(groups) * figsize_scale_intensity[0]),
        max(8, len(symptom_names_clean) * figsize_scale_intensity[1])
    ))
    sns.heatmap(median_intensity_matrix.astype(float), annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5, cbar_kws={'label': 'Intensidad Mediana'})
    plt.title(f'Intensidad Mediana de Síntomas por {group_by_col}', fontsize=14)
    plt.xlabel(group_by_col, fontsize=12)
    plt.ylabel('Síntoma Agrupado', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

    # Generación del Heatmap de Porcentaje de Ocurrencia
    plt.figure(figsize=(
        max(10, len(groups) * figsize_scale_occurrence[0]),
        max(8, len(symptom_names_clean) * figsize_scale_occurrence[1])
    ))
    # Formato de anotaciones para incluir porcentaje y conteo
    annotations_occurrence = occurrence_percentage_matrix.astype(float).round(1).astype(str) + "%\n(n=" + occurrence_count_matrix.astype(str) + ")"
    annotations_occurrence = annotations_occurrence.where(occurrence_count_matrix > 0, "n=" + occurrence_count_matrix.astype(str)) # Mostrar solo n=0 si no hay ocurrencias
    
    sns.heatmap(occurrence_percentage_matrix.astype(float), annot=annotations_occurrence, fmt="", cmap="Blues", linewidths=.5, cbar_kws={'label': 'Ocurrencia (%)'})
    plt.title(f'Porcentaje de Ocurrencia de Síntomas por {group_by_col}', fontsize=14)
    plt.xlabel(group_by_col, fontsize=12)
    plt.ylabel('Síntoma Agrupado', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

    # Muestra de las tablas numéricas
    print(f"\nTabla de Intensidad Mediana de Síntomas por {group_by_col} (NaN si no ocurre):")
    display(median_intensity_matrix.round(1)) # display para notebooks
    print(f"\nTabla de Porcentaje de Ocurrencia de Síntomas por {group_by_col} (%):")
    display(occurrence_percentage_matrix.round(1)) # display para notebooks

import random, torch, numpy as np
from .config import SEED, IMG_SIZE
import torchvision.transforms as T

# ------------------------------------------------------------------ SEED ALL --
def seed_everything(seed:int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------- DENORMALIZE IMAGE --
def denormalize_image(t: torch.Tensor) -> torch.Tensor:
    """Invierte la normalización ImagenNet para visualizar."""
    mean = torch.tensor([0.485,0.456,0.406], device=t.device)[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225], device=t.device)[:,None,None]
    return (t*std + mean).clamp(0,1)

# ------------- PEQUEÑO DEBUG VISUAL (opcional, no afecta producción) ---------
def show_tensor_image(t: torch.Tensor, title:str="img"):
    import matplotlib.pyplot as plt
    plt.imshow(denormalize_image(t).permute(1,2,0).cpu()); plt.title(title); plt.axis("off")
# ───────────────────────────────────────────────────────────── SMOTE helper ──
from imblearn.over_sampling import SMOTE
import numpy as np

def smote_balance(
    X: np.ndarray | pd.DataFrame,
    y: pd.Series | np.ndarray,
    seed: int = 42,
    k_max: int = 5,
):
    """
    Aplica SMOTE cuidando que k_neighbors < tamaño de la clase minoritaria.

    Retorna
    -------
    X_res, y_res : ndarray | DataFrame, ndarray
        Conjunto remuestreado (mismo tipo de X de entrada) y etiquetas.
    """
    uniq, cnts = np.unique(y, return_counts=True)
    k_neighbors = min(k_max, max(1, cnts.min() - 1))
    if k_neighbors < 1:
        # Demasiado pocas muestras -> devuelve sin cambios
        return X, y

    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X, y)

    # Mantén DataFrame si la entrada era DataFrame
    if isinstance(X, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    return X_res, y_res
