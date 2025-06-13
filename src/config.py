from pathlib import Path

# ─── Hyper-parameters shared by all training scripts ──────────────────────────
import torch as _torch     # importación local para no contaminar namespace

# --- Rutas del Proyecto ---
BASE_PROJECT_PATH = Path(__file__).resolve().parent.parent # Sube dos niveles desde src/config.pys

# --- Rutas de Datos Tabulares ---
TABULAR_DATA_SYNTHETIC_PATH = BASE_PROJECT_PATH / "data" / "brain_conditions_detailed_dataset.csv"
TABULAR_DATA_KAGGLE_METADATA_PATH = BASE_PROJECT_PATH / "data" / "Brain_Cancer raw MRI data" / "dataset.csv"

# --- Rutas de Imágenes ---
IMAGE_DIR_PATH = BASE_PROJECT_PATH / "data" / "Brain_Cancer raw MRI data" / "brain_cancer_IMG"
GLIOMA_IMAGE_PATH = IMAGE_DIR_PATH / "brain_glioma"
MENIN_IMAGE_PATH = IMAGE_DIR_PATH / "brain_menin"
TUMOR_IMAGE_PATH = IMAGE_DIR_PATH / "brain_tumor"

# --- Constantes para Parseo de Notas Clínicas ---
SYMPTOM_SPLIT_PATTERN = r',| and |;'
SYMPTOM_EXTRACTION_PATTERN = r"experiencing (.+?) (?:for|over|in|during|since|the past|the last|this|these)"
DURATION_EXTRACTION_PATTERN = r"(?:for|over|in|during|since|the past|the last|this|these) (.+?)(?:\.|$)"
INTENSITY_EXTRACTION_PATTERN = r"described as (.+?) in nature"

ADJECTIVES = (
    'persistent', 'frequent', 'chronic', 'recurrent', 'severe', 'mild', 'intense',
    'significant', 'moderate', 'debilitating', 'tolerable', 'slight', 'noticeable',
    'low-grade', 'high-grade'
)
ADJECTIVE_CLEANUP_PATTERN = rf"^({'|'.join(ADJECTIVES)})\s+"
CLASS_NAMES = ['Brain Glioma', 'Brain Meningioma', 'Brain Tumor'] 
INTENSITY_SCALE_MAP = {
    'mild': 1,
    'slight': 2, 'tolerable': 2, 'noticeable': 2, 'low-grade': 2, 'nogreat': 2, # nogreat as per original
    'moderate': 3,
    'significant': 4,
    'intense': 5,
    'severe': 6,
    'debilitating': 7,
    'high-grade': 8
}

# --- Mapeo para Agrupación de Síntomas ---
# Clave: nuevo nombre base del síntoma agrupado
# Valor: lista de nombres base de síntomas originales a agrupar
SYMPTOM_GROUPS_MAP = {
    'speech_issue': ['speech difficultie', 'speech difficulty', 'speech problem'],
    'seizure': ['seizure', 'recurring seizure'],
    'vision_problem': ['blurry vision', 'visual disturbance'],
    'motor_coordination_issue': ['coordination problem', 'motor dysfunction'],
    'balance_problem': ['balance issue'],
    'neurological_weakness': ['localized neurological weaknes'],
    'memory_problem': ['memory lapse'],
    'concentration_difficulty': ['difficulty concentrating'],
    'cognitive_impairment': ['difficulty concentrating', 'confusion'], # Ejemplo de agrupación más amplia
}

# --- Extensiones de archivo de imagen comunes ---
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tif']

# Dispositivo
DEVICE: str = "cuda" if _torch.cuda.is_available() else "cpu"

# Arquitectura base (timm)
MODEL_NAME: str = "efficientnet_b3a"

# Optimizador
LEARNING_RATE: float = 2e-4
WEIGHT_DECAY:  float = 1e-4

# ---------------------- MODEL & TRAINING HYPERPARAMETERS ----------------------
SEED              = 42
N_FOLDS           = 5
BATCH_SIZE        = 16
NUM_EPOCHS        = 15         
IMG_SIZE          = 256
MODEL_NAME        = "efficientnet_b3a"
LEARNING_RATE     = 2e-4
WEIGHT_DECAY      = 1e-4
PATIENCE_EARLY_ST = 3          # early-stopping patience

# -------------------------- OUTPUT / CHECKPOINTS -----------------------------
MODEL_DIR   = BASE_PROJECT_PATH / "models" 
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_CV_PATH   = MODEL_DIR / f"{MODEL_NAME}_best_cv.pth"
FINAL_PATH     = MODEL_DIR / f"{MODEL_NAME}_final.pth"


BASE_PROJECT_PATH = Path(__file__).resolve().parent.parent  # raíz del proyecto
PROCESSED_DATA_PATH = BASE_PROJECT_PATH / "data" / "processed" / "clinical_data_processed_with_paths.csv"
#-----hiper parámetros de entrenamiento treatment 
MODEL_TUMOR_CLASSIFIER_PATH = BASE_PROJECT_PATH / "models" / "tumor_final.pth"
NUM_CLASSES_CONDITION = 3 # O el número real de clases de 'condition'
N_FOLDS_TREATMENT = 5     # Ejemplo, o el valor que quieras para el modelo de tratamiento