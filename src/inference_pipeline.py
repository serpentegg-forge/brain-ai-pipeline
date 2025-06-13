# ============================== inference_pipeline.py =========================
# Entrada :  image_path, age, sex, clinical_note
# Salida   :  dict con condition/treatment (code & label) + probabilidades
# -----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing  import Union, Dict, Any

import sys, joblib, torch, pandas as pd
from PIL import Image
import torchvision.transforms as T                       # ⬅️  nuevo import

# ────────────────────────── local project import setup ───────────────────────
HERE = Path().resolve()
for p in [HERE, *HERE.parents]:
    if (p / "src").is_dir():
        PROJECT_ROOT = p
        break
else:
    raise RuntimeError(f"❌  No se encontró carpeta 'src' desde {HERE}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ───────────────────────────── artefactos entrenados ─────────────────────────
from src.config import (
    DEVICE, IMG_SIZE, SYMPTOM_GROUPS_MAP,
    INTENSITY_SCALE_MAP,                    # ⬅️  ya venía desde config
)

ROOT      = PROJECT_ROOT
MODELS    = ROOT / "models"

EFFNET_W  = MODELS / "tumor_final.pth"
PREPROC_W = MODELS / "treatment_preproc.joblib"
XGB_W     = MODELS / "treatment_xgb_tuned.joblib"
ENC_W     = MODELS / "label_encoders.joblib"

_preproc  = joblib.load(PREPROC_W)
_xgb      = joblib.load(XGB_W)
_enc      = joblib.load(ENC_W)
_le_cond  = _enc["condition"]
_le_treat = _enc["treatment"]
_EXP_COLS = list(_preproc.feature_names_in_)          # columnas esperadas

# ────────────────────────── modelos / transforms ─────────────────────────────
from src.model_training import get_model

_effnet = get_model(num_classes=len(_le_cond.classes_), pretrained=False)
_effnet.load_state_dict(torch.load(EFFNET_W, map_location=DEVICE))
_effnet.to(DEVICE).eval()

#  🔑  Transformaciones de validación (añadimos ToTensor antes de Normalize)
_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),                                          # ← imprescindible
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ───────────── utilidades EDA ya implementadas en src.data_preprocessing ─────
from src.data_preprocessing import _parse_single_clinical_note

# -----------------------------------------------------------------------------
@torch.no_grad()
def _predict_condition(img_path: Union[str, Path]) -> int:
    """Inferencia EfficientNet → código de condición."""
    img_path = Path(img_path)
    img = Image.open(img_path).convert("RGB")
    tensor = _tfms(img).unsqueeze(0).to(DEVICE)
    return int(_effnet(tensor).argmax(1).item())


def _symptom_aggregation(row_sym_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Aplica la misma lógica de agrupación (`SYMPTOM_GROUPS_MAP`) pero para un solo
    paciente. Devuelve un dict {col_name: intensity}.
    """
    out = row_sym_dict.copy()
    for new_base, old_bases in SYMPTOM_GROUPS_MAP.items():
        new_col = f"sym_{new_base}"
        max_val = 0.0
        for old in old_bases:
            old_col = f"sym_{old}"
            if old_col in out:
                max_val = max(max_val, out.pop(old_col))
        if max_val:
            out[new_col] = max_val
    return out


def _build_features(age: int, sex: str,
                    note_dict: Dict[str, Any],
                    cond_code: int) -> pd.DataFrame:
    """
    Construye un DataFrame con todas las columnas esperadas por el preprocesador.
    """
    base = {
        "age": age,
        "duration_days": note_dict["duration_days"],
        "sex_code": 1 if str(sex).upper().startswith("M") else 0,
        "condition_code": cond_code,
    }

    # síntomas individuales con la intensidad extraída
    sym_cols = {
        f"sym_{s}": note_dict["intensity_lvl"]
        for s in note_dict["symptoms_list"]
        if note_dict["intensity_lvl"] == note_dict["intensity_lvl"]  # descarta NaN
    }

    # aplica la misma lógica de agrupación utilizada en training
    sym_cols = _symptom_aggregation(sym_cols)
    base.update(sym_cols)

    # asegura presencia de TODAS las columnas esperadas
    for col in _EXP_COLS:
        base.setdefault(col, 0)

    return pd.DataFrame([base], columns=_EXP_COLS)


# ───────────────────────────────── PUBLIC API ────────────────────────────────
def predict_treatment(image_path: Union[str, Path],
                      age: int,
                      sex: str,
                      clinical_note: str) -> Dict[str, Any]:
    """
    Pipeline completo:
      1) EfficientNet ➜ predice condición (tumor type)
      2) Se procesan nota clínica + variables demográficas
      3) Preprocesador clínico + XGBoost ➜ predice tratamiento
    """
    # 1. condición tumoral
    cond_code = _predict_condition(image_path)
    cond_lbl  = _le_cond.inverse_transform([cond_code])[0]

    # 2. parseo nota clínica & ensamblado de features
    note_info = _parse_single_clinical_note(clinical_note)
    df_feats  = _build_features(age, sex, note_info, cond_code)
    X_proc    = _preproc.transform(df_feats)

    # 3. inferencia tratamiento
    treat_code = int(_xgb.predict(X_proc)[0])
    treat_lbl  = _le_treat.inverse_transform([treat_code])[0]
    probas     = _xgb.predict_proba(X_proc)[0]

    return {
        "condition_code":  cond_code,
        "condition_label": cond_lbl,
        "treatment_code":  treat_code,
        "treatment_label": treat_lbl,
        "probabilities":   probas.tolist(),
    }