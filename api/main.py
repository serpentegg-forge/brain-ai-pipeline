# api/main.py
# -----------------------------------------------------------------------
# FastAPI service – Brain-Cancer Treatment Recommender
# Exposes POST /predict that receives:
#   • image (multipart file jpg/png)
#   • age (int), sex ("M"/"F"), clinical_note (str)
# Returns JSON with condition & treatment predictions.
# -----------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import sys
import os
import tempfile
from typing import Dict, Any

# Calcula la raíz del proyecto (dos niveles por encima de /api)
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # …/brain-cancer-ai-pipeline
SRC_DIR      = PROJECT_ROOT / "src"

# Añade la raíz y /src al PYTHONPATH solo si no están ya
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Imports de la API
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Tu pipeline ya entrenado
from inference_pipeline import predict_treatment

app = FastAPI(
    title="Brain-Cancer AI Pipeline – Treatment Recommender",
    description=(
        "Inferencia de tipo de tumor (EfficientNet) + recomendación "
        "de tratamiento (XGBoost) a partir de imagen MRI y datos clínicos."
    ),
    version="1.0.0"
)

# ----------------------------- response schema --------------------------------
class PredictionOut(BaseModel):
    condition_code:  int         = Field(..., example=1)
    condition_label: str         = Field(..., example="Brain Glioma")
    treatment_code:  int         = Field(..., example=3)
    treatment_label: str         = Field(..., example="surgery")
    probabilities:   list[float] = Field(
        ..., example=[0.12, 0.05, 0.70, 0.13]
    )

# ----------------------------- API end-point ----------------------------------
@app.post("/predict", response_model=PredictionOut)
async def predict(
    image: UploadFile   = File(..., description="MRI slice (jpg/png)"),
    age: int            = Form(..., ge=0, lt=120),
    sex: str            = Form(..., regex="^(M|F|m|f)$"),
    clinical_note: str  = Form(..., min_length=10),
):
    # 1️⃣ Validar tipo MIME
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Solo se aceptan JPG/PNG")

    # 2️⃣ Guardar la imagen en un temp-file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
            data = await image.read()
            tmp.write(data)
            tmp_path = Path(tmp.name)

        # 3️⃣ Llamar al pipeline
        result: Dict[str, Any] = predict_treatment(
            image_path    = tmp_path,
            age           = age,
            sex           = sex,
            clinical_note = clinical_note
        )
        return JSONResponse(content=result)

    finally:
        # 4️⃣ Limpiar el archivo temporal
        if tmp_path and tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
