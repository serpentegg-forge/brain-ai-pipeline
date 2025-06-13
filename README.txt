Brain-Cancer AI Pipeline
========================

Repositorio      : https://github.com/serpentegg-forge/brain-cancer-ai-pipeline
Autor            : Juan Pablo Restrepo Urrea
Python probado   : 3.12


------------------------------------------------------------
0. Requisitos mínimos de entorno
------------------------------------------------------------

• Python 3.12.x  
• (Opcional) GPU CUDA 11.8+ para acelerar EfficientNet y XGBoost  
• Dependencias: instalar con «pip install -r requirements.txt»  
• Imágenes MRI en «data/Brain_Cancer raw MRI data/brain_cancer_IMG/…»  
• Datos tabulares en «data/brain_conditions_detailed_dataset.csv»

Crear un entorno virtual (ejemplo Windows):

    py -3.12 -m venv .venv
    .\.venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt

------------------------------------------------------------
1. Exploración y pre-procesamiento ─ Notebook 01
------------------------------------------------------------

Archivo: notebooks/01_EDA_and_Preprocessing.ipynb  
Salida: data/processed/clinical_data_processed_with_paths.csv

------------------------------------------------------------
2. Clasificador de tipo de tumor ─ Notebook 02
------------------------------------------------------------

Archivo: notebooks/02_MRI_Tumor_Classification_Model.ipynb  
Salida: models/tumor_final.pth

Parámetros clave (config.py):
IMG_SIZE=256, BATCH_SIZE=16, NUM_EPOCHS=15 …

------------------------------------------------------------
3. Recomendador de tratamiento ─ Notebook 03
------------------------------------------------------------

Archivo: notebooks/03_Treatment_Recommendation_Model.ipynb  
Pasos: split → preproc → SMOTE → tuning XGBoost → evaluación → SHAP  
Artefactos:  
    models/treatment_preproc.joblib  
    models/treatment_xgb_tuned.joblib  
    models/label_encoders.joblib

------------------------------------------------------------
4. Inferencia offline (script)
------------------------------------------------------------

    from inference_pipeline import predict_treatment

    out = predict_treatment(
        image_path=r"data/.../brain_glioma_0001.jpg",
        age=74,
        sex="F",
        clinical_note=(
            "Patient experiencing memory lapses and speech difficulties "
            "for the last few weeks, described as intense in nature."
        ),
    )
    print(out)

Formato de retorno:

    {
      "condition_code": 1,
      "condition_label": "Brain Glioma",
      "treatment_code": 3,
      "treatment_label": "surgery",
      "probabilities": [0.10, 0.05, 0.12, 0.73]
    }

La «clinical_note» debe seguir la sintaxis del dataset, por ejemplo:

    "Patient experiencing recurrent seizures and blurred vision for the past few months, described as severe in nature."

------------------------------------------------------------
5. API REST con FastAPI
------------------------------------------------------------

Arranque (Python 3.12):

    cd brain-cancer-ai-pipeline
    py -3.12 -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Prueba:

    import requests, pathlib, json
    url = "http://127.0.0.1:8000/predict"
    img = pathlib.Path("data/.../brain_glioma_0001.jpg")
    files = {"image": ("glioma.jpg", img.open("rb"), "image/jpeg")}
    data  = {
        "age": 74,
        "sex": "F",
        "clinical_note": "Patient is experiencing memory lapses ..."
    }
    r = requests.post(url, files=files, data=data)
    print(r.status_code, json.dumps(r.json(), indent=2))

Swagger / Redoc: http://127.0.0.1:8000/docs

------------------------------------------------------------
6. Estructura del repositorio
------------------------------------------------------------

brain-cancer-ai-pipeline/
├ api/                 (FastAPI)
│   └ main.py
├ data/
│   ├ Brain_Cancer raw MRI data/
│   └ processed/
├ models/
├ notebooks/
│   ├ 01_EDA_and_Preprocessing.ipynb
│   ├ 02_MRI_Tumor_Classification_Model.ipynb
│   └ 03_Treatment_Recommendation_Model.ipynb
├ src/
│   ├ config.py
│   ├ data_preprocessing.py
│   ├ image_processing.py
│   ├ model_training.py
│   ├ inference_pipeline.py
│   └ …
├ requirements.txt
└ README.txt  (este archivo)

------------------------------------------------------------
Preguntas frecuentes
------------------------------------------------------------

• ¿GPU obligatoria? No, pero acelera el entrenamiento.  
• Error «IProgress not found»: instalar/actualizar ipywidgets.  
• Error 415 «Solo se aceptan JPG/PNG»: verifica Content-Type en la petición.

Con esto puedes entrenar, inferir y servir el modelo de recomendación de tratamiento.
