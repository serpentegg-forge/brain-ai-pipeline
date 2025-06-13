
# scripts/infer_tumor_batch.py
import os, sys
from pathlib import Path
import numpy as np
import torch, timm
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd

# ─── 1. Parámetros ──────────────────────────────────────────────────────────
IMG_SIZE    = 256
MODEL_NAME  = "efficientnet_b3a"
NUM_CLASSES = 3
CLASS_NAMES = ['Brain Glioma', 'Brain Meningioma', 'Brain Tumor']
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH  = Path(__file__).parent.parent / "models" / "tumor_final.pth"

tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ─── 2. Utilidades ───────────────────────────────────────────────────────────
def load_model():
    print(f"→ Cargando modelo   {MODEL_PATH.name}  en  {DEVICE}")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    # === Asegurarse de usar Dropout+Linear como en el entrenamiento ===
    if isinstance(model.classifier, torch.nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.35),
            torch.nn.Linear(in_features, NUM_CLASSES)
        )

    # === Cargar el state_dict exactamente como fue entrenado ===
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model



def choose_files():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    files = filedialog.askopenfilenames(
        title="Selecciona hasta 5 MRIs",
        filetypes=[("Imágenes","*.jpg *.jpeg *.png *.tif *.tiff"), ("Todos","*.*")]
    )
    root.destroy()
    return list(files)[:5]          # máx 5

def preprocess(paths):
    tensors = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        x   = tfm(T.ToTensor()(img))
        tensors.append(x)
    return torch.stack(tensors).to(DEVICE)   # B×3×H×W

# ─── 3. Main ─────────────────────────────────────────────────────────────────
def main():
    model = load_model()

    while True:
        paths = choose_files()
        if not paths:
            print("⛔ No se seleccionaron imágenes. Saliendo.")
            break

        print(f"✔ {len(paths)} imagen(es) seleccionadas")
        batch = preprocess(paths)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy() * 100  # → %

        # 3a. Tabla completa en porcentaje
        df = pd.DataFrame(np.round(probs,1), columns=CLASS_NAMES,
                          index=[Path(p).name for p in paths])
        print("\n=== Probabilidades (%) ===")
        print("\n")
        display(df)

        # 3b. Dibujar grid 1×N
        n = len(paths)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
        if n == 1: axes = [axes]

        for ax, p, prob_vec in zip(axes, paths, probs):
            img = Image.open(p)
            idx = int(prob_vec.argmax())
            title = f"{CLASS_NAMES[idx]}\n{prob_vec[idx]:.1f}%"
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # 3c. Continuar o salir
        if input("¿Clasificar otras imágenes? [s/N]: ").strip().lower() != 's':
            break

if __name__ == "__main__":
    main()
