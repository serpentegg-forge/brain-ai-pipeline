import numpy as np
from pathlib import Path
from typing     import List, Tuple
import torch
from torch.utils.data import Dataset
from torchvision.io  import read_image
import torchvision.transforms as T
from .config import IMG_SIZE
from .utils  import seed_everything

# ---------------------------------------------------------------- MRIDataset -
class MRIDataset(Dataset):
    """Carga rutas y etiquetas desde un DataFrame."""
    def __init__(self, df, path_col:str="image_path", label_col:str="condition", transform=None):
        self.paths      = df[path_col].astype(str).tolist()
        labels_series   = df[label_col].astype("category")
        self.labels     = torch.tensor(labels_series.cat.codes.values, dtype=torch.long)
        self.label_names= list(labels_series.cat.categories)
        self.transform  = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor,int]:
        img = read_image(self.paths[idx]).float()/255.0
        if img.shape[0]==1:   img = img.repeat(3,1,1)        # grayscale → RGB
        elif img.shape[0]==4: img = img[:3]                   # RGBA → RGB
        if self.transform:    img = self.transform(img)
        return img, int(self.labels[idx])

# --------------------------------------------------------- TRANSFORM HELPERS -
def get_train_transforms(img_size:int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),  # aumento de rotación
        T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # desplazamientos pequeños
            scale=(0.9, 1.1)       # zoom in/out
        ),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size:int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size,img_size), antialias=True),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

# optional: función simple para visualizar un lote
def preview_augmentations(df, n:int=4):
    import matplotlib.pyplot as plt
    seed_everything()
    ds = MRIDataset(df.sample(n), transform=get_train_transforms())
    fig,ax = plt.subplots(1,n, figsize=(3*n,3))
    for i,(img,lab) in enumerate(ds):
        ax[i].imshow((img.permute(1,2,0)*0.229+0.485).clamp(0,1)); ax[i].set_title(ds.label_names[lab]); ax[i].axis("off")
    plt.show()
# ───────────────────────────────────────────── EfficientNet inference helper ──
import torch
from .model_training import get_model            # ya implementado :contentReference[oaicite:1]{index=1}
from .config          import NUM_CLASSES_CONDITION, BATCH_SIZE, DEVICE

@torch.no_grad()
def predict_conditions(
    df,
    model_weights_path: str | Path,
    path_col: str = "image_path",
    label_dummy_col: str = "condition_code",
    batch_size: int = BATCH_SIZE,
    device: str | torch.device = DEVICE,
):
    """
    Devuelve un np.array con los códigos de condición predichos
    usando el EfficientNet ya entrenado.
    """
    device = torch.device(device)
    model = get_model(num_classes=NUM_CLASSES_CONDITION, pretrained=False)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device).eval()

    ds = MRIDataset(
        df, path_col=path_col, label_col=label_dummy_col,
        transform=get_val_transforms()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    for x, _ in loader:
        x = x.to(device)
        preds_batch = model(x).argmax(1).cpu().numpy()
        preds.extend(preds_batch)
    return np.asarray(preds)
