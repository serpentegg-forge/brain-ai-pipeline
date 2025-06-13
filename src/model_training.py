# src/model_training.py

import matplotlib.pyplot as plt
import timm
import torch
import xgboost as xgb
import numpy as np 
from packaging import version
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from scipy.stats import uniform, randint
from .config import DEVICE, MODEL_NAME
from .utils import smote_balance

# ------------------------------------------------------------------ NETWORK --
def get_model(num_classes: int,
              model_name: str = MODEL_NAME,
              pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Linear):
        in_f = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.35),
            torch.nn.Linear(in_f, num_classes)
        )
    return model.to(DEVICE)

# ---------------------------------------------------------- ONE-EPOCH TRAIN --
def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

# ------------------------------------------------------------- EVALUATE LOOP -
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running = 0.0
    preds_list, trues_list = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        running += criterion(logits, y).item() * x.size(0)
        preds_list.append(logits.argmax(1).cpu())
        trues_list.append(y.cpu())
    preds = torch.cat(preds_list)
    trues = torch.cat(trues_list)
    acc = (preds == trues).float().mean().item()
    return running / len(loader.dataset), acc, trues.numpy(), preds.numpy()

# ------------------------------------------------------------- SIMPLE PLOT ---
def plot_history(hist: dict, title: str = "", save_path=None):
    if not hist.get("train_loss"):
        return
    epochs = range(1, len(hist["train_loss"]) + 1)
    plt.plot(epochs, hist["train_loss"], "o-", label="train_loss")
    if hist.get("val_loss"):
        plt.plot(epochs, hist["val_loss"], "o-", label="val_loss")
    if hist.get("val_acc"):
        plt.plot(epochs, hist["val_acc"], "s-", label="val_acc")
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ---------------------------------------------------------- SAVE / LOAD ------
def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(path: str, num_classes: int) -> torch.nn.Module:
    model = get_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

# ───────────────────────────────────────── XGBoost tuning & training ───────


_EXTENSIVE_PARAM_GRID = {
    "n_estimators": randint(100, 1001),
    "learning_rate": uniform(loc=0.005, scale=0.195),
    "max_depth": randint(3, 13),
    "min_child_weight": randint(1, 11),
    "gamma": uniform(loc=0, scale=0.5),
    "subsample": uniform(loc=0.5, scale=0.5),
    "colsample_bytree": uniform(loc=0.5, scale=0.5),
    "colsample_bylevel": uniform(loc=0.5, scale=0.5),
    "colsample_bynode": uniform(loc=0.5, scale=0.5),
    "reg_alpha": uniform(loc=0, scale=1.0),
    "reg_lambda": uniform(loc=0, scale=1.0),
    "max_delta_step": randint(0, 11),
}
# ───────────────────────────────────────── XGBoost tuning manual ─────────────
import random
from IPython.display import clear_output
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

def tune_xgb_classifier(
    X, y,
    param_grid: dict | None = None,
    cv_splits: int = 3,
    n_iter: int = 30,
    seed: int = 42,
    scoring: str = "f1_weighted",
    use_gpu: bool = True,
):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.utils.class_weight import compute_sample_weight
    import random
    from IPython.display import clear_output

    param_grid = param_grid or _EXTENSIVE_PARAM_GRID
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    sample_weights = compute_sample_weight("balanced", y)

    xgb_ver = version.parse(xgb.__version__)
    extra = {}
    if use_gpu:
        if xgb_ver >= version.parse("2.0.0"):
            extra.update(tree_method="hist", device="cuda")
        else:
            extra.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        extra.update(tree_method="hist", predictor="cpu_predictor")

    best_score = -float("inf")
    best_params = None
    best_model = None
    keys = list(param_grid.keys())
    random.seed(seed)

    for i in range(1, n_iter+1):
        # Muestreo robusto
        params = {}
        for k in keys:
            v = param_grid[k]
            if hasattr(v, "rvs"):
                # paso random_state distinto cada iter para diversificar
                params[k] = v.rvs(random_state=seed + i)
            else:
                params[k] = random.choice(v)

        clear_output(wait=True)
        print(f"Iteration {i}/{n_iter} — Params: {params}")

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(set(y)),
            eval_metric="mlogloss",
            random_state=seed,
            **extra,
            **params
        )

        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=1 if use_gpu else -1,
            fit_params={"sample_weight": sample_weights}
        )
        mean_score = scores.mean()
        print(f"CV {scoring}: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = model

    clear_output(wait=True)
    print("=== Best params ===", best_params)
    print(f"Best CV {scoring}: {best_score:.4f}")
    best_model.fit(X, y, sample_weight=sample_weights)

    return best_model, best_params, best_score


