# src/evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix as sk_confusion_matrix
)
from sklearn.calibration import calibration_curve

def plot_roc_curves(y_true, y_score, class_names):
    """
    y_true: array de etiquetas verdaderas (shape [n_samples])
    y_score: matriz de probabilidades (shape [n_samples, n_classes])
    """
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        RocCurveDisplay.from_predictions(
            (y_true == i).astype(int),
            y_score[:, i],
            name=name
        )
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()

def plot_precision_recall(y_true, y_score, class_names):
    """
    Dibuja las curvas Precision-Recall para cada clase usando PrecisionRecallDisplay.from_predictions.
    """
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        # binarizamos ground-truth para la clase i
        y_true_bin = (y_true == i).astype(int)
        # calculamos la curva
        PrecisionRecallDisplay.from_predictions(
            y_true_bin,
            y_score[:, i],
            name=name
        ).plot()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.show()

def print_additional_metrics(y_true, y_pred):
    """
    Imprime métricas adicionales: Balanced Accuracy, Matthews CorrCoef y Cohen's Kappa.
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc     = matthews_corrcoef(y_true, y_pred)
    kappa   = cohen_kappa_score(y_true, y_pred)
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Matthews CorrCoef : {mcc:.4f}")
    print(f"Cohen's Kappa     : {kappa:.4f}")

def plot_calibration(y_true_bin, y_prob, class_name, n_bins=10):
    """
    Curva de calibración para una sola clase binaria.
    y_true_bin: etiquetas binarizadas (0/1)
    y_prob   : probabilidades predichas para la clase positiva
    """
    prob_true, prob_pred = calibration_curve(y_true_bin, y_prob, n_bins=n_bins)
    plt.figure(figsize=(6,4))
    plt.plot(prob_pred, prob_true, marker='o', label=class_name)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfecta')
    plt.xlabel("Prob Predicha")
    plt.ylabel("Prob Observada")
    plt.title(f"Curva de Calibración: {class_name}")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names,
                          normalize=None,
                          cmap="Blues",
                          figsize=(6,5),
                          fmt=".2f",
                          title=None,
                          save_path=None):
    """
    Muestra la matriz de confusión.

    Parámetros:
    - y_true, y_pred: etiquetas verdaderas y predichas.
    - class_names: lista de nombres de clase en orden de índices.
    - normalize: None | 'true' | 'pred' | 'all' (como sklearn.metrics.confusion_matrix)
    - cmap, figsize, fmt: parámetros de seaborn heatmap.
    - title: título opcional.
    - save_path: Path para guardar la figura (opcional).
    """
    cm = sk_confusion_matrix(y_true, y_pred, normalize=normalize)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names,
                yticklabels=class_names)
    if title is None:
        title = "Matriz de Confusión"
        if normalize:
            title += f" (normalized={normalize})"
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Verdadera")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
