"""
evaluate.py

Cubre las Partes 8-14 y 21 del enunciado:

    - evaluate_model            -> Parte 8 (accuracy, precision, recall, F1)
    - plot_confusion_matrix     -> Parte 9
    - plot_predictions          -> Parte 10 (5 correctas + 5 incorrectas)
    - analyze_misclassified     -> Parte 11 (10 imágenes mal clasificadas
                                   con confianza)
    - plot_feature_maps         -> Parte 12 (>=16 feature maps de la
                                   primera capa conv)
    - predict_image             -> Parte 21 (inferencia de una sola imagen)

Uso típico (después de entrenar con train.py):

    python src/evaluate.py --experiment basic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, CIFAR10_CLASSES
from utils import get_dataloaders, get_device, denormalize


@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Parte 8. Devuelve accuracy, precision, recall, f1 (macro) y las
    listas de y_true / y_pred, evaluando SOLO sobre el loader de test.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES, zero_division=0))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def plot_confusion_matrix(y_true, y_pred, results_dir="results", prefix=""):
    """Parte 9. Guarda results/{prefix}confusion_matrix.png"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/{prefix}confusion_matrix.png", dpi=150)
    plt.close(fig)
    return cm


@torch.no_grad()
def plot_predictions(model, dataset, device, results_dir="results", prefix="", n_each=5):
    """
    Parte 10. Busca n_each predicciones correctas y n_each incorrectas
    (recorriendo el dataset) y guarda results/{prefix}predictions.png
    """
    import matplotlib.pyplot as plt

    model.eval()
    correct_samples, incorrect_samples = [], []

    idxs = np.random.permutation(len(dataset))
    for idx in idxs:
        if len(correct_samples) >= n_each and len(incorrect_samples) >= n_each:
            break
        image, label = dataset[idx]
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()

        if pred == label and len(correct_samples) < n_each:
            correct_samples.append((image, label, pred))
        elif pred != label and len(incorrect_samples) < n_each:
            incorrect_samples.append((image, label, pred))

    samples = correct_samples + incorrect_samples
    fig, axes = plt.subplots(2, n_each, figsize=(2.2 * n_each, 5))

    for col, (image, label, pred) in enumerate(samples):
        row = 0 if col < n_each else 1
        ax = axes[row, col % n_each]
        ax.imshow(denormalize(image))
        color = "green" if label == pred else "red"
        ax.set_title(
            f"Real: {CIFAR10_CLASSES[label]}\nPred: {CIFAR10_CLASSES[pred]}", color=color, fontsize=8
        )
        ax.axis("off")

    axes[0, 0].set_ylabel("Correctas", fontsize=10)
    axes[1, 0].set_ylabel("Incorrectas", fontsize=10)
    fig.tight_layout()

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{results_dir}/{prefix}predictions.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def analyze_misclassified(model, dataset, device, n=10):
    """
    Parte 11. Devuelve una lista de dicts con imagen mal clasificada,
    clase real, clase predicha y confianza (softmax) — para que el
    notebook los imprima y el estudiante responda las preguntas de
    análisis cualitativo (ambigüedad, tamaño del objeto, fondo, etc.).
    """
    model.eval()
    results = []
    idxs = np.random.permutation(len(dataset))

    for idx in idxs:
        if len(results) >= n:
            break
        image, label = dataset[idx]
        output = model(image.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

        if pred != label:
            results.append(
                {
                    "image": image,
                    "true_label": CIFAR10_CLASSES[label],
                    "pred_label": CIFAR10_CLASSES[pred],
                    "confidence": float(probs[pred]),
                }
            )

    return results


@torch.no_grad()
def plot_feature_maps(model, image, results_dir="results", prefix="", n_maps=16):
    """
    Parte 12. Aplica la imagen a la primera capa convolucional y
    guarda results/{prefix}feature_maps.png con al menos n_maps mapas.
    """
    import matplotlib.pyplot as plt

    model.eval()
    activations = model.get_first_conv_activations(image)  # (1, C, H, W)
    activations = activations.squeeze(0).cpu()

    n_maps = min(n_maps, activations.shape[0])
    cols = 4
    rows = (n_maps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()

    for i in range(n_maps):
        axes[i].imshow(activations[i], cmap="viridis")
        axes[i].set_title(f"Filtro {i}", fontsize=8)
        axes[i].axis("off")

    for i in range(n_maps, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{results_dir}/{prefix}feature_maps.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def predict_image(image: torch.Tensor, model, device=None, top_k: int = 3):
    """
    Parte 21. Recibe una imagen ya transformada (tensor 3x32x32,
    normalizada igual que en entrenamiento) y devuelve:
        - clase predicha
        - probabilidad de esa clase
        - top_k clases con sus probabilidades (para el bonus de Gradio)
    """
    device = device or get_device()
    model.eval()
    model.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    output = model(image)
    probs = F.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    top_indices = probs.argsort()[::-1][:top_k]

    return {
        "predicted_class": CIFAR10_CLASSES[pred_idx],
        "probability": float(probs[pred_idx]),
        "top_k": [(CIFAR10_CLASSES[i], float(probs[i])) for i in top_indices],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluar un modelo entrenado sobre CIFAR-10")
    parser.add_argument("--experiment", type=str, default="basic",
                         choices=["basic", "deep", "dropout", "augmentation"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    device = get_device()
    _, test_loader = get_dataloaders(data_root=args.data_root)
    test_dataset = test_loader.dataset

    model = build_model(args.experiment).to(device)
    model_path = Path(args.models_dir) / f"{args.experiment}_cifar10_cnn.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modelo cargado desde: {model_path}")

    metrics = evaluate_model(model, test_loader, device)
    plot_confusion_matrix(metrics["y_true"], metrics["y_pred"],
                           results_dir=args.results_dir, prefix=f"{args.experiment}_")
    plot_predictions(model, test_dataset, device,
                      results_dir=args.results_dir, prefix=f"{args.experiment}_")

    sample_image, _ = test_dataset[0]
    plot_feature_maps(model, sample_image, results_dir=args.results_dir, prefix=f"{args.experiment}_")

    print(f"\nGráficos guardados en '{args.results_dir}/' con prefijo '{args.experiment}_'")


if __name__ == "__main__":
    main()
