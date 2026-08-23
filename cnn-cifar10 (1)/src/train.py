"""
train.py

Training loop para CIFAR10CNN. Soporta los 4 experimentos obligatorios
del enunciado (Parte 13) mediante un flag --experiment:

    python src/train.py --experiment basic
    python src/train.py --experiment deep
    python src/train.py --experiment dropout
    python src/train.py --experiment augmentation

Cada corrida:
    1. Entrena el modelo registrando loss/accuracy por epoch.
    2. Guarda las curvas en results/{experiment}_training_loss.png y
       results/{experiment}_training_accuracy.png.
    3. Guarda los pesos en models/{experiment}_cifar10_cnn.pth.
    4. Imprime un resumen final (para copiar a la tabla comparativa,
       Parte 19).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import build_model
from utils import get_dataloaders, get_device, save_training_curves, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_quick(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluación rápida (usada solo para imprimir progreso durante el
    entrenamiento; la evaluación formal con métricas completas vive en
    evaluate.py)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Entrenar CNN sobre CIFAR-10")
    parser.add_argument(
        "--experiment",
        type=str,
        default="basic",
        choices=["basic", "deep", "dropout", "augmentation"],
        help="Qué experimento del enunciado correr (Parte 13).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Usando device: {device}")

    augment = args.experiment == "augmentation"
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        augment=augment,
    )

    model = build_model(args.experiment).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, train_accuracies = [], []

    print(f"\n=== Experimento: {args.experiment} ===")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | Accuracy: {acc:.1f}%")

    elapsed = time.time() - start
    print(f"\nEntrenamiento completado en {elapsed/60:.1f} min")

    # Evaluación rápida sobre test (referencia; usar evaluate.py para el
    # reporte formal con precision/recall/F1/matriz de confusión).
    test_loss, test_acc = evaluate_quick(model, test_loader, criterion, device)
    print(f"Test (rápido) | Loss: {test_loss:.4f} | Accuracy: {test_acc:.1f}%")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.models_dir) / f"{args.experiment}_cifar10_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en: {model_path}")

    save_training_curves(
        train_losses, train_accuracies, results_dir=args.results_dir, prefix=f"{args.experiment}_"
    )
    print(f"Curvas guardadas en: {args.results_dir}/{args.experiment}_training_*.png")


if __name__ == "__main__":
    main()
