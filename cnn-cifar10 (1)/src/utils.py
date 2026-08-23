"""
utils.py

Funciones compartidas por el notebook, train.py y evaluate.py:

- Carga de CIFAR-10 y DataLoaders.
- Transforms (con y sin Data Augmentation).
- Des-normalización de imágenes para visualizar.
- Guardado de curvas de entrenamiento (loss / accuracy).
- Fijar semillas para reproducibilidad.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def set_seed(seed: int = 42) -> None:
    """Fija semillas para numpy / torch / random (reproducibilidad)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Devuelve GPU si está disponible, si no CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_transform(augment: bool = False) -> transforms.Compose:
    """
    Transform de entrenamiento.

    augment=False -> solo ToTensor + Normalize (Experimentos 1, 2, 3).
    augment=True  -> RandomHorizontalFlip + RandomCrop (Experimento 4:
                      Data Augmentation).
    """
    if augment:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def get_test_transform() -> transforms.Compose:
    """El conjunto de test NUNCA lleva augmentation, solo normalización."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def get_dataloaders(
    data_root: str = "./data",
    batch_size: int = 64,
    augment: bool = False,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Descarga (si hace falta) CIFAR-10 y devuelve (train_loader, test_loader)."""
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=get_train_transform(augment=augment),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=get_test_transform(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convierte un tensor normalizado (3, H, W) de vuelta a un array
    (H, W, 3) en rango [0, 1] listo para matplotlib.
    """
    img = img_tensor.clone().detach().cpu()
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def save_training_curves(
    train_losses: list,
    train_accuracies: list,
    results_dir: str = "results",
    prefix: str = "",
) -> None:
    """
    Genera y guarda:
        results/{prefix}training_loss.png
        results/{prefix}training_accuracy.png

    Importa matplotlib de forma perezosa para que utils.py se pueda
    importar en entornos sin backend gráfico configurado.
    """
    import matplotlib.pyplot as plt

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, marker="o")
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accuracies, marker="o", color="green")
    plt.title("Training Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}training_accuracy.png", dpi=150)
    plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
