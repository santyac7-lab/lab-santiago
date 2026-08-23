"""
model.py

Arquitecturas CNN para clasificación de CIFAR-10.

Se define una única clase configurable (CIFAR10CNN) que cubre los
experimentos obligatorios del proyecto:

    Experimento 1 - CNN básica        -> CIFAR10CNN(variant="basic")
    Experimento 2 - CNN más profunda  -> CIFAR10CNN(variant="deep")
    Experimento 3 - CNN + Dropout     -> CIFAR10CNN(variant="basic", dropout=0.5)
    Experimento 4 - Data Augmentation -> se aplica en los transforms (ver utils.py),
                                          no cambia la arquitectura.

Todas las variantes reciben tensores de entrada (3, 32, 32) y devuelven
logits de tamaño (10,) — una por clase de CIFAR-10.
"""

from __future__ import annotations

import torch
import torch.nn as nn

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


class CIFAR10CNN(nn.Module):
    """
    CNN configurable para CIFAR-10.

    Parameters
    ----------
    variant : {"basic", "deep"}
        "basic": 3 bloques Conv-ReLU-MaxPool (32 -> 64 -> 128 filtros).
        "deep":  igual que "basic" pero con dos convoluciones por bloque
                 (Conv-ReLU-Conv-ReLU-MaxPool), lo que aumenta la
                 profundidad y el receptive field sin cambiar la
                 resolución espacial de salida.
    dropout : float
        Probabilidad de Dropout aplicada antes de la capa densa final.
        0.0 desactiva el Dropout (comportamiento del Experimento 1/2).
    num_classes : int
        Número de clases de salida (10 para CIFAR-10).
    """

    def __init__(self, variant: str = "basic", dropout: float = 0.0, num_classes: int = 10):
        super().__init__()

        if variant not in ("basic", "deep"):
            raise ValueError(f"variant debe ser 'basic' o 'deep', recibido: {variant}")

        self.variant = variant
        self.dropout_p = dropout

        if variant == "basic":
            self.features = nn.Sequential(
                # Bloque 1: 3 x 32 x 32 -> 32 x 16 x 16
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Bloque 2: 32 x 16 x 16 -> 64 x 8 x 8
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Bloque 3: 64 x 8 x 8 -> 128 x 4 x 4
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            flat_features = 128 * 4 * 4

        else:  # variant == "deep"
            self.features = nn.Sequential(
                # Bloque 1: 3 x 32 x 32 -> 32 x 16 x 16
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Bloque 2: 32 x 16 x 16 -> 64 x 8 x 8
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Bloque 3: 64 x 8 x 8 -> 128 x 4 x 4
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            flat_features = 128 * 4 * 4

        classifier_layers = [nn.Flatten(), nn.Linear(flat_features, 256), nn.ReLU(inplace=True)]

        if dropout > 0:
            classifier_layers.append(nn.Dropout(p=dropout))

        classifier_layers.append(nn.Linear(256, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_first_conv_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Devuelve los feature maps de la PRIMERA capa convolucional
        (antes de ReLU/Pool). Útil para la Parte 12 (visualización de
        feature maps). Espera x con shape (1, 3, 32, 32) o (3, 32, 32).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        device = next(self.parameters()).device
        x = x.to(device)

        first_conv = self.features[0]
        with torch.no_grad():
            return first_conv(x)


def build_model(experiment: str, num_classes: int = 10) -> CIFAR10CNN:
    """
    Factory que construye el modelo correspondiente a cada experimento
    obligatorio del enunciado (Parte 13).

    experiment: {"basic", "deep", "dropout", "augmentation"}
        "basic"        -> CNN básica (Experimento 1)
        "deep"         -> CNN más profunda (Experimento 2)
        "dropout"      -> CNN básica + Dropout 0.5 (Experimento 3)
        "augmentation" -> misma arquitectura que "basic"; el cambio real
                          está en los transforms de entrenamiento
                          (ver utils.get_train_transform(augment=True))
    """
    mapping = {
        "basic": dict(variant="basic", dropout=0.0),
        "deep": dict(variant="deep", dropout=0.0),
        "dropout": dict(variant="basic", dropout=0.5),
        "augmentation": dict(variant="basic", dropout=0.0),
    }
    if experiment not in mapping:
        raise ValueError(f"Experimento desconocido: {experiment}. Usa uno de {list(mapping)}")

    return CIFAR10CNN(num_classes=num_classes, **mapping[experiment])


if __name__ == "__main__":
    # Sanity check rápido: verifica que las dimensiones fluyen como se
    # documenta en la Parte 4 del enunciado.
    for exp in ["basic", "deep", "dropout", "augmentation"]:
        model = build_model(exp)
        dummy = torch.randn(2, 3, 32, 32)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{exp:12s}] output shape: {tuple(out.shape)} | parámetros: {n_params:,}")
