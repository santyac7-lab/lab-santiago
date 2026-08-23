# Clasificación de imágenes con CNN — CIFAR-10

## 1. Problema

Prototipo de visión artificial capaz de reconocer automáticamente el
tipo de objeto presente en una imagen, usando el dataset público
**CIFAR-10** (10 categorías: airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck).

## 2. Dataset

- **CIFAR-10**: 60,000 imágenes RGB de 32x32 píxeles, 10 clases
  balanceadas (5,000 train + 1,000 test por clase).
- Descarga automática vía `torchvision.datasets.CIFAR10`.
- Normalización: `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)`.
- Data Augmentation (Experimento 4): `RandomHorizontalFlip` +
  `RandomCrop(32, padding=4)`.

## 3. Arquitectura

Definida en [`src/model.py`](src/model.py), clase `CIFAR10CNN`
configurable en dos variantes:

**Variante "basic"** (3 bloques Conv-ReLU-MaxPool):

```text
INPUT 3x32x32
Conv2D 32 filtros (3x3, pad 1) -> ReLU -> MaxPool 2x2   => 32x16x16
Conv2D 64 filtros (3x3, pad 1) -> ReLU -> MaxPool 2x2   => 64x8x8
Conv2D 128 filtros (3x3, pad 1) -> ReLU -> MaxPool 2x2  => 128x4x4
Flatten -> 2048
Dense 256 -> ReLU
Dense 10
```

**Variante "deep"**: igual estructura pero con dos convoluciones por
bloque (Conv-ReLU-Conv-ReLU-MaxPool), aumentando la profundidad y el
receptive field sin cambiar la resolución espacial de salida.

Dropout (0.5) se agrega opcionalmente antes de la capa `Dense 10`
(Experimento 3).

## 4. Entrenamiento

| Parámetro | Valor |
|---|---|
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

Entrenar cualquiera de los 4 experimentos:

```bash
python src/train.py --experiment basic
python src/train.py --experiment deep
python src/train.py --experiment dropout
python src/train.py --experiment augmentation
```

Esto genera automáticamente:
- `models/{experimento}_cifar10_cnn.pth`
- `results/{experimento}_training_loss.png`
- `results/{experimento}_training_accuracy.png`

Evaluar un experimento ya entrenado:

```bash
python src/evaluate.py --experiment basic
```

Esto genera:
- `results/{experimento}_confusion_matrix.png`
- `results/{experimento}_predictions.png`
- `results/{experimento}_feature_maps.png`
- Reporte de accuracy / precision / recall / F1 en consola

## 5. Resultados

*(Completar tras entrenar — copiar de la salida de `evaluate.py` o del
notebook.)*

| Modelo | Train Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CNN básica | | | | | |
| CNN profunda | | | | | |
| CNN + Dropout | | | | | |
| CNN + Data Augmentation | | | | | |

**Mejor modelo:** *(completar y justificar)*

## 6. Gráficos

- `results/*_training_loss.png` — Loss por epoch
- `results/*_training_accuracy.png` — Accuracy por epoch
- `results/*_confusion_matrix.png` — Matriz de confusión
- `results/*_predictions.png` — 5 predicciones correctas + 5 incorrectas
- `results/*_feature_maps.png` — Feature maps de la primera capa convolucional

## 7. Experimentos

1. **CNN básica**: arquitectura de referencia (3 bloques Conv-Pool).
2. **CNN más profunda**: doble convolución por bloque.
3. **CNN + Dropout**: Dropout(0.5) antes de la capa de salida, para
   reducir overfitting.
4. **CNN + Data Augmentation**: `RandomHorizontalFlip` + `RandomCrop`
   sobre el set de entrenamiento (nunca sobre test).

*(Completar el análisis comparativo de impacto de cada experimento.)*

## 8. Análisis de errores

Ver `notebooks/01_cifar10_exploracion_entrenamiento.ipynb`, sección
"Parte 11", con 10 imágenes mal clasificadas, su confianza (softmax) y
el análisis cualitativo (ambigüedad, objeto pequeño, fondo, resolución,
similitud entre clases).

## 9. Conclusiones

*(Completar tras correr los 4 experimentos: qué tanto ayudó cada
técnica, qué clases resultaron más difíciles y por qué, y qué se
intentaría a continuación — p. ej. Transfer Learning con ResNet18.)*

## 10. Cómo reproducir

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Entrenar los 4 experimentos
python src/train.py --experiment basic
python src/train.py --experiment deep
python src/train.py --experiment dropout
python src/train.py --experiment augmentation

# Evaluar cada uno
python src/evaluate.py --experiment basic
python src/evaluate.py --experiment deep
python src/evaluate.py --experiment dropout
python src/evaluate.py --experiment augmentation

# O abrir el notebook interactivo
jupyter notebook notebooks/01_cifar10_exploracion_entrenamiento.ipynb
```

## Estructura del repositorio

```text
cnn-cifar10/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_cifar10_exploracion_entrenamiento.ipynb
├── src/
│   ├── model.py        # Arquitecturas CNN (basic / deep / dropout)
│   ├── train.py        # Training loop + CLI para los 4 experimentos
│   ├── evaluate.py      # Métricas, matriz de confusión, feature maps, inferencia
│   └── utils.py         # DataLoaders, transforms, plotting, seeds
├── results/              # Gráficos generados (loss, accuracy, matriz, etc.)
└── models/               # Pesos entrenados (.pth)
```
