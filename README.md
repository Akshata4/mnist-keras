# mnist-keras

A complete MNIST handwritten digit classifier built with Keras/TensorFlow, implemented as a Jupyter notebook.

## Overview

This project trains a fully-connected (MLP) neural network on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to classify handwritten digits (0–9).

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **98.43%** |
| Test Loss | 0.0609 |
| Best Val Accuracy | 98.68% |
| Misclassified (test) | 157 / 10,000 |
| Epochs Trained | 21 |
| Trainable Parameters | 570,506 |

## Model Architecture

A `Sequential` MLP with the following layers:

```
Input (784)
  → Dense(512, relu) → BatchNorm → Dropout(0.3)
  → Dense(256, relu) → BatchNorm → Dropout(0.3)
  → Dense(128, relu) → Dropout(0.2)
  → Dense(10, softmax)
```

**Total parameters:** 570,506 (~2.18 MB)

## Training Details

- **Optimizer:** Adam (initial lr = 1e-3)
- **Loss:** Categorical cross-entropy
- **Batch size:** 128
- **Max epochs:** 30 (early stopping triggered at epoch 21)
- **Validation split:** 10% of training data
- **Callbacks:**
  - `EarlyStopping` — monitors `val_loss`, patience=5, restores best weights
  - `ReduceLROnPlateau` — halves LR on plateau (patience=3, min_lr=1e-6)

## Dataset

- **Training set:** 60,000 images (28×28 grayscale)
- **Test set:** 10,000 images
- **Classes:** 10 digits (0–9)
- **Preprocessing:** Pixels normalized to [0, 1], images flattened to 784-dim vectors, labels one-hot encoded

## Notebook Structure

| Section | Description |
|---|---|
| 1. Imports | Libraries: TensorFlow 2.20, Keras 3.13, NumPy, Matplotlib, Seaborn, scikit-learn |
| 2. Load & Explore Data | Dataset shapes, pixel range, class distribution bar charts |
| 3. Preprocess Data | Normalize, flatten, one-hot encode |
| 4. Build Model | `build_model()` function, model summary |
| 5. Compile & Train | Adam optimizer, callbacks, training loop |
| 6. Training Curves | Accuracy & loss plots over epochs |
| 7. Evaluate on Test Set | Final test loss and accuracy |
| 8. Confusion Matrix | Heatmap of predictions vs. true labels |
| 9. Classification Report | Per-class precision, recall, F1-score |
| 10. Per-Class Accuracy | Bar chart comparing accuracy per digit |
| 11. Misclassified Examples | Grid of 20 misclassified samples with true/predicted labels |
| 12. Prediction Confidence | Random test samples with softmax probability bars |
| 13. Summary | Final metrics printout |

## Per-Class Performance

All 10 digit classes achieve ≥ 98% precision, recall, and F1-score on the test set.

## Requirements

- Python 3.12+
- TensorFlow 2.20 / Keras 3.13
- NumPy, Matplotlib, Seaborn, scikit-learn

## Usage

```bash
jupyter notebook notebooks/mnist_classifier.ipynb
```
