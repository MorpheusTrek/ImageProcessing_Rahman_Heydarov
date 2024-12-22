
# CIFAR-10 Image Classification Pipeline

This repository provides a comprehensive set of utilities for building, training, and evaluating convolutional neural networks (CNNs) for the CIFAR-10 image classification task. The pipeline is modular, featuring dataset preprocessing, model creation, training, and evaluation utilities.

---

## 🚀 Features

### 📂 **Dataset Utilities**
- Load, preprocess, and augment the CIFAR-10 dataset.
- Normalize images and split data into training, validation, and test sets.
- Perform random image augmentations such as:
  - Flipping
  - Brightness adjustment
  - Contrast adjustment
  - Cropping

### 🛠️ **Model Utilities**
- Build simple and regularized CNN architectures.
- Optimize models using hyperparameter tuning with Keras Tuner.
- Train models using advanced techniques like:
  - Early stopping
  - Learning rate reduction

### 📊 **Evaluation Utilities**
- Visualize training history with accuracy and loss plots.
- Generate detailed classification reports and confusion matrices.
- Visualize model architectures with layered diagrams.

---

## 📁 Repository Structure

| File Name         | Description                                             |
|-------------------|---------------------------------------------------------|
| `dataset_utils.py` | Functions for loading, preprocessing, and augmenting datasets. |
| `model_utils.py`   | Functions for creating and training CNN models, including regularization and hyperparameter tuning. |
| `eval_utils.py`    | Functions for evaluating model performance and visualizing results. |

---

## 🛠️ Getting Started

For ease of use, a Jupyter Notebook, **evaluation.ipynb**, is included. It provides:
- Step-by-step instructions to load preprocessed data.
- Easy-to-use cells for creating, training, and evaluating models.
- Visualizations of training history, confusion matrices, and classification reports.

Simply follow the notebook's cells in order, or dive into the scripts for more detailed control.

---

### 🔧 Prerequisites

Make sure you have Python installed along with the required libraries:
- **TensorFlow**
- **Keras Tuner**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **Seaborn**
- **VisualKeras**

Install dependencies using:
```bash
pip install tensorflow keras-tuner numpy scikit-learn matplotlib seaborn visualkeras
```

---

### ⚡ Running the Pipeline

#### **1. Load and Preprocess Data**
- Use `load_and_preprocess_data` to load and split the CIFAR-10 dataset into training, validation, and test sets.
- Use `augment_dataset` to apply data augmentation techniques.

#### **2. Create a Model**
Choose or customize a model from `model_utils.py`:
- `create_model`: Basic CNN.
- `create_regularized_model`: CNN with regularization (BatchNormalization, Dropout).
- `create_hypermodel`: For hyperparameter tuning.

#### **3. Train the Model**
Train your model using the following options:
- `train_model`: Standard training.
- `train_regularized_model`: Advanced training with early stopping and learning rate reduction.

#### **4. Evaluate the Model**
Evaluate your trained model using functions from `eval_utils.py`:
- `evaluation_report`: Generates a detailed classification report.
- `evaluation_matrix`: Displays a confusion matrix as a heatmap.

#### **5. Visualize Results**
- Use `plot_history` to view training performance (accuracy and loss plots).
- Use `visualize_model` to generate a diagram of your model's architecture.

---

## 📌 Example Workflow

Here’s a quick example of how to use the pipeline:

```python
from dataset_utils import load_and_preprocess_data, augment_dataset
from model_utils import create_model, train_model
from eval_utils import plot_history, evaluation_report

# Step 1: Load and preprocess data
x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()

# Step 2: Augment the training dataset
train_dataset = augment_dataset(x_train, y_train)

# Step 3: Create a CNN model
model = create_model()

# Step 4: Train the model
history = train_model(model, train_dataset, x_val, y_val)

# Step 5: Evaluate the model
evaluation_report(model, x_test, y_test)

# Step 6: Visualize training history
plot_history(history)
```
---
### ⚠️ Disclaimer

This repository's documentation, comments, and docstrings have been refined with the assistance of a large language model (LLM) to ensure clarity and readability.
