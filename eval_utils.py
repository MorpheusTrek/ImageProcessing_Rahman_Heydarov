import matplotlib.pyplot as plt
import numpy as np
import visualkeras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_history(history):
    """
    Plot the training and validation accuracy and loss curves from the model's training history.

    Args:
        history (tf.keras.callbacks.History): Training history object containing metrics for each epoch.

    Returns:
        None: Displays the plots for accuracy and loss.
    """
    plt.figure(figsize=(12, 4))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

def visualize_model(model):
    """
    Visualize the architecture of a given model using a layered diagram.

    Args:
        model (tf.keras.Model): The Keras model to visualize.

    Returns:
        PIL.Image.Image: An image representation of the model architecture.
    """
    return visualkeras.layered_view(model, legend=True)

# Class names for Cifar-10 Dataset
cifar_class_names = [
    'airplane', 'automobile', 'bird', 'cat', 
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

def evaluation_report(model, x_test, y_test, class_names = cifar_class_names):
    """
    Generate and display a classification report for the model's predictions.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        x_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): True labels for the test data.
        class_names (list): List of class names corresponding to label indices.

    Returns:
        None: Prints the classification report to the console.
    """
    # Predict class labels for the test dataset
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Generate a classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("Classification Report:\n", report)

def evaluation_matrix(model, x_test, y_test, class_names = cifar_class_names):
    """
    Compute and visualize the confusion matrix for the model's predictions.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        x_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): True labels for the test data.
        class_names (list): List of class names corresponding to label indices.

    Returns:
        None: Displays a heatmap of the confusion matrix.
    """
    # Compute the confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
