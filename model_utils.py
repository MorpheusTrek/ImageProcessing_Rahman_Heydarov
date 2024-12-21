from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_model():
    """
    Create a simple CNN model for image classification.

    The model consists of:
    - Four convolutional blocks, each with:
        - Two Conv2D layers with ReLU activation and same padding.
        - A MaxPooling2D layer to reduce spatial dimensions.
    - A Flatten layer to convert the 2D feature maps into a 1D vector.
    - A fully connected Dense layer for feature representation.
    - An output Dense layer with softmax activation for 10-class classification.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"),
        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256),
        layers.Dense(10, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_regularized_model():
    """
    Create a regularized CNN model for image classification.

    The model includes:
    - Four convolutional blocks, each with:
        - Two Conv2D layers with ReLU activation and same padding.
        - BatchNormalization for regularization.
        - A MaxPooling2D layer to reduce spatial dimensions.
        - Dropout for further regularization.
    - A GlobalAveragePooling2D layer for feature extraction.
    - A fully connected Dense layer with L2 regularization and ReLU activation.
    - An output Dense layer with softmax activation for 10-class classification.

    Returns:
        tf.keras.Model: Compiled regularized CNN model.
    """
    l2_reg = regularizers.l2(0.01)

    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"),
        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Fully Connected Layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=l2_reg),
        layers.Dense(10, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def train_model(model, train_dataset, x_val, y_val):
    """
    Train the model on the training dataset and validate on the validation set.

    Args:
        model (tf.keras.Model): The model to be trained.
        train_dataset (tf.data.Dataset): The training dataset.
        x_val (np.ndarray): Validation set images.
        y_val (np.ndarray): Validation set labels.

    Returns:
        tf.keras.callbacks.History: Training history containing loss and accuracy metrics.
    """
    history = model.fit(train_dataset, epochs=50, validation_data=(x_val, y_val))
    return history

def train_regularized_model(model, train_dataset, x_val, y_val):
    """
    Train the model with regularization and callbacks for early stopping and learning rate reduction.

    Args:
        model (tf.keras.Model): The model to be trained.
        train_dataset (tf.data.Dataset): The training dataset.
        x_val (np.ndarray): Validation set images.
        y_val (np.ndarray): Validation set labels.

    Callbacks:
        - EarlyStopping: Stops training when validation loss does not improve for a specified number of epochs.
        - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.

    Returns:
        tf.keras.callbacks.History: Training history containing loss and accuracy metrics.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    history = model.fit(train_dataset, epochs=50, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])
    return history

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        x_test (np.ndarray): Test set images.
        y_test (np.ndarray): Test set labels.

    Returns:
        None: Prints the test loss and accuracy metrics.
    """
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
