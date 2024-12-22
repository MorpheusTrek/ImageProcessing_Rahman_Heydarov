from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

"""
Model Creation Functions
"""

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

"""
Model Training
"""

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

"""
Hyperparameter Tuning
"""

def build_model(hp):
    """
    Build a CNN model for Keras Tuner with hyperparameter tuning.

    Args:
        hp (HyperParameters): Hyperparameters for tuning.

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
    ])


    model.add(layers.Dense(hp.Int("units", 128, 1024, step=128), activation='relu'))  # Tune units in dense layer
    model.add(layers.Dense(10, activation='softmax'))

    # Hyperparameters for optimizer and learning rate
    optimizer_choice = hp.Choice("optimizer", ["adam", "sgd"])
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])

    # Choose the optimizer
    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def tune_hyperparameters(train_dataset, x_val, y_val):
    """
    Tune hyperparameters using Keras Tuner's Random Search.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        x_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.

    Returns:
        dict: Best hyperparameters.
    """
    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=20,  # Number of hyperparameter combinations to try
        executions_per_trial=1,  # Number of models to train for each trial
        directory="hyperparameter_tuning",
        project_name="cnn_tuning"
    )

    # Perform hyperparameter search
    tuner.search(
        train_dataset,
        validation_data=(x_val, y_val),
        epochs=10,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
        ]

    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_hps, best_model

"""
Model Evaluation
"""

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
