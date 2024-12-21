import tensorflow as tf

from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    """
    Load the CIFAR-10 dataset, normalize pixel values, and split into training, validation, and test sets.

    Returns:
        tuple: x_train, y_train, x_val, y_val, x_test, y_test
        - x_train: Training images (numpy array).
        - y_train: Training labels (numpy array).
        - x_val: Validation images (numpy array).
        - y_val: Validation labels (numpy array).
        - x_test: Test images (numpy array).
        - y_test: Test labels (numpy array).
    """
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Split the training set into a training and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def augment_image(image, label):
    """
    Apply random augmentations to an image.

    Args:
        image (tf.Tensor): Input image tensor.
        label (tf.Tensor): Corresponding label tensor.

    Returns:
        tuple: Augmented image and label.
    """
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.2)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)  # Pad to 40x40
    image = tf.image.random_crop(image, size=[32, 32, 3])  # Random crop back to 32x32
    return image, label

def augment_dataset(x_train, y_train):
    """
    Create a TensorFlow dataset with augmentation, shuffling, batching, and prefetching.

    Args:
        x_train (numpy array): Training images.
        y_train (numpy array): Training labels.

    Returns:
        tf.data.Dataset: Augmented training dataset.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset
        .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)  # Apply augmentation
        .shuffle(buffer_size=5000)  # Shuffle the dataset
        .batch(64)  # Batch size
        .prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch for performance
    )
    return train_dataset
