"""
Image preprocessing pipeline for VGG19.

Provides functions to preprocess images for VGG19 input (ImageNet
normalization) and to reverse the preprocessing for visualization.
"""

import numpy as np
import tensorflow as tf


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """Preprocess an image tensor for VGG19 input.

    Applies the standard VGG19/ImageNet preprocessing:
    - Converts RGB to BGR channel order.
    - Subtracts ImageNet channel means.

    This uses tf.keras.applications.vgg19.preprocess_input which
    expects input in [0, 255] range with RGB channel order.

    Args:
        image: A 4D float32 tensor of shape (1, H, W, 3).
               Values should be in [0, 255] with RGB channel order.

    Returns:
        A preprocessed 4D float32 tensor suitable for VGG19 input.
    """
    return tf.keras.applications.vgg19.preprocess_input(
        tf.identity(image)
    )


def deprocess_image(image: tf.Tensor) -> np.ndarray:
    """Reverse VGG19 preprocessing to get a displayable image.

    Reverses the preprocessing:
    - Adds back ImageNet channel means.
    - Converts BGR back to RGB channel order.
    - Clips values to [0, 255].

    Args:
        image: A 4D tensor in VGG-preprocessed format (1, H, W, 3).

    Returns:
        A numpy array of shape (H, W, 3) in uint8 RGB format.
    """
    img = image.numpy().copy()
    if img.ndim == 4:
        img = img[0]

    # ImageNet mean values (BGR order as used by VGG preprocessing)
    imagenet_mean = np.array([103.939, 116.779, 123.68])

    # Add back the mean
    img += imagenet_mean

    # Convert BGR to RGB
    img = img[:, :, ::-1]

    # Clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img
