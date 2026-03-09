"""
Image utility functions for Neural Style Transfer.

Provides functions to convert tensors back to displayable/saveable
images and to save output images to disk.
"""

import os
import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet mean values used by VGG preprocessing
IMAGENET_MEAN = np.array([103.939, 116.779, 123.68])


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    """Convert a preprocessed VGG tensor back to a PIL Image.

    Reverses VGG19 preprocessing (BGR channel order, ImageNet mean
    subtraction) and clamps pixel values to [0, 255].

    Args:
        tensor: A 4D tensor of shape (1, H, W, 3) in VGG-preprocessed format.

    Returns:
        A PIL Image in RGB format.
    """
    # Remove batch dimension
    img = tensor.numpy().copy()
    if img.ndim == 4:
        img = img[0]

    # Reverse VGG preprocessing: add back ImageNet mean
    img += IMAGENET_MEAN

    # VGG uses BGR channel order, convert back to RGB
    img = img[:, :, ::-1]

    # Clamp values to valid pixel range
    img = np.clip(img, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


def save_image(tensor: tf.Tensor, path: str) -> None:
    """Save a tensor as an image file.

    Args:
        tensor: A 4D tensor of shape (1, H, W, 3) in VGG-preprocessed format.
        path: File path to save the image to.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    img = tensor_to_image(tensor)
    img.save(path)
    logger.info(f"Image saved to {path}")


def display_image(tensor: tf.Tensor, title: Optional[str] = None) -> None:
    """Display a tensor as an image using matplotlib.

    Args:
        tensor: A 4D tensor of shape (1, H, W, 3) in VGG-preprocessed format.
        title: Optional title for the displayed image.
    """
    import matplotlib.pyplot as plt

    img = tensor_to_image(tensor)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if title:
        plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
