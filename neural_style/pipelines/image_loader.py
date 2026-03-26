"""
Image loading pipeline for Neural Style Transfer.

Handles loading images from file paths, resizing to target dimensions,
and converting to tensors suitable for VGG19 preprocessing.
"""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(
    path: str,
    max_size: int = 512,
    target_shape: Tuple[int, int] = None,
) -> tf.Tensor:
    """Load an image from disk and prepare it as a tensor.

    Loads the image, resizes it while preserving aspect ratio
    (unless a specific target_shape is given), and returns a
    float32 tensor ready for VGG19 preprocessing.

    Args:
        path: Path to the image file.
        max_size: Maximum dimension (height or width). Supports up to 1024.
        target_shape: Optional (height, width) to force exact size. If provided,
                      max_size is ignored.

    Returns:
        A 4D float32 tensor of shape (1, H, W, 3) with values in [0, 255].

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    # Validate max_size
    max_size = min(max_size, 1024)

    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {path}")
    except Exception as e:
        raise ValueError(f"Failed to load image from {path}: {e}")

    logger.info(f"Loaded image from {path} — original size: {img.size}")

    if target_shape is not None:
        # Resize to exact target shape (height, width)
        img = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
    else:
        # Resize preserving aspect ratio
        width, height = img.size
        scale = max_size / max(width, height)
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

    logger.info(f"Resized image to: {img.size}")

    # Convert to numpy array and then to tensor
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension: (H, W, 3) → (1, H, W, 3)
    img_tensor = tf.expand_dims(img_array, axis=0)

    return img_tensor


def load_image_pair(
    content_path: str,
    style_path: str,
    max_size: int = 512,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load content and style images, resizing style to match content dimensions.

    Args:
        content_path: Path to the content image.
        style_path: Path to the style image.
        max_size: Maximum dimension for the content image.

    Returns:
        Tuple of (content_tensor, style_tensor), both with matching
        spatial dimensions and shape (1, H, W, 3).
    """
    content = load_image(content_path, max_size=max_size)

    # Get content dimensions to resize style image to match
    _, h, w, _ = content.shape
    style = load_image(style_path, target_shape=(h, w))

    logger.info(
        f"Image pair loaded — content: {content.shape}, style: {style.shape}"
    )

    return content, style
