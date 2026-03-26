"""
Gram Matrix computation for Neural Style Transfer.

The Gram matrix captures feature correlations in CNN activations,
representing the style information of an image by computing the
inner product between feature maps.
"""

import tensorflow as tf


def gram_matrix(feature_map: tf.Tensor) -> tf.Tensor:
    """Compute the Gram matrix of a feature map tensor.

    The Gram matrix G is computed as G = F^T * F, normalized by the
    total number of elements. This captures the correlations between
    different filter responses, encoding texture and style information.

    Args:
        feature_map: A 4D tensor of shape (batch, height, width, channels)
                     representing CNN feature activations.

    Returns:
        Gram matrix of shape (batch, channels, channels), normalized by
        the number of spatial locations (height * width).
    """
    # feature_map shape: (batch, height, width, channels)
    # Reshape to (batch, height*width, channels)
    shape = tf.shape(feature_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    num_locations = tf.cast(height * width, tf.float32)

    # Reshape: (batch, H*W, C)
    features = tf.reshape(feature_map, (batch_size, -1, channels))

    # Gram matrix: (batch, C, C) = (batch, C, H*W) @ (batch, H*W, C)
    gram = tf.linalg.einsum("bij,bik->bjk", features, features)

    # Normalize by the number of spatial locations
    gram = gram / num_locations

    return gram
