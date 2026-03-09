"""
Loss functions for Neural Style Transfer.

Implements content loss, style loss, and total loss computations
used to optimize the generated image.
"""

import tensorflow as tf
from typing import Dict, List

from utils.gram_matrix import gram_matrix


def content_loss(
    generated_features: tf.Tensor, content_features: tf.Tensor
) -> tf.Tensor:
    """Compute content loss between generated and content feature maps.

    Content loss is the Mean Squared Error (MSE) between the feature
    representations of the generated image and the content image,
    encouraging the generated image to preserve content structure.

    Args:
        generated_features: Feature map from the generated image.
        content_features: Feature map from the content image.

    Returns:
        Scalar tensor representing the content loss.
    """
    return tf.reduce_mean(tf.square(generated_features - content_features))


def style_loss_per_layer(
    generated_features: tf.Tensor, style_features: tf.Tensor
) -> tf.Tensor:
    """Compute style loss for a single layer using Gram matrices.

    Style loss is the MSE between the Gram matrices of the generated
    and style feature maps, encouraging similar texture and patterns.

    Args:
        generated_features: Feature map from the generated image.
        style_features: Feature map from the style image.

    Returns:
        Scalar tensor representing the style loss for this layer.
    """
    gram_gen = gram_matrix(generated_features)
    gram_style = gram_matrix(style_features)
    return tf.reduce_mean(tf.square(gram_gen - gram_style))


def style_loss(
    generated_features_dict: Dict[str, tf.Tensor],
    style_features_dict: Dict[str, tf.Tensor],
    style_layers: List[str],
    style_weights: Dict[str, float],
) -> tf.Tensor:
    """Compute total style loss across multiple layers.

    Aggregates style loss from multiple VGG19 layers, weighted by
    the per-layer style weights, to capture style at multiple scales.

    Args:
        generated_features_dict: Dict mapping layer name to generated features.
        style_features_dict: Dict mapping layer name to style features.
        style_layers: List of layer names to compute style loss for.
        style_weights: Dict mapping layer name to its weight.

    Returns:
        Scalar tensor representing the total weighted style loss.
    """
    total_style_loss = tf.constant(0.0)

    for layer_name in style_layers:
        gen_feat = generated_features_dict[layer_name]
        style_feat = style_features_dict[layer_name]
        weight = style_weights.get(layer_name, 1.0)

        layer_loss = style_loss_per_layer(gen_feat, style_feat)
        total_style_loss += weight * layer_loss

    # Average across the number of style layers
    total_style_loss /= len(style_layers)

    return total_style_loss


def total_loss(
    content_loss_val: tf.Tensor,
    style_loss_val: tf.Tensor,
    alpha: float,
    beta: float,
) -> tf.Tensor:
    """Compute total loss as a weighted combination of content and style loss.

    Total Loss = alpha * Content Loss + beta * Style Loss

    Args:
        content_loss_val: Scalar content loss.
        style_loss_val: Scalar style loss.
        alpha: Weight for content loss.
        beta: Weight for style loss.

    Returns:
        Scalar tensor representing the total loss.
    """
    return alpha * content_loss_val + beta * style_loss_val
