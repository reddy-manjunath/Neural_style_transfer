"""
Style Transfer Model using VGG19.

Builds a multi-output feature extractor from a pretrained VGG19 network
to extract content and style representations from images.
"""

import logging
from typing import Dict, List

import tensorflow as tf

logger = logging.getLogger(__name__)


class StyleTransferModel:
    """VGG19-based feature extractor for Neural Style Transfer.

    Loads a pretrained VGG19 model (ImageNet weights) and constructs
    a new model that outputs intermediate feature maps from specified
    content and style layers. All VGG19 weights are frozen.

    VGG19 layer mapping (Keras names):
        - conv1_1 → block1_conv1
        - conv2_1 → block2_conv1
        - conv3_1 → block3_conv1
        - conv4_1 → block4_conv1
        - conv4_2 → block4_conv2  (content layer)
        - conv5_1 → block5_conv1

    Attributes:
        content_layers: List of layer names for content extraction.
        style_layers: List of layer names for style extraction.
        model: A tf.keras.Model that returns a dict of features.
    """

    # Mapping from common notation to Keras VGG19 layer names
    LAYER_MAP = {
        "conv1_1": "block1_conv1",
        "conv2_1": "block2_conv1",
        "conv3_1": "block3_conv1",
        "conv4_1": "block4_conv1",
        "conv4_2": "block4_conv2",
        "conv5_1": "block5_conv1",
    }

    def __init__(
        self,
        content_layers: List[str],
        style_layers: List[str],
    ) -> None:
        """Initialize the style transfer feature extractor.

        Args:
            content_layers: Layer names for content feature extraction
                           (Keras naming convention, e.g., 'block4_conv2').
            style_layers: Layer names for style feature extraction
                         (Keras naming convention, e.g., 'block1_conv1').
        """
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.model = self._build_model()

        logger.info(
            f"StyleTransferModel initialized with "
            f"content_layers={content_layers}, style_layers={style_layers}"
        )

    def _build_model(self) -> tf.keras.Model:
        """Build a multi-output VGG19 feature extractor.

        Loads VGG19 pretrained on ImageNet (without the classifier head),
        freezes all layers, and constructs a model that returns feature
        maps from the specified content and style layers.

        Returns:
            A tf.keras.Model with multiple outputs.
        """
        # Load pretrained VGG19 without the top classification layers
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet"
        )

        # Freeze all weights — we only use VGG as a feature extractor
        vgg.trainable = False

        # Collect output layers
        all_layers = list(set(self.content_layers + self.style_layers))

        # Validate layer names
        available_layers = {layer.name for layer in vgg.layers}
        for layer_name in all_layers:
            if layer_name not in available_layers:
                raise ValueError(
                    f"Layer '{layer_name}' not found in VGG19. "
                    f"Available layers: {sorted(available_layers)}"
                )

        # Build outputs dict
        outputs = {
            layer_name: vgg.get_layer(layer_name).output
            for layer_name in all_layers
        }

        # Create multi-output model
        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False

        logger.info(f"VGG19 feature extractor built with {len(all_layers)} output layers")

        return model

    def extract_features(self, image: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Extract feature maps from an image.

        Args:
            image: A preprocessed 4D tensor of shape (1, H, W, 3).

        Returns:
            Dict mapping layer names to their feature map tensors.
        """
        return self.model(image)

    def get_content_features(
        self, features: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Filter features to return only content layer outputs.

        Args:
            features: Full feature dict from extract_features().

        Returns:
            Dict with only content layer features.
        """
        return {
            name: features[name]
            for name in self.content_layers
        }

    def get_style_features(
        self, features: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Filter features to return only style layer outputs.

        Args:
            features: Full feature dict from extract_features().

        Returns:
            Dict with only style layer features.
        """
        return {
            name: features[name]
            for name in self.style_layers
        }
