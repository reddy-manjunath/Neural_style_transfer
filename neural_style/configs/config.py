"""
Configuration module for Neural Style Transfer.

Contains all hyperparameters and default settings for the style transfer pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class StyleTransferConfig:
    """Configuration dataclass for Neural Style Transfer parameters.

    Attributes:
        image_size: Maximum dimension for input images (supports up to 1024).
        iterations: Number of optimization iterations.
        learning_rate: Learning rate for the Adam optimizer.
        alpha: Weight for content loss.
        beta: Weight for style loss.
        save_every: Save intermediate results every N iterations.
        content_layers: VGG19 layers used for content feature extraction.
        style_layers: VGG19 layers used for style feature extraction.
        style_weights: Weight for each style layer (should sum to 1.0).
        output_dir: Directory to save output images.
        init_mode: Initialization mode for the generated image ('content' or 'random').
    """

    # Image settings
    image_size: int = 512

    # Optimization settings
    iterations: int = 400
    learning_rate: float = 5.0
    alpha: float = 1.0
    beta: float = 1e4
    save_every: int = 50

    # VGG19 layer names (Keras naming convention)
    content_layers: List[str] = field(
        default_factory=lambda: ["block4_conv2"]
    )

    style_layers: List[str] = field(
        default_factory=lambda: [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
    )

    style_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "block1_conv1": 1.0,
            "block2_conv1": 0.8,
            "block3_conv1": 0.5,
            "block4_conv1": 0.3,
            "block5_conv1": 0.1,
        }
    )

    # Output settings
    output_dir: str = "outputs"
    init_mode: str = "content"  # 'content' or 'random'

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.image_size < 64 or self.image_size > 1024:
            raise ValueError(
                f"image_size must be between 64 and 1024, got {self.image_size}"
            )
        if self.iterations < 1:
            raise ValueError(
                f"iterations must be >= 1, got {self.iterations}"
            )
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.init_mode not in ("content", "random"):
            raise ValueError(
                f"init_mode must be 'content' or 'random', got '{self.init_mode}'"
            )
