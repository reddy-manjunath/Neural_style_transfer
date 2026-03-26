"""
Optimization pipeline for Neural Style Transfer.

Runs the iterative optimization loop that generates a stylized image
by minimizing content and style losses using gradient descent.
"""

import os
import time
import logging
from typing import Optional

import tensorflow as tf
import numpy as np

from configs.config import StyleTransferConfig
from models.style_transfer_model import StyleTransferModel
from pipelines.image_loader import load_image_pair
from pipelines.preprocessing import preprocess_image
from utils.loss_functions import content_loss, style_loss, total_loss
from utils.image_utils import save_image

logger = logging.getLogger(__name__)


def initialize_generated_image(
    content_image: tf.Tensor, mode: str = "content"
) -> tf.Variable:
    """Initialize the generated image as a trainable variable.

    Args:
        content_image: The preprocessed content image tensor.
        mode: 'content' to initialize from the content image,
              'random' to initialize with random noise.

    Returns:
        A tf.Variable representing the generated image to optimize.
    """
    if mode == "content":
        generated = tf.Variable(content_image, dtype=tf.float32)
    elif mode == "random":
        noise = tf.random.normal(content_image.shape, mean=0.0, stddev=50.0)
        generated = tf.Variable(noise, dtype=tf.float32)
    else:
        raise ValueError(f"Unknown init_mode: '{mode}'. Use 'content' or 'random'.")

    logger.info(f"Generated image initialized with mode='{mode}'")
    return generated


@tf.function
def _train_step(
    generated_image: tf.Variable,
    model: tf.keras.Model,
    content_targets: dict,
    style_targets: dict,
    content_layers: list,
    style_layers: list,
    style_weights: dict,
    alpha: float,
    beta: float,
    optimizer: tf.optimizers.Optimizer,
) -> tf.Tensor:
    """Perform a single optimization step.

    Uses tf.GradientTape to compute gradients of the total loss
    with respect to the generated image pixels.

    Args:
        generated_image: The tf.Variable to optimize.
        model: The VGG19 feature extractor model.
        content_targets: Target content features.
        style_targets: Target style features.
        content_layers: List of content layer names.
        style_layers: List of style layer names.
        style_weights: Per-layer style weights.
        alpha: Content loss weight.
        beta: Style loss weight.
        optimizer: The optimizer to use.

    Returns:
        The total loss value for this step.
    """
    with tf.GradientTape() as tape:
        # Extract features from the generated image
        generated_features = model(generated_image)

        # Compute content loss
        c_loss = tf.constant(0.0)
        for layer_name in content_layers:
            c_loss += content_loss(
                generated_features[layer_name],
                content_targets[layer_name],
            )

        # Compute style loss
        s_loss = style_loss(
            generated_features,
            style_targets,
            style_layers,
            style_weights,
        )

        # Compute total loss
        t_loss = total_loss(c_loss, s_loss, alpha, beta)

    # Compute and apply gradients
    gradients = tape.gradient(t_loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])

    return t_loss


def run_style_transfer(
    content_path: str,
    style_path: str,
    config: Optional[StyleTransferConfig] = None,
    output_path: Optional[str] = None,
) -> tf.Tensor:
    """Run the full Neural Style Transfer optimization pipeline.

    Loads content and style images, extracts target features,
    initializes the generated image, and iteratively optimizes
    it to minimize the combined content and style loss.

    Args:
        content_path: Path to the content image.
        style_path: Path to the style image.
        config: Configuration object. Uses defaults if None.
        output_path: Path to save the final output image.
                     Defaults to config.output_dir/stylized_output.jpg.

    Returns:
        The final stylized image tensor of shape (1, H, W, 3).
    """
    if config is None:
        config = StyleTransferConfig()

    config.validate()

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(config.output_dir, "stylized_output.jpg")

    logger.info("=" * 60)
    logger.info("Neural Style Transfer — Starting Optimization")
    logger.info("=" * 60)
    logger.info(f"Content image: {content_path}")
    logger.info(f"Style image:   {style_path}")
    logger.info(f"Iterations:    {config.iterations}")
    logger.info(f"Alpha:         {config.alpha}")
    logger.info(f"Beta:          {config.beta}")
    logger.info(f"Image size:    {config.image_size}")
    logger.info(f"Output:        {output_path}")
    logger.info("=" * 60)

    # ── Step 1: Load images ─────────────────────────────────────
    logger.info("Step 1/5: Loading images...")
    content_tensor, style_tensor = load_image_pair(
        content_path, style_path, max_size=config.image_size
    )

    # ── Step 2: Preprocess for VGG19 ────────────────────────────
    logger.info("Step 2/5: Preprocessing images for VGG19...")
    content_processed = preprocess_image(content_tensor)
    style_processed = preprocess_image(style_tensor)

    # ── Step 3: Build model and extract target features ─────────
    logger.info("Step 3/5: Building VGG19 feature extractor...")
    model = StyleTransferModel(
        content_layers=config.content_layers,
        style_layers=config.style_layers,
    )

    content_features = model.extract_features(content_processed)
    style_features = model.extract_features(style_processed)

    # Separate content and style targets
    content_targets = model.get_content_features(content_features)
    style_targets = model.get_style_features(style_features)

    # ── Step 4: Initialize generated image ──────────────────────
    logger.info("Step 4/5: Initializing generated image...")
    generated_image = initialize_generated_image(
        content_processed, mode=config.init_mode
    )

    # ── Step 5: Optimize ────────────────────────────────────────
    logger.info("Step 5/5: Running optimization loop...")
    optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)

    # Convert Python lists/dicts to work with tf.function
    content_layers_list = config.content_layers
    style_layers_list = config.style_layers

    start_time = time.time()

    for iteration in range(1, config.iterations + 1):
        loss_value = _train_step(
            generated_image=generated_image,
            model=model.model,
            content_targets=content_targets,
            style_targets=style_targets,
            content_layers=content_layers_list,
            style_layers=style_layers_list,
            style_weights=config.style_weights,
            alpha=config.alpha,
            beta=config.beta,
            optimizer=optimizer,
        )

        # Log progress
        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            logger.info(
                f"Iteration {iteration:4d}/{config.iterations} — "
                f"Loss: {loss_value.numpy():.4f} — "
                f"Time: {elapsed:.1f}s"
            )

        # Save intermediate results
        if config.save_every > 0 and iteration % config.save_every == 0:
            intermediate_path = os.path.join(
                config.output_dir,
                f"intermediate_iter_{iteration:04d}.jpg",
            )
            save_image(generated_image, intermediate_path)
            logger.info(f"Intermediate saved: {intermediate_path}")

    # Save final output
    total_time = time.time() - start_time
    save_image(generated_image, output_path)
    logger.info("=" * 60)
    logger.info(f"Optimization complete! Total time: {total_time:.1f}s")
    logger.info(f"Final output saved to: {output_path}")
    logger.info("=" * 60)

    return generated_image
