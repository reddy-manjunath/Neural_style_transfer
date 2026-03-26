"""
CLI script for Neural Style Transfer.

Provides a command-line interface to apply artistic style transfer
to content images using a pretrained VGG19 network.

Example usage:
    python run_style_transfer.py \\
        --content examples/content.jpg \\
        --style examples/style.jpg \\
        --iterations 400
"""

import os
import sys
import argparse
import logging

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import StyleTransferConfig
from pipelines.optimization_pipeline import run_style_transfer


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If True, set logging level to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer using VGG19 CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python run_style_transfer.py --content examples/content.jpg --style examples/style.jpg

  Custom iterations and weights:
    python run_style_transfer.py \\
        --content examples/content.jpg \\
        --style examples/style.jpg \\
        --iterations 500 \\
        --alpha 1.0 \\
        --beta 1e5

  High resolution output:
    python run_style_transfer.py \\
        --content examples/content.jpg \\
        --style examples/style.jpg \\
        --image-size 1024 \\
        --output outputs/high_res_output.jpg
        """,
    )

    # Required arguments
    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="Path to the content image.",
    )
    parser.add_argument(
        "--style",
        type=str,
        required=True,
        help="Path to the style image.",
    )

    # Optional arguments
    parser.add_argument(
        "--iterations",
        type=int,
        default=400,
        help="Number of optimization iterations (default: 400).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Content loss weight (default: 1.0).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e4,
        help="Style loss weight (default: 1e4).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Maximum image dimension in pixels (default: 512, max: 1024).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5.0,
        help="Adam optimizer learning rate (default: 5.0).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save intermediate results every N iterations (default: 50). Set to 0 to disable.",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["content", "random"],
        default="content",
        help="Initialization mode for generated image (default: content).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the stylized image (default: outputs/stylized_output.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output images (default: outputs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)

    # Validate input files exist
    if not os.path.isfile(args.content):
        logger.error(f"Content image not found: {args.content}")
        sys.exit(1)
    if not os.path.isfile(args.style):
        logger.error(f"Style image not found: {args.style}")
        sys.exit(1)

    # Build config from CLI arguments
    config = StyleTransferConfig(
        image_size=args.image_size,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        save_every=args.save_every,
        output_dir=args.output_dir,
        init_mode=args.init_mode,
    )

    logger.info("Starting Neural Style Transfer...")
    logger.info(f"Content: {args.content}")
    logger.info(f"Style:   {args.style}")

    # Run style transfer
    run_style_transfer(
        content_path=args.content,
        style_path=args.style,
        config=config,
        output_path=args.output,
    )

    logger.info("Done! Check the outputs directory for results.")


if __name__ == "__main__":
    main()
