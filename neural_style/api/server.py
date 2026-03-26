"""
FastAPI server for Neural Style Transfer.

Provides a REST API endpoint to perform style transfer on uploaded
images. Users can upload content and style images, configure
parameters, and receive the stylized result.
"""

import os
import sys
import io
import uuid
import logging
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import StyleTransferConfig
from pipelines.optimization_pipeline import run_style_transfer
from utils.image_utils import tensor_to_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI Application ─────────────────────────────────────────
app = FastAPI(
    title="Neural Style Transfer API",
    description=(
        "Apply artistic style transfer to images using a pretrained "
        "VGG19 CNN. Upload a content image and a style image to generate "
        "a stylized result."
    ),
    version="1.0.0",
)


@app.get("/")
async def root():
    """Health check and welcome endpoint."""
    return {
        "service": "Neural Style Transfer API",
        "version": "1.0.0",
        "endpoints": {
            "POST /style-transfer": "Apply style transfer to uploaded images",
        },
    }


@app.post("/style-transfer")
async def style_transfer(
    content_image: UploadFile = File(..., description="Content image file"),
    style_image: UploadFile = File(..., description="Style image file"),
    style_strength: Optional[float] = Form(
        1e4, description="Style loss weight (beta). Higher = stronger style."
    ),
    iterations: Optional[int] = Form(
        300, description="Number of optimization iterations."
    ),
    image_size: Optional[int] = Form(
        512, description="Maximum image dimension (64-1024)."
    ),
    learning_rate: Optional[float] = Form(
        5.0, description="Adam optimizer learning rate."
    ),
):
    """Apply Neural Style Transfer to uploaded images.

    Accepts a content image and a style image as multipart file uploads,
    runs the style transfer optimization, and returns the stylized image.

    Args:
        content_image: The content image to preserve structure from.
        style_image: The style image to transfer artistic style from.
        style_strength: Beta weight for style loss (default: 1e4).
        iterations: Number of optimization iterations (default: 300).
        image_size: Maximum image dimension in pixels (default: 512).
        learning_rate: Optimizer learning rate (default: 5.0).

    Returns:
        The stylized image as a JPEG response.
    """
    # Validate inputs
    if not content_image.content_type or not content_image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="content_image must be an image file.",
        )
    if not style_image.content_type or not style_image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="style_image must be an image file.",
        )

    # Clamp image_size
    image_size = max(64, min(image_size, 1024))

    # Create temporary files for the uploaded images
    request_id = uuid.uuid4().hex[:8]
    temp_dir = tempfile.mkdtemp(prefix=f"nst_{request_id}_")

    try:
        # Save uploaded files to temp directory
        content_path = os.path.join(temp_dir, "content.jpg")
        style_path = os.path.join(temp_dir, "style.jpg")

        content_data = await content_image.read()
        style_data = await style_image.read()

        with open(content_path, "wb") as f:
            f.write(content_data)
        with open(style_path, "wb") as f:
            f.write(style_data)

        logger.info(
            f"[{request_id}] Style transfer request — "
            f"iterations={iterations}, style_strength={style_strength}, "
            f"image_size={image_size}"
        )

        # Configure and run style transfer
        config = StyleTransferConfig(
            image_size=image_size,
            iterations=iterations,
            learning_rate=learning_rate,
            alpha=1.0,
            beta=style_strength,
            save_every=0,  # Don't save intermediates for API requests
            output_dir=temp_dir,
        )

        output_path = os.path.join(temp_dir, "output.jpg")
        result_tensor = run_style_transfer(
            content_path=content_path,
            style_path=style_path,
            config=config,
            output_path=output_path,
        )

        # Convert result to PIL image and stream as response
        result_image = tensor_to_image(result_tensor)
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format="JPEG", quality=95)
        img_buffer.seek(0)

        logger.info(f"[{request_id}] Style transfer complete")

        return StreamingResponse(
            img_buffer,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=stylized_{request_id}.jpg"
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Style transfer failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Style transfer failed: {str(e)}",
        )

    finally:
        # Clean up temp files
        import shutil

        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
