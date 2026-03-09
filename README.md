# 🎨 Neural Style Transfer using Convolutional Neural Networks

A production-ready implementation of **Neural Style Transfer** using a pretrained **VGG19** network in TensorFlow. This system takes a content image and a style image and generates a stylized output by optimizing pixel values to minimize content and style losses.

![Neural Style Transfer](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg)

---

## 📖 Table of Contents

- [Concept](#-concept)
- [Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Setup](#-setup)
- [Usage](#-usage)
- [API](#-api-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)

---

## 🧠 Concept

**Neural Style Transfer** (Gatys et al., 2015) uses deep CNN feature representations to separate and recombine the *content* and *style* of images. A pretrained VGG19 network acts as a feature extractor — it is never trained, only used to compute losses that guide the optimization of a generated image.

### Content vs. Style Representations

| Aspect | Content | Style |
|--------|---------|-------|
| **What it captures** | Object shapes, spatial structure | Textures, colors, patterns |
| **CNN layers used** | Deeper layers (e.g., `block4_conv2`) | Multiple layers (`block1_conv1` – `block5_conv1`) |
| **Representation** | Raw feature maps | Gram matrices (feature correlations) |
| **Preserved from** | Content image | Style image |

### Gram Matrix Intuition

The **Gram matrix** captures *which features tend to activate together* across an image. By computing the inner product of feature maps, it encodes texture and pattern information while discarding spatial layout. Two images with similar Gram matrices share similar textures, regardless of what objects appear in them.

```
G = F^T · F / N
```

Where `F` is the feature map reshaped to (channels × spatial_locations) and `N` is the number of spatial locations.

### Loss Functions

- **Content Loss** = MSE(features_generated, features_content)
- **Style Loss** = Σ MSE(Gram(features_generated), Gram(features_style)) / num_layers
- **Total Loss** = α × Content Loss + β × Style Loss

The hyperparameters `α` (alpha) and `β` (beta) control the trade-off between preserving content structure and applying artistic style.

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌─────────────┐
│ Content Img  │    │  Style Img   │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│     Image Loading & Resize      │
│     (pipelines/image_loader)    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     VGG19 Preprocessing         │
│     (pipelines/preprocessing)   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     VGG19 Feature Extractor     │
│     (models/style_transfer)     │
│     [Frozen Weights]            │
└──────────────┬──────────────────┘
               │
          ┌────┴────┐
          ▼         ▼
    ┌──────────┐ ┌──────────┐
    │ Content  │ │  Style   │
    │ Features │ │ Features │
    └────┬─────┘ └────┬─────┘
         │            │
         ▼            ▼
┌─────────────────────────────────┐
│        Loss Computation         │
│  Content Loss + Style Loss      │
│  (utils/loss_functions)         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Adam Optimizer Loop         │
│  Update generated image pixels  │
│  (pipelines/optimization)       │
└──────────────┬──────────────────┘
               │
               ▼
       ┌───────────────┐
       │ Stylized Image │
       └───────────────┘
```

---

## ⚙️ How It Works

1. **Load** content and style images, resize to target dimensions
2. **Preprocess** images for VGG19 (BGR conversion, ImageNet mean subtraction)
3. **Extract features** from both images using frozen VGG19 layers
4. **Initialize** the generated image (from content image or random noise)
5. **Optimize** the generated image using Adam optimizer:
   - Compute content loss from deeper layers
   - Compute style loss from Gram matrices across multiple layers
   - Backpropagate total loss to update pixel values
6. **Save** intermediate results and final stylized output

---

## 🚀 Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd neural_style

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### CLI — Basic

```bash
python run_style_transfer.py \
    --content examples/content.jpg \
    --style examples/style.jpg \
    --iterations 400
```

### CLI — Custom Parameters

```bash
python run_style_transfer.py \
    --content examples/content.jpg \
    --style examples/style.jpg \
    --iterations 500 \
    --alpha 1.0 \
    --beta 1e5 \
    --image-size 1024 \
    --learning-rate 5.0 \
    --save-every 100 \
    --output outputs/my_artwork.jpg
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--content` | *required* | Path to content image |
| `--style` | *required* | Path to style image |
| `--iterations` | 400 | Optimization iterations |
| `--alpha` | 1.0 | Content loss weight |
| `--beta` | 1e4 | Style loss weight |
| `--image-size` | 512 | Max image dimension (up to 1024) |
| `--learning-rate` | 5.0 | Adam optimizer learning rate |
| `--save-every` | 50 | Save intermediate results every N steps |
| `--init-mode` | content | Initialize from 'content' or 'random' |
| `--output` | outputs/stylized_output.jpg | Output file path |

---

## 🌐 API Usage

### Start the Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Or directly:

```bash
python api/server.py
```

### POST /style-transfer

Upload content and style images as multipart form data:

```bash
curl -X POST "http://localhost:8000/style-transfer" \
    -F "content_image=@examples/content.jpg" \
    -F "style_image=@examples/style.jpg" \
    -F "iterations=300" \
    -F "style_strength=10000" \
    --output stylized_result.jpg
```

#### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content_image` | file | *required* | Content image upload |
| `style_image` | file | *required* | Style image upload |
| `style_strength` | float | 1e4 | Style loss weight (beta) |
| `iterations` | int | 300 | Optimization iterations |
| `image_size` | int | 512 | Max image dimension |
| `learning_rate` | float | 5.0 | Optimizer learning rate |

### Interactive Docs

Visit `http://localhost:8000/docs` for the Swagger UI.

---

## 🔧 Configuration

All defaults are defined in `configs/config.py` using a Python dataclass:

```python
from configs.config import StyleTransferConfig

config = StyleTransferConfig(
    image_size=512,
    iterations=400,
    learning_rate=5.0,
    alpha=1.0,        # Content weight
    beta=1e4,         # Style weight
    save_every=50,
    init_mode="content",
)
```

### VGG19 Layers Used

| Layer (Keras name) | Purpose |
|--------------------|---------|
| `block1_conv1` | Style |
| `block2_conv1` | Style |
| `block3_conv1` | Style |
| `block4_conv1` | Style |
| `block4_conv2` | **Content** |
| `block5_conv1` | Style |

---

## 📁 Project Structure

```
neural_style/
│
├── models/
│   └── style_transfer_model.py   # VGG19 feature extractor
│
├── pipelines/
│   ├── image_loader.py           # Image loading & resizing
│   ├── preprocessing.py          # VGG19 preprocessing
│   └── optimization_pipeline.py  # Optimization loop
│
├── api/
│   └── server.py                 # FastAPI REST server
│
├── utils/
│   ├── gram_matrix.py            # Gram matrix computation
│   ├── loss_functions.py         # Content, style, total loss
│   └── image_utils.py            # Tensor ↔ image conversion
│
├── configs/
│   └── config.py                 # Hyperparameter config
│
├── examples/
│   ├── content.jpg               # Sample content image
│   └── style.jpg                 # Sample style image
│
├── outputs/                      # Generated images saved here
├── requirements.txt
├── README.md
└── run_style_transfer.py         # CLI entry point
```

---

## 📚 References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style.* arXiv:1508.06576
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
