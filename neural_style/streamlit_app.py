"""
Neural Style Transfer — Streamlit Application

A premium, gallery-inspired interface for running neural style transfer
using a pretrained VGG19 network. Upload content and style images,
configure parameters, and generate stunning stylized artwork.
"""

import os
import sys
import io
import tempfile
import logging

import streamlit as st
from PIL import Image

# ── Ensure project root is importable ──────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import StyleTransferConfig

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Style Transfer — Transform Images Into Art",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Fonts ──────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root Variables ────────────────────────────────── */
    :root {
        --primary: #2C2C2C;
        --accent: #C8956C;
        --accent-dark: #A87650;
        --accent-light: #E8C9AD;
        --bg: #FAF7F2;
        --surface: #FFFFFF;
        --surface-alt: #F0EBE3;
        --text: #1A1A1A;
        --text-muted: #6B6B6B;
        --border: #E5DED4;
        --shadow: 0 2px 16px rgba(60, 40, 20, 0.06);
        --shadow-lg: 0 8px 40px rgba(60, 40, 20, 0.10);
        --radius: 12px;
        --radius-sm: 8px;
        --radius-lg: 20px;
        --font-heading: 'DM Serif Display', Georgia, serif;
        --font-body: 'Inter', -apple-system, sans-serif;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global Overrides ──────────────────────────────── */
    .stApp {
        font-family: var(--font-body) !important;
    }

    .stApp > header {
        background-color: transparent !important;
    }

    .block-container {
        max-width: 1100px !important;
        padding-top: 1rem !important;
    }

    /* ── Typography ────────────────────────────────────── */
    h1, h2, h3 {
        font-family: var(--font-heading) !important;
        color: var(--primary) !important;
        letter-spacing: -0.01em;
    }

    /* ── Hero Section ──────────────────────────────────── */
    .hero-container {
        text-align: center;
        padding: 3rem 1rem 2rem;
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent-light), var(--surface-alt));
        color: var(--accent-dark);
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        border: 1px solid var(--border);
    }

    .hero-title {
        font-family: var(--font-heading) !important;
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        color: var(--primary);
        margin-bottom: 0.8rem;
        line-height: 1.15;
        font-weight: 400;
    }

    .hero-subtitle {
        font-family: var(--font-body);
        font-size: 1.08rem;
        color: var(--text-muted);
        max-width: 600px;
        margin: 0 auto 2rem;
        line-height: 1.7;
        font-weight: 300;
    }

    /* ── Cards ─────────────────────────────────────────── */
    .premium-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem;
        box-shadow: var(--shadow);
        transition: var(--transition);
        margin-bottom: 1.5rem;
    }

    .premium-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    .card-header {
        font-family: var(--font-heading);
        font-size: 1.3rem;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .card-text {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.65;
    }

    /* ── Steps Section ─────────────────────────────────── */
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--accent), var(--accent-dark));
        color: white;
        border-radius: 50%;
        font-family: var(--font-heading);
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }

    .step-title {
        font-family: var(--font-heading);
        font-size: 1.15rem;
        color: var(--primary);
        margin-bottom: 0.4rem;
    }

    .step-desc {
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* ── Upload Zones ──────────────────────────────────── */
    .upload-zone {
        background: var(--surface);
        border: 2px dashed var(--border);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        transition: var(--transition);
    }

    .upload-zone:hover {
        border-color: var(--accent);
        background: linear-gradient(135deg, var(--surface), var(--surface-alt));
    }

    .upload-label {
        font-family: var(--font-heading);
        font-size: 1.1rem;
        color: var(--primary);
        margin-bottom: 0.3rem;
    }

    .upload-hint {
        color: var(--text-muted);
        font-size: 0.82rem;
    }

    /* ── Section Styling ───────────────────────────────── */
    .section-header {
        text-align: center;
        margin: 3rem 0 1.8rem;
    }

    .section-title {
        font-family: var(--font-heading) !important;
        font-size: 1.8rem;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .section-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Result Display ────────────────────────────────── */
    .result-frame {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.2rem;
        box-shadow: var(--shadow-lg);
    }

    .result-label {
        text-align: center;
        font-family: var(--font-heading);
        font-size: 1rem;
        color: var(--accent-dark);
        margin-top: 0.8rem;
    }

    /* ── Tech Card ─────────────────────────────────────── */
    .tech-pill {
        display: inline-block;
        background: var(--surface-alt);
        color: var(--text);
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 4px;
        border: 1px solid var(--border);
    }

    /* ── Sidebar Styling ───────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    .sidebar-title {
        font-family: var(--font-heading);
        font-size: 1.3rem;
        color: var(--primary);
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    .param-label {
        font-weight: 600;
        font-size: 0.82rem;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }

    .param-hint {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.7rem 2rem !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.02em;
        transition: var(--transition) !important;
        box-shadow: 0 4px 12px rgba(200, 149, 108, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(200, 149, 108, 0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .stDownloadButton > button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(44, 44, 44, 0.2) !important;
    }

    .stDownloadButton > button:hover {
        background: #444 !important;
    }

    /* ── Divider ───────────────────────────────────────── */
    .elegant-divider {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light));
        margin: 1rem auto;
        border-radius: 2px;
    }

    /* ── Footer ────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 3rem 0 2rem;
        color: var(--text-muted);
        font-size: 0.82rem;
        border-top: 1px solid var(--border);
        margin-top: 4rem;
    }

    .footer a {
        color: var(--accent-dark);
        text-decoration: none;
    }

    /* ── Image Styling ─────────────────────────────────── */
    [data-testid="stImage"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ── Expander ───────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-family: var(--font-heading) !important;
        font-size: 1.05rem !important;
        color: var(--primary) !important;
    }

    /* ── Spinner ────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ── Progress Bar ──────────────────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-dark)) !important;
    }

    /* ── Hide Streamlit Default ─────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✦ Powered by VGG19 Deep Learning</div>
    <div class="hero-title">Transform Your Images<br>Into Masterpieces</div>
    <div class="hero-subtitle">
        Harness the power of convolutional neural networks to blend the content
        of any photograph with the artistic style of iconic paintings.
    </div>
    <div class="elegant-divider"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <div class="section-title">How It Works</div>
    <div class="section-subtitle">Three simple steps to create stunning AI-generated artwork</div>
    <div class="elegant-divider"></div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)

steps = [
    ("1", "📤", "Upload", "Select a content photo and a style reference image — any painting, texture, or pattern you love."),
    ("2", "⚙️", "Configure", "Fine-tune the style strength, iterations, and resolution to get exactly the look you want."),
    ("3", "🎨", "Generate", "Our VGG19 neural network optimizes your image pixel-by-pixel to create the final artwork."),
]

for col, (num, icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="premium-card" style="text-align: center; min-height: 220px;">
            <div class="step-number">{num}</div>
            <div style="font-size: 2rem; margin-bottom: 0.6rem;">{icon}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR — PARAMETERS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">🎨 Style Controls</div>
    <div class="sidebar-subtitle">Adjust these parameters to control the style transfer output.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="param-label">Style Strength (β)</div>', unsafe_allow_html=True)
    st.markdown('<div class="param-hint">Higher values apply more artistic style. Lower values preserve more of the original photo.</div>', unsafe_allow_html=True)
    style_strength = st.slider(
        "Style Strength",
        min_value=1e2,
        max_value=1e6,
        value=1e4,
        step=1e2,
        format="%.0f",
        label_visibility="collapsed",
    )

    st.markdown('<div class="param-label">Iterations</div>', unsafe_allow_html=True)
    st.markdown('<div class="param-hint">More iterations = better quality but longer processing time.</div>', unsafe_allow_html=True)
    iterations = st.slider(
        "Iterations",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        label_visibility="collapsed",
    )

    st.markdown('<div class="param-label">Image Size</div>', unsafe_allow_html=True)
    st.markdown('<div class="param-hint">Maximum dimension in pixels. Higher = more detail but slower.</div>', unsafe_allow_html=True)
    image_size = st.slider(
        "Image Size",
        min_value=256,
        max_value=1024,
        value=512,
        step=64,
        label_visibility="collapsed",
    )

    st.markdown('<div class="param-label">Learning Rate</div>', unsafe_allow_html=True)
    st.markdown('<div class="param-hint">Controls optimization speed. Default (5.0) works well for most cases.</div>', unsafe_allow_html=True)
    learning_rate = st.slider(
        "Learning Rate",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown('<div class="param-label">Initialization</div>', unsafe_allow_html=True)
    init_mode = st.selectbox(
        "Init Mode",
        options=["content", "random"],
        index=0,
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════
# STYLE TRANSFER STUDIO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <div class="section-title">Style Transfer Studio</div>
    <div class="section-subtitle">Upload your images below to create AI-generated artwork</div>
    <div class="elegant-divider"></div>
</div>
""", unsafe_allow_html=True)

col_content, col_style = st.columns(2)

with col_content:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-label">📷 Content Image</div>
        <div class="upload-hint">The photo whose structure you want to keep</div>
    </div>
    """, unsafe_allow_html=True)
    content_file = st.file_uploader(
        "Upload content image",
        type=["jpg", "jpeg", "png", "webp"],
        key="content_uploader",
        label_visibility="collapsed",
    )
    if content_file:
        content_img = Image.open(content_file)
        st.image(content_img, caption="Content Image", use_container_width=True)

with col_style:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-label">🖼️ Style Image</div>
        <div class="upload-hint">The painting or artwork whose style you want to apply</div>
    </div>
    """, unsafe_allow_html=True)
    style_file = st.file_uploader(
        "Upload style image",
        type=["jpg", "jpeg", "png", "webp"],
        key="style_uploader",
        label_visibility="collapsed",
    )
    if style_file:
        style_img = Image.open(style_file)
        st.image(style_img, caption="Style Image", use_container_width=True)


# ── Generate Button ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    generate_clicked = st.button(
        "✦  Generate Artwork",
        use_container_width=True,
        disabled=not (content_file and style_file),
    )


# ── Run Style Transfer ─────────────────────────────────────
if generate_clicked and content_file and style_file:
    # Import pipeline modules lazily to speed up initial page load
    from pipelines.optimization_pipeline import run_style_transfer
    from utils.image_utils import tensor_to_image

    with st.spinner(""):
        # Show progress
        status_container = st.container()
        with status_container:
            st.markdown("""
            <div class="premium-card" style="text-align: center;">
                <div class="card-header">🎨 Creating Your Artwork</div>
                <div class="card-text">The neural network is optimizing your image pixel by pixel.
                This may take a few minutes depending on your settings.</div>
            </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0, text="Initializing VGG19 model...")

        try:
            # Save uploaded images to temp files
            with tempfile.TemporaryDirectory(prefix="nst_streamlit_") as temp_dir:
                content_path = os.path.join(temp_dir, "content.jpg")
                style_path = os.path.join(temp_dir, "style.jpg")
                output_path = os.path.join(temp_dir, "output.jpg")

                # Reset file positions and save
                content_file.seek(0)
                style_file.seek(0)
                Image.open(content_file).convert("RGB").save(content_path, "JPEG")
                Image.open(style_file).convert("RGB").save(style_path, "JPEG")

                progress_bar.progress(10, text="Loading and preprocessing images...")

                # Configure
                config = StyleTransferConfig(
                    image_size=image_size,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    alpha=1.0,
                    beta=style_strength,
                    save_every=0,
                    output_dir=temp_dir,
                    init_mode=init_mode,
                )

                progress_bar.progress(20, text="Running style transfer optimization...")

                # Run style transfer
                result_tensor = run_style_transfer(
                    content_path=content_path,
                    style_path=style_path,
                    config=config,
                    output_path=output_path,
                )

                progress_bar.progress(95, text="Generating final image...")

                # Convert to PIL image
                result_image = tensor_to_image(result_tensor)
                progress_bar.progress(100, text="Complete!")

            # Display result
            st.markdown("""
            <div class="section-header">
                <div class="section-title">Your Artwork</div>
                <div class="elegant-divider"></div>
            </div>
            """, unsafe_allow_html=True)

            # Show result in a framed display
            result_col1, result_col2, result_col3 = st.columns([1, 3, 1])
            with result_col2:
                st.markdown('<div class="result-frame">', unsafe_allow_html=True)
                st.image(result_image, caption="Stylized Result", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Download button
            buf = io.BytesIO()
            result_image.save(buf, format="JPEG", quality=95)
            buf.seek(0)

            dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
            with dl_col2:
                st.download_button(
                    label="⬇  Download Artwork",
                    data=buf,
                    file_name="neural_style_transfer_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            # Show comparison
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <div class="section-title" style="font-size: 1.3rem;">Comparison</div>
            </div>
            """, unsafe_allow_html=True)

            comp_cols = st.columns(3)
            with comp_cols[0]:
                content_file.seek(0)
                st.image(Image.open(content_file), caption="Content", use_container_width=True)
            with comp_cols[1]:
                style_file.seek(0)
                st.image(Image.open(style_file), caption="Style", use_container_width=True)
            with comp_cols[2]:
                st.image(result_image, caption="Result", use_container_width=True)

        except Exception as e:
            st.error(f"**Style transfer failed:** {str(e)}")
            logger.error(f"Style transfer error: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════
# ABOUT THE TECHNOLOGY
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header" style="margin-top: 4rem;">
    <div class="section-title">About the Technology</div>
    <div class="section-subtitle">The science behind the art</div>
    <div class="elegant-divider"></div>
</div>
""", unsafe_allow_html=True)

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown("""
    <div class="premium-card">
        <div class="card-header">Neural Style Transfer</div>
        <div class="card-text">
            Based on the seminal paper by Gatys et al. (2015), this technique uses deep
            convolutional neural network feature representations to separate and recombine
            the <em>content</em> and <em>style</em> of images. The generated image is
            optimized pixel-by-pixel using gradient descent.
        </div>
    </div>
    """, unsafe_allow_html=True)

with tech_col2:
    st.markdown("""
    <div class="premium-card">
        <div class="card-header">How Losses Work</div>
        <div class="card-text">
            <strong>Content Loss</strong> preserves the spatial structure from your photo using
            deeper CNN layers. <strong>Style Loss</strong> captures textures and patterns via
            Gram matrices across multiple layers. The total loss balances both, guided by the
            α and β parameters you control.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Architecture Details (expandable)
with st.expander("🔍 Detailed Architecture"):
    st.markdown("""
    **VGG19 Feature Extractor** — A pretrained VGG19 network (frozen weights) acts as a
    multi-scale feature extractor. It is never trained — only used to compute losses.

    | Layer | Purpose |
    |-------|---------|
    | `block1_conv1` | Style features |
    | `block2_conv1` | Style features |
    | `block3_conv1` | Style features |
    | `block4_conv1` | Style features |
    | `block4_conv2` | **Content features** |
    | `block5_conv1` | Style features |

    **Gram Matrix** captures *which features activate together*, encoding texture information
    while discarding spatial layout: `G = F^T · F / N`

    **Optimization** uses Adam optimizer to update the generated image's pixel values directly,
    minimizing: `Total Loss = α × Content Loss + β × Style Loss`
    """)

# Tech Stack Pills
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <span class="tech-pill">TensorFlow</span>
    <span class="tech-pill">VGG19</span>
    <span class="tech-pill">Gram Matrices</span>
    <span class="tech-pill">Adam Optimizer</span>
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Streamlit</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 0.5rem;">
        <strong style="font-family: var(--font-heading); font-size: 1rem; color: var(--primary);">
            Neural Style Transfer
        </strong>
    </div>
    <div>
        Built with TensorFlow &amp; VGG19 · Based on
        <a href="https://arxiv.org/abs/1508.06576" target="_blank">Gatys et al., 2015</a>
    </div>
    <div style="margin-top: 0.3rem;">
        Open source under MIT License
    </div>
</div>
""", unsafe_allow_html=True)
