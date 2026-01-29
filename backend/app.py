import os
import base64
import io
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# Terminal logging utilities
def log_progress(message: str, progress: int = None):
    """Log progress to terminal with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if progress is not None:
        print(f"[{timestamp}] {message} ({progress}%)", flush=True)
    else:
        print(f"[{timestamp}] {message}", flush=True)


def log_section(title: str):
    """Log a section header."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


# -----------------------------
# App configuration + constants
# -----------------------------
APP_TITLE = "CytoAssist – AI Assisted FNAC Screening Tool"
APP_SUBTITLE = (
    "Decision-support system for cytology image screening. Not for clinical diagnosis."
)
DISCLAIMER_TEXT = (
    "This tool is for educational and research purposes only. "
    "It does not provide medical diagnoses."
)

# Default local path (place the .pth next to this backend app).
DEFAULT_MODEL_PATH = "cytoassist_resnet18.pth"

# ImageNet normalization (required)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class order assumption: index 0 = Benign, index 1 = Suspicious
# (If your training used a different order, swap these labels.)
CLASS_NAMES = ["Benign", "Suspicious"]


# -----------------------------
# Model loading + preprocessing
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> nn.Module:
    """Load a ResNet18 model on CPU.

    The checkpoint may be:
    - a state_dict
    - a dict containing 'state_dict'
    - a serialized nn.Module

    We build a ResNet18 with a 1-output sigmoid head and load weights robustly.
    """
    log_section("MODEL LOADING")
    log_progress("Initializing ResNet18 architecture...", 10)
    device = torch.device("cpu")

    # Build architecture (ResNet18) with a single sigmoid output.
    # Trained setup: Linear -> Sigmoid, output shape [1], representing P(Suspicious).
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid(),
    )
    log_progress("Architecture configured (1-output sigmoid head)", 30)

    # Allow running from repo root: resolve relative paths against this file's directory.
    resolved_model_path = model_path
    if not os.path.isabs(resolved_model_path):
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_model_path = os.path.join(backend_dir, resolved_model_path)

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Resolved path checked: {resolved_model_path}\n"
            "Tip: Update the model path in the sidebar to point to your local .pth file."
        )

    log_progress(f"Loading weights from: {os.path.basename(resolved_model_path)}", 50)
    checkpoint = torch.load(resolved_model_path, map_location=device)
    log_progress("Checkpoint loaded into memory", 60)

    # If a whole module was saved, use it directly.
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        # Extract possible state_dict
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Strip 'module.' prefix from DataParallel checkpoints
        if isinstance(state_dict, dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                key = k
                if key.startswith("module."):
                    key = key[len("module."):]
                new_state_dict[key] = v
            state_dict = new_state_dict

        # Load (allow non-strict in case checkpoint contains extra keys)
        log_progress("Loading state dict into model...", 80)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    log_progress("Model ready for inference ✓", 100)
    return model


@st.cache_resource(show_spinner=False)
def get_preprocess() -> transforms.Compose:
    """Preprocessing required by the trained ResNet18.

    Requirement: Resize to 224x224 and normalize with ImageNet mean/std.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def pil_from_upload(uploaded_file) -> Image.Image:
    """Read uploaded image into a RGB PIL image."""
    log_section("IMAGE PROCESSING")
    log_progress("Loading uploaded image...", 50)
    image = Image.open(uploaded_file).convert("RGB")
    log_progress(f"Image loaded: {image.size[0]}x{image.size[1]} pixels ✓", 100)
    return image


def _read_frontend_css() -> str:
        """Read the <style> block from frontend/index.html (if present)."""
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_index = os.path.join(backend_dir, "..", "frontend", "index.html")
        if not os.path.exists(frontend_index):
                return ""

        with open(frontend_index, "r", encoding="utf-8") as f:
                html_text = f.read()

        start = html_text.find("<style")
        if start == -1:
                return ""
        start = html_text.find(">", start)
        if start == -1:
                return ""
        end = html_text.find("</style>", start)
        if end == -1:
                return ""
        return html_text[start + 1 : end].strip()


def _to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
        """Convert a PIL image to a base64 data URL."""
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"


def render_frontend_results(
        original: Image.Image,
        overlay: Image.Image,
        suspicious_prob: float,
        benign_prob: float,
) -> None:
        """Render the frontend/index.html visual style with real backend outputs."""
        css = _read_frontend_css()
        # Ensure predictable sizing inside Streamlit iframe
        extra_css = """
        .container { max-width: 1000px; margin: 0 auto; padding: 0 16px; }
        body { background: transparent !important; }
        header { position: sticky; top: 0; z-index: 1; }
        """

        original_url = _to_data_url(original, fmt="PNG")
        overlay_url = _to_data_url(overlay, fmt="PNG")

        suspicious_pct = max(0.0, min(100.0, suspicious_prob * 100.0))
        benign_pct = max(0.0, min(100.0, benign_prob * 100.0))

        # Use the existing HTML structure (header, disclaimer box, cards, progress bars, images).
        rendered = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <style>
    {css}
    {extra_css}
    </style>
</head>
<body>
    <header>
        <div class=\"container header-content\">
            <div class=\"logo-section\">
                <h1>CytoAssist</h1>
                <p>AI-assisted cytology image screening for decision support</p>
            </div>
            <div class=\"header-badge\">Research Preview</div>
        </div>
    </header>

    <div class=\"container\">
        <section class=\"disclaimer-box\">
            <div class=\"disclaimer-icon\">
                <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
                    <path d=\"M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z\"></path>
                    <line x1=\"12\" y1=\"9\" x2=\"12\" y2=\"13\"></line>
                    <line x1=\"12\" y1=\"17\" x2=\"12.01\" y2=\"17\"></line>
                </svg>
            </div>
            <div class=\"disclaimer-text\">
                <strong>Important Medical Disclaimer</strong>
                <p>{DISCLAIMER_TEXT}</p>
            </div>
        </section>

        <main>
            <div class=\"results-grid\">
                <section class=\"card\">
                    <div class=\"card-title\">Screening Result</div>

                    <div style=\"margin-top: 1rem;\">
                        <div class=\"confidence-item\">
                            <div class=\"confidence-header\">
                                <span>Suspicious Features Confidence</span>
                                <span>{suspicious_pct:.1f}%</span>
                            </div>
                            <div class=\"progress-track\">
                                <div class=\"progress-bar bar-suspicious\" style=\"width: {suspicious_pct:.1f}%;\"></div>
                            </div>
                        </div>

                        <div class=\"confidence-item\">
                            <div class=\"confidence-header\">
                                <span>Benign Features Confidence</span>
                                <span>{benign_pct:.1f}%</span>
                            </div>
                            <div class=\"progress-track\">
                                <div class=\"progress-bar bar-benign\" style=\"width: {benign_pct:.1f}%;\"></div>
                            </div>
                        </div>
                    </div>
                </section>

                <section class=\"card\">
                    <div class=\"card-title\">Model Attention Map (Grad-CAM)</div>

                    <div class=\"viz-comparison\">
                        <div class=\"viz-box\">
                            <img src=\"{original_url}\" alt=\"Original Image\" />
                            <div class=\"viz-label\">Original</div>
                        </div>
                        <div class=\"viz-box\">
                            <img src=\"{overlay_url}\" alt=\"Heatmap Overlay\" />
                            <div class=\"viz-label\">Attention Map</div>
                        </div>
                    </div>

                    <p style=\"margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted); text-align: center;\">
                        Highlighted regions indicate areas influencing the model output.
                    </p>
                </section>
            </div>
        </main>
    </div>
</body>
</html>
"""

        components.html(rendered, height=980, scrolling=True)


# -----------------------------
# Grad-CAM implementation
# -----------------------------
class GradCAM:
    """Minimal Grad-CAM for ResNet-like models.

    - Targets the last conv layer (provided by caller)
    - Captures activations and gradients
    - Produces a normalized heatmap in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook: capture activations
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

        # Backward hook: capture gradients
        # Prefer full backward hook if available.
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
        else:
            self._bwd_handle = self.target_layer.register_backward_hook(self._backward_hook)  # type: ignore

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] is gradient w.r.t. the layer output
        self.gradients = grad_output[0].detach()

    def close(self):
        """Remove hooks to avoid side effects."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Generate a Grad-CAM heatmap.

        For a single-output sigmoid model, the backward target is the total output.
        """
        log_section("GRAD-CAM GENERATION")
        log_progress("Computing activation maps...", 20)
        device = torch.device("cpu")
        input_tensor = input_tensor.to(device)

        # Forward
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        log_progress("Forward pass complete", 40)

        # Backward target for single-output sigmoid model
        score = logits.sum()
        log_progress("Computing gradients via backpropagation...", 60)
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        
        log_progress("Gradients captured successfully", 70)

        # Global-average-pool gradients to get channel weights
        # shapes: activations [N, C, H, W], gradients [N, C, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=False)  # [N, H, W]

        # ReLU and normalize to [0, 1]
        cam = F.relu(cam)
        cam = cam[0]  # remove batch
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        log_progress("Heatmap normalized and ready ✓", 100)

        return cam.cpu().numpy()


def apply_colormap_and_overlay(
    base_image_rgb: Image.Image,
    heatmap_01: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a heatmap (values in [0,1]) on top of an RGB PIL image.

    Uses matplotlib colormap to avoid dependencies like OpenCV.
    """
    import matplotlib.cm as cm

    base_np = np.array(base_image_rgb).astype(np.float32) / 255.0

    # Resize heatmap to image size
    heatmap_img = Image.fromarray((heatmap_01 * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(base_image_rgb.size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0

    # Apply colormap (jet) -> RGBA -> RGB
    cmap = cm.get_cmap("jet")
    colored = cmap(heatmap_resized)[:, :, :3]  # drop alpha

    # Blend
    overlay = (1.0 - alpha) * base_np + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def predict_with_probs(model: nn.Module, input_tensor: torch.Tensor) -> Tuple[float, float, int]:
    """Run CPU inference and return (benign_prob, suspicious_prob, predicted_index)."""
    device = torch.device("cpu")
    with torch.no_grad():
        suspicious_prob = float(model(input_tensor.to(device)).item())

    benign_prob = 1.0 - suspicious_prob
    pred_idx = 1 if suspicious_prob >= 0.5 else 0
    return benign_prob, suspicious_prob, pred_idx


# -----------------------------
# Streamlit UI
# -----------------------------
# Clinical/neutral styling (no new colors beyond neutral + blue/gray tone)
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)

st.title(APP_TITLE)
st.write(APP_SUBTITLE)

# Visible disclaimer box at top
st.warning(DISCLAIMER_TEXT)

# Sidebar for model path override (useful for local Windows demos)
with st.sidebar:
    st.subheader("Model Settings")
    model_path = st.text_input(
        "Model path (.pth)",
        value=DEFAULT_MODEL_PATH,
        help=(
            "Default matches the provided path. "
            "For local demos, update this to where the .pth exists on your machine."
        ),
    )

st.divider()

st.header("Upload Image")
uploaded = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload an image to run decision-support screening and view a Grad-CAM attention map.")
    st.stop()

# Load and show image
log_section("IMAGE PROCESSING")
log_progress("Loading uploaded image...", 50)
original_pil = pil_from_upload(uploaded)
log_progress(f"Image loaded: {original_pil.size[0]}x{original_pil.size[1]} pixels ✓", 100)

# Layout: image + results
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.image(original_pil, caption="Uploaded Image", use_container_width=True)

with col_right:
    st.header("Prediction")

    # Run inference
    try:
        with st.spinner("Loading model on CPU..."):
            log_section("MODEL LOADING")
            log_progress("Initializing ResNet18 architecture...", 10)
            model = load_model(model_path)
            log_progress("Model ready for inference ✓", 100)

        log_progress("Preprocessing image (resize 224x224, normalize)...", 50)
        log_progress("Preprocessing image (resize 224x224, ImageNet normalization)...", 50)
        preprocess = get_preprocess()
        input_tensor = preprocess(original_pil).unsqueeze(0)  # [1, 3, 224, 224]
        log_progress("Preprocessing complete ✓", 100)
        log_progress("Preprocessing complete ✓", 100)

        with st.spinner("Running inference..."):
            log_section("INFERENCE")
            log_progress("Running forward pass through ResNet18...", 30)
            benign_prob, suspicious_prob, pred_idx = predict_with_probs(model, input_tensor)
            log_progress(f"Prediction complete: Suspicious={suspicious_prob*100:.1f}%, Benign={benign_prob*100:.1f}% ✓", 100)

        # Display required outputs
        st.write(f"Suspicious Confidence: {suspicious_prob * 100:.1f}%")
        st.write(f"Benign Confidence: {benign_prob * 100:.1f}%")

        # Keep phrasing as decision-support (no diagnosis claims)
        st.caption(
            "Outputs are confidence estimates for screening support only; "
            "they are not diagnostic statements."
        )

    except Exception as e:
        st.error("Model inference could not be completed.")
        st.code(str(e))
        st.stop()

st.divider()

st.header("Explainability")

# Explainability section: show original + Grad-CAM overlay
exp_left, exp_right = st.columns(2, gap="large")

with exp_left:
    st.image(original_pil, caption="Uploaded Image", use_container_width=True)

with exp_right:
    st.subheader("Model Attention Map (Grad-CAM)")

    try:
        # Target the last convolutional layer of ResNet18
        target_layer = model.layer4[-1].conv2
        cam = GradCAM(model=model, target_layer=target_layer)

        # Use the predicted class for the attention map
        with st.spinner("Generating Grad-CAM attention map..."):
            log_section("GRAD-CAM GENERATION")
            log_progress("Computing activation maps...", 20)
            heatmap_01 = cam.generate(input_tensor=input_tensor, class_index=pred_idx)
            log_progress("Heatmap normalized and ready ✓", 100)

        cam.close()

        overlay_pil = apply_colormap_and_overlay(original_pil, heatmap_01, alpha=0.45)
        st.image(overlay_pil, caption="Model Attention Map (Grad-CAM)", use_container_width=True)

        st.caption(
            "Grad-CAM highlights image regions that most influenced the model’s prediction."
        )

    except Exception as e:
        st.error("Grad-CAM could not be generated.")
        st.code(str(e))

st.divider()

# Render the provided frontend UI (index.html) with real backend outputs
st.header("Web UI")
render_frontend_results(
    original=original_pil,
    overlay=overlay_pil,
    suspicious_prob=suspicious_prob,
    benign_prob=benign_prob,
)

if st.button("Analyze Another Image"):
    st.rerun()
