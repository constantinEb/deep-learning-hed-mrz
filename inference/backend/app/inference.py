"""
Inference pipeline wrapper.
Reuses logic from training/mrz-field-segmentation/inference_test.py
"""
import sys
import base64
import io
from typing import Dict, Tuple
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, "/workspace/training/mrz-field-segmentation")

from inference_test import infer_sample, create_visualization
from app.model_loader import model_manager


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 string to numpy array.

    Args:
        base64_str: Base64-encoded image (with or without data URI prefix)

    Returns:
        numpy array (H, W, 3) in RGB format
    """
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    img_bytes = base64.b64decode(base64_str)

    img_pil = Image.open(io.BytesIO(img_bytes))

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    img_np = np.array(img_pil)

    return img_np


def encode_image_to_base64(img_np: np.ndarray, format: str = "PNG") -> str:
    """
    Encode numpy array to base64 string.

    Args:
        img_np: numpy array (H, W, 3) in RGB format
        format: Image format ("PNG" or "JPEG")

    Returns:
        Base64-encoded string with data URI prefix
    """
    img_pil = Image.fromarray(img_np.astype(np.uint8))

    buffer = io.BytesIO()
    img_pil.save(buffer, format=format)
    buffer.seek(0)

    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    mime_type = "image/png" if format == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{img_b64}"


def preprocess_image(img_rgb: np.ndarray, target_size: int = 384) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model inference.

    Args:
        img_rgb: Input image (H, W, 3) in RGB format
        target_size: Target size for model input

    Returns:
        Tuple of (preprocessed tensor, resized RGB image for visualization)
    """
    img_resized_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    img_gray = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2GRAY)

    img_normalized = img_gray.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)

    return img_tensor, img_resized_rgb


def run_inference(
    base64_image: str,
    model_type: str
) -> Dict[str, str]:
    """
    Run inference on a single image.

    Args:
        base64_image: Base64-encoded input image
        model_type: "hough_encoder" or "hed_mrz"

    Returns:
        Dictionary with base64-encoded visualization images:
        {
            "original": "data:image/png;base64,...",
            "heatmap": "data:image/png;base64,...",
            "overlay": "data:image/png;base64,...",
            "obb": "data:image/png;base64,..."
        }
    """
    model, postprocess_to_obbs, iou_binary, config = model_manager.get_model(model_type)
    img_rgb = decode_base64_image(base64_image)
    img_tensor, img_rgb_resized = preprocess_image(img_rgb, target_size=config.img_size)

    prob_map, binary_mask = infer_sample(
        model=model,
        img_tensor=img_tensor,
        device=model_manager.device
    )

    gt_mask = np.zeros_like(prob_map)

    vis_original, vis_heatmap, vis_overlay, vis_obb = create_visualization(
        img_rgb=img_rgb_resized,
        prob_map=prob_map,
        binary_mask=binary_mask,
        gt_mask=gt_mask,
        doc_info="",
        iou=0.0,
        postprocess_to_obbs=postprocess_to_obbs
    )

    result = {
        "original": encode_image_to_base64(vis_original, format="PNG"),
        "heatmap": encode_image_to_base64(vis_heatmap, format="PNG"),
        "overlay": encode_image_to_base64(vis_overlay, format="PNG"),
        "obb": encode_image_to_base64(vis_obb, format="PNG")
    }

    return result
