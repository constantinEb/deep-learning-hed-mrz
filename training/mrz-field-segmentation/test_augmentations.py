import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_dataloaders
from transformer import MRZTransform


def setup_args():
    parser = argparse.ArgumentParser(
        description="Test and visualize MRZ augmentation pipeline"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to generate (default: 8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="augmentation_test_output",
        help="Output directory for visualizations (default: augmentation_test_output)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/master-thesis-eberc/data/midv2020",
        help="Path to MIDV2020 dataset root"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)"
    )
    return parser.parse_args()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()

    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            # Grayscale: (1, H, W) -> (H, W)
            arr = arr[0]
        else:
            # RGB: (C, H, W) -> (H, W, C)
            arr = arr.transpose(1, 2, 0)

    # Clip to [0, 1] and scale to [0, 255]
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    return arr


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    mask_colored = np.zeros_like(image_rgb)
    mask_colored[:, :, 0] = mask

    overlay = cv2.addWeighted(image_rgb, 1.0, mask_colored, alpha, 0)

    return overlay


def create_comparison_grid(
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    output_path: str,
    title: str = "MRZ Augmentation Comparison"
):
    n_samples = len(samples)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    for idx, (img_orig, mask_orig, img_aug, mask_aug) in enumerate(samples):
        img_orig_np = tensor_to_numpy(img_orig)
        mask_orig_np = tensor_to_numpy(mask_orig)
        img_aug_np = tensor_to_numpy(img_aug)
        mask_aug_np = tensor_to_numpy(mask_aug)

        overlay_orig = create_overlay(img_orig_np, mask_orig_np)
        overlay_aug = create_overlay(img_aug_np, mask_aug_np)

        axes[idx, 0].imshow(overlay_orig)
        axes[idx, 0].set_title(f"Sample {idx + 1}: Original", fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(overlay_aug)
        axes[idx, 1].set_title(f"Sample {idx + 1}: Augmented", fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved grid visualization to: {output_path}")


def save_individual_sample(
    img_orig: torch.Tensor,
    mask_orig: torch.Tensor,
    img_aug: torch.Tensor,
    mask_aug: torch.Tensor,
    output_path: str,
    sample_idx: int
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    img_orig_np = tensor_to_numpy(img_orig)
    mask_orig_np = tensor_to_numpy(mask_orig)
    img_aug_np = tensor_to_numpy(img_aug)
    mask_aug_np = tensor_to_numpy(mask_aug)

    # Original image with overlay
    overlay_orig = create_overlay(img_orig_np, mask_orig_np)
    axes[0, 0].imshow(overlay_orig)
    axes[0, 0].set_title("Original Image + Mask", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Augmented image with overlay
    overlay_aug = create_overlay(img_aug_np, mask_aug_np)
    axes[0, 1].imshow(overlay_aug)
    axes[0, 1].set_title("Augmented Image + Mask", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Original mask only
    axes[1, 0].imshow(mask_orig_np, cmap='gray')
    axes[1, 0].set_title("Original Mask", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Augmented mask only
    axes[1, 1].imshow(mask_aug_np, cmap='gray')
    axes[1, 1].set_title("Augmented Mask", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    fig.suptitle(f"Sample {sample_idx + 1} - Detailed View", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = setup_args()

    print("MRZ Augmentation Test Script")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    aug_config = {
        'aug_perspective': True,
        'aug_rotate': True,
        'aug_blur': True,
        'aug_brightness_contrast': True,
        'aug_compression': True,
        'aug_perspective_scale': (0.05, 0.12),
        'aug_rotate_limit': 25,
        'aug_blur_limit': 7,
        'aug_brightness_limit': 0.3,
        'aug_contrast_limit': 0.3,
        'aug_compression_quality': (60, 100),
        'aug_prob': 0.95,
    }


    train_loader, val_loader, test_loader = create_dataloaders(
        datasets=["MIDV2020"],
        data_roots={"MIDV2020": args.data_root},
        midv2020_modalities=["photo"],
        batch_size=1,
        img_size=384,
        grayscale=True,
        split_ratios=(0.7, 0.15, 0.15),
        seed=args.seed,
        num_workers=0,
        pin_memory=False,
        aug_config=aug_config,
        use_kornia=True
    )

 
    transform = MRZTransform(
        img_size=384,
        train=True,
        aug_config=aug_config
    ).to(args.device)

    samples = []

    for i, (img, mask) in enumerate(train_loader):
        if i >= args.num_samples:
            break

        img = img.to(args.device)
        mask = mask.to(args.device)

        img_orig = img.clone()
        mask_orig = mask.clone()

        img_aug, mask_aug = transform(img, mask)

        samples.append((img_orig[0], mask_orig[0], img_aug[0], mask_aug[0]))

        if args.save_individual:
            individual_path = os.path.join(
                args.output_dir,
                f"sample_{i + 1:02d}_detailed.png"
            )
            save_individual_sample(
                img_orig[0], mask_orig[0],
                img_aug[0], mask_aug[0],
                individual_path,
                i
            )


    grid_path = os.path.join(args.output_dir, "augmentation_comparison_grid.png")
    create_comparison_grid(
        samples,
        grid_path,
        title=f"MRZ Augmentation Comparison (Kornia GPU-based) - {args.num_samples} Samples"
    )

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
