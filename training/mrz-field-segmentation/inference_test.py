import os
import sys
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Model imports - will be done dynamically based on trainer type
# from trainer_hough_encoder import HEDMRZ, postprocess_to_obbs, iou_binary
from dataset import create_dataloaders


def load_config_from_file(config_path: str):
    """Load config module from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def get_trainer_module(trainer_type: str):
    """Get the trainer module based on trainer type."""
    if trainer_type == "hough_encoder":
        from trainer_hough_encoder import HEDMRZ, postprocess_to_obbs, iou_binary
        config_name = "config_hough_encoder"
    elif trainer_type == "hed_mrz":
        import sys
        # Import from trainer-hed-mrz (with hyphen)
        spec = importlib.util.spec_from_file_location(
            "trainer_hed_mrz",
            os.path.join(os.path.dirname(__file__), "trainer-hed-mrz.py")
        )
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
        HEDMRZ = trainer_module.HEDMRZ
        postprocess_to_obbs = trainer_module.postprocess_to_obbs
        iou_binary = trainer_module.iou_binary
        config_name = "config_hed_mrz"
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Must be 'hough_encoder' or 'hed_mrz'")

    return HEDMRZ, postprocess_to_obbs, iou_binary, config_name


def load_model(
    checkpoint_path: str = None,
    run_dir: str = None,
    trainer_type: str = "hough_encoder",
    checkpoint_name: str = "best.pt",
    device: str = "cuda"
) -> tuple:
    """
    Load trained HEDMRZ model from checkpoint.

    Args:
        checkpoint_path: Direct path to checkpoint file (if not using run_dir)
        run_dir: Path to runs/ directory (loads config and checkpoint from there)
        trainer_type: Type of trainer ('hough_encoder' or 'hed_mrz')
        checkpoint_name: Name of checkpoint file (default: 'best.pt')
        device: Device to load model on

    Returns:
        tuple: (model, postprocess_to_obbs function, iou_binary function, config)
    """
    # Get trainer module
    HEDMRZ, postprocess_to_obbs, iou_binary, config_name = get_trainer_module(trainer_type)

    # Determine checkpoint path and config
    if run_dir:
        # Load from run directory
        checkpoint_path = os.path.join(run_dir, checkpoint_name)
        config_file = os.path.join(run_dir, f"{config_name}.py")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading from run directory: {run_dir}")
        print(f"  Checkpoint: {checkpoint_path}")

        # Load config from run directory if it exists
        if os.path.exists(config_file):
            print(f"  Config: {config_file}")
            config = load_config_from_file(config_file)
        else:
            print(f"  Config not found in run directory, using active config")
            if config_name == "config_hough_encoder":
                import config_hough_encoder as config
            else:
                import config_hed_mrz as config
    else:
        # Load from explicit checkpoint path and active config
        if not checkpoint_path:
            raise ValueError("Either checkpoint_path or run_dir must be provided")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading from checkpoint: {checkpoint_path}")
        print(f"Using active config: {config_name}")

        # Load active config
        if config_name == "config_hough_encoder":
            import config_hough_encoder as config
        else:
            import config_hed_mrz as config

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint for model architecture
    cfg_dict = checkpoint.get("cfg", {})
    img_size = cfg_dict.get("img_size", 384)
    n_angles = cfg_dict.get("n_angles", 128)
    sino_ch = cfg_dict.get("h_sino_channels", 32)

    print(f"Model architecture:")
    print(f"  img_size: {img_size}")
    print(f"  n_angles: {n_angles}")
    print(f"  sino_ch: {sino_ch}")

    # Create model
    model = HEDMRZ(
        img_size=img_size,
        n_angles=n_angles,
        sino_ch=sino_ch
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Print validation metrics if available
    val_iou_seg = checkpoint.get("val_mIoU_seg", "N/A")
    val_iou_obb = checkpoint.get("val_mIoU_obb", "N/A")
    print(f"Model loaded successfully!")
    print(f"  val_mIoU_seg: {val_iou_seg}")
    print(f"  val_mIoU_obb: {val_iou_obb}")

    return model, postprocess_to_obbs, iou_binary, config


@torch.no_grad()
def infer_sample(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

    if img_tensor.shape[1] != 1:
        img_tensor = img_tensor.mean(dim=1, keepdim=True)

    logits = model(img_tensor)  # (1, 1, H, W)
    prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
    binary_mask = (prob_map > 0.5).astype(np.float32)

    return prob_map, binary_mask


def tensor_to_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert image tensor (C, H, W) to numpy array (H, W, 3) for visualization."""
    img = img_tensor.cpu().numpy()  # (C, H, W)

    if img.shape[0] == 1:
        # Grayscale: (1, H, W) -> (H, W, 3)
        img = np.repeat(img[0][..., None], 3, axis=2)
    else:
        # RGB: (3, H, W) -> (H, W, 3)
        img = img.transpose(1, 2, 0)

    # Denormalize from [0, 1] to [0, 255]
    img = (img * 255).astype(np.uint8)

    return img


def create_visualization(
    img_rgb: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    gt_mask: np.ndarray,
    doc_info: str,
    iou: float,
    postprocess_to_obbs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W = prob_map.shape

    # 1. Original image
    vis_original = img_rgb.copy()

    # 2. Probability heatmap
    prob_colored = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)

    # 3. Binary mask overlay on original
    vis_overlay = img_rgb.copy()
    mask_colored = np.zeros_like(img_rgb)
    mask_colored[binary_mask > 0.5] = [0, 255, 0]
    vis_overlay = cv2.addWeighted(vis_overlay, 0.7, mask_colored, 0.3, 0)

    # 4. OBB detection
    vis_obb = img_rgb.copy()
    prob_map_resized = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
    obbs = postprocess_to_obbs(prob_map_resized, orig_hw=(H, W))

    for obb in obbs:
        obb_int = obb.astype(np.int32)
        cv2.drawContours(vis_obb, [obb_int], 0, (0, 255, 0), 2)

        for pt in obb_int:
            cv2.circle(vis_obb, tuple(pt), 5, (255, 0, 0), -1)

    return vis_original, prob_colored, vis_overlay, vis_obb


def generate_pdf_report(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    output_path: str,
    num_samples: int = 50,
    device: str = "cuda",
    checkpoint_info: str = "",
    postprocess_to_obbs = None,
    iou_binary = None
):
    all_samples = []
    all_iou_scores = []

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(test_loader, desc="Running inference")):
            batch_size = imgs.size(0)

            for i in range(batch_size):
                if len(all_samples) >= num_samples:
                    break

                img_tensor = imgs[i]  # (C, H, W)
                gt_mask = masks[i, 0].cpu().numpy()  # (H, W)

                prob_map, binary_mask = infer_sample(model, img_tensor, device)

                pred_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
                gt_mask_tensor = masks[i:i+1]
                iou = iou_binary(pred_mask_tensor, gt_mask_tensor)

                img_rgb = tensor_to_image(img_tensor)

                sample_info = test_loader.dataset.samples[batch_idx * test_loader.batch_size + i]
                doc_type = sample_info.get("doc_type", "unknown")
                modality = sample_info.get("modality", "unknown")
                doc_info = f"{doc_type} ({modality})"

                all_samples.append({
                    "img_rgb": img_rgb,
                    "prob_map": prob_map,
                    "binary_mask": binary_mask,
                    "gt_mask": gt_mask,
                    "doc_info": doc_info,
                    "iou": iou
                })
                all_iou_scores.append(iou)

            if len(all_samples) >= num_samples:
                break

    mean_iou = np.mean(all_iou_scores)
    std_iou = np.std(all_iou_scores)
    median_iou = np.median(all_iou_scores)

    print(f"\nInference Statistics:")
    print(f"  Samples: {len(all_samples)}")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Std IoU: {std_iou:.4f}")
    print(f"  Median IoU: {median_iou:.4f}")

    with PdfPages(output_path) as pdf:

        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, "MRZ Field Segmentation", ha="center", fontsize=24, weight="bold")
        fig.text(0.5, 0.5, "Inference Results on Test Set", ha="center", fontsize=16)
        fig.text(0.5, 0.4, f"Model: Hough Encoder (HEDMRZ)", ha="center", fontsize=12)
        if checkpoint_info:
            fig.text(0.5, 0.37, checkpoint_info, ha="center", fontsize=10, style="italic")
        fig.text(0.5, 0.32, f"Samples: {len(all_samples)}", ha="center", fontsize=12)
        fig.text(0.5, 0.27, f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}", ha="center", fontsize=12)
        fig.text(0.5, 0.22, f"Median IoU: {median_iou:.4f}", ha="center", fontsize=12)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        samples_per_page = 2
        num_pages = (len(all_samples) + samples_per_page - 1) // samples_per_page

        for page_idx in tqdm(range(num_pages), desc="Generating PDF pages"):
            fig, axes = plt.subplots(samples_per_page, 4, figsize=(16, 8))

            if samples_per_page == 1:
                axes = axes.reshape(1, -1)

            for row in range(samples_per_page):
                sample_idx = page_idx * samples_per_page + row

                if sample_idx >= len(all_samples):
                    for col in range(4):
                        axes[row, col].axis("off")
                    continue

                sample = all_samples[sample_idx]

                vis_original, vis_heatmap, vis_overlay, vis_obb = create_visualization(
                    sample["img_rgb"],
                    sample["prob_map"],
                    sample["binary_mask"],
                    sample["gt_mask"],
                    sample["doc_info"],
                    sample["iou"],
                    postprocess_to_obbs
                )

                axes[row, 0].imshow(vis_original)
                axes[row, 0].set_title(f"Original\n{sample['doc_info']}", fontsize=8)
                axes[row, 0].axis("off")

                axes[row, 1].imshow(vis_heatmap)
                axes[row, 1].set_title("Probability Heatmap", fontsize=8)
                axes[row, 1].axis("off")

                axes[row, 2].imshow(vis_overlay)
                axes[row, 2].set_title("Mask Overlay", fontsize=8)
                axes[row, 2].axis("off")

                axes[row, 3].imshow(vis_obb)
                axes[row, 3].set_title(f"OBB Detection\nIoU: {sample['iou']:.3f}", fontsize=8)
                axes[row, 3].axis("off")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    print(f"\nPDF report saved to: {output_path}")

    stats_path = output_path.replace(".pdf", "_stats.txt")
    with open(stats_path, "w") as f:
        f.write("MRZ Field Segmentation - Inference Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples: {len(all_samples)}\n")
        f.write(f"Mean IoU: {mean_iou:.6f}\n")
        f.write(f"Std IoU: {std_iou:.6f}\n")
        f.write(f"Median IoU: {median_iou:.6f}\n")
        f.write(f"Min IoU: {np.min(all_iou_scores):.6f}\n")
        f.write(f"Max IoU: {np.max(all_iou_scores):.6f}\n\n")
        f.write("Per-sample IoU scores:\n")
        for i, (sample, iou) in enumerate(zip(all_samples, all_iou_scores)):
            f.write(f"  {i+1:3d}. {sample['doc_info']:40s} IoU: {iou:.6f}\n")

    print(f"Statistics saved to: {stats_path}")


def main():
    """Main function for CLI inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on MRZ segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a run directory with hough_encoder trainer
  python inference_test.py --run_dir runs/20250123_140530-photo_scan_upright --trainer hough_encoder

  # Use a run directory with hed_mrz trainer and specific checkpoint
  python inference_test.py --run_dir runs/20250123_140530-photo_scan_upright --trainer hed_mrz --checkpoint last.pt

  # Use explicit checkpoint path with active config
  python inference_test.py --checkpoint path/to/model.pt --trainer hough_encoder
        """
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to runs/ directory (loads config and checkpoint from there)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename (default: best.pt) or full path if not using --run_dir"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        choices=["hough_encoder", "hed_mrz"],
        help="Trainer type: 'hough_encoder' or 'hed_mrz'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of test samples to include in PDF report (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: inference_results.pdf in run_dir or current dir)"
    )

    args = parser.parse_args()

    # Determine checkpoint path and output path
    if args.run_dir:
        checkpoint_path = None
        checkpoint_name = args.checkpoint
        output_path = args.output or os.path.join(args.run_dir, "inference_results.pdf")
    else:
        checkpoint_path = args.checkpoint
        checkpoint_name = None
        output_path = args.output or "inference_results.pdf"

    print("=" * 70)
    print("MRZ Field Segmentation - Inference")
    print("=" * 70)

    # Load model
    model, postprocess_to_obbs, iou_binary, config = load_model(
        checkpoint_path=checkpoint_path,
        run_dir=args.run_dir,
        trainer_type=args.trainer,
        checkpoint_name=checkpoint_name,
        device=args.device
    )

    print("\n" + "=" * 70)
    print("Loading test data...")
    print("=" * 70)

    # Create test dataloader
    # Note: This uses the same parameters as training
    # You may want to adjust these based on your needs
    _, _, test_loader = create_dataloaders(
        datasets=["MIDV500", "MIDV2020"],
        data_roots={"MIDV500": "../../data/midv500", "MIDV2020": "../../data/midv2020"},
        split_method="random",
        batch_size=8,
        img_size=getattr(config, 'img_size', 384),
        grayscale=getattr(config, 'grayscale', True),
        num_workers=getattr(config, 'num_workers', 4),
        pin_memory=getattr(config, 'pin_memory', True),
        use_kornia=False,
        midv2020_modalities=getattr(config, 'modalities', ["photo", "scan_upright", "scan_rotated", "clips"]),
    )

    print("\n" + "=" * 70)
    print("Generating PDF report...")
    print("=" * 70)

    # Generate PDF report
    generate_pdf_report(
        model=model,
        test_loader=test_loader,
        output_path=output_path,
        num_samples=args.num_samples,
        device=args.device,
        checkpoint_info=f"Trainer: {args.trainer}",
        postprocess_to_obbs=postprocess_to_obbs,
        iou_binary=iou_binary
    )

    print("\n" + "=" * 70)
    print("Inference completed!")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
