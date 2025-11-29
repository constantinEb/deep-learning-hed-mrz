"""
Configuration for FastAPI backend.
Loads from environment variables.
"""
import os
from typing import Optional


class Settings:
    HOUGH_ENCODER_CHECKPOINT: str = os.getenv(
        "HOUGH_ENCODER_CHECKPOINT",
        "/workspace/runs/20251123_125224-photo_scan_upright_scan_rotated_clips/best.pt"
    )
    HED_MRZ_CHECKPOINT: str = os.getenv(
        "HED_MRZ_CHECKPOINT",
        "/workspace/runs/hed-mrz-20251123_132036-photo_scan_upright_scan_rotated_clips/best.pt"
    )

    DEVICE: str = os.getenv("DEVICE", "cuda")

    IMG_SIZE: int = int(os.getenv("MODEL_IMG_SIZE", "384"))
    N_ANGLES: int = int(os.getenv("MODEL_N_ANGLES", "128"))
    SINO_CH: int = int(os.getenv("MODEL_SINO_CH", "32"))

    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    TRAINING_CODE_PATH: str = "/workspace/training/mrz-field-segmentation"


settings = Settings()
