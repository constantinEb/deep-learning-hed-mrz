"""
Singleton class for loading and managing models.
Models are loaded once at startup and reused across requests.
"""
import sys
import os
from typing import Tuple, Optional, Callable
import torch

# Add training code to Python path
sys.path.insert(0, "/workspace/training/mrz-field-segmentation")

from app.config import settings


class ModelManager:
    """Singleton for managing loaded models"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = settings.DEVICE
        self.models = {}
        self.postprocess_funcs = {}
        self.iou_funcs = {}
        self.configs = {}

        self._initialized = True

    def load_model(
        self,
        model_type: str,
        checkpoint_path: str
    ) -> Tuple[torch.nn.Module, Callable, Callable, object]:
        """
        Load a model from checkpoint.

        Args:
            model_type: "hough_encoder" or "hed_mrz"
            checkpoint_path: Path to .pt checkpoint file

        Returns:
            Tuple of (model, postprocess_to_obbs, iou_binary, config)
        """
        from inference_test import load_model as _load_model

        run_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path)

        print(f"Loading {model_type} model from {checkpoint_path}...")

        model, postprocess_fn, iou_fn, config = _load_model(
            run_dir=run_dir,
            trainer_type=model_type,
            checkpoint_name=checkpoint_name,
            device=self.device
        )

        print(f"{model_type} model loaded successfully!")

        return model, postprocess_fn, iou_fn, config

    def initialize_models(self):
        """Load both models at startup"""
        try:
            if os.path.exists(settings.HOUGH_ENCODER_CHECKPOINT):
                (
                    self.models["hough_encoder"],
                    self.postprocess_funcs["hough_encoder"],
                    self.iou_funcs["hough_encoder"],
                    self.configs["hough_encoder"]
                ) = self.load_model("hough_encoder", settings.HOUGH_ENCODER_CHECKPOINT)
            else:
                print(f"WARNING: hough_encoder checkpoint not found: {settings.HOUGH_ENCODER_CHECKPOINT}")

            if os.path.exists(settings.HED_MRZ_CHECKPOINT):
                (
                    self.models["hed_mrz"],
                    self.postprocess_funcs["hed_mrz"],
                    self.iou_funcs["hed_mrz"],
                    self.configs["hed_mrz"]
                ) = self.load_model("hed_mrz", settings.HED_MRZ_CHECKPOINT)
            else:
                print(f"WARNING: hed_mrz checkpoint not found: {settings.HED_MRZ_CHECKPOINT}")

        except Exception as e:
            print(f"ERROR loading models: {e}")
            raise

    def get_model(self, model_type: str) -> Tuple[torch.nn.Module, Callable, Callable, object]:
        """Get a loaded model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")

        return (
            self.models[model_type],
            self.postprocess_funcs[model_type],
            self.iou_funcs[model_type],
            self.configs[model_type]
        )

    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded"""
        return model_type in self.models


model_manager = ModelManager()
