import os
import torch
from datetime import datetime

# Data
modalities = ["photo", "scan_upright", "scan_rotated", "clips"]
filter_passport_only = True

# Splits
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
split_seed = 42

# Loader
num_workers = 16
pin_memory = True

# Model / Training
img_size = 384
grayscale = True
batch_size = 32
epochs = 150
lr = 3e-4
betas = (0.9, 0.99)
weight_decay = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loss
pos_weight = 1.0
bg_weight = 0.5
threshold = 0.5

# Hough/FHT branch
n_angles = 128
radon_pad = 1.0
h_sino_channels = 32

# Augmentation
aug_perspective = True
aug_rotate = True
aug_blur = True
aug_brightness_contrast = True
aug_compression = True
aug_perspective_scale = (0.05, 0.12)
aug_rotate_limit = 25
aug_blur_limit = 7
aug_brightness_limit = 0.3
aug_contrast_limit = 0.3
aug_compression_quality = (60, 100)
aug_prob = 0.95

# Logging
out_dir = "runs/mrz-segmentation"
save_every = 50
profile_timing = True
log_file = "training_log.csv"


def generate_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = "_".join(modalities)
    run_name = f"{timestamp}-{datasets_str}"
    return os.path.join("runs", run_name)


def to_dict():
    import sys
    current_module = sys.modules[__name__]
    return {
        k: v for k, v in vars(current_module).items()
        if not k.startswith('_')
        and not callable(v)
        and k not in ['sys', 'os', 'torch', 'datetime']
        and not k.startswith('__')
    }


def get_aug_config():
    return {
        'aug_perspective': aug_perspective,
        'aug_rotate': aug_rotate,
        'aug_blur': aug_blur,
        'aug_brightness_contrast': aug_brightness_contrast,
        'aug_compression': aug_compression,
        'aug_perspective_scale': aug_perspective_scale,
        'aug_rotate_limit': aug_rotate_limit,
        'aug_blur_limit': aug_blur_limit,
        'aug_brightness_limit': aug_brightness_limit,
        'aug_contrast_limit': aug_contrast_limit,
        'aug_compression_quality': aug_compression_quality,
        'aug_prob': aug_prob,
    }
