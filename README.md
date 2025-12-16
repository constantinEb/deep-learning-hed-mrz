# HED-MRZ: Deep Learning for MRZ Field Segmentation

Deep learning model for detecting Machine Readable Zone (MRZ) regions in passport images using a hybrid architecture combining convolutional and Hough/Radon transform branches.

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets

Download MIDV-2020 and/or MIDV-500 datasets (use download.sh) and place them in:
```
data/midv2020/ 
data/midv500/
```

### 3. Generate Training Masks

```bash
cd scripts
python generate_hed_mrz_masks_midv2020.py
python generate_hed_mrz_masks_midv500.py
```

### 4. Train Models

**HED-MRZ (Hybrid Architecture):**
```bash
cd training/mrz-field-segmentation
python trainer-hed-mrz.py
```

**Hough Encoder:**
```bash
cd training/mrz-field-segmentation
python trainer_hough_encoder.py
```

Configuration files: `config_hed_mrz.py` and `config_hough_encoder.py`

### 5. Run Inference

Create .env in infernce/ with:

HOUGH_ENCODER_CHECKPOINT=/workspace/runs/20251121_161220-photo_scan_upright_scan_rotated_clips/best.pt
HED_MRZ_CHECKPOINT=/workspace/runs/hed-mrz-20251123_132036-photo_scan_upright_scan_rotated_clips/best.pt

# Device Configuration (CPU-only)
DEVICE=cpu

# Model Architecture Parameters (must match training config)
MODEL_IMG_SIZE=384
MODEL_N_ANGLES=128
MODEL_SINO_CH=32

# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend Configuration
FRONTEND_PORT=8501
BACKEND_URL=http://backend:8000

```bash
cd inference/
docker-compose up -d
```

Access the web interface:
- **Streamlit UI**: http://localhost:8501 (upload images and visualize results)
- **API Documentation**: http://localhost:8000/docs

