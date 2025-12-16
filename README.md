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

```bash
cd training/mrz-field-segmentation
python inference_test.py --run_dir runs/hed-mrz-YYYYMMDD_HHMMSS-modalities --trainer hed_mrz
```

Replace `YYYYMMDD_HHMMSS` with training run timestamp. PDF reports are auto-generated.
