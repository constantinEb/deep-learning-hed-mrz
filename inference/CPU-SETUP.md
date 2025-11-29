# CPU-Only Inference Setup

The system has been configured for CPU-only inference. Here are the changes made:

## Changes from GPU Version

1. **docker-compose.yml**:
   - Default device set to `cpu` instead of `cuda`
   - GPU deployment section commented out
   - No NVIDIA runtime required

2. **backend/Dockerfile**:
   - Changed from `nvidia/cuda:12.8.0-runtime-ubuntu22.04` to `python:3.10-slim`
   - Removed CUDA-specific dependencies
   - Lighter Docker image

3. **backend/requirements.txt**:
   - Changed PyTorch index URL from CUDA to CPU version
   - Uses `--extra-index-url https://download.pytorch.org/whl/cpu`

4. **.env**:
   - `DEVICE=cpu` (explicitly set)

## Quick Start

### 1. Verify Checkpoint Paths

Edit `inference/.env` and update the checkpoint paths to match your trained models:

```bash
HOUGH_ENCODER_CHECKPOINT=/workspace/runs/YOUR_HOUGH_RUN/best.pt
HED_MRZ_CHECKPOINT=/workspace/runs/YOUR_HED_MRZ_RUN/best.pt
```

### 2. Build and Start

```bash
cd inference/
docker-compose up --build -d
```

### 3. Monitor Startup

```bash
# Watch backend logs (model loading takes 10-30 seconds)
docker-compose logs -f backend

# Check when models are loaded
docker-compose logs backend | grep "loaded successfully"
```

### 4. Access Services

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Performance Expectations

- **Model Loading**: 15-40 seconds (CPU is slower than GPU)
- **Inference Time**: 1-5 seconds per image (vs 100-500ms on GPU)
- **Memory**: ~2-4GB RAM per model

## Troubleshooting

### Models Taking Too Long to Load

This is normal on CPU. The backend health check has a 60-second start period to account for this.

### Out of Memory

If you run out of RAM:
1. Close other applications
2. Load only one model at a time (comment out one checkpoint in `.env`)
3. Reduce batch size in future training

### Check CPU Usage

```bash
# Monitor container resources
docker stats mrz-inference-backend
```

## Optional: Local Testing (No Docker)

If Docker is slow, you can run locally:

```bash
# Backend
cd inference/backend
pip install -r requirements.txt
export PYTHONPATH="../../training/mrz-field-segmentation:$PYTHONPATH"
export DEVICE=cpu
export HOUGH_ENCODER_CHECKPOINT="../../training/mrz-field-segmentation/runs/.../best.pt"
export HED_MRZ_CHECKPOINT="../../training/mrz-field-segmentation/runs/.../best.pt"
uvicorn app.main:app --reload

# Frontend (in separate terminal)
cd inference/frontend
pip install -r requirements.txt
export BACKEND_URL="http://localhost:8000"
streamlit run app.py
```

## Reverting to GPU

If you later get a GPU, you can revert by:

1. Uncomment GPU section in `docker-compose.yml`
2. Change `DEVICE=cuda` in `.env`
3. Update `backend/Dockerfile` base image to `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
4. Update `backend/requirements.txt` to use CUDA PyTorch
5. Rebuild: `docker-compose up --build`
