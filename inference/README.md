# MRZ Field Segmentation - Inference System

Docker-based inference system for HED-MRZ models with FastAPI backend and Streamlit frontend.

## Overview

This inference system provides:
- **FastAPI Backend**: REST API for model inference with GPU support
- **Streamlit Frontend**: User-friendly web interface for image upload and visualization
- **Two Models**: hough_encoder and hed_mrz
- **Docker Compose**: Easy deployment with GPU support

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- Trained model checkpoints in `../training/mrz-field-segmentation/runs/`

### 1. Configure Environment

Copy the example environment file and update paths to your model checkpoints:

```bash
cd inference/
cp .env.example .env
# Edit .env to set correct checkpoint paths
```

### 2. Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 3. Access Services

- **Streamlit UI**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Stop Services

```bash
docker-compose down
```

## Architecture

```
┌─────────────────────┐
│  Streamlit Frontend │ (Port 8501)
│   (User Interface)  │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│   FastAPI Backend   │ (Port 8000)
│   (Model Inference) │
└──────────┬──────────┘
           │
           ├─ Volume Mount: training/mrz-field-segmentation/
           └─ Volume Mount: training/mrz-field-segmentation/runs/
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "hough_encoder": true,
    "hed_mrz": true
  },
  "device": "cuda"
}
```

### Inference with hough_encoder

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 test_image.jpg)

# Send inference request
curl -X POST http://localhost:8000/predict/hough_encoder \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/jpeg;base64,${IMAGE_B64}\"}" \
  | jq '.'
```

### Inference with hed_mrz

```bash
curl -X POST http://localhost:8000/predict/hed_mrz \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/jpeg;base64,${IMAGE_B64}\"}" \
  | jq '.'
```

Response format:
```json
{
  "success": true,
  "model_type": "hough_encoder",
  "images": {
    "original": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "obb": "data:image/png;base64,..."
  },
  "error": null
}
```

## Environment Variables

Configure in `.env` file or docker-compose.yml:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOUGH_ENCODER_CHECKPOINT` | Path to hough_encoder model checkpoint | `/workspace/runs/.../best.pt` |
| `HED_MRZ_CHECKPOINT` | Path to hed_mrz model checkpoint | `/workspace/runs/.../best.pt` |
| `DEVICE` | Compute device (cuda/cpu) | `cuda` |
| `MODEL_IMG_SIZE` | Model input image size | `384` |
| `MODEL_N_ANGLES` | Radon transform angles | `128` |
| `MODEL_SINO_CH` | Sinogram channels | `32` |

## Directory Structure

```
inference/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI routes
│   │   ├── models.py        # Pydantic schemas
│   │   ├── inference.py     # Inference pipeline
│   │   ├── model_loader.py  # Model management
│   │   └── config.py        # Configuration
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml       # Service orchestration
├── .env.example             # Example configuration
└── README.md                # This file
```

## Development

### Local Testing (Without Docker)

**Backend:**
```bash
cd backend/
pip install -r requirements.txt

export PYTHONPATH="../training/mrz-field-segmentation:$PYTHONPATH"
export HOUGH_ENCODER_CHECKPOINT="../training/mrz-field-segmentation/runs/.../best.pt"
export HED_MRZ_CHECKPOINT="../training/mrz-field-segmentation/runs/.../best.pt"
export DEVICE="cuda"

uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend/
pip install -r requirements.txt

export BACKEND_URL="http://localhost:8000"

streamlit run app.py
```

### Updating Model Checkpoints

1. Train new models in `../training/mrz-field-segmentation/`
2. Update checkpoint paths in `.env` file
3. Restart services: `docker-compose restart`

No need to rebuild Docker images - checkpoints are mounted as volumes.

## Troubleshooting

### Models Not Loading

- Check checkpoint paths in `.env` match actual file locations
- Verify volumes are mounted correctly: `docker-compose config`
- Check backend logs: `docker-compose logs backend`

### CUDA Out of Memory

- Set `DEVICE=cpu` in `.env` to use CPU inference
- Monitor GPU usage: `nvidia-smi`

### Backend Not Responding

- Check health endpoint: `curl http://localhost:8000/health`
- View logs: `docker-compose logs backend`
- Verify training code is mounted: `docker exec mrz-inference-backend ls /workspace/training/mrz-field-segmentation`

### Frontend Can't Connect

- Ensure backend is healthy: `docker-compose ps`
- Check BACKEND_URL in frontend container: `docker exec mrz-inference-frontend env | grep BACKEND`
- Verify network: `docker network inspect mrz-inference-network`

## Performance

- **Model Loading**: ~10-30 seconds on startup
- **Inference Time**: ~100-500ms per image (GPU) / 1-5s (CPU)
- **Concurrent Requests**: Supported via FastAPI async

## GPU Support

### Requirements

- NVIDIA GPU with CUDA 12.8 support
- NVIDIA Docker runtime installed

### Verification

```bash
# Check GPU is accessible in container
docker-compose exec backend nvidia-smi

# Check PyTorch CUDA availability
docker-compose exec backend python -c "import torch; print(torch.cuda.is_available())"
```

### CPU-Only Mode

If no GPU available, set in `.env`:
```bash
DEVICE=cpu
```

And remove GPU configuration from `docker-compose.yml`:
```yaml
# Comment out or remove:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

## License

Same as parent project.
