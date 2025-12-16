"""
FastAPI application for MRZ field segmentation inference.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.models import PredictRequest, PredictResponse, HealthResponse
from app.inference import run_inference
from app.model_loader import model_manager
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MRZ Field Segmentation API",
    description="Inference API for MRZ (Machine Readable Zone) field segmentation using HED-MRZ models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    try:
        model_manager.initialize_models()
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "MRZ Field Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "hough_encoder": "/predict/hough_encoder",
            "hed_mrz": "/predict/hed_mrz"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "hough_encoder": model_manager.is_model_loaded("hough_encoder"),
            "hed_mrz": model_manager.is_model_loaded("hed_mrz")
        },
        "device": settings.DEVICE
    }


@app.post("/predict/hough_encoder", response_model=PredictResponse, tags=["Inference"])
async def predict_hough_encoder(request: PredictRequest):
    """
    Run inference with hough_encoder model.

    - **image**: Base64-encoded image (JPEG, PNG)

    Returns 4 visualization images: original, heatmap, overlay, obb
    """
    try:
        if not model_manager.is_model_loaded("hough_encoder"):
            raise HTTPException(status_code=503, detail="hough_encoder model not loaded")

        images = run_inference(
            base64_image=request.image,
            model_type="hough_encoder"
        )

        return {
            "success": True,
            "model_type": "hough_encoder",
            "images": images,
            "error": None
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {
            "success": False,
            "model_type": "hough_encoder",
            "images": {},
            "error": str(e)
        }


@app.post("/predict/hed_mrz", response_model=PredictResponse, tags=["Inference"])
async def predict_hed_mrz(request: PredictRequest):
    """
    Run inference with hed_mrz model.

    - **image**: Base64-encoded image (JPEG, PNG)

    Returns 4 visualization images: original, heatmap, overlay, obb
    """
    try:
        if not model_manager.is_model_loaded("hed_mrz"):
            raise HTTPException(status_code=503, detail="hed_mrz model not loaded")

        images = run_inference(
            base64_image=request.image,
            model_type="hed_mrz"
        )

        return {
            "success": True,
            "model_type": "hed_mrz",
            "images": images,
            "error": None
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {
            "success": False,
            "model_type": "hed_mrz",
            "images": {},
            "error": str(e)
        }
