"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class PredictRequest(BaseModel):
    """Request schema for /predict endpoints"""
    image: str = Field(
        ...,
        description="Base64-encoded image (JPEG, PNG). Example: 'data:image/jpeg;base64,/9j/4AAQ...'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
            }
        }


class PredictResponse(BaseModel):
    """Response schema for /predict endpoints"""
    success: bool = Field(..., description="Whether inference succeeded")
    model_type: Literal["hough_encoder", "hed_mrz"] = Field(..., description="Model type used")
    images: dict = Field(
        ...,
        description="Base64-encoded visualization images (original, heatmap, overlay, obb)"
    )
    error: Optional[str] = Field(None, description="Error message if success=False")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_type": "hough_encoder",
                "images": {
                    "original": "data:image/png;base64,iVBORw0KGgo...",
                    "heatmap": "data:image/png;base64,iVBORw0KGgo...",
                    "overlay": "data:image/png;base64,iVBORw0KGgo...",
                    "obb": "data:image/png;base64,iVBORw0KGgo..."
                },
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """Response schema for /health endpoint"""
    status: str = Field(..., description="Service health status")
    models_loaded: dict = Field(..., description="Status of loaded models")
    device: str = Field(..., description="Device being used (cuda/cpu)")
