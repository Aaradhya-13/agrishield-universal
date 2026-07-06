import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Import local pipeline engines
from pipeline import inspect_image_buffer
from classifier import classify_produce_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrishield-api")

app = FastAPI(
    title="AgriShield Universal API",
    description="Production inspection and logistics grid processing engine",
    version="1.0.0"
)

# Pydantic schemas for structured validation
class SurfaceMetrics(BaseModel):
    total_surface_area_px: int
    defect_area_px: int
    defect_ratio: float

class InspectionResponse(BaseModel):
    success: bool
    produce_type: str
    confidence: float
    surface_metrics: SurfaceMetrics
    visualizations: Dict[str, str]
    verdict: str
    error: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# FIXED: Changed endpoint path from /api/v1/inspect to /analyze to match index.html
@app.post("/analyze", response_model=InspectionResponse)
async def inspect_produce(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    try:
        # Read raw image bytes
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # 1. Run Classification Layer (Zero-shot transformer model)
        classification = classify_produce_type(contents)
        produce_type = classification.get("label", "Unknown")
        confidence = classification.get("confidence", 0.0)

        # 2. Run Computer Vision Pipeline (Color isolation matrix)
        metrics = inspect_image_buffer(contents)
        
        # Determine health verdict based on surface area ratio thresholds
        defect_ratio = metrics.get("defect_ratio", 0.0)
        if defect_ratio > 0.15:
            verdict = "REJECTED"
        elif defect_ratio > 0.05:
            verdict = "WARNING"
        else:
            verdict = "APPROVED"

        # FIXED: Mapped keys exactly to match the corrected pipeline.py return keys
        return InspectionResponse(
            success=True,
            produce_type=produce_type,
            confidence=confidence,
            surface_metrics=SurfaceMetrics(
                total_surface_area_px=metrics.get("total_surface_area_px", 0),
                defect_area_px=metrics.get("defect_area_px", 0),
                defect_ratio=defect_ratio
            ),
            visualizations=metrics.get("visualizations", {"thresh": "", "contours": ""}),
            verdict=verdict
        )

    except Exception as e:
        logger.error(f"Inspection processing failed: {str(e)}")
        return InspectionResponse(
            success=False,
            produce_type="Unknown",
            confidence=0.0,
            surface_metrics=SurfaceMetrics(total_surface_area_px=0, defect_area_px=0, defect_ratio=0.0),
            visualizations={"thresh": "", "contours": ""},
            verdict="ERROR",
            error=str(e)
        )

# --------------------------------------------------------------------------- #
# Static Dashboard Configuration
# FIXED: Points directly to the root execution directory instead of a /static folder
# --------------------------------------------------------------------------- #
_current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=_current_dir, html=True), name="static")
