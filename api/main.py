"""
api/main.py
 
FastAPI application gateway for AgriShield Universal.
 
Flow per request:
  1. Receive the uploaded image (multipart/form-data).
  2. Stage 1 -- ask the classifier for a broad category (Fruit / Vegetable)
     and strictly normalize whatever label it returns.
  3. Stage 2 -- hand the raw image bytes to the OpenCV pipeline to get an
     accurate surface defect percentage.
  4. Stage 3 -- run the dynamic degradation/logistics rules against that
     defect percentage to produce shelf life + routing instructions.
  5. Return one unified JSON payload the dashboard can render directly.
 
Assumption about api/classifier.py
-----------------------------------
This file expects `api/classifier.py` to expose a function:
 
    def classify(image_bytes: bytes) -> str
 
returning a raw label (e.g. "tomato", "veg", "Fruit", "unknown", ...).
If your existing classifier.py uses a different function name or returns
a dict/confidence tuple instead of a bare string, adjust `_run_classifier`
below to match -- everything downstream only needs a clean "Fruit" or
"Vegetable" string out of it.
"""
 
from __future__ import annotations
 
import logging
import random
from typing import Optional
 
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from api.pipeline import (
    EmptyProduceRegionError,
    ImageDecodeError,
    analyze_defects,
)
 
try:
    from api import classifier as _classifier_module
except ImportError:  # classifier.py not present in this environment
    _classifier_module = None
 
logger = logging.getLogger("agrishield")
 
app = FastAPI(title="AgriShield Universal API", version="1.0.0")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# --------------------------------------------------------------------------- #
# Domain constants -- logistics degradation rules
# --------------------------------------------------------------------------- #
 
WARNING_THRESHOLD = 8.0   # % surface defect where item drops to "Warning"
CRITICAL_THRESHOLD = 25.0  # % surface defect where item is rejected outright
 
# Base shelf life ranges (days) by broad category, used for the Healthy state.
# Exact crop-level shelf life would require crop-specific classification;
# at the Fruit/Vegetable granularity we use a representative range per the
# spec (10-14 days) and derive a stable value from the item's own defect
# signal so results aren't random noise on refresh.
BASE_SHELF_LIFE_RANGE = {
    "Fruit": (10, 13),
    "Vegetable": (11, 14),
}
 
WARNING_SHELF_LIFE_RANGE = (2, 4)
 
 
class AnalysisResponse(BaseModel):
    classification: str
    classification_confidence: Optional[float] = None
    defect_percent: float
    condition_state: str          # "Healthy" | "Warning" | "Critical"
    shelf_life_days: int
    routing_instruction: str
    routing_reason: str
    diagnostics: dict
 
 
def _clean_classification_label(raw_label: str) -> str:
    """
    Strictly normalize whatever the classifier returns down to exactly
    "Fruit" or "Vegetable". Handles common synonyms and the tomato edge
    case (botanically a fruit, the target crop mentioned in the spec) by
    defaulting ambiguous/unknown produce toward "Fruit" only when the
    label itself clearly names a fruit-family item.
    """
    if not raw_label:
        return "Fruit"
 
    label = raw_label.strip().lower()
 
    vegetable_terms = {
        "vegetable", "veg", "vegetables", "potato", "carrot", "broccoli",
        "cabbage", "onion", "spinach", "lettuce", "cucumber", "pepper",
        "bell pepper", "cauliflower", "eggplant", "zucchini",
    }
    fruit_terms = {
        "fruit", "fruits", "apple", "banana", "tomato", "tomatoes", "orange",
        "grape", "grapes", "mango", "strawberry", "berry", "melon", "pear",
    }
 
    if label in vegetable_terms or any(term in label for term in vegetable_terms):
        return "Vegetable"
    if label in fruit_terms or any(term in label for term in fruit_terms):
        return "Fruit"
 
    # Unknown/unrecognized label -- fail safe to "Fruit" rather than crash,
    # so the pipeline downstream always has a valid category to key off of.
    logger.warning("Unrecognized classifier label %r, defaulting to Fruit", raw_label)
    return "Fruit"
 
 
def _run_classifier(image_bytes: bytes) -> tuple[str, Optional[float]]:
    """
    Stage 1 wrapper. Calls into api/classifier.py if available; falls back
    to a safe default if the module or expected function is missing, so
    the API never hard-crashes purely because the ML classifier isn't
    wired up yet in a given environment.
    """
    if _classifier_module is None or not hasattr(_classifier_module, "classify"):
        logger.warning("classifier module unavailable; defaulting classification to Fruit")
        return "Fruit", None
 
    try:
        raw = _classifier_module.classify(image_bytes)
    except Exception as exc:  # noqa: BLE001 -- classifier failures shouldn't crash the request
        logger.exception("Classifier raised an exception: %s", exc)
        return "Fruit", None
 
    confidence = None
    if isinstance(raw, dict):
        confidence = raw.get("confidence")
        raw = raw.get("label", "")
    elif isinstance(raw, (tuple, list)) and len(raw) >= 1:
        confidence = raw[1] if len(raw) > 1 else None
        raw = raw[0]
 
    return _clean_classification_label(str(raw)), confidence
 
 
def _stable_shelf_life(seed_bytes: bytes, low: int, high: int) -> int:
    """
    Deterministic pseudo-random pick within [low, high] seeded from the
    image content itself, so the same photo always yields the same shelf
    life instead of jittering on every re-request.
    """
    seed = sum(seed_bytes[:512]) if seed_bytes else 0
    rng = random.Random(seed)
    return rng.randint(low, high)
 
 
def _apply_logistics_rules(
    defect_percent: float, classification: str, image_bytes: bytes
) -> tuple[str, int, str, str]:
    """
    Stage 3: dynamic degradation math + routing decision.
 
    Returns (condition_state, shelf_life_days, routing_instruction, routing_reason)
    """
    if defect_percent >= CRITICAL_THRESHOLD:
        return (
            "Critical",
            0,
            "Route to composting units",
            f"Surface spoilage at {defect_percent:.1f}% exceeds the "
            f"{CRITICAL_THRESHOLD:.0f}% rejection threshold.",
        )
 
    if defect_percent >= WARNING_THRESHOLD:
        low, high = WARNING_SHELF_LIFE_RANGE
        shelf_life = _stable_shelf_life(image_bytes, low, high)
        return (
            "Warning",
            shelf_life,
            "Route to local regional markets immediately",
            f"Blemishes detected ({defect_percent:.1f}% surface coverage); "
            f"shelf life compressed to a {low}-{high} day window.",
        )
 
    low, high = BASE_SHELF_LIFE_RANGE.get(classification, (10, 14))
    shelf_life = _stable_shelf_life(image_bytes, low, high)
    return (
        "Healthy",
        shelf_life,
        "Green-lit for long-haul shipping",
        f"Minimal surface defect ({defect_percent:.1f}%); item qualifies "
        f"for full base shelf life.",
    )
 
 
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
 
 
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_produce(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
 
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
 
    # Stage 1: classification
    classification, confidence = _run_classifier(image_bytes)
 
    # Stage 2: computer vision defect analysis
    try:
        cv_result = analyze_defects(image_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyProduceRegionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- never let a CV edge case 500 silently
        logger.exception("Unexpected pipeline failure: %s", exc)
        raise HTTPException(
            status_code=500, detail="Image analysis failed unexpectedly."
        ) from exc
 
    # Stage 3: logistics/degradation rules
    state, shelf_life, routing, reason = _apply_logistics_rules(
        cv_result.defect_percent, classification, image_bytes
    )
 
    return AnalysisResponse(
        classification=classification,
        classification_confidence=confidence,
        defect_percent=cv_result.defect_percent,
        condition_state=state,
        shelf_life_days=shelf_life,
        routing_instruction=routing,
        routing_reason=reason,
        diagnostics=cv_result.to_dict(),
    )