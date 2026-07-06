"""
api/pipeline.py

Computer Vision Layer for AgriShield Universal.

Responsible for turning raw image bytes into an accurate surface-defect
percentage. The pipeline is deliberately built in stages so that each
source of error (background clutter, flash glare, shadow) is stripped
out *before* the actual rot/necrosis detector ever sees the pixels.

Pipeline stages
---------------
1. Decode + normalize   -> safe decode, resize to a stable working size
2. Produce isolation    -> separate the item from plate/table/background
3. Glare & shadow purge -> remove blown-out highlights and deep shadow
4. Defect detection     -> adaptive thresholding on Cr/Cb (YCrCb) to
                           find necrotic / mold pixels that deviate from
                           the item's own healthy color cluster
5. Cleanup + scoring    -> morphological cleanup, area math, guard rails
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, asdict


# --------------------------------------------------------------------------- #
# Tunable constants (kept in one place so accuracy can be tuned without
# hunting through the algorithm body)
# --------------------------------------------------------------------------- #

WORKING_MAX_DIM = 800          # px, longest side after resize
MIN_PRODUCE_AREA_RATIO = 0.02  # ignore contours smaller than 2% of frame
GLARE_VALUE_THRESH = 235       # V channel (HSV) above this = potential glare
GLARE_SAT_THRESH = 60          # S channel (HSV) below this = potential glare
SHADOW_VALUE_THRESH = 35       # V channel below this = potential deep shadow
ADAPTIVE_BLOCK_SIZE = 35       # must be odd; local neighborhood for threshold
ADAPTIVE_C = 5                 # constant subtracted in adaptive threshold
MORPH_KERNEL_SIZE = 5
MIN_DEFECT_BLOB_AREA = 25       # px, discard sub-pixel noise speckles


class ImageDecodeError(ValueError):
    """Raised when the uploaded bytes cannot be decoded into an image."""


class EmptyProduceRegionError(ValueError):
    """Raised when no plausible produce region could be isolated."""


@dataclass
class DefectResult:
    defect_percent: float          # 0-100, accurate surface rot/mold coverage
    produce_pixel_count: int       # pixels counted as the actual item
    defect_pixel_count: int        # pixels counted as necrotic/mold
    glare_pixels_removed: int      # pixels excluded as camera flash glare
    shadow_pixels_removed: int     # pixels excluded as background shadow
    frame_width: int
    frame_height: int

    def to_dict(self) -> dict:
        return asdict(self)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ImageDecodeError("Received an empty file.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    if buffer.size == 0:
        raise ImageDecodeError("Received an empty file.")

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageDecodeError(
            "File could not be decoded as an image (corrupt or unsupported format)."
        )
    return image


def _resize_stable(image: np.ndarray, max_dim: int = WORKING_MAX_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _isolate_produce_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Stage 2: separate the produce item from plates, tables, and other
    background clutter.

    Strategy: blur -> grayscale -> Otsu threshold to get a rough
    foreground/background split, then keep only the single largest
    connected contour (the produce item is assumed to be the dominant
    object in frame). This avoids counting background texture as defect
    surface later on.
    """
    blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Otsu handles both bright (white plate) and dark (wood table) backgrounds
    # reasonably well since it picks the threshold from the image's own
    # bimodal histogram rather than a fixed constant.
    _, otsu_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # THRESH_BINARY_INV assumes background tends brighter than the object.
    # If that assumption is wrong (dark background, bright produce), the
    # mask will cover more than half the frame -- flip it back.
    if cv2.countNonZero(otsu_mask) > 0.6 * otsu_mask.size:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise EmptyProduceRegionError(
            "Could not isolate a produce region from the background."
        )

    largest = max(contours, key=cv2.contourArea)
    frame_area = bgr.shape[0] * bgr.shape[1]
    if cv2.contourArea(largest) < MIN_PRODUCE_AREA_RATIO * frame_area:
        raise EmptyProduceRegionError(
            "Detected produce region is too small to analyze reliably."
        )

    produce_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(produce_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return produce_mask


def _strip_glare_and_shadow(bgr: np.ndarray, produce_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Stage 3: within the isolated produce region, exclude camera flash
    glare (very bright, desaturated) and deep background-adjacent shadow
    (very dark) so neither is mistaken for necrotic tissue.

    Returns the refined mask plus pixel counts removed for each cause,
    used purely for transparency/telemetry in the API response.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    glare_mask = ((v > GLARE_VALUE_THRESH) & (s < GLARE_SAT_THRESH)).astype(np.uint8) * 255
    shadow_mask = (v < SHADOW_VALUE_THRESH).astype(np.uint8) * 255

    glare_in_produce = cv2.bitwise_and(glare_mask, produce_mask)
    shadow_in_produce = cv2.bitwise_and(shadow_mask, produce_mask)

    exclusion = cv2.bitwise_or(glare_in_produce, shadow_in_produce)
    refined_mask = cv2.bitwise_and(produce_mask, cv2.bitwise_not(exclusion))

    # Light morphological close to patch tiny holes left by pixel-level exclusion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

    glare_count = int(cv2.countNonZero(glare_in_produce))
    shadow_count = int(cv2.countNonZero(shadow_in_produce))
    return refined_mask, glare_count, shadow_count


def _detect_necrotic_defects(bgr: np.ndarray, refined_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Stage 4: the actual rot/mold detector.

    Works in YCrCb because the Cr (red-difference) and Cb (blue-difference)
    channels separate color shift from lighting brightness far better than
    RGB/HSV alone -- necrotic and moldy tissue tends to drift away from a
    healthy item's own dominant Cr/Cb cluster even under uneven lighting.

    Adaptive Gaussian thresholding (rather than one fixed global cutoff) is
    used so the detector adjusts to each item's own local color rather than
    assuming all tomatoes, or all produce, share one universal "healthy red".
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    produce_pixels = refined_mask > 0
    if not np.any(produce_pixels):
        raise EmptyProduceRegionError("No usable produce pixels remained after cleanup.")

    # Establish this specific item's own healthy-color baseline rather than
    # a hardcoded constant, so the model generalizes across crop types.
    healthy_cr_median = float(np.median(cr[produce_pixels]))
    healthy_cb_median = float(np.median(cb[produce_pixels]))

    block_size = ADAPTIVE_BLOCK_SIZE if ADAPTIVE_BLOCK_SIZE % 2 == 1 else ADAPTIVE_BLOCK_SIZE + 1

    cr_adaptive = cv2.adaptiveThreshold(
        cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, ADAPTIVE_C,
    )
    cb_adaptive = cv2.adaptiveThreshold(
        cb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, ADAPTIVE_C,
    )

    # A pixel only counts as a defect if BOTH channels disagree with the
    # item's own healthy baseline by a meaningful margin AND the adaptive
    # threshold flags it locally -- this dual-channel agreement cuts down
    # on single-channel false positives from lighting gradients.
    cr_deviation = np.abs(cr.astype(np.int16) - int(healthy_cr_median)) > 12
    cb_deviation = np.abs(cb.astype(np.int16) - int(healthy_cb_median)) > 12
    channel_agreement = cr_deviation & cb_deviation

    adaptive_flag = cv2.bitwise_or(cr_adaptive, cb_adaptive) > 0

    defect_raw = (channel_agreement & adaptive_flag & produce_pixels).astype(np.uint8) * 255

    # Stage 5: morphological cleanup to drop noise specks and fill small
    # gaps inside genuine defect blobs.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    defect_clean = cv2.morphologyEx(defect_raw, cv2.MORPH_OPEN, kernel)
    defect_clean = cv2.morphologyEx(defect_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        defect_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final_defect_mask = np.zeros_like(defect_clean)
    for c in contours:
        if cv2.contourArea(c) >= MIN_DEFECT_BLOB_AREA:
            cv2.drawContours(final_defect_mask, [c], -1, 255, thickness=cv2.FILLED)

    defect_pixel_count = int(cv2.countNonZero(final_defect_mask))
    return final_defect_mask, defect_pixel_count


def analyze_defects(image_bytes: bytes) -> DefectResult:
    """
    Public entry point used by the FastAPI gateway.

    Runs the full 5-stage pipeline and returns a DefectResult. Raises
    ImageDecodeError or EmptyProduceRegionError on unusable input so the
    API layer can translate those into clean HTTP error responses instead
    of a raw 500 / zero-division crash.
    """
    bgr = _decode_image(image_bytes)
    bgr = _resize_stable(bgr)

    produce_mask = _isolate_produce_mask(bgr)
    refined_mask, glare_count, shadow_count = _strip_glare_and_shadow(bgr, produce_mask)
    defect_mask, defect_pixel_count = _detect_necrotic_defects(bgr, refined_mask)

    produce_pixel_count = int(cv2.countNonZero(refined_mask))

    # Guard rail: never divide by zero, even if upstream checks are ever
    # loosened later. A produce region with 0 usable pixels reads as 0%
    # defect rather than crashing the request.
    if produce_pixel_count <= 0:
        defect_percent = 0.0
    else:
        defect_percent = (defect_pixel_count / produce_pixel_count) * 100.0
        defect_percent = max(0.0, min(100.0, round(defect_percent, 2)))

    h, w = bgr.shape[:2]
    return DefectResult(
        defect_percent=defect_percent,
        produce_pixel_count=produce_pixel_count,
        defect_pixel_count=defect_pixel_count,
        glare_pixels_removed=glare_count,
        shadow_pixels_removed=shadow_count,
        frame_width=w,
        frame_height=h,
    )
