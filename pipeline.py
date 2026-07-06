import cv2
import numpy as np
import base64

def inspect_image_buffer(image_bytes: bytes) -> dict:
    """
    Processes raw image bytes through an adaptive HSV color isolation matrix
    to calculate defect surface area coverage.
    """
    # Decode image array from memory buffer
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image bytes into valid matrix")

    # Image dimensional bounds
    h, w, _ = img.shape
    total_area = h * w

    # Convert to HSV color space for color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Broad masking profile targeting active rot/wilt discoloration spectra
    lower_defect = np.array([10, 30, 30])
    upper_defect = np.array([28, 255, 220])
    
    # Isolate defect area masks
    mask = cv2.inRange(hsv, lower_defect, upper_defect)
    
    # Clean noise using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Count total pixels within the threshold mask
    defect_area = int(cv2.countNonZero(thresh))
    defect_ratio = float(defect_area / total_area) if total_area > 0 else 0.0

    # Draw tracking contours on a canvas duplicate for UI display
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

    # Encode visualizations back to base64 text streams
    _, thresh_buf = cv2.imencode('.png', thresh)
    _, contour_buf = cv2.imencode('.png', contour_img)

    thresh_b64 = base64.b64encode(thresh_buf).decode('utf-8')
    contour_b64 = base64.b64encode(contour_buf).decode('utf-8')

    # FIXED: Normalized return dictionary keys with '_px' suffix to align perfectly with main.py expectations
    return {
        "total_surface_area_px": total_area,
        "defect_area_px": defect_area,
        "defect_ratio": defect_ratio,
        "visualizations": {
            "thresh": f"data:image/png;base64,{thresh_b64}",
            "contours": f"data:image/png;base64,{contour_b64}"
        }
    }
