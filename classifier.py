import io
from PIL import Image
from transformers import pipeline

try:
    # Open-vocabulary zero-shot classification engine
    classifier_engine = pipeline("zero-shot-image-classification", model="google/siglip-base-patch16-224")
except Exception:
    classifier_engine = None

def identify_crop_type(image_bytes: bytes) -> str:
    """
    Forces the vision network to evaluate broad biological categories.
    Hides specific crop names and strictly outputs 'Fruit' or 'Vegetable'.
    """
    if classifier_engine is None:
        return "Fruit / Vegetable"
        
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # We strictly limit the model's choices to these broad categories
        candidate_labels = ["fruit", "vegetable"]
        
        predictions = classifier_engine(
            image, 
            candidate_labels=candidate_labels,
            hypothesis_template="this is a picture of a {}"
        )
        
        if predictions and len(predictions) > 0:
            return predictions[0]['label'].title()
            
        return "Fruit / Vegetable"
        
    except Exception:
        return "Fruit / Vegetable"
    # Alias to match what api/main.py is looking to call
classify = identify_crop_typecd agrishield-universal