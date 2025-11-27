# backend/app/model_stub.py
# Simple model stub — replace predict_with_model with real model code later.

from PIL import Image
import io, base64, numpy as np

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

def preprocess_image_bytes(image_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    return arr

def predict_with_model(image_bytes):
    """
    Dummy predictor:
    - returns a uniform softmax over classes (placeholder)
    - returns a white heatmap PNG encoded as base64
    """
    # pretend we ran inference
    probs = np.ones(len(CLASSES)) / len(CLASSES)
    top_idx = int(np.argmax(probs))
    pred = {"class": CLASSES[top_idx], "probability": float(probs[top_idx])}

    # Create a dummy heatmap (white image)
    hm = Image.new("RGB", (224,224), color=(255,255,255))
    buf = io.BytesIO()
    hm.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"predictions": [pred], "heatmap_b64": heatmap_b64}
