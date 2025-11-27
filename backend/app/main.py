# backend/app/main.py (imports at top)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from io import BytesIO
from PIL import Image
import base64
from .model_stub import get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    heatmap: Optional[str] = None
    description: Optional[str] = None
    log_entry: Optional[Dict] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload JPG or PNG.")

    contents = await image.read()
    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    model = get_model(device="cpu")
    predicted_class, probabilities, heatmap_b64, description, log_info = model.predict(pil_image, save_log=True)

    return PredictionResponse(
        predicted_class=predicted_class,
        probabilities=probabilities,
        heatmap=heatmap_b64,
        description=description,
        log_entry=log_info
    )
