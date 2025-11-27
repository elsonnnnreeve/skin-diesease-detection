from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .model_stub import predict_with_model

app = FastAPI(title="Skin Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Expects: multipart/form-data with key "image".
    Returns: dict with predictions and heatmap_b64 (base64 PNG).
    """
    contents = await image.read()
    result = predict_with_model(contents)
    return result
