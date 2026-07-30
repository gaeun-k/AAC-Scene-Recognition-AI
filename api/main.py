from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import CLIPImageProcessor

from models.clip import CLIPClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_model.pt"
MODEL_NAME = "openai/clip-vit-base-patch32"
CLASS_NAMES = [
    "Cafe",
    "Convenience Store",
    "Hospital",
    "Pharmacy",
    "Public Transport",
    "Restaurant",
]


def load_model(device: torch.device) -> CLIPClassifier:
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {WEIGHTS_PATH}")

    model = CLIPClassifier(
        model_name=MODEL_NAME,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False,
    )

    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    app.state.model = load_model(device)
    yield


app = FastAPI(
    title="AAC Scene Recognition API",
    version="1.0.0",
    description="Returns the scene predicted by the trained CLIP classifier.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AAC Scene Recognition API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """Classify one JPEG, PNG, WebP, or BMP image uploaded as ``file``."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported.")

    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    inputs = app.state.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(app.state.device)

    with torch.inference_mode():
        probabilities = torch.softmax(app.state.model(pixel_values), dim=1)[0]
        class_index = int(torch.argmax(probabilities).item())

    return {
        "scene": CLASS_NAMES[class_index],
        "score": round(float(probabilities[class_index].item()), 4),
    }
