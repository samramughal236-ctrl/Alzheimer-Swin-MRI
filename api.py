from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import timm
from torchvision import transforms
import torch.nn.functional as F
import os
import io
import urllib.request

app = FastAPI(
    title="Alzheimer MRI Classification API",
    description="Swin Transformer-based MRI Dementia Classification API"
)

# ==========================================
# SETTINGS
# ==========================================

MODEL_URL = (
    "https://huggingface.co/samraAman/alzheimer-swin-model/"
    "resolve/main/best_model.pth"
)

MODEL_PATH = "/tmp/best_model.pth"

CLASSES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# DOWNLOAD MODEL
# ==========================================

def download_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading Swin Transformer model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("Model downloaded successfully.")

    return MODEL_PATH


# ==========================================
# LOAD MODEL
# ==========================================

_model = None


def load_model():

    global _model

    if _model is not None:
        return _model

    model_path = download_model()

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=4
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    _model = model

    return model


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Alzheimer MRI Classification API"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================
# MRI PREDICTION
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded file
        file_bytes = await file.read()

        # Open image
        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        # Load model
        model = load_model()

        # Preprocess
        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)

        # Prediction
        with torch.no_grad():

            output = model(image_tensor)

            probabilities = F.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = CLASSES[
            predicted.item()
        ]

        confidence_value = confidence.item()

        return {
            "prediction": predicted_class,
            "class": predicted_class,
            "label": predicted_class,
            "confidence": confidence_value,
            "confidence_percent": confidence_value * 100,
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(
                    CLASSES,
                    probabilities[0].tolist()
                )
            }
        }

    except Exception as e:

        return {
            "error": str(e)
        }