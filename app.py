import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Alzheimer MRI Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Alzheimer MRI Classification")
st.write("Swin Transformer-based MRI Dementia Classification")

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "best_model.pth"

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
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=4
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


# ==========================================
# LOAD MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ best_model.pth not found. "
        "Please place best_model.pth in the same folder as app.py."
    )

    st.stop()

model = load_model()

st.success("✅ Swin Transformer model loaded successfully")

st.write("Device:", device)

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

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

        confidence_value = (
            confidence.item() * 100
        )

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence_value:.2f}%"
        )

        # ==================================
        # ALL CLASS PROBABILITIES
        # ==================================

        st.subheader("Class Probabilities")

        for class_name, probability in zip(
            CLASSES,
            probabilities[0]
        ):

            st.write(
                f"{class_name}: "
                f"{probability.item() * 100:.2f}%"
            )

            st.progress(
                float(probability.item())
            )