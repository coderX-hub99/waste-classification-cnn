import json

import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import transforms


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Waste Classification CNN",
    page_icon="♻️",
    layout="centered"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. CNN MODEL ARCHITECTURE
# ============================================================

class WasteCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Second convolution block
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            64 * 32 * 32,
            128
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):

        # Convolution block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolution block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


# ============================================================
# 4. LOAD CLASS NAMES
# ============================================================

with open(
    "C:\ML PROJECTS\P4-DEEP LEARNING\CNN Image Classification\class_names.json",
    "r"
) as f:

    class_names = json.load(f)


num_classes = len(class_names)


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = WasteCNN(
    num_classes=num_classes
)


# ============================================================
# 6. LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    torch.load(
        "best_waste_cnn.pth",
        map_location=device
    )
)


# Move model to CPU/GPU
model = model.to(device)


# Evaluation mode
model.eval()


# ============================================================
# 7. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    
    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

def predict_image(
    image,
    model,
    transform,
    classes,
    device
):

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply the same preprocessing
    # used during validation/testing
    image_tensor = transform(image)

    # Add batch dimension
    # [3, 128, 128] → [1, 3, 128, 128]
    image_tensor = image_tensor.unsqueeze(0)

    # Move image to CPU/GPU
    image_tensor = image_tensor.to(device)

    # Evaluation mode
    model.eval()

    # No gradients needed for prediction
    with torch.no_grad():

        # CNN prediction
        outputs = model(image_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Get top 3 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        class_name = classes[index.item()]

        confidence = probability.item()

        results.append(
            (
                class_name,
                confidence
            )
        )

    return results


# ============================================================
# 9. STREAMLIT UI
# ============================================================

st.title("♻️ Waste Classification CNN")

st.write(
    "Upload an image and the trained CNN will "
    "predict its waste category."
)


# Show device
st.caption(
    f"Running on: {device}"
)


# ============================================================
# 10. IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# 11. PROCESS IMAGE
# ============================================================
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        results = predict_image(
            image,
            model,
            transform,
            class_names,
            device
        )

        # ==========================================
        # TOP PREDICTION
        # ==========================================

        predicted_class = results[0][0]
        confidence = results[0][1]

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )


        # ==========================================
        # TOP 3 PREDICTIONS
        # ==========================================

        st.subheader(
            "📊 Top 3 Predictions"
        )

        for class_name, probability in results:

            st.write(
                f"**{class_name}**: "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with PyTorch + CNN + Streamlit"
)