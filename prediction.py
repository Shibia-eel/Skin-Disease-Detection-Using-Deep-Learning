import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =================================================
# PATHS
# =================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stage 1: Normal / Diseased
NORMAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "normal_skin_detector.pth"
)

# Stage 2: 8 Disease Classification
DISEASE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "resnet18_skin_model_improved.pth"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

# =================================================
# DEVICE
# =================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# =================================================
# LOAD DISEASE CLASS NAMES
# =================================================

with open(CLASS_NAMES_PATH, "r") as f:
    disease_classes = json.load(f)

print("Disease Classes:", disease_classes)

# =================================================
# IMAGE TRANSFORMATION
# =================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =================================================
# STAGE 1 MODEL
# NORMAL / DISEASED
# =================================================

print("\nLoading Normal Skin Detector...")

normal_model = models.resnet18(weights=None)

normal_model.fc = nn.Linear(
    normal_model.fc.in_features,
    2
)

normal_model.load_state_dict(
    torch.load(
        NORMAL_MODEL_PATH,
        map_location=device
    )
)

normal_model = normal_model.to(device)
normal_model.eval()

# IMPORTANT:
# ImageFolder gave:
# 0 = diseased
# 1 = normal

normal_classes = [
    "diseased",
    "normal"
]

# =================================================
# STAGE 2 MODEL
# 8 DISEASE CLASSIFICATION
# =================================================

print("Loading 8-Class Disease Model...")

disease_model = models.resnet18(weights=None)

disease_model.fc = nn.Linear(
    disease_model.fc.in_features,
    len(disease_classes)
)

disease_model.load_state_dict(
    torch.load(
        DISEASE_MODEL_PATH,
        map_location=device
    )
)

disease_model = disease_model.to(device)
disease_model.eval()

print("Both models loaded successfully!")

# =================================================
# PREDICTION FUNCTION
# =================================================

def predict_image(image_path):

    # ---------------------------------------------
    # OPEN IMAGE
    # ---------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # ---------------------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------------------

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # =================================================
    # STAGE 1: NORMAL / DISEASED
    # =================================================

    with torch.no_grad():

        output = normal_model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    stage1_class = normal_classes[
        predicted.item()
    ]

    stage1_confidence = confidence.item() * 100

    # =================================================
    # IF NORMAL
    # =================================================

    if stage1_class == "normal":

        return {
            "status": "normal",
            "class": "Normal Skin",
            "confidence": stage1_confidence
        }

    # =================================================
    # IF DISEASED → STAGE 2
    # =================================================

    with torch.no_grad():

        output = disease_model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    disease = disease_classes[
        predicted.item()
    ]

    disease_confidence = confidence.item() * 100

    return {
        "status": "diseased",
        "class": disease,
        "confidence": disease_confidence,
        "stage1_confidence": stage1_confidence
    }


# =================================================
# MAIN PROGRAM
# =================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       TWO-STAGE SKIN DETECTION")
    print("========================================")

    image_path = input(
        "\nEnter image path: "
    ).strip('"')

    # ---------------------------------------------
    # CHECK IMAGE
    # ---------------------------------------------

    if not os.path.exists(image_path):

        print("\nERROR: Image not found.")

    else:

        try:

            result = predict_image(image_path)

            print("\n========================================")
            print("          PREDICTION RESULT")
            print("========================================")

            # -------------------------------------
            # NORMAL RESULT
            # -------------------------------------

            if result["status"] == "normal":

                print("Skin Status    : NORMAL")
                print("Prediction     : Normal Skin")
                print(
                    f"Confidence     : "
                    f"{result['confidence']:.2f}%"
                )

            # -------------------------------------
            # DISEASE RESULT
            # -------------------------------------

            else:

                print("Skin Status    : DISEASED")
                print(
                    "Disease        : "
                    + result["class"]
                )

                print(
                    f"Disease Confidence : "
                    f"{result['confidence']:.2f}%"
                )

                print(
                    f"Stage 1 Confidence : "
                    f"{result['stage1_confidence']:.2f}%"
                )

            print("========================================")

        except Exception as e:

            print("\nERROR during prediction:")
            print(e)