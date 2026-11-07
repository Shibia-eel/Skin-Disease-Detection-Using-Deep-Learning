import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "dataset_8class"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "resnet18_skin_model_improved.pth"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("       RESNET18 MODEL EVALUATION")
print("========================================")
print("Device:", device)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)


# ============================================================
# VALIDATION TRANSFORM
# IMPORTANT: NO RANDOM AUGMENTATION
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

print("\nTotal images:", len(dataset))


# ============================================================
# RECREATE SAME 80/20 SPLIT
# ============================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(dataset),
    generator=generator
).tolist()

val_indices = indices[train_size:]

val_dataset = Subset(
    dataset,
    val_indices
)

print("Validation images:", len(val_dataset))


# ============================================================
# DATA LOADER
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD RESNET18
# ============================================================

print("\nLoading trained ResNet18...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("          EVALUATION RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print("========================================")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("----------------------------------------")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nCONFUSION MATRIX")
print("----------------------------------------")
print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Skin Disease Classification - Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

# Display numbers inside cells
for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )
)

print("\nConfusion matrix saved as:")

print(
    os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )
)

print("\nConfusion matrix saved as:")

print(
    os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )
)