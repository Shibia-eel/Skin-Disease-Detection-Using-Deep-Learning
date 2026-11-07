import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ============================================================
# SETTINGS
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

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

# Smaller learning rate for fine-tuning
LEARNING_RATE = 0.0001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("      IMPROVED SKIN DISEASE RESNET18")
print("========================================")
print("Device:", device)
print("Dataset:", DATASET_DIR)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATA
# ============================================================

train_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

val_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

class_names = train_full.classes

print("\nClasses:", class_names)
print("Total images:", len(train_full))


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

targets = train_full.targets

indices = list(range(len(targets)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=targets
)

print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))


# ============================================================
# CREATE SUBSETS
# ============================================================

train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    val_full,
    val_indices
)


# ============================================================
# BALANCED SAMPLING
# ============================================================

train_targets = [
    targets[i]
    for i in train_indices
]

class_counts = torch.bincount(
    torch.tensor(train_targets),
    minlength=len(class_names)
).float()

print("\nTraining class counts:")

for i, name in enumerate(class_names):
    print(
        f"{name}: {int(class_counts[i])}"
    )


# Calculate weight for each class
class_weights = 1.0 / class_counts

# Weight for each individual training sample
sample_weights = torch.tensor([
    class_weights[label]
    for label in train_targets
], dtype=torch.double)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

# Use normal CrossEntropyLoss because
# WeightedRandomSampler already balances classes.
criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)


# ============================================================
# TRAINING
# ============================================================

best_macro_f1 = 0.0

for epoch in range(EPOCHS):

    print("\n----------------------------------------")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("----------------------------------------")


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    correct = 0
    total = 0
    running_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )


    train_accuracy = (
        100 * correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )


    val_accuracy = (
        100 * val_correct / val_total
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    print(
        "\nTraining Accuracy:",
        f"{train_accuracy:.2f}%"
    )

    print(
        "Validation Accuracy:",
        f"{val_accuracy:.2f}%"
    )

    print(
        "Validation Macro F1:",
        f"{macro_f1 * 100:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("\n*** Best model saved ***")


    scheduler.step()


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("       IMPROVED TRAINING COMPLETE")
print("========================================")

print(
    f"Best Macro F1: "
    f"{best_macro_f1 * 100:.2f}%"
)

print("\nModel saved to:")
print(MODEL_PATH)