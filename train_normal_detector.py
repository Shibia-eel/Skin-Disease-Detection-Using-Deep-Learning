import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ==============================
# SETTINGS
# ==============================

BASE_DIR = r"C:\Skin_Disease_Project"
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "normal_detector")
MODEL_PATH = os.path.join(BASE_DIR, "normal_skin_detector.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("========================================")
print("      NORMAL SKIN DETECTOR")
print("========================================")
print("Device:", device)


# ==============================
# TRANSFORMS
# ==============================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# LOAD DATASET
# ==============================

dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ==============================
# RESNET18
# ==============================

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)


# ==============================
# LOSS + OPTIMIZER
# ==============================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ==============================
# TRAINING
# ==============================

best_accuracy = 0.0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("--------------------------------")

    # TRAIN
    model.train()

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # VALIDATION
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Training Accuracy   : {train_accuracy:.2f}%")
    print(f"Validation Accuracy : {val_accuracy:.2f}%")

    # SAVE BEST MODEL
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved!")


print("\n========================================")
print("       TRAINING COMPLETED")
print("========================================")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Model saved to:")
print(MODEL_PATH)