import json
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path

from src.datasets_mask_crop import HAMMaskCropDataset

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("../data/processed")

EPOCHS = 15
LR = 1e-4

print(f"Using device: {DEVICE}")

# --------------------------------------------------
# Model
# --------------------------------------------------

class MultiModalEfficientNet(nn.Module):
    def __init__(self, meta_input_dim, num_classes=7):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.backbone.classifier = nn.Identity()

        image_feat_dim = 1280

        self.meta_net = nn.Sequential(
            nn.Linear(meta_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, metadata):
        img_feat = self.backbone(image)
        meta_feat = self.meta_net(metadata)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(combined)

# --------------------------------------------------
# Transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Load Data
# --------------------------------------------------

train_dataset = HAMMaskCropDataset(
    DATA_DIR / "train_with_bboxes.csv",
    transform=train_transform
)

val_dataset = HAMMaskCropDataset(
    DATA_DIR / "val_with_bboxes.csv",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

meta_dim = len(train_dataset.meta_cols)

model = MultiModalEfficientNet(meta_input_dim=meta_dim).to(DEVICE)

# --------------------------------------------------
# Loss
# --------------------------------------------------

with open(DATA_DIR / "class_weights.json") as f:
    class_weights_list = json.load(f)

class_weights = torch.tensor(class_weights_list).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------
# Training
# --------------------------------------------------


if __name__ == "__main__":

    best_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for images, metadata, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds, average="macro")

        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(DEVICE)
                metadata = metadata.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images, metadata)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Train Macro F1: {train_f1:.4f}")
        print(f"Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), DATA_DIR / "best_efficient_mask_crop.pth")
            print("New best model saved.")

    print(f"\nBest Efficient Mask-Crop Macro F1: {best_f1:.4f}")