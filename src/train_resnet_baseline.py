import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path

# --------------------------------------------------
# Device
# --------------------------------------------------

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

print(f"Using device: {DEVICE}")

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class HAMDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.meta_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        if "label" in self.meta_cols:
            self.meta_cols.remove("label")

        print("Metadata columns:", self.meta_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        metadata = torch.tensor(
            row[self.meta_cols].values.astype(np.float32)
        )

        label = torch.tensor(row["label"]).long()
        return image, metadata, label


# --------------------------------------------------
# Model
# --------------------------------------------------

class MultiModalResNet(nn.Module):
    def __init__(self, meta_input_dim, num_classes=7):
        super().__init__()

        # ImageNet pretrained ResNet34
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        image_feat_dim = 512

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

train_dataset = HAMDataset(DATA_DIR / "train.csv", train_transform)
val_dataset = HAMDataset(DATA_DIR / "val.csv", val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

meta_dim = len(train_dataset.meta_cols)

model = MultiModalResNet(meta_input_dim=meta_dim).to(DEVICE)


# --------------------------------------------------
# Loss
# --------------------------------------------------

with open(DATA_DIR / "class_weights.json") as f:
    class_weights_list = json.load(f)

class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

def train_one_epoch():
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, metadata, labels in tqdm(train_loader, desc="Training"):
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

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(train_loader), macro_f1


def validate():
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, metadata, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, metadata)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1


# --------------------------------------------------
# Run Training
# --------------------------------------------------

if __name__ == "__main__":
    best_f1 = 0

    for epoch in range(15):
        print(f"\nEpoch {epoch+1}/15")

        train_loss, train_f1 = train_one_epoch()
        val_f1 = validate()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Macro F1: {train_f1:.4f}")
        print(f"Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), DATA_DIR / "best_resnet34_baseline.pth")
            print("Saved best model.")

    print(f"\nBest Macro F1: {best_f1:.4f}")
