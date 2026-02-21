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
import os
import csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("../data/processed")

FREEZE_EPOCHS = 5
FINETUNE_EPOCHS = 15


# --------------------------------------------------
# Dataset (Same as baseline)
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
# Transfer Model (ResNet34 + Metadata)
# --------------------------------------------------

class MultiModalResNet(nn.Module):
    def __init__(self, meta_input_dim, num_classes=7):
        super().__init__()

        backbone = models.resnet34(weights=None)
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
# Load Segmentation Backbone
# --------------------------------------------------

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in loader:
            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, metadata)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1

def load_segmentation_backbone(model):
    seg_weights = torch.load(
        DATA_DIR / "segmentation_backbone.pth",
        map_location=DEVICE,
        weights_only=True
    )

    model_dict = model.backbone.state_dict()
    seg_weights = {k: v for k, v in seg_weights.items() if k in model_dict}
    model_dict.update(seg_weights)
    model.backbone.load_state_dict(model_dict)

    print("Segmentation backbone loaded.")


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


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
load_segmentation_backbone(model)


with open(DATA_DIR / "class_weights.json") as f:
    class_weights_list = json.load(f)

class_weights = torch.tensor(class_weights_list).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_f1 = 0

# -------- PHASE 1: Freeze --------

freeze_backbone(model)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

print("Phase 1: Freeze Backbone")

for epoch in range(FREEZE_EPOCHS):
    model.train()
    for images, metadata, labels in tqdm(train_loader):
        images, metadata, labels = images.to(DEVICE), metadata.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    val_f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} Val Macro F1: {val_f1:.4f}")


# -------- PHASE 2: Unfreeze --------

print("Phase 2: Unfreeze Full Model")
unfreeze_all(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for images, metadata, labels in tqdm(train_loader):
        images, metadata, labels = images.to(DEVICE), metadata.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    val_f1 = evaluate(model, val_loader)
    print(f"[FT {epoch+1}] Val Macro F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), DATA_DIR / "best_transfer_model.pth")


# --------------------------------------------------
# Log Results
# --------------------------------------------------

results_path = Path("../results/experiments_transfer.csv")

file_exists = results_path.exists()

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["model", "freeze_epochs", "finetune_epochs", "best_macro_f1"])
    writer.writerow(["resnet34_seg_transfer", FREEZE_EPOCHS, FINETUNE_EPOCHS, best_f1])

print("Transfer experiment logged.")