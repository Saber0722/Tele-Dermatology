import torch
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from models import SegmentationModel
from datasets import ISICSegmentationDataset
from losses import BCEDiceLoss
from utils import log_experiment, print_epoch, console


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4
PATIENCE = 5


def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def train():
    console.print("[bold green]Starting ISIC Segmentation Training[/bold green]")

    transform = get_transforms()

    full_dataset = ISICSegmentationDataset(
        "../data/raw/ISIC/ISIC2018_Task1-2_Training_Input",
        "../data/raw/ISIC/ISIC2018_Task1_Training_GroundTruth",
        transform
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    console.print(f"Train size: {len(train_dataset)}")
    console.print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SegmentationModel(pretrained=True).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    early_counter = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print_epoch(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0

            os.makedirs("../data/processed", exist_ok=True)

            torch.save(model.state_dict(),
                       "../data/processed/best_segmentation_model.pth")

            torch.save(model.backbone.state_dict(),
                       "../data/processed/segmentation_backbone.pth")

            console.print("[green]Best model saved[/green]")

        else:
            early_counter += 1
            if early_counter >= PATIENCE:
                console.print("[red]Early stopping triggered[/red]")
                break

    log_experiment(
        "../results/experiments.csv",
        {
            "model": "resnet34_segmentation",
            "img_size": IMG_SIZE,
            "loss": "BCE+Dice",
            "val_loss": best_val_loss
        }
    )


if __name__ == "__main__":
    train()