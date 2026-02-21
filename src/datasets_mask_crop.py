import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class HAMMaskCropDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # ---- Detect label column automatically ----
        possible_label_cols = ["label", "target", "dx_idx", "class"]
        self.label_col = None

        for col in possible_label_cols:
            if col in self.df.columns:
                self.label_col = col
                break

        if self.label_col is None:
            raise ValueError("No label column found in CSV.")

        # ---- Metadata columns (exclude non-meta columns) ----
        exclude_cols = [
            self.label_col,
            "image_path",
            "x_min", "y_min", "x_max", "y_max"
        ]

        self.meta_cols = [
            col for col in self.df.columns
            if col not in exclude_cols and
            self.df[col].dtype in ["int64", "float64"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image_np = np.array(image)

        # ---- Bounding Box ----
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])

        h, w, _ = image_np.shape

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        cropped = image_np[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            cropped = image_np

        image = Image.fromarray(cropped)

        if self.transform:
            image = self.transform(image)

        metadata = torch.tensor(
            row[self.meta_cols].values.astype(np.float32)
        )

        label = torch.tensor(row[self.label_col]).long()

        return image, metadata, label