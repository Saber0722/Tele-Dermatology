import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import cv2

from models import SegmentationModel  # use your segmentation model class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("../data/processed")

MODEL_PATH = DATA_DIR / "best_segmentation_model.pth"

# --------------------------------------------------
# Load Segmentation Model
# --------------------------------------------------

model = SegmentationModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------------------------------------------------
# Load CSV
# --------------------------------------------------

df = pd.read_csv(DATA_DIR / "ham10000_clean.csv")

bboxes = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image = Image.open(row["image_path"]).convert("RGB")
    image_np = np.array(image)

    h, w, _ = image_np.shape

    image_tensor = torch.tensor(image_np / 255.0).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
    image_resized = torch.nn.functional.interpolate(image_tensor, size=(256,256))

    with torch.no_grad():
        mask = torch.sigmoid(model(image_resized))

    mask_np = mask.squeeze().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8)

    coords = np.column_stack(np.where(mask_np > 0))

    if len(coords) == 0:
        bboxes.append((0,0,w,h))
        continue

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    scale_x = w / 256
    scale_y = h / 256

    x_min = int(x_min * scale_x)
    x_max = int(x_max * scale_x)
    y_min = int(y_min * scale_y)
    y_max = int(y_max * scale_y)

    bboxes.append((x_min, y_min, x_max, y_max))

df["x_min"], df["y_min"], df["x_max"], df["y_max"] = zip(*bboxes)

df.to_csv(DATA_DIR / "ham10000_with_bboxes.csv", index=False)

print("Bounding boxes saved.")