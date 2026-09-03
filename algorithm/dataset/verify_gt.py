import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "synthetic_data"
with (DATA_DIR / "ground_truth_combined.csv").open(newline="") as f:
    row = list(csv.DictReader(f))[0]  # pair 0

pair_id = row["pair_id"]
search = Image.open(DATA_DIR / f"pair_{pair_id}" / "search.png").convert("L")
reference = Image.open(DATA_DIR / f"pair_{pair_id}" / "reference.png").convert("L")

x, y = float(row["gt_x"]), float(row["gt_y"])
r = 50  # half of 100x100 footprint

search_boxed = search.convert("RGB")
draw = ImageDraw.Draw(search_boxed)
draw.rectangle([x-r, y-r, x+r, y+r], outline=(255,0,0), width=2)
search_boxed.save("gt_check.png")

footprint = search.crop((x-r, y-r, x+r, y+r)).resize((1000, 1000), Image.BICUBIC)
footprint.save("footprint_upscaled.png")

a = np.array(footprint, dtype=np.float32)
b = np.array(reference, dtype=np.float32)
a = (a - a.mean()) / (a.std() + 1e-6)
b = (b - b.mean()) / (b.std() + 1e-6)
correlation = float((a * b).mean())

print(f"GT box at ({x:.1f}, {y:.1f})")
print(f"Correlation between footprint and reference: {correlation:.3f}")
print("(should be well above 0.5 if ground truth is correct)")