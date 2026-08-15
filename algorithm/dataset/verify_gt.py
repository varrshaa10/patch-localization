import csv
import numpy as np
from PIL import Image, ImageDraw

with open("../tests/synthetic_data/ground_truth.csv") as f:
    row = list(csv.DictReader(f))[0]  # pair 0

search = Image.open(row["search_path"]).convert("L")
reference = Image.open(row["ref_path"]).convert("L")

x, y = float(row["gt_x"]), float(row["gt_y"])
r = 50  # half of 100x100 footprint

# draw the box for visual reference
search_boxed = search.convert("RGB")
draw = ImageDraw.Draw(search_boxed)
draw.rectangle([x-r, y-r, x+r, y+r], outline=(255,0,0), width=2)
search_boxed.save("gt_check.png")

# crop the claimed footprint and blow it up to reference size for direct comparison
footprint = search.crop((x-r, y-r, x+r, y+r)).resize((1000, 1000), Image.BICUBIC)
footprint.save("footprint_upscaled.png")

# numeric similarity check: correlate footprint vs reference
a = np.array(footprint, dtype=np.float32)
b = np.array(reference, dtype=np.float32)
a = (a - a.mean()) / (a.std() + 1e-6)
b = (b - b.mean()) / (b.std() + 1e-6)
correlation = float((a * b).mean())

print(f"GT box at ({x:.1f}, {y:.1f})")
print(f"Correlation between footprint and reference: {correlation:.3f}")
print("(should be well above 0.5 if ground truth is correct)")