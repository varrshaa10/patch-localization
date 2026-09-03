import argparse
import csv
import os
from pathlib import Path

import cv2

p = argparse.ArgumentParser()
p.add_argument("--pair_id", required=True)
args = p.parse_args()

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "synthetic_data"
with (DATA_DIR / "ground_truth_combined.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

row = next(r for r in rows if r["pair_id"] == args.pair_id)
gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

search = cv2.imread(str(DATA_DIR / f"pair_{args.pair_id}" / "search.png"))
ref = cv2.imread(str(DATA_DIR / f"pair_{args.pair_id}" / "reference.png"))

if search is None or ref is None:
    raise FileNotFoundError(f"Missing dataset images for pair {args.pair_id}")

marked = search.copy()
cv2.drawMarker(marked, (int(gt_x), int(gt_y)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)

half = ref.shape[0] // 2
crop = search[max(0, int(gt_y) - half):int(gt_y) + half, max(0, int(gt_x) - half):int(gt_x) + half]

output_dir = Path(__file__).resolve().parents[1] / "docs" / "examples"
output_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_dir / f"gt_check_pair_{args.pair_id}_marked.png"), marked)
cv2.imwrite(str(output_dir / f"gt_check_pair_{args.pair_id}_crop.png"), crop)
cv2.imwrite(str(output_dir / f"gt_check_pair_{args.pair_id}_ref.png"), ref)

print(f"Pair {args.pair_id}: gt=({gt_x},{gt_y}), arch={row['architecture']}, marker={row['add_marker']}")
print("Saved: marked search image, cropped region at gt, and reference image side by side.")