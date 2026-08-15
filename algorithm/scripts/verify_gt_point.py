import cv2
import csv
import argparse

p = argparse.ArgumentParser()
p.add_argument("--pair_id", required=True)
args = p.parse_args()

with open("../tests/synthetic_data/ground_truth.csv") as f:
    rows = list(csv.DictReader(f))

row = next(r for r in rows if r["pair_id"] == args.pair_id)
gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

search = cv2.imread(row["search_path"])
ref = cv2.imread(row["ref_path"])

# draw crosshair on search image at ground truth point
marked = search.copy()
cv2.drawMarker(marked, (int(gt_x), int(gt_y)), (0,0,255), cv2.MARKER_CROSS, 40, 3)

# crop a region around gt point matching reference size, for side-by-side comparison
half = ref.shape[0] // 2
crop = search[max(0,int(gt_y)-half):int(gt_y)+half, max(0,int(gt_x)-half):int(gt_x)+half]

import os
output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "examples")
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, f"gt_check_pair_{args.pair_id}_marked.png"), marked)
cv2.imwrite(os.path.join(output_dir, f"gt_check_pair_{args.pair_id}_crop.png"), crop)
cv2.imwrite(os.path.join(output_dir, f"gt_check_pair_{args.pair_id}_ref.png"), ref)

print(f"Pair {args.pair_id}: gt=({gt_x},{gt_y}), arch={row['architecture']}, marker={row['add_marker']}")
print("Saved: marked search image, cropped region at gt, and reference image side by side.")