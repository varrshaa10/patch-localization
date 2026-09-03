import csv, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r'C:\Users\svars\OneDrive\Desktop\patch-localization').resolve()
sys.path.insert(0, str(ROOT / 'algorithm' / 'core'))
from infer import coarse_to_fine_search
from ncc import ncc_match_multi
from rotation_search import rotate_image
from scale_rotation_search import scale_template

threshold = 0.001
held_path = ROOT / 'algorithm' / 'tests' / 'synthetic_data_phase2_v2' / 'ground_truth_combined.csv'
rows = list(csv.DictReader(open(held_path, newline='')))
results = []
for r in rows:
    pair_id = r['pair_id']
    category = r['category']
    pair_dir = ROOT / 'algorithm' / 'tests' / 'synthetic_data_phase2_v2' / f'pair_{pair_id}'
    image = cv2.imread(str(pair_dir / 'search.png'))
    template = cv2.imread(str(pair_dir / 'reference.png'))
    best, _, _, _, _ = coarse_to_fine_search(
        image, template,
        angle_range=5, angle_step=1,
        zoom_min=8.0, zoom_max=12.0, zoom_step=0.5,
    )
    angle = float(best[3])
    scale = float(best[4])
    rotated = rotate_image(template, angle)
    scaled_template = scale_template(rotated, scale)
    peaks = ncc_match_multi(image, scaled_template, num_peaks=20, min_distance=15)
    peak_scores = [float(s) for _, _, s in peaks]
    if peak_scores:
        best_peak = max(peak_scores)
        remaining = [x for x in peak_scores if x != best_peak]
        median_of_rest = float(np.median(remaining)) if remaining else 0.0
        score = best_peak - median_of_rest
    else:
        score = 0.0
    pred = 1 if score >= threshold else 0
    true = 0 if category == 'absent' else 1
    results.append((pred, true, score, category))
TP = sum(1 for p, t, _, _ in results if p == 1 and t == 1)
FP = sum(1 for p, t, _, _ in results if p == 1 and t == 0)
FN = sum(1 for p, t, _, _ in results if p == 0 and t == 1)
TN = sum(1 for p, t, _, _ in results if p == 0 and t == 0)
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
print('held_out_rows', len(results))
print('present', sum(1 for _, t, _, _ in results if t == 1))
print('absent', sum(1 for _, t, _, _ in results if t == 0))
print('threshold', threshold)
print('precision', precision)
print('recall', recall)
print('f1', f1)
print('tp', TP, 'fp', FP, 'fn', FN, 'tn', TN)
