import csv
import time
from pathlib import Path

import cv2

from infer import coarse_to_fine_search

DATASET_ROOT = Path(r'C:\Users\svars\OneDrive\Desktop\patch-localization\algorithm\tests\synthetic_data_phase2')
CSV_PATH = DATASET_ROOT / 'ground_truth_combined.csv'

rows = list(csv.DictReader(CSV_PATH.open(newline='')))
present = [r for r in rows if r.get('category') != 'absent' and str(r.get('found', '1')) not in ('0', 'False', 'false', '')]
total = len(present)
print(f'benchmark_coarse_to_fine: starting {total} present pairs', flush=True)

load_times = []
search_times = []
errors = []
within5 = 0
for idx, row in enumerate(present, start=1):
    print(f'pair {idx}/{total}', flush=True)
    pair_id = row['ref_path'].split('/')[-2]
    ref_path = DATASET_ROOT / pair_id / 'reference.png'
    search_path = DATASET_ROOT / pair_id / 'search.png'

    t0 = time.perf_counter()
    ref = cv2.imread(str(ref_path))
    search = cv2.imread(str(search_path))
    load_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    best, _, _, _, _ = coarse_to_fine_search(search, ref)
    search_times.append(time.perf_counter() - t0)

    x, y, _, _, _ = best
    gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])
    err = ((x - gt_x) ** 2 + (y - gt_y) ** 2) ** 0.5
    errors.append(err)
    if err <= 5:
        within5 += 1

print(f'present_count={len(present)}')
print(f'avg_image_load={sum(load_times) / len(load_times):.6f}s')
print(f'avg_grid_search={sum(search_times) / len(search_times):.6f}s')
print(f'avg_total={(sum(load_times) + sum(search_times)) / len(present):.6f}s')
print(f'mean_error={sum(errors) / len(errors):.3f}px')
print(f'median_error={sorted(errors)[len(errors) // 2]:.3f}px')
print(f'within5={within5}/{len(present)}')
