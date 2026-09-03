import csv
import random
import statistics
import time
from pathlib import Path

import cv2

from infer import full_grid_search, coarse_to_fine_search

DATASET_ROOT = Path(r'C:\Users\svars\OneDrive\Desktop\patch-localization\algorithm\tests\synthetic_data_phase2')
CSV_PATH = DATASET_ROOT / 'ground_truth_combined.csv'

rows = list(csv.DictReader(CSV_PATH.open(newline='')))

# Full-all-80 benchmark for non-sample run; if too slow, use the 20-pair sample below.
# Here we keep the default as a deterministic 20-pair sample to stay feasible.
# Sample design: 5 nominal, 5 degraded, 5 absent, 5 extra random from remaining pairs.
nominal = [r for r in rows if r.get('category') == 'present_nominal']
degraded = [r for r in rows if r.get('category') == 'present_degraded']
absent = [r for r in rows if r.get('category') == 'absent']
random.seed(0)
subset = (
    nominal[:5] + degraded[:5] + absent[:5] +
    random.sample([r for r in rows if r not in nominal[:5] and r not in degraded[:5] and r not in absent[:5]], 5)
)


def benchmark(label, fn, subset_rows):
    total = len(subset_rows)
    print(f'{label}: starting {total} pairs', flush=True)
    timings = []
    errors = []
    within5 = 0
    for idx, row in enumerate(subset_rows, start=1):
        print(f'pair {idx}/{total}', flush=True)
        pair_id = row['ref_path'].split('/')[-2]
        ref = cv2.imread(str(DATASET_ROOT / pair_id / 'reference.png'))
        search = cv2.imread(str(DATASET_ROOT / pair_id / 'search.png'))
        t0 = time.perf_counter()
        best, _, _, _, _ = fn(search, ref)
        timings.append(time.perf_counter() - t0)
        gt_x_raw = row.get('gt_x')
        gt_y_raw = row.get('gt_y')
        if gt_x_raw not in (None, '') and gt_y_raw not in (None, ''):
            gt_x = float(gt_x_raw)
            gt_y = float(gt_y_raw)
            x, y, _, _, _ = best
            err = ((x - gt_x) ** 2 + (y - gt_y) ** 2) ** 0.5
            errors.append(err)
            if err <= 5:
                within5 += 1
    print(f'=== {label} ===')
    print(f'count={len(timings)}')
    print(f'median={statistics.median(timings):.6f}s')
    print(f'mean={sum(timings)/len(timings):.6f}s')
    print(f'max={max(timings):.6f}s')
    print(f'within_5px_present={within5}/{len(errors)}')
    print()


benchmark('full_grid_sample_20', full_grid_search, subset)
benchmark('pyramid_sample_20', coarse_to_fine_search, subset)
