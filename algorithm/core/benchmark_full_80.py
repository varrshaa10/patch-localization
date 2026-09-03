import csv
import statistics
import time
from pathlib import Path

import cv2

from infer import full_grid_search, coarse_to_fine_search

DATASET_ROOT = Path(r'C:\Users\svars\OneDrive\Desktop\patch-localization\algorithm\tests\synthetic_data_phase2')
CSV_PATH = DATASET_ROOT / 'ground_truth_combined.csv'


def benchmark(label, search_fn, include_absent=True):
    rows = list(csv.DictReader(CSV_PATH.open(newline='')))
    if not include_absent:
        rows = [r for r in rows if r.get('category') != 'absent']

    total = len(rows)
    print(f'{label}: starting {total} pairs', flush=True)
    timings = []
    errors = []
    within5 = 0
    for idx, row in enumerate(rows, start=1):
        print(f'pair {idx}/{total}', flush=True)
        pair_id = row['ref_path'].split('/')[-2]
        ref_path = DATASET_ROOT / pair_id / 'reference.png'
        search_path = DATASET_ROOT / pair_id / 'search.png'

        t0 = time.perf_counter()
        ref = cv2.imread(str(ref_path))
        search = cv2.imread(str(search_path))
        load_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        best, _, _, _, _ = search_fn(search, ref)
        search_time = time.perf_counter() - t0
        total = load_time + search_time
        timings.append(total)

        gt_x = row.get('gt_x')
        gt_y = row.get('gt_y')
        if gt_x not in (None, '') and gt_y not in (None, ''):
            x, y, _, _, _ = best
            gt_x = float(gt_x); gt_y = float(gt_y)
            err = ((x - gt_x) ** 2 + (y - gt_y) ** 2) ** 0.5
            errors.append(err)
            if err <= 5:
                within5 += 1

    print(f'=== {label} ===')
    print(f'rows={len(rows)}')
    print(f'avg_total={sum(timings) / len(timings):.6f}s')
    print(f'median_total={statistics.median(timings):.6f}s')
    print(f'max_total={max(timings):.6f}s')
    print(f'avg_load={sum(t for t in timings) / len(timings):.6f}s')
    if errors:
        print(f'mean_error={sum(errors) / len(errors):.3f}px')
        print(f'median_error={statistics.median(errors):.3f}px')
        print(f'within_5px={within5}/{len(errors)}')
    print()


benchmark('full_grid', full_grid_search, include_absent=True)
benchmark('coarse_to_fine', coarse_to_fine_search, include_absent=True)
benchmark('full_grid_present_only', full_grid_search, include_absent=False)
benchmark('coarse_to_fine_present_only', coarse_to_fine_search, include_absent=False)
