import csv
import statistics
import time
from pathlib import Path

import cv2

from infer import full_grid_search, coarse_to_fine_search

DATASET_ROOT = Path(r'C:\Users\svars\OneDrive\Desktop\patch-localization\algorithm\tests\synthetic_data_phase2')
CSV_PATH = DATASET_ROOT / 'ground_truth_combined.csv'

rows = list(csv.DictReader(CSV_PATH.open(newline='')))
subset = [r for r in rows if r['ref_path'].endswith('/pair_0/reference.png') or r['ref_path'].endswith('/pair_1/reference.png') or r['ref_path'].endswith('/pair_2/reference.png') or r['ref_path'].endswith('/pair_3/reference.png') or r['ref_path'].endswith('/pair_4/reference.png') or r['ref_path'].endswith('/pair_5/reference.png') or r['ref_path'].endswith('/pair_6/reference.png') or r['ref_path'].endswith('/pair_7/reference.png') or r['ref_path'].endswith('/pair_8/reference.png') or r['ref_path'].endswith('/pair_9/reference.png')]

for label, fn in [('full_grid', full_grid_search), ('coarse_to_fine', coarse_to_fine_search)]:
    total = len(subset)
    print(f'{label}: starting {total} pairs', flush=True)
    timings=[]
    for idx, row in enumerate(subset, start=1):
        print(f'pair {idx}/{total}', flush=True)
        pair = row['ref_path'].split('/')[-2]
        ref = str((DATASET_ROOT / pair / 'reference.png').resolve())
        srch = str((DATASET_ROOT / pair / 'search.png').resolve())
        t0=time.perf_counter(); ref_im=cv2.imread(ref); sr_im=cv2.imread(srch)
        t0=time.perf_counter(); fn(sr_im, ref_im)
        timings.append(time.perf_counter() - t0)
    print(label)
    print(f'count={len(timings)}')
    print(f'median={statistics.median(timings):.6f}s')
    print(f'mean={sum(timings)/len(timings):.6f}s')
    print(f'max={max(timings):.6f}s')
