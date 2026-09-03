import csv
import math
from pathlib import Path
import cv2
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'algorithm' / 'core'))
import infer as inf

root = Path(__file__).resolve().parent / 'algorithm' / 'tests' / 'synthetic_data_phase2'
with open(root / 'ground_truth_combined.csv', newline='') as f:
    rows = list(csv.DictReader(f))
present = [r for r in rows if int(float(r['found'])) == 1]

for label, clear_cache in [('without_clear', False), ('with_clear', True)]:
    hits = 0
    sample = []
    for row in present:
        pid = str(int(float(row['pair_id'])))
        ref_path = root / f'pair_{pid}' / 'reference.png'
        search_path = root / f'pair_{pid}' / 'search.png'
        ref = cv2.imread(str(ref_path))
        search = cv2.imread(str(search_path))
        if clear_cache:
            inf._TEMPLATE_CACHE.clear()
        best, *_ = inf.coarse_to_fine_search(search, ref)
        x, y = best[0], best[1]
        gt_x = float(row['gt_x'])
        gt_y = float(row['gt_y'])
        err = math.hypot(x - gt_x, y - gt_y)
        if err <= 5:
            hits += 1
        if len(sample) < 5:
            sample.append((pid, err, (x, y), (gt_x, gt_y)))
    print(label, hits, '/', len(present), 'sample=', sample)
