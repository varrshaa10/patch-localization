import csv, math, sys
from pathlib import Path
import cv2
sys.path.insert(0, 'algorithm/core')
import infer as inf
root = Path('algorithm/tests/synthetic_data_phase2')
with open(root / 'ground_truth_combined.csv', newline='') as f:
    rows = list(csv.DictReader(f))
present = [r for r in rows if int(float(r['found'])) == 1]

hits = 0
for row in present:
    inf._TEMPLATE_CACHE.clear()
    pid = str(row['pair_id'])
    ref = cv2.imread(str(root / f'pair_{pid}' / 'reference.png'))
    search = cv2.imread(str(root / f'pair_{pid}' / 'search.png'))
    best, *_ = inf.coarse_to_fine_search(search, ref)
    x, y = best[0], best[1]
    gt_x = float(row['gt_x']); gt_y = float(row['gt_y'])
    if math.hypot(x - gt_x, y - gt_y) <= 5:
        hits += 1
print('cache_cleared_within5', hits, '/', len(present))
