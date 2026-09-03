import csv, time, sys
from pathlib import Path
import cv2
sys.path.insert(0, r'C:\Users\svars\OneDrive\Desktop\patch-localization\algorithm\core')
from infer import full_grid_search, multi_peak_search

csv_path = Path(r'C:\Users\svars\OneDrive\Desktop\patch-localization\algorithm\tests\synthetic_data_phase2\ground_truth_combined.csv')
rows = list(csv.DictReader(open(csv_path, newline='')))
present = [r for r in rows if r.get('category') != 'absent' and str(r.get('found', '1')) not in ('0', 'False', 'false', '')]

for label, fn in [('full', full_grid_search), ('coarse', multi_peak_search)]:
    load_times=[]; search_times=[]; within5=0; total_err=[]
    print(f'== {label} ==')
    for r in present:
        pair = r['ref_path'].split('/')[-2]
        ref = str((csv_path.parent / pair / 'reference.png').resolve())
        srch = str((csv_path.parent / pair / 'search.png').resolve())
        t0 = time.perf_counter(); ref_im = cv2.imread(ref); sr_im = cv2.imread(srch); load_times.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); best, _, _, _, _ = fn(sr_im, ref_im); search_times.append(time.perf_counter()-t0)
        x, y, _, _, _ = best
        gt_x, gt_y = float(r['gt_x']), float(r['gt_y'])
        err = ((x-gt_x)**2 + (y-gt_y)**2)**0.5
        total_err.append(err)
        if err <= 5:
            within5 += 1
    print(f'avg_image_load={sum(load_times)/len(load_times):.6f}s')
    print(f'avg_search={sum(search_times)/len(search_times):.6f}s')
    print(f'avg_total={(sum(load_times)+sum(search_times))/len(present):.6f}s')
    print(f'within5={within5}/{len(present)}')
    print(f'mean_error={sum(total_err)/len(total_err):.3f}px median_error={sorted(total_err)[len(total_err)//2]:.3f}px max_error={max(total_err):.3f}px')
