import csv
import time
from pathlib import Path

import cv2

from infer import full_grid_search, multi_peak_search

DATASET_ROOT = Path(__file__).resolve().parents[1] / "tests" / "synthetic_data_phase2"
CSV_PATH = DATASET_ROOT / "ground_truth_combined.csv"


def evaluate(label, search_fn):
    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    present = [r for r in rows if r.get("category") != "absent"]
    total = len(present)
    print(f"{label}: starting {total} present pairs", flush=True)
    load_times = []
    search_times = []
    errors = []
    within5 = 0

    for idx, row in enumerate(present, start=1):
        print(f"pair {idx}/{total}", flush=True)
        pair_id = row["ref_path"].split("/")[-2]
        ref_path = DATASET_ROOT / pair_id / "reference.png"
        search_path = DATASET_ROOT / pair_id / "search.png"

        t0 = time.perf_counter()
        ref = cv2.imread(str(ref_path))
        search = cv2.imread(str(search_path))
        load_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        best, _, _, _, _ = search_fn(search, ref)
        search_times.append(time.perf_counter() - t0)

        x, y, _, _, _ = best
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        error = ((x - gt_x) ** 2 + (y - gt_y) ** 2) ** 0.5
        errors.append(error)
        if error <= 5:
            within5 += 1

    print(f"\n=== {label} ===")
    print(f"present_pairs={len(present)}")
    print(f"avg_image_load={sum(load_times) / len(load_times):.6f}s")
    print(f"avg_grid_search={sum(search_times) / len(search_times):.6f}s")
    print(f"avg_total={(sum(load_times) + sum(search_times)) / len(present):.6f}s")
    print(f"mean_error={sum(errors) / len(errors):.3f}px")
    print(f"median_error={sorted(errors)[len(errors) // 2]:.3f}px")
    print(f"max_error={max(errors):.3f}px")
    print(f"within_5px={within5}/{len(present)}")


if __name__ == "__main__":
    evaluate("full_grid", full_grid_search)
    evaluate("coarse_to_fine", lambda image, reference: multi_peak_search(image, reference, coarse_to_fine=True))
