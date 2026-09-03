import csv
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1] / "tests" / "synthetic_data"
gt_rows = list(csv.DictReader((data_dir / "ground_truth_combined.csv").open(newline="")))
results = list(csv.DictReader(open("batch_results.csv")))

hard_ids = {r["pair_id"] for r in gt_rows if r["add_marker"] == "False"}

print(f"{'pair_id':>8} {'arch':>8} {'pixel_error':>12}")
for r in results:
    if r["pair_id"] in hard_ids:
        print(f"{r['pair_id']:>8} {r['architecture']:>8} {float(r['pixel_error']):>12.2f}")